"""Export GPT-SoVITS v1/v2/v2Pro/v2ProPlus/v5dev/v5turbo checkpoints to ONNX.

The command uses synthetic tracing inputs. It never loads a reference recording or
SSL/speaker encoders, and importing this module does not load models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

V5_VERSIONS = ("v5dev", "v5turbo")
SUPPORTED_VERSIONS = ("v1", "v2", "v2Pro", "v2ProPlus", "v5dev", "v5turbo")
PRO_VERSIONS = ("v2Pro", "v2ProPlus")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export GPT-SoVITS v1/v2/v2Pro/v2ProPlus/v5dev/v5turbo checkpoints to ONNX."
    )
    parser.add_argument(
        "--gpt-model", required=True, type=Path, help="GPT checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--sovits-model", required=True, type=Path, help="SoVITS checkpoint (.pth)."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for exported files."
    )
    parser.add_argument(
        "--name", help="Output filename prefix (default: output directory name)."
    )
    parser.add_argument(
        "--version",
        choices=("auto",) + SUPPORTED_VERSIONS,
        default="auto",
        help="Expected SoVITS version. 'auto' reads checkpoint metadata (default: auto).",
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default: 17)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for synthetic tracing inputs (default: 0).",
    )
    parser.add_argument(
        "--pretrained-sovits-model",
        type=Path,
        default=None,
        help="v5 LoRA base checkpoint (default: pretrained_models/gsv-v5-pretrained/s2G<version>.pth).",
    )
    parser.add_argument(
        "--vocoder-model",
        type=Path,
        default=None,
        help="v5 vocoder checkpoint (default: pretrained_models/gsv-v5-pretrained/vocoder.pth).",
    )
    return parser


def detect_sovits_version(
    path: Path, checkpoint: dict[str, Any], expected: str = "auto"
) -> str:
    """Resolve the inference architecture from detector metadata and checkpoint shape."""
    if not path.is_file():
        raise FileNotFoundError(f"SoVITS checkpoint not found: {path}")
    from process_ckpt import get_sovits_version_from_path_fast

    try:
        _symbol_version, detected_version, is_lora = get_sovits_version_from_path_fast(
            str(path)
        )
    except KeyError as exc:
        with path.open("rb") as checkpoint_file:
            header = checkpoint_file.read(2)
        raise ValueError(f"Unsupported SoVITS checkpoint header: {header!r}") from exc
    if detected_version not in SUPPORTED_VERSIONS or (
        is_lora and detected_version not in V5_VERSIONS
    ):
        raise ValueError(
            f"Unsupported SoVITS version {detected_version!r}; supported versions: "
            + ", ".join(SUPPORTED_VERSIONS)
        )

    if detected_version in PRO_VERSIONS or detected_version in V5_VERSIONS:
        version = detected_version
    else:
        text_embedding = checkpoint["weight"].get("enc_p.text_embedding.weight")
        version = (
            "v1"
            if text_embedding is not None and text_embedding.shape[0] == 322
            else "v2"
        )

    if expected != "auto" and expected != version:
        raise ValueError(
            f"Checkpoint is {version}, but --version requested {expected}."
        )
    return version


def load_sovits_checkpoint(path: Path) -> dict[str, Any]:
    """Load a regular or two-byte-tagged checkpoint through the canonical loader."""
    from process_ckpt import load_sovits_new

    checkpoint = load_sovits_new(str(path))
    if not isinstance(checkpoint, dict) or not {"config", "weight"}.issubset(
        checkpoint
    ):
        raise ValueError("SoVITS checkpoint must contain 'config' and 'weight'.")
    return checkpoint


def waveform_to_spectrogram(y, n_fft: int, hop_size: int, win_size: int):
    import torch

    window = torch.hann_window(win_size, dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return spec.abs()


def _make_vits_wrapper(model, pro: bool):
    from torch import nn

    class LegacyVitsWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, text, semantic, reference_spectrogram, noise):
            return self.model(semantic, text, reference_spectrogram, noise=noise)[0, 0]

    class ProVitsWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(
            self, text, semantic, reference_spectrogram, noise, speaker_embedding
        ):
            # Matches SynthesizerTrn.decode(..., sv_emb=...) in production TTS.
            return self.model(
                semantic,
                text,
                reference_spectrogram,
                sv_emb=speaker_embedding,
                noise=noise,
            )[0, 0]

    return ProVitsWrapper() if pro else LegacyVitsWrapper()


def validate_vits_state_dict(incompatible) -> None:
    """Require every inference key and allow only training-only enc_q extras."""
    if incompatible.missing_keys:
        raise ValueError(
            f"Missing SoVITS checkpoint keys: {incompatible.missing_keys[:5]}"
        )
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("enc_q.")
    ]
    if unexpected:
        raise ValueError(f"Unexpected SoVITS checkpoint keys: {unexpected[:5]}")


def waveform_to_mel_v5(y):
    """32 kHz reference waveform to normalized v5 conditioning mel (outside ONNX)."""
    from module.mel_processing import mel_spectrogram_torch

    mel = mel_spectrogram_torch(
        y,
        n_fft=1280,
        hop_size=320,
        win_size=1280,
        num_mels=100,
        sampling_rate=32000,
        fmin=0,
        fmax=None,
        center=False,
    )
    return (mel + 12) / 14 * 2 - 1


def denorm_spec_v5(x):
    return (x + 1) / 2 * 14 - 12


def build_v5_vocoder(vocoder_path: Path | None = None):
    import torch
    from module.models_onnx import Generator

    resolved_path = vocoder_path or (
        Path(__file__).resolve().parent
        / "pretrained_models"
        / "gsv-v5-pretrained"
        / "vocoder.pth"
    )
    vocoder = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    vocoder.remove_weight_norm()
    vocoder.load_state_dict(
        torch.load(resolved_path, map_location="cpu", weights_only=False), strict=True
    )
    return vocoder.eval()


def build_vits_model(
    checkpoint: dict[str, Any],
    version: str,
    pretrained_sovits_path: Path | None = None,
):
    """Build inference weights and export config for every supported version."""
    from module.models_onnx import SynthesizerTrn

    if version in V5_VERSIONS:
        from module.models_onnx import SynthesizerTrnV3

    hps = checkpoint["config"]
    model_config = dict(hps["model"])
    model_config["semantic_frame_rate"] = "25hz"
    model_config["version"] = version
    data = hps["data"]
    model_class = SynthesizerTrnV3 if version in V5_VERSIONS else SynthesizerTrn
    model = model_class(
        data["filter_length"] // 2 + 1,
        hps["train"]["segment_size"] // data["hop_length"],
        n_speakers=data["n_speakers"],
        **model_config,
    ).eval()
    if version in V5_VERSIONS and "lora_rank" in checkpoint:
        from peft import LoraConfig, get_peft_model

        base_path = pretrained_sovits_path or (
            Path(__file__).resolve().parent
            / "pretrained_models"
            / "gsv-v5-pretrained"
            / f"s2G{version}.pth"
        )
        if not base_path.is_file():
            raise FileNotFoundError(f"v5 base SoVITS checkpoint not found: {base_path}")
        base_ckpt = load_sovits_checkpoint(base_path)
        validate_vits_state_dict(
            model.load_state_dict(base_ckpt["weight"], strict=False)
        )
        lora_rank = int(checkpoint["lora_rank"])
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        model.cfm = get_peft_model(model.cfm, lora_config)
        incompatible_lora = model.load_state_dict(checkpoint["weight"], strict=False)
        unexpected_lora = [
            key
            for key in incompatible_lora.unexpected_keys
            if not key.startswith("enc_q.")
        ]
        if unexpected_lora:
            raise ValueError(
                f"Unexpected v5 LoRA checkpoint keys: {unexpected_lora[:5]}"
            )
        model.cfm = model.cfm.merge_and_unload()
        model.eval()
    else:
        validate_vits_state_dict(
            model.load_state_dict(checkpoint["weight"], strict=False)
        )
    export_config = {
        "filter_length": data["filter_length"],
        "hop_length": data["hop_length"],
        "win_length": data["win_length"],
        "sampling_rate": data["sampling_rate"],
        "inter_channels": model.inter_channels,
    }
    if version in V5_VERSIONS:
        export_config.update(
            sampling_rate=48000,
            reference_sampling_rate=data["sampling_rate"],
            mel_channels=100,
            sample_steps=4 if version == "v5turbo" else 32,
            cfg_rate=0.0 if version == "v5turbo" else 1.30,
            noise_temperature=0.875,
        )
    return model, export_config


def build_t2s_model(gpt_path: Path):
    import torch
    from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule

    checkpoint = torch.load(gpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"config", "weight"}.issubset(
        checkpoint
    ):
        raise ValueError("GPT checkpoint must contain 'config' and 'weight'.")
    config = checkpoint["config"]
    lightning = Text2SemanticLightningModule(config, "onnx_export", is_train=False)
    lightning.load_state_dict(checkpoint["weight"])
    lightning.eval()
    model = lightning.model
    # Split decoder graphs do not expose top_k as an input; keep the configured
    # export-time constant as an int, which torch.topk accepts in eager mode.
    model.top_k = int(config["inference"]["top_k"])
    model.early_stop_num = torch.tensor(
        [50 * config["data"]["max_sec"]], dtype=torch.long
    )
    model.init_onnx()
    model.max_sec = config["data"]["max_sec"]
    return model


def _export_t2s(
    torch, t2s, vits, inputs, output_dir: Path, name: str, opset: int
) -> None:
    from torch import nn

    class T2SEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = t2s.onnx_encoder
            self.vits = vits

        def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
            prompt = self.vits.extract_latent(ssl_content)[0, 0].unsqueeze(0)
            bert = torch.cat(
                [ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1
            ).unsqueeze(0)
            phonemes = torch.cat([ref_seq, text_seq], 1)
            return self.encoder(phonemes, bert), prompt

    encoder = T2SEncoder().eval()
    torch.onnx.export(
        encoder,
        inputs,
        output_dir / f"{name}_t2s_encoder.onnx",
        input_names=["ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content"],
        output_names=["x", "prompts"],
        dynamic_axes={
            "ref_seq": {1: "ref_length"},
            "text_seq": {1: "text_length"},
            "ref_bert": {0: "ref_length"},
            "text_bert": {0: "text_length"},
            "ssl_content": {2: "ssl_length"},
            "x": {1: "phoneme_length"},
            "prompts": {1: "prompt_length"},
        },
        opset_version=opset,
    )
    x, prompts = encoder(*inputs)
    first_random = torch.rand((t2s.vocab_size,), dtype=x.dtype).clamp_(1e-6, 1 - 1e-6)
    first_inputs = (x, prompts, first_random)
    first_dynamic_axes = {
        "x": {1: "x_length"},
        "prompts": {1: "prompt_length"},
        "y": {1: "y_length"},
        "k": {1: "cache_length"},
        "v": {1: "cache_length"},
        "y_emb": {1: "embedding_length"},
        "x_example": {1: "x_length"},
    }
    torch.onnx.export(
        t2s.first_stage_decoder,
        first_inputs,
        output_dir / f"{name}_t2s_fsdec.onnx",
        input_names=["x", "prompts", "sampling_random"],
        output_names=["y", "k", "v", "y_emb", "x_example"],
        dynamic_axes=first_dynamic_axes,
        opset_version=opset,
    )
    decoder_outputs = t2s.first_stage_decoder(*first_inputs)
    step_random = torch.rand((t2s.vocab_size,), dtype=x.dtype).clamp_(1e-6, 1 - 1e-6)
    stage_inputs = (*decoder_outputs, step_random)
    stage_dynamic_axes = {
        "iy": {1: "y_length"},
        "ik": {1: "cache_length"},
        "iv": {1: "cache_length"},
        "iy_emb": {1: "embedding_length"},
        "ix_example": {1: "x_length"},
        "y": {1: "next_y_length"},
        "k": {1: "next_cache_length"},
        "v": {1: "next_cache_length"},
        "y_emb": {1: "next_embedding_length"},
        "samples": {1: "sample_length"},
    }
    torch.onnx.export(
        t2s.stage_decoder,
        stage_inputs,
        output_dir / f"{name}_t2s_sdec.onnx",
        input_names=["iy", "ik", "iv", "iy_emb", "ix_example", "sampling_random"],
        output_names=["y", "k", "v", "y_emb", "logits", "samples"],
        dynamic_axes=stage_dynamic_axes,
        opset_version=opset,
    )


def _export_vits(
    torch, model, config, version: str, output_dir: Path, name: str, opset: int
) -> None:
    pro = version in PRO_VERSIONS
    wrapper = _make_vits_wrapper(model, pro).eval()
    text = torch.ones((1, 8), dtype=torch.long)
    semantic = torch.ones((1, 1, 16), dtype=torch.long)
    reference_spectrogram = torch.randn(
        (1, config["filter_length"] // 2 + 1, 100), dtype=torch.float32
    )
    noise = torch.randn(
        (1, config["inter_channels"], semantic.shape[2] * 2), dtype=torch.float32
    )
    export_args = (text, semantic, reference_spectrogram, noise)
    input_names = ["text", "semantic", "reference_spectrogram", "noise"]
    dynamic_axes = {
        "text": {1: "text_length"},
        "semantic": {2: "semantic_length"},
        "reference_spectrogram": {2: "reference_frames"},
        "noise": {2: "vits_frame_length"},
        "audio": {0: "audio_samples"},
    }
    if pro:
        export_args += (torch.randn((1, 20480), dtype=torch.float32),)
        input_names.append("speaker_embedding")
    torch.onnx.export(
        wrapper,
        export_args,
        output_dir / f"{name}_vits.onnx",
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )


def _export_v5_acoustic(
    torch, model, vocoder, config, output_dir: Path, name: str, opset: int
):
    from torch import nn

    class EncpWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, text, semantic, reference_spectrogram):
            return self.model(semantic, text, reference_spectrogram)

    class VocoderWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocoder = vocoder

        def forward(self, mel):
            return self.vocoder(mel)[0, 0]

    torch.onnx.export(
        EncpWrapper().eval(),
        (
            torch.ones((1, 16), dtype=torch.long),
            torch.ones((1, 1, 32), dtype=torch.long),
            torch.randn((1, config["filter_length"] // 2 + 1, 64)),
        ),
        output_dir / f"{name}_vits_encp.onnx",
        input_names=["text", "semantic", "reference_spectrogram"],
        output_names=["fea", "ge"],
        dynamic_axes={
            "text": {1: "text_length"},
            "semantic": {2: "semantic_length"},
            "reference_spectrogram": {2: "reference_frames"},
            "fea": {2: "feature_frames"},
        },
        opset_version=opset,
    )
    torch.onnx.export(
        model.cfm,
        (
            torch.randn((1, 100, 64)),
            torch.randn((1, 100, 64)),
            torch.tensor([0.5], dtype=torch.float32),
            torch.randn((1, 512, 64)),
        ),
        output_dir / f"{name}_vits_cfm.onnx",
        input_names=["x", "cond", "t", "mu"],
        output_names=["v_pred"],
        dynamic_axes={
            "x": {0: "batch", 2: "frames"},
            "cond": {0: "batch", 2: "frames"},
            "t": {0: "batch"},
            "mu": {0: "batch", 2: "frames"},
            "v_pred": {0: "batch", 2: "frames"},
        },
        opset_version=opset,
    )
    torch.onnx.export(
        VocoderWrapper().eval(),
        (torch.randn((1, 100, 64), dtype=torch.float32),),
        output_dir / f"{name}_vocoder.onnx",
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={"mel": {2: "mel_frames"}, "audio": {0: "audio_samples"}},
        opset_version=opset,
    )


def validate_export_args(args: argparse.Namespace) -> tuple[Path, str]:
    if args.opset < 17:
        raise ValueError("--opset must be at least 17.")
    output_dir = args.output_dir
    name = args.name or output_dir.resolve().name
    if (
        not name
        or Path(name).name != name
        or any(char in name for char in '<>:"/\\|?*')
    ):
        raise ValueError("--name must be a filename-safe prefix, not a path.")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("--output-dir must be a directory.")
    if not args.gpt_model.is_file():
        raise FileNotFoundError(f"GPT checkpoint not found: {args.gpt_model}")
    if not args.sovits_model.is_file():
        raise FileNotFoundError(f"SoVITS checkpoint not found: {args.sovits_model}")
    return output_dir, name


def export(args: argparse.Namespace) -> None:
    output_dir, name = validate_export_args(args)

    import torch

    checkpoint = load_sovits_checkpoint(args.sovits_model)
    version = detect_sovits_version(args.sovits_model, checkpoint, args.version)
    vits, vits_config = build_vits_model(
        checkpoint, version, args.pretrained_sovits_model
    )
    if version in V5_VERSIONS:
        vocoder = build_v5_vocoder(args.vocoder_model)
    t2s = build_t2s_model(args.gpt_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    inputs = (
        torch.ones((1, 8), dtype=torch.long),
        torch.ones((1, 16), dtype=torch.long),
        torch.randn((8, 1024), dtype=torch.float32),
        torch.randn((16, 1024), dtype=torch.float32),
        torch.randn((1, 768, 100), dtype=torch.float32),
    )
    _export_t2s(torch, t2s, vits, inputs, output_dir, name, args.opset)
    if version in V5_VERSIONS:
        _export_v5_acoustic(
            torch, vits, vocoder, vits_config, output_dir, name, args.opset
        )
    else:
        _export_vits(torch, vits, vits_config, version, output_dir, name, args.opset)

    # Retain the legacy metadata keys used by existing v1/v2 consumers.
    metadata = {
        "Folder": name,
        "Name": name,
        "Type": "GPT-SoVits",
        "Rate": vits_config["sampling_rate"],
        "NumLayers": t2s.num_layers,
        "EmbeddingDim": t2s.embedding_dim,
        "Dict": "BasicDict",
        "BertPath": "chinese-roberta-wwm-ext-large",
        "AddBlank": False,
        "Version": version,
        "Opset": args.opset,
        "RequiresSpeakerEmbedding": version in PRO_VERSIONS,
        "SpeakerEmbeddingSize": 20480 if version in PRO_VERSIONS else None,
        "SpeakerEmbeddingKind": (
            "SV.compute_embedding3 flattened fbank features"
            if version in PRO_VERSIONS
            else None
        ),
        "BatchSize": 1,
        "VitsReferenceInput": "reference_spectrogram",
        "ReferenceSpectrogramShape": [
            1,
            vits_config["filter_length"] // 2 + 1,
            "reference_frames",
        ],
        "VitsSpeed": 1,
        "VitsNoiseShape": [1, vits_config["inter_channels"], "semantic_length * 2"],
        "T2SSamplingRandomShape": [t2s.vocab_size],
        "T2SSamplingRandomDistribution": "uniform (0, 1)",
        "RuntimeMustStopOnEOS": True,
        "RuntimeMaxSemanticTokens": int(t2s.max_sec * 50),
    }
    if version in V5_VERSIONS:
        metadata.update(
            ReferenceSamplingRate=vits_config["reference_sampling_rate"],
            VitsNoiseShape=[1, 100, "cfm_total_frames"],
            VitsSampleSteps=vits_config["sample_steps"],
            VitsCfgRate=vits_config["cfg_rate"],
            VitsNoiseTemperature=vits_config["noise_temperature"],
            AcousticGraphs=[
                f"{name}_vits_encp.onnx",
                f"{name}_vits_cfm.onnx",
                f"{name}_vocoder.onnx",
            ],
        )
    (output_dir / f"{name}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    export(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

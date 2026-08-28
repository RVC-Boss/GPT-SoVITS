from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "GPT_SoVITS" / "onnx_export.py"
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
spec = importlib.util.spec_from_file_location("onnx_export_under_test", MODULE_PATH)
onnx_export = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(onnx_export)


def cli_args(tmp_path, *extra):
    gpt = tmp_path / "g.ckpt"
    sovits = tmp_path / "s.pth"
    gpt.touch()
    sovits.touch()
    return onnx_export.build_parser().parse_args(
        [
            "--gpt-model",
            str(gpt),
            "--sovits-model",
            str(sovits),
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_import_has_no_model_side_effects():
    assert "torch" not in onnx_export.__dict__


def test_cli_defaults_to_opset_17(tmp_path):
    args = cli_args(tmp_path)
    assert args.opset == 17
    assert args.version == "auto"


def test_cli_rejects_invalid_name_before_loading_models(tmp_path, monkeypatch):
    args = cli_args(tmp_path, "--name", "bad/name")
    called = False

    def must_not_load(*_args):
        nonlocal called
        called = True
        raise AssertionError("model loading must not run")

    monkeypatch.setattr(onnx_export, "detect_sovits_version", must_not_load)
    with pytest.raises(ValueError, match="filename-safe"):
        onnx_export.export(args)
    assert not called


def test_cli_rejects_opset_below_17(tmp_path):
    with pytest.raises(ValueError, match="at least 17"):
        onnx_export.validate_export_args(cli_args(tmp_path, "--opset", "16"))


@pytest.mark.parametrize(
    ("header", "version"),
    [(b"00", "v1"), (b"01", "v2"), (b"05", "v2Pro"), (b"06", "v2ProPlus")],
)
def test_detects_supported_two_byte_headers(tmp_path, header, version):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(header + b"synthetic checkpoint payload")
    assert onnx_export.detect_sovits_version(checkpoint) == version


@pytest.mark.parametrize("header", [b"02", b"03", b"04", b"99"])
def test_rejects_unsupported_two_byte_headers(tmp_path, header):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(header + b"synthetic checkpoint payload")
    with pytest.raises(ValueError, match="Unsupported"):
        onnx_export.detect_sovits_version(checkpoint)


def test_expected_version_must_match_header(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"05synthetic checkpoint payload")
    with pytest.raises(ValueError, match="--version requested v2"):
        onnx_export.detect_sovits_version(checkpoint, "v2")


@pytest.mark.parametrize("header", [b"05", b"06"])
def test_canonical_loader_restores_tagged_zip_header(tmp_path, header):
    checkpoint = tmp_path / "model.pth"
    torch.save({"config": {}, "weight": {}}, checkpoint)
    data = checkpoint.read_bytes()
    checkpoint.write_bytes(header + data[2:])
    assert onnx_export.load_sovits_checkpoint(checkpoint) == {
        "config": {},
        "weight": {},
    }


@pytest.mark.parametrize(
    ("version", "deleted_key"),
    [("v2", "dec.weight"), ("v2Pro", "sv_emb.weight")],
)
def test_deleting_regular_or_pro_checkpoint_key_fails_model_build(
    monkeypatch, version, deleted_key
):
    class FakeSynthesizer:
        inter_channels = 192

        def __init__(self, *_args, **kwargs):
            self.version = kwargs["version"]

        def eval(self):
            return self

        def load_state_dict(self, weight, strict):
            expected = {"dec.weight"}
            if self.version in onnx_export.PRO_VERSIONS:
                expected.add("sv_emb.weight")
            return SimpleNamespace(
                missing_keys=sorted(expected - weight.keys()), unexpected_keys=[]
            )

    monkeypatch.setitem(
        sys.modules,
        "module.models_onnx",
        SimpleNamespace(SynthesizerTrn=FakeSynthesizer),
    )
    weight = {"dec.weight": torch.tensor(1), "sv_emb.weight": torch.tensor(1)}
    del weight[deleted_key]
    checkpoint = {
        "config": {
            "model": {},
            "data": {
                "filter_length": 2048,
                "hop_length": 640,
                "win_length": 2048,
                "sampling_rate": 32000,
                "n_speakers": 0,
            },
            "train": {"segment_size": 20480},
        },
        "weight": weight,
    }
    with pytest.raises(
        ValueError, match=f"Missing SoVITS checkpoint keys.*{deleted_key}"
    ):
        onnx_export.build_vits_model(checkpoint, version)


def test_only_enc_q_unexpected_state_keys_are_allowed():
    onnx_export.validate_vits_state_dict(
        SimpleNamespace(missing_keys=[], unexpected_keys=["enc_q.pre.weight"])
    )
    with pytest.raises(ValueError, match="Unexpected SoVITS checkpoint keys"):
        onnx_export.validate_vits_state_dict(
            SimpleNamespace(missing_keys=[], unexpected_keys=["dec.unknown"])
        )


@pytest.mark.parametrize("version", ["v2", "v2Pro"])
def test_vits_export_contract_has_explicit_noise_and_opset_17(
    tmp_path, monkeypatch, version
):
    calls = []
    monkeypatch.setattr(
        torch.onnx, "export", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    model = torch.nn.Identity()
    config = {
        "sampling_rate": 32000,
        "filter_length": 2048,
        "hop_length": 640,
        "win_length": 2048,
        "inter_channels": 192,
    }
    onnx_export._export_vits(torch, model, config, version, tmp_path, "example", 17)
    export_args, kwargs = calls[0]
    model_inputs = export_args[1]
    assert kwargs["opset_version"] == 17
    assert kwargs["input_names"][:4] == [
        "text",
        "semantic",
        "reference_spectrogram",
        "noise",
    ]
    assert model_inputs[2].shape == (1, 1025, 100)
    assert model_inputs[3].shape == (1, 192, 32)
    assert kwargs["dynamic_axes"]["reference_spectrogram"] == {2: "reference_frames"}
    assert kwargs["dynamic_axes"]["noise"] == {2: "vits_frame_length"}
    assert "speaker_embedding" not in kwargs["dynamic_axes"]
    if version == "v2Pro":
        assert kwargs["input_names"][-1] == "speaker_embedding"
        assert model_inputs[-1].shape == (1, 20480)


class TinyEncoder(torch.nn.Module):
    def forward(self, phonemes, bert):
        return torch.zeros((1, phonemes.shape[1], 4))


class TinyVits(torch.nn.Module):
    def extract_latent(self, ssl):
        return torch.ones((1, 1, 4), dtype=torch.long)


class GrowingFirst(torch.nn.Module):
    def forward(self, x, prompts, sampling_random):
        y = torch.cat([prompts, torch.ones((1, 1), dtype=torch.long)], 1)
        k = torch.zeros((2, x.shape[1], 1, 4))
        v = torch.zeros_like(k)
        y_emb = torch.zeros((1, y.shape[1], 4))
        return y, k, v, y_emb, x[:, :, 0]


class GrowingStage(torch.nn.Module):
    def forward(self, y, k, v, y_emb, x_example, sampling_random):
        next_y = torch.cat([y, torch.ones((1, 1), dtype=torch.long)], 1)
        next_k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1))
        next_v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1))
        next_emb = torch.cat([y_emb, torch.zeros((1, 1, y_emb.shape[2]))], 1)
        return next_y, next_k, next_v, next_emb, torch.zeros((1, 8)), next_y[:, -1:]


def tiny_t2s():
    return SimpleNamespace(
        onnx_encoder=TinyEncoder(),
        first_stage_decoder=GrowingFirst(),
        stage_decoder=GrowingStage(),
        vocab_size=8,
    )


def test_t2s_export_contract_has_sampling_inputs_and_dynamic_outputs(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        torch.onnx, "export", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    inputs = (
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 3), dtype=torch.long),
        torch.zeros((2, 1024)),
        torch.zeros((3, 1024)),
        torch.zeros((1, 768, 4)),
    )
    onnx_export._export_t2s(
        torch, tiny_t2s(), TinyVits(), inputs, tmp_path, "example", 17
    )
    assert len(calls) == 3
    first = calls[1][1]
    stage = calls[2][1]
    assert first["input_names"][-1] == "sampling_random"
    assert stage["input_names"][-1] == "sampling_random"
    assert first["opset_version"] == stage["opset_version"] == 17
    assert {"y", "k", "v", "y_emb", "x_example"} <= first["dynamic_axes"].keys()
    assert {"y", "k", "v", "y_emb", "samples"} <= stage["dynamic_axes"].keys()


def test_t2s_recurrent_feedback_lengths_grow():
    first = GrowingFirst()
    stage = GrowingStage()
    outputs = first(
        torch.zeros((1, 3, 4)), torch.ones((1, 2), dtype=torch.long), torch.rand(8)
    )
    initial_y, initial_k, _, initial_emb, _ = outputs
    outputs = stage(*outputs, torch.rand(8))
    outputs = stage(*outputs[:4], outputs[4] * 0, torch.rand(8))
    assert outputs[0].shape[1] == initial_y.shape[1] + 2
    assert outputs[1].shape[1] == initial_k.shape[1] + 2
    assert outputs[3].shape[1] == initial_emb.shape[1] + 2


def test_explicit_t2s_sampling_is_reproducible():
    from AR.models.t2s_model_onnx import multinomial_sample_one_no_sync

    probabilities = torch.tensor([0.1, 0.3, 0.6])
    random_values = torch.tensor([0.2, 0.4, 0.8])
    first = multinomial_sample_one_no_sync(probabilities, random_values)
    second = multinomial_sample_one_no_sync(probabilities, random_values)
    assert torch.equal(first, second)


def test_export_metadata_documents_runtime_contract(tmp_path, monkeypatch):
    class DummyT2S:
        num_layers = 12
        embedding_dim = 512
        max_sec = 10
        vocab_size = 1025

    args = cli_args(tmp_path, "--name", "example")
    monkeypatch.setattr(onnx_export, "detect_sovits_version", lambda *_: "v2Pro")
    monkeypatch.setattr(onnx_export, "load_sovits_checkpoint", lambda *_: {})
    monkeypatch.setattr(
        onnx_export,
        "build_vits_model",
        lambda *_: (
            object(),
            {
                "sampling_rate": 32000,
                "filter_length": 2048,
                "hop_length": 640,
                "win_length": 2048,
                "inter_channels": 192,
            },
        ),
    )
    monkeypatch.setattr(onnx_export, "build_t2s_model", lambda *_: DummyT2S())
    monkeypatch.setattr(onnx_export, "_export_t2s", lambda *_: None)
    monkeypatch.setattr(onnx_export, "_export_vits", lambda *_: None)
    onnx_export.export(args)
    metadata = json.loads(
        (args.output_dir / "example.json").read_text(encoding="utf-8")
    )
    assert metadata["Opset"] == 17
    assert metadata["BatchSize"] == 1
    assert metadata["VitsReferenceInput"] == "reference_spectrogram"
    assert metadata["ReferenceSpectrogramShape"] == [1, 1025, "reference_frames"]
    assert metadata["VitsSpeed"] == 1
    assert metadata["VitsNoiseShape"] == [1, 192, "semantic_length * 2"]
    assert metadata["RequiresSpeakerEmbedding"] is True
    assert metadata["SpeakerEmbeddingSize"] == 20480
    assert "flattened fbank" in metadata["SpeakerEmbeddingKind"]
    assert metadata["T2SSamplingRandomShape"] == [1025]
    assert metadata["RuntimeMustStopOnEOS"] is True
    assert metadata["RuntimeMaxSemanticTokens"] == 500


def test_minimal_vits_wrapper_real_onnx_and_ort(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    class TinyVitsDecoder(torch.nn.Module):
        def forward(self, semantic, text, reference_spectrogram, noise=None):
            value = (
                semantic.float().mean()
                + text.float().mean()
                + reference_spectrogram.mean()
                + noise.mean()
            )
            return value.reshape(1, 1, 1).expand(1, 1, 4)

    wrapper = onnx_export._make_vits_wrapper(TinyVitsDecoder(), pro=False)
    output = tmp_path / "vits.onnx"
    inputs = (
        torch.ones((1, 3), dtype=torch.long),
        torch.ones((1, 1, 4), dtype=torch.long),
        torch.randn((1, 9, 5)),
        torch.randn((1, 4, 8)),
    )
    torch.onnx.export(
        wrapper,
        inputs,
        output,
        opset_version=17,
        input_names=["text", "semantic", "reference_spectrogram", "noise"],
        output_names=["audio"],
        dynamic_axes={
            "text": {1: "text_length"},
            "semantic": {2: "semantic_length"},
            "reference_spectrogram": {2: "reference_frames"},
            "noise": {2: "vits_frame_length"},
            "audio": {0: "audio_samples"},
        },
    )
    model = onnx.load(output)
    onnx.checker.check_model(model)
    operators = {node.op_type for node in model.graph.node}
    assert not any("STFT" in operator.upper() for operator in operators)
    assert not any("RANDOM" in operator.upper() for operator in operators)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    result = session.run(
        ["audio"],
        {
            "text": inputs[0].numpy(),
            "semantic": inputs[1].numpy(),
            "reference_spectrogram": inputs[2].numpy(),
            "noise": inputs[3].numpy(),
        },
    )[0]
    assert result.shape == (4,)
    assert torch.isfinite(torch.from_numpy(result)).all()


def test_t2s_sampling_core_real_onnx_and_ort(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from AR.models.t2s_model_onnx import sample

    class SamplingCore(torch.nn.Module):
        def forward(self, logits, previous_tokens, sampling_random):
            token, _probs = sample(
                logits,
                previous_tokens,
                sampling_random,
                top_p=1.0,
                repetition_penalty=1.1,
            )
            return token

    module = SamplingCore().eval()
    output = tmp_path / "t2s_sampling.onnx"
    logits = torch.tensor([0.2, 1.0, -0.4, 0.7], dtype=torch.float32)
    previous_tokens = torch.tensor([1, 3], dtype=torch.long)
    sampling_random = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
    torch.onnx.export(
        module,
        (logits.clone(), previous_tokens, sampling_random),
        output,
        opset_version=17,
        input_names=["logits", "previous_tokens", "sampling_random"],
        output_names=["token"],
    )

    model = onnx.load(output)
    onnx.checker.check_model(model)
    assert "sampling_random" in {value.name for value in model.graph.input}
    operators = {node.op_type for node in model.graph.node}
    assert not any("RANDOM" in operator.upper() for operator in operators)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    feed = {
        "logits": logits.numpy(),
        "previous_tokens": previous_tokens.numpy(),
        "sampling_random": sampling_random.numpy(),
    }
    ort_token = session.run(["token"], feed)[0]
    torch_token = module(logits.clone(), previous_tokens, sampling_random).numpy()
    assert ort_token.shape == torch_token.shape
    assert (ort_token == torch_token).all()

    feed["sampling_random"] = torch.tensor(
        [0.8, 0.6, 0.4, 0.2], dtype=torch.float32
    ).numpy()
    second_token = session.run(["token"], feed)[0]
    assert second_token.shape == torch_token.shape


def test_vits_wrapper_forwards_explicit_noise(monkeypatch):
    class DummyVits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise = None

        def forward(self, semantic, text, refer, noise=None):
            self.noise = noise
            return torch.zeros((1, 1, 4))

    model = DummyVits()
    wrapper = onnx_export._make_vits_wrapper(model, pro=False)
    noise = torch.randn(1, 4, 6)
    wrapper(
        torch.ones((1, 3), dtype=torch.long),
        torch.ones((1, 1, 3), dtype=torch.long),
        torch.randn(1, 4, 8),
        noise,
    )
    assert model.noise is noise


def test_python_waveform_helper_produces_magnitude_spectrogram():
    spectrogram = onnx_export.waveform_to_spectrogram(
        torch.randn(1, 64), n_fft=16, hop_size=4, win_size=16
    )
    assert spectrogram.ndim == 3
    assert spectrogram.shape[:2] == (1, 9)
    assert not torch.is_complex(spectrogram)


def test_full_t2s_forward_uses_compatible_split_decoder_signatures():
    from AR.models.t2s_model_onnx import Text2SemanticDecoder

    class First:
        def __init__(self):
            self.called = False

        def __call__(self, x, prompts, sampling_random=None):
            self.called = True
            return (
                prompts,
                torch.zeros((1, 1, 1, 1)),
                torch.zeros((1, 1, 1, 1)),
                torch.zeros((1, 1, 1)),
                torch.zeros((1, x.shape[1])),
            )

    class Stage:
        def __init__(self):
            self.called = False

        def __call__(self, y, k, v, y_emb, x_example, sampling_random=None):
            self.called = True
            sample = torch.tensor([[2]])
            return (
                torch.cat([y, sample], 1),
                k,
                v,
                y_emb,
                torch.tensor([[0.0, 0.0, 1.0]]),
                sample,
            )

    first = First()
    stage = Stage()
    decoder = SimpleNamespace(
        early_stop_num=torch.tensor([-1]),
        EOS=2,
        onnx_encoder=lambda x, bert: x,
        first_stage_decoder=first,
        stage_decoder=stage,
    )
    output, steps = Text2SemanticDecoder.forward(
        decoder,
        torch.zeros((1, 2, 4)),
        torch.ones((1, 1), dtype=torch.long),
        torch.zeros((1, 1024, 2)),
    )
    assert first.called and stage.called
    assert output.shape[1] == 2
    assert steps == 1


def test_full_t2s_infer_sample_fallback_has_no_signature_error():
    from AR.models.t2s_model_onnx import Text2SemanticDecoder

    class Transformer:
        def __call__(self, x, mask=None, cache=None):
            return x

    class PredictEOS:
        def __call__(self, x):
            logits = torch.zeros((x.shape[0], 3))
            logits[:, 2] = 10
            return logits

    decoder = SimpleNamespace(
        top_k=torch.tensor([1]),
        early_stop_num=torch.tensor([-1]),
        num_layers=1,
        EOS=2,
        onnx_encoder=lambda x, bert: x,
        ar_audio_embedding=torch.nn.Embedding(3, 4),
        ar_audio_position=torch.nn.Identity(),
        h=Transformer(),
        ar_predict_layer=PredictEOS(),
    )
    output, steps = Text2SemanticDecoder.infer(
        decoder,
        torch.zeros((1, 2, 4)),
        torch.ones((1, 1), dtype=torch.long),
        torch.zeros((1, 1024, 2)),
    )
    assert output.shape[1] == 2
    assert steps == 0

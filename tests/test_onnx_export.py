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
sys.path[:0] = [str(ROOT), str(ROOT / "GPT_SoVITS")]
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
    ("detected_version", "embedding_rows", "resolved_version"),
    [
        ("v2", 322, "v1"),
        ("v1", 400, "v2"),
        ("v2", None, "v2"),
        ("v2Pro", 322, "v2Pro"),
        ("v2ProPlus", 400, "v2ProPlus"),
    ],
)
def test_resolves_inference_version_from_detector_and_checkpoint_structure(
    tmp_path, monkeypatch, detected_version, embedding_rows, resolved_version
):
    import process_ckpt

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    monkeypatch.setattr(
        process_ckpt,
        "get_sovits_version_from_path_fast",
        lambda _path: ("v2", detected_version, False),
    )
    weight = {}
    if embedding_rows is not None:
        weight["enc_p.text_embedding.weight"] = torch.empty((embedding_rows, 192))

    assert (
        onnx_export.detect_sovits_version(checkpoint_path, {"weight": weight})
        == resolved_version
    )


@pytest.mark.parametrize(
    ("detected_version", "is_lora"),
    [("v3", False), ("v4", False), ("unknown", False), ("v2", True)],
)
def test_rejects_lora_and_unsupported_detector_versions(
    tmp_path, monkeypatch, detected_version, is_lora
):
    import process_ckpt

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    monkeypatch.setattr(
        process_ckpt,
        "get_sovits_version_from_path_fast",
        lambda _path: ("v2", detected_version, is_lora),
    )
    with pytest.raises(ValueError, match="Unsupported"):
        onnx_export.detect_sovits_version(checkpoint_path, {"weight": {}})


def test_expected_version_must_match_resolved_structure(tmp_path, monkeypatch):
    import process_ckpt

    checkpoint_path = tmp_path / "official-v1.pth"
    checkpoint_path.touch()
    monkeypatch.setattr(
        process_ckpt,
        "get_sovits_version_from_path_fast",
        lambda _path: ("v2", "v2", False),
    )
    checkpoint = {"weight": {"enc_p.text_embedding.weight": torch.empty((322, 192))}}
    with pytest.raises(ValueError, match="Checkpoint is v1.*--version requested v2"):
        onnx_export.detect_sovits_version(checkpoint_path, checkpoint, "v2")


@pytest.mark.parametrize("header", [b"05", b"06"])
def test_canonical_loader_restores_tagged_zip_header(tmp_path, header):
    checkpoint = tmp_path / "model.pth"
    with checkpoint.open("wb") as checkpoint_file:
        torch.save({"config": {}, "weight": {}}, checkpoint_file)
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


@pytest.mark.parametrize("length", [3, 5, 7])
def test_relative_embeddings_narrow_matches_original_slice(length):
    from module.attentions_onnx import MultiHeadAttention

    attention = MultiHeadAttention(4, 4, 2, window_size=4).eval()
    relative = torch.arange(18, dtype=torch.float32).reshape(1, 9, 2)
    pad_length = max(length - (attention.window_size + 1), 0)
    slice_start = max((attention.window_size + 1) - length, 0)
    padded = torch.nn.functional.pad(relative, (0, 0, pad_length, pad_length, 0, 0))
    expected = padded[:, slice_start : slice_start + 2 * length - 1]

    actual = attention._get_relative_embeddings(relative, length)

    assert actual.shape == (1, 2 * length - 1, 2)
    assert torch.equal(actual, expected)


def test_relative_embeddings_exports_rank1_slice_starts(tmp_path):
    onnx = pytest.importorskip("onnx")
    from module.attentions_onnx import MultiHeadAttention

    class RelativeEmbeddingSlice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MultiHeadAttention(4, 4, 2, window_size=4).eval()

        def forward(self, x):
            return self.attention._get_relative_embeddings(
                self.attention.emb_rel_k, x.size(1)
            )

    model = RelativeEmbeddingSlice().eval()
    output_path = tmp_path / "relative_embeddings.onnx"
    torch.onnx.export(
        model,
        torch.zeros((1, 5)),
        output_path,
        input_names=["x"],
        output_names=["relative_embeddings"],
        dynamic_axes={
            "x": {1: "length"},
            "relative_embeddings": {1: "relative_length"},
        },
        opset_version=17,
    )

    exported = onnx.shape_inference.infer_shapes(onnx.load(output_path))
    onnx.checker.check_model(exported)
    ranks = {}
    for value in [
        *exported.graph.input,
        *exported.graph.output,
        *exported.graph.value_info,
    ]:
        ranks[value.name] = len(value.type.tensor_type.shape.dim)
    for initializer in exported.graph.initializer:
        ranks[initializer.name] = len(initializer.dims)
    for node in exported.graph.node:
        if node.op_type == "Constant":
            tensor = next(
                (
                    attribute.t
                    for attribute in node.attribute
                    if attribute.name == "value"
                ),
                None,
            )
            if tensor is not None:
                ranks[node.output[0]] = len(tensor.dims)

    slices = [node for node in exported.graph.node if node.op_type == "Slice"]
    assert slices
    for node in slices:
        assert (
            ranks.get(node.input[1]) == 1
        ), f"Slice starts {node.input[1]!r} must have rank 1; ranks={ranks}"


def test_relative_embeddings_ort_dynamic_length_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from module.attentions_onnx import MultiHeadAttention

    class RelativeEmbeddingSlice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MultiHeadAttention(4, 4, 2, window_size=4).eval()

        def forward(self, x):
            return self.attention._get_relative_embeddings(
                self.attention.emb_rel_k, x.size(1)
            )

    model = RelativeEmbeddingSlice().eval()
    output_path = tmp_path / "relative_embeddings_ort.onnx"
    torch.onnx.export(
        model,
        torch.zeros((1, 5)),
        output_path,
        input_names=["x"],
        output_names=["relative_embeddings"],
        dynamic_axes={
            "x": {1: "length"},
            "relative_embeddings": {1: "relative_length"},
        },
        opset_version=17,
    )
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    for length in (3, 7):
        x = torch.zeros((1, length))
        expected = model(x)
        actual = session.run(["relative_embeddings"], {"x": x.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), expected)


def test_onnx_encoder_does_not_register_unused_speaker_projection():
    from module.attentions_onnx import Encoder

    encoder = Encoder(8, 16, 2, 2).eval()
    official_state = {
        key: value
        for key, value in encoder.state_dict().items()
        if "spk_emb_linear" not in key
    }
    incompatible = encoder.load_state_dict(official_state, strict=True)

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert not any("spk_emb_linear" in key for key in encoder.state_dict())
    output = encoder(torch.randn((1, 8, 4)), torch.ones((1, 1, 4)))
    assert output.shape == (1, 8, 4)


def test_real_inference_layer_missing_key_still_fails_closed():
    missing_key = "enc_p.encoder_text.attn_layers.0.conv_q.weight"
    with pytest.raises(ValueError, match=missing_key):
        onnx_export.validate_vits_state_dict(
            SimpleNamespace(missing_keys=[missing_key], unexpected_keys=[])
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


@pytest.mark.parametrize("shape", [(5, 1, 8), (3, 2, 4)])
def test_mha_qkv_reshape_matches_original_unflatten_expression(shape):
    from AR.modules.patched_mha_with_cache_onnx import (
        _reshape_qkv_for_attention,
    )

    query = torch.randn(shape)
    projected = torch.randn((*shape[:-1], shape[-1] * 3))
    original = (
        projected.unflatten(-1, (3, query.size(-1)))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
        .contiguous()
    )
    replacement = _reshape_qkv_for_attention(projected, query)

    assert replacement.shape == (3, *shape)
    assert torch.equal(replacement, original)
    for index in range(3):
        assert torch.equal(replacement[index], original[index])


def test_mha_qkv_reshape_onnxruntime_dynamic_shape_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from AR.modules.patched_mha_with_cache_onnx import (
        _reshape_qkv_for_attention,
    )

    class QKVReshape(torch.nn.Module):
        def forward(self, projected, query):
            return _reshape_qkv_for_attention(projected, query)

    output_path = tmp_path / "qkv_reshape_ort.onnx"
    initial_query = torch.randn((5, 1, 8))
    initial_projected = torch.randn((5, 1, 24))
    torch.onnx.export(
        QKVReshape().eval(),
        (initial_projected, initial_query),
        output_path,
        input_names=["projected", "query"],
        output_names=["qkv"],
        dynamic_axes={
            "projected": {0: "sequence", 1: "batch", 2: "packed_embedding"},
            "query": {0: "sequence", 1: "batch", 2: "embedding"},
            "qkv": {1: "sequence", 2: "batch", 3: "embedding"},
        },
        opset_version=17,
    )

    query = torch.randn((3, 2, 4))
    projected = torch.randn((3, 2, 12))
    expected = _reshape_qkv_for_attention(projected, query).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["qkv"], {"projected": projected.numpy(), "query": query.numpy()}
    )[0]
    assert actual.shape == expected.shape
    assert actual == pytest.approx(expected)


def test_mha_qkv_reshape_exports_opset17_without_unflatten(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from AR.modules.patched_mha_with_cache_onnx import (
        _reshape_qkv_for_attention,
    )

    class QKVReshape(torch.nn.Module):
        def forward(self, projected, query):
            return _reshape_qkv_for_attention(projected, query)

    model = QKVReshape().eval()
    projected = torch.randn((4, 2, 24))
    query = torch.randn((4, 2, 8))
    output_path = tmp_path / "qkv_reshape.onnx"
    torch.onnx.export(
        model,
        (projected, query),
        output_path,
        input_names=["projected", "query"],
        output_names=["qkv"],
        dynamic_axes={
            "projected": {0: "sequence", 1: "batch"},
            "query": {0: "sequence", 1: "batch"},
            "qkv": {1: "sequence", 2: "batch"},
        },
        opset_version=17,
    )

    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    assert all(node.op_type != "Unflatten" for node in exported.graph.node)

    dynamic_projected = torch.randn((7, 1, 24))
    dynamic_query = torch.randn((7, 1, 8))
    expected = model(dynamic_projected, dynamic_query).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            "projected": dynamic_projected.numpy(),
            "query": dynamic_query.numpy(),
        },
    )[0]
    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))


@pytest.mark.parametrize(
    ("query_length", "key_length"),
    [(4, 4), (1, 5)],
)
def test_onnx_attention_matches_sdpa_with_float_mask(query_length, key_length):
    from AR.modules.patched_mha_with_cache_onnx import (
        _scaled_dot_product_attention_onnx,
    )

    q = torch.randn((1, 2, query_length, 8))
    k = torch.randn((1, 2, key_length, 8))
    v = torch.randn((1, 2, key_length, 8))
    mask = torch.zeros((1, 1, query_length, key_length))
    mask[..., -1] = float("-inf")

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, mask, dropout_p=0.0, is_causal=False
    )
    actual = _scaled_dot_product_attention_onnx(q, k, v, mask)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_onnx_attention_rejects_causal_hint():
    from AR.modules.patched_mha_with_cache_onnx import (
        _scaled_dot_product_attention_onnx,
    )

    tensor = torch.randn((1, 1, 2, 4))
    with pytest.raises(ValueError, match="is_causal=False"):
        _scaled_dot_product_attention_onnx(
            tensor, tensor, tensor, torch.zeros((1, 1, 2, 2)), is_causal=True
        )


def test_onnx_attention_exports_basic_opset17_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    from AR.modules.patched_mha_with_cache_onnx import (
        _scaled_dot_product_attention_onnx,
    )

    class TinyAttention(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return _scaled_dot_product_attention_onnx(q, k, v, mask)

    model = TinyAttention().eval()
    q = torch.randn((1, 2, 3, 8))
    k = torch.randn((1, 2, 5, 8))
    v = torch.randn((1, 2, 5, 8))
    mask = torch.zeros((1, 1, 3, 5))
    output_path = tmp_path / "basic_attention.onnx"
    dynamic_axes = {
        "q": {2: "query_length"},
        "k": {2: "key_length"},
        "v": {2: "key_length"},
        "mask": {2: "query_length", 3: "key_length"},
        "output": {2: "query_length"},
    }
    torch.onnx.export(
        model,
        (q, k, v, mask),
        output_path,
        input_names=["q", "k", "v", "mask"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    op_types = {node.op_type for node in exported.graph.node}
    assert {"MatMul", "Transpose", "Add", "Softmax"} <= op_types
    assert "ScaledDotProductAttention" not in op_types
    assert not any("Random" in op_type or op_type == "Dropout" for op_type in op_types)


def test_onnx_attention_ort_dynamic_length_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from AR.modules.patched_mha_with_cache_onnx import (
        _scaled_dot_product_attention_onnx,
    )

    class TinyAttention(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return _scaled_dot_product_attention_onnx(q, k, v, mask)

    model = TinyAttention().eval()
    output_path = tmp_path / "basic_attention_ort.onnx"
    initial = (
        torch.randn((1, 2, 3, 8)),
        torch.randn((1, 2, 5, 8)),
        torch.randn((1, 2, 5, 8)),
        torch.zeros((1, 1, 3, 5)),
    )
    torch.onnx.export(
        model,
        initial,
        output_path,
        input_names=["q", "k", "v", "mask"],
        output_names=["output"],
        dynamic_axes={
            "q": {2: "query_length"},
            "k": {2: "key_length"},
            "v": {2: "key_length"},
            "mask": {2: "query_length", 3: "key_length"},
            "output": {2: "query_length"},
        },
        opset_version=17,
    )

    q = torch.randn((1, 2, 1, 8))
    k = torch.randn((1, 2, 7, 8))
    v = torch.randn((1, 2, 7, 8))
    mask = torch.zeros((1, 1, 1, 7))
    mask[..., -2:] = float("-inf")
    expected = model(q, k, v, mask)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["output"],
        {
            "q": q.numpy(),
            "k": k.numpy(),
            "v": v.numpy(),
            "mask": mask.numpy(),
        },
    )[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-5, atol=1e-6)


def test_build_t2s_model_uses_constant_int_top_k_in_real_sampling_path(
    tmp_path, monkeypatch
):
    from AR.models.t2s_model_onnx import Text2SemanticDecoder, sample

    class FakeModel:
        def __init__(self):
            self.ar_text_embedding = torch.nn.Identity()
            self.bert_proj = torch.nn.Identity()
            self.ar_text_position = torch.nn.Identity()
            self.ar_audio_embedding = torch.nn.Identity()
            self.ar_audio_position = torch.nn.Identity()
            self.h = torch.nn.Identity()
            self.ar_predict_layer = torch.nn.Identity()
            self.loss_fct = torch.nn.Identity()
            self.ar_accuracy_metric = torch.nn.Identity()
            self.num_layers = 1

        def init_onnx(self):
            Text2SemanticDecoder.init_onnx(self)

    class FakeLightning:
        def __init__(self, config, _name, is_train):
            assert is_train is False
            self.model = FakeModel()

        def load_state_dict(self, weight):
            assert weight == {"model.weight": "loaded"}

        def eval(self):
            return self

    config = {
        "inference": {"top_k": 3},
        "data": {"max_sec": 10},
    }
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "config": config,
            "weight": {"model.weight": "loaded"},
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "AR.models.t2s_lightning_module_onnx",
        SimpleNamespace(Text2SemanticLightningModule=FakeLightning),
    )

    model = onnx_export.build_t2s_model(tmp_path / "gpt.ckpt")

    assert type(model.top_k) is int
    assert model.top_k == 3
    assert model.first_stage_decoder.top_k == 3
    assert model.stage_decoder.top_k == 3
    token, probabilities = sample(
        torch.arange(8, dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.full((8,), 0.5),
        top_k=model.first_stage_decoder.top_k,
    )
    assert token.shape == (1,)
    assert torch.count_nonzero(probabilities) == 3


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
        top_k=1,
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


@pytest.mark.parametrize("version", ["v5dev", "v5turbo"])
@pytest.mark.parametrize("is_lora", [False, True])
def test_v5_detects_base_and_lora(tmp_path, monkeypatch, version, is_lora):
    import process_ckpt

    path = tmp_path / "v5.pth"
    path.touch()
    monkeypatch.setattr(
        process_ckpt,
        "get_sovits_version_from_path_fast",
        lambda _: ("v2", version, is_lora),
    )
    assert onnx_export.detect_sovits_version(path, {"weight": {}}, version) == version
    with pytest.raises(ValueError, match="--version requested"):
        onnx_export.detect_sovits_version(path, {"weight": {}}, "v2")


def _tiny_v5_dit(text_dim=32):
    from f5_tts.model import DiT

    return DiT(
        dim=64,
        depth=2,
        heads=4,
        dim_head=16,
        ff_mult=2,
        text_dim=text_dim,
        conv_layers=2,
        use_step_embedding=False,
    ).eval()


def _strict_onnx_session(path):
    import onnx
    import onnxruntime as ort

    graph = onnx.shape_inference.infer_shapes(
        onnx.load(path), check_type=True, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(graph, full_check=True)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_v5_cfm_real_onnx_and_ort(tmp_path):
    from module.models_onnx import CFMV5Onnx

    torch.manual_seed(19)
    original = _tiny_v5_dit()
    adapted = _tiny_v5_dit()
    adapted.load_state_dict(original.state_dict())
    model = CFMV5Onnx(100, adapted).eval()
    assert model.estimator.state_dict().keys() == original.state_dict().keys()
    initial = (
        torch.randn(1, 100, 24),
        torch.randn(1, 100, 24),
        torch.tensor([0.5]),
        torch.randn(1, 32, 24),
    )
    path = tmp_path / "cfm.onnx"
    torch.onnx.export(
        model,
        initial,
        path,
        opset_version=17,
        input_names=["x", "cond", "t", "mu"],
        output_names=["v_pred"],
        dynamic_axes={
            "x": {0: "batch", 2: "frames"},
            "cond": {0: "batch", 2: "frames"},
            "t": {0: "batch"},
            "mu": {0: "batch", 2: "frames"},
            "v_pred": {0: "batch", 2: "frames"},
        },
    )
    session = _strict_onnx_session(path)
    dynamic = (
        torch.randn(2, 100, 35),
        torch.randn(2, 100, 35),
        torch.tensor([0.25, 0.75]),
        torch.randn(2, 32, 35),
    )
    with torch.no_grad():
        expected = original(
            dynamic[0], dynamic[1], torch.tensor([35, 35]), dynamic[2], text0=dynamic[3]
        ).transpose(2, 1)
        adapted_output = model(*dynamic)
    actual = torch.from_numpy(
        session.run(
            None, dict(zip(["x", "cond", "t", "mu"], [x.numpy() for x in dynamic]))
        )[0]
    )
    assert (adapted_output - expected).abs().max().item() < 1e-5
    assert (actual - expected).abs().max().item() < 1e-5


def _v5_config():
    return {
        "model": dict(
            inter_channels=192,
            hidden_channels=192,
            filter_channels=192,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.0,
            resblock="1",
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_rates=[2],
            upsample_initial_channel=16,
            upsample_kernel_sizes=[4],
            gin_channels=512,
        ),
        "data": dict(
            filter_length=2048,
            hop_length=640,
            win_length=2048,
            sampling_rate=32000,
            n_speakers=0,
        ),
        "train": {"segment_size": 20480},
    }


def test_v5_encp_and_vocoder_real_onnx_and_ort(tmp_path, monkeypatch):
    from module import models_onnx

    monkeypatch.setattr(models_onnx, "DiT", lambda **_: _tiny_v5_dit(text_dim=512))
    config = _v5_config()
    model = models_onnx.SynthesizerTrnV3(
        1025, 32, semantic_frame_rate="25hz", version="v5dev", **config["model"]
    ).eval()
    vocoder = models_onnx.Generator(
        100, "1", [3], [[1, 3, 5]], [2], 16, [4], gin_channels=0, is_bias=True
    ).eval()
    vocoder.remove_weight_norm()
    onnx_export._export_v5_acoustic(
        torch, model, vocoder, {"filter_length": 2048}, tmp_path, "tiny", 17
    )
    encp = _strict_onnx_session(tmp_path / "tiny_vits_encp.onnx")
    cfm = _strict_onnx_session(tmp_path / "tiny_vits_cfm.onnx")
    assert [value.name for value in cfm.get_inputs()] == ["x", "cond", "t", "mu"]
    voc = _strict_onnx_session(tmp_path / "tiny_vocoder.onnx")
    text = torch.ones(1, 11, dtype=torch.long)
    codes = torch.randint(0, 1024, (1, 1, 19))
    refer = torch.randn(1, 1025, 43)
    with torch.no_grad():
        expected_fea, expected_ge = model(codes, text, refer)
    fea, ge = encp.run(
        None,
        {
            "text": text.numpy(),
            "semantic": codes.numpy(),
            "reference_spectrogram": refer.numpy(),
        },
    )
    assert fea.shape == (1, 512, 76)
    assert ge.shape == (1, 512, 1)
    torch.testing.assert_close(
        torch.from_numpy(fea), expected_fea, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(torch.from_numpy(ge), expected_ge, rtol=1e-4, atol=1e-5)
    mel = torch.randn(1, 100, 35)
    with torch.no_grad():
        expected_audio = vocoder(mel)[0, 0]
    audio = voc.run(None, {"mel": mel.numpy()})[0]
    torch.testing.assert_close(
        torch.from_numpy(audio), expected_audio, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("version", ["v5dev", "v5turbo"])
def test_v5_lora_build_merges_real_peft_and_requires_base(
    tmp_path, monkeypatch, version
):
    from module import models_onnx
    from peft import LoraConfig, get_peft_model

    class SmallSynthesizer(torch.nn.Module):
        inter_channels = 192

        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.cfm = models_onnx.CFMV5Onnx(100, _tiny_v5_dit())

    monkeypatch.setattr(models_onnx, "SynthesizerTrnV3", SmallSynthesizer)
    base = SmallSynthesizer()
    base_path = tmp_path / "base.pth"
    with base_path.open("wb") as base_file:
        torch.save({"config": _v5_config(), "weight": base.state_dict()}, base_file)
    base.cfm = get_peft_model(
        base.cfm,
        LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"], r=2, lora_alpha=2
        ),
    )
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
    weight = {
        name: value for name, value in base.state_dict().items() if "lora_" in name
    }
    checkpoint = {"config": _v5_config(), "weight": weight, "lora_rank": 2}
    with pytest.raises(FileNotFoundError, match="v5 base SoVITS checkpoint not found"):
        onnx_export.build_vits_model(checkpoint, version, tmp_path / "missing.pth")
    merged, config = onnx_export.build_vits_model(checkpoint, version, base_path)
    reference = base.cfm.merge_and_unload()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(merged.cfm.state_dict()[name], value)
    assert not any("lora_" in name for name in merged.state_dict())
    assert config["sampling_rate"] == 48000
    assert config["reference_sampling_rate"] == 32000
    assert config["sample_steps"] == (4 if version == "v5turbo" else 32)
    assert config["cfg_rate"] == (0.0 if version == "v5turbo" else 1.30)


def test_v5_mel_helpers_match_production_preprocessing():
    from module.mel_processing import mel_spectrogram_torch

    waveform = torch.randn(1, 6400) * 0.1
    expected = mel_spectrogram_torch(
        waveform, 1280, 100, 32000, 320, 1280, 0, None, center=False
    )
    normalized = onnx_export.waveform_to_mel_v5(waveform)
    torch.testing.assert_close(onnx_export.denorm_spec_v5(normalized), expected)


@pytest.mark.parametrize(
    "version,steps,cfg", [("v5dev", 32, 1.30), ("v5turbo", 4, 0.0)]
)
def test_v5_export_dispatches_three_graphs_and_metadata(
    tmp_path, monkeypatch, version, steps, cfg
):
    args = cli_args(
        tmp_path,
        "--name",
        "v5",
        "--pretrained-sovits-model",
        "base.pth",
        "--vocoder-model",
        "voc.pth",
    )
    acoustic, vocoder = object(), object()
    t2s = SimpleNamespace(num_layers=12, embedding_dim=512, max_sec=10, vocab_size=1025)
    config = dict(
        filter_length=2048,
        hop_length=640,
        win_length=2048,
        sampling_rate=48000,
        reference_sampling_rate=32000,
        inter_channels=192,
        sample_steps=steps,
        cfg_rate=cfg,
        noise_temperature=0.875,
    )
    monkeypatch.setattr(onnx_export, "load_sovits_checkpoint", lambda _: {})
    monkeypatch.setattr(onnx_export, "detect_sovits_version", lambda *_: version)

    def build(_checkpoint, actual_version, base_path):
        assert actual_version == version
        assert base_path == Path("base.pth")
        return acoustic, config

    vocoder_calls = []

    def build_vocoder(path):
        assert path == Path("voc.pth")
        vocoder_calls.append(path)
        return vocoder

    exported = []

    def export_acoustic(
        _torch, model, actual_vocoder, actual_config, directory, name, opset
    ):
        assert (
            model is acoustic and actual_vocoder is vocoder and actual_config is config
        )
        assert directory == args.output_dir and name == "v5" and opset == 17
        exported.append(name)

    monkeypatch.setattr(onnx_export, "build_vits_model", build)
    monkeypatch.setattr(onnx_export, "build_v5_vocoder", build_vocoder)
    monkeypatch.setattr(onnx_export, "build_t2s_model", lambda _: t2s)
    monkeypatch.setattr(onnx_export, "_export_t2s", lambda *_: None)
    monkeypatch.setattr(onnx_export, "_export_v5_acoustic", export_acoustic)
    onnx_export.export(args)
    assert exported == ["v5"]
    assert vocoder_calls == [Path("voc.pth")]
    metadata = json.loads((args.output_dir / "v5.json").read_text(encoding="utf-8"))
    assert metadata["Rate"] == 48000
    assert metadata["ReferenceSamplingRate"] == 32000
    assert metadata["VitsNoiseShape"] == [1, 100, "cfm_total_frames"]
    assert metadata["VitsSampleSteps"] == steps
    assert metadata["VitsCfgRate"] == cfg
    assert metadata["VitsNoiseTemperature"] == 0.875
    assert metadata["AcousticGraphs"] == [
        "v5_vits_encp.onnx",
        "v5_vits_cfm.onnx",
        "v5_vocoder.onnx",
    ]

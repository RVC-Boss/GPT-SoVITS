"""Checkpoint compatibility tests for the inference-only SoVITS models."""

from collections import OrderedDict
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from module.models import ResidualCouplingBlock, SynthesizerTrn, SynthesizerTrnV3
from module.modules import Flip, ResidualCouplingLayer
from module.quantize import ResidualVectorQuantizer


class QuantizerInferenceTests(unittest.TestCase):
    def test_legacy_ema_buffers_load_and_preserve_known_tokens_and_features(self):
        quantizer = ResidualVectorQuantizer(dimension=2, n_q=2, bins=3).eval()
        container = nn.ModuleDict({"voice": quantizer})
        weights = OrderedDict(container.state_dict())
        weights["voice.vq.layers.0._codebook.embed"] = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
        )
        weights["voice.vq.layers.1._codebook.embed"] = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        for layer in range(2):
            prefix = f"voice.vq.layers.{layer}._codebook."
            weights[prefix + "inited"] = torch.ones(1)
            weights[prefix + "cluster_size"] = torch.full((3,), 50.0)
            weights[prefix + "embed_avg"] = torch.full((3, 2), 200.0)
        incompatible = container.load_state_dict(weights, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        with torch.inference_mode():
            codes = quantizer.encode(torch.tensor([[[2.7, 0.1, 0.0], [0.1, 2.8, 0.0]]]))
            torch.testing.assert_close(codes, torch.tensor([[[1, 2, 0]], [[1, 2, 0]]]))
            torch.testing.assert_close(
                quantizer.decode(codes),
                torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]),
                rtol=0,
                atol=0,
            )

    def test_strict_loading_rejects_missing_embeddings_and_unknown_buffers(self):
        quantizer = ResidualVectorQuantizer(dimension=2, n_q=1, bins=3)
        key = "vq.layers.0._codebook.embed"
        missing = OrderedDict(quantizer.state_dict())
        del missing[key]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            quantizer.load_state_dict(missing, strict=True)
        unknown = OrderedDict(quantizer.state_dict())
        unknown["vq.layers.0._codebook.unrecognized"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            quantizer.load_state_dict(unknown, strict=True)


class WaveformFlowInferenceTests(unittest.TestCase):
    def test_inverse_coupling_subtracts_shift_and_masks_padding(self):
        layer = ResidualCouplingLayer(4, 4, 3, 1, 1).eval()
        with torch.no_grad():
            layer.post.weight.zero_()
            layer.post.bias.copy_(torch.tensor([1.0, -2.0]))
        value = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [5., 6., 7.], [8., 9., 10.]]])
        mask = torch.tensor([[[1., 1., 0.]]])
        expected = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [4., 5., 0.], [10., 11., 0.]]])
        with torch.inference_mode():
            torch.testing.assert_close(layer(value, mask), expected, rtol=0, atol=0)
            torch.testing.assert_close(Flip()(value), value.flip(1), rtol=0, atol=0)

    def test_flow_runs_inverse_transforms_in_reverse_layer_order(self):
        flow = ResidualCouplingBlock(4, 4, 3, 1, 1, n_flows=2).eval()
        with torch.no_grad():
            for index, shift in ((0, [1., 2.]), (2, [10., 20.])):
                flow.flows[index].post.weight.zero_()
                flow.flows[index].post.bias.copy_(torch.tensor(shift))
        value = torch.tensor([[[1.], [2.], [3.], [4.]]])
        with torch.inference_mode():
            actual = flow(value, torch.ones(1, 1, 1))
        torch.testing.assert_close(actual, torch.tensor([[[-19.], [-8.], [2.], [2.]]]), rtol=0, atol=0)


class SynthesizerCheckpointTests(unittest.TestCase):
    versions = ("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo")
    mel_versions = {"v3", "v4", "v5dev", "v5turbo"}

    def model(self, version):
        cls = SynthesizerTrnV3 if version in self.mel_versions else SynthesizerTrn
        # Meta tensors exercise the complete parameter tree without allocating
        # full diffusion weights or requiring model downloads and training config.
        with torch.device("meta"):
            return nn.ModuleDict({"voice": cls(
                spec_channels=705, inter_channels=192, hidden_channels=192,
                filter_channels=64, n_heads=2, n_layers=2, kernel_size=3,
                p_dropout=0, resblock="2", resblock_kernel_sizes=[3],
                resblock_dilation_sizes=[[1, 3]], upsample_rates=[2],
                upsample_initial_channel=16, upsample_kernel_sizes=[4],
                gin_channels=512, semantic_frame_rate="25hz", version=version,
            )})

    def test_all_versions_strictly_load_legacy_training_only_parameters(self):
        for version in self.versions:
            with self.subTest(version=version):
                model = self.model(version)
                weights = OrderedDict(model.state_dict())
                if version in self.mel_versions:
                    weights["voice.linear_mel.weight"] = torch.empty(100, 512, 1, device="meta")
                    weights["voice.linear_mel.bias"] = torch.empty(100, device="meta")
                else:
                    weights["voice.enc_q.pre.weight"] = torch.empty(192, 705, 1, device="meta")
                    weights["voice.enc_q.pre.bias"] = torch.empty(192, device="meta")
                prefix = "voice.quantizer.vq.layers.0._codebook."
                weights[prefix + "inited"] = torch.empty(1, device="meta")
                weights[prefix + "cluster_size"] = torch.empty(1024, device="meta")
                weights[prefix + "embed_avg"] = torch.empty(1024, 768, device="meta")
                incompatible = model.load_state_dict(weights, strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                self.assertIs(type(model["voice"]).forward, nn.Module.forward)

    def test_training_key_filters_keep_inference_mismatch_errors_visible(self):
        for version in ("v2", "v4"):
            with self.subTest(version=version):
                model = self.model(version)
                missing = OrderedDict(model.state_dict())
                del missing["voice.enc_p.ssl_proj.weight"]
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    model.load_state_dict(missing, strict=True)
                unknown = OrderedDict(model.state_dict())
                unknown["voice.enc_q_extra.weight"] = torch.empty(1, device="meta")
                with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                    model.load_state_dict(unknown, strict=True)
                unknown = OrderedDict(model.state_dict())
                unknown["voice.linear_mel_extra.weight"] = torch.empty(1, device="meta")
                with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                    model.load_state_dict(unknown, strict=True)


if __name__ == "__main__":
    unittest.main()

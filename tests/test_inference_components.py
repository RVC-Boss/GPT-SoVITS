"""Inference-only speaker, vocoder, and bandwidth-extension compatibility."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torchaudio.compliance.kaldi import fbank as reference_fbank

from GPT_SoVITS.BigVGAN.bigvgan import AttrDict, BigVGAN
from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
from GPT_SoVITS.eres2net.kaldi import fbank
from GPT_SoVITS.sv import SV
from tools.AP_BWE_main.models.model import APNet_BWE_Model
from tools.audio_sr import AP_BWE


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / ".codex" / "tmp"


class InferenceComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_speaker_features_match_kaldi(self):
        for dtype in (torch.float32, torch.float64):
            for length in (400, 1600, 16001):
                for dither in (0.0, 0.1):
                    with self.subTest(dtype=dtype, length=length, dither=dither), torch.random.fork_rng():
                        torch.manual_seed(41)
                        audio = torch.randn(1, length, dtype=dtype)
                        torch.manual_seed(42)
                        expected = reference_fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=dither)
                        torch.manual_seed(42)
                        actual = fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=dither)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _speaker_model(self):
        return ERes2NetV2(num_blocks=[1, 1, 1, 1], m_channels=16, baseWidth=24, scale=4, expansion=4).eval()

    def test_original_speaker_checkpoint_ignores_only_unused_projection(self):
        model = self._speaker_model()
        state = dict(model.state_dict())
        state.update({"seg_1.weight": torch.zeros(2, 2), "seg_1.bias": torch.zeros(2)})
        with patch("GPT_SoVITS.sv.ERes2NetV2", return_value=model), patch("GPT_SoVITS.sv.torch.load", return_value=state):
            speaker = SV("cpu", False)
        self.assertFalse(hasattr(model, "pool"))
        self.assertFalse(hasattr(model, "seg_1"))
        with torch.inference_mode():
            features = speaker.compute_embedding3(torch.zeros(1, 1600))
        self.assertEqual(features.shape, (1, 5120))
        self.assertTrue(torch.isfinite(features).all())

    def test_speaker_checkpoint_rejects_missing_or_unexpected_features(self):
        for defect in ("missing", "unexpected"):
            with self.subTest(defect=defect):
                model = self._speaker_model()
                state = dict(model.state_dict())
                if defect == "missing":
                    state.pop("conv1.weight")
                else:
                    state["seg_2.weight"] = torch.zeros(1)
                with patch("GPT_SoVITS.sv.ERes2NetV2", return_value=model), patch("GPT_SoVITS.sv.torch.load", return_value=state):
                    with self.assertRaises(RuntimeError):
                        SV("cpu", False)

    def test_bigvgan_loads_both_local_checkpoint_layouts(self):
        config = AttrDict(
            num_mels=4,
            upsample_initial_channel=8,
            upsample_rates=[2],
            upsample_kernel_sizes=[4],
            resblock="1",
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            activation="snakebeta",
            snake_logscale=True,
        )
        with torch.random.fork_rng(), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            torch.manual_seed(42)
            model = BigVGAN(config).eval()
            audio_features = torch.randn(1, 4, 8)
            with torch.inference_mode():
                expected = model(audio_features)
            directory = Path(directory)
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for normalized in (True, False):
                with self.subTest(normalized=normalized):
                    if not normalized:
                        model.remove_weight_norm()
                    torch.save({"generator": model.state_dict()}, directory / "bigvgan_generator.pt")
                    loaded = BigVGAN.from_pretrained(directory).eval()
                    self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
                    with torch.inference_mode():
                        torch.testing.assert_close(loaded(audio_features), expected, rtol=1e-6, atol=1e-7)
            self.assertFalse(hasattr(loaded, "save_pretrained"))

    def test_bandwidth_extension_loads_generator_and_resamples(self):
        config = AttrDict(n_fft=16, hop_size=4, win_size=16, hr_sampling_rate=48000, ConvNeXt_channels=8, ConvNeXt_layers=1)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            directory = Path(directory)
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            checkpoint = directory / "generator.zip"
            generator = APNet_BWE_Model(config).eval()
            torch.save({"generator": generator.state_dict()}, checkpoint)
            extender = AP_BWE("cpu", AttrDict, checkpoint)
            output, sampling_rate = extender(torch.zeros(1, 1600), 16000)
        self.assertEqual(sampling_rate, 48000)
        self.assertEqual(output.shape, (4800,))
        self.assertTrue(torch.isfinite(torch.from_numpy(output)).all())


if __name__ == "__main__":
    unittest.main()

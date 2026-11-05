"""V5 acoustic scheduling/CFM tests; no checkpoints or reference recordings needed."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
from module.models_v5 import CFMV5, synthesize_v5_mel
from module.v5_acoustic import V5AcousticGraphRunner, synthesize_v5_mel_batch

from GPT_SoVITS.f5_tts.model.backbones.dit import DiT


class ToyCFM(torch.nn.Module):
    in_channels = 2
    noise_temperature = 0.875
    use_conditioner_cache = True
    use_static_cache = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def inference(self, mu, lengths, prompt, steps, inference_cfg_rate=0.0, initial_noise=None, all_valid=False):
        noise = (
            torch.randn(mu.shape[0], 2, mu.shape[1], device=mu.device, dtype=mu.dtype)
            if initial_noise is None
            else initial_noise
        )
        return mu.transpose(2, 1) * self.scale + noise * 0.1 + prompt.mean(-1, keepdim=True) * 0.3


class SchedulerTests(unittest.TestCase):
    def test_rng_order_lengths_and_rolling_context(self):
        model = SimpleNamespace(version="v5turbo", cfm=ToyCFM())
        for reference_length in (17, 400, 600):
            reference = torch.randn(1, 2, reference_length)
            mel = torch.randn_like(reference)
            chunk = min(1000 - min(reference_length, 500), 640)
            targets = [torch.randn(1, 2, n) for n in (5, chunk, chunk * 2 + 13, chunk + 8, chunk * 3, 5)]
            torch.manual_seed(22)
            expected = [synthesize_v5_mel(model, reference, target, mel, 4, 0.0) for target in targets]
            expected_rng = torch.get_rng_state()
            torch.manual_seed(22)
            stats = []
            actual = synthesize_v5_mel_batch(model, reference, targets, mel, max_batch=4, stats=stats)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertEqual([x.shape[-1] for x in actual], [x.shape[-1] for x in targets])
            self.assertTrue(any(x["batch"] > 1 for x in stats))

    def test_invalid_inputs(self):
        model = SimpleNamespace(cfm=ToyCFM())
        reference = torch.zeros(1, 2, 10)
        for targets, batch in (([], 4), ([reference], 0), ([reference[..., :0]], 4)):
            with self.assertRaises(ValueError):
                synthesize_v5_mel_batch(model, reference, targets, reference, max_batch=batch)
        with self.assertRaises(ValueError):
            synthesize_v5_mel_batch(model, reference[..., :0], [reference], reference)


class CFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)
        cls.cfm = CFMV5(
            4, DiT(dim=32, depth=1, heads=2, mel_dim=4, text_dim=16, conv_layers=1, use_step_embedding=False)
        ).eval()
        cls.mu = torch.randn(2, 15, 16)
        cls.prompt = torch.randn(2, 4, 5)
        cls.lengths = torch.tensor([15, 15])

    def test_explicit_noise_and_cfg(self):
        for cfg in (0.0, 1.3):
            torch.manual_seed(7)
            expected = self.cfm.inference(self.mu, self.lengths, self.prompt, 4, inference_cfg_rate=cfg)
            torch.manual_seed(7)
            noise = torch.randn(2, 4, 15)
            actual = self.cfm.inference(
                self.mu, self.lengths, self.prompt, 4, inference_cfg_rate=cfg, initial_noise=noise
            )
            self.assertTrue(torch.equal(expected, actual))
            optimized = self.cfm.inference(
                self.mu, self.lengths, self.prompt, 4, inference_cfg_rate=cfg, initial_noise=noise, all_valid=True
            )
            torch.testing.assert_close(actual, optimized, atol=1e-5, rtol=1e-5)

    def test_padded_mask_is_preserved(self):
        cache = self.cfm.estimator.prepare_static_cache(
            torch.zeros(2, 4, 15), torch.tensor([15, 11]), self.mu.transpose(2, 1)
        )
        self.assertTrue(cache["mask"][0].all())
        self.assertTrue(cache["mask"][1, :11].all())
        self.assertFalse(cache["mask"][1, 11:].any())
        cache = self.cfm.estimator.prepare_static_cache(
            torch.zeros(2, 4, 15), self.lengths, self.mu.transpose(2, 1), all_valid=True, use_negative_condition=False
        )
        self.assertIsNone(cache["mask"])
        self.assertIs(cache["condition"], cache["negative_condition"])

    def test_invalid_noise(self):
        with self.assertRaises(ValueError):
            self.cfm.inference(self.mu, self.lengths, self.prompt, 4, initial_noise=torch.zeros(2, 4, 14))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA Graph tests require CUDA")
class GraphTests(unittest.TestCase):
    def test_replay_eviction_and_model_invalidation(self):
        runner = V5AcousticGraphRunner(max_entries=2)
        model = ToyCFM().eval().cuda()
        for length in (6, 6, 7, 8, 6):
            mu = torch.randn(1, length, 2, device="cuda")
            prompt = torch.randn(1, 2, 2, device="cuda")
            noise = torch.randn(1, 2, length, device="cuda")
            expected = model.inference(mu, None, prompt, 4, initial_noise=noise)
            actual = runner(model, mu, prompt, noise, 4, 0.0).clone()
            self.assertTrue(torch.equal(expected, actual))
            self.assertLessEqual(len(runner.entries), 2)
        self.assertEqual(runner.hits, 1)
        self.assertEqual(runner.evictions, 2)
        model = ToyCFM().eval().cuda()
        model.scale.data.fill_(0.7)
        expected = model.inference(mu, None, prompt, 4, initial_noise=noise)
        self.assertTrue(torch.equal(runner(model, mu, prompt, noise, 4, 0.0), expected))
        self.assertEqual(len(runner.entries), 1)
        runner.clear()
        self.assertFalse(runner.entries)
        self.assertIsNone(runner.model_context)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

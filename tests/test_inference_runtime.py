"""CPU-only regression tests for inference model and checkpoint compatibility."""

from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from AR.models.t2s_inference import Text2SemanticInference
from AR.models.t2s_model import Text2SemanticDecoder
from GPT_SoVITS.inference.checkpoints import get_sovits_version_from_path_fast, load_sovits_new


class InferenceModelTests(unittest.TestCase):
    def test_legacy_model_prefix_loads_strictly_and_preserves_inference(self):
        config = {
            "model": {
                "hidden_dim": 16,
                "embedding_dim": 16,
                "head": 2,
                "n_layer": 1,
                "vocab_size": 9,
                "phoneme_vocab_size": 12,
                "dropout": 0.0,
                "EOS": 8,
            }
        }
        with torch.random.fork_rng():
            torch.manual_seed(31415)
            legacy_decoder = Text2SemanticDecoder(config).eval()
            # Training checkpoints store decoder parameters below self.model.
            legacy_weights = OrderedDict(
                ("model." + name, tensor.clone())
                for name, tensor in legacy_decoder.state_dict().items()
            )
            inference = Text2SemanticInference(config).eval()
            incompatible = inference.load_state_dict(legacy_weights, strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            self.assertEqual(set(inference.state_dict()), set(legacy_weights))
            for name, tensor in inference.state_dict().items():
                torch.testing.assert_close(tensor, legacy_weights[name], rtol=0, atol=0)

            phones = torch.tensor([[1, 2, 3]], dtype=torch.long)
            lengths = torch.tensor([3], dtype=torch.long)
            prompt = torch.tensor([[1, 4]], dtype=torch.long)
            bert = torch.randn(1, 1024, 3)
            with torch.inference_mode(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                torch.manual_seed(7)
                expected = legacy_decoder.infer(
                    phones, lengths, prompt, bert, top_k=1, early_stop_num=2
                )
                torch.manual_seed(7)
                actual = inference.model.infer(
                    phones, lengths, prompt, bert, top_k=1, early_stop_num=2
                )
            self.assertGreater(actual.shape[1], prompt.shape[1])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TextFeatureModelTests(unittest.TestCase):
    def test_encoder_loads_mlm_checkpoint_without_changing_text_features(self):
        from transformers import AutoModel, BertConfig, BertForMaskedLM

        config = BertConfig(
            vocab_size=24, hidden_size=16, num_hidden_layers=4,
            num_attention_heads=2, intermediate_size=32,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        )
        temporary_root = ROOT / ".codex" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with torch.random.fork_rng(), tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            torch.manual_seed(31415)
            legacy = BertForMaskedLM(config).eval()
            legacy.save_pretrained(directory)
            encoder = AutoModel.from_pretrained(directory, add_pooling_layer=False).eval()
            inputs = torch.tensor([[1, 5, 3, 9, 2]])
            with torch.inference_mode():
                expected = legacy(inputs, output_hidden_states=True).hidden_states[-3]
                actual = encoder(inputs, output_hidden_states=True).hidden_states[-3]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertIsNone(encoder.pooler)
            self.assertFalse(hasattr(encoder, "cls"))


class CheckpointLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".codex" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="checkpoint-test-", dir=temporary_root
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.checkpoint = {
            "weight": {"decoder.weight": torch.arange(6).reshape(2, 3).float()},
            "config": {"model": {"version": "v2"}},
            "info": "inference checkpoint fixture",
        }
        buffer = BytesIO()
        torch.save(self.checkpoint, buffer)
        self.payload = buffer.getvalue()
        self.assertTrue(self.payload.startswith(b"PK"))

    def assert_loaded_checkpoint(self, path):
        loaded = load_sovits_new(str(path))
        self.assertEqual(loaded["config"], self.checkpoint["config"])
        self.assertEqual(loaded["info"], self.checkpoint["info"])
        torch.testing.assert_close(
            loaded["weight"]["decoder.weight"],
            self.checkpoint["weight"]["decoder.weight"],
            rtol=0,
            atol=0,
        )

    def test_unmodified_zip_checkpoint_loads(self):
        path = self.directory / "legacy.pth"
        path.write_bytes(self.payload)
        self.assertEqual(get_sovits_version_from_path_fast(str(path)), ("v1", "v1", False))
        self.assert_loaded_checkpoint(path)

    def test_all_supported_version_headers_load_the_original_zip_payload(self):
        versions = {
            b"00": ("v1", "v1", False),
            b"01": ("v2", "v2", False),
            b"02": ("v2", "v3", False),
            b"03": ("v2", "v3", True),
            b"04": ("v2", "v4", True),
            b"05": ("v2", "v2Pro", False),
            b"06": ("v2", "v2ProPlus", False),
            b"07": ("v2", "v5dev", True),
            b"08": ("v2", "v5turbo", True),
        }
        for header, expected in versions.items():
            with self.subTest(header=header):
                path = self.directory / (header.decode("ascii") + ".pth")
                path.write_bytes(header + self.payload[2:])
                self.assertEqual(get_sovits_version_from_path_fast(str(path)), expected)
                self.assert_loaded_checkpoint(path)

    def test_unknown_version_header_is_rejected(self):
        path = self.directory / "unsupported.pth"
        path.write_bytes(b"99" + self.payload[2:])
        with self.assertRaisesRegex(ValueError, "Unsupported SoVITS checkpoint header"):
            get_sovits_version_from_path_fast(str(path))

    def test_checkpoint_with_legacy_hparams_metadata_loads(self):
        # Original SoVITS training saved utils.HParams objects in config, so
        # checkpoint loading must still resolve that pickle type after pruning.
        legacy_utils = types.ModuleType("utils")
        exec(
            "class HParams:\n"
            "    def __init__(self, **values):\n"
            "        for key, value in values.items():\n"
            "            if isinstance(value, dict):\n"
            "                value = HParams(**value)\n"
            "            setattr(self, key, value)\n"
            "    def __getitem__(self, key):\n"
            "        return getattr(self, key)\n",
            legacy_utils.__dict__,
        )
        path = self.directory / "legacy-hparams.pth"
        with patch.dict(sys.modules, {"utils": legacy_utils}):
            metadata = legacy_utils.HParams(model={"version": "v2"})
            torch.save({"config": metadata, "weight": self.checkpoint["weight"]}, path)
        payload = path.read_bytes()
        for header in (b"PK", b"03"):
            with self.subTest(header=header):
                path.write_bytes(header + payload[2:])
                loaded = load_sovits_new(str(path))
                self.assertEqual(loaded["config"]["model"]["version"], "v2")
                torch.testing.assert_close(
                    loaded["weight"]["decoder.weight"],
                    self.checkpoint["weight"]["decoder.weight"],
                    rtol=0,
                    atol=0,
                )


class AudioRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(
        torchaudio.__version__.split("+", 1)[0] == "2.7.1",
        "audio decoder check requires the supported torchaudio 2.7.1 runtime",
    )
    def test_reference_wav_can_be_decoded_without_torchcodec(self):
        buffer = BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00\xff\x7f\x00\x80\x00\x40")
        buffer.seek(0)
        waveform, sample_rate = torchaudio.load(buffer)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(tuple(waveform.shape), (1, 4))
        torch.testing.assert_close(
            waveform,
            torch.tensor([[0.0, 32767 / 32768, -1.0, 0.5]]),
            rtol=0,
            atol=0,
        )


class LanguageRuntimeTests(unittest.TestCase):
    def test_korean_mecab_backend_applies_pronunciation_rules(self):
        import nltk
        nltk.data.find("corpora/cmudict.zip")
        from text.korean import _g2p, g2p

        self.assertTrue(type(_g2p.mecab).__module__.startswith("mecab"))
        self.assertEqual(_g2p("국물"), "궁물")
        self.assertTrue(g2p("안녕하세요."))


if __name__ == "__main__":
    unittest.main()

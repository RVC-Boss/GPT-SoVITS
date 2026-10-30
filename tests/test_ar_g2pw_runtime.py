"""Inference attention/cache and G2PW input preparation regressions."""

import builtins
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
from AR.modules.activation import MultiheadAttention
from AR.modules.transformer import TransformerEncoder, TransformerEncoderLayer


def load_g2pw_module(name):
    package_name = "_g2pw_input_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "GPT_SoVITS/text/g2pw")]
    spec = importlib.util.spec_from_file_location(package_name + "." + name, Path(package.__path__[0]) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package_name: package}):
        spec.loader.exec_module(module)
    return module


class AttentionInferenceTests(unittest.TestCase):
    def test_pytorch_attention_function_is_not_monkey_patched(self):
        self.assertEqual(torch.nn.functional.multi_head_attention_forward.__module__, "torch.nn.functional")

    def test_attention_matches_pytorch_with_checkpoint_parameters(self):
        torch.manual_seed(54)
        custom = MultiheadAttention(16, 2, batch_first=True).eval()
        reference = torch.nn.MultiheadAttention(16, 2, dropout=0, batch_first=True).eval()
        reference.load_state_dict(custom.state_dict(), strict=True)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        with torch.inference_mode():
            for attention_mask in (None, mask, mask.expand(4, -1, -1)):
                for need_weights in (False, True):
                    with self.subTest(mask=attention_mask is not None, need_weights=need_weights):
                        actual = custom(x, x, x, attn_mask=attention_mask, need_weights=need_weights)
                        expected = reference(x, x, x, attn_mask=attention_mask, need_weights=need_weights)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_cached_continuation_matches_full_causal_sequence(self):
        for norm_first in (False, True):
            with self.subTest(norm_first=norm_first):
                torch.manual_seed(37)
                encoder = TransformerEncoder(TransformerEncoderLayer(16, 2, 32, batch_first=True,
                    norm_first=norm_first), 2).eval()
                x = torch.randn(2, 7, 16)
                mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
                cache = {"first_infer": 1, "stage": 0, "all_stage": 2, "k": [None, None], "v": [None, None]}
                with torch.inference_mode():
                    expected = encoder(x, mask=mask)
                    outputs = [encoder(x[:, :4], mask=mask[:4, :4], cache=cache)]
                    cache["first_infer"] = 0
                    for position in range(4, 7):
                        outputs.append(encoder(x[:, position:position + 1],
                            mask=mask[position:position + 1, :position + 1], cache=cache))
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(cache["stage"], 0)
                self.assertEqual([item.shape[0] for item in cache["k"]], [7, 7])


class G2PWInferenceTests(unittest.TestCase):
    def test_config_keeps_runtime_settings_and_discards_training_options(self):
        module = load_g2pw_module("utils")
        temp_root = ROOT / ".codex/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root, prefix="g2pw-config-") as folder:
            config_path = Path(folder) / "config.py"
            config_path.write_text("lr = 0.1\nnum_iter = 100\n", encoding="utf-8")
            defaults = module.load_config(config_path)
            self.assertEqual(vars(defaults), {"model_source": "bert-base-chinese", "use_mask": True,
                "use_char_phoneme": False})
            config_path.write_text("model_source = 'local-bert'\nuse_mask = False\nuse_char_phoneme = True\n", encoding="utf-8")
            custom = module.load_config(config_path)
            self.assertEqual(vars(custom), {"model_source": "local-bert", "use_mask": False,
                "use_char_phoneme": True})

    def test_onnx_inputs_preserve_queries_masks_and_long_text_alignment(self):
        module = load_g2pw_module("input_processing")
        class Tokenizer:
            def tokenize(self, word):
                return list(word)
            def convert_tokens_to_ids(self, tokens):
                return [sum(map(ord, token)) for token in tokens]
        labels, char2phonemes = module.get_phoneme_labels([("行", "xing2"), ("行", "hang2"), ("重", "zhong4")])
        output = module.prepare_onnx_input(Tokenizer(), labels, char2phonemes, ["行", "重"],
            ["我们行abc", "中" * 40 + "重" + "国" * 40], [2, 40], use_mask=True, max_len=12)
        self.assertEqual(output["input_ids"].shape, (2, 12))
        np.testing.assert_array_equal(output["char_ids"], [0, 1])
        np.testing.assert_array_equal(output["position_ids"], [3, 6])
        np.testing.assert_array_equal(output["attention_masks"].sum(axis=1), [8, 12])
        np.testing.assert_array_equal(output["phoneme_masks"], [[1, 1, 0], [0, 0, 1]])
        self.assertEqual(output["input_ids"][1, output["position_ids"][1]], ord("重"))

    def test_pronunciation_dictionary_import_never_writes_source_files(self):
        package_name = "_g2pw_dictionary_tests"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "GPT_SoVITS/text/g2pw")]
        backend = types.ModuleType(package_name + ".onnx_api")
        backend.G2PWOnnxConverter = object
        spec = importlib.util.spec_from_file_location(package_name + ".g2pw", Path(package.__path__[0]) / "g2pw.py")
        module = importlib.util.module_from_spec(spec)
        original_open = builtins.open
        def read_only_open(file, mode="r", *args, **kwargs):
            self.assertFalse(any(flag in mode for flag in "wax+"), f"unexpected write to {file}")
            return original_open(file, mode, *args, **kwargs)
        with patch.dict(sys.modules, {package_name: package, package_name + ".onnx_api": backend}), \
                patch("builtins.open", side_effect=read_only_open):
            spec.loader.exec_module(module)
        self.assertEqual(module.correct_pronunciation("一不小心", ["yi1", "bu4", "xiao3", "xin1"]),
            ["yi2", "bu4", "xiao3", "xin1"])
        self.assertFalse((Path(package.__path__[0]) / "polyphonic.pickle").exists())
        self.assertFalse((Path(package.__path__[0]) / "polyphonic.md5").exists())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

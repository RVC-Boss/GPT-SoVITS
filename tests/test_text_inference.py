"""Actual multilingual inference phoneme and normalization regressions."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

SNAPSHOTS = {
    ("zh", "v1", "你好。"): (["n", "i2", "h", "ao3", "."], [2, 2, 1], "你好."),
    ("zh", "v2", "你好。"): (["n", "i2", "h", "ao3", "."], [2, 2, 1], "你好."),
    ("en", "v1", "Hello, world!"): (
        ["HH", "AH0", "L", "OW1", ",", "W", "ER1", "L", "D", "!"], None, "Hello, world!"
    ),
    ("en", "v2", "Hello, world!"): (
        ["HH", "AH0", "L", "OW1", ",", "W", "ER1", "L", "D", "!"], None, "Hello, world!"
    ),
    ("ja", "v1", "こんにちは。"): (
        ["k", "o", "UNK", "N", "n", "i", "ch", "i", "w", "a", "."], None, "こんにちは。"
    ),
    ("ja", "v2", "こんにちは。"): (
        ["k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "."], None, "こんにちは。"
    ),
    ("ko", "v2", "국물."): (["ㄱ", "ㅜ", "ㅇ", "ㅁ", "ㅜ", "ㄹ", "."], None, "국물."),
    ("yue", "v2", "你好。"): (["Yn", "Yei5", "Yh", "You2", "."], [2, 2, 1], "你好."),
}


class TextInferenceTests(unittest.TestCase):
    def test_language_phonemes_and_alignment(self):
        from text.cleaner import clean_text

        for (language, version, text), expected in SNAPSHOTS.items():
            with self.subTest(language=language, version=version):
                if language == "zh" and version != "v1":
                    model = ROOT / "GPT_SoVITS/text/G2PWModel/g2pW.onnx"
                    tokenizer = ROOT / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json"
                    if not model.is_file() or not tokenizer.is_file():
                        self.skipTest("Local G2PW ONNX model and tokenizer are required")
                self.assertEqual(clean_text(text, language, version), expected)

    def test_traditional_input_conversion(self):
        from text.zh_normalization.char_convert import tranditional_to_simplified

        self.assertEqual(tranditional_to_simplified("繁體中文，重慶銀行，臺灣與香港。"), "繁体中文，重庆银行，台湾与香港。")
        self.assertEqual(tranditional_to_simplified("ABC123."), "ABC123.")


if __name__ == "__main__":
    unittest.main()

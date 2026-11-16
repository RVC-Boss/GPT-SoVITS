"""Regressions for reading numbers in Chinese text normalization, without loading any model."""

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent / "GPT_SoVITS")]
from text.zh_normalization.text_normlization import TextNormalizer  # noqa: E402


class NegativeNumberTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = TextNormalizer()

    def test_negative_decimals(self):
        self.assertEqual(self.normalizer.normalize_sentence("-1.5"), "负一点五")
        self.assertEqual(self.normalizer.normalize_sentence("-0.5"), "负零点五")
        self.assertEqual(self.normalizer.normalize_sentence("余额为-3.75元"), "余额为负三点七五元")

    def test_negative_integers(self):
        self.assertEqual(self.normalizer.normalize_sentence("-10"), "负十")
        self.assertEqual(self.normalizer.normalize_sentence("温度是-1。"), "温度是负一。")


if __name__ == "__main__":
    unittest.main()

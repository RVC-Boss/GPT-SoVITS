"""Regressions for reading dates in Chinese text normalization, without loading any model."""

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent / "GPT_SoVITS")]
from text.zh_normalization.text_normlization import TextNormalizer  # noqa: E402


class DateTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = TextNormalizer()

    def test_single_digit_month_and_day(self):
        self.assertEqual(self.normalizer.normalize_sentence("2024/1/1"), "二零二四年一月一日")
        self.assertEqual(self.normalizer.normalize_sentence("2024-3-8"), "二零二四年三月八日")
        self.assertEqual(self.normalizer.normalize_sentence("2024/1/31"), "二零二四年一月三十一日")
        self.assertEqual(self.normalizer.normalize_sentence("2024/10/1"), "二零二四年十月一日")
        self.assertEqual(self.normalizer.normalize_sentence("会议在2024/9/27举行"), "会议在二零二四年九月二十七日举行")

    def test_two_digit_month_and_day(self):
        self.assertEqual(self.normalizer.normalize_sentence("2024/09/27"), "二零二四年九月二十七日")
        self.assertEqual(self.normalizer.normalize_sentence("2024-01-01"), "二零二四年一月一日")

    def test_not_a_date(self):
        self.assertEqual(self.normalizer.normalize_sentence("3/4"), "四分之三")
        self.assertNotIn("年", self.normalizer.normalize_sentence("2024/13/1"))
        self.assertNotIn("年", self.normalizer.normalize_sentence("2024/1/32"))


if __name__ == "__main__":
    unittest.main()

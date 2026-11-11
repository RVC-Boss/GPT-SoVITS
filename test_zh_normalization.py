"""Regressions for Chinese text normalization, without loading any model."""

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent / "GPT_SoVITS")]
from text.zh_normalization.chronology import RE_TIME_RANGE, replace_time  # noqa: E402


def verbalize_time_range(text):
    return RE_TIME_RANGE.sub(replace_time, text)


class TimeRangeTests(unittest.TestCase):
    def test_end_time_minutes_are_read_from_end_time(self):
        self.assertEqual(verbalize_time_range("8:30-12:15"), "八点半至十二点十五分")
        self.assertEqual(verbalize_time_range("9:00-9:30"), "九点至九点半")

    def test_half_hour_on_both_ends(self):
        self.assertEqual(verbalize_time_range("8:30~9:30"), "八点半至九点半")


if __name__ == "__main__":
    unittest.main()

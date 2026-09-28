import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from expend import normalize


class TimeTests(unittest.TestCase):
    def test_midnight_is_twelve_am(self):
        self.assertEqual(normalize("0:00"), "twelve o'clock a.m.")
        self.assertEqual(normalize("00:30"), "twelve thirty a.m.")

    def test_noon_and_afternoon_stay(self):
        self.assertEqual(normalize("12:00"), "twelve o'clock p.m.")
        self.assertEqual(normalize("4:00"), "four o'clock a.m.")
        self.assertEqual(normalize("13:00"), "one o'clock p.m.")
        self.assertEqual(normalize("13:30"), "one thirty p.m.")


if __name__ == "__main__":
    unittest.main()

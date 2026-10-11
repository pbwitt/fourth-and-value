"""Eastern presentation preserves instants across date and daylight-saving changes."""
from datetime import datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from display_time import eastern_time
from nhl.v2.arbitrage import _when


class EasternTimeTests(unittest.TestCase):
    def test_date_rollover_and_seasonal_offsets(self):
        self.assertEqual(eastern_time('2026-10-10T01:30:00Z'), 'Oct 09, 2026 at 09:30 PM EDT')
        self.assertEqual(eastern_time('2026-01-01T02:30:00+00:00'), 'Dec 31, 2025 at 09:30 PM EST')

    def test_daylight_saving_boundaries(self):
        cases = {
            '2026-03-08T06:59:00Z': '01:59 AM EST',
            '2026-03-08T07:00:00Z': '03:00 AM EDT',
            '2026-11-01T05:30:00Z': '01:30 AM EDT',
            '2026-11-01T06:30:00Z': '01:30 AM EST',
        }
        for instant, label in cases.items():
            with self.subTest(instant=instant):
                self.assertEqual(eastern_time(instant, '%I:%M %p %Z'), label)

    def test_requires_known_source_zone(self):
        with self.assertRaises(ValueError):
            eastern_time(datetime(2026, 10, 10, 12))

    def test_arbitrage_uses_eastern_calendar_date(self):
        self.assertEqual(_when('2026-10-11T02:00:00Z'), 'Oct 10, 10:00 PM EDT')
        self.assertEqual(_when(None), 'not yet checked')


if __name__ == '__main__':
    unittest.main()

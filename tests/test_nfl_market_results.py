"""Tests for the Market Results builder: main-line choice, no-vig pricing and stable output."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_market_results as mr

ROOT = Path(__file__).resolve().parents[1]


def offers(rows):
    d = pd.DataFrame(rows, columns=['bookmaker', 'point', 'over_price', 'under_price'])
    d['p_over'] = [mr.no_vig(a, b) for a, b in zip(d.over_price, d.under_price)]
    return d


class MarketResultsTests(unittest.TestCase):
    def test_no_vig_removes_margin(self):
        self.assertAlmostEqual(mr.no_vig(-110, -110), 0.5)
        self.assertAlmostEqual(mr.no_vig(-140, 120) + mr.no_vig(120, -140), 1.0)
        self.assertGreater(mr.no_vig(-140, 120), 0.5)

    def test_main_line_is_most_offered_and_averages_books(self):
        line = mr.main_line(offers([['a', 4.5, -110, -110], ['b', 4.5, -130, 110], ['c', 5.5, 120, -140]]))
        self.assertEqual(line['line'], 4.5)
        self.assertEqual(line['books'], 2)
        self.assertAlmostEqual(line['p_over'], (0.5 + mr.no_vig(-130, 110)) / 2)
        self.assertAlmostEqual(line['dec_over'], (mr.decimal(-110) + mr.decimal(-130)) / 2)

    def test_main_line_tie_goes_to_line_nearest_median(self):
        line = mr.main_line(offers([['a', 3.5, -110, -110], ['b', 4.5, -110, -110], ['c', 4.5, -105, -115],
                                    ['d', 5.5, -110, -110], ['e', 5.5, -110, -110], ['f', 6.5, -110, -110]]))
        self.assertEqual(line['line'], 4.5)

    def test_pair_sides_needs_both_prices_at_the_same_line(self):
        d = pd.DataFrame([
            dict(stat_game='g', player='p', market_std='receptions', bookmaker='a', point=4.5, side='over', price=-110),
            dict(stat_game='g', player='p', market_std='receptions', bookmaker='a', point=4.5, side='under', price=-110),
            dict(stat_game='g', player='p', market_std='receptions', bookmaker='b', point=5.5, side='over', price=120),
        ])
        paired = mr.pair_sides(d)
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired.p_over.iloc[0], 0.5)

    def test_write_keeps_file_when_only_build_time_changes(self):
        with TemporaryDirectory() as tmp:
            old_out, mr.OUT, old_root = mr.OUT, Path(tmp), mr.ROOT
            mr.ROOT = Path(tmp)
            try:
                payload = mr.build_nhl(2026)
                mr.write(payload)
                first = (Path(tmp) / 'nhl.json').read_text()
                mr.write({**payload, 'generated_at': '2099-01-01T00:00:00+00:00'})
                self.assertEqual((Path(tmp) / 'nhl.json').read_text(), first)
            finally:
                mr.OUT, mr.ROOT = old_out, old_root

    def test_published_data_matches_its_market_config(self):
        for sport in ['nfl', 'nhl']:
            data = json.loads((ROOT / f'docs/markets/data/{sport}.json').read_text())
            self.assertEqual(set(data['rows']), {m['key'] for m in data['markets']})
            self.assertTrue(data['notes']['timing'])
            for rows in data['rows'].values():
                for r in rows:
                    self.assertEqual(len(r), len(data['fields']))
                    self.assertTrue(0 < r[5] < 1)


if __name__ == '__main__':
    unittest.main()

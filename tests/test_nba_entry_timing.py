"""NBA entry timing: earlier entry prices against the last pregame snapshot, from committed runs only."""
import copy
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nba import entry_timing as t  # noqa: E402
from test_nba_shadow_grade import AFTER_TIP, LATE, NOW, START, event, prop, summary  # noqa: E402

# Oct 20, 5 PM ET: a day-before price and a line the close no longer posts.
DAY_BEFORE = (prop(24.5, 120, -150, 0.60, 0.55, book='b') + prop(24.5, -110, -110, 0.60, 0.55)
              + prop(23.5, 130, -160, 0.62, 0.52, book='b') + prop(23.5, -110, -110, 0.62, 0.52))
# Oct 21, 7 AM ET: the game-morning price.
MORNING = prop(24.5, 105, -125, 0.62, 0.50, book='b') + prop(24.5, -110, -110, 0.62, 0.50)


class BucketTests(unittest.TestCase):
    def test_buckets_use_eastern_dates_and_noon(self):
        start = t.g.parse(START)  # Oct 21, 10 PM ET
        self.assertEqual(t.bucket(t.g.parse('2026-10-21T03:59:00Z'), start), 'day_before')   # Oct 20, 11:59 PM ET
        self.assertEqual(t.bucket(t.g.parse('2026-10-21T04:00:00Z'), start), 'game_morning')  # midnight ET
        self.assertEqual(t.bucket(t.g.parse('2026-10-21T15:59:00Z'), start), 'game_morning')  # 11:59 AM ET
        self.assertIsNone(t.bucket(t.g.parse('2026-10-21T16:00:00Z'), start))                # noon ET

    def test_line_move_direction_depends_on_the_side(self):
        close = t.g.snapshot_units(prop(24.5, -110, -110, 0.6, 0.55) + prop(24.5, 100, -120, 0.6, 0.55, book='b'))
        self.assertEqual(t.line_move(('player_points', '99', 23.5), True, close), 'toward')   # Over 23.5 vs close 24.5
        self.assertEqual(t.line_move(('player_points', '99', 23.5), False, close), 'away')   # Under 23.5 vs close 24.5
        self.assertIsNone(t.line_move(('player_points', '99', 24.5), True, close))
        self.assertIsNone(t.line_move(('player_rebounds', '99', 7.5), True, close))


class BuildTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runs = self.root / 'artifacts/nba/runs'
        self.runs.mkdir(parents=True)
        for name, at, rows in [('20261020T210000Z', '2026-10-20T21:00:00Z', DAY_BEFORE),
                               ('20261021T110000Z', '2026-10-21T11:00:00Z', MORNING),
                               ('20261021T210000Z', '2026-10-21T21:00:00Z', LATE),
                               ('20261022T030000Z', '2026-10-22T03:00:00Z', AFTER_TIP)]:
            with gzip.open(self.runs / f'{name}.json.gz', 'wt') as f:
                json.dump(dict(checked_at=at, rows=rows), f)
        self.calls = []

    def fetch(self, url):
        self.calls.append(url)
        return copy.deepcopy(dict(events=[event()]) if '/scoreboard?' in url else summary())

    def build(self, **kw):
        return t.build(2026, self.runs, self.root, kw.pop('fetch', self.fetch), offline=kw.pop('offline', False),
                       now=NOW, version='nba-v1.0', **kw)

    def test_entries_are_scored_against_the_last_pregame_snapshot(self):
        payload = self.build()
        self.assertEqual(payload['coverage'], dict(games_tipped=1, games_with_day_before_props=1, games_with_game_morning_props=1))
        bets = {(b['bucket'], b['line']): b for b in payload['bets']}
        self.assertEqual(set(bets), {('day_before', 24.5), ('day_before', 23.5), ('game_morning', 24.5)})
        morning = bets[('game_morning', 24.5)]
        # Over at +105 when the market said 50%; the close says 55% and its best Over price is +120.
        self.assertEqual((morning['side'], morning['entry_price'], morning['close_price']), ('ref', 105.0, 120.0))
        self.assertAlmostEqual(morning['closing_value'], 0.55 * 2.05 - 1)
        self.assertAlmostEqual(morning['movement'], 0.05)
        self.assertAlmostEqual(morning['close_price_value'], 0.55 * 2.2 - 1)
        self.assertEqual((morning['result'], round(morning['profit'], 6), round(morning['close_profit'], 6)), ('win', 1.05, 1.2))
        # A line the close no longer posts: no closing value, but the number held is recorded.
        moved = bets[('day_before', 23.5)]
        self.assertNotIn('closing_value', moved)
        self.assertEqual(moved['line_move'], 'toward')
        self.assertEqual(moved['result'], 'win', 'settled at the line actually bet')
        day = payload['buckets']['day_before']
        self.assertEqual((day['bets'], day['same_line_at_close'], day['line_moved']['toward']), (2, 1, 1))
        self.assertAlmostEqual(day['closing_value']['mean'], round(0.55 * 2.2 - 1, 5))
        self.assertIsNone(day['closing_value']['ci95'], 'no interval from one game')

    def test_game_lines_and_snapshots_after_tipoff_never_count(self):
        payload = self.build()
        self.assertTrue(all(b['market'] not in t.g.GAME_MARKETS for b in payload['bets']))
        self.assertTrue(all(b['close_at'] == '2026-10-21T21:00:00+00:00' for b in payload['bets']))

    def test_offline_uses_saved_results_only(self):
        payload = self.build(offline=True, fetch=lambda url: self.fail('offline must not fetch'))
        self.assertEqual(payload['buckets']['game_morning']['bets'], 1)
        self.assertEqual(payload['buckets']['game_morning']['results']['graded'], 0)

    def test_writes_json_and_an_eastern_time_summary(self):
        payload = self.build()
        out = self.root / 'out'
        t.write(payload, out)
        self.assertEqual(json.loads((out / 'entry-timing.json').read_text())['schema'], 1)
        text = (out / 'ENTRY_TIMING.md').read_text()
        self.assertIn('EDT', text)
        self.assertNotIn('UTC', text)
        self.assertIn('| Game morning (before noon ET) | 1 | 1 |', text)

    def test_empty_archive(self):
        empty = self.root / 'empty'
        empty.mkdir()
        payload = t.build(2026, empty, self.root, None, offline=True, now=NOW, version='nba-v1.0')
        self.assertEqual(payload['buckets']['day_before']['bets'], 0)
        self.assertIn('| Day before | 0 | 0 |', t.markdown(payload))


if __name__ == '__main__':
    unittest.main()

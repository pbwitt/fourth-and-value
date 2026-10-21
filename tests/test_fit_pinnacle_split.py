"""Pinnacle-vs-US scoring joins graded lines without lookahead."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import fit_pinnacle_split as fit  # noqa: E402

FIELDS = ['event_id', 'commence_time', 'market', 'player', 'side', 'line', 'price', 'last_update', 'captured_at']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def board_row(side, outcome, market_p, decided='2026-10-21T11:00:00Z'):
    return dict(event_id='e1', market='totals', player='', side=side, line=5.5, outcome=outcome,
                market_probability=market_p, decision_at=decided, commence_time='2026-10-21T23:00:00Z')


class Join(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        quotes = [['e1', '2026-10-21T23:00:00Z', 'totals', '', side, 5.5, price, None, at]
                  for at, prices in (('2026-10-21T10:00:00Z', (-120, 100)), ('2026-10-21T15:00:00Z', (-200, 170)))
                  for side, price in zip(('Over', 'Under'), prices)]
        write(self.root / 'artifacts/pinnacle/reference/nhl/2026-10-21.json',
              dict(sport='nhl', fields=FIELDS, quotes=quotes, runs=[]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_the_capture_before_our_decision(self):
        write(self.root / 'docs/blog/nhl-recap-2026-10-26.json',
              dict(board=[board_row('Over', 'win', .52), board_row('Under', 'loss', .48)]))
        chances = fit.no_vig(fit.pinnacle_quotes(self.root))
        lines = fit.board_lines(self.root, chances)
        self.assertEqual(len(lines), 1)                       # one side per line
        expected = fit.implied(-120) / (fit.implied(-120) + fit.implied(100))
        self.assertAlmostEqual(lines[0]['pin'], expected)     # not the 3 PM capture

    def test_line_decided_before_any_capture_is_skipped(self):
        write(self.root / 'docs/blog/nhl-recap-2026-10-26.json',
              dict(board=[board_row('Over', 'win', .52, decided='2026-10-21T09:00:00Z')]))
        chances = fit.no_vig(fit.pinnacle_quotes(self.root))
        self.assertEqual(fit.board_lines(self.root, chances), [])

    def test_log_loss_mix(self):
        lines = [dict(pin=.6, us=.5, y=True, game='g')]
        self.assertLess(fit.loss(lines, 1), fit.loss(lines, 0))


if __name__ == '__main__':
    unittest.main()

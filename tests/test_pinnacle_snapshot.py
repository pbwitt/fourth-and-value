"""Public Pinnacle reference: Pinnacle's own two-sided no-vig chance, only for lines in our feeds,
and an honest record of what was actually checked."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pinnacle_snapshot as ps

NOW = datetime(2026, 10, 24, 11, 9, tzinfo=timezone.utc)


def quote(market, side, line, price, player='', event='e1', sport='NHL'):
    return dict(sport=sport, event_id=event, commence_time='2026-10-24T23:00:00Z', market=market,
                player=player, side=side, line=line, price=price, last_update='2026-10-24T11:08:00Z')


class PublicReference(unittest.TestCase):
    def test_no_vig_pairs_both_sides_and_spread_signs(self):
        fair = ps.no_vig([quote('player_assists', 'Over', 0.5, 124, 'A B'), quote('player_assists', 'Under', 0.5, -151, 'A B'),
                          quote('spreads', 'Home', -1.5, 180), quote('spreads', 'Away', 1.5, -220),
                          quote('player_points', 'Over', 0.5, -110, 'C D')])          # one side only: no chance
        self.assertAlmostEqual(fair[('e1', 'player_assists', 'a b', 'over', 0.5)], 0.4260, places=4)
        self.assertAlmostEqual(sum(v for k, v in fair.items() if k[1] == 'spreads'), 1)
        self.assertNotIn(('e1', 'player_points', 'c d', 'over', 0.5), fair)

    def test_public_file_keeps_our_lines_and_what_was_checked(self):
        quotes = [quote('player_assists', 'Over', 0.5, 124, 'A B'), quote('player_assists', 'Under', 0.5, -151, 'A B'),
                  quote('player_receptions', 'Under', 2.5, 124, 'X Y', 'e2', 'NFL'),
                  quote('player_receptions', 'Over', 2.5, -151, 'X Y', 'e2', 'NFL')]
        ours = {'NHL': [quote('player_assists', 'Over', 0.5, 300, 'A B')], 'NFL': [quote('player_receptions', 'Under', 2.5, 120, 'X Y', 'e2', 'NFL')]}
        public = ps.public_reference(quotes, ours, dict(games=['NHL'], props=['e1', 'e2']), NOW)
        self.assertEqual((public['status'], public['decision_date'], public['use']), ('recorded', '2026-10-24', 'reference_only'))
        self.assertEqual([(r['market'], r['side']) for r in public['lines']], [('player_assists', 'Over'), ('player_receptions', 'Under')],
                         'only outcomes our feeds carry are published')
        self.assertEqual(public['lines'][1]['market_std'], 'receptions')
        self.assertEqual((public['checked_games'], public['checked_events']), (['NHL'], ['e1', 'e2']))

    def test_a_refused_snapshot_publishes_nothing_as_not_posted(self):
        public = ps.public_reference([], {'NHL': [quote('player_assists', 'Over', 0.5, 300, 'A B')]}, dict(games=[], props=[]), NOW)
        self.assertEqual((public['status'], public['lines'], public['checked_events']), ('unavailable', [], []))


if __name__ == '__main__':
    unittest.main()

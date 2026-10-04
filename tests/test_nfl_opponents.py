"""NFL prop opponents: each player is matched to the other team in his game, never his own."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from make_player_prop_params import apply_defensive_adjustment, create_home_away_map, create_opponent_map


def props(*games):
    return pd.DataFrame([dict(player=player, home_team=home, away_team=away, game=f'{away} @ {home}')
                         for player, home, away in games])


LOGS = pd.DataFrame([dict(player='Josh Allen', recent_team='BUF'), dict(player='Drake Maye', recent_team='NE'),
                     dict(player='Matthew Stafford', recent_team='LA'), dict(player='Sam Darnold', recent_team='SEA'),
                     dict(player='Traded Player', recent_team='NYJ')])
PROPS = props(('Josh Allen', 'Buffalo Bills', 'New England Patriots'), ('Drake Maye', 'Buffalo Bills', 'New England Patriots'),
              ('Matthew Stafford', 'Los Angeles Rams', 'Seattle Seahawks'), ('Sam Darnold', 'Los Angeles Rams', 'Seattle Seahawks'),
              ('Traded Player', 'Buffalo Bills', 'New England Patriots'), ('No Logs', 'Buffalo Bills', 'New England Patriots'))


class OpponentTests(unittest.TestCase):
    def test_home_and_away_players_face_the_other_team(self):
        opponents = create_opponent_map(PROPS, LOGS)
        self.assertEqual(opponents['Josh Allen'], 'NE', 'a home player faces the visitors, not his own defense')
        self.assertEqual(opponents['Drake Maye'], 'BUF')
        self.assertEqual(opponents['Matthew Stafford'], 'SEA', 'the Rams are LA in nflverse logs')
        self.assertEqual(opponents['Sam Darnold'], 'LA')

    def test_unmatched_teams_get_no_opponent(self):
        opponents = create_opponent_map(PROPS, LOGS)
        self.assertNotIn('Traded Player', opponents, 'no guess when his team is in neither side')
        self.assertNotIn('No Logs', opponents)

    def test_home_away_flags(self):
        home = create_home_away_map(PROPS, LOGS)
        self.assertEqual([home['Josh Allen'], home['Drake Maye'], home['Matthew Stafford'], home['Sam Darnold']],
                         [True, False, True, False])
        self.assertIsNone(home['Traded Player'])

    def test_defense_factor_range(self):
        self.assertAlmostEqual(apply_defensive_adjustment(100, 'pass_yds', .5), 115)
        self.assertAlmostEqual(apply_defensive_adjustment(100, 'pass_yds', 1), 100)
        self.assertAlmostEqual(apply_defensive_adjustment(100, 'pass_yds', 2), 70)


if __name__ == '__main__':
    unittest.main()

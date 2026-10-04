"""NFL prop traces for the player snapshot: every Normal market, its spread, opponent and venue."""
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from make_player_prop_params import build_params
from nfl_prop_diagnostics import opponent_trace


def week(player, team, opponent, position, **stats):
    base = dict(rushing_yards=0, rushing_attempts=0, rushing_tds=0, carries=0, receptions=0, receiving_yards=0, receiving_tds=0,
                targets=0, passing_yards=0, passing_tds=0, interceptions=0, attempts=0, completions=0)
    return [dict(base, player=player, recent_team=team, position=position, gsis_id=player, season=2026, week=w,
                 opponent_team=opponent, **stats) for w in range(1, 5)]


LOGS = pd.DataFrame(week('Run Back', 'BUF', 'NE', 'RB', carries=16, rushing_attempts=16, rushing_yards=72, targets=3, receptions=2, receiving_yards=15)
                    + week('Wide Out', 'NE', 'BUF', 'WR', targets=8, receptions=5, receiving_yards=75)
                    + week('Quarter Back', 'BUF', 'NE', 'QB', attempts=34, completions=22, passing_yards=240))
CANDS = pd.DataFrame([dict(player=p, market_std=m) for p, m in [('Run Back', 'rush_yds'), ('Run Back', 'rush_attempts'),
                      ('Wide Out', 'receptions'), ('Wide Out', 'recv_yds'), ('Quarter Back', 'pass_yds')]])
RATINGS = pd.DataFrame(dict(pass_def_rating=[1.2, 0.8, 1.0], rush_def_rating=[0.9, 1.1, 1.0], games=[4, 4, 4],
                            pass_yds_per_game=[190.0, 250.0, 220.0], rush_yds_per_game=[130.0, 100.0, 115.0], season=2026),
                       index=['NE', 'BUF', 'MIA'])


class TraceTests(unittest.TestCase):
    def setUp(self):
        params = build_params(CANDS, LOGS, 2026, 5, defensive_ratings=RATINGS,
                              opponent_map={'Run Back': 'NE', 'Wide Out': 'BUF', 'Quarter Back': 'NE'},
                              home_away_map={'Run Back': True, 'Wide Out': False, 'Quarter Back': True}, career_df=pd.DataFrame())
        self.traces = {(r.player, r.market_std): (r, json.loads(r.projection_diagnostics)) for r in params.itertuples()
                       if isinstance(r.projection_diagnostics, str)}

    def test_every_normal_market_has_a_trace_ending_at_its_mean(self):
        self.assertEqual(len(self.traces), 5)
        for (player, market), (row, trace) in self.traces.items():
            self.assertAlmostEqual(trace['mean_stages']['final'], row.mu, places=4, msg=market)
            self.assertAlmostEqual(trace['sigma'], row.sigma, places=4)

    def test_rushing_and_receiving_components(self):
        rush = self.traces[('Run Back', 'rush_yds')][1]
        self.assertEqual(rush['family'], 'rush')
        self.assertAlmostEqual(rush['carries'] * rush['yards_per_carry'], rush['mean_stages']['before_adjustments'], places=3)
        self.assertEqual(rush['current_sample'][-1], dict(season=2026, week=4, opponent_team='NE', carries=16, rushing_yards=72))
        catch = self.traces[('Wide Out', 'recv_yds')][1]
        self.assertAlmostEqual(catch['targets'] * catch['catch_rate'] * catch['yards_per_reception'],
                               catch['mean_stages']['before_adjustments'], places=3)

    def test_opponent_rank_and_venue(self):
        rush = self.traces[('Run Back', 'rush_yds')][1]
        self.assertEqual(rush['opponent'], dict(team='NE', kind='rush', rating=0.9, rank=3, of=3, allowed=130.0, league=115.0,
                                                games=4, season=2026), 'run defense for rushing props')
        self.assertTrue(rush['home'])
        qb = self.traces[('Quarter Back', 'pass_yds')][1]
        self.assertEqual(qb['opponent']['rank'], 1, 'the highest rating is the toughest defense')
        self.assertIsNone(opponent_trace('pass_yds', 'Nobody', {}, RATINGS))
        bare = RATINGS[['pass_def_rating', 'rush_def_rating']]
        self.assertEqual(opponent_trace('receptions', 'Wide Out', {'Wide Out': 'BUF'}, bare),
                         dict(team='BUF', kind='pass', rating=0.8, rank=3, of=3), 'older ratings frames still trace')

    def test_passing_sample_names_the_opponent(self):
        qb = self.traces[('Quarter Back', 'pass_yds')][1]
        self.assertEqual(qb['current_sample'][-1]['opponent_team'], 'NE')


class DefensiveRatingTests(unittest.TestCase):
    def test_yards_allowed_and_data_season_are_kept_for_display(self):
        import os, tempfile
        from make_player_prop_params import calculate_defensive_ratings
        rows = [dict(team=t, opponent_team=o, week=w, receiving_yards=r, rushing_yards=u)
                for w in (1, 2) for t, o, r, u in [('BUF', 'NE', 200, 90), ('NE', 'BUF', 260, 120), ('MIA', 'NYJ', 230, 100), ('NYJ', 'MIA', 230, 110)]]
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, 'data'))
            pd.DataFrame(rows).to_parquet(os.path.join(root, 'data/weekly_player_stats_2025.parquet'))
            here = os.getcwd()
            os.chdir(root)
            try:
                ratings = calculate_defensive_ratings(2026, 1)
            finally:
                os.chdir(here)
        self.assertEqual(int(ratings.at['NE', 'season']), 2025, 'week 1 uses last season and says so')
        self.assertAlmostEqual(ratings.at['NE', 'pass_yds_per_game'], 200.0)
        self.assertAlmostEqual(ratings.at['BUF', 'rush_yds_per_game'], 120.0)
        self.assertGreater(ratings.at['NE', 'pass_def_rating'], ratings.at['BUF', 'pass_def_rating'])


if __name__ == '__main__':
    unittest.main()

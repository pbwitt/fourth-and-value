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


class VersusTests(unittest.TestCase):
    def test_games_against_this_weeks_opponent_are_context_only(self):
        base = dict(rushing_yards=0, rushing_attempts=0, rushing_tds=0, carries=0, receptions=0, receiving_yards=0, receiving_tds=0,
                    targets=0, passing_yards=0, passing_tds=0, interceptions=0, attempts=0, completions=0)
        career = pd.DataFrame([dict(base, player='Run Back', recent_team='BUF', team='BUF', position='RB', gsis_id='Run Back', season=s, week=w,
                                    season_type='REG', opponent_team=o, rushing_yards=y, carries=c, rushing_attempts=c, game_id=g)
                               for s, w, o, y, c, g in [(2023, 3, 'NE', 40, 10, None), (2024, 7, 'NYJ', 99, 20, '2024_07_BUF_NYJ'),
                                                        (2025, 9, 'NE', 110, 22, '2025_09_NE_BUF'), (2026, 2, 'NE', 72, 16, '2026_02_BUF_NE'),
                                                        (2026, 6, 'NE', 500, 40, '2026_06_NE_BUF')]])
        args = dict(defensive_ratings=RATINGS, opponent_map={'Run Back': 'NE', 'Wide Out': 'BUF', 'Quarter Back': 'NE'},
                    home_away_map={'Run Back': True, 'Wide Out': False, 'Quarter Back': True})
        with_history = build_params(CANDS, LOGS, 2026, 5, career_df=career, **args)
        relabeled = build_params(CANDS, LOGS, 2026, 5, career_df=career.assign(opponent_team='MIA'), **args)
        trace = {(r.player, r.market_std): json.loads(r.projection_diagnostics) for r in with_history.itertuples()
                 if isinstance(r.projection_diagnostics, str)}
        rush = trace[('Run Back', 'rush_yds')]['versus']
        self.assertEqual(rush['team'], 'NE')
        self.assertEqual(rush['values'], [40.0, 110.0, 72.0], 'only NE games, oldest first, none from this week on')
        self.assertEqual(rush['since'], '2023 Wk 3')
        self.assertEqual(rush['home'], [None, True, False], 'venue from the game id; unknown without one')
        self.assertEqual(trace[('Run Back', 'rush_attempts')]['versus']['values'], [10.0, 22.0, 16.0])
        self.assertIsNone(trace[('Wide Out', 'recv_yds')].get('versus'), 'no meetings, no section')
        self.assertEqual(list(with_history.mu), list(relabeled.mu), 'history against the opponent never changes a forecast')
        again = {(r.player, r.market_std): json.loads(r.projection_diagnostics) for r in relabeled.itertuples()
                 if isinstance(r.projection_diagnostics, str)}
        self.assertIsNone(again[('Run Back', 'rush_yds')].get('versus'))


class PositionTests(unittest.TestCase):
    def test_params_keep_each_players_position(self):
        params = build_params(CANDS, LOGS, 2026, 5, career_df=pd.DataFrame())
        self.assertEqual(dict(zip(params.player, params.position)), {'Run Back': 'RB', 'Wide Out': 'WR', 'Quarter Back': 'QB'})

    def test_every_market_of_a_player_gets_his_position(self):
        from make_props_edges import attach_positions
        merged = pd.DataFrame(dict(player_key=['runback', 'runback', 'nobody'], market_std=['rush_yds', 'first_td', 'rush_yds']))
        params = pd.DataFrame(dict(player_key=['runback', 'runback'], market_std=['rush_yds', 'rush_attempts'], position=['RB', 'RB']))
        out = attach_positions(merged, params, 'player_key')
        self.assertEqual(out.player_position.tolist()[:2], ['RB', 'RB'], 'unmodeled markets too')
        self.assertTrue(pd.isna(out.player_position.iloc[2]))
        self.assertNotIn('player_position', attach_positions(merged, params.drop(columns='position'), 'player_key'))

    def test_board_rows_and_static_cards_show_the_position(self):
        import tempfile
        import build_props_site
        row = dict(game_id='g', game='A @ B', player='Run Back', bookmaker='draftkings', bookmaker_title='DraftKings', market_std='rush_yds',
                   name='over', point=70.5, price=-110, mu=72.2, model_prob=.47, push_prob=0, model_status='Uncalibrated',
                   last_update='2026-10-04T12:00:00Z', commence_time='2026-10-04T17:00:00Z', home_team='B', away_team='A',
                   player_position='RB')
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
            pd.DataFrame([row, dict(row, player='Other Back', player_position=None)]).to_csv(f.name, index=False)
        records = build_props_site.prepare_records(f.name)
        self.assertEqual([r['player_position'] for r in records], ['RB', None])
        self.assertIn('<h2>Run Back <span class="pc-pos">RB</span></h2>', build_props_site.static_card(records[0]))
        self.assertIn('<h2>Other Back</h2>', build_props_site.static_card(records[1]))


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

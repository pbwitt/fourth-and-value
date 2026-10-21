"""NBA v1: coherent score matrix, no same-day leakage, joint combos, shadow-mode inference."""
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba import model as M  # noqa: E402
from nba import inference  # noqa: E402

PARAMS = json.loads((Path(__file__).resolve().parents[1] / 'artifacts/nba/model/nba-v1.json').read_text())
TEAMS = {'1': ('Atlanta Hawks', 'ATL'), '2': ('Boston Celtics', 'BOS'), '3': ('LA Clippers', 'LAC'), '4': ('Denver Nuggets', 'DEN')}


def history(days=40, bump=0):
    """Synthetic league: four teams, round-robin daily, two players per team."""
    games, teams, players = [], [], []
    start = date(2025, 11, 1)
    pairs = [('1', '2'), ('3', '4'), ('2', '3'), ('4', '1'), ('1', '3'), ('2', '4')]
    for d in range(days):
        day = (start + timedelta(days=d)).isoformat()
        for j, (h, a) in enumerate(pairs[(2 * d) % 6:(2 * d) % 6 + 2]):
            gid = f'{d}-{j}'
            hp, ap = 110 + (d % 7) + (bump if d == days - 1 else 0), 105 + (d % 5)
            games.append(dict(GAME_ID=gid, GAME_DATE=day, SEASON_TYPE='regular', NEUTRAL=False, PERIODS=4,
                              HOME_TEAM_ID=h, HOME_TEAM_NAME=TEAMS[h][0], AWAY_TEAM_ID=a, AWAY_TEAM_NAME=TEAMS[a][0],
                              HOME_PTS=hp, AWAY_PTS=ap))
            for tid, oid, home, pts in ((h, a, True, hp), (a, h, False, ap)):
                teams.append(dict(GAME_ID=gid, GAME_DATE=day, SEASON_TYPE='regular', TEAM_ID=tid, TEAM_NAME=TEAMS[tid][0],
                                  TEAM_ABBREVIATION=TEAMS[tid][1], OPP_TEAM_ID=oid, HOME=home, PTS=pts,
                                  OPP_PTS=hp + ap - pts, FGA=88, FTA=22, OREB=10, TOV=13, REB=44, AST=25, FG3M=12,
                                  BLK=5, STL=8, PERIOD_PTS=[pts // 4] * 3 + [pts - 3 * (pts // 4)]))
                for k in range(2):
                    players.append(dict(GAME_ID=gid, GAME_DATE=day, SEASON_TYPE='regular', TEAM_ID=tid, HOME=home,
                                        PLAYER_ID=f'{tid}{k}', PLAYER_NAME=f'Player {TEAMS[tid][1]} {k}', POSITION='G',
                                        STARTER=k == 0, DNP=False, MIN=34.0 - 10 * k, PTS=22 - 8 * k, REB=5, AST=6 - 2 * k,
                                        FG3M=2, BLK=1, STL=1, TOV=2))
    return dict(games=games, teams=teams, players=players)


class ScoreMatrix(unittest.TestCase):
    def setUp(self):
        _, _, teams = M.game_table(history(), PARAMS['team_config'])
        self.f = teams.features('1', '2', date(2025, 12, 15))
        self.matrix, self.mh, self.ma = M.score_matrix(PARAMS['game'], self.f)

    def test_probabilities_are_coherent(self):
        self.assertAlmostEqual(self.matrix.sum(), 1, places=9)
        self.assertLess(np.trace(self.matrix), 1e-9, 'no NBA game ends tied')
        home, _ = M.game_market(self.matrix, 'h2h', True, None)
        away, _ = M.game_market(self.matrix, 'h2h', False, None)
        self.assertAlmostEqual(home + away, 1, places=9)
        for line in (220.5, 221.0):
            over, push = M.game_market(self.matrix, 'totals', True, line)
            under, push2 = M.game_market(self.matrix, 'totals', False, line)
            self.assertAlmostEqual(over + under + push, 1, places=9)
            self.assertEqual(push, push2)
            self.assertEqual(push == 0, line % 1 != 0, 'push mass only on whole lines')
        fav, push = M.game_market(self.matrix, 'spreads', True, -3.0)
        dog, push2 = M.game_market(self.matrix, 'spreads', False, 3.0)
        self.assertAlmostEqual(fav + dog + push, 1, places=9)
        self.assertGreater(push, 0)

    def test_same_day_results_do_not_leak(self):
        base, _, _ = M.game_table(history(), PARAMS['team_config'])
        bumped, _, _ = M.game_table(history(bump=30), PARAMS['team_config'])
        last = max(r['game']['GAME_DATE'] for r in base)
        a = [r['features']['naive_home'] for r in base if r['game']['GAME_DATE'] == last]
        b = [r['features']['naive_home'] for r in bumped if r['game']['GAME_DATE'] == last]
        self.assertEqual(a, b, "a game's own result never enters its features")


class PlayerDraws(unittest.TestCase):
    def test_combos_are_joint_sums_and_sides_complement(self):
        rates = {st: v for st, v in zip(M.STATS, (0.7, 0.2, 0.18, 0.08, 0.02, 0.03, 0.06))}
        draws, minutes = M.simulate(PARAMS['players'], rates, 32, 5, n=5000, seed=3)
        self.assertTrue(((minutes > 0) & (minutes <= 48)).all())
        over, push, mean = M.player_market(draws, 'player_points_rebounds_assists', 'Over', 34.0)
        under, push2, _ = M.player_market(draws, 'player_points_rebounds_assists', 'Under', 34.0)
        self.assertAlmostEqual(over + under + push, 1, places=9)
        self.assertAlmostEqual(mean, float((draws['PTS'] + draws['REB'] + draws['AST']).mean()))
        # Shared minutes make points and assists positively correlated, unlike independent marginals.
        self.assertGreater(np.corrcoef(draws['PTS'], draws['AST'])[0, 1], 0.02)
        same = M.simulate(PARAMS['players'], rates, 32, 5, n=5000, seed=3)[0]
        self.assertTrue((same['PTS'] == draws['PTS']).all(), 'seeded draws are reproducible')

    def test_median_is_the_inverted_cdf_middle_draw(self):
        draws = {st: np.zeros(7, dtype=int) for st in M.STATS}
        draws['PTS'] = np.array([30, 10, 12, 11, 40, 9, 13])
        self.assertEqual(M.player_median(draws, 'player_points'), 12.0)
        self.assertGreater(float(draws['PTS'].mean()), 17.5, 'right-skewed: the mean clears a line the median misses')
        self.assertLess(M.player_market(draws, 'player_points', 'Over', 17.5)[0], .5)

    def test_calibration_shrinks_toward_even(self):
        self.assertAlmostEqual(float(M.calibrate(0.8, 1.0)), 0.8)
        self.assertLess(float(M.calibrate(0.8, 0.7)), 0.8)
        self.assertAlmostEqual(float(M.calibrate(0.5, 0.7)), 0.5)


class Inference(unittest.TestCase):
    def rows(self):
        base = dict(event_id='e1', commence_time='2025-12-11T00:30:00Z', home_team='Atlanta Hawks',
                    away_team='Boston Celtics', game='Boston Celtics @ Atlanta Hawks', book='a', price=-110)
        return [dict(base, market='totals', side='Over', line=219.5, player=''),
                dict(base, market='h2h', side='Atlanta Hawks', line=None, player=''),
                dict(base, market='player_points', side='Over', line=21.5, player='Player ATL 0'),
                dict(base, market='player_points_rebounds_assists', side='Under', line=33.5, player='Player ATL 0'),
                dict(base, market='player_points', side='Over', line=12.5, player='Player BOS 1'),
                dict(base, market='player_points', side='Over', line=12.5, player='Player LAC 0'),
                dict(base, market='player_points', side='Over', line=9.5, player='Nobody Here')]

    def fetch(self, url):
        if url.endswith('/injuries'):
            return dict(injuries=[dict(injuries=[dict(status='Out', athlete=dict(links=[dict(href='https://www.espn.com/nba/player/_/id/21/x')]))])])
        tid = url.split('/teams/')[1].split('/')[0]
        return dict(athletes=[dict(id=f'{tid}{k}', displayName=f'Player {TEAMS[tid][1]} {k}') for k in range(2)])

    def test_forecasts_reasons_and_shadow_metadata(self):
        rows = self.rows()
        meta = inference.add_model(rows, history(), datetime(2025, 12, 10, 18, tzinfo=timezone.utc), PARAMS, fetch=self.fetch)
        total, h2h, pts, pra, injured, elsewhere, unknown = rows
        for row in (total, h2h, pts, pra):
            self.assertTrue(0 < row['model_probability'] < 1, row)
            self.assertEqual(row['model_version'], PARAMS['version'])
        self.assertEqual(pts['espn_player_id'], '10')
        self.assertAlmostEqual(pts['model_ev_pct'], 100 * (pts['model_probability'] * (1 + 100 / 110) - 1 + pts['model_push_probability']), places=1)
        self.assertEqual(injured['model_unsupported_reason'], 'Listed out on the ESPN injury report')
        self.assertEqual(elsewhere['model_unsupported_reason'], 'Player not on either current roster')
        self.assertIsNone(unknown['model_probability'])
        self.assertNotIn('model_checked_at', meta, 'shadow mode never makes NBA editorial/Top Picks eligible')
        self.assertEqual(meta['model_coverage']['forecast'], 4)
        # The median sits on the side of a half line that the model favors; the mean need not.
        for row in (total, pts, pra):
            favored = row['model_probability'] / (1 - row['model_push_probability']) > .5
            self.assertEqual(row['model_median'] > row['line'] if row['side'] == 'Over' else row['model_median'] < row['line'], favored, row)
        self.assertIsNone(injured['model_median'])


if __name__ == '__main__':
    unittest.main()

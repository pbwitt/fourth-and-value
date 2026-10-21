"""Season-form window and quarterback starts for the NFL baselines."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import career_baseline as cb
from make_player_prop_params import estimate_pass_latents, exponential_weighted_mean


def qb(player, team, week, attempts, season=2026):
    return dict(player=player, recent_team=team, position='QB', season=season, week=week,
                attempts=attempts, completions=round(attempts * .65), passing_yards=7 * attempts)


class StartsTests(unittest.TestCase):
    def test_relief_appearances_leave_and_partial_starts_stay(self):
        df = pd.DataFrame([qb('Starter', 'BUF', 1, 34), qb('Backup', 'BUF', 1, 3),
                           qb('Starter', 'BUF', 2, 9),  # injury exit: still the team's leading passer
                           dict(player='Receiver', recent_team='BUF', position='WR', season=2026, week=1, attempts=1)])
        kept = cb.qb_starts(df)
        self.assertEqual(sorted(zip(kept.player, kept.week)), [('Receiver', 1), ('Starter', 1), ('Starter', 2)])

    def test_without_teams_nothing_is_dropped(self):
        df = pd.DataFrame([qb('Starter', 'BUF', 1, 34), qb('Backup', 'BUF', 1, 3)]).drop(columns='recent_team')
        self.assertEqual(len(cb.qb_starts(df)), 2)

    def test_a_backups_relief_games_do_not_set_his_starting_volume(self):
        # His only games this season are relief appearances; the estimate comes from starts.
        career = pd.DataFrame([qb('Vet', 'NE', w, 33, season=2025) for w in range(1, 18)]
                              + [qb('Starter', 'MIA', w, 31, season=2025) for w in range(1, 18)])
        logs = pd.DataFrame([qb('Starter', 'BUF', w, 35) for w in (1, 2)] + [qb('Backup', 'BUF', w, 4) for w in (1, 2)])
        idx = pd.Index(['Backup', 'Starter'])
        on = estimate_pass_latents(logs, idx, career_df=career, season=2026)['volume_mu']
        try:
            cb.QB_STARTS_ONLY = False
            off = estimate_pass_latents(logs, idx, career_df=career, season=2026)['volume_mu']
        finally:
            cb.QB_STARTS_ONLY = True
        self.assertLess(off['Backup'], 25, 'four-attempt relief games dragged his projection down')
        self.assertGreater(on['Backup'], 30, 'with no starts yet, the pool of starts supplies it')
        self.assertAlmostEqual(on['Starter'], off['Starter'], places=6)


class RecencyTests(unittest.TestCase):
    def test_a_games_weight_halves_every_six_games(self):
        values = pd.Series([1.0] + [0.0] * 6)  # oldest first
        newest = pd.Series([0.0] * 6 + [1.0])
        self.assertAlmostEqual(exponential_weighted_mean(values, cb.RECENT_ALPHA) * 2,
                               exponential_weighted_mean(newest, cb.RECENT_ALPHA), places=9)
        self.assertGreaterEqual(cb.RECENT_GAMES, 17, 'every current-season game counts')


if __name__ == '__main__':
    unittest.main()

"""The NFL spread shadow is logged only: it reads the frozen pregame archive and never edits it."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import nfl_spread_shadow as shadow

CONFIG = dict(version='test-shadow', pct_per_point={'receptions': 1.0, 'pass_attempts': 0.5},
              multiplier_limits=[0.85, 1.15], first_prospective_week=8)


def props_frame(week=8):
    def row(player, market, home, mu, name, point=4.5, prob=0.5):
        diag = None if home is None else json.dumps({'home': home})
        return dict(season=2026, week=week, game_id='evt1', commence_time='2026-11-01T18:00:00Z', game='A @ B',
                    player=player, market_std=market, mu=mu, sigma=2.0, lam=np.nan, point=point, name=name,
                    consensus_line=point, prob_devig=prob, projection_diagnostics=diag)
    return pd.DataFrame([
        row('Home WR', 'receptions', True, 5.0, 'over', prob=0.52), row('Home WR', 'receptions', True, 5.0, 'under', prob=0.48),
        row('Away WR', 'receptions', False, 5.0, 'over'), row('Away WR', 'receptions', False, 5.0, 'under'),
        row('Away RB', 'rush_attempts', False, 15.0, 'over', point=14.5),
        row('Untraced QB', 'pass_attempts', None, 33.0, 'over', point=32.5),
    ])


def lines_frame():
    # Home team favored by 3 at two books, 3.5 at a third: median -3.0.
    return pd.DataFrame(dict(event_id=['evt1'] * 3, spread_home_line=[-3.0, -3.0, -3.5]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SpreadShadowTests(unittest.TestCase):
    def test_multiplier_signs_limits_and_scope(self):
        self.assertAlmostEqual(shadow.multiplier('receptions', 7, CONFIG), 1.07)
        self.assertAlmostEqual(shadow.multiplier('receptions', -7, CONFIG), 0.93)
        self.assertEqual(shadow.multiplier('receptions', 40, CONFIG), 1.15)
        self.assertEqual(shadow.multiplier('receptions', -40, CONFIG), 0.85)
        self.assertTrue(np.isnan(shadow.multiplier('rush_attempts', 7, CONFIG)))

    def test_team_spread_is_from_the_players_side(self):
        out = shadow.build_shadow(props_frame(), lines_frame(), CONFIG).set_index('player')
        self.assertNotIn('Away RB', out.index)                     # rushing is out of scope
        self.assertEqual(out.loc['Home WR', 'team_spread'], -3.0)   # favorite
        self.assertEqual(out.loc['Away WR', 'team_spread'], 3.0)    # underdog
        self.assertAlmostEqual(out.loc['Home WR', 'shadow_mu'], 5.0 * 0.97)
        self.assertAlmostEqual(out.loc['Away WR', 'shadow_mu'], 5.0 * 1.03)
        self.assertGreater(out.loc['Away WR', 'shadow_p_over'], out.loc['Away WR', 'model_p_over'])
        self.assertLess(out.loc['Home WR', 'shadow_p_over'], out.loc['Home WR', 'model_p_over'])
        self.assertAlmostEqual(out.loc['Home WR', 'market_p_over'], 0.52)
        self.assertEqual(out.loc['Untraced QB', 'skipped'], 'no home/away trace')
        self.assertTrue(np.isnan(out.loc['Untraced QB', 'shadow_mu']))

    def test_build_and_report_leave_the_pregame_archive_untouched(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            config = root / 'config.json'
            config.write_text(json.dumps(CONFIG))
            for week in (7, 8):
                pregame = root / f'reports/nfl-weekly/2026/week-{week}/pregame'
                pregame.mkdir(parents=True)
                props_frame(week).to_csv(pregame / 'props.csv', index=False)
                lines_frame().to_csv(pregame / 'totals_spreads.csv', index=False)
                (pregame / 'manifest.json').write_text(json.dumps({'snapshot_at': '2026-10-28T11:00:00+00:00'}))
                before = {p.name: digest(p) for p in pregame.iterdir()}
                manifest = shadow.build(2026, week, root=root, config_path=config, now='2026-11-01T20:00:00Z')
                self.assertEqual(before, {p.name: digest(p) for p in pregame.iterdir()})
                self.assertEqual(manifest['in_sample'], week < 8)
                self.assertEqual(manifest['adjusted'], 2)
                again = shadow.build(2026, week, root=root, config_path=config, now='2026-11-02T20:00:00Z')
                self.assertEqual(again['built_at'], manifest['built_at'])   # recovery reruns do not rewrite it
                review = root / f'reports/nfl-weekly/2026/week-{week}/review'
                review.mkdir(parents=True)
                pd.DataFrame(dict(player=['Home WR', 'Away WR', 'Untraced QB'], market_std=['receptions'] * 2 + ['pass_attempts'],
                                  actual=[3, 7, 40])).to_csv(review / 'probability_comparison.csv', index=False)
            summary = shadow.report(2026, root=root, config_path=config, now='2026-11-01T20:00:00Z')
            self.assertEqual([w['week'] for w in summary['weeks']], [7, 8])
            self.assertEqual([w['in_sample'] for w in summary['weeks']], [True, False])
            self.assertEqual(summary['prospective']['n'], 2)           # the untraced QB is never graded
            # The favorite went under and the underdog went over, so the shadow beats the current model.
            self.assertLess(summary['prospective']['shadow_log_loss'], summary['prospective']['model_log_loss'])
            readme = (root / 'reports/nfl-weekly/2026/spread-shadow/README.md').read_text()
            self.assertIn('EST', readme)
            self.assertNotIn('UTC', readme)


if __name__ == '__main__':
    unittest.main()

"""NFL calibration provenance and validation helpers (no network, no paid calls)."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import make_player_prop_params as mpp
import make_props_edges as edges
import nfl_validation as v


class CalibrationCompatibilityTests(unittest.TestCase):
    def test_legacy_and_mismatched_artifacts_are_not_presented_as_validated(self):
        legacy = json.loads((ROOT/'models/nfl_prop_calibration.json').read_text())
        strict = {}
        status = edges.calibration_status(legacy, policy=strict)
        self.assertFalse(status['compatible'])
        self.assertFalse(status['label'].startswith('Calibration fitted'))
        other = dict(legacy, _provenance=dict(model_version='nfl-props-old', run_id='r0'))
        self.assertFalse(edges.calibration_status(other, policy=strict)['label'].startswith('Calibration fitted'))
        # Owner policy keeps NFL in Top Picks, with a label that never claims validation.
        owner = dict(allow_incompatible_for_top_picks=True)
        for artifact in (legacy, other):
            status = edges.calibration_status(artifact, policy=owner)
            self.assertTrue(status['label'].startswith('Calibration fitted'))
            self.assertIn('not validated', status['label'])
            self.assertFalse(status['compatible'])

    def test_repository_policy_keeps_nfl_eligible_and_labelled(self):
        policy = edges.calibration_policy()
        self.assertTrue(policy['allow_incompatible_for_top_picks'])
        self.assertEqual(policy['decided_by'], 'owner')
        legacy = json.loads((ROOT/'models/nfl_prop_calibration.json').read_text())
        self.assertIn('not validated', edges.calibration_status(legacy)['label'])
        current = dict(legacy, _provenance=dict(model_version=mpp.MODEL_VERSION, run_id='r1', evaluation=dict(status='x')))
        status = edges.calibration_status(current)
        self.assertTrue(status['compatible'])
        self.assertTrue(status['label'].startswith('Calibration fitted (r1)'))
        self.assertEqual(edges.calibration_status(None)['label'], 'Uncalibrated')

    def test_saved_validation_records_provenance_and_decision(self):
        report = json.loads((ROOT/'reports/nfl-validation/2026-10-05/validation.json').read_text())
        self.assertEqual(report['model']['model_version'], mpp.MODEL_VERSION)
        self.assertTrue(report['cutoff_verification']['identical_with_future_rows_supplied'])
        self.assertFalse(report['decision']['installed'])
        for key in ('calibration_fit_for_evaluation', 'calibration_evaluation', 'deployment_fit', 'prospective'):
            self.assertIn(key, report['windows'])
        deploy = json.loads((ROOT/'reports/nfl-validation/2026-10-05/calibration-deployment.json').read_text())
        self.assertEqual(deploy['_provenance']['model_version'], mpp.MODEL_VERSION)
        # The production artifact was deliberately not replaced.
        self.assertNotIn('_provenance', json.loads((ROOT/'models/nfl_prop_calibration.json').read_text()))
        for market in report['book_line_evaluation']['markets'].values():
            self.assertGreater(market['games'], 0); self.assertGreaterEqual(market['rows'], market['forecasts'])


class HelperTests(unittest.TestCase):
    def test_participation_never_becomes_a_zero_or_loss(self):
        played = {(v.name_key('Active Player'), 'KC')}
        values = {}
        self.assertEqual(v.outcome_value(played, values, 'Active Player', 'KC', 'receptions'), 0.0)  # snaps, no box score row
        self.assertIsNone(v.outcome_value(played, values, 'Inactive Player', 'KC', 'receptions'))  # no snaps: void
        self.assertEqual(v.name_key('Michael Penix Jr.'), v.name_key('Michael Penix'))

    def test_clustered_scores_and_difference_orientation(self):
        frame = pd.DataFrame(dict(game_id=['a', 'a', 'b', 'c'], outcome=[1, 0, 1, 0], x=[.9, .2, .8, .3], y=[.5, .5, .5, .5]))
        scores = v.clustered(frame, ['x', 'y'], reps=200)
        self.assertAlmostEqual(scores['x']['brier'], float(np.mean((frame.x-frame.outcome)**2)))
        d = v.difference(scores, 'y', 'x')
        self.assertAlmostEqual(d['difference'], -scores['x_minus_y']['difference'])
        self.assertLessEqual(d['ci95'][0], d['ci95'][1])

    def test_curves_bins_and_consensus(self):
        cal = {'_eligible_markets': ['receptions'], 'markets': {'receptions': dict(x=[0, 1], y=[.4, .6])}}
        self.assertAlmostEqual(v.apply_curve(cal, 'receptions', .5), .5)
        self.assertAlmostEqual(v.apply_curve(cal, 'receptions', 1), .6)
        self.assertEqual(v.apply_curve(cal, 'anytime_td', .9), .9)
        lo, hi = v.wilson(30, 100)
        self.assertTrue(lo < .3 < hi)
        offers = pd.DataFrame(dict(game_id=['g']*4, player=['p']*4, market_std=['receptions']*4, point=[2.5]*4,
                                   bookmaker=['a', 'a', 'b', 'b'], side=['over', 'under', 'over', 'under'], price=[-110, -110, 120, -140]))
        c = v.devig_consensus(offers)
        self.assertEqual(int(c.books.iloc[0]), 2)
        self.assertTrue(.4 < c.market_over.iloc[0] < .5)


if __name__ == '__main__':
    unittest.main()

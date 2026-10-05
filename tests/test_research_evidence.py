"""Structured facts, decision ledger, NHL pilot, late check, MLB gates and research grading."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import decision_ledger as ledger
import late_research as late
import research_facts as facts
import research_grading as grading
import research_outcomes as outcomes
from mlb import gates
from nhl.v2 import deployment_pilot as pilot
from nhl.v2.data import iso

NOW = datetime(2026, 10, 5, 15, tzinfo=timezone.utc)


def row(**extra):
    base = dict(candidate_id='c1', offer_id='o1', forecast_id='f1', sport='NHL', game='Away @ Home', game_id='g1',
                commence_time=iso(NOW+timedelta(hours=8)), player='Example Player', home_team='Home', away_team='Away',
                market='player_shots_on_goal', side='Over', line=2.5, book='b', price=110, quoted_at=iso(NOW-timedelta(minutes=5)))
    return dict(base, **extra)


def src(**extra):
    base = dict(source_id='s1', url='https://www.nhl.com/news/example', title='Lines', published_at=iso(NOW-timedelta(hours=2)),
                retrieved_at=iso(NOW-timedelta(minutes=30)), excerpt='Example Player will play 21 minutes tonight', candidate_ids=['c1'])
    return dict(base, **extra)


def item(**extra):
    base = dict(source_id='s1', excerpt='Example Player will play 21 minutes tonight', interpretation='More ice time.',
                kind='deployment', direction='supports', represented_in='neither', assumption='ice_time',
                materiality='consequential', verification='official', effect='quantified', applies_to='this_game')
    return dict(base, **extra)


class FactTests(unittest.TestCase):
    def test_fact_record_keeps_provenance_and_point_in_time(self):
        [f] = facts.build_facts(dict(evidence=[item()]), row(), {'s1': src()}, iso(NOW))
        self.assertEqual(f['schema_version'], 'research-fact-1')
        for key in ('url', 'published_at', 'retrieved_at', 'content_sha256'):
            self.assertIn(key, f['source'])
        self.assertEqual(f['event']['player'], 'Example Player')
        self.assertEqual((f['assumption'], f['effect'], f['usage']), ('ice_time', 'quantified', 'not_represented'))
        self.assertIsNone(f['probability_adjustment'])
        self.assertTrue(f['freshness']['event_starts_after_decision'])
        with self.assertRaises(ValueError):  # No backfill with later information.
            facts.build_facts(dict(evidence=[item()]), row(), {'s1': src(retrieved_at=iso(NOW+timedelta(minutes=1)))}, iso(NOW))

    def test_verification_overrides_and_overlap_guard(self):
        [f] = facts.build_facts(dict(evidence=[item()]), row(), {'s1': src(source_kind='professional_opinion')}, iso(NOW))
        self.assertEqual((f['verification'], f['classification_basis']), ('opinion', 'source_kind_override'))
        [f] = facts.build_facts(dict(evidence=[item()]), row(), {'s1': src(url='https://www.espn.com/nhl/story')}, iso(NOW))
        self.assertEqual(f['verification'], 'secondary_report')
        [f] = facts.build_facts(dict(evidence=[item(represented_in='model_features')]), row(), {'s1': src()}, iso(NOW))
        self.assertEqual(f['usage'], 'excluded_already_in_model')
        legacy = item(); [legacy.pop(k) for k in ('assumption', 'materiality', 'verification', 'effect', 'applies_to')]
        [f] = facts.build_facts(dict(evidence=[legacy]), row(), {'s1': src()}, iso(NOW))
        self.assertEqual(f['classification_basis'], 'not_classified')

    def test_conflicting_directions_are_marked(self):
        a, b = facts.build_facts(dict(evidence=[item(), item(direction='concern', excerpt='Example Player will play 21 minutes')]),
                                 row(), {'s1': src()}, iso(NOW))
        self.assertEqual(a['conflicts_with'], [b['fact_id']])


class PilotTests(unittest.TestCase):
    def setUp(self):
        feed = json.loads((ROOT/'docs/nhl/data/latest.json').read_text())
        self.dist = feed['model_distribution']
        self.row = next(dict(r, sport='NHL', game_id=r['nhl_game_id']) for r in feed['rows']
                        if r.get('market') == 'player_shots_on_goal' and r.get('final_probability') is not None and r['side'] == 'Over')
        self.cutoff = self.row['model_inputs']['feature_cutoff']

    def fact(self, **extra):
        f = facts.build_facts(dict(evidence=[item()]), dict(self.row, candidate_id='c1'), {'s1': src(published_at=iso(NOW-timedelta(hours=1)))}, iso(NOW))[0]
        f.update(extra)
        return f

    def test_baseline_is_reproduced_and_preserved(self):
        rec = pilot.assess(self.row, [], self.dist, NOW)
        self.assertTrue(rec['baseline']['reproduces_published_forecast'])
        self.assertAlmostEqual(rec['baseline']['win'], self.row['final_probability'], places=6)
        self.assertEqual(rec['status'], 'no_facts')
        self.assertIsNone(rec['adjusted'])
        bad = dict(self.row, final_probability=self.row['final_probability']+.05)
        self.assertEqual(pilot.assess(bad, [self.fact()], self.dist, NOW)['status'], 'baseline_mismatch')

    def test_reported_minutes_recompute_through_the_model(self):
        rec = pilot.assess(self.row, [self.fact()], self.dist, NOW)
        self.assertEqual(rec['status'], 'supported_adjustment')
        self.assertEqual(rec['adjusted']['toi'], 21)
        base = rec['baseline']
        self.assertAlmostEqual(rec['adjusted']['mean'], base['mean']*21/base['projected_toi'])
        self.assertEqual(rec['adjusted']['win'] > base['win'], 21 > base['projected_toi'])
        self.assertEqual(rec['evaluation_status'], 'shadow_only')

    def test_double_counting_and_unvalidated_facts_do_not_adjust(self):
        old = self.fact(); old['source'] = dict(old['source'], published_at='2026-01-01T00:00:00Z', updated_at=None)
        rec = pilot.assess(self.row, [old], self.dist, NOW)
        self.assertEqual(rec['facts_considered'][0]['reason'], 'predates_model_feature_cutoff')
        self.assertIsNone(rec['adjusted'])
        rec = pilot.assess(self.row, [self.fact(represented_in='model_features', usage='excluded_already_in_model')], self.dist, NOW)
        self.assertEqual(rec['facts_considered'][0]['reason'], 'already_in_model_features')
        rec = pilot.assess(self.row, [self.fact(assumption='goalie')], self.dist, NOW)
        self.assertEqual(rec['unvalidated_context'][0]['assumption'], 'goalie'); self.assertIsNone(rec['adjusted'])

    def test_unquantified_deployment_gives_a_range_not_a_point(self):
        rec = pilot.assess(self.row, [self.fact(effect='scenario', excerpt='Example Player moves to the first line')], self.dist, NOW)
        self.assertEqual(rec['status'], 'scenario_only')
        lo, hi = rec['scenario_range']['win_conditional']
        self.assertLessEqual(lo, hi)
        self.assertIsNone(rec['adjusted'])
        rec = pilot.assess(self.row, [self.fact(effect='unresolved')], self.dist, NOW)
        self.assertEqual(rec['status'], 'unresolved_fact')


def selection(rows):
    return dict(selected=rows)


def sel_row(i, **extra):
    r = dict(sport='MLB', game_id=f'g{i}', game='A @ H', home_team='H', away_team='A', commence_time=iso(NOW+timedelta(hours=3)),
             player=f'P{i}', market='batter_hits', market_std=None, side='Over', line=1.5, book='b', price=120,
             quoted_at=iso(NOW-timedelta(minutes=3)), forecast_at=iso(NOW-timedelta(minutes=10)), model_probability=.5,
             model_push_probability=0, other_book_probability=.44, other_books=2, _card_value=.001*(10-i),
             research_state=dict(gate='model_case_only'), mlb_game_id=100+i)
    r.update(extra)
    return r


class LedgerTests(unittest.TestCase):
    def test_three_versions_on_one_universe_and_immutability(self):
        rows = [sel_row(0, research_state=dict(gate='adverse_fact')), sel_row(1), sel_row(2, model_probability=None, _card_value=None,
                research_state=dict(gate='not_reviewed'))]
        card = [rows[1]]
        pilot_rec = {}
        with tempfile.TemporaryDirectory() as td:
            led = ledger.freeze(selection(rows), card, NOW, edition_id='e1', ledgers=Path(td), pilot=pilot_rec)
            d = {e['player']: e['decisions'] for e in led['entries']}
            self.assertEqual(d['P0']['baseline']['decision'], 'select')
            self.assertEqual((d['P0']['research_filtered']['decision'], d['P0']['research_filtered']['reason']), ('pass', 'adverse_fact'))
            self.assertEqual(d['P1']['research_filtered']['decision'], 'select')
            self.assertEqual(d['P2']['baseline']['decision'], 'not_eligible')
            self.assertEqual(d['P1']['adjusted']['adjustment'], 'none_supported')
            self.assertTrue(ledger.verify(led))
            # Same edition re-frozen with different content is a conflict, never an overwrite.
            with self.assertRaises(ValueError):
                ledger.freeze(selection([sel_row(1, price=130)]), card, NOW, edition_id='e1', ledgers=Path(td))
            tampered = deepcopy(led); tampered['entries'][0]['price'] = 999
            with self.assertRaises(ValueError):
                ledger.verify(tampered)

    def test_started_events_cannot_be_frozen_and_adjustments_reevaluate(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                ledger.freeze(selection([sel_row(0, commence_time=iso(NOW))]), [], NOW, edition_id='e2', ledgers=Path(td))
            r = sel_row(0)
            key = ledger.outcome_key(r)
            adj = {key: dict(status='supported_adjustment', adjusted_win_conditional=.4, push=0, ev_threshold=.02, record_id='p1')}
            led = ledger.freeze(selection([r]), [r], NOW, edition_id='e3', ledgers=Path(td), pilot=adj)
            a = led['entries'][0]['decisions']['adjusted']
            self.assertEqual((a['decision'], a['reason'], a['adjustment']), ('pass', 'adjusted_ev_below_threshold', 'nhl_deployment_pilot'))
            self.assertEqual(led['entries'][0]['decisions']['research_filtered']['decision'], 'select')


class GradingTests(unittest.TestCase):
    def frozen(self, td):
        rows = [sel_row(0), sel_row(1, game_id='g0'), sel_row(2, research_state=dict(gate='material_question'))]
        return ledger.freeze(selection(rows), rows[:2], NOW, edition_id='e9', ledgers=Path(td))

    def test_settlement_pushes_voids_and_identity(self):
        e = dict(market='batter_hits', side='Over', line=1.5)
        self.assertEqual(grading.settle(e, dict(status='final', value=2)), 'won')
        self.assertEqual(grading.settle(dict(e, line=2), dict(status='final', value=2)), 'push')
        self.assertEqual(grading.settle(e, dict(status='did_not_participate')), 'void')
        self.assertEqual(grading.settle(e, dict(status='unknown')), 'pending')
        self.assertEqual(grading.settle(e, None), 'pending')
        g = dict(market='spreads', side='H', home_team='H', line=-1.5)
        self.assertEqual(grading.settle(g, dict(status='final', home_score=5, away_score=3)), 'won')
        self.assertEqual(grading.settle(dict(g, line=-2), dict(status='final', home_score=5, away_score=3)), 'push')

    def test_grades_versions_without_touching_the_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            led = self.frozen(td)
            before = json.dumps(led, sort_keys=True)
            ids = {e['player']: e['entry_id'] for e in led['entries']}
            result = grading.grade(led, {ids['P0']: dict(status='final', value=2), ids['P1']: dict(status='final', value=0),
                                         ids['P2']: dict(status='did_not_participate')}, now=NOW+timedelta(days=1))
            self.assertEqual(json.dumps(led, sort_keys=True), before)
            base = result['versions']['baseline']['returns']
            # The baseline ignores research and also selects P2, whose non-participation is a void.
            self.assertEqual((base['won'], base['lost'], base['voids']), (1, 1, 1))
            self.assertAlmostEqual(base['units'], .2)
            self.assertEqual(base['games'], 1)  # P0 and P1 share game g0; voids are not settled
            filtered = result['versions']['research_filtered']['returns']
            self.assertEqual((filtered['won'], filtered['lost'], filtered['voids']), (1, 1, 0))
            self.assertEqual(result['research_effect']['filtered_out']['voids'], 1)
            self.assertEqual(result['versions']['research_filtered']['coverage']['selected'], 2)
            self.assertEqual(result['universe']['eligible'], 3)

    def test_movement_is_separate_from_closing(self):
        e = dict(price=120, line=1.5, commence_time=iso(NOW+timedelta(hours=3)), probabilities=dict(market_conditional=.44))
        early = grading.movement(e, dict(last_pregame_at=iso(NOW+timedelta(hours=1)), other_fair=.47, book_line=1.5))
        self.assertAlmostEqual(early['market_probability_move'], .03)
        self.assertAlmostEqual(early['entry_price_value_vs_later'], 2.2*.47-1)
        self.assertEqual(early['closing_comparison']['status'], 'no_verified_close')
        close = grading.movement(e, dict(last_pregame_at=iso(NOW+timedelta(hours=2, minutes=45)), other_fair=.47, book_line=1.5))
        self.assertEqual(close['closing_comparison']['status'], 'verified_close_same_line')
        moved = grading.movement(e, dict(last_pregame_at=iso(NOW+timedelta(hours=2, minutes=45)), other_fair=.47, book_line=2.5))
        self.assertEqual((moved['status'], moved['closing_comparison']['status']), ('line_changed', 'no_verified_close'))

    def test_outcome_adapters_never_turn_missing_records_into_losses(self):
        games = [dict(id=101, home_score=4, away_score=2, teams=dict(home=dict(batters=[dict(name='P1', hits=2)], pitchers=[]),
                                                                      away=dict(batters=[], pitchers=[])))]
        e = dict(entry_id='x', mlb_game_id=101, market='batter_hits', player='P1')
        self.assertEqual(outcomes.mlb([e], games)['x'], dict(status='final', value=2))
        self.assertEqual(outcomes.mlb([dict(e, player='Nobody')], games)['x']['status'], 'did_not_participate')
        self.assertEqual(outcomes.mlb([dict(e, mlb_game_id=999)], games)['x']['status'], 'unknown')
        n = outcomes.nhl([dict(entry_id='y', nhl_game_id=1, player_id=7, market='player_shots_on_goal')],
                         [dict(game_id=1, home_score=3, away_score=2)], [])
        self.assertEqual(n['y']['status'], 'unknown')


class GateTests(unittest.TestCase):
    def report(self):
        return json.loads((ROOT/'docs/mlb/data/validation.json').read_text())

    def test_postseason_pitcher_outs_over_is_blocked_despite_aggregate_pass(self):
        audit = self.report()['postseason']['pitcher_outs']
        self.assertTrue(audit['passed'])
        over = dict(market='pitcher_outs', side='Over', model_probability=.85, model_push_probability=0)
        self.assertIn('overstated', gates.reasons(audit, over)[0])
        under = dict(market='pitcher_outs', side='Under', model_probability=.76, model_push_probability=0)
        self.assertEqual(gates.reasons(audit, under), [])  # P(Over)=.24 bin: overs overstated, so unders are not.

    def test_count_bias_blocks_only_the_favoured_side(self):
        audit = dict(passed=True, mean_prediction=16.0, mean_actual=14.0, forecasts=500,
                     calibration_bins=[dict(predicted=p/10+.05, observed=p/10+.05, n=100) for p in range(10)])
        self.assertTrue(gates.bias_reason(audit, dict(market='pitcher_outs', side='Over')))
        self.assertIsNone(gates.bias_reason(audit, dict(market='pitcher_outs', side='Under')))
        audit['mean_bias'] = dict(ci95=[-.5, 3.0])  # Interval includes zero: not established.
        self.assertIsNone(gates.bias_reason(audit, dict(market='pitcher_outs', side='Over')))

    def test_thin_ranges_and_unestablished_skill(self):
        audit = dict(calibration_bins=[dict(predicted=.55, observed=.5, n=12)], brier_difference=dict(ci95=[-.01, .002]))
        r = dict(market='h2h', side='H', home_team='H', model_probability=.55, model_push_probability=0)
        self.assertEqual(len(gates.reasons(audit, r)), 2)
        self.assertIn('not established', gates.reasons(audit, r)[0])
        self.assertEqual(gates.reference_side(dict(market='h2h', side='A', home_team='H')), False)


class LateCheckTests(unittest.TestCase):
    def test_material_change_detection_and_versions(self):
        r = dict(row(), candidate_id='c1')
        original = [src()]
        new = src(url='https://www.nhl.com/news/goalie-update', title='Goalie scratch update', excerpt='Example Player and Home goalie update')
        self.assertEqual(late.material_sources(r, [src()], original, NOW), [])
        found = late.material_sources(r, [src(), new], original, NOW)
        self.assertEqual([f['url'] for f in found], [new['url']])
        self.assertEqual(late.lineup_change(dict(sport='MLB', lineup_status='Not published'), dict(lineup_status='Both batting orders published'))[0]['kind'], 'lineup_published')
        self.assertEqual(late.change(dict(gate='model_case_only'), dict(gate='adverse_fact')), 'now_adverse_fact')
        self.assertEqual(late.change(dict(gate='model_case_only'), dict(gate='verified_context')), 'unchanged_eligibility')
        with tempfile.TemporaryDirectory() as td:
            index, records = Path(td)/'index.json', Path(td)/'records'
            record = dict(schema='late-reassessment-1', decision_date='2026-10-05', reassessed_at=iso(NOW), basis_edition_id='e1',
                          identity=dict(sport='NHL', game_id='g1'), current=dict(price=110, research_state=dict(label='x')), decision_change='unchanged_eligibility')
            first = late.publish(deepcopy(record), index, records)
            second = late.publish(dict(deepcopy(record), reassessed_at=iso(NOW+timedelta(hours=1))), index, records)
            self.assertEqual((first['version'], second['version']), (1, 2))
            self.assertEqual(len(json.loads(index.read_text())['reassessments']), 2)

    def test_disabled_check_never_pays(self):
        with tempfile.TemporaryDirectory() as td, patch.object(late.analyst_review, 'review') as paid:
            root = Path(td)
            (root/'docs/briefing').mkdir(parents=True)
            r = dict(row(), game_id='g1', commence_time=iso(NOW+timedelta(minutes=90)), review_sources=[])
            (root/'docs/briefing/morning-card.json').write_text(json.dumps(dict(kind='morning', decision_date='2026-10-05', edition_id='e1', rows=[r])))
            with patch.object(late.analyst_review, 'selected', return_value=dict(selected=[dict(r, review_bet_key='k')])), \
                 patch.object(late.evidence, 'collect', return_value=([src(url='https://www.nhl.com/news/injury', title='Injury report')], {})):
                summary = late.run(NOW, dict(late_check=dict(enabled=False)), feeds={}, execute=True, root=root, archive=root/'a')
            self.assertEqual(summary['status'], 'disabled')
            paid.assert_not_called()


if __name__ == '__main__':
    unittest.main()

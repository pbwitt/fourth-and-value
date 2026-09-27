"""Prospective NHL selection, evidence, spending and decision boundaries."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from nhl import analyst
from nhl.v2 import astra, evidence
from nhl.v2.candidates import shortlist, exclusion
from nhl.v2.data import ROOT, iso, write_json
from nhl.v2.decisions import validate, record_decision, evaluate
from nhl.v2.pricing import price

NOW = datetime(2026, 9, 26, 14, 30, tzinfo=timezone.utc)
CONFIG = json.loads((ROOT/'config/nhl_analyst.json').read_text())


def row(i=1, **changes):
    base = dict(offer_id=f'o{i}', forecast_id=f'f{i}', nhl_game_id=i, event_id=f'e{i}',
                home_team='Washington Capitals', away_team='Pittsburgh Penguins', player='',
                game='Pittsburgh Penguins @ Washington Capitals', book='test', book_label='Synthetic Book',
                market='totals', market_label='Total goals', line=6, side='Under', price=100,
                commence_time=iso(NOW+timedelta(hours=9)), quoted_at=iso(NOW-timedelta(minutes=1)),
                decision_at=iso(NOW), model_data_checked_at=iso(NOW-timedelta(hours=1)),
                model_version='nhl-v2.1', validation_status='experimental_predictive_evaluation_only',
                settlement_verified=True, settlement_profile='nhl_full_game_ot_so',
                sensitivity=dict(assumption='Synthetic fixture', win_min=.58, win_max=.72),
                other_books=0, market_probability=None, consensus_ev=None, key_drivers=[], uncertainties=[])
    base.update(price(dict(win=.66, push=.05, loss=.29), 100,
                      scenarios=[dict(win=.58, push=.07, loss=.35)]))
    return dict(base, **changes)


def state(rows=None):
    return dict(status='ready', last_success_at=iso(NOW), snapshot_id='snapshot1', rows=rows if rows is not None else [row()])


def board():
    return shortlist(state(), NOW, CONFIG)


def source(b=None, **changes):
    b = b or board()
    return dict(dict(source_id='s1', title='Washington Capitals practice notes',
                     url='https://www.nhl.com/news/synthetic-test-fixture',
                     published_at=iso(NOW-timedelta(hours=1)), retrieved_at=iso(NOW),
                     excerpt='The Capitals said the starting goalie will be announced after warmups. This is synthetic evidence for tests.',
                     candidate_ids=[b['candidates'][0]['candidate_id']]), **changes)


def response(b=None):
    b = b or board()
    review = dict(candidate_id=b['candidates'][0]['candidate_id'], status='needs_information',
                  assessment=dict(verdict='wait', reason='The goalie assumption needs verification.',
                      model_case='The sensitivity range qualifies at this quote.', price_case='The quote clears the scenario minimum.',
                      context_case='Starting goalie remains unverified.', blocking_checks=['Verify the starting goalie.']),
                  countercase='Unconfirmed goalie assumptions could change the scoring forecast.',
                  open_checks=['Verify the starting goalie before deciding.'], evidence=[dict(
                      source_id='s1', excerpt='the starting goalie will be announced after warmups',
                      interpretation='Wait for the announcement and check the forecast assumptions.',
                      kind='goalie', direction='context', represented_in='unknown')])
    return dict(status='completed', output=[dict(type='message', content=[dict(type='output_text', text=json.dumps({'reviews': [review]}))])],
                usage={'input_tokens': 1000, 'output_tokens': 200})


class CandidateTests(unittest.TestCase):
    def test_model_without_consensus_qualifies_market_without_model_does_not(self):
        r = row()
        self.assertIsNone(exclusion(r, NOW, CONFIG))
        r.update(independent_probability=None, market_probability=.9, other_books=6, consensus_ev=40)
        self.assertEqual(shortlist(state([r]), NOW, CONFIG)['candidates'], [])

    def test_fail_closed_on_missing_stale_future_or_invalid_inputs(self):
        for change in [dict(forecast_id=None), dict(nhl_game_id=None), dict(settlement_verified=False),
                       dict(quoted_at=iso(NOW-timedelta(minutes=31))), dict(quoted_at=iso(NOW+timedelta(seconds=1))),
                       dict(decision_at=iso(NOW-timedelta(minutes=31))), dict(model_data_checked_at=iso(NOW-timedelta(hours=36))),
                       dict(estimated_ev=float('nan')), dict(loss_probability=.9), dict(estimated_ev=.99),
                       dict(market='player_points', player_id=None), dict(commence_time=iso(NOW+timedelta(days=1))),
                       dict(commence_time=iso(NOW)), dict(sensitivity=None)]:
            with self.subTest(change=change):
                self.assertIsNotNone(exclusion(row(**change), NOW, CONFIG))

    def test_push_and_minimum_price_rank_gate(self):
        good = row()
        self.assertAlmostEqual(good['estimated_ev'], .37)
        self.assertIsNone(exclusion(good, NOW, CONFIG))
        self.assertEqual(exclusion(row(minimum_acceptable_decimal=2.1), NOW, CONFIG), 'fails_sensitivity_price')
        self.assertEqual(exclusion(row(rank_score=-.0001), NOW, CONFIG), 'insufficient_model_value')

    def test_all_games_best_identical_offer_deterministic_and_no_source_mutation(self):
        rows = [row(i) for i in range(1, 8)] + [row(1, offer_id='other-book', book='other', rank_score=.05)]
        before = deepcopy(rows)
        b = shortlist(state(rows), NOW, CONFIG)
        self.assertEqual(len(b['candidates']), 7)
        self.assertEqual(len({r['nhl_game_id'] for r in b['candidates']}), 7)
        self.assertEqual(b['candidates'][0]['offer_id'], 'o1')
        self.assertTrue(all(r['recommendation'] is False for r in b['candidates']))
        self.assertEqual(rows, before)
        self.assertEqual(b, shortlist(state(rows), NOW, CONFIG))

    def test_error_or_old_feed_cannot_reuse_shortlist(self):
        for s in [dict(state(), status='feed_error'), dict(state(), model_error='failed'),
                  dict(state(), last_success_at=iso(NOW-timedelta(hours=1)))]:
            b = shortlist(s, NOW, CONFIG)
            self.assertEqual(b['status'], 'unavailable')
            self.assertEqual(b['candidates'], [])


class EvidenceAndAstraTests(unittest.TestCase):
    def test_source_time_identity_and_host_guards(self):
        b = board(); r = b['candidates'][0]
        self.assertTrue(evidence.usable(source(), r, NOW))
        for changes in [dict(published_at=iso(NOW+timedelta(seconds=1))), dict(retrieved_at=iso(NOW+timedelta(seconds=1))),
                        dict(published_at=iso(NOW-timedelta(days=4))), dict(candidate_ids=[]),
                        dict(url='https://nhl.com.evil.example/news'), dict(url='http://www.nhl.com/news'),
                        dict(url='https://user:secret@nhl.com/news')]:
            self.assertFalse(evidence.usable(source(**changes), r, NOW))
        self.assertFalse(evidence.trusted('https://127.0.0.1/news'))
        self.assertFalse(evidence.matches(r, 'New York goalie news'))

    def test_schema_source_excerpt_and_identity_enforced(self):
        b = board(); original = deepcopy(b)
        got = astra.parse_response(response(), b, [source()], NOW)
        self.assertEqual(got[0]['offer_id'], b['candidates'][0]['offer_id'])
        self.assertIsNone(got[0]['probability_adjustment'])
        self.assertEqual(b, original)
        edits = [lambda r: r.update(candidate_id='wrong'), lambda r: r.update(probability=.8),
                 lambda r: r.update(status='research_support'), lambda r: r.update(countercase='Win confidence is 88%.'),
                 lambda r: r['evidence'][0].update(source_id='invented'),
                 lambda r: r['evidence'][0].update(excerpt='The goalie was officially confirmed.')]
        for edit in edits:
            raw = response(); value = json.loads(raw['output'][0]['content'][0]['text']); edit(value['reviews'][0])
            raw['output'][0]['content'][0]['text'] = json.dumps(value)
            with self.assertRaises(ValueError):
                astra.parse_response(raw, b, [source()], NOW)
        for raw in [dict(response(), status='incomplete'), dict(response(), output=[dict(type='message', content=[dict(type='refusal')])])]:
            with self.assertRaises(ValueError):
                astra.parse_response(raw, b, [source()], NOW)

    def test_payload_has_no_tools_or_substitute_model_and_budget_includes_schema(self):
        p = astra.payload(board(), [source()], NOW, CONFIG)
        self.assertEqual(p['model'], 'gpt-6-astra')
        self.assertFalse(p['store']); self.assertNotIn('tools', p)
        self.assertGreater(astra.bounds(p, CONFIG), .2)
        for bad in [dict(p, model='other'), dict(p, max_output_tokens=9000), dict(p, tools=[{'type': 'web_search'}]),
                    dict(p, text={'huge_schema': 'a'*26000})]:
            with self.assertRaises(ValueError):
                astra.bounds(bad, CONFIG)

    def test_budget_reservation_duplicate_rolling_cap_and_uncertain_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'budget.json'
            self.assertEqual(astra.reserve(path, 'today', NOW, .6, 1), 'reserved')
            self.assertEqual(astra.reserve(path, 'today', NOW, .6, 1), 'already_attempted')
            astra.settle_budget(path, 'today')
            self.assertEqual(json.loads(path.read_text())['entries'][0]['charge_usd'], .6)
            self.assertEqual(astra.reserve(path, 'later', NOW, .6, 1), 'budget_exhausted')
            self.assertEqual(astra.reserve(path, 'next-week', NOW+timedelta(days=8), .6, 1), 'reserved')

    def test_legacy_paid_entrypoint_routes_to_shared_queue(self):
        # The shared workflow tests cover actual paid NHL review. The old NHL
        # --astra flag can never create a second allowance or a duplicate call.
        with patch.object(astra,'call_api') as call:
            b=board(); before=deepcopy(b['candidates'])
            result=analyst.review(b,CONFIG,clock=lambda:NOW)
            self.assertEqual(result['review_status'],'shared_research_queue')
            self.assertEqual(result['candidates'],before)
            call.assert_not_called()


class DecisionTests(unittest.TestCase):
    def record(self, b):
        r = b['candidates'][0]
        return dict(board_id=b['board_id'], candidate_id=r['candidate_id'], offer_id=r['offer_id'], forecast_id=r['forecast_id'],
                    recorded_at=iso(NOW), analyst='Test Analyst', decision='select', reason='Synthetic context checked',
                    double_counting_check='Unknown; no probability adjustment', source_url=source()['url'],
                    source_published_at=source()['published_at'], price_confirmed=True, context_checked=True)

    def test_decisions_append_only_and_no_retrospective_selection(self):
        b = board(); r = self.record(b)
        self.assertEqual(validate(r, b, NOW, CONFIG)['evaluation_status'], 'prospective_shadow_only')
        for bad in [dict(r, recorded_at=iso(NOW+timedelta(minutes=31))), dict(r, offer_id='another'), dict(r, price_confirmed=False),
                    dict(r, source_published_at=iso(NOW+timedelta(minutes=1)))]:
            with self.assertRaises(ValueError):
                validate(bad, b, NOW+timedelta(hours=1), CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp); public = archive/'public.json'
            write_json(archive/'boards'/f"{b['board_id']}.json", b)
            record_decision(r, NOW, CONFIG, archive, public)
            with self.assertRaises(ValueError):
                record_decision(dict(r, decision='pass'), NOW, CONFIG, archive, public)

    def test_shadow_grading_preserves_screened_cohort_and_excludes_late_imports(self):
        b = board(); r = self.record(b)
        g = dict(game_id=1, home_score=3, away_score=2)
        early = validate(r, b, NOW, CONFIG)
        result = evaluate([b, b], [early], [g], [])
        self.assertEqual(result['metrics']['all_screened']['count'], 1)
        self.assertEqual(result['metrics']['human_selected']['count'], 1)
        late = dict(early, ingested_at=iso(NOW+timedelta(hours=10)))
        result = evaluate([b], [late], [g], [])
        self.assertEqual(result['metrics']['human_selected']['count'], 0)


if __name__ == '__main__':
    unittest.main()

"""Research reliability: partial failures, retries, idempotency, budget accounting and diagnostics."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyst_review as analyst
import morning_card as card
import morning_operations as ops
import research_budget as budget
from nhl.v2 import astra, evidence
from nhl.v2.data import iso, write_json
from test_analyst_review import NOW, CONFIG, feeds, board, source, response


def batch(n=3):
    b = board()
    b['candidates'] = [dict(b['candidates'][0], candidate_id=f'c{i}', review_key=f'["k{i}"]') for i in range(n)]
    return b


def multi_response(b, edit=None):
    raw = response(b)
    value = json.loads(raw['output'][0]['content'][0]['text'])
    template = value['reviews'][0]
    value['reviews'] = []
    for i, r in enumerate(b['candidates']):
        q = deepcopy(template); q['candidate_id'] = r['candidate_id']; q['evidence'] = []
        q['assessment'].update(verdict='consider', blocking_checks=[])
        if edit: edit(i, q)
        value['reviews'].append(q)
    raw['output'][0]['content'][0]['text'] = json.dumps(value)
    return raw


class ValidationTests(unittest.TestCase):
    def test_one_invalid_candidate_does_not_discard_valid_reviews(self):
        b = batch()
        def edit(i, q):
            if i == 1: q['assessment']['reason'] = 'Confidence is 80%.'
        accepted, rejected = astra.parse_response(multi_response(b, edit), b, [], NOW, schema=analyst.SCHEMA,
                                                  prompt_version=analyst.PROMPT_VERSION, partial=True)
        self.assertEqual([q['candidate_id'] for q in accepted], ['c0', 'c2'])
        self.assertEqual(rejected, [dict(candidate_id='c1', category='numeric_confidence')])
        with self.assertRaises(astra.ReviewError) as caught:
            astra.parse_response(multi_response(b, edit), b, [], NOW, schema=analyst.SCHEMA)
        self.assertEqual(caught.exception.category, 'numeric_confidence')

    def test_negated_certainty_is_accepted_and_affirmative_rejected(self):
        b = batch(1)
        for text, ok in [('Supports the planned starting role, not a guaranteed workload.', True),
                         ('There is no guaranteed role here.', True), ('This is a guaranteed workload.', False),
                         ('A lock for volume.', False)]:
            raw = multi_response(b, lambda i, q: q['assessment'].update(context_case=text))
            accepted, rejected = astra.parse_response(raw, b, [], NOW, schema=analyst.SCHEMA, partial=True)
            self.assertEqual(bool(accepted), ok, text)

    def test_batch_level_failures_raise_categories(self):
        b = batch(2)
        raw = multi_response(b); raw['status'] = 'incomplete'
        with self.assertRaises(astra.ReviewError) as caught:
            astra.parse_response(raw, b, [], NOW, schema=analyst.SCHEMA, partial=True)
        self.assertEqual(caught.exception.category, 'response_incomplete')
        raw = multi_response(b); value = json.loads(raw['output'][0]['content'][0]['text']); value['reviews'].pop()
        raw['output'][0]['content'][0]['text'] = json.dumps(value)
        with self.assertRaises(astra.ReviewError) as caught:
            astra.parse_response(raw, b, [], NOW, schema=analyst.SCHEMA, partial=True)
        self.assertEqual(caught.exception.category, 'candidate_identity_mismatch')
        raw = multi_response(b); raw['output'][0]['content'][0]['text'] = 'not json'
        with self.assertRaises(astra.ReviewError):
            astra.parse_response(raw, b, [], NOW, schema=analyst.SCHEMA, partial=True)

    def test_unsupported_positive_status_rejects_only_that_candidate(self):
        b = batch(2)
        raw = multi_response(b, lambda i, q: q.update(status='research_support') if i == 0 else None)
        accepted, rejected = astra.parse_response(raw, b, [], NOW, schema=analyst.SCHEMA, partial=True)
        self.assertEqual(rejected[0]['category'], 'unsupported_positive_status')
        self.assertEqual(len(accepted), 1)


class RunnerTests(unittest.TestCase):
    def run_review(self, raw, td, b=None):
        b = b or batch()
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic'}), patch.object(evidence, 'collect', return_value=([], {'status': 'no_usable_reporting'})), \
             patch.object(astra, 'checkpoint'), patch.object(astra, 'call_api', return_value=raw), \
             patch.object(analyst, 'selected', return_value={'selected': [dict(review_key=r['review_key']) for r in b['candidates']]}):
            return analyst.review(b, feeds(), CONFIG, Path(td), lambda: NOW)

    def test_partial_batch_records_failures_and_charges_once(self):
        b = batch()
        raw = multi_response(b, lambda i, q: q['assessment'].update(reason='Confidence is 80%.') if i == 2 else None)
        with tempfile.TemporaryDirectory() as td:
            result = self.run_review(raw, td, b)
            self.assertEqual(result['review_status'], 'partially_completed')
            self.assertEqual(result['batch']['rejected'], [dict(candidate_id='c2', category='numeric_confidence')])
            self.assertEqual(result['candidates'][2]['research_failure']['category'], 'numeric_confidence')
            self.assertNotIn('qualitative_review', result['candidates'][2])
            self.assertTrue(all('facts' in r['qualitative_review'] for r in result['candidates'][:2]))
            entries = [e for e in json.loads((Path(td)/'daily-budget.json').read_text())['entries'] if not e.get('legacy')]
            self.assertEqual(len(entries), 1); self.assertEqual(entries[0]['status'], 'settled')

    def test_transport_failure_is_categorized_and_never_retried(self):
        b = batch()
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic'}), \
             patch.object(evidence, 'collect', return_value=([], {})), patch.object(astra, 'checkpoint'), \
             patch.object(astra, 'call_api', side_effect=astra.ReviewError('api_http_error', 'Astra HTTP 500', http_status=500)), \
             patch.object(analyst, 'selected', return_value={'selected': [dict(review_key=r['review_key']) for r in b['candidates']]}):
            result = analyst.review(b, feeds(), CONFIG, Path(td), lambda: NOW)
            self.assertEqual((result['review_status'], result['review_error']), ('review_unavailable', 'api_http_error'))
            self.assertEqual(result['batch']['http_status'], 500)
            entry = [e for e in json.loads((Path(td)/'daily-budget.json').read_text())['entries'] if not e.get('legacy')][0]
            self.assertEqual(entry['status'], 'uncertain_reservation_retained')
            self.assertEqual(entry['charge_usd'], entry['reserved_usd'])
        self.assertNotIn('api_http_error', analyst.RETRYABLE)

    def test_queue_retries_validation_and_checkpoint_failures_once(self):
        b = board(); b['candidates'] = batch(4)['candidates']
        b['_research_queue'] = dict(pending=list(b['candidates']), sources=[], diagnostics={}, statuses=[], batches=[])
        calls = []
        def fake(batch_board, *args, **kwargs):
            ids = [r['candidate_id'] for r in batch_board['candidates']]
            calls.append((ids, batch_board['_attempt']))
            if len(calls) == 1:  # validation rejects c1
                for r in batch_board['candidates']:
                    if r['candidate_id'] != 'c1': r['qualitative_review'] = {'assessment': {'verdict': 'consider'}}
                return dict(batch_board, review_status='partially_completed',
                            batch=dict(candidate_ids=ids, status='partially_completed', rejected=[dict(candidate_id='c1', category='numeric_confidence')]))
            if len(calls) == 2:  # checkpoint failure: nothing sent
                return dict(batch_board, review_status='review_unavailable', batch=dict(candidate_ids=ids, status='review_unavailable', category='checkpoint_failed'))
            for r in batch_board['candidates']: r['qualitative_review'] = {'assessment': {'verdict': 'consider'}}
            return dict(batch_board, review_status='completed', batch=dict(candidate_ids=ids, status='completed'))
        with patch.object(analyst, 'review', side_effect=fake):
            analyst.run_queue([b], feeds(), dict(CONFIG, max_review_batches=8), Path('/unused'), lambda: NOW)
        self.assertEqual(calls[0], (['c0', 'c1', 'c2'], 0))
        self.assertEqual(calls[1], (['c3', 'c1'], 1))      # c1 retried once alongside c3
        # The checkpoint failure retries c3 once; c1 already used its single retry this run.
        self.assertEqual(calls[2], (['c3'], 1))
        self.assertEqual(len(calls), 3)
        self.assertEqual(b['reviewed_count'], 3)
        self.assertEqual(b['pending_count'], 1)
        self.assertEqual(b['coverage_summary']['batches_attempted'], 3)

    def test_budget_exhaustion_stops_queue_with_accurate_counts(self):
        b = board(); b['candidates'] = batch(6)['candidates']
        b['_research_queue'] = dict(pending=list(b['candidates']), sources=[], diagnostics={}, statuses=[], batches=[])
        def fake(batch_board, *args, **kwargs):
            ids = [r['candidate_id'] for r in batch_board['candidates']]
            return dict(batch_board, review_status='budget_exhausted', batch=dict(candidate_ids=ids, status='budget_exhausted'))
        with patch.object(analyst, 'review', side_effect=fake) as call:
            analyst.run_queue([b], feeds(), CONFIG, Path('/unused'), lambda: NOW)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(b['review_status'], 'budget_exhausted')
        self.assertEqual((b['reviewed_count'], b['pending_count']), (0, 6))
        self.assertEqual(b['coverage_summary']['not_attempted'], 3)


class BudgetTests(unittest.TestCase):
    def test_release_only_unsent_and_summary_never_double_counts(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'ledger.json'
            self.assertEqual(budget.reserve('a', NOW, .2, path=path, legacy=(), cap=2.75), 'reserved')
            self.assertEqual(budget.reserve('b', NOW, .2, path=path, legacy=(), cap=2.75), 'reserved')
            self.assertEqual(budget.reserve('c', NOW, .2, path=path, legacy=(), cap=2.75), 'reserved')
            budget.release_unsent('a', 'checkpoint_failed', path=path)
            budget.settle('b', dict(input_tokens=1000, output_tokens=100), path=path)
            budget.settle('c', None, path=path)
            with self.assertRaises(ValueError):
                budget.release_unsent('b', 'late', path=path)
            self.assertEqual(budget.reserve('a', NOW, .2, path=path, legacy=(), cap=2.75), 'already_attempted')
            s = budget.ledger_summary(NOW, path=path, legacy=())
            self.assertEqual(s['released_not_sent_usd'], .2)
            self.assertAlmostEqual(s['charged_or_reserved_usd'], s['settled_actual_usd']+s['uncertain_retained_usd']+s['outstanding_reserved_usd'])
            self.assertEqual(s['uncertain_retained_usd'], .2)


class RecoveryTests(unittest.TestCase):
    def edition(self, root, now, **sports):
        base = dict(refresh_status='success', available_at_publication=True, batch_statuses=['completed'], review_status='completed')
        research = {s: dict(base, **sports.get(s, {})) for s in ('NFL', 'MLB', 'NHL')}
        write_json(root/'docs/briefing/morning-card.json', dict(schema_version=1, kind='morning', status='research_incomplete',
            decision_date='2026-09-27', published_at=iso(now-timedelta(minutes=30)), edition_id='e1', rows=[],
            research=dict(sports=research)))
        write_json(root/'docs/props/top-picks.json', dict(generated_at=iso(now-timedelta(minutes=32))))
        write_json(root/'docs/mlb/data/latest.json', dict(model_checked_at=iso(now-timedelta(minutes=31))))

    def test_healthy_sports_are_reused_after_an_unrelated_refresh_failure(self):
        now = datetime(2026, 9, 27, 11, 35, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.edition(root, now, NHL=dict(refresh_status='failure', available_at_publication=False))
            scope = ops.gate(root, now)['scope']
            self.assertEqual(scope['reused'], ['NFL', 'MLB']); self.assertEqual(scope['refresh'], ['NHL'])
            self.assertEqual(scope['reasons']['NHL'], 'nhl_quotes_expire_in_30_minutes')

    def test_failed_batches_old_feeds_and_explicit_tests_refresh(self):
        now = datetime(2026, 9, 27, 11, 35, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.edition(root, now, NFL=dict(batch_statuses=['completed', 'review_unavailable']))
            self.assertEqual(ops.gate(root, now)['scope']['reused'], ['MLB'])
            self.assertEqual(ops.gate(root, now+timedelta(minutes=40))['scope']['reused'], [])
            self.assertEqual(ops.gate(root, now, test_edition=True)['scope']['reused'], [])

    def test_reused_feed_that_expires_is_an_issue(self):
        selection = dict(coverage=[dict(sport=s, available=s != 'NFL') for s in ('NFL', 'MLB', 'NHL')])
        reviews = dict(sports={s: dict(candidate_count=0, reviewed_count=0, review_status='no_candidates') for s in ('NFL', 'MLB', 'NHL')})
        health = card.research_health({}, reviews, selection, NOW, dict(NFL='reused', MLB='reused', NHL='success'))
        self.assertIn('NFL: reused feed expired before publication', health['issues'])
        self.assertFalse(any(i.startswith('MLB') for i in health['issues']))

    def test_card_issues_name_failure_categories_and_candidates(self):
        selection = dict(coverage=[dict(sport='MLB', available=True)])
        board = dict(candidate_count=3, reviewed_count=2, review_status='partially_reviewed', batch_statuses=['partially_completed'],
                     batches=[dict(index=1, status='partially_completed', candidate_ids=['a', 'b', 'c'], rejected=[dict(candidate_id='c', category='numeric_confidence')])],
                     coverage_summary=dict(failure_categories={'numeric_confidence': 1}, failed=[dict(candidate_id='c', category='numeric_confidence')]),
                     evidence_status=dict(status='available', failures=[dict(host='x', stage='feed_unavailable', category='timeout')], coverage={'a': 1, 'c': 0}))
        health = card.research_health({}, dict(sports=dict(MLB=board)), selection, NOW, dict(MLB='success'))
        self.assertIn('MLB: some assessments failed (numeric_confidence ×1)', health['issues'])
        self.assertEqual(health['sports']['MLB']['batches'][0]['rejected'][0]['candidate_id'], 'c')
        self.assertEqual(health['sports']['MLB']['source_diagnostics']['failures'], {'feed_unavailable:timeout': 1})
        self.assertEqual(health['sports']['MLB']['source_diagnostics']['candidates_without_sources'], ['c'])


class EvidenceTests(unittest.TestCase):
    def test_retrieval_failures_are_sanitized_categories(self):
        import requests
        response = requests.Response(); response.status_code = 403
        self.assertEqual(evidence.failure_category(requests.HTTPError(response=response)), 'http_403')
        self.assertEqual(evidence.failure_category(requests.Timeout('https://secret.example/?token=x')), 'timeout')
        self.assertEqual(evidence.failure_category(ValueError('Publisher not allowed')), 'not_allowed')


if __name__ == '__main__':
    unittest.main()

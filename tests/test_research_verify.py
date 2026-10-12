"""Targeted verification: one bounded, budgeted search per strongest candidate."""
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyst_review as analyst
import model_api
import research_verify as verify
from nhl.v2 import astra, evidence
from nhl.v2.data import iso
from test_analyst_review import NOW, CONFIG, KEY, board, feeds, source

URL = 'https://www.espn.com/mlb/story/synthetic-injury'


def reply(findings, searched=(URL,)):
    """A Claude reply converted to the shared response fields."""
    content = [dict(type='server_tool_use', id='s1', name='web_search', input={'query': 'q'}),
               dict(type='web_search_tool_result', tool_use_id='s1', content=[
                   dict(type='web_search_result', url=u, title='t') for u in searched]),
               dict(type='text', text=json.dumps(dict(findings=findings)))]
    message = dict(id='m', model=astra.MODEL, stop_reason='end_turn', content=content,
                   usage=dict(input_tokens=20000, output_tokens=900, server_tool_use={'web_search_requests': 1}))
    return model_api.from_claude(message, verify.search_request(board()['candidates'][0], NOW))


def fetched(rows, urls, clock):
    return [dict(source(dict(sport='MLB', candidates=rows)), source_id='v1', url=u) for u in urls], []


class RequestTests(unittest.TestCase):
    def test_request_is_one_search_about_one_bet_and_is_budgeted(self):
        row = board()['candidates'][0]
        r = verify.search_request(row, NOW)
        self.assertEqual((r['model'], r['max_tool_calls'], r['tool_choice']), (astra.MODEL, 1, 'required'))
        self.assertEqual(json.loads(r['input'])['bet']['player'], 'Synthetic Player')
        self.assertNotIn('model_probability', r['input'])
        amount = verify.bounds(r)
        self.assertGreater(amount, verify.SEARCH_CONTEXT_TOKENS*astra.INPUT_RATE)
        self.assertLess(amount, 1)
        for bad in [dict(r, max_tool_calls=3), dict(r, model='other'), dict(r, max_output_tokens=verify.MAX_OUTPUT_TOKENS+1),
                    dict(r, instructions='x'*verify.MAX_REQUEST_BYTES)]:
            with self.assertRaises(ValueError):
                verify.bounds(bad)

    def test_only_trusted_links_the_search_returned_are_kept(self):
        kept = verify.findings(reply([
            dict(url=URL, stance='against', kind='injury', note='Listed as questionable.'),
            dict(url='https://www.espn.com/invented', stance='supports', kind='role', note='Not from the search.'),
            dict(url='https://evil.test/x', stance='context', kind='other', note='Untrusted.')],
            searched=(URL, 'https://evil.test/x')))
        self.assertEqual([f['url'] for f in kept], [URL])
        with self.assertRaises(ValueError):
            verify.findings(dict(reply([]), status='incomplete'))
        with self.assertRaises(ValueError):
            verify.findings(reply([dict(url=URL, stance='maybe', kind='injury', note='x')]))


class RunTests(unittest.TestCase):
    def test_paid_search_is_reserved_settled_fetched_and_never_repeated(self):
        rows = board()['candidates']
        found = [dict(url=URL, stance='against', kind='injury', note='Listed as questionable.')]
        with tempfile.TemporaryDirectory() as td, patch.object(astra, 'checkpoint') as checkpoint, \
             patch.object(astra, 'call_api', return_value=reply(found)) as call, \
             patch.object(evidence, 'targeted', side_effect=fetched) as fetch:
            archive = Path(td)
            sources, records = verify.run(rows, rows, CONFIG, archive, lambda: NOW)
            self.assertEqual(records[0]['status'], 'completed')
            self.assertEqual([s['url'] for s in sources], [URL])
            checkpoint.assert_called_once(); call.assert_called_once()
            (entry,) = json.loads((archive/'daily-budget.json').read_text())['entries']
            self.assertEqual(entry['status'], 'settled')
            self.assertAlmostEqual(entry['charge_usd'], 20000*astra.INPUT_RATE+900*astra.OUTPUT_RATE+.01)
            self.assertLess(entry['charge_usd'], entry['reserved_usd'])
            again, records = verify.run(rows, rows, CONFIG, archive, lambda: NOW+timedelta(minutes=30))
            self.assertEqual(records[0]['status'], 'reused')
            self.assertEqual([s['url'] for s in again], [URL])
            call.assert_called_once()
            self.assertEqual(fetch.call_count, 2)

    def test_budget_failure_and_bad_reply_fail_closed(self):
        rows = board()['candidates']
        with tempfile.TemporaryDirectory() as td, patch.object(astra, 'checkpoint'), \
             patch.object(astra, 'call_api') as call:
            sources, records = verify.run(rows, rows, dict(CONFIG, daily_budget_usd=.05), Path(td), lambda: NOW)
            self.assertEqual((sources, records[0]['status']), ([], 'budget_exhausted'))
            call.assert_not_called()
        with tempfile.TemporaryDirectory() as td, patch.object(astra, 'checkpoint'), \
             patch.object(astra, 'call_api', return_value=dict(reply([]), status='incomplete')):
            sources, records = verify.run(rows, rows, CONFIG, Path(td), lambda: NOW)
            self.assertEqual((sources, records[0]['status']), ([], 'verification_unavailable'))
            (entry,) = json.loads((Path(td)/'daily-budget.json').read_text())['entries']
            self.assertEqual(entry['status'], 'settled')

    def test_queue_verifies_the_strongest_first_and_adds_sources_to_the_review(self):
        b = board(); base = b['candidates'][0]
        b['candidates'] = [dict(base, candidate_id=str(i), review_key=f'k{i}') for i in range(6)]
        b['sources'] = []
        b['_research_queue'] = dict(pending=list(b['candidates']), sources=[], diagnostics={}, statuses=[])
        seen = []
        def reviewed(batch, *args, **kwargs):
            seen.append([s['url'] for s in batch['_prepared_evidence'][0]])
            return dict(batch, review_status='completed')
        def searched(rows, batch, config, archive, clock):
            return [dict(source(dict(sport='MLB', candidates=batch)), source_id='v'+r['candidate_id'], url=URL+r['candidate_id']) for r in rows], \
                [dict(candidate_id=r['candidate_id'], status='completed') for r in rows]
        with patch.object(analyst, 'review', side_effect=reviewed), \
             patch.object(analyst.research_verify, 'run', side_effect=searched) as run:
            analyst.run_queue([b], feeds(), dict(CONFIG, verification_candidates=3, verification_per_batch=2),
                              Path('/unused'), lambda: NOW)
        self.assertEqual([[r['candidate_id'] for r in c.args[0]] for c in run.call_args_list], [['0', '1'], ['3']])
        self.assertEqual(seen[0], [URL+'0', URL+'1'])
        self.assertEqual([s['url'] for s in b['sources']], [URL+'0', URL+'1', URL+'3'])
        self.assertTrue(all('excerpt' not in s for s in b['sources']))
        self.assertEqual(len(b['verification']), 3)


if __name__ == '__main__':
    unittest.main()

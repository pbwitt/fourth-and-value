"""Replay the recorded October 5, 2026 research artifacts through the current code.

Uses immutable request/response/board archives committed under artifacts/analyst/ and
the published card. No network and no paid calls.
"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import analyst_review as analyst
from nhl.v2 import astra
from nhl.v2.data import stamp

ARCHIVE = ROOT/'artifacts/analyst'


def replay(request_id):
    packet = json.loads((ARCHIVE/'requests'/f'{request_id}.json').read_text())
    board = json.loads((ARCHIVE/'boards'/f"{packet['board_id']}.json").read_text())
    request = json.loads(packet['request']['input'])
    rows = []
    for c in request['candidates']:
        original = next(r for r in board['candidates'] if analyst.normalized(r)['candidate_id'] == c['candidate_id'])
        rows.append(dict(analyst.normalized(original), offer_id='o', forecast_id='f'))
    supplied = {s['source_id']: s for s in request['sources']}
    sources = [dict(s, excerpt=supplied[s['source_id']]['excerpt']) for s in packet['collected_sources'] if s['source_id'] in supplied]
    response = json.loads((ARCHIVE/'responses'/f'{request_id}.json').read_text())['response']
    schema = packet['request']['text']['format']['schema']
    return astra.parse_response(response, {'candidates': rows}, sources, stamp(request['review_asof']), schema=schema,
                                prompt_version=request['prompt_version'], partial=True)


class October5ReplayTests(unittest.TestCase):
    def test_negation_false_positive_no_longer_discards_a_paid_batch(self):
        # 12:09 ET MLB batch: "not a guaranteed workload" rejected all three reviews.
        accepted, rejected = replay('4a3416733392378777efeb36')
        self.assertEqual((len(accepted), rejected), (3, []))

    def test_one_unsupported_status_rejects_only_its_candidate(self):
        # 12:39 ET MLB batch: one research_support without a supporting item.
        accepted, rejected = replay('402bdc842301251fadd8a3e1')
        self.assertEqual(len(accepted), 2)
        self.assertEqual([r['category'] for r in rejected], ['unsupported_positive_status'])

    def test_unsent_reservation_is_identifiable_from_the_ledger(self):
        # 12:39:53 NFL batch: reservation committed, no response, never settled.
        ledger = json.loads((ARCHIVE/'daily-budget.json').read_text())
        entry = next(e for e in ledger['entries'] if 'd7db6ef2c1c7f109b580' in e['key'])
        self.assertEqual(entry['status'], 'reserved')
        self.assertFalse((ARCHIVE/'responses/6605451629579aec9a19be91.json').exists())

    def test_published_card_rows_are_model_case_only_not_corroborated(self):
        card = json.loads((ROOT/'docs/briefing/cards/2026-10-05-466e6f14cadd62482bcc79b8.json').read_text())
        script = ("const {researchState}=require('./docs/assets/briefing-picks.js');"
                  "const card=JSON.parse(require('fs').readFileSync(0,'utf8'));"
                  "process.stdout.write(JSON.stringify(card.rows.map(r=>researchState(r,Date.parse(card.published_at)))));")
        states = json.loads(subprocess.run(['node', '-e', script], input=json.dumps(card), capture_output=True, text=True,
                                           check=True, cwd=ROOT).stdout)
        self.assertEqual(len(states), 5)
        self.assertEqual({s['gate'] for s in states}, {'model_case_only'})
        self.assertEqual({s['evidence'] for s in states}, {'none_verified'})
        self.assertTrue(all(s['label'] == 'Consider · model case only' for s in states))


if __name__ == '__main__':
    unittest.main()

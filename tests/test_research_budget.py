"""Actual daily boundary, legacy spend, concurrency, and unknown billable outcomes."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import research_budget as budget

NOW=datetime(2026,9,27,14,tzinfo=timezone.utc)

class BudgetTests(unittest.TestCase):
    def test_calendar_reset_including_dst_and_no_rolling_reset(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'daily.json'
            # Both repetitions of 01:30 on the fall-back day share one allowance.
            a=datetime(2026,11,1,5,30,tzinfo=timezone.utc)
            self.assertEqual(budget.reserve('a',a,2,path=p,legacy=()),'reserved')
            self.assertEqual(budget.reserve('b',a+timedelta(hours=1),1,path=p,legacy=()),'budget_exhausted')
            self.assertEqual(budget.reserve('c',datetime(2026,11,2,4,59,tzinfo=timezone.utc),1,path=p,legacy=()),'budget_exhausted')
            self.assertEqual(budget.reserve('d',datetime(2026,11,2,5,tzinfo=timezone.utc),2,path=p,legacy=()),'reserved')

    def test_unknown_response_retains_full_cost_and_duplicate_never_retries(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'daily.json'
            budget.reserve('a',NOW,2.5,path=p,legacy=())
            budget.settle('a',None,path=p)
            self.assertEqual(budget.reserve('a',NOW,.1,path=p,legacy=()),'already_attempted')
            self.assertEqual(budget.reserve('b',NOW,.3,path=p,legacy=()),'budget_exhausted')
            self.assertEqual(budget.usage_summary(NOW,path=p,legacy=())['remaining_usd'],.25)

    def test_legacy_import_idempotent_and_search_fee_included(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'daily.json';old=Path(td)/'old.json'
            old.write_text(json.dumps(dict(entries=[dict(key='old',at=budget.iso(NOW),charge_usd=2)])))
            self.assertEqual(budget.reserve('a',NOW,.8,path=p,legacy=(old,)),'budget_exhausted')
            self.assertEqual(budget.reserve('a',NOW,.5,path=p,legacy=(old,)),'reserved')
            budget.settle('a',dict(input_tokens=1000,output_tokens=100),path=p,search_calls=1)
            self.assertAlmostEqual(budget.usage_summary(NOW,path=p,legacy=(old,))['charged_or_reserved_usd'],2.0275)

    def test_concurrent_reservations_cannot_each_spend_full_balance(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'daily.json'
            with ThreadPoolExecutor(max_workers=5) as pool:
                result=list(pool.map(lambda i:budget.reserve(str(i),NOW,1,path=p,legacy=()),range(5)))
            self.assertEqual(result.count('reserved'),2)
            self.assertEqual(result.count('budget_exhausted'),3)

    def test_unexpected_usage_halts_further_spending_and_invalid_cap_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'daily.json'
            with self.assertRaises(ValueError):budget.reserve('bad',NOW,.1,path=p,legacy=(),cap=3)
            budget.reserve('a',NOW,.1,path=p,legacy=())
            budget.settle('a',dict(input_tokens=20000,output_tokens=0),path=p)
            self.assertEqual(budget.reserve('b',NOW,.1,path=p,legacy=()),'budget_halted')

if __name__=='__main__':unittest.main()

class ReservePolicyTests(unittest.TestCase):
    def test_later_allowance_is_part_of_same_day(self):
        cfg={'daily_budget_usd':2.75,'later_reserve_usd':.75}
        self.assertEqual(budget.run_cap(NOW,cfg),2)
        self.assertEqual(budget.run_cap(NOW+timedelta(hours=3),cfg),2.75)

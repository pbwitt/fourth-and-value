"""Shared Odds API credit budget: cost estimates, the shrinking core reserve and the per-run cap."""
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import odds_budget
from odds_budget import CORE, OPTIONAL, CreditBudget, CreditFloorReached, estimate_cost


class Response:
    def __init__(self, remaining=None, last=None, payload=None, status=200):
        self.headers = {}
        if remaining is not None:
            self.headers['x-requests-remaining'] = str(remaining)
        if last is not None:
            self.headers['x-requests-last'] = str(last)
        self.payload = payload if payload is not None else []
        self.status_code = status
        self.ok = status < 400

    def json(self):
        return self.payload


class EstimateCost(unittest.TestCase):
    def test_documented_costs(self):
        self.assertEqual(estimate_cost('sports'), 0)
        self.assertEqual(estimate_cost('sports/icehockey_nhl/events', {'dateFormat': 'iso'}), 0)
        self.assertEqual(estimate_cost('sports/icehockey_nhl/odds', {'markets': 'h2h,spreads,totals', 'regions': 'us'}), 3)
        self.assertEqual(estimate_cost('sports/icehockey_nhl/odds', {'markets': 'totals', 'regions': 'us,eu'}), 2)
        self.assertEqual(estimate_cost('sports/nfl/events/abc/odds', {'markets': ','.join(['m'] * 8)}), 8)
        self.assertEqual(estimate_cost('sports/nfl/scores'), 1)
        self.assertEqual(estimate_cost('sports/nfl/scores', {'daysFrom': 3}), 2)
        self.assertEqual(estimate_cost('historical/sports/nhl/odds', {'markets': 'h2h,spreads,totals'}), 30)
        self.assertEqual(estimate_cost('historical/sports/nhl/events', {'date': 'x'}), 1)
        self.assertEqual(estimate_cost('https://api.the-odds-api.com/v4/sports/nfl/odds?apiKey=k', {'markets': 'totals'}), 1)


def clock(value):
    return lambda: datetime.fromisoformat(value)


class Reserve(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ('ODDS_CREDIT_FLOOR', 'ODDS_CORE_DAILY', 'ODDS_RUN_CAP'):
            os.environ.pop(name, None)

    def test_core_refreshes_may_spend_to_zero(self):
        budget = CreditBudget(tier=CORE)
        budget.observe(Response(remaining=10).headers)
        budget.ensure(10)                       # lands exactly on zero
        with self.assertRaisesRegex(CreditFloorReached, '0-credit floor'):
            budget.ensure(11)

    def test_optional_jobs_leave_the_core_reserve_until_the_reset(self):
        # 16.5 days before the November 1 reset: 500 x 16.5 = 8,250 kept for core refreshes.
        budget = CreditBudget(clock=clock('2026-10-15T12:00:00+00:00'))
        self.assertEqual(budget.tier, OPTIONAL, 'unknown jobs are optional')
        self.assertAlmostEqual(budget.reserve(), 8250)
        budget.observe(Response(remaining=8300).headers)
        budget.ensure(50)
        with self.assertRaisesRegex(CreditFloorReached, 'kept for the NFL, NHL and MLB refreshes'):
            budget.ensure(51)

    def test_reserve_shrinks_to_zero_at_the_monthly_reset(self):
        late = CreditBudget(clock=clock('2026-10-31T20:00:00+00:00'))   # four hours left
        self.assertAlmostEqual(late.reserve(), 500 * 4 / 24)
        self.assertEqual(odds_budget.next_reset(datetime(2026, 12, 15, tzinfo=timezone.utc)),
                         datetime(2027, 1, 1, tzinfo=timezone.utc))
        fresh = CreditBudget(clock=clock('2026-11-01T00:00:00+00:00'))
        self.assertAlmostEqual(fresh.reserve(), 500 * 30, msg='a full month is reserved right after the reset')

    def test_free_requests_always_allowed(self):
        budget = CreditBudget()
        budget.observe(Response(remaining=5).headers)
        budget.ensure(0)

    def test_unknown_balance_allows_one_small_request_only(self):
        budget = CreditBudget()
        budget.ensure(odds_budget.UNKNOWN_BALANCE_MAX)
        with self.assertRaisesRegex(CreditFloorReached, 'balance is unknown'):
            budget.ensure(odds_budget.UNKNOWN_BALANCE_MAX + 1)

    def test_preflight_supplies_the_balance(self):
        budget = CreditBudget(tier=CORE)
        budget.ensure(500, preflight=lambda b: b.observe(Response(remaining=9000).headers))
        self.assertEqual(budget.remaining, 9000)
        low = CreditBudget(clock=clock('2026-10-31T00:00:00+00:00'))   # 500 reserved
        with self.assertRaises(CreditFloorReached):
            low.ensure(5, preflight=lambda b: b.observe(Response(remaining=501).headers))

    def test_run_cap(self):
        budget = CreditBudget(tier=CORE)
        budget.observe(Response(remaining=15000, last=1995).headers)
        budget.ensure(5)
        with self.assertRaisesRegex(CreditFloorReached, 'per-run cap'):
            budget.ensure(6)

    def test_environment_can_only_tighten(self):
        with mock.patch.dict(os.environ, {'ODDS_CREDIT_FLOOR': '-5', 'ODDS_CORE_DAILY': '100', 'ODDS_RUN_CAP': '9000'}):
            loose = CreditBudget()
        self.assertEqual((loose.floor, loose.core_daily, loose.run_cap), (0, 500, 2000))
        with mock.patch.dict(os.environ, {'ODDS_CREDIT_FLOOR': '3000', 'ODDS_CORE_DAILY': '800', 'ODDS_RUN_CAP': '300'}):
            strict = CreditBudget(tier=CORE)
        self.assertEqual((strict.floor, strict.core_daily, strict.run_cap), (3000, 800, 300))
        self.assertEqual(strict.reserve(), 3000, 'a raised floor applies to core refreshes too')

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            CreditBudget(tier='urgent')


class SharedClient(unittest.TestCase):
    """The NBA client is also the NHL and MLB client."""

    def test_paid_request_refused_below_floor(self):
        from nba.pipeline import CreditFloorError, FeedError, OddsClient
        client = OddsClient('key', 'icehockey_nhl')
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(url)
            return Response(remaining=2, last=0 if url.endswith('/events') else 3)

        with mock.patch('nba.pipeline.requests.get', fake_get):
            client.get('events', dateFormat='iso')                      # free; reports 2 left
            with self.assertRaises(CreditFloorError) as caught:
                client.get('odds', regions='us', markets='h2h,spreads,totals')
        self.assertIsInstance(caught.exception, FeedError, 'existing feed-error handling applies')
        self.assertEqual(len(calls), 1, 'the refused request never reached the provider')

    def test_card_sports_are_core_and_nba_is_optional(self):
        from nba.pipeline import OddsClient
        self.assertEqual(OddsClient('key', 'icehockey_nhl').budget.tier, CORE)
        self.assertEqual(OddsClient('key', 'baseball_mlb').budget.tier, CORE)
        self.assertEqual(OddsClient('key').budget.tier, OPTIONAL)

    def test_normal_balance_unchanged(self):
        from nba.pipeline import OddsClient
        client = OddsClient('key', 'basketball_nba')
        with mock.patch('nba.pipeline.requests.get', lambda *a, **k: Response(remaining=15000, last=3, payload=[1])):
            self.assertEqual(client.get('odds', regions='us', markets='h2h,spreads,totals'), [1])
        self.assertEqual(client.quota_remaining, '15000')
        self.assertEqual(client.budget.spent, 3)


if __name__ == '__main__':
    unittest.main()

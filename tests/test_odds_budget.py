"""Shared Odds API credit floor: cost estimates, the 2,000 floor and the per-run cap."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import odds_budget
from odds_budget import CreditBudget, CreditFloorReached, estimate_cost


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


class Floor(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('ODDS_CREDIT_FLOOR', None)
        os.environ.pop('ODDS_RUN_CAP', None)

    def test_refuses_a_request_that_could_cross_the_floor(self):
        budget = CreditBudget()
        budget.observe(Response(remaining=2010).headers)
        budget.ensure(10)                       # lands exactly on 2,000
        with self.assertRaisesRegex(CreditFloorReached, '2000-credit floor'):
            budget.ensure(11)

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
        budget = CreditBudget()
        budget.ensure(500, preflight=lambda b: b.observe(Response(remaining=9000).headers))
        self.assertEqual(budget.remaining, 9000)
        low = CreditBudget()
        with self.assertRaises(CreditFloorReached):
            low.ensure(5, preflight=lambda b: b.observe(Response(remaining=2001).headers))

    def test_run_cap(self):
        budget = CreditBudget()
        budget.observe(Response(remaining=15000, last=1995).headers)
        budget.ensure(5)
        with self.assertRaisesRegex(CreditFloorReached, 'per-run cap'):
            budget.ensure(6)

    def test_environment_can_only_tighten(self):
        with mock.patch.dict(os.environ, {'ODDS_CREDIT_FLOOR': '500', 'ODDS_RUN_CAP': '9000'}):
            loose = CreditBudget()
        self.assertEqual((loose.floor, loose.run_cap), (2000, 2000))
        with mock.patch.dict(os.environ, {'ODDS_CREDIT_FLOOR': '3000', 'ODDS_RUN_CAP': '300'}):
            strict = CreditBudget()
        self.assertEqual((strict.floor, strict.run_cap), (3000, 300))


class SharedClient(unittest.TestCase):
    """The NBA client is also the NHL and MLB client."""

    def test_paid_request_refused_below_floor(self):
        from nba.pipeline import CreditFloorError, FeedError, OddsClient
        client = OddsClient('key', 'icehockey_nhl')
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(url)
            return Response(remaining=2002, last=0 if url.endswith('/events') else 3)

        with mock.patch('nba.pipeline.requests.get', fake_get):
            client.get('events', dateFormat='iso')                      # free; reports 2,002 left
            with self.assertRaises(CreditFloorError) as caught:
                client.get('odds', regions='us', markets='h2h,spreads,totals')
        self.assertIsInstance(caught.exception, FeedError, 'existing feed-error handling applies')
        self.assertEqual(len(calls), 1, 'the refused request never reached the provider')

    def test_normal_balance_unchanged(self):
        from nba.pipeline import OddsClient
        client = OddsClient('key', 'basketball_nba')
        with mock.patch('nba.pipeline.requests.get', lambda *a, **k: Response(remaining=15000, last=3, payload=[1])):
            self.assertEqual(client.get('odds', regions='us', markets='h2h,spreads,totals'), [1])
        self.assertEqual(client.quota_remaining, '15000')
        self.assertEqual(client.budget.spent, 3)


if __name__ == '__main__':
    unittest.main()

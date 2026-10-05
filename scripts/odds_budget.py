"""Shared Odds API credit floor for every job that spends credits.

The owner's rule: no job may take the shared Odds API balance below 2,000 credits,
and no single run may spend more than 2,000 without approval. Every paid request
goes through ``CreditBudget.ensure`` first and ``CreditBudget.observe`` after.

``ODDS_CREDIT_FLOOR`` and ``ODDS_RUN_CAP`` can make the limits stricter, never looser.
The balance comes from the provider's ``x-requests-remaining`` header. Until a
response has reported it, only a free preflight or one small request is allowed.
"""
from __future__ import annotations

import os
import re

FLOOR = 2000
RUN_CAP = 2000
UNKNOWN_BALANCE_MAX = 50   # the most one request may cost before the balance is known
SPORTS_URL = 'https://api.the-odds-api.com/v4/sports'   # free; reports the balance


class CreditFloorReached(RuntimeError):
    """A paid Odds API request was refused to protect the shared balance."""


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _limit(name, default, stricter):
    value = _number(os.getenv(name))
    return default if value is None else stricter(default, value)


def estimate_cost(path, params=None):
    """Upper bound on one request's cost from the documented Odds API rules.

    Event lists and the sports list are free; odds cost markets x regions; scores
    cost 1 (2 with daysFrom); historical odds cost 10x and historical events 1.
    """
    params = params or {}
    path = re.sub(r'^https?://[^/]+/v4/', '', str(path)).split('?')[0].strip('/')
    historical = path.startswith('historical/')
    if historical and path.endswith('/events'):
        return 1
    if path in ('sports', '') or path.endswith('/events'):
        return 0
    if path.endswith('/scores'):
        return 2 if params.get('daysFrom') else 1
    markets = len([m for m in str(params.get('markets') or 'h2h').split(',') if m.strip()]) or 1
    regions = len([r for r in str(params.get('regions') or 'us').split(',') if r.strip()]) or 1
    return markets * regions * (10 if historical else 1)


class CreditBudget:
    def __init__(self, floor=None, run_cap=None, label='odds'):
        self.floor = max(FLOOR, floor or 0, _limit('ODDS_CREDIT_FLOOR', FLOOR, max))
        self.run_cap = min(RUN_CAP, run_cap or RUN_CAP, _limit('ODDS_RUN_CAP', RUN_CAP, min))
        self.remaining = None
        self.spent = 0.0
        self.label = label

    def observe(self, headers):
        """Record the balance and cost the provider reported for a response."""
        if headers is None:
            return
        left = _number(headers.get('x-requests-remaining'))
        if left is not None:
            self.remaining = left
        last = _number(headers.get('x-requests-last'))
        if last:
            self.spent += last

    def ensure(self, cost, preflight=None):
        """Raise CreditFloorReached unless a request costing ``cost`` is allowed."""
        if cost <= 0:
            return
        if self.spent + cost > self.run_cap:
            raise CreditFloorReached(
                f'{self.label}: this run has spent {self.spent:.0f} credits; the per-run cap is '
                f'{self.run_cap:.0f}. Larger jobs need the owner\'s approval.')
        if self.remaining is None and preflight is not None:
            try:
                preflight(self)
            except Exception:
                pass   # an unknown balance is handled below
        if self.remaining is None:
            if cost > UNKNOWN_BALANCE_MAX:
                raise CreditFloorReached(
                    f'{self.label}: the Odds API balance is unknown, so a {cost:.0f}-credit request was refused.')
            return
        if self.remaining - cost < self.floor:
            raise CreditFloorReached(
                f'{self.label}: {self.remaining:.0f} credits left; this request could take the shared '
                f'balance below the {self.floor:.0f}-credit floor.')


def requests_preflight(api_key, session=None):
    """A free /sports call that reports the balance, for budgets that need it first."""
    def run(budget):
        import requests
        response = (session or requests).get(SPORTS_URL, params={'apiKey': api_key}, timeout=20)
        budget.observe(response.headers)
    return run


def urllib_preflight(api_key):
    """The same free balance check for scripts that use urllib instead of requests."""
    def run(budget):
        import urllib.parse
        import urllib.request
        url = SPORTS_URL + '?' + urllib.parse.urlencode({'apiKey': api_key})
        request = urllib.request.Request(url, headers={'User-Agent': 'fourth-and-value/1.0'})
        with urllib.request.urlopen(request, timeout=20) as response:
            budget.observe(response.headers)
    return run

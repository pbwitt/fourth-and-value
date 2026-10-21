"""Shared Odds API credit budget for every job that spends credits.

The provider resets the balance at midnight UTC on the 1st of each month (observed
October 1, 2026) and unused credits do not carry over, so nothing is held back for its
own sake. The owner's rules:

- Core jobs, the NFL, NHL and MLB refreshes the morning card is built from and, from
  October 17, 2026, the NBA refresh, may spend the balance down to zero.
- Optional jobs (the Pinnacle snapshot, editorial prices) must leave a
  reserve for the core refreshes until the reset: ``CORE_DAILY`` credits for each day
  left, shrinking to zero at the reset. Unknown jobs are optional.
- No single run may spend more than 2,000 credits without the owner's approval.

Every paid request goes through ``CreditBudget.ensure`` first and ``CreditBudget.observe``
after. ``ODDS_CREDIT_FLOOR``, ``ODDS_CORE_DAILY`` and ``ODDS_RUN_CAP`` can make the limits
stricter, never looser. The balance comes from the provider's ``x-requests-remaining``
header. Until a response has reported it, only a free preflight or one small request is
allowed. The Live Odds and closing-lines Edge Functions keep their own 2,000-credit
reserves, so they stop first.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

FLOOR = 0           # core refreshes may spend to zero; the balance resets monthly
# Credits a day the core refreshes use: NFL, NHL and MLB ~450-550 in October 2026, plus
# about 350 for NBA on a full slate (up to 16 games x 11 prop markets, twice a day).
CORE_DAILY = 850
RUN_CAP = 2000
CORE, OPTIONAL = 'core', 'optional'
UNKNOWN_BALANCE_MAX = 50   # the most one request may cost before the balance is known
SPORTS_URL = 'https://api.the-odds-api.com/v4/sports'   # free; reports the balance


class CreditFloorReached(RuntimeError):
    """A paid Odds API request was refused to protect the shared balance."""


def next_reset(now):
    """The provider's monthly reset: midnight UTC on the 1st."""
    now = now.astimezone(timezone.utc)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc)


def core_reserve(now, core_daily=CORE_DAILY):
    """Credits optional jobs must leave for the core refreshes until the reset."""
    return core_daily * (next_reset(now) - now).total_seconds() / 86400


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

    Event lists and the sports list are free; odds cost markets x regions, where each
    group of ten named bookmakers counts as one region; scores
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
    if params.get('bookmakers'):
        # Named bookmakers take priority over regions; each group of ten bills as one region.
        books = len([b for b in str(params['bookmakers']).split(',') if b.strip()]) or 1
        regions = -(-books // 10)
    else:
        regions = len([r for r in str(params.get('regions') or 'us').split(',') if r.strip()]) or 1
    return markets * regions * (10 if historical else 1)


class CreditBudget:
    def __init__(self, floor=None, run_cap=None, label='odds', tier=OPTIONAL, clock=None):
        if tier not in (CORE, OPTIONAL):
            raise ValueError(f'unknown credit tier {tier!r}')
        self.floor = max(FLOOR, floor or 0, _limit('ODDS_CREDIT_FLOOR', FLOOR, max))
        self.core_daily = _limit('ODDS_CORE_DAILY', CORE_DAILY, max)
        self.run_cap = min(RUN_CAP, run_cap or RUN_CAP, _limit('ODDS_RUN_CAP', RUN_CAP, min))
        self.tier = tier
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.remaining = None
        self.spent = 0.0
        self.label = label

    def reserve(self):
        """Credits this job must leave: the floor, plus the core reserve for optional jobs."""
        if self.tier == CORE:
            return self.floor
        return max(self.floor, core_reserve(self.clock(), self.core_daily))

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
        reserve = self.reserve()
        if self.remaining - cost < reserve:
            if self.tier == OPTIONAL and reserve > self.floor:
                raise CreditFloorReached(
                    f'{self.label}: {self.remaining:.0f} credits left; this request could use the '
                    f'{reserve:.0f} credits kept for the NFL, NHL, MLB and NBA refreshes until the monthly reset.')
            raise CreditFloorReached(
                f'{self.label}: {self.remaining:.0f} credits left; this request could take the shared '
                f'balance below the {reserve:.0f}-credit floor.')


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

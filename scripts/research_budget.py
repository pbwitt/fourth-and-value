"""One conservative, durable research allowance per Eastern calendar day.

Callers hold data/analyst/review.lock throughout a run; each ledger mutation is
also flock-protected for standalone use. Pending/unknown requests consume their
full reservation. The cap covers this application's research, not an API account.
"""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from nhl.v2.data import ROOT, iso, stamp
from nhl.v2.astra import MODEL, RATES

CAP = 2.75
PATH = ROOT/'artifacts/analyst/daily-budget.json'
LEGACY = (ROOT/'artifacts/analyst/budget.json', ROOT/'artifacts/nhl/analyst/budget.json')
ET = ZoneInfo('America/New_York')


def day(now):
    return now.astimezone(ET).date().isoformat()


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock outside the committed artifact tree; reservation itself is committed.
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def read(path, legacy=None):
    if legacy is None: legacy=LEGACY if path.resolve()==PATH.resolve() else ()
    ledger = json.loads(path.read_text()) if path.exists() else {'version': 2, 'entries': []}
    for source in legacy:
        if not source.exists():
            continue
        for entry in json.loads(source.read_text())['entries']:
            key = 'legacy:'+str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source)+':'+entry['key']
            if not any(e['key'] == key for e in ledger['entries']):
                ledger['entries'].append(dict(entry, key=key, legacy=True, day=day(stamp(entry['at']))))
    for e in ledger['entries']:
        if type(e.get('charge_usd')) not in (float,int) or not math.isfinite(e['charge_usd']) or e['charge_usd']<0:
            raise ValueError('Invalid budget ledger; spending stopped')
        if stamp(e['at']).tzinfo is None:
            raise ValueError('Budget timestamp lacks timezone')
    return ledger


def save(path, ledger):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(ledger, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def daily_limit(now, config=None):
    config = config or {}
    cap = min(CAP, config.get('daily_budget_usd', CAP))
    override = config.get('test_budget_override') or {}
    if override.get('date') == day(now):
        cap = override.get('limit_usd')
        if not isinstance(override.get('reason'), str) or not override['reason'].strip():
            raise ValueError('Test allowance requires an authorization reason')
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Invalid daily research allowance')
    return cap


def run_cap(now, config):
    cap=daily_limit(now,config)
    # This reserve protects noon-and-later source/price rechecks; it is not a
    # second allowance. All stages charge the same calendar-day ledger.
    if now.astimezone(ET).hour<12:
        cap-=config.get('later_reserve_usd',.75)
    return max(0,cap)


def usage_summary(now, path=PATH, legacy=None, config=None):
    ledger = read(path, legacy)
    used = sum(e['charge_usd'] for e in ledger['entries'] if e.get('day', day(stamp(e['at']))) == day(now))
    cap = daily_limit(now,config)
    return dict(day=day(now), timezone=str(ET), limit_usd=cap, charged_or_reserved_usd=round(used, 6),
                remaining_usd=round(max(0, cap-used), 6))


def reserve(key, now, amount, *, path=PATH, legacy=None, cap=CAP, config=None):
    limit = daily_limit(now,config)
    if not math.isfinite(amount) or amount <= 0 or not 0 < cap <= limit:
        raise ValueError('Invalid research reservation')
    with locked(path):
        ledger = read(path, legacy)
        if ledger.get('halted'):
            return 'budget_halted'
        if any(e['key'] == key for e in ledger['entries']):
            return 'already_attempted'
        used = sum(e['charge_usd'] for e in ledger['entries'] if e.get('day', day(stamp(e['at']))) == day(now))
        if used+amount > cap+1e-9:
            return 'budget_exhausted'
        ledger['entries'].append(dict(key=key, at=iso(now), day=day(now), charge_usd=amount,
            reserved_usd=amount, status='reserved', daily_limit_usd=limit,
            limit_reason=(config or {}).get('test_budget_override',{}).get('reason')
                if (config or {}).get('test_budget_override',{}).get('date')==day(now) else None))
        save(path, ledger)
    return 'reserved'


def settle(key, usage=None, *, path=PATH, search_calls=0, model=MODEL):
    with locked(path):
        ledger = read(path, ())
        entry = next(e for e in ledger['entries'] if e['key'] == key)
        valid = usage and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens'))
        if valid:
            # Overcounts cached input intentionally; conservative billing, no credit assumptions.
            # Same model rates as the reservation, so a normal call cannot look like an overrun.
            rate_in, rate_out = RATES[model]
            charge = round(usage['input_tokens']*rate_in+usage['output_tokens']*rate_out+search_calls*.01, 6)
            entry.update(status='settled', usage=usage, search_calls=search_calls, charge_usd=charge)
            if charge > entry['reserved_usd']+1e-6:
                entry['status'] = 'reservation_overrun_stop'
                ledger['halted'] = True
        else:
            entry['status'] = 'uncertain_reservation_retained'
        save(path, ledger)
    return entry

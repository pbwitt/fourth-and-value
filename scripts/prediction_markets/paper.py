"""Private, locked paper ledger. IOC simulations never submit exchange orders."""
from contextlib import contextmanager
from decimal import Decimal as D
import fcntl
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from .feed import ROOT, atomic_json
from .pricing import decimal, fingerprint, fresh, iso, sweep, timestamp


@contextmanager
def ledger_file(path):
    path = Path(path).resolve()
    if path == ROOT/'docs' or ROOT/'docs' in path.parents:
        raise ValueError('Paper ledgers must be outside the public docs directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path)+'.lock', 'a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else dict(schema_version=1, mode='paper', orders=[])
        if ledger.get('schema_version') != 1 or ledger.get('mode') != 'paper' or not isinstance(ledger.get('orders'), list):
            raise ValueError('Unrecognized ledger; refusing to reset it')
        yield ledger
        atomic_json(path, ledger, private=True)


def run_intent(snapshot, intent, ledger, config, now):
    """One durable ID = one attempt. A changed intent must receive a new ID."""
    order_id = intent.get('id')
    if not isinstance(order_id, str) or not 1 <= len(order_id) <= 120:
        raise ValueError('Each paper intent needs a stable id')
    digest = fingerprint(intent)
    existing = next((o for o in ledger['orders'] if o['id'] == order_id), None)
    if existing:
        if existing['intent_hash'] != digest:
            raise ValueError('Intent ID already used for different instructions')
        return existing
    record = dict(id=order_id, intent_hash=digest, mode='paper', created_at=iso(now),
                  intent=intent, status='rejected', reason=None, snapshot_id=snapshot.get('snapshot_id'),
                  ticker=intent.get('ticker'), side=intent.get('side'), cost_dollars='0')

    def reject(reason):
        record['reason'] = reason
        ledger['orders'].append(record)
        return record

    if intent.get('mode') != 'paper':
        return reject('Only explicit paper intents are accepted')
    max_age = config['paper_snapshot_max_seconds']
    if snapshot.get('schema_version') != 1 or snapshot.get('status') not in ('ready', 'partial') or not snapshot.get('snapshot_id'):
        return reject('Market snapshot unavailable')
    if not fresh(snapshot.get('generated_at'), now, max_age):
        return reject('Snapshot is stale or future-dated')
    row = next((r for r in snapshot['rows'] if r['ticker'] == intent.get('ticker')), None)
    if not row or not fresh(row.get('observed_at'), now, max_age):
        return reject('Contract quote missing, stale or future-dated')
    record['event_ticker'] = row['event_ticker']
    if row.get('status') not in ('active', 'open') or timestamp(row['close_time']) <= now:
        return reject('Contract is not open')
    if intent.get('side') not in ('yes', 'no') or intent.get('rules_hash') != row['rules_hash']:
        return reject('Side or reviewed contract rules do not match')
    # Market close/expiration is never substituted for game start. The operator
    # supplies a verified start and source for this initial pregame simulator.
    try:
        if timestamp(intent['commence_time']) <= now or not intent.get('start_time_source'):
            return reject('Verified future game start required')
        if timestamp(intent['expires_at']) <= now:
            return reject('Intent has expired')
    except (ValueError, KeyError, TypeError, AttributeError):
        return reject('Verified start time and intent expiry required')
    fee = row.get('fee')
    if not fee or not fresh(fee.get('observed_at'), now, max_age):
        return reject('Current fee estimate unavailable')
    if fee.get('valid_until') and timestamp(fee['valid_until']) <= now:
        return reject('Fee schedule changed since observation')
    consumed = {}
    for previous in ledger['orders']:
        if (previous.get('snapshot_id'), previous.get('ticker'), previous.get('side')) != (
                snapshot['snapshot_id'], row['ticker'], intent['side']):
            continue
        for fill in previous.get('simulation', {}).get('fills', []):
            key = fill['price_dollars']
            consumed[key] = str(decimal(consumed.get(key, '0')) + decimal(fill['contracts']))
    try:
        result = sweep(row['asks'][intent['side']], intent['contracts'], fee, intent['limit_price_dollars'], consumed)
        cost = decimal(result['total_estimate_dollars'])
    except (ValueError, KeyError, TypeError):
        return reject('Invalid order quantity, limit or book')
    if not decimal(result['filled_contracts']):
        return reject('No observed liquidity at or below the limit')
    limits = config['paper_limits']
    today = now.astimezone(ZoneInfo('America/New_York')).date()
    filled_orders = [o for o in ledger['orders'] if o.get('simulation')]
    daily = sum((decimal(o['cost_dollars']) for o in filled_orders
                 if timestamp(o['created_at']).astimezone(ZoneInfo('America/New_York')).date() == today), D(0))
    open_orders = [o for o in filled_orders if o.get('settlement') is None]
    exposure = sum((decimal(o['cost_dollars']) for o in open_orders), D(0))
    event = sum((decimal(o['cost_dollars']) for o in open_orders if o['event_ticker'] == row['event_ticker']), D(0))
    for total, limit, label in ((cost, limits['max_order_dollars'], 'order'),
                               (daily+cost, limits['max_daily_dollars'], 'daily'),
                               (exposure+cost, limits['max_open_dollars'], 'open exposure'),
                               (event+cost, limits['max_event_dollars'], 'event exposure')):
        if total > decimal(limit):
            return reject('Paper ' + label + ' limit exceeded')
    # Optional expected payout is supplied explicitly, never inferred from an
    # exchange price, a qualitative review, or a sportsbook's odds.
    forecast = intent.get('forecast')
    record['expected_profit_dollars'] = None
    if forecast is not None:
        try:
            if (forecast['rules_hash'] != row['rules_hash'] or forecast['side'] != intent['side'] or
                    not forecast['source'] or not fresh(forecast['at'], now, max_age) or
                    forecast.get('includes_partial_settlements') is not True):
                return reject('Forecast does not cover the exact current contract and settlement states')
            expected = decimal(forecast['expected_payout_per_contract'])
            if not 0 <= expected <= 1:
                raise ValueError('Invalid forecast')
            record['expected_profit_dollars'] = str(expected*decimal(result['filled_contracts']) - cost)
        except (ValueError, KeyError, TypeError):
            return reject('Invalid explicit payout forecast')
    record.update(status='simulated_partial' if decimal(result['unfilled_contracts']) else 'simulated_filled',
                  reason='Observed-depth IOC simulation; queue, latency and future price movement are not modeled',
                  cost_dollars=str(cost), simulation=result, settlement=None)
    ledger['orders'].append(record)
    return record


def settle(ledger, markets, now):
    """Only explicit finalized per-YES settlement values may grade paper fills."""
    changed = 0
    for order in ledger['orders']:
        if not order.get('simulation') or order.get('settlement') is not None:
            continue
        market = markets.get(order['ticker'], {})
        if market.get('ticker') != order['ticker'] or market.get('status') != 'finalized':
            continue
        try:
            value = decimal(market.get('settlement_value_dollars'))
            if not 0 <= value <= 1:
                continue
        except ValueError:
            continue
        payout = (value if order['side'] == 'yes' else 1-value)*decimal(order['simulation']['filled_contracts'])
        order['settlement'] = dict(observed_at=iso(now), yes_value_dollars=str(value),
                                   payout_dollars=str(payout), profit_dollars=str(payout-decimal(order['cost_dollars'])))
        changed += 1
    return changed

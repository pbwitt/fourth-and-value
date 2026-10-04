"""Dollar/contract math. Preserve decimals, missing fees and partial settlement."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json

D = Decimal
ONE = D('1')


def decimal(value):
    if value is None or isinstance(value, bool):
        raise ValueError('Missing or invalid decimal')
    try:
        result = D(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid decimal') from exc
    if not result.is_finite():
        raise ValueError('Non-finite decimal')
    return result


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return result.astimezone(timezone.utc)


def utcnow():
    return datetime.now(timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def fresh(value, now, seconds):
    try:
        return 0 <= (now - timestamp(value)).total_seconds() <= seconds
    except (ValueError, TypeError, AttributeError):
        return False


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rules_hash(market):
    return fingerprint({k: market.get(k) for k in (
        'ticker', 'event_ticker', 'market_type', 'notional_value_dollars',
        'rules_primary', 'rules_secondary', 'settlement_bounds_type',
        'settlement_floor_dollars', 'strike_type', 'floor_strike', 'cap_strike',
        'custom_strike', 'yes_sub_title', 'no_sub_title')})


def asks(orderbook, side):
    """YES buys consume NO bids at 1-bid; NO buys consume YES bids."""
    if side not in ('yes', 'no'):
        raise ValueError('Side must be yes or no')
    book = orderbook.get('orderbook_fp')
    if not isinstance(book, dict):
        raise ValueError('Fixed-point order book missing')
    bids = book.get('no_dollars' if side == 'yes' else 'yes_dollars')
    if not isinstance(bids, list):
        raise ValueError('Order-book side missing')
    levels = {}
    for level in bids:
        if not isinstance(level, list) or len(level) != 2:
            raise ValueError('Malformed order-book level')
        price, count = map(decimal, level)
        if not 0 < price < 1 or count < 0 or count != count.quantize(D('.01')):
            raise ValueError('Invalid order-book price or quantity')
        if count:
            ask = ONE - price
            if ask in levels:
                raise ValueError('Duplicate order-book price')
            levels[ask] = count
    return [[str(p), str(n)] for p, n in sorted(levels.items())]


def fee_model(series, changes, event_ticker, now):
    """Resolve event overrides at observation time; unknown fees remain unknown."""
    kind, multiplier = series.get('fee_type'), series.get('fee_multiplier')
    relevant = [c for c in changes if c.get('event_ticker') == event_ticker]
    past = sorted((c for c in relevant if timestamp(c['scheduled_ts']) <= now),
                  key=lambda c: timestamp(c['scheduled_ts']))
    if past:
        override = past[-1]
        if override.get('fee_type_override') is not None:
            kind = override['fee_type_override']
        if override.get('fee_multiplier_override') is not None:
            multiplier = override['fee_multiplier_override']
    if kind not in ('quadratic', 'quadratic_with_maker_fees'):
        return None
    try:
        multiplier = decimal(multiplier)
    except ValueError:
        return None
    if multiplier < 0:
        return None
    future = [timestamp(c['scheduled_ts']) for c in relevant if timestamp(c['scheduled_ts']) > now]
    return dict(type=kind, multiplier=str(multiplier), taker_rate=str(D('.07') * multiplier),
                observed_at=iso(now), valid_until=iso(min(future)) if future else None,
                basis='Series fee metadata and event fee overrides; taker estimate; cent rounding per price level')


def sweep(levels, quantity, fee, limit='1', consumed=None):
    """IOC simulation. Never presume liquidity beyond the observed ladder.

    Round each price-level cash debit upward to cents, conservatively ignoring
    the exchange's finer direct-member precision and rounding rebates. This is
    an estimate, not a reproduction of exchange fill accounting.
    """
    quantity, limit = decimal(quantity), decimal(limit)
    if not 0 < quantity <= 100000 or quantity != quantity.quantize(D('.01')):
        raise ValueError('Quantity must be 0.01–100000 contracts, in hundredths')
    if not 0 < limit <= 1:
        raise ValueError('Limit must be greater than zero and at most one dollar')
    remaining, cost, fees = quantity, D(0), D(0)
    fills = []
    rate = decimal(fee['taker_rate']) if fee else None
    if rate is not None and rate < 0:
        raise ValueError('Negative fee rate')
    previous = D(0)
    for value, size in levels:
        price, count = decimal(value), decimal(size)
        if not previous < price < 1 or count < 0:
            raise ValueError('Invalid or unsorted ask ladder')
        previous = price
        if price > limit or remaining <= 0:
            break
        available = max(D(0), count - decimal((consumed or {}).get(str(price), '0')))
        take = min(remaining, available)
        if take <= 0:
            continue
        premium = take * price
        charge = ((premium + rate * take * price * (1-price)).quantize(D('.01'), rounding=ROUND_CEILING)
                  - premium) if rate is not None else None
        fills.append(dict(price_dollars=str(price), contracts=str(take),
                          premium_dollars=str(premium), fee_estimate_dollars=str(charge) if charge is not None else None))
        cost += premium
        if charge is not None:
            fees += charge
        remaining -= take
    filled = quantity - remaining
    total = cost + fees if rate is not None else None
    unit = total / filled if filled and total is not None else None
    equivalent = (100 * (1-unit) / unit if unit <= D('.5') else -100 * unit / (1-unit)) if unit and 0 < unit < 1 else None
    return dict(requested_contracts=str(quantity), filled_contracts=str(filled),
                unfilled_contracts=str(remaining), premium_dollars=str(cost),
                fee_estimate_dollars=str(fees) if total is not None else None,
                total_estimate_dollars=str(total) if total is not None else None,
                average_price_dollars=str(cost/filled) if filled else None,
                cost_per_contract_dollars=str(unit) if unit is not None else None,
                payout_equivalent_american=float(equivalent) if equivalent is not None else None,
                fills=fills)

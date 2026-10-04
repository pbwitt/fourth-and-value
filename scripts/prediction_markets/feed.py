"""Collect exact contract snapshots; only compare explicitly verified outcomes."""
import json
import os
from pathlib import Path
import tempfile

from .client import ticker
from .pricing import asks, decimal, fee_model, fingerprint, fresh, iso, rules_hash, sweep, timestamp, utcnow

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            os.fchmod(handle.fileno(), 0o600 if private else 0o644)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def comparisons(row, mappings, books, now, max_age):
    """No fuzzy matching. A map is invalidated by changed contract rules.

    A verified map alone is insufficient: the sportsbook source must independently
    carry the same verified settlement profile. NFL's current quote feed does not.
    """
    matches = []
    for mapping in mappings:
        if mapping.get('ticker') != row['ticker'] or mapping.get('side') not in ('yes', 'no'):
            continue
        profile = mapping.get('settlement_profile')
        if (mapping.get('rules_hash') != row['rules_hash'] or not profile or
                mapping.get('settlement_verified') is not True or not mapping.get('evidence_url') or
                not fresh(mapping.get('verified_at'), now, 7*86400)):
            continue
        for book in books:
            identity = ('sport', 'game_id', 'player', 'market', 'side', 'line', 'commence_time')
            expected = mapping.get('sportsbook_outcome', {})
            if not all(k in expected and k in book and expected[k] == book[k] for k in identity):
                continue
            if (book.get('settlement_verified') is not True or book.get('settlement_profile') != profile or
                    not fresh(book.get('quoted_at'), now, max_age)):
                continue
            if timestamp(book['commence_time']) <= now:
                continue
            try:
                odds = decimal(book['price'])
            except (ValueError, KeyError):
                continue
            if abs(odds) < 100:
                continue
            dec = 1 + (odds/100 if odds > 0 else 100/abs(odds))
            matches.append(dict(side=mapping['side'], book=book['book'], price=book['price'],
                                quoted_at=book['quoted_at'], outcome=expected,
                                cost_per_dollar_payout=str(1/dec), settlement_profile=profile))
    return matches


def collect(client, config, books=(), clock=utcnow, market_tickers=None):
    started = clock()
    rows, errors, coverage = [], [], []
    event_fees = {}
    max_markets = int(config['max_markets'])
    if not 1 <= max_markets <= 100:
        raise ValueError('max_markets must be 1–100')
    if market_tickers is not None:
        market_tickers = sorted({ticker(t) for t in market_tickers})
        if len(market_tickers) > max_markets:
            raise ValueError('Requested contracts exceed the observation cap')
        if any(not any(t.startswith(s['ticker']+'-') for s in config['series']) for t in market_tickers):
            raise ValueError('Requested contract is outside the configured series')
    for source in config['series']:
        series_id = ticker(source['ticker'])
        targets = [t for t in market_tickers or [] if t.startswith(series_id+'-')]
        if market_tickers is not None and not targets:
            continue
        coverage_row = dict(series=series_id, discovered=0, sampled=0, discovery_complete=False)
        coverage.append(coverage_row)
        try:
            series = client.get('/series/' + series_id)['series']
            series_observed = clock()
            if market_tickers is not None:
                markets, complete = [client.get('/markets/'+t)['market'] for t in targets], True
            else:
                markets, complete = client.pages('/markets', 'markets', series_ticker=series_id,
                                                 status='open', limit=100,
                                                 max_pages=int(config['max_pages_per_series']))
            coverage_row.update(discovered=len(markets), discovery_complete=complete)
            # Close time orders the sampling pool; it is never treated as kickoff.
            markets = sorted(markets, key=lambda m: (m.get('close_time', ''), m.get('ticker', '')))
        except Exception as exc:
            errors.append(dict(scope=series_id, error=type(exc).__name__, detail='Series discovery unavailable'))
            continue
        seen = set()
        for market in markets:
            if len(rows) >= max_markets:
                break
            market_id = market.get('ticker')
            try:
                ticker(market_id)
                if market_id in seen:
                    continue
                seen.add(market_id)
                if not market_id.startswith(series_id+'-'):
                    raise ValueError('Market does not belong to requested series')
                if market.get('market_type') != 'binary' or decimal(market.get('notional_value_dollars')) != 1:
                    continue
                if market.get('status') not in ('active', 'open') or timestamp(market['close_time']) <= clock():
                    continue
                event_id = ticker(market['event_ticker'])
                if event_id not in event_fees:
                    try:
                        changes, fee_complete = client.pages('/events/fee_changes', 'event_fee_changes',
                                                             event_ticker=event_id, limit=100)
                        if not fee_complete:
                            raise ValueError('Incomplete fee history')
                        event_fees[event_id] = changes
                    except Exception as exc:
                        event_fees[event_id] = None
                        errors.append(dict(scope=event_id, error=type(exc).__name__, detail='Fee override history unavailable'))
                raw = client.get('/markets/' + market_id + '/orderbook')
                observed = clock()
                model = fee_model(series, event_fees[event_id], event_id, observed) if event_fees[event_id] is not None else None
                if model:
                    model['observed_at'] = iso(series_observed)
                # Event changes can expire a fee estimate before the normal quote TTL.
                row = dict(venue='kalshi', sport=source['sport'], ticker=market_id, event_ticker=event_id,
                           series_ticker=series_id, title=market.get('title', market_id),
                           yes_label=market.get('yes_sub_title', ''), no_label=market.get('no_sub_title', ''),
                           status=market['status'], close_time=market['close_time'],
                           observed_at=iso(observed), source_updated_at=market.get('updated_time'),
                           commence_time=None, model_probability=None,
                           rules_primary=market.get('rules_primary', ''), rules_secondary=market.get('rules_secondary', ''),
                           rules_hash=rules_hash(market), fee=model,
                           contract_terms_url=series.get('contract_terms_url'),
                           url=('https://kalshi.com/markets/' + series_id.lower() + '/' + source['url_slug'] + '/' + event_id.lower()) if source.get('url_slug') else None,
                           asks={side: asks(raw, side) for side in ('yes', 'no')})
                row['estimates'] = {count: {side: sweep(row['asks'][side], count, model) for side in ('yes', 'no')}
                                    for count in ('1', '10', '100')}
                row['sportsbook_comparisons'] = comparisons(row, config.get('mappings', []), books, observed,
                                                           config['public_snapshot_max_seconds'])
                rows.append(row)
                coverage_row['sampled'] += 1
            except Exception as exc:
                errors.append(dict(scope=str(market_id), error=type(exc).__name__, detail='Contract snapshot unavailable'))
    bounded = any(not c['discovery_complete'] or c['sampled'] < c['discovered'] for c in coverage)
    result = dict(schema_version=1, venue='kalshi', mode='observation', started_at=iso(started),
                  generated_at=iso(clock()), status='partial' if errors and rows else 'error' if errors else 'ready',
                  quote_max_seconds=config['public_snapshot_max_seconds'], display_contracts=config['display_contracts'],
                  coverage=coverage, bounded=bounded, errors=errors, rows=rows)
    result['snapshot_id'] = fingerprint(result)
    return result

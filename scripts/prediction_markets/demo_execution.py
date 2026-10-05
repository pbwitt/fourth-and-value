"""Private demo IOC execution. Reserve durably before sending; reconcile before reuse."""
from contextlib import contextmanager
from datetime import timedelta
from decimal import Decimal as D, ROUND_CEILING
import fcntl
import json
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from .client import ticker
from .demo_client import DemoAPIError
from .feed import ROOT, atomic_json
from .pricing import asks, decimal, fee_model, fingerprint, fresh, iso, rules_hash, sweep, timestamp, utcnow

FINAL = ('executed', 'canceled')


def quantity(value):
    result = decimal(value)
    if not 0 <= result <= 100000 or result != result.quantize(D('.01')):
        raise ValueError('Quantity must be in hundredths of contracts')
    return result


def complete(client, path, key, **params):
    rows, done = client.pages(path, key, limit=100, **params)
    if not done:
        raise ValueError('Incomplete account or fee pagination')
    return rows


class Journal:
    def __init__(self, path, account):
        self.path, self.account = Path(path).expanduser().resolve(), account
        if ROOT == self.path or ROOT in self.path.parents:
            if ROOT/'data'/'prediction-markets' not in self.path.parents:
                raise ValueError('Keep demo journals outside the repository or under data/prediction-markets')

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(str(self.path)+'.lock', 'a') as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists() and self.path.stat().st_mode & 0o077:
                raise ValueError('Demo journal must be owner-only (chmod 600)')
            self.data = (json.loads(self.path.read_text()) if self.path.exists() else
                         dict(schema_version=1, mode='kalshi_demo', account=self.account, orders=[]))
            if (self.data.get('schema_version') != 1 or self.data.get('mode') != 'kalshi_demo' or
                    self.data.get('account') != self.account or not isinstance(self.data.get('orders'), list)):
                raise ValueError('Demo journal/account mismatch; refusing to reset it')
            ids = set()
            for order in self.data['orders']:
                if order['id'] in ids or order['state'] not in ('submitting', 'uncertain', 'open', 'terminal', 'cancel_pending'):
                    raise ValueError('Invalid demo journal')
                ids.add(order['id'])
                if decimal(order['reserved_dollars']) <= 0 or decimal(order['cost_dollars']) < 0:
                    raise ValueError('Invalid demo journal amounts')
            yield self

    def save(self):
        atomic_json(self.path, self.data, private=True)
        # The rename and directory entry must survive a crash before the POST.
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class DemoExecutor:
    def __init__(self, client, journal_path, config, clock=utcnow):
        if config.get('schema_version') != 1 or not isinstance(config.get('enabled'), bool):
            raise ValueError('Invalid demo configuration')
        if not isinstance(config.get('allowed_series'), list) or not config['allowed_series']:
            raise ValueError('Demo series allowlist required')
        for series in config['allowed_series']:
            ticker(series)
        if not 0 < decimal(config['quote_max_seconds']) <= 30:
            raise ValueError('Demo quotes must expire within 30 seconds')
        for key in ('max_order_dollars', 'max_daily_dollars', 'max_event_dollars', 'max_open_dollars'):
            if decimal(config[key]) <= 0:
                raise ValueError('Positive demo limits required')
        self.client, self.config, self.clock = client, config, clock
        self.journal = Journal(journal_path, client.account_fingerprint)

    def inspect(self, market_id):
        market_id = ticker(market_id)
        series_id = next((s for s in self.config['allowed_series'] if market_id.startswith(s+'-')), None)
        if not series_id:
            raise ValueError('Contract outside the demo series allowlist')
        observed = self.clock()
        market = self.client.get('/markets/'+market_id)['market']
        if (market.get('ticker') != market_id or market.get('market_type') != 'binary' or
                decimal(market.get('notional_value_dollars')) != 1 or not market.get('rules_primary')):
            raise ValueError('Exact $1 binary contract rules required')
        event = ticker(market['event_ticker'])
        if not event.startswith(series_id+'-'):
            raise ValueError('Unexpected event identity')
        series = self.client.get('/series/'+series_id)['series']
        if series.get('ticker') != series_id:
            raise ValueError('Unexpected series identity')
        changes = complete(self.client, '/events/fee_changes', 'event_fee_changes', event_ticker=event)
        fee = fee_model(series, changes, event, self.clock())
        book = self.client.get('/markets/'+market_id+'/orderbook')
        return dict(mode='demo', observed_at=iso(observed), market=market, rules_hash=rules_hash(market),
                    fee=fee, asks={s: asks(book, s) for s in ('yes', 'no')})

    def _validate(self, intent, quote, now):
        market, fee = quote['market'], quote['fee']
        if (intent.get('mode') != 'demo' or intent.get('side') not in ('yes', 'no') or
                intent.get('rules_hash') != quote['rules_hash']):
            raise ValueError('Explicit demo mode, side and reviewed rules hash required')
        if (not fresh(quote['observed_at'], now, self.config['quote_max_seconds']) or
                market.get('status') not in ('active', 'open') or timestamp(market['close_time']) <= now):
            raise ValueError('Contract or quote is no longer current')
        if (timestamp(intent['commence_time']) <= now or not intent.get('start_time_source') or
                timestamp(intent['expires_at']) <= now):
            raise ValueError('Verified future game start and unexpired intent required')
        if (not fee or not fresh(fee['observed_at'], now, self.config['quote_max_seconds']) or
                (fee.get('valid_until') and timestamp(fee['valid_until']) <= now+timedelta(seconds=30))):
            raise ValueError('Known, current fees required without an imminent schedule change')
        count, limit = quantity(intent['contracts']), decimal(intent['limit_price_dollars'])
        if count <= 0 or not 0 < limit < 1 or limit != limit.quantize(D('.0001')):
            raise ValueError('Positive quantity and a four-decimal limit strictly between $0 and $1 required')
        yes_price = limit if intent['side'] == 'yes' else 1-limit
        valid_tick = False
        for band in market.get('price_ranges', []):
            start, end, step = (decimal(band[k]) for k in ('start', 'end', 'step'))
            if step > 0 and start <= yes_price <= end and (yes_price-start) % step == 0:
                valid_tick = True
        if not valid_tick:
            raise ValueError('Limit is outside the published YES price ticks')
        estimate = sweep(quote['asks'][intent['side']], count, fee, limit)
        if decimal(estimate['filled_contracts']) <= 0:
            raise ValueError('No observed liquidity at the limit')
        # Reserve the entire requested size: the book may deepen before the IOC.
        # Worst quadratic fee at p=.5, plus <1 cent rounding per .01-contract
        # fill. This intentionally generous demo bound also covers debit rounding.
        reserve = (count*limit + count*decimal(fee['taker_rate'])/4 + count).quantize(D('.01'), rounding=ROUND_CEILING)
        return count, yes_price, reserve, estimate

    def _limits(self, orders, reserve, event, now):
        today = now.astimezone(ZoneInfo('America/New_York')).date()
        def held(order):
            return decimal(order['cost_dollars'] if order['state'] == 'terminal' else order['reserved_dollars'])
        daily = sum((held(o) for o in orders if timestamp(o['created_at']).astimezone(ZoneInfo('America/New_York')).date() == today), D(0))
        opened = [o for o in orders if o.get('settlement') is None]
        total = sum((held(o) for o in opened), D(0))
        event_total = sum((held(o) for o in opened if o['event_ticker'] == event), D(0))
        for amount, key in ((reserve, 'max_order_dollars'), (daily+reserve, 'max_daily_dollars'),
                            (total+reserve, 'max_open_dollars'), (event_total+reserve, 'max_event_dollars')):
            if amount > decimal(self.config[key]):
                raise ValueError('Demo '+key+' exceeded')

    def submit(self, intent):
        order_id = intent.get('id')
        if not isinstance(order_id, str) or not 1 <= len(order_id) <= 120:
            raise ValueError('Stable intent ID required')
        digest = fingerprint(intent)
        with self.journal.locked() as journal:
            orders = journal.data['orders']
            existing = next((o for o in orders if o['id'] == order_id), None)
            if existing:
                if existing['intent_hash'] != digest:
                    raise ValueError('Intent ID already used for different instructions')
                return existing
            if not self.config['enabled']:
                raise ValueError('Demo submissions disabled in config')
            if any(o['state'] != 'terminal' for o in orders):
                raise ValueError('Reconcile or cancel the outstanding demo order before another submission')
            quote = self.inspect(intent['ticker'])
            now = self.clock()
            count, yes_price, reserve, estimate = self._validate(intent, quote, now)
            self._limits(orders, reserve, quote['market']['event_ticker'], now)
            balance = self.client.get('/portfolio/balance', subaccount=0,
                                      exchange_index=quote['market'].get('exchange_index'))
            cash = balance.get('balance_dollars')
            if cash is None:
                cents = balance.get('balance')
                if type(cents) is not int:
                    raise ValueError('Demo balance unavailable')
                cash = D(cents)/100
            if decimal(cash) < reserve:
                raise ValueError('Insufficient available demo balance')
            # Balance/network latency may have expired the observed book or intent.
            now = self.clock()
            self._validate(intent, quote, now)
            self._limits(orders, reserve, quote['market']['event_ticker'], now)
            client_id = str(uuid.uuid5(uuid.NAMESPACE_URL, 'fv-demo:'+self.client.account_fingerprint+':'+order_id))
            payload = dict(ticker=intent['ticker'], client_order_id=client_id,
                           side='bid' if intent['side'] == 'yes' else 'ask', count=f'{count:.2f}',
                           price=f'{yes_price:.4f}', time_in_force='immediate_or_cancel',
                           self_trade_prevention_type='taker_at_cross', cancel_order_on_pause=True, subaccount=0)
            if quote['market'].get('exchange_index') is not None:
                payload['exchange_index'] = quote['market']['exchange_index']
            record = dict(id=order_id, intent_hash=digest, intent=intent, client_order_id=client_id,
                          order_id=None, state='submitting', created_at=iso(now),
                          ticker=intent['ticker'], event_ticker=quote['market']['event_ticker'], side=intent['side'],
                          reserved_dollars=str(reserve), cost_dollars='0', filled_contracts='0', fills=[],
                          settlement=None, payload=payload, estimate=estimate, quote=quote)
            orders.append(record)
            journal.save()  # Never send if this fails. A crash after this retains the reservation.
            try:
                send_time = self.clock()
                self._validate(intent, quote, send_time)
                if send_time.astimezone(ZoneInfo('America/New_York')).date() != now.astimezone(ZoneInfo('America/New_York')).date():
                    raise ValueError('Day changed while reserving; no order sent')
                result = self.client.create_order(payload)
                if result.get('client_order_id') != client_id or not isinstance(result.get('order_id'), str) or not result['order_id']:
                    raise ValueError('Unrecognized create acknowledgement')
                record['order_id'] = result['order_id']
                record['state'] = 'uncertain'
                journal.save()
                self._reconcile(record)
            except (DemoAPIError, ValueError, KeyError, TypeError, AttributeError):
                record.update(state='uncertain', issue='Submission or reconciliation incomplete; reservation retained; no automatic retry')
            journal.save()
            return record

    def _reconcile(self, record):
        if not record['order_id']:
            candidates = complete(self.client, '/portfolio/orders', 'orders', ticker=record['ticker'], subaccount=0)
            matches = [o for o in candidates if o.get('client_order_id') == record['client_order_id']]
            if len(matches) != 1 or not matches[0].get('order_id'):
                raise ValueError('Order absent or ambiguous; keep its reservation and do not resend')
            record['order_id'] = matches[0]['order_id']
        order = self.client.get('/portfolio/orders/'+record['order_id'])['order']
        self._identity(order, record, client_id=True)
        count = quantity(order['fill_count_fp'])
        remaining = quantity(order['remaining_count_fp'])
        requested = quantity(record['intent']['contracts'])
        if quantity(order['initial_count_fp']) != requested or count > requested or count+remaining > requested:
            raise ValueError('Order quantities disagree')
        status = order.get('status')
        if status not in (*FINAL, 'resting') or (status == 'executed' and count != requested):
            raise ValueError('Unrecognized order lifecycle state')
        raw = complete(self.client, '/portfolio/fills', 'fills', order_id=record['order_id'], ticker=record['ticker'], subaccount=0)
        fills = {}
        for fill in raw:
            self._identity(fill, record)
            fill_id = fill.get('fill_id')
            if not isinstance(fill_id, str) or not fill_id:
                raise ValueError('Fill identity missing')
            size, price, fee = quantity(fill['count_fp']), decimal(fill[record['side']+'_price_dollars']), decimal(fill['fee_cost'])
            if (size <= 0 or not 0 < price < 1 or price > decimal(record['intent']['limit_price_dollars']) or
                    fee < 0 or decimal(fill['yes_price_dollars'])+decimal(fill['no_price_dollars']) != 1):
                raise ValueError('Invalid fill price, size or fee')
            normalized = dict(fill_id=fill_id, contracts=str(size.normalize()),
                              price_dollars=str(price.normalize()), fee_dollars=str(fee.normalize()))
            if fill_id in fills and fills[fill_id] != normalized:
                raise ValueError('Conflicting duplicate fill')
            fills[fill_id] = normalized
        if sum((decimal(f['contracts']) for f in fills.values()), D(0)) != count:
            raise ValueError('Order and fill counts disagree; reconciliation may be lagging')
        if any(fills.get(f['fill_id']) != f for f in record['fills']):
            raise ValueError('Previously observed fills changed or disappeared')
        cost = sum((decimal(f['contracts'])*decimal(f['price_dollars'])+decimal(f['fee_dollars']) for f in fills.values()), D(0))
        if cost > decimal(record['reserved_dollars']):
            raise ValueError('Actual cost exceeded demo reservation; manual review required')
        record.update(state='terminal' if status in FINAL else 'open', exchange_status=status,
                      cost_dollars=str(cost), filled_contracts=str(count), fills=sorted(fills.values(), key=lambda f:f['fill_id']),
                      reconciled_at=iso(self.clock()), issue=None)

    @staticmethod
    def _identity(value, record, client_id=False):
        if (value.get('order_id') != record['order_id'] or value.get('ticker') != record['ticker'] or
                type(value.get('subaccount_number')) is not int or value['subaccount_number'] != 0 or
                value.get('outcome_side') != record['side'] or
                (record['payload'].get('exchange_index') is not None and
                 value.get('exchange_index') != record['payload']['exchange_index']) or
                value.get('book_side') != ('bid' if record['side'] == 'yes' else 'ask') or
                (client_id and value.get('client_order_id') != record['client_order_id'])):
            raise ValueError('Order/fill account, contract, ID or direction mismatch')

    def reconcile(self):
        with self.journal.locked() as journal:
            for record in journal.data['orders']:
                if record.get('settlement') is not None:
                    continue
                try:
                    self._reconcile(record)
                    if record['state'] == 'terminal':
                        market = self.client.get('/markets/'+record['ticker'])['market']
                        if market.get('ticker') != record['ticker']:
                            raise ValueError('Settlement market identity mismatch')
                        if market.get('status') == 'finalized' and market.get('settlement_value_dollars') is not None:
                            value = decimal(market['settlement_value_dollars'])
                            if not 0 <= value <= 1:
                                raise ValueError('Invalid settlement value')
                            payout = (value if record['side'] == 'yes' else 1-value)*decimal(record['filled_contracts'])
                            record['settlement'] = dict(observed_at=iso(self.clock()), yes_value_dollars=str(value),
                                                       payout_dollars=str(payout), profit_dollars=str(payout-decimal(record['cost_dollars'])))
                except (DemoAPIError, ValueError, KeyError, TypeError, AttributeError):
                    record.update(state='uncertain', issue='Reconciliation incomplete; reservation retained; no automatic retry')
                journal.save()
            return journal.data['orders']

    def cancel(self, intent_id):
        with self.journal.locked() as journal:
            record = next((o for o in journal.data['orders'] if o['id'] == intent_id), None)
            if not record or not record['order_id']:
                raise ValueError('Reconcile this journal-owned order ID before cancellation')
            if record['state'] == 'terminal':
                return record
            record['state'] = 'cancel_pending'
            journal.save()
            try:
                self.client.cancel_order(record['order_id'], record['ticker'])
                self._reconcile(record)  # Fills may race cancellation; never zero the cost from its ACK.
            except (DemoAPIError, ValueError, KeyError, TypeError, AttributeError):
                record.update(state='uncertain', issue='Cancellation unconfirmed; reconcile before continuing')
            journal.save()
            return record

    def status(self):
        with self.journal.locked() as journal:
            return journal.data['orders']


def summary(record):
    return {k: record.get(k) for k in ('id', 'order_id', 'state', 'exchange_status', 'ticker', 'side',
                                      'reserved_dollars', 'cost_dollars', 'filled_contracts', 'settlement', 'issue')}

"""Fixture-only demo execution tests. Never use exchange credentials or networking."""
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prediction_markets.demo_client import DemoClient, DemoAPIError, NoRedirect, signature
from prediction_markets.demo_execution import DemoExecutor, Journal
from prediction_markets.feed import ROOT
from prediction_markets.pricing import iso, rules_hash

NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
TICKER = 'KXNFLGAME-26OCT05AB-A'
MARKET = dict(ticker=TICKER, event_ticker='KXNFLGAME-26OCT05AB', market_type='binary',
              notional_value_dollars='1.0000', status='active', rules_primary='Exact fixture rules: half settlement on tie.',
              close_time=iso(NOW+timedelta(days=2)), price_ranges=[dict(start='0', end='1', step='0.01')])
CONFIG = json.loads((ROOT/'config/kalshi_demo.json').read_text()) | {'enabled': True}


def intent(**changes):
    return dict(id='demo-test-1', mode='demo', ticker=TICKER, side='yes', contracts='1.00',
                limit_price_dollars='0.50', rules_hash=rules_hash(MARKET),
                commence_time=iso(NOW+timedelta(days=1)), start_time_source='https://example.test/schedule',
                expires_at=iso(NOW+timedelta(minutes=5))) | changes


class FixtureClient:
    account_fingerprint = 'test-account'

    def __init__(self):
        self.market = deepcopy(MARKET)
        self.series = dict(ticker='KXNFLGAME', fee_type='quadratic', fee_multiplier='1')
        self.book = {'orderbook_fp': {'yes_dollars': [['0.60', '10.00']], 'no_dollars': [['0.50', '10.00']]}}
        self.balance = {'balance_dollars': '100.00'}
        self.changes, self.orders, self.fills, self.sent = [], {}, {}, []
        self.after_create = lambda: None
        self.after_balance = lambda: None
        self.cancel_hook = lambda: None
        self.pages_complete = True
        self.order_visible = True
        self.before_post = lambda: None

    def get(self, path, **params):
        if path == '/markets/'+TICKER:
            return {'market': deepcopy(self.market)}
        if path.startswith('/series/'):
            return {'series': deepcopy(self.series)}
        if path.endswith('/orderbook'):
            return deepcopy(self.book)
        if path == '/portfolio/balance':
            self.after_balance()
            return deepcopy(self.balance)
        if path.startswith('/portfolio/orders/'):
            return {'order': deepcopy(self.orders[path.rsplit('/', 1)[1]])}
        raise AssertionError('Unexpected path '+path)

    def pages(self, path, key, **params):
        if key == 'event_fee_changes':
            return deepcopy(self.changes), True
        if key == 'orders':
            return deepcopy(list(self.orders.values())) if self.order_visible else [], self.pages_complete
        if key == 'fills':
            return deepcopy(self.fills[params['order_id']]), self.pages_complete
        raise AssertionError('Unexpected list '+path)

    def create_order(self, payload):
        self.before_post()
        self.sent.append(deepcopy(payload))
        order_id = 'exchange-'+str(len(self.sent))
        side = 'yes' if payload['side'] == 'bid' else 'no'
        yes = payload['price']
        self.orders[order_id] = dict(order_id=order_id, client_order_id=payload['client_order_id'],
                                     ticker=payload['ticker'], subaccount_number=0, outcome_side=side, book_side=payload['side'],
                                     initial_count_fp=payload['count'], fill_count_fp=payload['count'], remaining_count_fp='0',
                                     status='executed')
        self.fills[order_id] = [dict(order_id=order_id, fill_id='fill-'+order_id, ticker=payload['ticker'],
                                    subaccount_number=0, outcome_side=side, book_side=payload['side'],
                                    count_fp=payload['count'], yes_price_dollars=yes, no_price_dollars=str(1-D(yes)), fee_cost='0.02')]
        self.after_create()
        return {'order_id': order_id, 'client_order_id': payload['client_order_id'], 'fill_count': payload['count'], 'remaining_count': '0'}

    def cancel_order(self, order_id, market_ticker):
        self.cancel_hook()
        self.orders[order_id].update(status='canceled', remaining_count_fp='0')
        return {'order_id': order_id, 'reduced_by': '0.50'}


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'journal.json'
        self.client = FixtureClient()
        self.now = NOW
        self.config = deepcopy(CONFIG)
        self.engine = DemoExecutor(self.client, self.path, self.config, clock=lambda: self.now)

    def disk(self):
        return json.loads(self.path.read_text())['orders']

    def uncertain(self):
        def timeout():
            raise DemoAPIError()
        self.client.after_create = timeout
        return self.engine.submit(intent())

    def test_yes_ioc_actual_fees_and_private_durable_reservation(self):
        self.client.before_post = lambda: self.assertEqual(self.disk()[0]['state'], 'submitting')
        row = self.engine.submit(intent())
        self.assertEqual((row['state'], D(row['cost_dollars'])), ('terminal', D('.52')))
        self.assertGreater(D(row['reserved_dollars']), D(row['cost_dollars']))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.client.sent[0]['time_in_force'], 'immediate_or_cancel')
        self.assertEqual(self.client.sent[0]['price'], '0.5000')
        self.assertNotIn('expiration_time', self.client.sent[0])

    def test_no_uses_yes_ask_price_and_no_purchase_cost(self):
        row = self.engine.submit(intent(side='no', limit_price_dollars='0.40'))
        self.assertEqual((self.client.sent[0]['side'], self.client.sent[0]['price']), ('ask', '0.6000'))
        self.assertEqual(D(row['cost_dollars']), D('.42'))

    def test_repeat_id_returns_outcome_and_changed_id_contents_fail(self):
        self.engine.submit(intent())
        self.engine.submit(intent())
        with self.assertRaisesRegex(ValueError, 'already used'):
            self.engine.submit(intent(contracts='0.5'))
        self.assertEqual(len(self.client.sent), 1)

    def test_concurrent_repeat_posts_once(self):
        def run(_):
            engine = DemoExecutor(self.client, self.path, self.config, clock=lambda: NOW)
            return engine.submit(intent())
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, range(2)))
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(results[0]['order_id'], results[1]['order_id'])

    def test_timeout_after_acceptance_reconciles_without_second_post(self):
        self.assertEqual(self.uncertain()['state'], 'uncertain')
        self.assertEqual(self.engine.submit(intent())['state'], 'uncertain')
        with self.assertRaisesRegex(ValueError, 'outstanding'):
            self.engine.submit(intent(id='next'))
        row = self.engine.reconcile()[0]
        self.assertEqual((row['state'], D(row['cost_dollars'])), ('terminal', D('.52')))
        self.assertEqual(len(self.client.sent), 1)

    def test_missing_or_duplicate_remote_order_never_releases_reservation(self):
        self.uncertain()
        self.client.order_visible = False
        row = self.engine.reconcile()[0]
        self.assertEqual(row['state'], 'uncertain')
        self.client.order_visible = True
        self.client.orders['duplicate'] = deepcopy(self.client.orders['exchange-1'])
        self.client.orders['duplicate']['order_id'] = 'duplicate'
        self.assertEqual(self.engine.reconcile()[0]['state'], 'uncertain')
        self.assertEqual(len(self.client.sent), 1)

    def test_crash_during_request_survives_restart(self):
        self.client.after_create = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.engine.submit(intent())
        self.assertEqual(self.disk()[0]['state'], 'submitting')
        restarted = DemoExecutor(self.client, self.path, self.config, clock=lambda: NOW)
        self.assertEqual(restarted.reconcile()[0]['state'], 'terminal')
        self.assertEqual(len(self.client.sent), 1)

    def test_failed_durable_reservation_sends_nothing(self):
        with patch.object(Journal, 'save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.engine.submit(intent())
        self.assertFalse(self.client.sent)

    def test_incomplete_pagination_keeps_reservation(self):
        self.uncertain()
        self.client.pages_complete = False
        self.assertEqual(self.engine.reconcile()[0]['state'], 'uncertain')

    def test_partial_ioc_records_actual_cost(self):
        def partial():
            self.client.orders['exchange-1'].update(fill_count_fp='0.5', status='canceled')
            self.client.fills['exchange-1'][0]['count_fp'] = '0.5'
        self.client.after_create = partial
        row = self.engine.submit(intent())
        self.assertEqual((row['state'], D(row['cost_dollars']), D(row['filled_contracts'])), ('terminal', D('.27'), D('.5')))

    def test_cancellation_accounts_for_racing_fill_even_when_disabled(self):
        def resting():
            self.client.orders['exchange-1'].update(fill_count_fp='0.5', remaining_count_fp='0.5', status='resting')
            self.client.fills['exchange-1'][0]['count_fp'] = '0.5'
        self.client.after_create = resting
        self.assertEqual(self.engine.submit(intent())['state'], 'open')
        self.config['enabled'] = False
        # A prior fill cannot change: a racing fill must have its own identity.
        def distinct_race():
            self.client.orders['exchange-1']['fill_count_fp'] = '0.75'
            extra = self.client.fills['exchange-1'][0] | {'fill_id':'later-fill', 'count_fp':'0.25', 'fee_cost':'0.01'}
            self.client.fills['exchange-1'].append(extra)
        self.client.cancel_hook = distinct_race
        row = self.engine.cancel('demo-test-1')
        self.assertEqual((row['state'], D(row['cost_dollars'])), ('terminal', D('.405')))
        self.assertEqual(len(row['fills']), 2)

    def test_failed_cancel_keeps_hold(self):
        self.uncertain()
        self.client.orders['exchange-1'].update(status='resting')
        self.engine.reconcile()
        self.client.cancel_hook = lambda: (_ for _ in ()).throw(DemoAPIError())
        self.assertEqual(self.engine.cancel('demo-test-1')['state'], 'uncertain')

    def test_duplicate_fill_deduplicated_but_conflicts_block(self):
        self.engine.submit(intent())
        fill = self.client.fills['exchange-1'][0]
        self.client.fills['exchange-1'].append(deepcopy(fill))
        self.assertEqual(len(self.engine.reconcile()[0]['fills']), 1)
        self.client.fills['exchange-1'][1]['fee_cost'] = '0.03'
        self.assertEqual(self.engine.reconcile()[0]['state'], 'uncertain')

    def test_missing_fees_or_direction_account_id_mismatch_block(self):
        for key, value in [('fee_cost', None), ('outcome_side','no'), ('subaccount_number',1),
                           ('order_id','other'), ('count_fp','0.5'), ('yes_price_dollars','0.6')]:
            with self.subTest(key=key):
                self.client.after_create = lambda k=key,v=value: self.client.fills['exchange-1'][0].update({k:v})
                row = self.engine.submit(intent())
                self.assertEqual(row['state'], 'uncertain')
                self.path.unlink()
                self.client = FixtureClient()
                self.engine.client = self.client

    def test_lagging_fills_never_replace_previously_confirmed_cost(self):
        self.engine.submit(intent())
        self.client.fills['exchange-1'] = []
        self.assertEqual(self.engine.reconcile()[0]['state'], 'uncertain')
        self.assertEqual(D(self.disk()[0]['cost_dollars']), D('.52'))

    def test_full_requested_size_reserved_even_with_thin_book(self):
        self.client.book['orderbook_fp']['no_dollars'][0][1] = '0.01'
        with self.assertRaisesRegex(ValueError, 'max_order_dollars'):
            self.engine.submit(intent(contracts='2'))
        self.assertFalse(self.client.sent)

    def test_balance_latency_expires_quote_before_post(self):
        self.client.after_balance = lambda: setattr(self, 'now', NOW+timedelta(seconds=31))
        with self.assertRaisesRegex(ValueError, 'current'):
            self.engine.submit(intent())
        self.assertFalse(self.client.sent)

    def test_invalid_or_expired_intents_never_post(self):
        for changes in [dict(mode='live'), dict(side='bid'), dict(rules_hash='changed'), dict(contracts='0'),
                        dict(contracts='0.001'), dict(limit_price_dollars='0.501'), dict(contracts='NaN'),
                        dict(commence_time=iso(NOW)), dict(expires_at=iso(NOW)), dict(start_time_source='')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.engine.submit(intent(**changes))
        self.assertFalse(self.client.sent)

    def test_disabled_unknown_fees_no_ticks_and_low_balance_block(self):
        self.config['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'disabled'):
            self.engine.submit(intent())
        self.config['enabled'] = True
        self.client.series['fee_type'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'fees'):
            self.engine.submit(intent())
        self.client.series['fee_type'] = 'quadratic'
        self.client.market['price_ranges'] = []
        with self.assertRaisesRegex(ValueError, 'ticks'):
            self.engine.submit(intent())
        self.client.market = deepcopy(MARKET)
        self.client.balance = {'balance': 1}
        with self.assertRaisesRegex(ValueError, 'balance'):
            self.engine.submit(intent())
        self.assertFalse(self.client.sent)

    def test_upcoming_fee_change_blocks(self):
        self.client.changes = [dict(event_ticker=MARKET['event_ticker'], scheduled_ts=iso(NOW+timedelta(seconds=20)), fee_multiplier_override='2')]
        with self.assertRaisesRegex(ValueError, 'fees'):
            self.engine.submit(intent())

    def test_settlement_half_payout_and_daily_gross_cap(self):
        self.engine.submit(intent())
        self.client.market.update(status='finalized', settlement_value_dollars='0.5')
        row = self.engine.reconcile()[0]
        self.assertEqual(D(row['settlement']['profit_dollars']), D('-.02'))
        self.config['max_daily_dollars'] = '2'
        self.client.market = deepcopy(MARKET)
        with self.assertRaisesRegex(ValueError, 'max_daily_dollars'):
            self.engine.submit(intent(id='second'))
        self.now = NOW+timedelta(days=1)
        later = intent(id='second', commence_time=iso(NOW+timedelta(days=2)), expires_at=iso(self.now+timedelta(minutes=5)))
        self.assertEqual(self.engine.submit(later)['state'], 'terminal')

    def test_event_and_open_caps(self):
        self.engine.submit(intent())
        for name in ('max_event_dollars', 'max_open_dollars'):
            self.config[name] = '2'
            with self.assertRaisesRegex(ValueError, name):
                self.engine.submit(intent(id='second'))
            self.config[name] = '10'

    def test_missing_settlement_is_not_inferred(self):
        self.engine.submit(intent())
        self.client.market.update(status='finalized', result='yes')
        self.assertIsNone(self.engine.reconcile()[0]['settlement'])

    def test_account_or_corrupt_journal_never_reset(self):
        self.engine.submit(intent())
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            with Journal(self.path, 'other-account').locked():
                pass
        self.path.write_text('{')
        with self.assertRaises(ValueError):
            self.engine.submit(intent())
        self.assertEqual(self.path.read_text(), '{')

    def test_public_journal_rejected(self):
        with self.assertRaises(ValueError):
            Journal(ROOT/'docs'/'private.json', 'account')


class SigningTests(unittest.TestCase):
    def test_rsa_and_ed25519_sign_exact_path_without_query(self):
        message = b'123GET/trade-api/v2/portfolio/orders'
        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        sig = signature(rsa_key, '123', 'GET', '/trade-api/v2/portfolio/orders?cursor=secret')
        rsa_key.public_key().verify(base64.b64decode(sig), message,
                                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
        key = ed25519.Ed25519PrivateKey.generate()
        key.public_key().verify(base64.b64decode(signature(key, '123', 'GET', '/trade-api/v2/portfolio/orders')), message)

    def test_host_signed_headers_timeout_and_endpoint_allowlist(self):
        key = ed25519.Ed25519PrivateKey.generate()
        seen = []
        class Opener:
            def open(self, request, timeout):
                seen.append((request, timeout))
                return io.BytesIO(b'{}')
        client = DemoClient('demo-key', key, opener=Opener(), clock=lambda: 1)
        client.get('/portfolio/orders', cursor='next')
        req, timeout = seen[0]
        self.assertEqual(req.full_url, 'https://external-api.demo.kalshi.co/trade-api/v2/portfolio/orders?cursor=next')
        self.assertEqual(timeout, 15)
        headers = {k.lower():v for k,v in req.header_items()}
        key.public_key().verify(base64.b64decode(headers['kalshi-access-signature']), b'1000GET/trade-api/v2/portfolio/orders')
        for method, path in [('POST','/portfolio/orders'), ('POST','/portfolio/transfers'),
                             ('GET','https://external-api.kalshi.com/portfolio/orders'), ('DELETE','/portfolio/events/orders/../x')]:
            with self.assertRaises(ValueError):
                client.request(method, path)
        self.assertEqual(len(seen), 1)

    def test_no_redirect_no_retry_and_sanitized_errors(self):
        self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://external-api.kalshi.com'))
        class Opener:
            count = 0
            def open(self, request, timeout):
                self.count += 1
                raise HTTPError(request.full_url, 403, 'SECRET', {}, io.BytesIO(b'SECRET'))
        opener = Opener()
        client = DemoClient('demo-key', ed25519.Ed25519PrivateKey.generate(), opener=opener)
        with self.assertRaises(DemoAPIError) as error:
            client.get('/portfolio/balance')
        self.assertNotIn('SECRET', str(error.exception))
        self.assertEqual(opener.count, 1)

    def test_pagination_repeated_cursor_rejected(self):
        client = DemoClient('demo-key', ed25519.Ed25519PrivateKey.generate())
        with patch.object(client, 'get', return_value={'orders':[], 'cursor':'repeated'}):
            with self.assertRaisesRegex(ValueError, 'Repeated'):
                client.pages('/portfolio/orders', 'orders')

    def test_only_demo_environment_and_private_key_permissions(self):
        with patch.dict(os.environ, {'KALSHI_API_KEY_ID':'live'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'KALSHI_DEMO'):
                DemoClient.from_environment()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'key.pem'
            key = ed25519.Ed25519PrivateKey.generate()
            path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
            env = {'KALSHI_DEMO_API_KEY_ID':'demo', 'KALSHI_DEMO_PRIVATE_KEY_PATH':str(path)}
            with patch.dict(os.environ, env, clear=True):
                path.chmod(0o644)
                with self.assertRaisesRegex(ValueError, 'owner-only'):
                    DemoClient.from_environment()
                path.chmod(0o600)
                self.assertTrue(DemoClient.from_environment().account_fingerprint)


if __name__ == '__main__':
    unittest.main()

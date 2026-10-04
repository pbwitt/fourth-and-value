from copy import deepcopy
from datetime import timedelta
from decimal import Decimal as D
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prediction_markets.client import PublicClient, ticker
from prediction_markets.feed import collect, comparisons
from prediction_markets.paper import ledger_file, run_intent, settle
from prediction_markets.pricing import asks, decimal, fee_model, fresh, iso, rules_hash, sweep, timestamp

NOW = timestamp('2026-10-04T16:00:00Z')
CONFIG = json.loads((Path(__file__).resolve().parents[1]/'config/prediction_markets.json').read_text())
SERIES = dict(fee_type='quadratic_with_maker_fees', fee_multiplier=1,
              contract_terms_url='https://assets.kalshi.com/contract_terms/FOOTBALLGAMEWIN.pdf')
MARKET = dict(ticker='KXNFLGAME-TEST-BUF', event_ticker='KXNFLGAME-TEST', market_type='binary',
              notional_value_dollars='1.0000', status='active', close_time='2026-10-05T20:00:00Z',
              title='Buffalo wins', yes_sub_title='Buffalo', no_sub_title='Buffalo',
              rules_primary='If Buffalo wins, resolves Yes.', rules_secondary='Ties settle at $0.50.',
              updated_time='2026-09-29T10:00:00Z')
BOOK = dict(orderbook_fp=dict(no_dollars=[['0.4000', '100.00'], ['0.4500', '2.50']],
                             yes_dollars=[['0.5300', '100.00']]))


class FakeClient:
    def __init__(self, markets=None, fail_fees=False, fail_book=False, complete=True):
        self.markets = [deepcopy(MARKET)] if markets is None else markets
        self.fail_fees, self.fail_book, self.complete = fail_fees, fail_book, complete

    def get(self, path, **params):
        if path.startswith('/series/'):
            return dict(series=deepcopy(SERIES))
        if path.endswith('/orderbook'):
            if self.fail_book:
                raise OSError('test book unavailable')
            return deepcopy(BOOK)
        if path.startswith('/markets/'):
            return dict(market=deepcopy(MARKET))
        raise AssertionError(path)

    def pages(self, path, key, **params):
        if path == '/markets':
            return self.markets, self.complete
        if path == '/events/fee_changes':
            if self.fail_fees:
                raise OSError('test fees unavailable')
            return [], True
        raise AssertionError(path)


def fixture():
    return collect(FakeClient(), CONFIG, clock=lambda: NOW)


def intent(snapshot, **overrides):
    return dict(dict(id='paper-1', mode='paper', ticker=MARKET['ticker'], side='yes', contracts='2',
                     limit_price_dollars='.56', rules_hash=snapshot['rows'][0]['rules_hash'],
                     commence_time='2026-10-04T18:00:00Z', start_time_source='https://example.com/schedule',
                     expires_at='2026-10-04T17:00:00Z'), **overrides)


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.fee = fee_model(SERIES, [], MARKET['event_ticker'], NOW)

    def test_complement_ladders_preserve_fixed_point_and_fractional_sizes(self):
        self.assertEqual(asks(BOOK, 'yes'), [['0.5500', '2.50'], ['0.6000', '100.00']])
        self.assertEqual(asks(BOOK, 'no'), [['0.4700', '100.00']])
        self.assertEqual(asks({'orderbook_fp': {'yes_dollars': [], 'no_dollars': []}}, 'yes'), [])
        with self.assertRaises(ValueError):
            asks({'orderbook_fp': {'no_dollars': [['0.4000', 'NaN']]}}, 'yes')
        with self.assertRaises(ValueError):
            asks({'orderbook_fp': {'no_dollars': [['0.4000', '2'], ['0.4000', '3']]}}, 'yes')

    def test_sweep_costs_actual_levels_including_fees(self):
        q = sweep(asks(BOOK, 'yes'), '10', self.fee)
        self.assertEqual(D(q['premium_dollars']), D('5.875'))
        self.assertEqual(D(q['total_estimate_dollars']), D('6.05'))
        self.assertEqual(D(q['fee_estimate_dollars']), D('.175'))
        self.assertEqual([f['contracts'] for f in q['fills']], ['2.50', '7.50'])
        self.assertEqual(D(q['average_price_dollars']), D('.5875'))

    def test_limit_and_missing_liquidity_do_not_create_full_fill(self):
        q = sweep(asks(BOOK, 'yes'), '10', self.fee, '.56')
        self.assertEqual(D(q['filled_contracts']), D('2.5'))
        self.assertEqual(D(q['unfilled_contracts']), D('7.5'))
        self.assertEqual(sweep([], '1', self.fee)['payout_equivalent_american'], None)

    def test_unknown_fees_withhold_total_and_equivalent_odds(self):
        q = sweep(asks(BOOK, 'yes'), '1', None)
        self.assertEqual(D(q['premium_dollars']), D('.55'))
        for key in ('fee_estimate_dollars', 'total_estimate_dollars', 'payout_equivalent_american'):
            self.assertIsNone(q[key])

    def test_event_fee_override_clear_and_future_change(self):
        event = MARKET['event_ticker']
        changes = [dict(event_ticker=event, scheduled_ts=iso(NOW-timedelta(hours=1)),
                        fee_type_override='quadratic', fee_multiplier_override=2),
                   dict(event_ticker=event, scheduled_ts=iso(NOW+timedelta(hours=1)),
                        fee_type_override=None, fee_multiplier_override=None)]
        model = fee_model(SERIES, changes, event, NOW)
        self.assertEqual(D(model['taker_rate']), D('.14'))
        self.assertEqual(model['valid_until'], iso(NOW+timedelta(hours=1)))
        self.assertEqual(D(fee_model(SERIES, changes, event, NOW+timedelta(hours=2))['taker_rate']), D('.07'))
        self.assertIsNone(fee_model(dict(fee_type='unknown', fee_multiplier=1), [], event, NOW))
        self.assertIsNone(fee_model(dict(fee_type='quadratic'), [], event, NOW))

    def test_invalid_decimals_and_quantities_are_rejected(self):
        for value in (None, True, 'nan', 'Infinity', '-Infinity', 'bad'):
            with self.assertRaises(ValueError):
                decimal(value)
        for value in ('0', '-1', '.001', '100001'):
            with self.assertRaises(ValueError):
                sweep([], value, self.fee)

    def test_quote_times_require_timezone_and_reject_future(self):
        self.assertFalse(fresh('2026-10-04T16:00:00', NOW, 30))
        self.assertFalse(fresh(iso(NOW+timedelta(seconds=1)), NOW, 30))
        self.assertFalse(fresh(iso(NOW-timedelta(seconds=31)), NOW, 30))


class FeedTests(unittest.TestCase):
    def test_live_observation_does_not_invent_start_probability_or_book_match(self):
        data = fixture()
        row = data['rows'][0]
        self.assertEqual(data['status'], 'ready')
        self.assertEqual(row['source_updated_at'], MARKET['updated_time'])
        self.assertEqual(row['observed_at'], iso(NOW))
        self.assertIsNone(row['model_probability'])
        self.assertIsNone(row['commence_time'])
        self.assertEqual(row['sportsbook_comparisons'], [])
        self.assertEqual(set(row['estimates']), {'1', '10', '100'})
        self.assertNotEqual(rules_hash(MARKET), rules_hash(dict(MARKET, rules_secondary='Different tie rule')))

    def test_fee_failure_keeps_gross_quote_but_blocks_all_in_cost(self):
        data = collect(FakeClient(fail_fees=True), CONFIG, clock=lambda: NOW)
        self.assertEqual(data['status'], 'partial')
        self.assertIsNone(data['rows'][0]['fee'])
        self.assertIsNone(data['rows'][0]['estimates']['10']['yes']['total_estimate_dollars'])

    def test_book_failure_and_empty_board_are_distinct(self):
        failed = collect(FakeClient(fail_book=True), CONFIG, clock=lambda: NOW)
        empty = collect(FakeClient(markets=[]), CONFIG, clock=lambda: NOW)
        self.assertEqual(failed['status'], 'error')
        self.assertEqual(empty['status'], 'ready')
        self.assertEqual(empty['rows'], [])

    def test_sampling_and_pagination_limits_are_explicit(self):
        markets = [dict(MARKET, ticker=f'KXNFLGAME-TEST-{i}') for i in range(3)]
        data = collect(FakeClient(markets=markets, complete=False), dict(CONFIG, max_markets=1), clock=lambda: NOW)
        self.assertTrue(data['bounded'])
        self.assertEqual(data['coverage'][0], dict(series='KXNFLGAME', discovered=3, sampled=1, discovery_complete=False))

    def test_paper_refresh_targets_only_requested_contracts(self):
        client = FakeClient()
        with patch.object(client, 'pages', wraps=client.pages) as calls:
            data = collect(client, CONFIG, clock=lambda: NOW, market_tickers=[MARKET['ticker']])
            self.assertEqual(len(data['rows']), 1)
            self.assertTrue(all(c.args[0] != '/markets' for c in calls.call_args_list))
        with self.assertRaises(ValueError):
            collect(client, CONFIG, market_tickers=['UNCONFIGURED-SERIES'])

    def test_only_verified_exact_settlement_match_can_compare(self):
        row = fixture()['rows'][0]
        outcome = dict(sport='NFL', game_id='g1', player='', market='h2h', side='Buffalo', line=None,
                       commence_time='2026-10-04T18:00:00Z')
        mapping = dict(ticker=row['ticker'], side='yes', rules_hash=row['rules_hash'], verified_at=iso(NOW),
                       evidence_url='https://example.com/rules', sportsbook_outcome=outcome,
                       settlement_verified=True, settlement_profile='test_identical_payouts')
        book = dict(outcome, book='test-book', price=-110, quoted_at=iso(NOW),
                    settlement_verified=True, settlement_profile='test_identical_payouts')
        self.assertEqual(len(comparisons(row, [mapping], [book], NOW, 900)), 1)
        for overrides in ({'settlement_profile':'push_refund'}, {'line':.5}, {'side':'Other team'},
                          {'settlement_verified':False}, {'quoted_at':iso(NOW-timedelta(hours=1))}):
            self.assertEqual(comparisons(row, [mapping], [dict(book, **overrides)], NOW, 900), [])
        self.assertEqual(comparisons(row, [dict(mapping, rules_hash='changed')], [book], NOW, 900), [])

    def test_public_client_is_bounded_get_only_and_cursors_checked(self):
        client = PublicClient(max_requests=0)
        with self.assertRaises(ValueError):
            client.get('/markets')
        with self.assertRaises(ValueError):
            client.get('/portfolio/orders')
        with self.assertRaises(ValueError):
            client.get('/markets/../portfolio/orders')
        with self.assertRaises(ValueError):
            ticker('../portfolio')
        with patch.object(client, 'get', side_effect=[dict(markets=[1], cursor='a'), dict(markets=[2], cursor='b')]):
            self.assertEqual(client.pages('/markets', 'markets', max_pages=2), ([1,2], False))
        with patch.object(client, 'get', return_value=dict(markets=[], cursor='a')):
            with self.assertRaises(ValueError):
                client.pages('/markets', 'markets')


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = fixture()
        self.ledger = dict(schema_version=1, mode='paper', orders=[])

    def run_order(self, **kwargs):
        return run_intent(self.snapshot, intent(self.snapshot, **kwargs), self.ledger, CONFIG, NOW)

    def test_paper_fill_is_idempotent_and_does_not_invent_edge(self):
        a = self.run_order()
        b = self.run_order()
        self.assertIs(a, b)
        self.assertEqual(len(self.ledger['orders']), 1)
        self.assertEqual(a['status'], 'simulated_filled')
        self.assertIsNone(a['expected_profit_dollars'])
        with self.assertRaises(ValueError):
            self.run_order(contracts='3')

    def test_partial_fill_consumes_depth_once(self):
        self.run_order()
        second = self.run_order(id='second')
        self.assertEqual(second['status'], 'simulated_partial')
        self.assertEqual(D(second['simulation']['filled_contracts']), D('.5'))
        third = self.run_order(id='third')
        self.assertEqual(third['status'], 'rejected')

    def test_stale_future_closed_rules_and_missing_fees_reject(self):
        for field, value in [('observed_at', iso(NOW-timedelta(seconds=31))),
                             ('observed_at', iso(NOW+timedelta(seconds=1))), ('status', 'closed'), ('fee', None)]:
            snapshot = fixture()
            snapshot['rows'][0][field] = value
            result = run_intent(snapshot, intent(snapshot), deepcopy(self.ledger), CONFIG, NOW)
            self.assertEqual(result['status'], 'rejected')
        for change in ({'mode':'live'}, {'rules_hash':'changed'}, {'commence_time':iso(NOW)},
                       {'start_time_source':''}, {'expires_at':iso(NOW)}, {'side':'maybe'}):
            result = run_intent(self.snapshot, intent(self.snapshot, **change), deepcopy(self.ledger), CONFIG, NOW)
            self.assertEqual(result['status'], 'rejected')

    def test_changed_fee_and_stale_snapshot_reject(self):
        self.snapshot['rows'][0]['fee']['valid_until'] = iso(NOW)
        self.assertEqual(self.run_order()['status'], 'rejected')
        self.snapshot = fixture()
        self.snapshot['generated_at'] = iso(NOW-timedelta(seconds=31))
        self.assertEqual(self.run_order(id='new')['status'], 'rejected')

    def test_order_daily_event_and_open_caps_include_fees(self):
        for key in CONFIG['paper_limits']:
            config = deepcopy(CONFIG)
            config['paper_limits'][key] = '1.10'
            result = run_intent(self.snapshot, intent(self.snapshot), deepcopy(self.ledger), config, NOW)
            self.assertEqual(result['status'], 'rejected', key)
            self.assertIn('limit exceeded', result['reason'])

    def test_daily_spend_does_not_reset_after_settlement(self):
        self.run_order()
        settle(self.ledger, {MARKET['ticker']:dict(MARKET, status='finalized', settlement_value_dollars='1')}, NOW)
        config = deepcopy(CONFIG)
        config['paper_limits']['max_daily_dollars'] = '1.30'
        result = run_intent(self.snapshot, intent(self.snapshot, id='second'), self.ledger, config, NOW)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('daily', result['reason'])

    def test_explicit_payout_forecast_covers_partial_settlements(self):
        forecast = dict(source='test model', at=iso(NOW), rules_hash=self.snapshot['rows'][0]['rules_hash'],
                        side='yes', includes_partial_settlements=True, expected_payout_per_contract='.60')
        result = self.run_order(forecast=forecast)
        self.assertEqual(D(result['expected_profit_dollars']), D('1.20')-D(result['cost_dollars']))
        invalid = self.run_order(id='new', forecast=dict(forecast, includes_partial_settlements=False))
        self.assertEqual(invalid['status'], 'rejected')

    def test_settlement_requires_explicit_final_value_and_handles_no_and_half(self):
        yes = self.run_order()
        no = self.run_order(id='no', side='no', limit_price_dollars='.50')
        self.assertEqual(settle(self.ledger, {MARKET['ticker']:dict(MARKET, status='finalized', result='yes')}, NOW), 0)
        markets = {MARKET['ticker']:dict(MARKET, status='finalized', settlement_value_dollars='.50')}
        self.assertEqual(settle(self.ledger, markets, NOW), 2)
        self.assertEqual(D(yes['settlement']['payout_dollars']), D('1'))
        self.assertEqual(D(no['settlement']['payout_dollars']), D('1'))
        self.assertEqual(settle(self.ledger, markets, NOW), 0)

    def test_private_ledger_roundtrip_and_malformed_file_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ledger.json'
            with ledger_file(path) as ledger:
                run_intent(self.snapshot, intent(self.snapshot), ledger, CONFIG, NOW)
            with ledger_file(path) as ledger:
                run_intent(self.snapshot, intent(self.snapshot), ledger, CONFIG, NOW)
                self.assertEqual(len(ledger['orders']), 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                with ledger_file(path):
                    pass
            self.assertEqual(path.read_text(), '{broken')


if __name__ == '__main__':
    if '--fixture' in sys.argv:
        print(json.dumps(fixture()))
    else:
        unittest.main()

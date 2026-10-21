"""Pinnacle rides along with the core refreshes at no extra cost and never reaches the feeds."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import check_sharp_reference as check  # noqa: E402
import sharp_reference as ref  # noqa: E402
from odds_budget import estimate_cost  # noqa: E402

NOW = datetime(2026, 10, 21, 11, 5, tzinfo=timezone.utc)   # 7:05 AM EDT
KICK = (NOW + timedelta(hours=8)).isoformat().replace('+00:00', 'Z')


def book(key, over=-110, under=-110, line=5.5):
    return dict(key=key, title=key.title(), last_update='2026-10-21T11:00:00Z', markets=[dict(
        key='totals', last_update='2026-10-21T11:00:00Z',
        outcomes=[dict(name='Over', point=line, price=over), dict(name='Under', point=line, price=under)])])


def event(eid='e1', books=('draftkings', 'pinnacle')):
    return dict(id=eid, sport_key='icehockey_nhl', commence_time=KICK, home_team='Home', away_team='Away',
                bookmakers=[book(b) for b in books])


class Params(unittest.TestCase):
    def test_us_region_becomes_ten_named_books(self):
        swapped = ref.odds_params(dict(regions='us', markets='totals'))
        self.assertNotIn('regions', swapped)
        books = swapped['bookmakers'].split(',')
        self.assertEqual(len(books), 10)
        self.assertIn('pinnacle', books)
        self.assertEqual(swapped['markets'], 'totals')

    def test_other_requests_unchanged(self):
        for params in (dict(regions='eu'), dict(regions='us,eu'), dict(bookmakers='pinnacle'), dict(dateFormat='iso')):
            self.assertEqual(ref.odds_params(params), params)

    def test_ten_named_books_cost_one_region(self):
        swapped = ref.odds_params(dict(regions='us', markets='h2h,spreads,totals'))
        self.assertEqual(estimate_cost('sports/icehockey_nhl/odds', swapped), 3)
        self.assertEqual(estimate_cost('sports/nfl/events/a/odds', dict(bookmakers=','.join(['b'] * 11), markets='m')), 2)


class Split(unittest.TestCase):
    def test_list_and_single_event(self):
        stripped, quotes = ref.split([event('e1'), event('e2')], '2026-10-21T11:05:00Z')
        self.assertEqual([[b['key'] for b in e['bookmakers']] for e in stripped], [['draftkings']] * 2)
        self.assertEqual(len(quotes), 4)
        self.assertEqual(quotes[0]['side'], 'Over')
        one, quotes = ref.split(event('e3'))
        self.assertEqual([b['key'] for b in one['bookmakers']], ['draftkings'])
        self.assertEqual({q['event_id'] for q in quotes}, {'e3'})

    def test_original_payload_not_mutated_and_odd_shapes_pass(self):
        original = event()
        ref.split(original)
        self.assertEqual(len(original['bookmakers']), 2)
        self.assertEqual(ref.split({'message': 'x'}), ({'message': 'x'}, []))
        self.assertEqual(ref.split([1, 2]), ([1, 2], []))


class Save(unittest.TestCase):
    def test_runs_merge_by_eastern_date_and_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = ref.Recorder('icehockey_nhl')
            rec.take([event()], '2026-10-21T11:05:00Z')
            first = rec.save(NOW, root=tmp)
            self.assertEqual(first.name, '2026-10-21.json')
            self.assertEqual(first.parent.name, 'nhl')
            again = ref.Recorder('icehockey_nhl')
            again.take([event()], '2026-10-21T20:00:00Z')        # same prices: no new quotes
            again.save(NOW + timedelta(hours=9), root=tmp)       # 4:05 PM EDT, same Eastern date
            data = json.loads(first.read_text())
            self.assertEqual(len(data['quotes']), 2)
            self.assertEqual(len(data['runs']), 2)
            self.assertEqual(data['quotes'][0][-1], '2026-10-21T11:05:00Z')
            late = NOW + timedelta(hours=16)                      # 03:05 UTC Oct 22 is 11:05 PM EDT Oct 21
            self.assertEqual(ref.path_for('nhl', late, tmp).name, '2026-10-21.json')
            self.assertEqual(ref.path_for('nhl', late + timedelta(hours=1), tmp).name, '2026-10-22.json')

    def test_save_never_raises(self):
        rec = ref.Recorder('nhl')
        with mock.patch.object(ref, 'save', side_effect=OSError('disk')):
            self.assertIsNone(rec.save(NOW))


class Response:
    def __init__(self, payload):
        self.payload, self.status_code, self.ok = payload, 200, True
        self.headers = {'x-requests-remaining': '9000', 'x-requests-last': '1'}

    def json(self):
        return self.payload


class Client(unittest.TestCase):
    def test_shared_client_requests_pinnacle_and_hides_it(self):
        from nba.pipeline import OddsClient
        client = OddsClient('key', 'icehockey_nhl')
        with mock.patch('nba.pipeline.requests.get', return_value=Response([event()])) as get:
            payload = client.get('odds', regions='us', markets='totals', oddsFormat='american')
        sent = get.call_args.kwargs['params']
        self.assertNotIn('regions', sent)
        self.assertIn('pinnacle', sent['bookmakers'])
        self.assertEqual([b['key'] for b in payload[0]['bookmakers']], ['draftkings'])
        self.assertEqual(len(client.reference.quotes), 2)
        with mock.patch('nba.pipeline.requests.get', return_value=Response([])) as get:
            client.get('events', dateFormat='iso')
        self.assertEqual(get.call_args.kwargs['params'].get('dateFormat'), 'iso')
        self.assertNotIn('bookmakers', get.call_args.kwargs['params'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Check(unittest.TestCase):
    def feeds(self, root, games=4):
        rows = [dict(event_id=f'g{i}', commence_time=KICK) for i in range(games)]
        write(root / 'docs/nhl/data/latest.json', dict(status='ready', last_success_at='2026-10-21T11:00:00Z', rows=rows))
        write(root / 'docs/mlb/data/latest.json', dict(status='ready', last_success_at='2026-10-20T11:00:00Z', rows=rows))

    def test_missing_reference_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.feeds(root)
            rows, errors, _ = check.check(root, NOW)
            self.assertEqual(len(errors), 1)
            self.assertIn('NHL', errors[0])
            self.assertFalse(next(r for r in rows if r['sport'] == 'mlb')['expected'])  # refreshed yesterday

    def test_recorded_reference_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.feeds(root)
            rec = ref.Recorder('nhl')
            rec.take([event()])
            rec.save(NOW, root=root)
            _, errors, _ = check.check(root, NOW)
            self.assertEqual(errors, [])
            text = check.report(*check.check(root, NOW), NOW)
            self.assertIn('Status: OK', text)

    def test_zero_quotes_errors_on_a_full_slate_warns_on_one_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.feeds(root, games=4)
            ref.Recorder('nhl').save(NOW, root=root)            # a run that got no Pinnacle quotes
            self.assertEqual(len(check.check(root, NOW)[1]), 1)
            self.feeds(root, games=1)
            write(root / 'docs/nba/data/latest.json', dict(status='ready', last_success_at='2026-10-21T10:47:00Z',
                                                           rows=[dict(event_id='n1', commence_time=KICK)] * 5))
            rec = ref.Recorder('nba')
            rec.take([event('n1')])
            rec.save(NOW, root=root)
            _, errors, warnings = check.check(root, NOW)
            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 1)


if __name__ == '__main__':
    unittest.main()

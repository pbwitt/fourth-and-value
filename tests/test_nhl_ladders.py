"""One-week NHL milestone-prop test: collection window, contract mapping, refresh
wiring that never touches published rows, the compact research file and grading."""
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl.refresh import SPORT, iso, record_milestones, refresh
from nhl.v2 import ladder_test, ladders

NOW = datetime(2026, 10, 5, 14, tzinfo=timezone.utc)          # 10 AM Eastern, inside the window
GAME = dict(nhl_game_id=2026020030, season=20262027, game_type=2, commence_time='2026-10-05T23:00:00Z',
            home_team='Buffalo Sabres', away_team='Chicago Blackhawks')
EVENT = dict(id='event1', sport_key=SPORT, commence_time=GAME['commence_time'],
             home_team=GAME['home_team'], away_team=GAME['away_team'])
RULES = {'draftkings': {'player': 'nhl_player_ot_no_so_participation', 'source': 'https://example.test/rules'}}


def outcome(name, price, point=None, player='Tage Thompson'):
    return {'name': name, 'price': price, 'description': player, **({} if point is None else {'point': point})}


def bookmaker(key, markets, at=NOW):
    return dict(key=key, title=key.title(), last_update=iso(at),
                markets=[dict(key=k, last_update=iso(at), outcomes=v) for k, v in markets.items()])


class WindowTests(unittest.TestCase):
    def test_eastern_window_and_24_hour_horizon(self):
        self.assertTrue(ladders.wanted(EVENT, NOW))
        self.assertFalse(ladders.wanted({**EVENT, 'commence_time': '2026-10-06T23:00:00Z'}, NOW), 'more than 24 hours out')
        self.assertFalse(ladders.wanted(EVENT, datetime(2026, 10, 5, 23, 30, tzinfo=timezone.utc)), 'already started')
        self.assertFalse(ladders.wanted({**EVENT, 'commence_time': '2026-10-04T02:00:00Z'},
                                        datetime(2026, 10, 3, 23, tzinfo=timezone.utc)), 'before the window')
        self.assertTrue(ladders.active(datetime(2026, 10, 11, 3, tzinfo=timezone.utc)), '11 PM Eastern on Oct 10')
        self.assertFalse(ladders.active(datetime(2026, 10, 11, 4, 30, tzinfo=timezone.utc)), 'Oct 11 Eastern: over')
        self.assertFalse(ladders.wanted({**EVENT, 'commence_time': None}, NOW))


class QuoteTests(unittest.TestCase):
    def test_milestones_become_their_base_contracts(self):
        event = {**EVENT, 'nhl_game_id': GAME['nhl_game_id'], 'bookmakers': [
            bookmaker('draftkings', {
                'player_goal_scorer_anytime': [outcome('Yes', 150), outcome('No', -190), outcome('Over', 150)],
                'player_shots_on_goal_alternate': [outcome('Over', -130, 2.5), outcome('Over', 160, 3.5),
                    outcome('Over', 300, 4.5, player=''), outcome('Over', 300), outcome('Over', 50, 5.5)],
                'player_shots_on_goal': [outcome('Over', -110, 2.5), outcome('Under', -110, 2.5)]}),
            bookmaker('betmgm', {'player_points_alternate': [outcome('Over', 120, 1.5)]}),
            bookmaker('fanduel', {'player_points_alternate': [outcome('Over', 120, 1.5)]}, at=NOW - timedelta(hours=25))]}
        rows = ladders.quotes(event, NOW, RULES)
        self.assertEqual([(r['book'], r['offered_market'], r['offered_side'], r['market'], r['side'], r['line'], r['price']) for r in rows], [
            ('draftkings', 'player_goal_scorer_anytime', 'Yes', 'player_goals', 'Over', 0.5, 150.0),
            ('draftkings', 'player_goal_scorer_anytime', 'No', 'player_goals', 'Under', 0.5, -190.0),
            ('draftkings', 'player_shots_on_goal_alternate', 'Over', 'player_shots_on_goal', 'Over', 2.5, -130.0),
            ('draftkings', 'player_shots_on_goal_alternate', 'Over', 'player_shots_on_goal', 'Over', 3.5, 160.0),
            ('betmgm', 'player_points_alternate', 'Over', 'player_points', 'Over', 1.5, 120.0)])
        self.assertEqual(rows[0]['nhl_game_id'], GAME['nhl_game_id'])
        self.assertEqual([rows[0]['settlement_profile'], rows[0]['settlement_verified']], ['nhl_player_ot_no_so_participation', True])
        self.assertEqual([rows[-1]['settlement_profile'], rows[-1]['settlement_verified']], ['unverified:betmgm:player_points', False])
        self.assertAlmostEqual(rows[0]['book_probability'], .4)
        self.assertEqual(ladders.quotes({**event, 'commence_time': '2026-10-05T13:00:00Z'}, NOW, RULES), [], 'started games')


class Client:
    requests, quota_remaining = 3, '17000'

    def __init__(self):
        self.prop_markets = []

    def get(self, suffix, **params):
        if suffix == 'events':
            return [EVENT]
        if suffix == 'odds':
            return [{**EVENT, 'bookmakers': []}]
        self.prop_markets.append(params['markets'])
        return {**EVENT, 'bookmakers': [bookmaker('draftkings', {
            'player_shots_on_goal': [outcome('Over', -110, 2.5), outcome('Under', -110, 2.5)],
            'player_goal_scorer_anytime': [outcome('Yes', 150)],
            'player_shots_on_goal_alternate': [outcome('Over', 160, 3.5)]})]}


class RefreshTests(unittest.TestCase):
    def test_milestones_ride_on_the_prop_request_and_stay_out_of_published_rows(self):
        client = Client()
        state = refresh(client, NOW, [GAME], {})
        self.assertEqual(client.prop_markets, ['player_shots_on_goal,player_goals,player_assists,player_points,'
            'player_goal_scorer_anytime,player_shots_on_goal_alternate,player_points_alternate'])
        self.assertEqual({r['market'] for r in state['rows']}, {'player_shots_on_goal'})
        self.assertEqual(len(state['rows']), 2, 'the published rows are exactly the base props')
        self.assertEqual(sorted(r['offered_market'] for r in state['milestone_quotes']),
                         ['player_goal_scorer_anytime', 'player_shots_on_goal_alternate'])

    def test_no_milestones_requested_outside_the_window_or_beyond_24_hours(self):
        for now, start in [(datetime(2026, 10, 12, 14, tzinfo=timezone.utc), '2026-10-12T23:00:00Z'),
                           (NOW, '2026-10-06T23:00:00Z')]:
            client = Client()
            game, event = {**GAME, 'commence_time': start}, {**EVENT, 'commence_time': start}
            with patch.object(Client, 'get', lambda self, suffix, **p: [event] if suffix == 'events' else
                              [{**event, 'bookmakers': []}] if suffix == 'odds' else
                              (self.prop_markets.append(p['markets']) or {**event, 'bookmakers': []})):
                state = refresh(client, now, [game], {})
            self.assertEqual(client.prop_markets, ['player_shots_on_goal,player_goals,player_assists,player_points'])
            self.assertEqual(state['milestone_quotes'], [])

    def test_a_refused_milestone_market_falls_back_to_the_standard_props(self):
        from nhl.refresh import FeedError
        client = Client()
        original = Client.get
        def get(self, suffix, **params):
            if 'alternate' in params.get('markets', ''):
                self.prop_markets.append(params['markets'])
                raise FeedError('Odds provider returned HTTP 422')
            return original(self, suffix, **params)
        with patch.object(Client, 'get', get), redirect_stderr(io.StringIO()) as err:
            state = refresh(client, NOW, [GAME], {})
        self.assertEqual(len(client.prop_markets), 2)
        self.assertEqual(client.prop_markets[1], 'player_shots_on_goal,player_goals,player_assists,player_points')
        self.assertEqual(len(state['rows']), 2)
        self.assertEqual(state['milestone_quotes'], [])
        self.assertIn('requesting the standard props only', err.getvalue())

    def test_a_milestone_failure_never_breaks_the_refresh(self):
        with patch('nhl.v2.ladders.quotes', side_effect=KeyError('bookmakers')), redirect_stderr(io.StringIO()) as err:
            state = refresh(Client(), NOW, [GAME], {})
        self.assertEqual(len(state['rows']), 2)
        self.assertIn('milestone quotes skipped', err.getvalue())
        with patch('nhl.v2.ladders.record', side_effect=ValueError('Stale independent-model inputs')), redirect_stderr(io.StringIO()) as err:
            record_milestones([dict(x=1)], dict(snapshot_id='s'), NOW)
        self.assertIn('milestone test skipped (ValueError)', err.getvalue())


class RecordTests(unittest.TestCase):
    def test_compact_research_file_with_plain_values(self):
        priced = [dict({k: None for k in ladders.KEEP}, player='Tage Thompson', price=150.0, book_probability=np.float64(.4),
                       estimated_ev=float('nan'), other_books=np.int64(3), key_drivers=['not kept'], model_inputs={'big': 1})]
        manifest = dict(version='nhl-v2.1', artifact_sha256='abc')
        with tempfile.TemporaryDirectory() as tmp, \
             patch('nhl.v2.inference.bundle', return_value=({}, manifest)), \
             patch('nhl.v2.inference.live_history', return_value=([], [], iso(NOW))), \
             patch('nhl.v2.inference.annotate', return_value=priced) as annotate:
            self.assertIsNone(ladders.record([], dict(snapshot_id='s', events=[]), NOW, out=tmp))
            path = ladders.record([dict(price=150)], dict(snapshot_id='snapshot123456789', events=[]), NOW, out=tmp)
            with gzip.open(path, 'rt') as f:
                saved = json.load(f)
        self.assertEqual(annotate.call_count, 1)
        self.assertEqual(saved['model_version'], 'nhl-v2.1')
        self.assertEqual(saved['window'], ['2026-10-04', '2026-10-11'])
        self.assertEqual(list(saved['rows'][0]), ladders.KEEP)
        self.assertEqual([saved['rows'][0]['book_probability'], saved['rows'][0]['estimated_ev'], saved['rows'][0]['other_books']], [.4, None, 3])
        self.assertTrue(Path(path).name.endswith('-snapshot1234.json.gz'))


def quote(**changes):
    row = dict(event_id='e1', nhl_game_id=10, commence_time='2026-10-05T23:00:00Z', game='Chicago @ Buffalo',
               home_team='Buffalo', away_team='Chicago', book='draftkings', offered_market='player_shots_on_goal_alternate',
               offered_side='Over', market='player_shots_on_goal', player='Tage Thompson', player_id=7, side='Over',
               line=2.5, price=150.0, book_probability=.4, quoted_at='2026-10-05T13:00:00Z', settlement_verified=True,
               offer_id='o1', fair_probability=None, other_book_probability=None, other_books=0, model_probability=.45,
               conditional_probability=.45, push_probability=0.0, estimated_ev=.125, projected_mean=2.6, model_status='x')
    row.update(changes)
    return row


class ReportTests(unittest.TestCase):
    SNAPS = [
        dict(snapshot_id='s1', decided_at='2026-10-05T11:00:00Z', model_version='nhl-v2.1',
             rows=[quote(price=120.0, book_probability=100/220, estimated_ev=-.01, offer_id='o0')]),
        dict(snapshot_id='s2', decided_at='2026-10-05T13:05:00Z', model_version='nhl-v2.1', rows=[
            quote(), quote(book='fanduel', price=130.0, book_probability=100/230, estimated_ev=.035, offer_id='o2'),
            quote(book='betmgm', price=200.0, book_probability=1/3, estimated_ev=None, settlement_verified=False, offer_id='o3'),
            quote(player_id=8, player='Scratched Skater', offered_market='player_goal_scorer_anytime', offered_side='Yes',
                  market='player_goals', line=.5, offer_id='o4', estimated_ev=.05)]),
        # A decision after puck drop never counts.
        dict(snapshot_id='s3', decided_at='2026-10-05T23:30:00Z', model_version='nhl-v2.1',
             rows=[quote(price=500.0, book_probability=1/6, estimated_ev=1.0, offer_id='o5')])]
    GAMES = [dict(game_id=10, season=20262027, home_score=3, away_score=2)]
    PLAYERS = [dict(game_id=10, player_id=7, season=20262027, shots=3, goals=0, assists=1, points=1)]

    def test_last_pregame_decision_best_verified_pick_and_void(self):
        graded = ladder_test.grade(self.SNAPS, self.GAMES, self.PLAYERS)
        shots = next(r for r in graded if r['offered_market'] == 'player_shots_on_goal_alternate')
        self.assertEqual([shots['decided_at'], shots['result'], shots['books']], ['2026-10-05T13:05:00Z', 'won', 3])
        self.assertEqual(shots['pick']['book'], 'draftkings', 'highest estimated EV among verified prices')
        self.assertEqual(shots['best']['book'], 'betmgm', 'best price is shown even when its rules are unverified')
        scorer = next(r for r in graded if r['offered_market'] == 'player_goal_scorer_anytime')
        self.assertEqual(scorer['result'], 'unresolved_participation', 'no box-score line: void, never a loss')

        report = ladder_test.summarize(graded, self.SNAPS, NOW)
        two = report['value_bets']['0.02']['all']
        self.assertEqual([two['count'], two['net_units']], [1, 1.5])
        self.assertEqual(report['value_bets']['0.10']['all']['count'], 1)
        self.assertEqual(report['results'], {'won': 1, 'unresolved_participation': 1})
        cal = report['calibration']['by_market']['player_shots_on_goal_alternate']
        self.assertEqual([cal['contracts'], cal['hit_rate']], [1, 1.0])
        self.assertAlmostEqual(cal['model_brier'], .55 ** 2)
        self.assertAlmostEqual(cal['best_price_brier'], (2 / 3) ** 2)
        text = ladder_test.markdown(report)
        self.assertIn('| 2% | All | 1 | +1.50 |', text)
        self.assertIn('too few to judge', text)

    def test_command_writes_report_and_skips_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = io.StringIO()
            with patch('sys.stdout', out):
                self.assertEqual(ladder_test.main(['--root', str(root)]), 0)
            self.assertIn('no quotes collected yet', out.getvalue())
            self.assertFalse((root / 'report.json').exists())
            for snap in self.SNAPS:
                with gzip.open(root / f"{snap['snapshot_id']}.json.gz", 'wt') as f:
                    json.dump(snap, f)
            history = root / 'history.json'
            history.write_text(json.dumps(dict(games=self.GAMES, players=self.PLAYERS)))
            with patch('sys.stdout', io.StringIO()):
                ladder_test.main(['--root', str(root), '--history', str(history)])
            self.assertEqual(json.loads((root / 'report.json').read_text())['contracts'], 2)
            self.assertIn('# NHL milestone-prop test', (root / 'REPORT.md').read_text())


if __name__ == '__main__':
    unittest.main()

"""Tests for the NBA Market Analytics ledger: pregame snapshot choice, ESPN settlement, voids and saved results."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_market_results as mr
from nba_weekly_results import fetch_results

ROOT = Path(__file__).resolve().parents[1]
START = '2026-10-22T02:00:00Z'   # Wednesday, Oct 21, 10 PM ET
HOME, AWAY = 'Los Angeles Lakers', 'Los Angeles Clippers'


def event(identifier='401000001', final=True, home_score='115', away_score='108'):
    return dict(id=identifier, date=START, competitions=[dict(
        id=identifier, date=START, status=dict(type=dict(completed=final, state='post' if final else 'in')),
        competitors=[dict(homeAway='home', score=home_score, team=dict(id='13', displayName=HOME)),
                     dict(homeAway='away', score=away_score, team=dict(id='12', displayName='LA Clippers'))])])


def summary(game=None):
    athletes = [dict(athlete=dict(id='99', displayName='José Player Jr.'), stats=['35', '25', '8', '7', '3-7', '2', '1', '4']),
                dict(athlete=dict(id='100', displayName='Ben Bench'), didNotPlay=True, stats=[])]
    return dict(header=copy.deepcopy(game or event()), boxscore=dict(players=[dict(
        team=dict(id='13', displayName=HOME), statistics=[dict(
            labels=['MIN', 'PTS', 'REB', 'AST', '3PT', 'BLK', 'STL', 'TO'], athletes=athletes)])]))


def quote(market, side, line, price, book='a', player='', start=START, quoted=None, event_id='evt1'):
    return dict(event_id=event_id, commence_time=start, quoted_at=quoted, home_team=HOME, away_team=AWAY,
                market=market, side=side, line=line, price=price, book=book, player=player)


def pair(market, line, over, under, player='', book='a'):
    return [quote(market, 'Over', line, over, book, player), quote(market, 'Under', line, under, book, player)]


QUOTES = (pair('player_points', 24.5, -110, -110, 'Jose Player Jr') + pair('player_points', 24.5, -120, 100, 'Jose Player Jr', 'b')
          + pair('player_points_rebounds_assists', 38.5, -115, -105, 'Jose Player Jr')
          + pair('player_threes', 2.5, 120, -150, 'Jose Player Jr')
          + pair('player_rebounds', 6.5, -110, -110, 'Ben Bench')
          + pair('totals', 224.5, -110, -110)
          + [quote('spreads', HOME, -6.5, -110), quote('spreads', AWAY, 6.5, -110),
             quote('h2h', HOME, None, -250), quote('h2h', AWAY, None, 200)])


class PregameSnapshotTests(unittest.TestCase):
    def test_uses_last_snapshot_before_tipoff(self):
        runs = [dict(checked_at='2026-10-21T15:00:00Z', rows=[quote('totals', 'Over', 226.5, -110)]),
                dict(checked_at='2026-10-21T21:00:00Z', rows=[quote('totals', 'Over', 224.5, -110)]),
                dict(checked_at='2026-10-22T03:00:00Z', rows=[quote('totals', 'Over', 230.5, -110)])]
        at, quotes = mr.nba_pregame_quotes(runs)['evt1']
        self.assertEqual(str(at), '2026-10-21 21:00:00+00:00')
        self.assertEqual(quotes[0]['line'], 224.5)

    def test_quote_timed_after_tipoff_is_ignored(self):
        late = dict(checked_at='2026-10-22T01:59:00Z', rows=[quote('totals', 'Over', 224.5, -110, quoted='2026-10-22T02:01:00Z')])
        self.assertEqual(mr.nba_pregame_quotes([late]), {})


class GameRowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fetch = lambda url: copy.deepcopy(dict(events=[event()]) if '/scoreboard?' in url else summary())
        probe = [dict(event_id='evt1', commence_time=START, home_team=HOME, away_team=AWAY)]
        games, self.players, resolved = fetch_results('2026-10-21', '2026-10-21', probe, directory.name, fetch)
        self.game = games[resolved[0]['nba_game_id']]

    def rows(self, quotes=QUOTES):
        return {r['market']: r for r in mr.nba_game_rows(quotes, self.game, self.players, 0)}

    def test_props_settle_from_espn_box_score_by_name(self):
        r = self.rows()
        self.assertEqual((r['player_points']['actual'], r['player_points']['label'], r['player_points']['books']),
                         (25.0, 'Jose Player Jr', 2))
        self.assertAlmostEqual(r['player_points']['p_over'], (0.5 + mr.no_vig(-120, 100)) / 2)
        self.assertEqual((r['player_points_rebounds_assists']['actual'], r['player_threes']['actual']), (40.0, 3.0))

    def test_player_who_did_not_play_is_void_not_zero(self):
        self.assertNotIn('player_rebounds', self.rows())

    def test_total_spread_and_moneyline(self):
        r = self.rows()
        self.assertEqual((r['totals']['actual'], r['totals']['line'], r['totals']['label']), (223.0, 224.5, 'LAC @ LAL'))
        self.assertEqual((r['spreads']['label'], r['spreads']['line'], r['spreads']['actual']), ('LAL -6.5', 6.5, 7.0))
        self.assertEqual((r['h2h']['label'], r['h2h']['line'], r['h2h']['actual']), ('LAL', 0.0, 7.0))
        self.assertGreater(r['h2h']['p_over'], 0.5)

    def test_road_favorite_is_expressed_from_its_side(self):
        r = self.rows([quote('spreads', HOME, 3.5, -110), quote('spreads', AWAY, -3.5, -110)])['spreads']
        self.assertEqual((r['label'], r['line'], r['actual']), ('LAC -3.5', 3.5, -7.0))

    def test_unknown_team_name_falls_back_to_the_name(self):
        self.assertEqual((mr.nba_code('LA Clippers'), mr.nba_code('Philadelphia 76ers'), mr.nba_code('Seattle Sonics')),
                         ('LAC', 'PHI', 'Seattle Sonics'))


class BuildTests(unittest.TestCase):
    NOW = '2026-10-23T12:00:00Z'

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runs = self.root / 'artifacts/nba/runs'
        self.runs.mkdir(parents=True)
        self.calls, self.down = [], False
        self.save('20261021T210000Z', dict(checked_at='2026-10-21T21:00:00Z', rows=QUOTES + [
            # Not started at build time: never graded and never fetched.
            quote('totals', 'Over', 230.5, -110, start='2026-10-24T23:30:00Z', event_id='later'),
            quote('totals', 'Under', 230.5, -110, start='2026-10-24T23:30:00Z', event_id='later')]))

    def save(self, name, run):
        with gzip.open(self.runs / f'{name}.json.gz', 'wt') as f:
            json.dump(run, f)

    def fetch(self, url):
        self.calls.append(url)
        if self.down:
            raise requests.ConnectionError('down')
        return copy.deepcopy(dict(events=[event()]) if '/scoreboard?' in url else summary())

    def build(self, offline=False):
        return mr.build_nba(2026, self.runs, self.root, self.fetch, offline, now=self.NOW)

    def test_end_to_end(self):
        payload = self.build()
        counts = {k: len(v) for k, v in payload['rows'].items() if v}
        self.assertEqual(counts, {'totals': 1, 'spreads': 1, 'h2h': 1, 'player_points': 1, 'player_threes': 1,
                                  'player_points_rebounds_assists': 1})
        self.assertEqual(payload['sport'], 'nba')
        self.assertEqual(payload['fields'][:6], ['period', 't', 'label', 'game', 'line', 'p_over'])
        self.assertEqual([(p['label'], p['games']) for p in payload['periods']], [('Week of Oct 19', 1)])
        self.assertIn('median 5.0 hours before tipoff', payload['periods'][0]['captured'])
        self.assertIn('median of 5.0 hours before tipoff', payload['notes']['timing'])
        self.assertEqual(payload['through'], 'Oct 21')
        self.assertEqual(sorted(u.split('?')[0].rsplit('/', 1)[1] for u in self.calls), ['scoreboard', 'summary'])
        self.assertTrue(all('20261024' not in u for u in self.calls), 'games not yet played are not fetched')

    def test_saved_results_are_reused_without_requests(self):
        first = self.build()
        self.calls.clear()
        self.assertEqual(self.build()['rows'], first['rows'])
        self.assertEqual(self.calls, [], 'a final game with a box score is not fetched again')
        self.down = True
        self.assertEqual(self.build(offline=True)['rows'], first['rows'])

    def test_unfinished_game_is_fetched_again_later(self):
        original = summary
        self.fetch = lambda url: (self.calls.append(url), copy.deepcopy(
            dict(events=[event(final=False)]) if '/scoreboard?' in url else original()))[1]
        self.assertEqual(sum(len(v) for v in self.build()['rows'].values()), 0)
        self.calls.clear()
        self.fetch = BuildTests.fetch.__get__(self)
        self.assertEqual(len(self.build()['rows']['totals']), 1)
        self.assertTrue(self.calls)

    def test_feed_failure_publishes_saved_results_only(self):
        self.down = True
        payload = self.build()
        self.assertEqual(sum(len(v) for v in payload['rows'].values()), 0)
        self.assertEqual(payload['periods'], [])
        self.assertIn('not graded yet', payload['notes']['empty'])

    def test_offline_never_fetches(self):
        self.assertEqual(sum(len(v) for v in self.build(offline=True)['rows'].values()), 0)
        self.assertEqual(self.calls, [])


class PageTests(unittest.TestCase):
    def test_hub_has_nba_markers_links_and_redirect(self):
        hub = (ROOT / 'docs/markets/index.html').read_text()
        for marker in ('market-status:nba:start', 'market-table:nba:start', 'data-sport-link="nba"', 'data-for="nba"'):
            self.assertIn(marker, hub)
        self.assertIn('?sport=nba', (ROOT / 'docs/markets/nba/index.html').read_text())

    def test_static_summary_publishes_into_the_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / 'index.html'
            page.write_text((ROOT / 'docs/markets/index.html').read_text())
            payload = mr.package('nba', 2026, mr.NBA_MARKETS, [], [], mr.SEASON_WINDOWS,
                                 dict(empty='NBA results are not graded yet.'))
            with mock.patch.object(mr, 'ROOT', Path(directory)):
                mr.publish_static(payload, page)
            self.assertIn('<!-- market-status:nba:start -->NBA results are not graded yet.<!-- market-status:nba:end -->',
                          page.read_text())


if __name__ == '__main__':
    unittest.main()

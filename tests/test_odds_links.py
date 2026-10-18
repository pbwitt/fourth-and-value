"""Sportsbook links: requested with every odds call, stored as the provider sends them,
and filled in per reader state only when shown."""
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import odds_links
from nba import pipeline
from nba.pipeline import flatten, iso
import fetch_all_player_props as nfl_fetch

NOW = datetime(2026, 10, 18, 11, tzinfo=timezone.utc)
SLIP = 'https://sportsbook.draftkings.com/event/123?outcomes=0QA1'
GAME = 'https://sportsbook.draftkings.com/event/123'
STATE_SLIP = 'https://sports.{state}.betmgm.com/en/sports/events/456?options=789'


def event(outcome_link=None, market_link=None, book_link=None):
    outcome = dict(name='Over', description='Sean Monahan', point=0.5, price=310)
    market = dict(key='player_assists', outcomes=[outcome])
    book = dict(key='draftkings', title='DraftKings', last_update=iso(NOW), markets=[market])
    for target, link in ((outcome, outcome_link), (market, market_link), (book, book_link)):
        if link:
            target['link'] = link
    return dict(id='g', sport_key='icehockey_nhl', home_team='Minnesota Wild', away_team='Columbus Blue Jackets',
                commence_time=iso(NOW + timedelta(hours=7)), bookmakers=[book])


class OddsLinksTests(unittest.TestCase):
    def test_most_specific_https_link_wins(self):
        self.assertEqual(odds_links.bet_link(dict(link=GAME), dict(link=GAME), dict(link=SLIP)), (SLIP, 'bet'))
        self.assertEqual(odds_links.bet_link(dict(link=GAME), dict(link=GAME + '/m'), {}), (GAME + '/m', 'market'))
        self.assertEqual(odds_links.bet_link(dict(link=GAME), {}, {}), (GAME, 'event'))
        self.assertEqual(odds_links.bet_link({}, {}, dict(link='javascript:alert(1)')), (None, None))
        self.assertEqual(odds_links.bet_link({}, {}, dict(link='http://insecure.example/')), (None, None))
        self.assertEqual(odds_links.link_fields({}, {}, {}), {}, 'no empty fields on rows without links')
        self.assertEqual(odds_links.link_fields({}, {}, dict(link=SLIP)), dict(bet_link=SLIP, bet_link_level='bet'))

    def test_state_is_filled_per_reader(self):
        self.assertEqual(odds_links.for_state(STATE_SLIP, 'MD'), 'https://sports.md.betmgm.com/en/sports/events/456?options=789')
        self.assertEqual(odds_links.for_state(STATE_SLIP, 'dc'), 'https://sports.dc.betmgm.com/en/sports/events/456?options=789')
        self.assertEqual(odds_links.for_state('https://{STATE}.betrivers.com/#event/1', 'md'), 'https://md.betrivers.com/#event/1')
        for state in (None, '', 'maryland', 'm1'):
            self.assertIsNone(odds_links.for_state(STATE_SLIP, state), state)
        self.assertEqual(odds_links.for_state(SLIP, None), SLIP, 'links without a state need none')
        self.assertIsNone(odds_links.for_state('https://x.example/{region}', 'md'), 'unknown placeholders are never shown')
        self.assertTrue(odds_links.is_state_specific(STATE_SLIP))
        self.assertFalse(odds_links.is_state_specific(SLIP))

    def test_shared_flatten_keeps_the_link_on_nhl_mlb_and_nba_rows(self):
        markets = {'player_assists': 'Assists'}
        rows = flatten(event(SLIP, GAME, GAME), NOW, 'icehockey_nhl', markets, list(markets))
        self.assertEqual((rows[0]['bet_link'], rows[0]['bet_link_level']), (SLIP, 'bet'))
        rows = flatten(event(None, None, GAME), NOW, 'icehockey_nhl', markets, list(markets))
        self.assertEqual(rows[0]['bet_link_level'], 'event')
        rows = flatten(event(), NOW, 'icehockey_nhl', markets, list(markets))
        self.assertNotIn('bet_link', rows[0])

    def test_shared_client_asks_for_links_on_odds_only_at_the_same_cost(self):
        budget = Mock(ensure=Mock(), observe=Mock())
        client = pipeline.OddsClient('key', 'icehockey_nhl', budget=budget)
        response = Mock(ok=True, headers={}, json=Mock(return_value={}))
        with patch.object(pipeline.requests, 'get', return_value=response) as get:
            client.get('events/g/odds', regions='us', markets='player_assists', oddsFormat='american')
            self.assertEqual(get.call_args.kwargs['params']['includeLinks'], 'true')
            self.assertEqual(budget.ensure.call_args.args[0], 1, 'one market, one region: links cost nothing extra')
            client.get('events')
            self.assertNotIn('includeLinks', get.call_args.kwargs['params'])
        # Links are optional: if the provider refuses them, the refresh asks once without.
        refused = Mock(ok=False, status_code=422, headers={}, json=Mock(return_value={}))
        with patch.object(pipeline.requests, 'get', side_effect=[refused, response]) as get:
            self.assertEqual(client.get('events/g/odds', regions='us', markets='player_assists'), {})
            self.assertEqual([c.kwargs['params'].get('includeLinks') for c in get.call_args_list], ['true', None])
        with patch.object(pipeline.requests, 'get', return_value=Mock(ok=False, status_code=401, headers={})):
            with self.assertRaises(pipeline.FeedError):
                client.get('events/g/odds', regions='us', markets='player_assists')

    def test_nfl_props_fetch_requests_and_writes_links(self):
        payload = dict(id='g', commence_time='2026-10-18T20:05:00Z', home_team='Los Angeles Rams', away_team='Arizona Cardinals',
                       bookmakers=[dict(key='betrivers', title='BetRivers', last_update='2026-10-18T11:07:17Z',
                                        link='https://{state}.betrivers.com/#event/9',
                                        markets=[dict(key='player_receptions', outcomes=[
                                            dict(name='Over', description='Blake Corum', point=1.5, price=150,
                                                 link='https://{state}.betrivers.com/#event/9?coupon=single|1|'),
                                            dict(name='Under', description='Blake Corum', point=1.5, price=-190)])])])
        with patch.object(nfl_fetch, 'req_json', return_value=payload) as req, patch.object(nfl_fetch.BUDGET, 'ensure'):
            rows = nfl_fetch.fetch_event_props('key', 'americanfootball_nfl', 'g', ['player_receptions'], 'us', 8, 0)
        self.assertIn('&includeLinks=true&', req.call_args.args[0])
        self.assertEqual([(r['name'], r['bet_link_level']) for r in rows], [('Over', 'bet'), ('Under', 'event')])
        self.assertTrue(rows[0]['bet_link'].endswith('coupon=single|1|'))
        with patch.object(nfl_fetch, 'req_json', side_effect=[{'__422__': True, 'body': 'x'}, payload]) as req, \
                patch.object(nfl_fetch.BUDGET, 'ensure'):
            rows = nfl_fetch.fetch_event_props('key', 'americanfootball_nfl', 'g', ['player_receptions'], 'us', 8, 0)
        self.assertNotIn('includeLinks', req.call_args.args[0], 'a refusal is retried without links')
        self.assertEqual(len(rows), 2)


class PropsFeedTests(unittest.TestCase):
    def test_links_ride_on_the_shortlist_feed_not_the_full_board(self):
        from build_props_site import prepare_records
        with TemporaryDirectory() as td:
            path = Path(td) / 'merged.csv'
            fields = ['game_id', 'commence_time', 'home_team', 'away_team', 'game', 'bookmaker', 'bookmaker_title', 'market',
                      'market_std', 'player', 'name', 'price', 'point', 'last_update', 'bet_link', 'bet_link_level']
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow(dict(game_id='g', commence_time='2026-10-18T20:05:00Z', home_team='Los Angeles Rams',
                                away_team='Arizona Cardinals', game='Arizona Cardinals @ Los Angeles Rams', bookmaker='betrivers',
                                bookmaker_title='BetRivers', market='receptions', market_std='receptions', player='Blake Corum',
                                name='over', price=150, point=1.5, last_update='2026-10-18T11:07:17Z',
                                bet_link='https://{state}.betrivers.com/#event/9', bet_link_level='bet'))
                w.writerow(dict(game_id='g', commence_time='2026-10-18T20:05:00Z', home_team='Los Angeles Rams',
                                away_team='Arizona Cardinals', game='Arizona Cardinals @ Los Angeles Rams', bookmaker='fanduel',
                                bookmaker_title='FanDuel', market='receptions', market_std='receptions', player='Blake Corum',
                                name='over', price=140, point=1.5, last_update='2026-10-18T11:07:17Z'))
            shortlist = prepare_records(path, links=True)
            self.assertEqual([r['bet_link_level'] for r in shortlist], ['bet', None])
            self.assertIsNone(shortlist[1]['bet_link'])
            json.dumps(shortlist, allow_nan=False)
            self.assertNotIn('bet_link', prepare_records(path)[0], 'the full board stays small')


if __name__ == '__main__':
    unittest.main()

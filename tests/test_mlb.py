import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mlb.refresh import (SPORT, MARKETS, PROPS, PHASES, FeedError, compare, context, flatten,
                         history_lookup, iso, load_history, match_events, refresh, schedule, validate)

NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
GAME = dict(mlb_game_id=1, season=2026, game_type='R', phase='Regular season',
            commence_time='2026-09-22T00:00:00Z', home_team='New York Yankees', away_team='Athletics',
            home_pitcher={'id':1,'fullName':'Test Pitcher'}, away_pitcher=None, game_number=1,
            doubleheader=False, venue='Test Park', lineup_status='Batting lineups not verified')
EVENT = dict(id='event1', sport_key=SPORT, commence_time=GAME['commence_time'],
             home_team=GAME['home_team'], away_team=GAME['away_team'])


class MLBTests(unittest.TestCase):
    def test_regular_and_all_playoff_rounds(self):
        games = [dict(gamePk=i, gameType=phase, season='2026', gameDate=GAME['commence_time'],
            status=dict(abstractGameState='Preview', detailedState='Scheduled'),
            teams={s:{'team':{'id':j+1,'name':s}} for j,s in enumerate(['home','away'])}) for i,phase in enumerate(PHASES)]
        invalid = [dict(resumedFrom='2026-09-20'),dict(gameType='S'),dict(gameType='A'),dict(season='2025'),
            dict(gameDate='2025-12-01T00:00:00Z'),dict(status={'abstractGameState':'Final'}),
            dict(status={'abstractGameState':'Preview','detailedState':'Postponed'}),
            dict(status={'abstractGameState':'Preview','detailedState':'Delayed Start'}),
            dict(status={'abstractGameState':'Preview','detailedState':'Scheduled','startTimeTBD':True}),
            dict(teams={'home':{'team':{'name':'TBD'}},'away':{'team':{'name':'TBD'}}})]
        payload = {'dates':[{'games':games+[{**games[0],**change,'gamePk':i+100} for i,change in enumerate(invalid)]}]}
        with patch('mlb.refresh.official_json',return_value=payload): result=schedule(NOW)
        self.assertEqual({g['game_type'] for g in result},set(PHASES))
        self.assertEqual(len(result),5)

    def test_doubleheaders_match_by_time_not_just_teams(self):
        second = {**GAME,'mlb_game_id':2,'commence_time':'2026-09-22T03:00:00Z','game_number':2,'doubleheader':True}
        event2 = {**EVENT,'id':'event2','commence_time':second['commence_time']}
        pairs=match_events([EVENT,event2],[GAME,second])
        self.assertEqual([g['mlb_game_id'] for _,g in pairs],[1,2])
        self.assertEqual(match_events([EVENT],[GAME,GAME]),[])
        self.assertEqual(match_events([EVENT,{**EVENT,'id':'duplicate'}],[GAME]),[])

    def test_team_alias_and_time_mismatch(self):
        self.assertEqual(len(match_events([{**EVENT,'away_team':'Oakland Athletics'}],[GAME])),1)
        self.assertEqual(match_events([{**EVENT,'commence_time':'2026-09-22T02:00:00Z'}],[GAME]),[])

    def test_baseball_prop_pairing(self):
        event = {**EVENT,'bookmakers':[dict(key='a',title='A',markets=[dict(key='pitcher_strikeouts',last_update=iso(NOW),
            outcomes=[dict(name=s,description='Test Pitcher',point=5.5,price=-110) for s in ['Over','Under']])])]}
        rows=compare(flatten(event,NOW,SPORT,MARKETS,list(PROPS)))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['fair_probability'],.5)

    def test_innings_are_outs_not_decimal_numbers(self):
        history={'pitching':{'testpitcher':dict(player_id=1,stat=dict(inningsPitched='5.2',outs=17,
            strikeOuts=17,earnedRuns=2,gamesPlayed=1,gamesStarted=1))}}
        row=context(dict(player='Test Pitcher',market='pitcher_strikeouts'),GAME,history)
        self.assertEqual(row['stat_context']['k_per_nine'],27)
        self.assertAlmostEqual(row['stat_context']['era'],54/17)
        self.assertEqual(row['starter_status'],'Listed probable starter')
        self.assertIsNone(row['model_probability'])
        self.assertEqual(row['stat_context']['innings'],'5.2')

    def test_missing_and_ambiguous_stats_do_not_invent_probabilities(self):
        self.assertIsNone(context(dict(player='Unknown',market='batter_hits'),GAME,{})['stat_context'])
        h=dict(fetched_at=iso(NOW),season=2026,groups={'hitting':[dict(name='Same Name',player_id=1),dict(name='Same Name',player_id=2)]})
        self.assertEqual(history_lookup(h,NOW),{'hitting':{}})
        h['fetched_at']=iso(NOW-timedelta(hours=37));self.assertEqual(history_lookup(h,NOW),{})

    def test_season_history_excludes_today_and_postseason(self):
        payload=lambda group:dict(stats=[dict(group={'displayName':group},splits=[],totalSplits=0)])
        with patch('mlb.refresh.read_json',return_value={}),patch('mlb.refresh.save_json'),patch('mlb.refresh.official_json',side_effect=[payload('hitting'),payload('pitching')]) as get:
            result=load_history(NOW)
        self.assertEqual(result['through_date'],'2026-09-20')
        self.assertEqual(get.call_args.kwargs['gameType'],'R')
        self.assertEqual(get.call_args.kwargs['endDate'],'2026-09-20')

    def test_incomplete_stats_and_schedule_do_not_pass(self):
        with patch('mlb.refresh.read_json',return_value={}),patch('mlb.refresh.official_json',return_value={'stats':[{'group':{'displayName':'hitting'},'splits':[],'totalSplits':10}]}):
            self.assertIn('error',load_history(NOW))
        with patch('mlb.refresh.official_json',return_value={}):
            with self.assertRaises(FeedError):schedule(NOW)

    def test_empty_slate_does_not_spend_on_props(self):
        class Client:
            requests=1
            quota_remaining=None
            def get(self,suffix,**params):
                assert suffix=='events'
                return []
        result=refresh(Client(),NOW,[],{})
        self.assertEqual(result['status'],'waiting_for_markets')
        self.assertEqual(validate(result,NOW),[])

    def test_quote_expiry_and_started_game_guard(self):
        state=dict(status='ready',season=2026,last_success_at=iso(NOW),rows=[])
        self.assertTrue(validate({**state,'last_success_at':iso(NOW-timedelta(hours=13))},NOW))
        state['rows']=[dict(game_type='R',commence_time=iso(NOW-timedelta(minutes=1)),quoted_at=iso(NOW))]
        self.assertTrue(validate(state,NOW))


if __name__ == '__main__':unittest.main()

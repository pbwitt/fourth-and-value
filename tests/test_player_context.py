import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from player_context import nhl_context, mlb_context, windows
from mlb.models import State
from nhl.v2.features import History
from nba.pipeline import add_baselines


class PlayerContextTests(unittest.TestCase):
    def test_nhl_observed_form_is_distinct_from_weighted_model_inputs(self):
        history=History()
        for day in range(1,8):
            history.add_player(dict(player_id=1,game_date=f'2026-09-{day:02}',available_at=f'2026-09-{day+1:02}T12:00:00Z',
                shots=day,goals=0,assists=1,points=1,toi=20,position='F'))
        features=history.player_features(1,'F','2026-09-10',datetime(2026,9,10,tzinfo=timezone.utc))
        original=copy.deepcopy(features)
        result=nhl_context(history.players[1],features,0,'opportunity_nb','nhl-v2.1')
        self.assertEqual(result['recent'],[dict(games=5,mean=5,workload=20),dict(games=7,mean=4,workload=20)])
        self.assertAlmostEqual(result['inputs'][1]['value'],features['opportunity_means'][0]/features['projected_toi']*60,places=3)
        self.assertTrue(all(i['used'] for i in result['inputs']))
        self.assertIn('30-day',result['note'])
        self.assertEqual(features,original)
        self.assertEqual(result['through'],'2026-09-07')

    def test_mlb_uses_same_predate_window_and_decimal_innings(self):
        history=State()
        history.pitchers[1]=[dict(date='2026-09-01',outs=17,strikeOuts=0),dict(date='2026-09-02',outs=16,strikeOuts=6),
                             dict(date='2026-09-03',outs=99,strikeOuts=99),dict(date='2024-01-01',outs=99,strikeOuts=99)]
        features=dict(starter_outs5=15,starter_pitches5=85,starter_bf=22,starter_k_rate=.25,opp_k_rate=.20,starter_rest=1)
        result=mlb_context(history,'2026-09-03',{'id':1},'pitcher_strikeouts',features,{'kind':'rolling'})
        self.assertEqual(result['sample_games'],2)
        self.assertEqual(result['recent'],[dict(games=2,mean=3,workload=5.5)])
        self.assertEqual(result['through'],'2026-09-02')
        used=[i['label'] for i in result['inputs'] if i['used']]
        self.assertEqual(used,['Pitcher strikeout rate','Opponent strikeout rate','Batters faced / start'])
        self.assertEqual(result['inputs'][0]['value'],5)
        self.assertEqual(result['inputs'][2]['value'],25)
        self.assertIn('decimal innings',result['note'])

    def test_selected_model_controls_input_labels(self):
        history=State()
        history.pitchers[1]=[dict(date='2026-09-01',outs=17,strikeOuts=1)]
        f=dict(starter_outs5=15,starter_k_rate=.25)
        result=mlb_context(history,'2026-09-03',{'id':1},'pitcher_outs',f,{'kind':'boosting','features':['starter_k_rate']})
        self.assertEqual([i['label'] for i in result['inputs'] if i['used']],['Pitcher strikeout rate'])
        self.assertIsNone(result['inputs'][1]['value'])
        batter=mlb_context(history,'2026-09-03',{'id':2},'batter_hits',{}, {'kind':'rolling'})
        self.assertEqual(batter['recent'],[])
        self.assertIsNone(batter['through'])

    def test_nba_complete_games_share_stat_and_minutes_sample(self):
        games=[dict(GAME_ID=str(i),PLAYER_ID=1,PLAYER_NAME='Test Player',MIN=30,PTS=20,
                    GAME_DATE=f'2026-08-{i:02}') for i in range(1,26)]
        games.append(dict(games[-1],GAME_ID='missing',GAME_DATE='2026-08-26',PTS=None,MIN=99))
        row=dict(player='Test Player',market='player_points',market_label='Points',line=20,side='Under')
        result=add_baselines([row],{'players':games},datetime(2026,9,1,tzinfo=timezone.utc))[0]
        c=result['player_context']
        self.assertEqual(c['sample_games'],25)
        self.assertTrue(all(w['mean']==20 and w['workload']==30 for w in c['recent']))
        self.assertIsNone(result['model_probability'])
        self.assertEqual(result['baseline_push'],1)

    def test_missing_observations_are_not_zero(self):
        self.assertEqual(windows([dict(stat=None,toi=None)],[5,10],'stat','toi'),[dict(games=1,mean=None,workload=None)])


if __name__=='__main__':unittest.main()

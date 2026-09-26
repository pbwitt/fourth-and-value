from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_selection as selection
import editorial_writer as writer

class VarietyTests(unittest.TestCase):
    def select(self,pools,state=None):
        state=state or {'slots':{}}
        calls=[]
        def discover(sport,excluded):
            calls.append(sport)
            return next(({'event_id':event} for event in pools.get(sport,[]) if event not in excluded),None)
        chosen=selection.select(state,['NFL','MLB','NBA','NHL'],discover)
        return state,chosen,calls
    def test_distinct_sports_win_before_same_sport(self):
        state,chosen,calls=self.select({'MLB':['a','b'],'NHL':['c']})
        self.assertEqual([c['sport'] for c in chosen.values()],['MLB','NHL'])
        self.assertEqual(calls,['NFL','MLB','NBA','NHL'])
    def test_all_sports_checked_before_mlb_fallback(self):
        state,chosen,calls=self.select({'MLB':['a','b']})
        self.assertEqual(calls[:4],['NFL','MLB','NBA','NHL'])
        self.assertEqual([c['event_id'] for c in chosen.values()],['a','b'])
        self.assertEqual(chosen[1]['selection'],'same_sport_fallback')
    def test_same_sport_fallback_applies_to_every_sport(self):
        for sport in ['NFL','MLB','NBA','NHL']:
            with self.subTest(sport=sport):
                _,chosen,_=self.select({sport:['a','b']})
                self.assertEqual([c['sport'] for c in chosen.values()],[sport,sport])
                self.assertEqual(chosen[1]['selection'],'same_sport_fallback')

    def test_cannot_duplicate_one_qualifying_matchup(self):
        state,chosen,_=self.select({'MLB':['a']})
        self.assertEqual(len(chosen),1)
    def test_retry_preserves_published_event_and_budget_slot(self):
        state={'allocation':[['MLB','matchup:a'],['NFL','matchup:bad']], 'slots':{
            '0-mlb':{'status':'published','event_id':'a'},'1-nfl':{'status':'waiting_for_data'}}}
        state,chosen,_=self.select({'MLB':['a','b']},state)
        self.assertEqual(state['allocation'],[['MLB','matchup:a'],['MLB','matchup:b']])
        self.assertEqual(state['slots']['0-mlb']['status'],'published')
        self.assertNotIn('1-nfl',state['slots'])
    def test_started_and_rejected_paid_slots_are_not_reassigned(self):
        state={'allocation':[['NFL','matchup:a'],['MLB','matchup:b']], 'slots':{
            '0-nfl':{'status':'started'},'1-mlb':{'status':'skipped'}}}
        state,chosen,_=self.select({'MLB':['c']},state)
        self.assertEqual(state['allocation'],[['NFL','matchup:a'],['MLB','matchup:b']])
    def test_unknown_legacy_matchup_cannot_get_same_sport_duplicate(self):
        state={'allocation':[['MLB','news-market']], 'slots':{'0-mlb':{'status':'published'}}}
        _,chosen,_=self.select({'MLB':['a']},state)
        self.assertEqual(len(chosen),1)

class ModelQualification(unittest.TestCase):
    now=datetime(2026,9,26,12,tzinfo=timezone.utc)
    def packet(self):
        return {'sport':'MLB','data_readiness':{'ready':True},'markets':[{'id':'total-a','event_id':'a','game':'Away @ Home'}],
            'model_rows':[{'id':'model-a','event_id':'a','model_mean':5.2,'model_version':'v1','model_input_through':'2026-09-25'}],
            'target_game':{'event_id':'a','game':'Away @ Home'}}
    def test_prices_alone_do_not_qualify(self):
        packet=self.packet();packet['model_rows']=[]
        with self.assertRaisesRegex(ValueError,'model inputs'):writer.require_model(packet,self.now)
    def test_other_game_or_old_inputs_do_not_qualify(self):
        packet=self.packet();packet['model_rows'][0]['event_id']='b'
        with self.assertRaises(ValueError):writer.require_model(packet,self.now)
        packet=self.packet();packet['model_rows'][0]['model_input_through']='2026-09-23'
        with self.assertRaises(ValueError):writer.require_model(packet,self.now)
    def test_qualifying_matchup_passes(self):writer.require_model(self.packet(),self.now)
    def test_missing_nfl_feed_cannot_claim_current_data_checks_passed(self):
        status=writer.data_readiness('NFL',{}, {'generated_at':self.now.isoformat()},self.now)
        self.assertFalse(status['ready']);self.assertIn('model inputs',status['reason'])
    def test_alternate_matchup_is_tried_when_first_lacks_reporting(self):
        packet=self.packet();packet.pop('target_game')
        packet['markets'].append({'id':'total-b','event_id':'b','game':'Other @ Team'})
        packet['model_rows'].append(dict(packet['model_rows'][0],id='model-b',event_id='b'))
        with patch.object(writer,'evidence',return_value=packet),patch.object(writer.reporting,'collect',side_effect=[[],[{'url':'https://www.mlb.com/news/a'}]]):
            result=writer.discover_matchup('MLB',self.now)
        self.assertEqual(result['event_id'],'b')

if __name__=='__main__':unittest.main()

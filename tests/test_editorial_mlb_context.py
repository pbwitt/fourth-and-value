from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_mlb_context as context
import editorial_ideas as ideas
import editorial_writer as writer

class WildcardContextTests(unittest.TestCase):
    def row(self,identifier,division_rank='1',league_rank='1',wild=None):
        return {'team':{'id':identifier,'name':f'Team {identifier}'},'divisionRank':division_rank,'leagueRank':league_rank,
            'wildCardRank':wild,'wins':80,'losses':70,'records':{'splitRecords':[{'type':'home','wins':45,'losses':30},{'type':'away','wins':35,'losses':40}]}}
    def test_bracket_uses_division_leaders_and_wildcards_not_six_best_records(self):
        groups=[]
        for league in (103,104):
            groups.append({'league':{'id':league},'teamRecords':[self.row(league*10+i,league_rank=str(i)) for i in (1,2,3)]+[self.row(league*10+i,'2',str(i-2),str(i-3)) for i in (4,5,6)]})
        pairs=context.projected_pairs({'records':groups})
        self.assertEqual(len(pairs),4)
        self.assertEqual([(h['team']['id'],a['team']['id']) for _,_,h,a in pairs[:2]],[(1033,1036),(1034,1035)])
    def test_head_to_head_deduplicates_and_excludes_unfinished_future_and_other_games(self):
        def game(pk,day='2026-09-24',state='Final',away=2):
            return {'gamePk':pk,'officialDate':day,'gameType':'R','status':{'abstractGameState':state},'teams':{'home':{'team':{'id':1},'score':5},'away':{'team':{'id':away},'score':3}}}
        games={'dates':[{'games':[game(1),game(1),game(2,state='Live'),game(3,day='2026-09-27'),game(4,away=3)]}]}
        result=context.matchup(103,3,self.row(1),self.row(2),games,'2026-09-25')
        self.assertEqual(result['head_to_head'],{'games':1,'higher_seed_wins':1,'lower_seed_wins':0,'higher_seed_runs':5,'lower_seed_runs':3})
        self.assertEqual(result['lower_seed'],6)
        self.assertEqual(result['higher_seed_team']['home_record'],'45-30')
    def test_missing_splits_or_incomplete_bracket_are_rejected(self):
        with self.assertRaises(ValueError):context.projected_pairs({})
        r=self.row(1);r['records']={}
        with self.assertRaises(ValueError):context.record(r)
    def test_missing_archive_is_reported_with_limited_search_scope(self):
        with TemporaryDirectory() as td:
            result=context.archive_context(td,{'higher_seed_team':{'team':'Cubs'},'lower_seed_team':{'team':'Padres'}})
        self.assertEqual(result['matched_forecasts'],[])
        self.assertIn('not indexed',result['scope'])
    def test_wildcard_topic_does_not_turn_instruction_words_into_names(self):
        row={'sport':'MLB','idea':'Talk about MLB wildcard matchups. Use head to head records. Do a paragraph. We do not need predictions.'}
        self.assertTrue(context.applies(row));matched,terms=ideas.context(row,{})
        self.assertEqual(matched,[]);self.assertEqual(terms,('wild card','wildcard','playoff','postseason'))
    def test_exact_payload_fit_preserves_records_and_requested_angle(self):
        packet={'markets':[{'id':'game'}],'reporting':[{'excerpt':'x'*1500} for _ in range(5)],'required_records':'x'*7000}
        assignment={'requested_angle':'Specific request','evidence':packet}
        request=writer.fit_assignment('x'*5000,assignment)
        restored=json.loads(request['input'])
        self.assertEqual(restored['requested_angle'],'Specific request')
        self.assertEqual(restored['evidence']['required_records'],packet['required_records'])
        self.assertEqual(restored['evidence']['markets'],[{'id':'game'}])
        self.assertLessEqual(len(request['input'].encode())+len(request['instructions'].encode()),18000)


class VerifiedUnspentRecoveryTests(unittest.TestCase):
    def test_only_explicitly_verified_zero_cost_attempt_can_be_reserved_again(self):
        from unittest.mock import patch
        import editorial_budget as budget
        now=datetime(2026,9,26,14,tzinfo=timezone.utc)
        with TemporaryDirectory() as td,patch.object(budget,'PATH',Path(td)/'budget.json'):
            entry={'key':'request','at':now.isoformat(),'status':'settled','charge_usd':0}
            budget.PATH.write_text(json.dumps({'entries':[entry]}))
            self.assertFalse(budget.reserve('request',now,9))
            entry['verified_pre_request_failure']='Payload failed before checkpoint and API call'
            for status,charge in [('settled',0.1),('uncertain-reservation-retained',0)]:
                budget.PATH.write_text(json.dumps({'entries':[dict(entry,status=status,charge_usd=charge)]}))
                self.assertFalse(budget.reserve('request',now,9))
            budget.PATH.write_text(json.dumps({'entries':[entry]}))
            self.assertTrue(budget.reserve('request',now,9))
            saved=budget.read()['entries'][0]
            self.assertEqual(saved['prior_attempts'][0]['charge_usd'],0)
            self.assertNotIn('verified_pre_request_failure',saved)
            self.assertFalse(budget.reserve('request',now,9))

if __name__=='__main__':unittest.main()

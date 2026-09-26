from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json,sys,unittest
from unittest.mock import patch
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
    def test_recent_form_uses_completed_results_and_distinct_venue_windows(self):
        def game(pk,day,home,score,state='Final',kind='R',number=1):
            return {'gamePk':pk,'officialDate':day,'gameNumber':number,'gameType':kind,'status':{'abstractGameState':state},
                'teams':{'home':{'team':{'id':home},'score':score[0]},'away':{'team':{'id':2 if home==1 else 1},'score':score[1]}}}
        rows=[game(i,f'2026-09-{i:02}',1 if i%2 else 2,(5,3)) for i in range(1,13)]
        rows += [rows[0],game(13,'2026-09-13',1,(9,1),state='Live'),game(14,'2026-09-14',1,(9,1)),
            game(15,'2026-09-13',1,(9,1),kind='S'),game(16,'2026-09-13',1,(3,3)),game(17,'2026-09-13',1,(None,3))]
        completed=context.completed_games({'dates':[{'games':list(reversed(rows))}]},'2026-09-13')
        self.assertEqual(len(completed),12)
        result=context.recent_form(1,completed)
        self.assertEqual(result,{'games':10,'from':'2026-09-03','through':'2026-09-12','wins':5,'losses':5,'runs_for':40,'runs_against':40,'run_differential':0})
        away=context.recent_form(1,completed,venue='away')
        self.assertEqual((away['games'],away['wins'],away['losses'],away['run_differential']),(6,0,6,-12))
        self.assertNotIn('cover_rate',away)
        with self.assertRaises(ValueError):context.recent_form(999,completed)
        double=[game(22,'2026-09-13',1,(1,5),number=2),game(23,'2026-09-13',1,(5,1),number=1)]
        result=context.recent_form(1,context.completed_games({'dates':[{'games':double}]},'2026-09-13'),limit=1)
        self.assertEqual((result['wins'],result['losses']),(0,1))
    def test_missing_archive_is_reported_with_limited_search_scope(self):
        with TemporaryDirectory() as td:
            result=context.archive_context(td,{'higher_seed_team':{'team':'Cubs'},'lower_seed_team':{'team':'Padres'}})
        self.assertEqual(result['matched_forecasts'],[])
        self.assertIn('not indexed',result['scope'])
    def test_wildcard_topic_does_not_turn_instruction_words_into_names(self):
        row={'sport':'MLB','idea':'Talk about MLB wildcard matchups. Use head to head records. Do a paragraph. We do not need predictions.'}
        self.assertTrue(context.applies(row));matched,terms=ideas.context(row,{})
        self.assertEqual(matched,[]);self.assertEqual(terms,('wild card','wildcard','playoff','postseason'))
    def test_team_playoff_request_does_not_become_opponent_game_preview(self):
        row={'sport':'MLB','idea':'Give me a rundown on the pirates playoff chances.'}
        matched,terms=ideas.context(row,{'markets':[{'id':'today','game':'Pittsburgh Pirates @ Detroit Tigers'}]})
        self.assertTrue(context.applies(row));self.assertEqual(matched,[])
        self.assertIn('playoff picture',terms);self.assertNotIn('tigers',terms)
        pirates=self.row(3);pirates['team']['name']='Pittsburgh Pirates'
        group={'league':{'id':104},'teamRecords':[pirates]}
        self.assertEqual(context.requested_team({'records':[group]},row),(group,pirates))
        self.assertIsNone(context.requested_team({'records':[group]},{'idea':'Are the Tiger teams alive?'}))
    def test_official_status_drives_playoff_outlook_without_inventing_probability(self):
        now=datetime(2026,9,26,16,tzinfo=timezone.utc)
        team=self.row(3,'3','8','5');team['team']['name']='Pirates'
        team.update(gamesPlayed=160,divisionGamesBack='20.0',wildCardGamesBack='6.0',
            eliminationNumber='E',wildCardEliminationNumber='E',clinched=False)
        leader=self.row(1);third=self.row(2,'2','6','3')
        group={'league':{'id':104},'teamRecords':[leader,third,team]};standings={'records':[group]}
        def game(pk,day,state):
            return {'gamePk':pk,'officialDate':day,'gameType':'R','status':{'abstractGameState':state},
                'teams':{'home':{'team':{'id':3,'name':'Pirates'},'score':5},'away':{'team':{'id':4,'name':'Tigers'},'score':2}}}
        history={'dates':[{'date':'2026-09-25','games':[game(1,'2026-09-25','Final')]}]}
        remaining={'dates':[{'date':'2026-09-26','games':[game(2,'2026-09-26','Preview'),game(2,'2026-09-26','Preview')]}]}
        with patch.object(context,'get',return_value=(remaining,'https://statsapi.mlb.com/schedule')):
            result,_=context.playoff_outlook(group,team,standings,history,'2026-09-25',now)
            self.assertEqual(result['postseason_status'],'eliminated')
            self.assertEqual(result['last_10']['wins'],1)
            self.assertEqual(len(result['remaining_scheduled_games']),1)
            self.assertNotIn('playoff_probability',result)
            team['wildCardEliminationNumber']='2'
            result,_=context.playoff_outlook(group,team,standings,history,'2026-09-25',now)
            self.assertEqual(result['postseason_status'],'not_clinched')
            team['clinched']=True
            result,_=context.playoff_outlook(group,team,standings,history,'2026-09-25',now)
            self.assertEqual(result['postseason_status'],'clinched')
            del team['wildCardEliminationNumber']
            with self.assertRaisesRegex(ValueError,'Official playoff status'):context.playoff_outlook(group,team,standings,history,'2026-09-25',now)
    def test_exact_payload_fit_preserves_records_and_requested_angle(self):
        packet={'markets':[{'id':'game'}],'reporting':[{'excerpt':'x'*1500} for _ in range(5)],'required_records':'x'*7000}
        assignment={'requested_angle':'Specific request','evidence':packet}
        request=writer.fit_assignment('x'*5000,assignment)
        restored=json.loads(request['input'])
        self.assertEqual(restored['requested_angle'],'Specific request')
        self.assertEqual(restored['evidence']['required_records'],packet['required_records'])
        self.assertEqual(restored['evidence']['markets'],[{'id':'game'}])
        self.assertLessEqual(len(request['input'].encode())+len(request['instructions'].encode()),18000)
    def test_overview_compaction_preserves_statistics_and_real_forecasts(self):
        recent={'games':10,'wins':7,'losses':3,'runs_for':50,'runs_against':30}
        forecast={'matched_forecasts':[{'game':'Team A @ Team B','model_mean':4.5}], 'scope':'Retained editorial snapshots only'}
        packet={'requested_topic':True,'methods':'Unrelated player-prop methods','model_status':'Daily model',
            'model_validation':{'test_start':'2026-01-01'},'model_summary':{'history_through':'2026-09-25'},
            'markets':[],'model_rows':[],'reporting':[],
            'statistical_context':{'series':[
                {'id':'s1','higher_seed_team':{'last_10':recent},'historical_models':{'matched_forecasts':[],'scope':'Checked archive'}},
                {'id':'s2','historical_models':forecast}]}}
        result=writer.compact(packet)
        self.assertEqual(result['statistical_context']['series'][0]['higher_seed_team']['last_10'],recent)
        self.assertEqual(result['statistical_context']['series'][1]['historical_models'],forecast)
        self.assertNotIn('methods',result)
        self.assertNotIn('historical_models',result['statistical_context']['series'][0])
        self.assertIn('historical_models',packet['statistical_context']['series'][0])


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

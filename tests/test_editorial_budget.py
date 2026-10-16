from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_budget as b
import editorial_writer as w
import editorial_sources as sources

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'budget.json'
        p=patch.object(b,'PATH',self.path);p.start();self.addCleanup(p.stop)
        self.now=datetime(2026,9,22,12,tzinfo=timezone.utc)
    def test_rolling_limit_and_duplicate(self):
        self.assertTrue(b.reserve('a',self.now,b.maximum()))
        self.assertFalse(b.reserve('a',self.now,9))
        self.assertFalse(b.reserve('b',self.now,b.maximum()))
        self.assertTrue(b.reserve('b',self.now+timedelta(days=8),b.maximum()))
    def test_uncertain_calls_keep_reservation(self):
        b.reserve('a',self.now,9);b.settle('a',[],False)
        self.assertEqual(b.used(b.read(),self.now),b.maximum())
        b.settle('a',[{'input_tokens':1000,'output_tokens':500}],True)
        self.assertAlmostEqual(b.used(b.read(),self.now),.0375)
    def test_no_unbounded_tools_or_input(self):
        p=w.payload('test',{},'write')
        self.assertNotIn('tools',p)
        p['tools']=[{'type':'web_search'}]
        with self.assertRaises(ValueError):b.bounds(p,'write')
        with self.assertRaises(ValueError):w.payload('x'*18001,{},'write')
    def test_sources_are_bounded_and_redirects_checked(self):
        self.assertFalse(sources.trusted('https://espn.com.evil.test/news'))
        self.assertFalse(sources.trusted('http://espn.com/news'))
        self.assertEqual(sources.text_content('<script type="application/ld+json">{"articleBody":"Real reporting"}</script>'),'Real reporting')
    def test_sonnet_writes_opus_audits_and_each_is_charged_at_its_rate(self):
        self.assertEqual((w.ed.CFG['writer']['model'],w.audit_model(w.ed.CFG['writer'])),('claude-sonnet-5-5','claude-opus-5-5'))
        root=Path(self.temp.name);docs=root/'docs';state=root/'runs'
        (docs/'editorial').mkdir(parents=True)
        w.ed.write_json(state/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-09-22','excerpt':'Private source excerpt'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        article={'publish':True,'title':'A substantive market analysis headline','excerpt':'A substantial original summary of the matchup and market.','sections':[{'heading':'Context','text':'Experimental '+'Analysis '*149,'source_ids':['a','b']} for _ in range(4)],'sources':[{k:v for k,v in r.items() if k!='excerpt'} for r in reporting],'market_ids':['game','model-game']}
        def response(value):return {'status':'completed','usage':{'input_tokens':1000,'output_tokens':800},'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
        packet={'sport':'MLB','data_readiness':{'ready':True},'markets':[{'id':'game','event_id':'game','game':'Away @ Home','commence_time':'2026-09-23T00:00:00Z'}],'model_rows':[{'id':'model-game','event_id':'game','game':'Away @ Home','model_mean':5.2,'model_version':'v1','model_input_through':'2026-09-21'}]}
        class FixedDate(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,9,22,12,tzinfo=timezone.utc)
        with patch.object(w,'datetime',FixedDate),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w,'evidence',return_value=packet),patch.object(w.reporting,'collect',return_value=reporting),patch.object(b,'checkpoint'),patch.object(w.ed,'render_home'),patch.object(w,'call_api',side_effect=[response(article),response({'pass':True})]) as api:
            w.run(self.now,1)
        self.assertEqual([c.args[0]['model'] for c in api.call_args_list],['claude-sonnet-5-5','claude-opus-5-5'])
        # Articles are public: the writer and the audit both carry the publication-voice rule.
        self.assertTrue(all(w.PUBLIC_VOICE in c.args[0]['instructions'] for c in api.call_args_list))
        (entry,)=b.read()['entries']
        self.assertEqual(entry['status'],'settled')
        self.assertAlmostEqual(entry['charge_usd'],(1000*2.5+800*10+1000*5+800*20)/1e6)
        self.assertLess(b.maximum({'write':'claude-sonnet-5-5','audit':'claude-opus-5-5'}),b.maximum())
    def test_refused_call_settles_at_zero_and_stops_paid_slots(self):
        root=Path(self.temp.name);docs=root/'docs';state=root/'runs'
        (docs/'editorial').mkdir(parents=True)
        w.ed.write_json(state/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-09-22','excerpt':'Private source excerpt'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        packet={'sport':'MLB','data_readiness':{'ready':True},'markets':[{'id':'game','event_id':'game','game':'Away @ Home','commence_time':'2026-09-23T00:00:00Z'}],'model_rows':[{'id':'model-game','event_id':'game','game':'Away @ Home','model_mean':5.2,'model_version':'v1','model_input_through':'2026-09-21'}]}
        class FixedDate(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,9,22,12,tzinfo=timezone.utc)
        refused=w.model_api.NotBilled('Claude HTTP 401 authentication_error')
        with patch.object(w,'datetime',FixedDate),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w,'evidence',return_value=packet),patch.object(w.reporting,'collect',return_value=reporting),patch.object(b,'checkpoint'),patch.object(w.ed,'render_home'),patch.object(w,'call_api',side_effect=refused):
            w.run(self.now,1)
        (entry,)=b.read()['entries']
        self.assertEqual((entry['status'],entry['charge_usd']),('settled',0))
        self.assertTrue(json.loads((state/'2026-09-22.json').read_text())['funding_required'])
    def test_full_publish_and_repeat_without_new_spend(self):
        root=Path(self.temp.name);docs=root/'docs';state=root/'runs'
        (docs/'editorial').mkdir(parents=True)
        w.ed.write_json(state/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-09-22','excerpt':'Private source excerpt'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        article={'publish':True,'title':'A substantive market analysis headline','excerpt':'A substantial original summary of the matchup and market.', 'sections':[{'heading':'Context','text':'Experimental '+'Analysis '*149,'source_ids':['a','b']} for _ in range(4)],'sources':[{k:v for k,v in r.items() if k!='excerpt'} for r in reporting],'market_ids':['game','model-game']}
        def response(value):return {'status':'completed','usage':{'input_tokens':1000,'output_tokens':800},'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
        packet={'sport':'MLB','data_readiness':{'ready':True},'markets':[{'id':'game','event_id':'game','game':'Away @ Home','commence_time':'2026-09-23T00:00:00Z'}],'model_rows':[{'id':'model-game','event_id':'game','game':'Away @ Home','model_mean':5.2,'model_version':'v1','model_input_through':'2026-09-21'}]}
        class FixedDate(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,9,22,12,tzinfo=timezone.utc)
        with patch.object(w,'datetime',FixedDate),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w,'evidence',return_value=packet),patch.object(w.reporting,'collect',return_value=reporting),patch.object(b,'checkpoint'),patch.object(w.ed,'render_home'),patch.object(w,'call_api',side_effect=[response(article),response({'pass':True})]) as api:
            w.run(self.now,1)
            self.assertEqual(api.call_count,2)
            public=json.loads(next((docs/'editorial/evidence').glob('*.json')).read_text())
            self.assertNotIn('excerpt',public['reporting'][0])
            catalog=json.loads((docs/'editorial/published.json').read_text())
            self.assertEqual(catalog[0]['featured_until'],'2026-09-23T00:00:00+00:00')
            w.run(self.now,1)
            self.assertEqual(api.call_count,2)
    def story(self,replies):
        """Run one MLB story against these model replies; return the API mock, slot and docs path."""
        root=Path(self.temp.name);docs=root/'docs';state=root/'runs'
        (docs/'editorial').mkdir(parents=True)
        w.ed.write_json(state/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-09-22','excerpt':'Private source excerpt'} for s,url in [('s1','https://mlb.com/news/a'),('s2','https://www.espn.com/mlb/b')]]
        packet={'sport':'MLB','data_readiness':{'ready':True},'markets':[{'id':'game','event_id':'game','game':'Away @ Home','commence_time':'2026-09-23T00:00:00Z'}],'model_rows':[{'id':'model-game','event_id':'game','game':'Away @ Home','model_mean':5.2,'model_version':'v1','model_input_through':'2026-09-21'}]}
        def article(title):return {'publish':True,'title':title,'excerpt':'A substantial original summary of the matchup and market.','sections':[{'heading':'Context','text':'Experimental '+'Analysis '*149,'source_ids':['s1','s2']} for _ in range(4)],'sources':[{k:v for k,v in r.items() if k!='excerpt'} for r in reporting],'market_ids':['game','model-game']}
        def response(value):return {'status':'completed','usage':{'input_tokens':1000,'output_tokens':800},'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
        class FixedDate(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,9,22,12,tzinfo=timezone.utc)
        replies=[response(article(r) if isinstance(r,str) else r) for r in replies]
        with patch.object(w,'datetime',FixedDate),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w,'evidence',return_value=packet),patch.object(w.reporting,'collect',return_value=reporting),patch.object(b,'checkpoint'),patch.object(w.ed,'render_home'),patch.object(w,'call_api',side_effect=replies) as api,patch('builtins.print'):
            w.run(self.now,1)
        return api,json.loads((state/'2026-09-22.json').read_text())['slots']['0-mlb'],docs
    def test_rejected_draft_gets_one_revision_against_the_audit_findings(self):
        findings='Section 2 says "our evidence states" the lineup is unknown; state it as of the quote time.'
        api,slot,docs=self.story(['A first market analysis headline',{'pass':False,'reason':findings},'A revised market analysis headline',{'pass':True,'reason':'Checks hold.'}])
        calls=[c.args[0] for c in api.call_args_list]
        self.assertEqual([c['model'] for c in calls],['claude-sonnet-5-5','claude-opus-5-5','claude-sonnet-5-5','claude-opus-5-5'])
        revision=json.loads(calls[2]['input'])
        self.assertEqual(revision['audit_findings'],findings)
        self.assertEqual(revision['rejected_draft']['title'],'A first market analysis headline')
        self.assertIn('rejected_draft',calls[2]['instructions'])
        self.assertEqual(calls[2]['max_output_tokens'],b.LIMITS['revise'][1])
        # The fresh audit sees the revised draft and the same evidence.
        self.assertEqual(json.loads(calls[3]['input'])['article']['title'],'A revised market analysis headline')
        self.assertEqual(json.loads(calls[3]['input'])['evidence'],json.loads(calls[1]['input'])['evidence'])
        self.assertEqual((slot['status'],slot['first_audit_reason'],slot['audit_reason']),('published',findings,'Checks hold.'))
        self.assertIn('revision_usage',slot);self.assertIn('revision_review_usage',slot)
        catalog=json.loads((docs/'editorial/published.json').read_text())
        self.assertEqual(catalog[0]['title'],'A revised market analysis headline')
        (entry,)=b.read()['entries']
        self.assertAlmostEqual(entry['charge_usd'],2*(1000*2.5+800*10+1000*5+800*20)/1e6)
    def test_second_rejection_stands_without_a_third_attempt(self):
        api,slot,docs=self.story(['A first market analysis headline',{'pass':False,'reason':'First problem.'},'A revised market analysis headline',{'pass':False,'reason':'Second problem.'}])
        self.assertEqual(api.call_count,4)
        self.assertEqual((slot['status'],slot['first_audit_reason'],slot['reason']),('skipped','First problem.','Factual audit did not pass: Second problem.'))
        self.assertFalse((docs/'editorial/published.json').exists())
    def test_story_reservation_covers_a_revision(self):
        models={'write':'claude-sonnet-5-5','revise':'claude-sonnet-5-5','audit':'claude-opus-5-5'}
        self.assertAlmostEqual(b.maximum(models,b.STORY),b.maximum(models)+b.maximum(models,('revise','audit')),places=5)
        self.assertLess(b.maximum(models,b.STORY),w.ed.CFG['writer']['weekly_budget_usd']/10)
    def test_writer_uses_configured_effort_and_audit_stays_low(self):
        with patch.dict(w.ed.CFG['writer'],{'reasoning_effort':'medium'}):
            self.assertEqual(w.payload('x',{},'write')['reasoning']['effort'],'medium')
            self.assertEqual(w.payload('x',{},'audit')['reasoning']['effort'],'low')
        self.assertEqual(w.payload('x',{},'write')['max_output_tokens'],b.LIMITS['write'][1])
        # Claude's thinking shares the ceiling with the short JSON verdict.
        self.assertGreaterEqual(w.payload('x',{},'audit')['max_output_tokens'],4000)
    def test_requested_length_clears_validation_floor(self):
        # Drafts land near the bottom of the requested range, so it must sit above the floor.
        import re
        for prompt,floor in ((w.PROMPT,550),(w.OVERVIEW_PROMPT,400),(w.TEAM_PLAYOFF_PROMPT,400)):
            low=int(re.search(r'(\d{3})–\d{3} word',prompt).group(1))
            self.assertGreaterEqual(low,floor+80)
            self.assertIn(f'under {floor} words are rejected',prompt)
    def test_checkpoint_retries_when_another_job_pushed_first(self):
        results=iter([0,0,0,0, 0,1, 0,0])  # config x2, add, commit; first push loses the race; retry wins
        calls=[]
        def run(command,**kw):
            calls.append(command[1]);return type('R',(),{'returncode':next(results)})()
        with patch.dict('os.environ',{'GITHUB_ACTIONS':'true'}),patch.object(b,'PATH',b.ed.ROOT/'docs/editorial/budget.json'),patch.object(b.subprocess,'run',side_effect=run),patch.object(b.time,'sleep'):
            b.checkpoint(b.ed.ROOT/'docs/editorial/runs/x.json')
        self.assertEqual(calls[-4:],['pull','push','pull','push'])
        results=iter([0,0,0,0]+[0,1]*3)
        with patch.dict('os.environ',{'GITHUB_ACTIONS':'true'}),patch.object(b,'PATH',b.ed.ROOT/'docs/editorial/budget.json'),patch.object(b.subprocess,'run',side_effect=lambda c,**k:type('R',(),{'returncode':next(results)})()),patch.object(b.time,'sleep'):
            with self.assertRaises(RuntimeError):b.checkpoint(b.ed.ROOT/'docs/editorial/runs/x.json')
    def test_checkpoint_failure_prevents_paid_request(self):
        root=Path(self.temp.name)
        w.ed.write_json(root/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        with patch.object(w,'STATE',root),patch.object(w,'evidence',return_value={'data_readiness':{'ready':True},'markets':[{'id':'q1'}]}),patch.object(w.reporting,'collect',return_value=[{'excerpt':'source'}]),patch.object(b,'checkpoint',side_effect=RuntimeError('Checkpoint failed')),patch.object(w.ed,'render_home'),patch.object(w,'call_api') as api:
            w.run(self.now,1);api.assert_not_called()
        self.assertEqual(b.used(b.read(),self.now),0)

if __name__=='__main__':unittest.main()

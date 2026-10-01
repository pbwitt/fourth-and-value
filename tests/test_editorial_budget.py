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
    def test_full_publish_and_repeat_without_new_spend(self):
        root=Path(self.temp.name);docs=root/'docs';state=root/'runs'
        (docs/'editorial').mkdir(parents=True)
        w.ed.write_json(state/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-09-22','excerpt':'Private source excerpt'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        article={'publish':True,'title':'A substantive market analysis headline','excerpt':'A substantial original summary of the matchup and market.', 'sections':[{'heading':'Context','text':'Analysis '*150,'source_ids':['a','b']} for _ in range(4)],'sources':[{k:v for k,v in r.items() if k!='excerpt'} for r in reporting],'market_ids':['game','model-game']}
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
    def shadow_run(self,responses,day=datetime(2026,10,3,12,tzinfo=timezone.utc),checkpoint=None):
        """Run one daily story with the Sol comparison active; returns (root, api mock)."""
        root=Path(self.temp.name);docs=root/'docs';state=root/'runs'
        (docs/'editorial').mkdir(parents=True,exist_ok=True)
        w.ed.write_json(state/f'{day.date()}.json',{'allocation':[['MLB','news-market']],'slots':{}})
        reporting=[{'id':s,'title':'Report','url':url,'published_at':str(day.date()),'excerpt':'Private source excerpt'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        packet={'sport':'MLB','data_readiness':{'ready':True},'markets':[{'id':'game','event_id':'game','game':'Away @ Home','commence_time':'2026-10-04T00:00:00Z'}],'model_rows':[{'id':'model-game','event_id':'game','game':'Away @ Home','model_mean':5.2,'model_version':'v1','model_input_through':'2026-10-02'}]}
        class FixedDate(datetime):
            @classmethod
            def now(cls,tz=None):return day
        writer=dict(w.ed.CFG['writer'],shadow={'model':'gpt-6.1-sol','start':'2026-10-02','until':'2026-10-09'})
        with patch.dict(w.ed.CFG,{'writer':writer}),patch.object(w,'datetime',FixedDate),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w,'TRIAL',root/'trial'),patch.object(w,'evidence',return_value=packet),patch.object(w.reporting,'collect',return_value=reporting),patch.object(b,'checkpoint',side_effect=checkpoint),patch.object(w.ed,'render_home'),patch.object(w,'call_api',side_effect=responses) as api:
            w.run(day,1)
            w.run(day,1)
        return root,api,reporting
    def article(self,reporting,title='A substantive market analysis headline'):
        return {'publish':True,'title':title,'excerpt':'A substantial original summary of the matchup and market.','sections':[{'heading':'Context','text':'Analysis '*150,'source_ids':['a','b']} for _ in range(4)],'sources':[{k:v for k,v in r.items() if k!='excerpt'} for r in reporting],'market_ids':['game','model-game']}
    @staticmethod
    def response(value):return {'status':'completed','usage':{'input_tokens':1000,'output_tokens':800},'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
    def test_shadow_draft_is_audited_but_never_published(self):
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-10-03'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        calls=[self.response(self.article(reporting)),self.response({'pass':True}),
               self.response(self.article(reporting,'A different comparison draft headline')),self.response({'pass':False,'reason':'Unsupported claim'})]
        root,api,_=self.shadow_run(calls)
        self.assertEqual([c.args[0]['model'] for c in api.call_args_list],['gpt-6-astra','gpt-6-astra','gpt-6.1-sol','gpt-6-astra'])
        catalog=json.loads((root/'docs/editorial/published.json').read_text())
        self.assertEqual([a['title'] for a in catalog],['A substantive market analysis headline'])
        (key,slot),=json.loads((root/'runs/2026-10-03.json').read_text())['slots'].items()
        self.assertEqual((slot['status'],slot['shadow']['status']),('published','audit_failed'))
        record=json.loads((root/f'trial/2026-10-03-{key}.json').read_text())
        self.assertEqual(record['astra']['status'],'published')
        self.assertEqual(record['shadow']['article']['title'],'A different comparison draft headline')
        self.assertNotIn('Private source excerpt',json.dumps(record))
        entries={e['key']:e for e in b.read()['entries']}
        self.assertEqual(entries[f'2026-10-03-{key}-shadow']['status'],'settled')
        self.assertAlmostEqual(entries[f'2026-10-03-{key}-shadow']['charge_usd'],1000*2.5e-6+800*10e-6+1000*12.5e-6+800*50e-6)
    def test_shadow_failure_cannot_change_the_edition(self):
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-10-03'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        root,api,_=self.shadow_run([self.response(self.article(reporting)),self.response({'pass':True}),StopIteration()])
        (slot,)=json.loads((root/'runs/2026-10-03.json').read_text())['slots'].values()
        self.assertEqual((slot['status'],slot['shadow']['status']),('published','failed'))
        self.assertEqual(api.call_count,3)
    def test_no_shadow_spend_without_a_durable_checkpoint(self):
        root,api,_=self.shadow_run([],checkpoint=RuntimeError('Checkpoint failed'))
        api.assert_not_called()
        self.assertEqual(b.used(b.read(),datetime(2026,10,3,12,tzinfo=timezone.utc)),0)
    def test_shadow_never_takes_budget_a_real_story_needs(self):
        day=datetime(2026,10,3,12,tzinfo=timezone.utc)
        # Leave room for this story and a little more, but not for a comparison plus another story.
        data={'version':2,'entries':[{'key':'earlier','at':day.isoformat(),'status':'settled','charge_usd':round(9-b.maximum()-0.3,6)}]}
        w.ed.write_json(self.path,data)
        reporting=[{'id':s,'title':'Report','url':url,'published_at':'2026-10-03'} for s,url in [('a','https://mlb.com/news/a'),('b','https://www.espn.com/mlb/b')]]
        root,api,_=self.shadow_run([self.response(self.article(reporting)),self.response({'pass':True})])
        self.assertEqual([c.args[0]['model'] for c in api.call_args_list],['gpt-6-astra','gpt-6-astra'])
        (slot,)=json.loads((root/'runs/2026-10-03.json').read_text())['slots'].values()
        self.assertEqual(slot['status'],'published');self.assertNotIn('shadow',slot)
    def test_shadow_window_and_rates(self):
        cfg={'shadow':{'model':'gpt-6.1-sol','start':'2026-10-02','until':'2026-10-09'}}
        self.assertEqual(w.shadow_model(cfg,'2026-10-05',None),'gpt-6.1-sol')
        self.assertIsNone(w.shadow_model(cfg,'2026-10-10',None))
        self.assertIsNone(w.shadow_model(cfg,'2026-10-01',None))
        self.assertIsNone(w.shadow_model(cfg,'2026-10-05','idea'))
        self.assertIsNone(w.shadow_model({'shadow':{'model':'unknown','start':'2026-10-02','until':'2026-10-09'}},'2026-10-05',None))
        self.assertLess(b.maximum({'write':'gpt-6.1-sol'}),b.maximum())
        self.assertLess(b.bounds(w.payload('test',{},'write','gpt-6.1-sol'),'write'),b.bounds(w.payload('test',{},'write'),'write'))
    def test_checkpoint_failure_prevents_paid_request(self):
        root=Path(self.temp.name)
        w.ed.write_json(root/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        with patch.object(w,'STATE',root),patch.object(w,'evidence',return_value={'data_readiness':{'ready':True},'markets':[{'id':'q1'}]}),patch.object(w.reporting,'collect',return_value=[{'excerpt':'source'}]),patch.object(b,'checkpoint',side_effect=RuntimeError('Checkpoint failed')),patch.object(w.ed,'render_home'),patch.object(w,'call_api') as api:
            w.run(self.now,1);api.assert_not_called()
        self.assertEqual(b.used(b.read(),self.now),0)

if __name__=='__main__':unittest.main()

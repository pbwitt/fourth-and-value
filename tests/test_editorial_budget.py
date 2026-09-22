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
        article={'publish':True,'title':'A substantive market analysis headline','excerpt':'A substantial original summary of the matchup and market.', 'sections':[{'heading':'Context','text':'Analysis '*150,'source_ids':['a','b']} for _ in range(4)],'sources':[{k:v for k,v in r.items() if k!='excerpt'} for r in reporting],'market_ids':['game']}
        def response(value):return {'status':'completed','usage':{'input_tokens':1000,'output_tokens':800},'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
        packet={'markets':[{'id':'game','commence_time':'2026-09-23T00:00:00Z'}],'model_rows':[]}
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
    def test_checkpoint_failure_prevents_paid_request(self):
        root=Path(self.temp.name)
        w.ed.write_json(root/'2026-09-22.json',{'allocation':[['MLB','news-market']],'slots':{}})
        with patch.object(w,'STATE',root),patch.object(w,'evidence',return_value={}),patch.object(w.reporting,'collect',return_value=[{'excerpt':'source'}]),patch.object(b,'checkpoint',side_effect=RuntimeError('Checkpoint failed')),patch.object(w.ed,'render_home'),patch.object(w,'call_api') as api:
            w.run(self.now,1);api.assert_not_called()
        self.assertEqual(b.used(b.read(),self.now),0)

if __name__=='__main__':unittest.main()

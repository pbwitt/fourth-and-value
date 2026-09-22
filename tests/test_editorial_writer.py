import copy
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_writer as w

class WriterGuards(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,9,22,16,tzinfo=timezone.utc)
        self.urls=['https://mlb.com/news/a','https://apnews.com/article/b']
        self.response={'output':[{'action':{'sources':[{'url':u} for u in self.urls]}}]}
        self.article={'publish':True,'title':'A specific and substantial headline','excerpt':'A sufficiently substantial summary for this sports story.',
            'sections':[{'heading':'Context','text':'word '*150,'source_ids':['a']} for _ in range(4)],
            'sources':[{'id':i,'url':u,'published_at':'2026-09-22'} for i,u in zip(['a','b'],self.urls)],'market_ids':['q1']}
        self.packet={'markets':[{'id':'q1'}],'model_rows':[]}
    def test_diverse_allocation(self):
        slots=w.slots([{'sport':'NFL'},{'sport':'MLB'}])
        self.assertEqual(len(slots),6)
        self.assertEqual(sum(s=='MLB' for s,_ in slots),2)
        self.assertEqual({s for s,_ in slots},{'MLB','NFL','NBA','NHL'})
    def test_valid(self):
        self.assertEqual(w.validate(self.article,self.response,self.packet,self.now),600)
    def test_invented_market_rejected(self):
        self.article['market_ids']=['fake']
        with self.assertRaises(ValueError):w.validate(self.article,self.response,self.packet,self.now)
    def test_unvisited_source_rejected(self):
        self.article['sources'][0]['url']='https://mlb.com/invented'
        with self.assertRaises(ValueError):w.validate(self.article,self.response,self.packet,self.now)
    def test_future_reporting_rejected(self):
        self.article['sources'][0]['published_at']='2026-09-23'
        with self.assertRaises(ValueError):w.validate(self.article,self.response,self.packet,self.now)
    def test_missing_citations_rejected(self):
        self.article['sections'][0]['source_ids']=[]
        with self.assertRaises(ValueError):w.validate(self.article,self.response,self.packet,self.now)
    def test_stale_reporting_rejected(self):
        for s in self.article['sources']:s['published_at']='2026-08-01'
        with self.assertRaises(ValueError):w.validate(self.article,self.response,self.packet,self.now)
    def test_old_price_is_removed_from_recent_model_context(self):
        from unittest.mock import patch
        board={'model_checked_at':'2026-09-22T12:00:00Z','rows':[{
            'event_id':'game','market':'pitcher_strikeouts','player':'Example',
            'game':'Example game','commence_time':'2026-09-23T00:00:00Z',
            'quoted_at':'2026-09-22T08:00:00Z','is_model_pick':True,
            'model_mean':5.3,'price':-110,'model_ev_pct':12}]}
        def fake_load(path,default):
            return {} if path==w.ed.PUBLIC/'latest.json' else board
        with patch.object(w,'load',side_effect=fake_load):
            packet=w.evidence('MLB',self.now)
            self.assertEqual(packet['model_rows'],[])
            self.assertEqual(len(packet['model_references']),1)
            self.assertNotIn('price',packet['model_references'][0])
            self.assertNotIn('model_ev_pct',packet['model_references'][0])
            board['model_checked_at']='2026-09-20T12:00:00Z'
            self.assertEqual(w.evidence('MLB',self.now)['model_references'],[])
    def test_kill_switch_prevents_paid_calls(self):
        from unittest.mock import patch
        with patch.dict(w.ed.CFG,{'writing_enabled':False}),patch.object(w,'call_api') as api:
            w.run(self.now)
            api.assert_not_called()
    def test_rerun_does_not_make_paid_calls(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            with patch.object(w,'STATE',Path(directory)),patch.object(w,'call_api') as api,patch.object(w.ed,'render_home'),patch('builtins.print'):
                day='2026-09-22'
                w.ed.write_json(Path(directory)/(day+'.json'),{'allocation':[['NFL','news-market']], 'slots':{'0-nfl':{'status':'started'}}})
                w.run(self.now)
                api.assert_not_called()

if __name__=='__main__':unittest.main()

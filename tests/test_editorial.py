import importlib.util
from datetime import datetime, timezone, timedelta
import unittest
import json
import tempfile
from unittest.mock import patch, Mock
from pathlib import Path

spec=importlib.util.spec_from_file_location('editorial',Path(__file__).resolve().parents[1]/'scripts/editorial.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NOW=datetime(2026,9,22,12,tzinfo=timezone.utc)

def event():
    return {'id':'abc','commence_time':(NOW+timedelta(hours=5)).isoformat(),'away_team':'A','home_team':'B',
            'bookmakers':[{'key':k,'last_update':NOW.isoformat(),'markets':[{'key':'totals','outcomes':[{'name':'Over','point':p,'price':-110},{'name':'Under','point':p,'price':-110}]}]} for k,p in [('one',40.5),('two',41.5)]]}

class EditorialTests(unittest.TestCase):
    def test_private_publication_has_stable_timestamp_for_homepage_order(self):
        row=dict(id='45981557-219d-4866-b015-56c1aa2c1933',status='publishing',
                 approved_hash='approved',approved_by='editor',user_id='reader',
                 updated_at=NOW.isoformat(),publish_on=NOW.date().isoformat(),
                 title='New featured story',body='Supported analysis. '*20,byline='Author',
                 sources='https://www.mlb.com/',kind='analysis',sport='MLB',featured=True)
        queue=Mock(ok=True,status_code=200);queue.json.return_value=[row]
        user=Mock(ok=True);user.json.return_value={'app_metadata':{'fv_editor':True}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'DOCS',Path(tmp)), patch.dict(m.os.environ,{'SUPABASE_URL':'https://example.test','SUPABASE_SERVICE_ROLE_KEY':'test'}), patch.object(m.requests,'get',side_effect=[queue,user,queue,user]):
            receipt=Path(tmp)/'receipt.json'
            m.publish_approved(NOW,receipt)
            catalog_path=Path(tmp)/'editorial/published.json'
            article=json.loads(catalog_path.read_text())[0]
            self.assertEqual(article['published_at'],NOW.isoformat())
            earlier=dict(article,title='Earlier automated story',published_at=(NOW-timedelta(hours=1)).isoformat())
            self.assertEqual(sorted([earlier,article],key=lambda a:(a['date'],a.get('published_at','')),reverse=True)[0]['title'],row['title'])
            m.publish_approved(NOW+timedelta(hours=1),receipt)
            self.assertEqual(json.loads(catalog_path.read_text())[0]['published_at'],NOW.isoformat())

    def test_feature_expiry_and_opinion_separation(self):
        a={'kind':'Analysis','date':'2026-09-22','featured_until':NOW.isoformat()}
        self.assertFalse(m.featured_now(a,NOW))
        self.assertTrue(m.featured_now(a,NOW-timedelta(seconds=1)))
        a.pop('featured_until')
        self.assertFalse(m.featured_now(a,NOW+timedelta(days=4)))
        a['kind']='Opinion'
        self.assertFalse(m.featured_now(a,NOW))
    def test_started_and_stale_quotes_excluded(self):
        e=event();e['commence_time']=(NOW-timedelta(seconds=1)).isoformat()
        self.assertEqual(m.summarize_events('NFL',[e],NOW,{}),[])
        e=event();e['bookmakers'][0]['last_update']=(NOW-timedelta(hours=7)).isoformat()
        self.assertEqual(m.summarize_events('NFL',[e],NOW,{}),[])
    def test_unpaired_line_excluded(self):
        e=event();e['bookmakers'][0]['markets'][0]['outcomes'][1]['point']=42.5
        self.assertEqual(m.summarize_events('NFL',[e],NOW,{}),[])
    def test_change_uses_matched_books_not_composition(self):
        previous={'games':[{'id':'abc','sport':'NFL','books':{'one':40.5,'two':41.5,'gone':80}}]}
        g=m.summarize_events('NFL',[event()],NOW,previous)[0]
        self.assertEqual(g['change'],0)
        self.assertEqual(g['median'],41)
        self.assertEqual(g['minimum'],40.5)
    def test_first_snapshot_not_an_opener(self):
        g=m.summarize_events('NFL',[event()],NOW,{})[0]
        self.assertIsNone(g['change']);self.assertIn('First comparable',g['change_label'])
    def test_stale_homepage_does_not_promote_prices(self):
        g=m.summarize_events('NFL',[event()],NOW,{})
        c=m.context({'generated_at':(NOW-timedelta(hours=7)).isoformat(),'games':g},NOW)
        self.assertEqual(c['cards'],[])
    def test_movement_card_names_books_and_prices(self):
        g=m.summarize_events('NFL',[event()],NOW,{'generated_at':(NOW-timedelta(hours=1)).isoformat(),'games':[{'sport':'NFL','id':'abc','books':{'one':39.5,'two':40.5}}]})
        cards=m.market_cards(g)
        self.assertEqual(len(cards),1)
        self.assertEqual(cards[0]['kind'],'Movement')
        self.assertIn('up 1',cards[0]['text'])
        self.assertIn('one · Over 40.5 (-110)',cards[0]['prices'][0])
        self.assertIn('two · Under 41.5 (-110)',cards[0]['prices'][1])
    def test_equal_lines_are_not_labeled_disagreement(self):
        e=event()
        for book in e['bookmakers']:
            for out in book['markets'][0]['outcomes']:out['point']=41.5
        cards=m.market_cards(m.summarize_events('NFL',[e],NOW,{}))
        self.assertEqual(cards[0]['kind'],'Next up')
        self.assertIn('All 2 books',cards[0]['text'])
    def test_editorial_content_is_escaped(self):
        html=m.ENV.get_template('article.html').render(title='<script>alert(1)</script>',paragraphs=['<img src=x onerror=alert(1)>'],links=[])
        self.assertNotIn('<script>alert',html);self.assertIn('&lt;img',html)
    def test_sources_cannot_be_script_urls(self):
        for url in ['javascript:alert(1)','http://example.com','https://user:secret@example.com']:
            self.assertFalse(m.safe_url(url))
        self.assertTrue(m.safe_url('https://www.nfl.com/news/example'))

if __name__=='__main__':unittest.main()

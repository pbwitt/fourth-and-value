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
    def test_contributor_needs_explicit_editor_publication_authorization(self):
        base=dict(id='45981557-219d-4866-b015-56c1aa2c1933',status='approved',
                  requires_review=True,approved_hash='fingerprint',approved_by='editor',
                  user_id='reader',updated_at=NOW.isoformat(),publish_on=None,
                  title='Contributor opinion',body='Completed opinion. '*20,byline='Contributor',
                  sources='',kind='opinion',sport='NFL',featured=True)
        for updates in ({'approved_by':None},{'approved_hash':None},{'status':'review'},
                        {'status':'submitted'},{'status':'archived'}):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as tmp:
                queue=Mock(ok=True,status_code=200);queue.json.return_value=[dict(base,**updates)]
                with patch.object(m,'DOCS',Path(tmp)), patch.dict(m.os.environ,{'SUPABASE_URL':'https://example.test','SUPABASE_SERVICE_ROLE_KEY':'test'}), patch.object(m.requests,'get',return_value=queue), patch.object(m.requests,'patch') as claim:
                    m.publish_approved(NOW,Path(tmp)/'receipt.json')
                claim.assert_not_called()
                self.assertFalse((Path(tmp)/'editorial/published.json').exists())
                self.assertFalse((Path(tmp)/'editorial/articles').exists())

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
    def test_home_lead_prefers_newest_featured_piece(self):
        story=lambda i,url,**k:{'title':f'S{i}','url':url,'featured':True,'kind':'Analysis',**k}
        fallback={'url':'/briefing/'}
        articles=[story(i,f'/editorial/articles/{i}.html') for i in range(4)]
        self.assertEqual(m.home_lead(articles,fallback),articles[0])
        self.assertEqual(m.home_lead([dict(articles[0],featured=False)]+articles[1:],fallback),articles[1])
        self.assertEqual(m.home_lead([],fallback),fallback)
        # Features exclude the lead and opinion, and top up with older analysis when thin.
        older=[story(9,'/blog/older.html',featured=None),story(8,'/editorial/opinion.html',kind='Opinion')]
        features=m.home_features(articles[:2],articles[:2]+older,articles[0])
        self.assertEqual([a['url'] for a in features],[articles[1]['url'],'/blog/older.html'])
    def test_opinion_promotion_expires_without_entering_analysis(self):
        opinion=dict(title='Opinion',kind='Opinion',date='2026-09-22',
                     url='/editorial/articles/opinion.html',featured=True,
                     featured_until=(NOW+timedelta(days=3)).isoformat())
        self.assertEqual(m.featured_opinions([opinion],NOW),[opinion])
        self.assertFalse(m.featured_now(opinion,NOW))
        self.assertEqual(m.featured_opinions([opinion],NOW+timedelta(days=3)),[])
        self.assertEqual(m.featured_opinions([dict(opinion,featured=False)],NOW),[])
        self.assertEqual(m.featured_opinions([dict(opinion,featured_until=None)],NOW),[])
        older=dict(opinion,title='Older',url='/editorial/older.html',featured=False,featured_until=None,date='2026-09-01')
        take,others=m.home_take([opinion,older],NOW)
        self.assertEqual((take,others),(opinion,[older]))
        take,others=m.home_take([older,opinion],NOW+timedelta(days=3))
        self.assertEqual((take,others),(older,[opinion]),'after the feature ends the newest opinion takes the slot')
        self.assertEqual(m.home_take([],NOW),(None,[]))
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
    def test_market_rows_give_numbers_moves_first(self):
        moved=m.summarize_events('NFL',[event()],NOW,{'generated_at':(NOW-timedelta(hours=1)).isoformat(),'games':[{'sport':'NFL','id':'abc','books':{'one':39.5,'two':40.5}}]})
        rows=m.market_rows(moved)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['change'],'Total 40 → 41')
        self.assertEqual(rows[0]['detail'],'since 7:00 AM ET')
        self.assertEqual(rows[0]['over'],{'text':'O 40.5 (-110)','book':'one'})
        self.assertEqual(rows[0]['under'],{'text':'U 41.5 (-110)','book':'two'})
        split=m.market_rows(m.summarize_events('NFL',[event()],NOW,{}))[0]
        self.assertEqual((split['change'],split['detail']),('Books split 40.5 to 41.5','a 1-point gap'))
        e=event()
        for book in e['bookmakers']:
            for out in book['markets'][0]['outcomes']:out['point']=41.5
        same=m.market_rows(m.summarize_events('NFL',[e],NOW,{}))[0]
        self.assertEqual((same['change'],same['detail']),('Total 41.5 at all 2 books','only the prices differ'))
        self.assertEqual(m.market_rows([]),[])
    def test_rundown_names_lowest_and_highest_line_books(self):
        e=event()
        e['bookmakers'].append({'key':'three','title':'Three','last_update':NOW.isoformat(),'markets':[{'key':'totals','outcomes':[{'name':'Over','point':40.5,'price':-105},{'name':'Under','point':40.5,'price':-115}]}]})
        g=m.summarize_events('NFL',[e],NOW,{})
        c=m.context({'generated_at':NOW.isoformat(),'games':g},NOW)
        row=c['games'][0]
        self.assertEqual((row['low']['label'],row['low']['line'],row['low']['over']),('Three',40.5,'-105'))
        self.assertEqual((row['high']['label'],row['high']['line'],row['high']['under']),('two',41.5,'-110'))
        self.assertIn('Book prices pulled Sep 22, 8:00 AM ET',c['pulled_label'])
        html=m.ENV.get_template('briefing.html').render(**c,title='t',url='/',evidence_url='/',live_picks=False)
        self.assertNotIn('matched books',html);self.assertIn('Lowest line',html)
    def test_lone_book_total_does_not_set_the_range(self):
        e=event()
        e['bookmakers']=[{'key':k,'title':k.title(),'last_update':NOW.isoformat(),'markets':[{'key':'totals','outcomes':[
            {'name':'Over','point':p,'price':o},{'name':'Under','point':p,'price':u}]}]}
            for k,p,o,u in [('a',5.5,-135,114),('b',5.5,-140,120),('c',6.0,-110,-110),('d',6.0,-108,-108),('fan',6.5,115,-140)]]
        g=m.summarize_events('NHL',[e],NOW,{})
        c=m.context({'generated_at':NOW.isoformat(),'games':g},NOW,model_totals={})
        row=c['games'][0]
        self.assertEqual((row['high']['label'],row['high']['line']),('D',6.0))
        self.assertEqual([(x['label'],x['line']) for x in row['high']['lone']],[('Fan',6.5)])
        self.assertEqual((row['low']['label'],row['low']['line'],row['low']['lone']),('A',5.5,[]))
        self.assertIn('Highest under total: D · Under 6 (-108)',c['cards'][0]['prices'][1])
        html=m.ENV.get_template('briefing.html').render(**c,title='t',url='/',evidence_url='/',live_picks=False)
        self.assertIn('Only Fan: 6.5 (O +115 / U -140)',html)
        # Three or fewer books keep the full range.
        three=m.summarize_events('NFL',[event()],NOW,{})
        self.assertEqual(m.context({'generated_at':NOW.isoformat(),'games':three},NOW,model_totals={})['games'][0]['high']['line'],41.5)
    def test_price_movement_summary_is_fresh_and_observed_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'mlb/data').mkdir(parents=True);(root/'nhl/data').mkdir(parents=True)
            summary=dict(picks=5,observed=4,same_line=3,same_line_beat=2,average_probability_move=.021,line_moves=1,line_moves_favorable=1)
            (root/'mlb/data/line-movement.json').write_text(json.dumps(dict(updated_at=NOW.isoformat(),summary=summary)))
            (root/'nhl/data/line-movement.json').write_text(json.dumps(dict(updated_at=(NOW-timedelta(days=4)).isoformat(),summary=summary)))
            rows=m.line_movement(NOW,root)
        self.assertEqual([(r['sport'],r['beat'],r['same_line'],r['average']) for r in rows],[('MLB',2,3,'+2.1 pp')])
        html=m.ENV.get_template('briefing.html').render(games=[],movement=rows,coverage={},news=[],sports=[],title='t',url='/',evidence_url='/',live_picks=False)
        self.assertIn('Price movement after our picks',html);self.assertIn('2 of 3',html)
    def test_rundown_shows_fresh_nhl_model_total_only(self):
        board={'status':'ready','model_error':None,'model_prediction_at':(NOW-timedelta(hours=1)).isoformat(),'model_version':'nhl-test',
               'rows':[{'event_id':'abc','player':'','projected_home_reg_goals':3.0,'projected_away_reg_goals':3.0},
                       {'event_id':'abc','player':'','projected_home_reg_goals':9.0,'projected_away_reg_goals':9.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'nhl.json';path.write_text(json.dumps(board))
            totals=m.nhl_model_totals(NOW,path)
            # Regulation ties add exactly one settlement goal: P(tie | Poisson 3,3) is about 0.1667.
            self.assertAlmostEqual(totals['abc']['total'],6.1667,places=3)
            self.assertEqual(m.nhl_model_totals(NOW+timedelta(hours=37),path),{})
            path.write_text(json.dumps(dict(board,model_error='Independent model unavailable')))
            self.assertEqual(m.nhl_model_totals(NOW,path),{})
        for sport,shown in [('NHL',True),('NFL',False)]:
            g=m.summarize_events(sport,[event()],NOW,{})
            c=m.context({'generated_at':NOW.isoformat(),'games':g},NOW,model_totals=totals)
            html=m.ENV.get_template('briefing.html').render(**c,title='t',url='/',evidence_url='/',live_picks=False)
            self.assertIn('<th>Model total</th>',html)
            self.assertEqual('<strong>6.17</strong>' in html,shown)
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

class NewsTests(unittest.TestCase):
    API='https://site.api.espn.test/nfl/news';RSS='https://rss.espn.test/nfl'
    CFG={'news_api':{'NFL':API},'news_feeds':{'NFL':RSS}}

    def article(self,title,hours_ago,link,**extra):
        published=(NOW-timedelta(hours=hours_ago)).strftime('%Y-%m-%dT%H:%M:%SZ')
        return {'headline':title,'published':published,'links':{'web':{'href':link}},**extra}

    def response(self,payload=None,content=None):
        r=Mock(status_code=200);r.raise_for_status.return_value=None
        r.json.return_value=payload;r.content=content
        return r

    def test_espn_api_headlines_skip_paywalled_stale_unsafe_and_repeated_links(self):
        api=self.response({'articles':[
            self.article('  Newest   headline ',1,'https://www.espn.com/nfl/story/_/id/1/a'),
            self.article('ESPN+ only',1,'https://www.espn.com/nfl/insider/story/_/id/2/b',premium=True),
            self.article('Too old',40,'https://www.espn.com/nfl/story/_/id/3/c'),
            self.article('Not https',2,'http://www.espn.com/nfl/story/_/id/4/d'),
            self.article('Repeat',3,'https://www.espn.com/nfl/story/_/id/1/a'),
            self.article('Second',5,'https://www.espn.com/nfl/story/_/id/5/e'),
            {'headline':'No date','links':{'web':{'href':'https://www.espn.com/nfl/story/_/id/6/f'}}}]})
        with patch.dict(m.CFG,self.CFG), patch.object(m.requests,'get',return_value=api) as get:
            items,status=m.fetch_news(NOW)
        self.assertEqual(get.call_args.args[0],self.API,'the API is tried first')
        self.assertEqual([i['title'] for i in items],['Newest headline','Second'])
        self.assertEqual(items[0]['source'],'ESPN')
        self.assertEqual(status,{'NFL':'2 recent headlines'})

    def test_rss_is_the_fallback_and_total_failure_says_why(self):
        pub=(NOW-timedelta(hours=2)).strftime('%a, %d %b %Y %H:%M:%S GMT')
        rss=self.response(content=f'<rss><channel><item><title>From RSS</title><link>https://www.espn.com/nfl/story/_/id/9/r</link><pubDate>{pub}</pubDate></item></channel></rss>'.encode())
        with patch.dict(m.CFG,self.CFG), patch.object(m.requests,'get',side_effect=[m.requests.ConnectionError('refused'),rss]):
            items,status=m.fetch_news(NOW)
        self.assertEqual([i['title'] for i in items],['From RSS'])
        self.assertEqual(status,{'NFL':'1 recent headlines'})

        html=self.response(payload=None,content=b'<html>moved</html')
        html.json.side_effect=ValueError('Expecting value')
        with patch.dict(m.CFG,self.CFG), patch.object(m.requests,'get',return_value=html), patch('builtins.print') as log:
            items,status=m.fetch_news(NOW)
        self.assertEqual((items,status),([],{'NFL':'News feed unavailable'}))
        logged=log.call_args.args[0]
        self.assertIn('ESPN news API: ValueError',logged);self.assertIn('ESPN RSS: ParseError',logged)

if __name__=='__main__':unittest.main()

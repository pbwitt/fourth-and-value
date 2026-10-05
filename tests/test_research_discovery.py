"""Blind discovery cannot manufacture prices, forecasts, sources or identities."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import research_discovery as discovery
from test_analyst_review import NOW,feeds,CONFIG
from nhl.v2.data import iso
from nfl_game_quotes import publish
import tempfile

class DiscoveryTests(unittest.TestCase):
    def test_slate_blind_to_model_price_rank_and_complete_games(self):
        f=feeds();f['MLB']['rows'] += [dict(f['MLB']['rows'][0],mlb_game_id=i) for i in range(20)]
        games=discovery.slate(discovery.catalog(f,NOW)); self.assertEqual(len(games),22)
        request=discovery.search_request(games,NOW)
        self.assertNotIn('model_prob',request['input']);self.assertNotIn('model_ev',request['input'])
        self.assertNotIn('edge_bps',request['input']);self.assertEqual(request['max_tool_calls'],1)
        self.assertLess(discovery.bounds(request),2.75)
        request['max_tool_calls']=2
        with self.assertRaises(ValueError):discovery.bounds(request)

    def test_real_quote_only_no_probability_fallback_and_stale_excluded(self):
        offers=discovery.catalog(feeds(),NOW);r=offers[0]
        direction=dict(game_id=r['game_id'],sport=r['sport'],market=r['market'],player=r['player'],side=r['side'],
            hypothesis='Opportunity may exceed this threshold.',question='Confirm deployment.',source_urls=['https://www.nfl.com/news/test'])
        result=discovery.resolve([direction],offers,iso(NOW))
        self.assertEqual(len(result),1);self.assertEqual(result[0]['price'],r['price']);self.assertEqual(result[0]['line'],r['line'])
        self.assertIsNone(result[0]['model_prob']);self.assertIsNone(result[0]['estimated_ev'])
        self.assertEqual(discovery.resolve([dict(direction,player='Invented Player')],offers,iso(NOW)),[])
        self.assertEqual(discovery.catalog(feeds(),NOW+timedelta(minutes=100)),[])

    def test_unknown_search_urls_or_games_never_verify(self):
        games=discovery.slate(discovery.catalog(feeds(),NOW));g=games[0]
        d=dict(game_id=g['game_id'],sport=g['sport'],market=g['markets'][0],player='',side='Over',
            hypothesis='A conditional research direction.',question='Confirm starters.',source_urls=['https://www.espn.com/example'])
        response=dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(dict(directions=[d])))])])
        self.assertEqual(discovery.response_directions(response,games),[])
        response['output'].append(dict(type='web_search_call',action=dict(sources=[dict(url=d['source_urls'][0])])))
        self.assertEqual(len(discovery.response_directions(response,games)),1)
        d['game_id']='invented';response['output'][0]['content'][0]['text']=json.dumps(dict(directions=[d]))
        with self.assertRaises(ValueError):discovery.response_directions(response,games)

    def test_game_quotes_preserve_line_book_source_time_and_no_model(self):
        with tempfile.TemporaryDirectory() as td:
            events=[dict(id='g1',home_team='Home Team',away_team='Away Team',commence_time=iso(NOW+timedelta(hours=5)),
                bookmakers=[dict(key='book',title='Book',last_update=iso(NOW),markets=[dict(key='h2h',last_update=iso(NOW-timedelta(minutes=1)),
                    outcomes=[dict(name='Home Team',price=-120),dict(name='Away Team',price=110)])])])]
            value=publish(events,NOW,Path(td));r=value['rows'][0]
            self.assertEqual(r['quoted_at'],iso(NOW-timedelta(minutes=1)));self.assertIsNone(r['line'])
            self.assertIsNone(r['model_probability']);self.assertTrue(list((Path(td)/'data/nfl/lines/raw').glob('*.json')))
            self.assertEqual(len(discovery.catalog({'NFLGames':value},NOW)),2)

if __name__=='__main__':unittest.main()

class PipelineTests(unittest.TestCase):
    def test_full_queue_rotates_sports_and_preserves_pending_on_budget_exhaustion(self):
        from unittest.mock import patch
        import analyst_review as analyst
        from nhl.v2 import astra,evidence
        calls=[]
        f=feeds()
        f['NFL']['rows']=[dict(f['NFL']['rows'][0],game_id='n'+str(i)) for i in range(4)]
        f['MLB']['rows']=[dict(f['MLB']['rows'][0],mlb_game_id=i+1) for i in range(4)]
        def response(request):
            rows=json.loads(request['input'])['candidates'];calls.append([r['sport'] for r in rows])
            reviews=[dict(candidate_id=r['candidate_id'],status='needs_information',assessment=dict(verdict='consider',
                reason='The case merits human review.',model_case='The forecast is experimental.',price_case='Check this executable quote.',
                context_case='No additional reporting verified.',blocking_checks=[]),countercase='Forecast assumptions may fail.',
                open_checks=['Reconfirm the available price.'],evidence=[]) for r in rows]
            return dict(status='completed',usage=dict(input_tokens=1000,output_tokens=100),
                output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(dict(reviews=reviews)))])])
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
            patch.object(evidence,'collect',return_value=([],{})),patch.object(astra,'checkpoint'), \
            patch.object(astra,'call_api',side_effect=response):
            p=Path(td)
            result=analyst.prepare(f,NOW,dict(CONFIG,discovery_enabled=False),run_review=True,
                archive=p/'archive',public=p/'reviews.json',clock=lambda:NOW)
            self.assertEqual(calls,[['NFL']*3,['MLB']*3,['NFL'],['MLB']])
            self.assertEqual(result['sports']['NFL']['reviewed_count'],4)
            self.assertEqual(result['sports']['MLB']['reviewed_count'],4)
            self.assertTrue(all(r['qualitative_review']['probability_adjustment'] is None for b in result['sports'].values() for r in b['candidates']))
            # New quote with unchanged price/model/reporting reuses an interpretation;
            # the original review timestamp is retained, not silently made fresh.
            calls.clear()
            f['NFL']['rows'][0]['last_update']=iso(NOW)
            second=analyst.prepare(f,NOW+timedelta(seconds=1),dict(CONFIG,discovery_enabled=False),run_review=True,
                archive=p/'archive',public=p/'reviews.json',clock=lambda:NOW+timedelta(seconds=1))
            self.assertEqual(calls,[])
            self.assertEqual(second['sports']['NFL']['candidates'][0]['qualitative_review']['reviewed_at'],iso(NOW))

class RegressionTests(unittest.TestCase):
    def test_normalizing_changed_offer_cannot_count_old_assessment_as_complete(self):
        import analyst_review as analyst
        row=analyst.selected(feeds(),NOW)['selected'][0]
        row['qualitative_review']=dict(offer_id='old',forecast_id='old',status='needs_information')
        row['reviewed_candidate']={'price':-110}
        current=analyst.normalized(row)
        self.assertNotIn('qualitative_review',current)
        self.assertNotIn('reviewed_candidate',current)

    def test_failed_feed_cannot_supply_independent_discovery_quotes(self):
        f=feeds();f['NFL']['status']='feed_error';f['MLB']['status']='feed_error'
        self.assertEqual(discovery.catalog(f,NOW),[])

class DiscoveryRunTests(unittest.TestCase):
    def test_paid_protocol_is_reserved_archived_matched_and_not_retried(self):
        from unittest.mock import patch
        from nhl.v2 import astra
        f=feeds();offer=discovery.catalog(f,NOW)[0];calls=[]
        direction=dict(sport=offer['sport'],game_id=offer['game_id'],market=offer['market'],player=offer['player'],side=offer['side'],
            hypothesis='Check whether the expected opportunity exceeds the offered threshold.',question='Confirm current role.',source_urls=['https://www.nfl.com/news/synthetic'])
        response=dict(status='completed',usage=dict(input_tokens=1000,output_tokens=100),output=[
            dict(type='web_search_call',action=dict(sources=[dict(url=direction['source_urls'][0])])),
            dict(type='message',content=[dict(type='output_text',text=json.dumps(dict(directions=[direction])))])])
        with tempfile.TemporaryDirectory() as td,patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
             patch.object(astra,'checkpoint') as checkpoint,patch.object(astra,'call_api',return_value=response) as call:
            p=Path(td)
            result=discovery.run(f,NOW,CONFIG,archive=p,clock=lambda:NOW,public=p/'discovery.json',execute=True)
            self.assertEqual(result['status'],'completed');self.assertEqual(len(result['candidates']),1)
            self.assertEqual(result['candidates'][0]['price'],offer['price']);self.assertIsNone(result['candidates'][0]['model_prob'])
            self.assertEqual(result['candidates'][0]['original_forecast']['model_prob'],offer['model_prob'])
            self.assertEqual(result['budget']['charged_or_reserved_usd'],round(1000*discovery.astra.INPUT_RATE+100*discovery.astra.OUTPUT_RATE+.01,6))
            checkpoint.assert_called_once();call.assert_called_once()
            self.assertEqual(len(list((p/'requests').glob('*.json'))),1)
            again=discovery.run(f,NOW,CONFIG,archive=p,clock=lambda:NOW,public=p/'discovery.json',execute=True)
            # A same-day rerun with no new game or question is idempotent: no second paid call.
            self.assertEqual(again['status'],'reused_same_day');call.assert_called_once()
            self.assertEqual(len(again['submitted_games']),len(result['submitted_games']))

class QueuePriorityTests(unittest.TestCase):
    def test_independent_idea_is_researched_before_large_model_pool(self):
        from unittest.mock import patch
        import analyst_review as analyst
        from nhl.v2 import evidence
        f=feeds();rows=analyst.selected(f,NOW)['selected']
        rows=[analyst.normalized(r) for r in rows]
        quantitative=next(r for r in rows if r['sport']=='MLB')
        idea=dict(quantitative,candidate_id='independent',discovery_origin='independent_research')
        b=dict(sport='MLB',session='morning',candidates=[quantitative,idea])
        with patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}),patch.object(evidence,'collect',return_value=([],{})):
            result=analyst.review_batches(b,f,CONFIG,Path('/unused'),lambda:NOW,{})
        self.assertEqual(result['_research_queue']['pending'][0]['candidate_id'],'independent')

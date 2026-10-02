from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_writer as w

class DataOrdering(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,9,23,12,tzinfo=timezone.utc)
        self.board={'status':'ready','last_success_at':self.now.isoformat(),'model_checked_at':self.now.isoformat(),'model_status':'Independent MLB forecasts available','model_summary':{'history_through':'2026-09-22'}}
        self.briefing={'generated_at':self.now.isoformat()}
    def test_current_mlb_history_required(self):
        self.assertTrue(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
        self.board['model_summary']['history_through']='2026-09-21'
        self.assertFalse(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
    def test_stale_or_failed_refresh_rejected(self):
        self.board['last_success_at']=(self.now-timedelta(minutes=91)).isoformat()
        self.assertFalse(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
        self.board['last_success_at']=self.now.isoformat();self.board['status']='feed_error'
        self.assertFalse(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
    def test_feed_without_any_model_check_is_unvalidated_not_stale(self):
        board={'status':'ready','last_success_at':self.now.isoformat(),'model_status':'Historical baselines; NBA predictions are not validated'}
        status=w.data_readiness('NBA',board,self.briefing,self.now)
        self.assertFalse(status['ready']);self.assertIn('no validated model forecasts',status['reason'])
        board['model_error']='upstream failed'
        self.assertIn('missing or stale',w.data_readiness('NBA',board,self.briefing,self.now)['reason'])
    def test_nhl_experimental_forecasts_are_usable_and_labelled(self):
        game=dict(event_id='e1',game='Away @ Home',home_team='Home',away_team='Away',commence_time=(self.now+timedelta(hours=8)).isoformat(),player='',projected_home_reg_goals=3.0,projected_away_reg_goals=2.5)
        board={'status':'ready','last_success_at':self.now.isoformat(),'model_prediction_at':self.now.isoformat(),'model_version':'nhl-v2.1','history_through_date':'2026-09-21',
               'model_status':'Experimental independent forecasts; market blend and recommendations disabled',
               'rows':[dict(game,market='h2h',side='Home'),dict(game,market='totals',side='Over'),
                       dict(game,market='player_shots_on_goal',market_label='Shots on goal',player='Skater',projected_mean=2.1,side='Under')]}
        adapted=w.nhl_board(board)
        self.assertEqual(adapted['model_checked_at'],board['model_prediction_at'])
        self.assertTrue(w.data_readiness('NHL',adapted,self.briefing,self.now)['ready'])
        self.assertFalse(w.data_readiness('NHL',board,self.briefing,self.now)['ready'])
        games=[r for r in adapted['rows'] if r['market']=='game_projection']
        self.assertEqual(len(games),1);self.assertAlmostEqual(games[0]['model_mean'],5.5)
        self.assertIn('Home 3.00, Away 2.50',games[0]['model_mean_label'])
        self.assertTrue(all('not validated against betting prices' in r['model_status'] for r in adapted['rows']))
        # Two days of history lag is accepted for NHL only.
        packet={'sport':'NHL','model_rows':[],'model_references':[r for r in adapted['rows'] if r['model_mean'] is not None]}
        self.assertEqual(len(w.qualified_models(packet,self.now)),2)
        self.assertEqual(w.qualified_models(dict(packet,sport='MLB'),self.now),[])
        for broken in ({'model_error':'failed'},{'history_error':'failed'},{'model_prediction_at':None}):
            self.assertNotIn('model_checked_at',w.nhl_board(dict(board,**broken)))
    def test_automatic_articles_require_post_start_models_but_manual_requests_do_not(self):
        self.board['model_checked_at']='2026-09-23T10:45:00Z'  # 6:45 ET; within 90 minutes.
        with patch.dict('os.environ',{'EDITORIAL_REQUIRE_MORNING_MODELS':'true'}):
            for sport in ('MLB','NFL'):
                status=w.data_readiness(sport,self.board,self.briefing,self.now)
                self.assertFalse(status['ready']);self.assertIn('7:05',status['reason'])
            self.board['model_checked_at']='2026-09-23T11:10:00Z'
            self.assertTrue(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
            self.board['model_error']='upstream failed'
            self.assertFalse(w.data_readiness('NFL',self.board,self.briefing,self.now)['ready'])
        self.board.pop('model_error');self.board['model_checked_at']='2026-09-23T10:45:00Z'
        with patch.dict('os.environ',{'EDITORIAL_REQUIRE_MORNING_MODELS':'false'}):
            self.assertTrue(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
    def test_delayed_automatic_writer_rechecks_window_before_using_fresh_data(self):
        now=self.now.replace(hour=16)  # Noon ET, even if the planner ran earlier.
        self.board['model_checked_at']=now.isoformat()
        self.briefing['generated_at']=now.isoformat()
        with patch.dict('os.environ',{'EDITORIAL_REQUIRE_MORNING_MODELS':'true'}):
            status=w.data_readiness('NFL',self.board,self.briefing,now)
            self.assertFalse(status['ready']);self.assertIn('noon',status['reason'])
        with patch.dict('os.environ',{'EDITORIAL_REQUIRE_MORNING_MODELS':'false'}):
            self.assertTrue(w.data_readiness('NFL',self.board,self.briefing,now)['ready'])
    def test_previous_day_briefing_rejected_even_within_six_hours(self):
        now=datetime(2026,9,23,5,tzinfo=timezone.utc)
        self.briefing['generated_at']='2026-09-23T03:00:00Z'
        self.assertFalse(w.data_readiness('NFL',{},self.briefing,now)['ready'])
    def test_compaction_preserves_real_market_and_model(self):
        packet={'markets':[{'id':str(i),'game':'Yankees at Rays','quotes':[{'line':7.5+j/2,'over_price':-110,'under_price':-110,'label':'Book '+str(j),'quoted_at':self.now.isoformat()} for j in range(30)]} for i in range(3)],'model_rows':[{'id':'model','model_mean':5.2,'model_inputs':{'last_five':5}}], 'reporting':[{'title':'Yankees news','excerpt':'facts '*1000,'url':'https://example.org/'+str(i)} for i in range(3)],'methods':'Methods '*300}
        result=w.compact(packet)
        self.assertTrue(result['markets']);self.assertTrue(result['model_rows'])
        self.assertLessEqual(len(result['markets'][0]['quotes']),4)
        self.assertLess(len(json.dumps(result).encode()),11500)
    def test_target_compaction_keeps_matching_prop_and_drops_unrelated_model(self):
        packet={'markets':[
            {'id':'total-padres','game':'San Diego Padres @ Los Angeles Dodgers','commence_time':'2026-09-25T02:11:00Z','quotes':[]},
            {'id':'total-pirates','game':'St. Louis Cardinals @ Pittsburgh Pirates','commence_time':'2026-09-24T16:36:00Z','quotes':[]}],
            'model_rows':[
                {'id':'model-padres','event_id':'padres','game':'San Diego Padres @ Los Angeles Dodgers','market':'pitcher_strikeouts','player':'Starter','model_ev_pct':8.2,'model_mean':6.4,'is_model_pick':True},
                {'id':'model-pirates','event_id':'pirates','game':'St. Louis Cardinals @ Pittsburgh Pirates','market':'h2h','model_ev_pct':22.4,'model_mean':-.1,'is_model_pick':True}],
            'model_references':[],
            'model_availability':[
                {'event_id':'padres','game':'San Diego Padres @ Los Angeles Dodgers','available':True,'forecast_count':5,'eligible_pick_count':1},
                {'event_id':'pirates','game':'St. Louis Cardinals @ Pittsburgh Pirates','available':True,'forecast_count':4,'eligible_pick_count':1}],
            'reporting':[
                {'title':'Padres move closer to clinching','url':'https://mlb.com/padres','excerpt':'facts '*800},
                {'title':'Dodgers prepare for postseason','url':'https://cbssports.com/dodgers','excerpt':'facts '*800}],
            'methods':'Methods '*300}
        result=w.compact(packet)
        self.assertEqual(result['target_game']['event_id'],'padres')
        self.assertEqual([row['id'] for row in result['model_rows']],['model-padres'])
        self.assertEqual(result['model_rows'][0]['market'],'pitcher_strikeouts')
        self.assertTrue(all('Padres' in row['game'] or 'Dodgers' in row['game'] for row in result['markets']))

    def test_target_preserves_unavailable_reason_and_research_reference(self):
        packet={'markets':[{'id':'total-game','game':'San Diego Padres @ Los Angeles Dodgers','quotes':[]}],
            'model_rows':[],
            'model_references':[{'id':'reference-game','event_id':'game','game':'San Diego Padres @ Los Angeles Dodgers','market':'totals','model_mean':7.8,'model_status':'Research forecast: this market did not pass the applicable validation checks'}],
            'model_availability':[{'event_id':'game','game':'San Diego Padres @ Los Angeles Dodgers','available':True,'forecast_count':1,'eligible_pick_count':0,'statuses':['Research forecast: this market did not pass the applicable validation checks']}],
            'reporting':[{'title':'Padres Dodgers preview','url':'https://mlb.com/padres-dodgers','excerpt':'facts '*1200}],
            'methods':'Methods '*400}
        result=w.compact(packet)
        self.assertEqual(result['target_game']['model_availability']['eligible_pick_count'],0)
        self.assertEqual(len(result['model_references']),1)
        self.assertIn('Research forecast',result['model_references'][0]['model_status'])

    def test_explicit_unavailable_reason_survives(self):
        packet={'markets':[{'id':'total-game','game':'San Diego Padres @ Los Angeles Dodgers','quotes':[]}],
            'model_rows':[],'model_references':[],
            'model_availability':[{'event_id':'game','game':'San Diego Padres @ Los Angeles Dodgers','available':False,'forecast_count':0,'eligible_pick_count':0,'reason':'Waiting for both probable starters','statuses':['Waiting for both probable starters']}],
            'reporting':[{'title':'Padres Dodgers preview','url':'https://mlb.com/padres-dodgers','excerpt':'facts'}]}
        result=w.compact(packet)
        self.assertFalse(result['target_game']['model_availability']['available'])
        self.assertEqual(result['target_game']['model_availability']['reason'],'Waiting for both probable starters')

    def test_missing_data_stops_before_research_or_spend(self):
        with TemporaryDirectory() as directory,patch.object(w,'STATE',Path(directory)),patch.object(w,'evidence',return_value={'data_readiness':{'ready':False,'reason':'stale'}}),patch.object(w.reporting,'collect') as collect,patch.object(w,'call_api') as api,patch.object(w.budget,'reserve') as reserve,patch.object(w.ed,'render_home'):
            w.run(self.now)
            api.assert_not_called();reserve.assert_not_called();collect.assert_not_called()
    def test_empty_packet_not_publishable(self):
        with self.assertRaises(ValueError):w.require_data({'data_readiness':{'ready':True},'markets':[],'model_rows':[]})

if __name__=='__main__':unittest.main()

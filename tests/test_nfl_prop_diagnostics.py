"""Independent inputs, exact offers, alternate prices and hypothetical sensitivity."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import nfl_prop_diagnostics as d
from make_player_prop_params import build_params
import make_player_prop_params as model

AT='2026-09-27T10:44:02Z'

def fixture():
    frame=[]
    for book,line,under,over in [('bovada',212.5,-115,-105),('bovada',242.5,-240,175),('other',211.5,-113,-113)]:
        for side,price in [('under',under),('over',over)]:
            frame.append(dict(game_id='g1',player='Test QB',market_std='pass_yds',bookmaker=book,
                name=side,point=line,price=price,last_update=AT,model_prob=.98 if side=='under' else .02,
                model_prob_raw=.914365 if side=='under' else .085635,mu=144.751236,sigma=71.446743))
    row=dict(game_id='g1',player='Test QB',market_std='pass_yds',book='bovada',side='under',line=242.5,
        price=-240,quoted_at=AT,consensus_line=211.5)
    return pd.DataFrame(frame),row

class DiagnosticsTests(unittest.TestCase):
    def test_production_csv_preserves_trace_through_schema_cleanup(self):
        props=pd.DataFrame([dict(player='Test QB',market='player_pass_yds')])
        params=pd.DataFrame([dict(player='Test QB',market_std='pass_yds',mu=200.,sigma=60.,
            lam=float('nan'),projection_diagnostics=json.dumps({'attempts':30}))])
        with TemporaryDirectory() as temp, patch.object(sys,'argv',['params','--season','2026','--week','3','--out',temp+'/params.csv']), \
             patch.object(model,'load_props_candidates',return_value=props),patch.object(model,'load_props',return_value=props), \
             patch.object(model,'fetch_recent_game_logs',return_value=pd.DataFrame()), \
             patch('career_baseline.load_career_logs',return_value=pd.DataFrame()), \
             patch.object(model,'calculate_defensive_ratings',return_value=pd.DataFrame()), \
             patch.object(model,'create_opponent_map',return_value={}),patch.object(model,'create_home_away_map',return_value={}), \
             patch.object(model,'build_params',return_value=params):
            model.main()
            result=pd.read_csv(temp+'/params.csv')
            self.assertEqual(json.loads(result.iloc[0].projection_diagnostics),{'attempts':30})
            self.assertEqual(result.iloc[0].mu,200.)

    def test_alternate_ladder_is_not_independent_exact_line_consensus(self):
        f,r=fixture();before=f.copy(deep=True);context=d.build_context(f,AT)
        got=d.review_diagnostics(r,context,AT)
        pd.testing.assert_frame_equal(f,before)
        self.assertEqual(got['other_books_at_exact_line'],0)
        self.assertEqual(got['offered_book_central_quote']['point'],212.5)
        self.assertEqual(got['other_book_central_quotes'][0]['point'],211.5)
        self.assertAlmostEqual(got['offered_book_nearby_quotes'][-1]['prob_devig'],.66)
        self.assertAlmostEqual(got['raw_distribution_stress']['mean_at_break_even'],203.819084,places=4)
        self.assertLess(got['raw_distribution_stress']['ev_per_unit'],0)
        self.assertTrue(got['median_is_not_expected_mean'])

    def test_no_context_transfer_between_snapshots_or_changed_offers(self):
        f,r=fixture();context=d.build_context(f,AT)
        self.assertIsNone(d.review_diagnostics(r,context,'2026-09-27T11:00:00Z'))
        for key,value in [('game_id','g2'),('player','Other QB'),('price',-230),('line',232.5),('quoted_at','2026-09-27T10:45:00Z')]:
            self.assertIsNone(d.review_diagnostics(dict(r,**{key:value}),context,AT))
        self.assertIsNone(d.review_diagnostics(r,None,AT))

    def test_quote_pairing_requires_contemporary_opposite_side(self):
        f,r=fixture(); f.loc[(f.bookmaker=='bovada')&(f.name=='over'),'last_update']='2026-09-27T09:44:02Z'
        got=d.review_diagnostics(r,d.build_context(f,AT),AT)
        self.assertIsNone(got['offered_book_central_quote'])
        self.assertTrue(all(q['prob_devig'] is None for q in got['offered_book_nearby_quotes']))
        f.loc[f.bookmaker=='other','last_update']='2026-09-27T09:44:02Z'
        self.assertEqual(d.review_diagnostics(r,d.build_context(f,AT),AT)['other_book_central_quotes'],[])

    def test_integer_lines_have_no_unmodeled_push_sensitivity(self):
        f,r=fixture();f.loc[f.point.eq(242.5),'point']=242;r['line']=242
        self.assertIsNone(d.review_diagnostics(r,d.build_context(f,AT),AT)['raw_distribution_stress'])

    def test_calibration_exposes_endpoint_without_implying_tail_sample_support(self):
        artifact=json.loads((Path(__file__).resolve().parents[1]/'models/nfl_prop_calibration.json').read_text())
        trace=d.calibration_trace('pass_yds',.914365,artifact)
        self.assertTrue(trace['outside_fitted_range'])
        self.assertEqual(trace['endpoint_probabilities'][-1],.98)
        self.assertIsNone(trace['tail_sample_size'])
        self.assertIsNone(trace['market_sample_size'])
        self.assertEqual(d.calibration_trace('unknown',.9,artifact),{'status':'not_calibrated'})

    def test_projection_trace_obeys_cutoff_and_does_not_alter_forecast(self):
        # A future 600-yard game cannot appear in the sample or change the forecast.
        cands=pd.DataFrame([dict(player='Test QB',market_std='pass_yds')])
        logs=pd.DataFrame([dict(player='Test QB',season=2026,week=w,position='QB',attempts=a,completions=c,
            passing_yards=y,rushing_yards=0,carries=0,receptions=0,targets=0,receiving_yards=0,passing_tds=0,interceptions=0,rushing_tds=0,receiving_tds=0) for w,a,c,y in [(1,5,3,18),(3,60,50,600)]])
        got=build_params(cands,logs,2026,3).iloc[0]
        prior=build_params(cands,logs.iloc[:1],2026,3).iloc[0]
        self.assertEqual(got.mu,prior.mu);self.assertEqual(got.sigma,prior.sigma)
        trace=json.loads(got.projection_diagnostics)
        self.assertEqual([s['week'] for s in trace['current_sample']],[1])
        self.assertEqual(trace['current_sample'][0]['attempts'],5)
        self.assertAlmostEqual(trace['mean_stages']['final'],got.mu,places=5)

if __name__=='__main__':unittest.main()

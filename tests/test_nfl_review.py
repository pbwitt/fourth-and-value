"""Behavioral regressions for pricing identity, pushes, chronology and freshness."""
import math
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from market_math import add_market_comparisons, outcome_probabilities, expected_profit, implied_probability
from validate_data_freshness import validate_props_freshness
from nfl_build_team_features import add_rolling_features
from nfl_find_consensus_edges import find_consensus_edges


class PricingTests(unittest.TestCase):
    def quote(self,game='g1',book='A',point=50.5,side='over',price=-110):
        return dict(game_id=game,player='A Player',market_std='rush_yds',bookmaker=book,point=point,name=side,price=price)

    def test_devig_pairs_exact_event_line_and_distinct_book(self):
        rows=[self.quote(),self.quote(side='under'),self.quote(),self.quote(point=60.5),
              self.quote(game='g2',side='under'),self.quote(book='B',price=100),self.quote(book='B',side='under',price=-120)]
        out=add_market_comparisons(pd.DataFrame(rows))
        self.assertAlmostEqual(out.iloc[0].prob_devig,.5)
        self.assertEqual(out.iloc[0].book_count,2)
        self.assertTrue(math.isnan(out.iloc[3].prob_devig))
        self.assertTrue(math.isnan(out.iloc[4].prob_devig))
        self.assertEqual(len(out),len(rows))
        self.assertAlmostEqual(out.iloc[0].consensus_prob,(.5+(.5/(.5+120/220)))/2)

    def test_binary_missing_opposite_is_not_devigged(self):
        a=self.quote(point=None,side='yes',price=300);a['market_std']='anytime_td'
        b=dict(a,name='no',price=-400)
        self.assertTrue(math.isnan(add_market_comparisons(pd.DataFrame([a])).iloc[0].prob_devig))
        self.assertAlmostEqual(add_market_comparisons(pd.DataFrame([a,b])).iloc[0].prob_devig,.25/1.05)

    def test_conflicting_duplicate_quote_is_unavailable(self):
        out=add_market_comparisons(pd.DataFrame([self.quote(),self.quote(price=-120),self.quote(side='under')]))
        self.assertTrue(out.prob_devig.isna().all())

    def test_poisson_integer_push_and_half_line(self):
        for market in ['pass_tds','pass_interceptions','interceptions']:
            over,push=outcome_probabilities(market,'over',1,lam=1)
            under,push2=outcome_probabilities(market,'under',1,lam=1)
            self.assertAlmostEqual(push,math.exp(-1));self.assertAlmostEqual(push,push2)
            self.assertAlmostEqual(over+under,1)
            self.assertAlmostEqual(under*(1-push),math.exp(-1))
            self.assertEqual(outcome_probabilities(market,'under',.5,lam=0),(1.,0.))
            self.assertEqual(outcome_probabilities(market,'over',.5,lam=0),(0.,0.))

    def test_normal_push_and_ev(self):
        p,push=outcome_probabilities('receptions','over',5,mu=5,sigma=2)
        self.assertAlmostEqual(p,.5);self.assertGreater(push,0)
        self.assertEqual(outcome_probabilities('receptions','under',5.5,mu=5,sigma=2)[1],0)
        self.assertAlmostEqual(expected_profit(.55,-110),5.)
        self.assertAlmostEqual(expected_profit(.55,-110,.1),4.5)
        self.assertTrue(math.isnan(implied_probability(0)))
        self.assertTrue(math.isnan(implied_probability(float('inf'))))
        self.assertTrue(math.isnan(outcome_probabilities('pass_tds','invalid',1,lam=1)[0]))

    def test_freshness_uses_quotes_not_file_modification(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'quotes.csv'
            pd.DataFrame([dict(commence_time='2026-09-10T00:20:00Z',last_update='2026-09-08T20:00:00Z')]).to_csv(p,index=False)
            self.assertTrue(validate_props_freshness(p,48,now='2026-09-09T00:00:00Z'))
            self.assertFalse(validate_props_freshness(p,1,now='2026-09-09T00:00:00Z'))
            pd.DataFrame([dict(commence_time='2026-09-10T00:20:00Z',last_update=None)]).to_csv(p,index=False)
            self.assertFalse(validate_props_freshness(p,48,now='2026-09-09T00:00:00Z'))

    def test_rolling_features_respect_date_and_original_index(self):
        cols=['off_epa_per_play','off_success_rate','off_pass_epa','off_rush_epa','off_explosive_play_rate',
              'off_third_down_conv','off_red_zone_td_rate','def_epa_per_play','def_success_rate','points_scored','points_allowed']
        rows=[]
        for date,week,score in [('2026-09-09',1,30),('2025-12-14',15,10),('2026-09-17',2,60)]:
            rows.append(dict(team='A',game_id=date,game_date=date,week=week,**{c:score for c in cols}))
        out=add_rolling_features(pd.DataFrame(rows,index=[7,3,9]),[3])
        self.assertTrue(math.isnan(out.loc[3,'points_scored_L3']))
        self.assertEqual(out.loc[7,'points_scored_L3'],10)
        self.assertEqual(out.loc[9,'points_scored_L3'],20)

    def test_totals_uses_points_not_odds_and_clears_empty_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);pred=p/'pred.csv';odds=p/'odds.csv';out=p/'edges.csv'
            pd.DataFrame([dict(game='A @ B',total_pred=45)]).to_csv(pred,index=False)
            rows=[dict(game='A @ B',market='totals',bookmaker=b,name=side,point=line,price=-110)
                  for b,line in [('a',45),('b',45),('c',47)] for side in ['Over','Under']]
            pd.DataFrame(rows).to_csv(odds,index=False)
            result=find_consensus_edges(str(pred),str(odds),str(out))
            self.assertEqual(len(result),1);self.assertEqual(result.iloc[0]['bet'],'UNDER');self.assertEqual(result.iloc[0]['line'],47)
            for r in rows:r['point']=45
            pd.DataFrame(rows).to_csv(odds,index=False)
            find_consensus_edges(str(pred),str(odds),str(out))
            self.assertTrue(pd.read_csv(out).empty)

class GradingTests(unittest.TestCase):
    def test_missing_stat_and_wrong_date_never_settle_as_push(self):
        from grade_bets_nfl import grade_bet
        bet=dict(market_type='recv_yds', side='over', line=50, player='A Player',
                 game_date='2026-09-13', stake_dollars=100, odds=-110)
        wrong=pd.DataFrame([dict(player='A Player',game_date='2025-09-13',receiving_yards=80)])
        self.assertIsNone(grade_bet(bet.copy(),wrong))
        missing=pd.DataFrame([dict(player='A Player',game_date='2026-09-13',receiving_yards=np.nan)])
        self.assertIsNone(grade_bet(bet.copy(),missing))
        correct=pd.DataFrame([dict(player='A Player',game_date='2026-09-13',receiving_yards=50)])
        self.assertEqual(grade_bet(bet.copy(),correct)['status'],'push')

if __name__=='__main__':
    unittest.main()

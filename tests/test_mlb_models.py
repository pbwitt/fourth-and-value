import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mlb.model_data import normalize, BAT_KEYS, PITCH_KEYS, manifest
from mlb.models import State, dataset, pmf, outcome, joint, game_outcome, expected_return, TARGETS
from mlb.predict import published_lineups, pick_reason, attach, load_bundle
from mlb.train import partition, score

NOW=datetime(2026,9,21,18,tzinfo=timezone.utc)


def game(day,game_id=1,runs=4):
    result=dict(id=game_id,date=day,start=day+'T20:00:00Z',season=2026,game_type='R',venue=1,
                home_id=1,away_id=2,home_score=runs,away_score=3,teams={})
    for side,team_id in [('home',1),('away',2)]:
        batting=dict.fromkeys(BAT_KEYS,0);batting.update(plateAppearances=38,hits=8,totalBases=12,runs=runs if side=='home' else 3,strikeOuts=8)
        pitching=dict.fromkeys(PITCH_KEYS,0);pitching.update(outs=27,battersFaced=38,strikeOuts=8,numberOfPitches=140,runs=3)
        starter={**pitching,'id':team_id+10,'name':f'Pitcher {side}','outs':18,'gamesStarted':1,'numberOfPitches':90}
        batters=[{**batting,'id':team_id*100+slot,'name':f'Batter {team_id} {slot}','slot':slot,'plateAppearances':4,'hits':1} for slot in range(1,10)]
        result['teams'][side]=dict(id=team_id,name=side,starter=starter['id'],batting=batting,pitching=pitching,batters=batters,pitchers=[starter])
    return result


def lineup_box():
    return dict(teams={side:dict(team={'id':team_id},teamStats={},players={str(slot):dict(person={'id':team_id*100+slot,'fullName':f'Batter {team_id} {slot}'},battingOrder=str(slot*100)) for slot in range(1,10)}) for side,team_id in [('home',1),('away',2)]})


class ModelTests(unittest.TestCase):
    def test_current_future_and_same_day_results_cannot_enter_features(self):
        games=[game(f'2026-08-{day:02}',day) for day in range(1,21)]
        games.append(game('2026-08-20',99))
        original,_=dataset(games)
        changed=copy.deepcopy(games)
        changed[-2]['teams']['home']['batting']['runs']=99
        changed[-2]['teams']['home']['batters'][0]['hits']=20
        altered,_=dataset(changed)
        for target in TARGETS:
            self.assertTrue(original[target])
            self.assertEqual([r['x'] for r in original[target]],[r['x'] for r in altered[target]])
        next_day,_=dataset(games+[game('2026-08-21',100)])
        next_changed,_=dataset(changed+[game('2026-08-21',100)])
        self.assertNotEqual(next_day['team_runs'][-2]['x']['team_runs'],next_changed['team_runs'][-2]['x']['team_runs'])

    def test_box_season_totals_are_never_features(self):
        meta=game('2026-08-01');box=lineup_box()
        for side in ['home','away']:
            team=box['teams'][side];observed=meta['teams'][side]
            team['team']['name']=side;team['teamStats']={k:observed[k] for k in ['batting','pitching']}
            for p in team['players'].values():p['stats']={'batting':observed['batters'][0]}
            team['players']['starter']=dict(person={'id':observed['starter'],'fullName':'Starter'},stats={'pitching':observed['pitchers'][0]},seasonStats={'pitching':{'strikeOuts':999}})
        first=normalize(meta,box)
        box['teams']['home']['players']['starter']['seasonStats']['pitching']['strikeOuts']=999999
        self.assertEqual(first,normalize(meta,box))

    def test_distribution_and_integer_push_math(self):
        for target in TARGETS:
            model=dict(target=target,alpha=.3,sigma=3)
            mass=pmf([1,5],model)
            np.testing.assert_allclose(mass.sum(axis=1),1)
            self.assertTrue((mass>=0).all())
            over,push=outcome(mass[1],5,'Over');under,_=outcome(mass[1],5,'Under')
            self.assertAlmostEqual(over+under+push,1)
            self.assertGreater(push,0)
        self.assertAlmostEqual(expected_return(.5,.1,100),.1)
        self.assertAlmostEqual(expected_return(.55,0,-110),.05)

    def test_game_matrix_has_no_ties_and_correct_spread_sides(self):
        home=np.array([0,0,0,1.]);away=np.array([0,1,0,0.]);matrix=joint(home,away)
        self.assertEqual(np.trace(matrix),0)
        self.assertEqual(game_outcome(matrix,'h2h',home=True),(1,0))
        self.assertEqual(game_outcome(matrix,'h2h',home=False),(0,0))
        self.assertEqual(game_outcome(matrix,'spreads',-1.5,True),(1,0))
        self.assertEqual(game_outcome(matrix,'spreads',1.5,False),(0,0))
        self.assertEqual(game_outcome(matrix,'totals',4),(0,1))

    def test_lineups_require_nine_unique_original_slots_and_no_game_action(self):
        fixture=lineup_box();g=dict(home_team_id=1,away_team_id=2)
        self.assertEqual(len(published_lineups(fixture,g)),2)
        fixture['teams']['home']['players']['1']['battingOrder']='101'
        self.assertNotIn('home',published_lineups(fixture,g))
        fixture['teams']['away']['teamStats']={'batting':{'plateAppearances':1}}
        self.assertEqual(published_lineups(fixture,g),{})

    def test_pick_guards_do_not_promote_consensus_or_bad_models(self):
        row=dict(market='pitcher_strikeouts',game_type='R',line=5.5,quoted_at=NOW.isoformat(),commence_time=(NOW+timedelta(hours=1)).isoformat(),
            fair_probability=.5,paired_books=3,model_ev_pct=5,model_edge_pp=4,best_price=True)
        report={'regular':{'pitcher_strikeouts':{'passed':True}}}
        self.assertIsNone(pick_reason(row,report,NOW))
        self.assertIsNotNone(pick_reason(row,{},NOW))
        for change in [dict(line=20.5),dict(paired_books=1),dict(fair_probability=None),dict(best_price=False),dict(model_ev_pct=31),dict(model_edge_pp=2),dict(quoted_at=(NOW-timedelta(hours=2)).isoformat())]:
            self.assertIsNotNone(pick_reason({**row,**change},report,NOW))
        self.assertIsNotNone(pick_reason({**row,'game_type':'W'},report,NOW))

    def test_live_missing_starters_and_lineups_withhold_forecasts(self):
        history=State()
        for day in range(1,21):history.update(game(f'2026-08-{day:02}',day))
        models={t:dict(target=t,kind='rolling',alpha=.2,sigma=3) for t in TARGETS}
        bundle=dict(models=models,state=history,version='test',report=dict(input_through='2026-09-20',training_through='2026-07-22'))
        g=dict(mlb_game_id=1,commence_time=(NOW+timedelta(hours=1)).isoformat(),game_type='R',venue_id=1,home_team_id=1,away_team_id=2,
            home_pitcher={'id':11,'fullName':'Pitcher home'},away_pitcher={'id':12,'fullName':'Pitcher away'})
        row=dict(mlb_game_id=1,market='batter_hits',player='Batter 1 1',model_probability=None)
        result=attach(dict(rows=[row],events=[g]),NOW,lambda endpoint:{},bundle)
        self.assertIsNone(result['rows'][0]['model_probability'])
        self.assertIn('published',result['rows'][0]['model_status'])
        g['home_pitcher']=None
        result=attach(dict(rows=[row],events=[g]),NOW,lambda endpoint:lineup_box(),bundle)
        self.assertIsNone(result['rows'][0]['model_probability'])
        self.assertFalse(result['rows'][0]['is_model_pick'])
        self.assertIn('probable starters',result['rows'][0]['model_status'])

    def test_held_out_dates_are_disjoint_and_worse_than_reference_fails(self):
        samples={'team_runs':[dict(date=f'2026-08-{d:02}',game_type='R') for d in range(1,31)]}
        train,cal,test=partition(samples,'2026-08-10','2026-08-20','2026-08-30')
        self.assertEqual([len(v['team_runs']) for v in [train,cal,test]],[10,10,10])
        self.assertFalse(score([.9]*200,[0]*200,[.5]*200,list(range(200)))['passed'])

    def test_live_all_markets_price_distributions_and_deduplicate_picks(self):
        history=State()
        for day in range(1,21):history.update(game(f'2026-08-{day:02}',day))
        models={t:dict(target=t,kind='rolling',alpha=.2,sigma=3) for t in TARGETS}
        report=dict(input_through='2026-09-20',training_through='2026-07-22',regular={m:{'passed':True} for m in ['h2h','spreads','totals',*TARGETS[1:]]})
        bundle=dict(models=models,state=history,version='test',report=report)
        g=dict(mlb_game_id=1,commence_time=(NOW+timedelta(hours=1)).isoformat(),game_type='R',venue_id=1,home_team_id=1,away_team_id=2,
            home_pitcher={'id':11,'fullName':'Pitcher home'},away_pitcher={'id':12,'fullName':'Pitcher away'})
        base=dict(mlb_game_id=1,game_type='R',home_team='Home',quoted_at=NOW.isoformat(),commence_time=g['commence_time'],
            side='Over',line=.5,price=110,book_probability=1/2.1,fair_probability=.5,paired_books=2,best_price=True)
        rows=[]
        for market in ['h2h','spreads','totals',*TARGETS[1:]]:
            row={**base,'market':market,'market_label':market,'player':'Pitcher home' if market.startswith('pitcher_') else 'Batter 1 1'}
            if market in ['h2h','spreads']:row.update(player='',side='Home',line=None if market=='h2h' else -1.5)
            if market=='totals':row.update(player='',line=8)
            if market=='pitcher_strikeouts':row['line']=5
            if market=='pitcher_outs':row['line']=15
            rows.append(row)
        rows.append(copy.deepcopy(rows[0]))
        result=attach(dict(rows=rows,events=[g]),NOW,lambda endpoint:lineup_box(),bundle)
        self.assertEqual(result['model_summary']['forecasts'],10)
        for row in rows:
            p,push=row['model_probability'],row['model_push_probability']
            self.assertGreaterEqual(p,0);self.assertLessEqual(p+push,1+1e-9)
            self.assertAlmostEqual(row['model_ev_pct'],100*expected_return(p,push,row['price']))
        self.assertLessEqual(sum(r['is_model_pick'] for r in rows if r['market']=='h2h'),1)
        self.assertGreater(next(r for r in rows if r['market']=='totals')['model_push_probability'],0)

    def test_stale_model_bundle_is_rejected(self):
        with patch('mlb.predict.MODEL_PATH') as path,patch('mlb.predict.joblib.load',return_value={'source_signature':'test','history_fetched_date':'2026-09-19'}),patch('mlb.predict.signature',return_value='test'):
            path.exists.return_value=True
            with self.assertRaisesRegex(ValueError,'daily refresh'):load_bundle(NOW)

    def test_history_cutoff_is_eastern_not_utc(self):
        with patch('mlb.model_data.get',return_value={'dates':[]}) as get:
            manifest(datetime(2026,9,22,1,tzinfo=timezone.utc))
            self.assertEqual(get.call_args.kwargs['endDate'],'2026-09-20')


if __name__=='__main__':unittest.main()

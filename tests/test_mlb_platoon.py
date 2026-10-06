import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mlb import model_data
from mlb.models import State, dataset, without_platoon, same_hand, PLATOON, PLATOON_LIVE, TARGETS
from mlb.predict import attach
from mlb import platoon_study as study
from test_mlb_models import game, lineup_box, NOW

# Home batters bat left, away batters right; the home starter (11) is a lefty, the away starter (12) a righty.
HANDS={**{100+s:dict(bats='L',throws='R') for s in range(1,10)},**{200+s:dict(bats='R',throws='R') for s in range(1,10)},
       11:dict(bats='L',throws='L'),12:dict(bats='R',throws='R')}


def games(days=20):
    return [game(f'2026-08-{day:02}',day) for day in range(1,days+1)]


class PlatoonFeatureTests(unittest.TestCase):
    def test_matchup_codes(self):
        self.assertEqual(same_hand('L','L'),1)
        self.assertEqual(same_hand('R','L'),0)
        self.assertEqual(same_hand('S','L'),0,'a switch hitter always has the platoon edge')
        self.assertEqual(same_hand(None,'L'),.5)
        self.assertEqual(same_hand('L',None),.5)

    def test_production_model_has_no_platoon_features_until_validated(self):
        self.assertFalse(PLATOON_LIVE)
        plain,_=dataset(games())
        for target in TARGETS:
            self.assertTrue(plain[target])
            self.assertFalse(set(PLATOON)&set(plain[target][0]['x']))

    def test_ablation_changes_only_the_platoon_columns(self):
        plain,_=dataset(games())
        platoon,_=dataset(games(),HANDS)
        stripped=without_platoon(platoon)
        for target in TARGETS:
            self.assertEqual(plain[target],stripped[target],'same rows, outcomes and other features')
        self.assertTrue(set(PLATOON)<=set(platoon['batter_hits'][0]['x']))
        self.assertTrue({'starter_left','opp_starter_left','team_adv_share','opp_adv_share'}<=set(platoon['team_runs'][0]['x']))

    def test_matchup_values_follow_the_starters(self):
        samples,_=dataset(games(),HANDS)
        home=next(r for r in samples['batter_hits'] if r['side']=='home')['x']
        away=next(r for r in samples['batter_hits'] if r['side']=='away')['x']
        # Left-handed home batters face the right-handed away starter; right-handed visitors face a lefty.
        self.assertEqual((home['batter_same_hand'],home['opp_starter_left'],home['starter_left']),(0,0,1))
        self.assertEqual((away['batter_same_hand'],away['opp_starter_left']),(0,1))
        for x in [home,away]:
            self.assertTrue(0<=x['batter_same_share']<=1)
            self.assertGreater(x['team_adv_share'],.9,'every batter in these lineups has the edge')
        unknown,_=dataset(games(),{})
        x=unknown['batter_hits'][0]['x']
        self.assertEqual((x['batter_same_hand'],x['team_adv_share'],x['opp_starter_left']),(.5,.5,.5))

    def test_current_and_future_games_cannot_enter_platoon_features(self):
        history=games()+[game('2026-08-20',99)]
        original,_=dataset(history,HANDS)
        changed=copy.deepcopy(history)
        # Same-day box score: different batters, a different reliever and different PA.
        changed[-2]['teams']['home']['batters'][0]['plateAppearances']=9
        changed[-2]['teams']['home']['pitchers'].append(dict(changed[-2]['teams']['home']['pitchers'][0],id=13,gamesStarted=0,battersFaced=30))
        altered,_=dataset(changed,{**HANDS,13:dict(bats='L',throws='L')})
        for target in TARGETS:
            self.assertEqual([r['x'] for r in original[target]],[r['x'] for r in altered[target]])
        # The next date does see it: a left-handed reliever lowers right-handed visitors' same-hand share.
        hands={**HANDS,13:dict(bats='L',throws='L')}
        later,_=dataset(changed+[game('2026-08-21',100)],hands)
        unchanged,_=dataset(history+[game('2026-08-21',100)],hands)
        self.assertLess(later['batter_hits'][-1]['x']['batter_same_share'],unchanged['batter_hits'][-1]['x']['batter_same_share'])

    def test_live_forecasts_use_the_same_platoon_inputs(self):
        history=State(HANDS)
        for day in range(1,21):history.update(game(f'2026-08-{day:02}',day))
        models={t:dict(target=t,kind='rolling',alpha=.2,sigma=3) for t in TARGETS}
        report=dict(input_through='2026-09-20',training_through='2026-07-22',regular={m:{'passed':True} for m in ['h2h','spreads','totals',*TARGETS[1:]]})
        g=dict(mlb_game_id=1,commence_time=(NOW+timedelta(hours=1)).isoformat(),game_type='R',venue_id=1,home_team_id=1,away_team_id=2,
            home_pitcher={'id':11,'fullName':'Pitcher home'},away_pitcher={'id':12,'fullName':'Pitcher away'})
        row=dict(mlb_game_id=1,game_type='R',home_team='Home',quoted_at=NOW.isoformat(),commence_time=g['commence_time'],side='Over',line=.5,
                 price=110,book_probability=1/2.1,fair_probability=.5,paired_books=2,best_price=True,market='batter_hits',market_label='Hits',player='Batter 1 1')
        result=attach(dict(rows=[row],events=[g]),NOW,lambda endpoint:lineup_box(),dict(models=models,state=history,version='test',report=report))
        self.assertIsNotNone(result['rows'][0]['model_probability'])
        x=history.features(dict(date='2026-09-21',game_type='R',venue=1,home_id=1,away_id=2,home_starter=11,away_starter=12),'home',101,1)
        self.assertEqual(x['batter_same_hand'],0)


class HandednessCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.patch=patch.object(model_data,'PLAYERS',Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def test_seasons_are_cached_current_season_refreshes_daily_and_gaps_are_looked_up(self):
        history=[{**game('2025-09-01',1),'season':2025},game('2026-08-01',2)]
        history[1]['teams']['home']['batters'][0]['id']=777
        calls=[]
        def fetch(endpoint,**params):
            calls.append((endpoint,params))
            if endpoint=='people':
                return {'people':[dict(id=777,batSide={'code':'S'},pitchHand={'code':'R'})]}
            return {'people':[dict(id=101,batSide={'code':'L'},pitchHand={'code':'R'}),dict(id=11,batSide={'code':'R'},pitchHand={'code':'L'}),
                              dict(id=5,batSide={},pitchHand={'code':'X'})]}
        with patch.object(model_data,'get',side_effect=fetch):
            hands=model_data.update_players(history,NOW)
            self.assertEqual(hands[101],dict(bats='L',throws='R'))
            self.assertEqual(hands[11]['throws'],'L')
            self.assertEqual(hands[5],dict(bats=None,throws=None))
            self.assertEqual(hands[777]['bats'],'S')
            self.assertEqual([c[0] for c in calls].count('sports/1/players'),2)
            self.assertIn('777',next(c[1]['personIds'] for c in calls if c[0]=='people').split(','))
            calls.clear();model_data.update_players(history,NOW)
            self.assertNotIn(('sports/1/players',{'season':2025}),calls,'a past season is fetched once')
            self.assertNotIn(('sports/1/players',{'season':2026}),calls,'already refreshed today')
            calls.clear();model_data.update_players(history,NOW+timedelta(days=1))
            self.assertEqual(calls[0],('sports/1/players',{'season':2026}))

    def test_failed_refresh_keeps_a_cache_but_cannot_start_without_one(self):
        history=[game('2026-08-01',2)]
        with patch.object(model_data,'get',side_effect=RuntimeError('MLB history request failed')):
            with self.assertRaisesRegex(RuntimeError,'handedness unavailable'):model_data.update_players(history,NOW)
        (Path(self.tmp.name)/'2026.json').write_text(json.dumps(dict(season=2026,fetched_date='2026-09-01',people={'101':dict(bats='L',throws='R')})))
        with patch.object(model_data,'get',side_effect=RuntimeError('MLB history request failed')):
            self.assertEqual(model_data.update_players(history,NOW)[101]['bats'],'L')


class StudyRuleTests(unittest.TestCase):
    def market(self,brier,log_loss=None,ece=(0,-.01,.01)):
        log_loss=log_loss or brier
        names=['mean','low','high']
        return dict(forecasts=100,games=50,brier=dict(zip(names,brier)),log_loss=dict(zip(names,log_loss)),ece=dict(zip(names,ece)))

    def reports(self,flip=None):
        passed={m:{'passed':True} for m in study.MARKETS}
        report={'with':copy.deepcopy(passed),'without':passed}
        if flip:report['with'][flip]['passed']=False
        return report

    def test_decision_rule(self):
        neutral={m:self.market((0,-1e-4,1e-4)) for m in study.MARKETS}
        self.assertTrue(study.decide(neutral,self.reports(),self.reports())['ship'],'neutral with no harm ships')
        hurt={**neutral,'batter_hits':self.market((2e-4,1e-4,3e-4))}
        self.assertFalse(study.decide(hurt,self.reports(),self.reports())['ship'])
        log_hurt={**neutral,'batter_hits':self.market((0,-1e-4,1e-4),(2e-4,1e-4,3e-4))}
        self.assertEqual(study.decide(log_hurt,self.reports(),self.reports())['worse'],['batter_hits'])
        slightly_worse={m:self.market((1e-5,-1e-4,1e-4)) for m in study.MARKETS}
        self.assertFalse(study.decide(slightly_worse,self.reports(),self.reports())['ship'],'positive average and no gain')
        # A noisy market's large negative change cannot outvote precise small harms.
        noisy={**slightly_worse,'h2h':self.market((-20e-4,-60e-4,20e-4))}
        self.assertGreater(study.decide(noisy,self.reports(),self.reports())['weighted_brier_change'],0)
        # Identical forecasts (zero-width interval) cannot pull the average to zero.
        unchanged={**slightly_worse,'pitcher_strikeouts':self.market((0,0,0))}
        self.assertAlmostEqual(study.decide(unchanged,self.reports(),self.reports())['weighted_brier_change'],1e-5)
        self.assertFalse(study.decide(unchanged,self.reports(),self.reports())['ship'])
        better={**slightly_worse,'batter_home_runs':self.market((-3e-4,-5e-4,-1e-4))}
        self.assertTrue(study.decide(better,self.reports(),self.reports())['ship'])
        drift={**better,'totals':self.market((0,-1e-4,1e-4),ece=(.01,.002,.018))}
        self.assertFalse(study.decide(drift,self.reports(),self.reports())['ship'],'calibration gap rose')
        noise={**better,'h2h':self.market((0,-1e-4,1e-4),ece=(.006,-.01,.02))}
        self.assertTrue(study.decide(noise,self.reports(),self.reports())['ship'],'a rise within its interval is noise')
        self.assertFalse(study.decide(better,self.reports(),self.reports('batter_rbis'))['ship'],'postseason check flipped')

    def test_paired_intervals_resample_games(self):
        rng=np.random.default_rng(1)
        inverse,weights=study.resample([1,1,2,2,3,3],rng)
        self.assertEqual(weights.shape,(study.BOOTSTRAP,3))
        np.testing.assert_array_equal(weights.sum(axis=1),3)
        result=study.interval([.1,.1,-.1,-.1,0,0],inverse,weights)
        self.assertAlmostEqual(result['mean'],0)
        self.assertLessEqual(result['low'],0);self.assertGreaterEqual(result['high'],0)
        # Bootstrap gaps use exactly score()'s bins: unit weights reproduce its calibration gap.
        p=np.random.default_rng(2).random(60);y=(p>.5).astype(int);ids=np.repeat(np.arange(20),3)
        inverse,_=study.resample(ids,rng)
        gap=study.calibration_gaps(p,y,inverse,np.ones((1,20)))[0]
        self.assertAlmostEqual(gap,study.score(p,y,[.5]*60,ids)['ece'],places=5)
        a={m:dict(p=[.6,.4],y=[1,0],b=[.5,.5],ids=[1,2]) for m in study.MARKETS}
        b={m:dict(p=[.5,.5],y=[1,0],b=[.5,.5],ids=[1,2]) for m in study.MARKETS}
        self.assertLess(study.paired(a,b,rng)['totals']['brier']['mean'],0)
        b['totals']['ids']=[1,3]
        with self.assertRaisesRegex(ValueError,'identical forecasts'):study.paired(a,b,rng)

    def test_fold_zero_is_train_py_split(self):
        from mlb.train import chronology
        history=games(5)
        latest,train_end,cal_end=chronology(history)
        self.assertEqual(study.fold_dates(history,0),(train_end,cal_end,latest.isoformat()))
        self.assertEqual(study.fold_dates(history,1),('2026-05-07','2026-06-06','2026-07-06'))


if __name__=='__main__':unittest.main()

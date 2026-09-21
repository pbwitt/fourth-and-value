"""Pregame MLB features and discrete forecast distributions.

State is updated only after an entire historical date has been forecast.
No current-game box score or market price enters any feature vector.
"""
from collections import defaultdict
from datetime import date, timedelta
import math

import numpy as np
from scipy.stats import nbinom, norm, poisson

VERSION = 'mlb-v1.0'
TARGETS = ['team_runs', 'pitcher_strikeouts', 'pitcher_outs', 'batter_hits',
           'batter_total_bases', 'batter_home_runs', 'batter_rbis']
STAT = dict(zip(TARGETS[1:], ['strikeOuts', 'outs', 'hits', 'totalBases', 'homeRuns', 'rbi']))
SUPPORT = dict(zip(TARGETS, [35, 25, 36, 10, 24, 6, 15]))
LINES = {'totals':[6.5,7.5,8.5,9.5,10.5], 'h2h':[0], 'spreads':[-1.5,1.5],
         'pitcher_strikeouts':[2.5,3.5,4.5,5.5,6.5,7.5,8.5],
         'pitcher_outs':[12.5,15.5,17.5,18.5], 'batter_hits':[.5,1.5,2.5],
         'batter_total_bases':[.5,1.5,2.5,3.5], 'batter_home_runs':[.5], 'batter_rbis':[.5,1.5]}


def avg(rows, key, prior, weight=0):
    return (sum(r.get(key,0) for r in rows)+weight*prior)/(len(rows)+weight) if rows or weight else prior


def rate(rows, num, den, prior, weight):
    return (sum(r.get(num,0) for r in rows)+weight*prior)/(sum(r.get(den,0) for r in rows)+weight)


class State:
    def __init__(self):
        self.batters=defaultdict(list)
        self.pitchers=defaultdict(list)
        self.teams=defaultdict(list)
        self.parks=defaultdict(list)
        self.last_date=None

    def past(self, collection, key, day, limit):
        cutoff=(date.fromisoformat(day)-timedelta(days=370)).isoformat()
        return [r for r in collection.get(key,[]) if cutoff<=r['date']<day][-limit:]

    def team(self, team_id, day):
        rows=self.past(self.teams,team_id,day,40)
        recent=[r for r in rows if (date.fromisoformat(day)-date.fromisoformat(r['date'])).days<=3]
        return dict(team_games=len(rows), runs=avg(rows,'runs',4.5,12), allowed=avg(rows,'allowed',4.5,12),
            pa=avg(rows,'pa',38,12), k_rate=rate(rows,'k','pa',.225,400),
            hit_rate=rate(rows,'hits','pa',.22,400), hr_rate=rate(rows,'hr','pa',.03,400),
            bb_rate=rate(rows,'bb','pa',.085,400), tb_rate=rate(rows,'tb','pa',.36,400),
            bullpen_ra9=27*rate(rows,'bull_runs','bull_outs',4.3/27,180),
            bullpen_workload=sum(r['bull_pitches'] for r in recent))

    def pitcher(self, player_id, day):
        rows=self.past(self.pitchers,player_id,day,15)
        recent=rows[-5:]
        rest=min(30,(date.fromisoformat(day)-date.fromisoformat(rows[-1]['date'])).days) if rows else 30
        return dict(starts=len(rows), outs5=avg(recent,'outs',15.5,2), outs15=avg(rows,'outs',15.5,3),
            pitches5=avg(recent,'numberOfPitches',85,2), bf=avg(rows,'battersFaced',22,3),
            k_rate=rate(rows,'strikeOuts','battersFaced',.225,100),
            bb_rate=rate(rows,'baseOnBalls','battersFaced',.085,100),
            hit_rate=rate(rows,'hits','battersFaced',.22,100),
            hr_rate=rate(rows,'homeRuns','battersFaced',.03,100),
            ra9=27*rate(rows,'runs','outs',4.3/27,90), rest=rest)

    def features(self, game, side, player_id=None, slot=0):
        day=game['date'];other='away' if side=='home' else 'home'
        team_id=game[side+'_id'];opp_id=game[other+'_id']
        starter=game[side+'_starter'];opponent=game[other+'_starter']
        team=self.team(team_id,day);opp=self.team(opp_id,day)
        pitcher=self.pitcher(starter,day);op=self.pitcher(opponent,day)
        park=self.past(self.parks,game['venue'],day,100)
        park_factor=avg(park,'total',9,40)/9
        base={'home':float(side=='home'),'postseason':float(game['game_type']!='R'), 'park_factor':park_factor}
        base.update({'team_'+k:v for k,v in team.items()})
        base.update({'opp_'+k:v for k,v in opp.items()})
        base.update({'starter_'+k:v for k,v in pitcher.items()})
        base.update({'opp_starter_'+k:v for k,v in op.items()})
        if player_id is not None:
            rows=self.past(self.batters,player_id,day,60)
            starts=[r for r in rows if r['slot']>0][-20:]
            pa=max(3.0,min(5.4,4.7-.115*slot-.1*(side=='home')+.04*(team['runs']-4.5)))
            base.update(batter_games=len(rows),batter_pa=sum(r['plateAppearances'] for r in rows),
                lineup_slot=slot,projected_pa=pa,recent_pa=avg(starts,'plateAppearances',pa,5),
                batter_hit_rate=rate(rows,'hits','plateAppearances',.22,100),
                batter_tb_rate=rate(rows,'totalBases','plateAppearances',.36,100),
                batter_hr_rate=rate(rows,'homeRuns','plateAppearances',.03,150),
                batter_rbi_rate=rate(rows,'rbi','plateAppearances',.11,100),
                batter_k_rate=rate(rows,'strikeOuts','plateAppearances',.225,100),
                batter_bb_rate=rate(rows,'baseOnBalls','plateAppearances',.085,100))
        return base

    def update(self, game):
        day=game['date']
        for side in ['home','away']:
            team=game['teams'][side];other=game['teams']['away' if side=='home' else 'home']
            starter=next(p for p in team['pitchers'] if p['id']==team['starter'])
            b,p=team['batting'],team['pitching']
            self.teams[team['id']].append(dict(date=day,runs=b['runs'],allowed=other['batting']['runs'],
                pa=b['plateAppearances'],k=b['strikeOuts'],hits=b['hits'],hr=b['homeRuns'],bb=b['baseOnBalls'],
                tb=b['totalBases'],bull_runs=max(0,p['runs']-starter['runs']),
                bull_outs=max(0,p['outs']-starter['outs']),bull_pitches=max(0,p['numberOfPitches']-starter['numberOfPitches'])))
            self.pitchers[starter['id']].append({**starter,'date':day})
            for batter in team['batters']:
                self.batters[batter['id']].append({**batter,'date':day})
        self.parks[game['venue']].append(dict(date=day,total=game['home_score']+game['away_score']))
        self.last_date=day


def game_input(game):
    return {**game, 'home_starter':game['teams']['home']['starter'], 'away_starter':game['teams']['away']['starter']}


def dataset(games):
    state=State();samples={target:[] for target in TARGETS};by_day=defaultdict(list)
    for game in games:by_day[game['date']].append(game)
    for day in sorted(by_day):
        for game in by_day[day]:
            data=game_input(game)
            for side in ['home','away']:
                team=game['teams'][side];features=state.features(data,side)
                meta=dict(game_id=game['id'],date=day,game_type=game['game_type'],side=side,
                          home_score=game['home_score'],away_score=game['away_score'])
                if features['team_team_games']<10 or features['opp_team_games']<10:
                    continue
                samples['team_runs'].append({**meta,'x':features,'y':team['batting']['runs']})
                starter=next(p for p in team['pitchers'] if p['id']==team['starter'])
                if features['starter_starts']>=3:
                    for target in ['pitcher_strikeouts','pitcher_outs']:
                        samples[target].append({**meta,'player_id':starter['id'],'x':features,'y':starter[STAT[target]]})
                for batter in team['batters']:
                    if not 1<=batter['slot']<=9:continue
                    x=state.features(data,side,batter['id'],batter['slot'])
                    if x['batter_pa']<50:continue
                    for target in TARGETS[3:]:
                        samples[target].append({**meta,'player_id':batter['id'],'x':x,'y':batter[STAT[target]]})
        # All games on a date share exactly the same pre-date information.
        for game in by_day[day]:state.update(game)
    return samples,state


def baseline(x,target):
    if target=='team_runs':return max(.5,(x['team_runs']+x['opp_allowed'])/2*x['park_factor'])
    if target=='pitcher_outs':return x['starter_outs5']
    if target=='pitcher_strikeouts':
        return x['starter_bf']*x['starter_k_rate']*(x['opp_k_rate']/.225)**.5
    key={'batter_hits':'hit','batter_total_bases':'tb','batter_home_runs':'hr','batter_rbis':'rbi'}[target]
    return x['projected_pa']*x['batter_'+key+'_rate']


def means(model,samples):
    if model['kind']=='rolling':
        values=np.array([baseline(s['x'],model['target']) for s in samples])
    else:
        matrix=np.array([[s['x'][k] for k in model['features']] for s in samples])
        values=model['estimator'].predict(matrix)
    return np.clip(values,.01,SUPPORT[model['target']]*.7)


def raw_cdf(mu,model):
    support=SUPPORT[model['target']];k=np.arange(support)[None,:];mu=np.asarray(mu)[:,None]
    if model['target']=='pitcher_outs':
        return norm.cdf((k+.5-mu)/model['sigma'])
    alpha=model['alpha']
    if alpha<.001:return poisson.cdf(k,mu)
    size=1/alpha
    return nbinom.cdf(k,size,size/(size+mu))


def pmf(mu,model):
    cdf=raw_cdf(mu,model)
    if model.get('calibrator') is not None:
        cdf=model['calibrator'].predict(cdf.ravel()).reshape(cdf.shape)
    # A small raw-distribution component prevents 0/1 certainty in unseen tails.
    cdf=.98*cdf+.02*raw_cdf(mu,model)
    cdf=np.maximum.accumulate(np.clip(cdf,0,1),axis=1)
    return np.diff(np.column_stack([np.zeros(len(cdf)),cdf,np.ones(len(cdf))]),axis=1)


def outcome(mass,line,side='Over'):
    values=np.arange(len(mass));win=values>line if side=='Over' else values<line
    return float(mass[win].sum()),float(mass[values==line].sum())


def joint(home,away):
    values=np.outer(home,away)
    np.fill_diagonal(values,0)  # Completed MLB games cannot end tied.
    return values/values.sum()


def game_outcome(j,market,line=None,home=True,side='Over'):
    h,a=np.indices(j.shape)
    if market=='h2h':return float(j[h>a if home else a>h].sum()),0.0
    values=h+a if market=='totals' else (h-a if home else a-h)+line
    boundary=line if market=='totals' else 0
    wins=values>boundary if market!='totals' or side=='Over' else values<boundary
    return float(j[wins].sum()),float(j[values==boundary].sum())


def expected_return(win,push,american):
    profit=american/100 if american>0 else 100/abs(american)
    return win*profit-(1-win-push)

"""Reproducible chronological predictive evaluation; never fabricate betting returns."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import scipy
import sklearn

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='nhl.v2'
from . import VERSION, FEATURE_SCHEMA
from .data import ROOT, load, write_json, digest
from .features import build, PLAYER_STATS
from .models import TeamModel, PlayerModel, TEAM_CANDIDATES, PLAYER_CANDIDATES, game_outcome


def binary_metrics(p,y):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9); y=np.asarray(y,dtype=float)
    bins=[]
    for lo in np.linspace(0,.9,10):
        mask=(np.floor(p*10)==round(lo*10))
        if mask.any(): bins.append(dict(lower=round(float(lo),1),count=int(mask.sum()),forecast=float(p[mask].mean()),observed=float(y[mask].mean())))
    return dict(n=len(y),log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
                brier=float(np.mean((p-y)**2)),calibration=bins,
                ece=float(sum(b['count']*abs(b['forecast']-b['observed']) for b in bins)/len(y)))


def interval_mean(values,groups):
    """Game-cluster bootstrap; avoids treating related players/markets as independent."""
    clustered=defaultdict(list)
    for v,g in zip(values,groups): clustered[g].append(float(v))
    sums=np.array([sum(v) for v in clustered.values()]); counts=np.array([len(v) for v in clustered.values()])
    rng=np.random.default_rng(712)
    indexes=rng.integers(0,len(sums),(500,len(sums)))
    means=sums[indexes].sum(axis=1)/counts[indexes].sum(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def team_metrics(model,rows,games,save=None):
    means=model.predict(rows)
    by_id=defaultdict(dict)
    for r,m in zip(rows,means): by_id[r['game_id']]['home' if r['home'] else 'away']=float(m)
    p,observed,errors,nll,ids = defaultdict(list),defaultdict(list),[],[],[]
    for g in games:
        m=by_id[g['game_id']]; joint=model.joint(m['home'],m['away'])
        h,a=g['home_score'],g['away_score']
        nll.append(-np.log(max(joint[h,a],1e-12))); ids.append(g['game_id'])
        H,A=np.indices(joint.shape); errors.append(float((joint*(H+A)).sum())-(h+a))
        forecasts={
            'moneyline':game_outcome(joint,'h2h',None)['win'],
            'total_5.5':game_outcome(joint,'totals',5.5)['win'],
            'total_6.5':game_outcome(joint,'totals',6.5)['win'],
            'puck_-1.5':game_outcome(joint,'spreads',-1.5)['win']}
        actuals={'moneyline':int(h>a),'total_5.5':int(h+a>5.5),'total_6.5':int(h+a>6.5),'puck_-1.5':int(h-a>1.5)}
        for k,v in forecasts.items(): p[k].append(v); observed[k].append(actuals[k])
        if save:
            save.write(json.dumps(dict(game_id=g['game_id'],game_date=g['game_date'],season=g['season'],
                model=model.kind,home_reg_mean=m['home'],away_reg_mean=m['away'],probabilities=forecasts,
                actual_home=h,actual_away=a,score_nll=float(nll[-1]),settlement='full_game_ot_so',
                decision_at=next(r['decision_at'] for r in rows if r['game_id']==g['game_id'])))+'\n')
    return dict(games=len(games),joint_log_loss=float(np.mean(nll)),joint_log_loss_ci=interval_mean(nll,ids),
                total_mae=float(np.mean(np.abs(errors))),total_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                markets={k:binary_metrics(p[k],observed[k]) for k in p})


def player_metrics(model,rows,save=None,only=None):
    pmfs=model.fast_pmfs(rows); target=np.array([r['targets'] for r in rows]); result={}
    for j,stat in enumerate(PLAYER_STATS):
        if only and stat not in only: continue
        p=pmfs[j]; y=target[:,j]; mean=p@np.arange(p.shape[1]); nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
        line=2.5 if j==0 else .5
        over=p[:,np.arange(p.shape[1])>line].sum(axis=1)
        lower=np.argmax(p.cumsum(axis=1)>=.05,axis=1); upper=np.argmax(p.cumsum(axis=1)>=.95,axis=1)
        result[stat]=dict(count_log_loss=float(nll.mean()),count_log_loss_ci=interval_mean(nll,[r['game_id'] for r in rows]),
            mae=float(np.mean(abs(mean-y))),rmse=float(np.sqrt(np.mean((mean-y)**2))),
            observed_zero=float(np.mean(y==0)),predicted_zero=float(p[:,0].mean()),
            interval90_coverage=float(np.mean((y>=lower)&(y<=upper))),
            rps=float(np.mean(np.sum((p.cumsum(axis=1)-(np.arange(p.shape[1])[None,:]>=y[:,None]))**2,axis=1))),
            reference_line=line,**binary_metrics(over,y>line))
        if save:
            for i,r in enumerate(rows):
                save.write(json.dumps(dict(game_id=r['game_id'],player_id=r['player_id'],game_date=r['game_date'],
                    season=r['season'],decision_at=r['decision_at'],feature_cutoff=r['feature_cutoff'],
                    market=stat,model=model.kind,mean=float(mean[i]),line=line,over=float(over[i]),
                    actual=int(y[i]),count_log_loss=float(nll[i]),conditional_on_participation=True))+'\n')
    return result


def window(rows):
    return dict(start=min(r['game_date'] for r in rows),end=max(r['game_date'] for r in rows),n=len(rows))


def run(history_root,output,model_dir):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    games,players,manifests=load(history_root)
    if sorted({g['season'] for g in games}) != [20222023,20232024,20242025,20252026]:
        raise ValueError('Evaluation requires all four specified seasons; do not silently change the final test')
    if any(sum(g['season']==s for g in games)!=1312 for s in [20222023,20232024,20242025,20252026]):
        raise ValueError('Incomplete regular season')
    print('Building strictly lagged team and player features...',flush=True)
    cache=Path(history_root)/'features.joblib'
    feature_key=digest([manifests,Path(__file__).with_name('features.py').read_text()])
    cached=joblib.load(cache) if cache.exists() else {}
    if cached.get('key')==feature_key:
        tr,pr=cached['rows']
    else:
        tr,pr=build(games,players)
        joblib.dump(dict(key=feature_key,rows=(tr,pr)),cache,compress=3)
    report=dict(version=VERSION,feature_schema=FEATURE_SCHEMA,history_sha256=digest(manifests),
        source_manifests=manifests,validation=[],final={},
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        limitations=['Reconstructed next-day availability; original publication/revision timestamps unavailable.',
          'Player evaluation is conditional on actual participation; no retrospective lineup features.',
          'Reference lines are fixed diagnostic thresholds, not claims of posted prices.',
          'No timestamped settled quote sample: market benchmark, learned blend, ROI and CLV unavailable.',
          'No final-test-driven policy selection; all production candidates remain experimental.'])
    for season in [20232024,20242025]:
        train=[r for r in tr if r['season']<season]; valid=[r for r in tr if r['season']==season]
        tg=[g for g in games if g['season']<season]; vg=[g for g in games if g['season']==season]
        pt=[r for r in pr if r['season']<season]; pv=[r for r in pr if r['season']==season]
        fold=dict(season=season,training=window(train),validation=window(valid),team={},player={})
        for name in TEAM_CANDIDATES:
            m=TeamModel(name).fit(train,tg); fold['team'][name]=team_metrics(m,valid,vg)
            print(f'Validation {season} {name}: {fold["team"][name]["joint_log_loss"]:.5f}',flush=True)
        for name in PLAYER_CANDIDATES:
            m=PlayerModel(name).fit(pt); fold['player'][name]=player_metrics(m,pv)
            print(f'Validation {season} {name}: '+str({s:round(v['count_log_loss'],4) for s,v in fold['player'][name].items()}),flush=True)
        report['validation'].append(fold)
    selected_team=min(TEAM_CANDIDATES,key=lambda n:np.mean([f['team'][n]['joint_log_loss'] for f in report['validation']]))
    selected_shots=min(PLAYER_CANDIDATES,key=lambda n:np.mean([f['player'][n]['shots']['count_log_loss'] for f in report['validation']]))
    selected_scoring=min(PLAYER_CANDIDATES,key=lambda n:np.mean([f['player'][n][s]['count_log_loss'] for f in report['validation'] for s in PLAYER_STATS[1:]]))
    selection=dict(team=selected_team,shots=selected_shots,scoring=selected_scoring,chosen_using='2023–24 and 2024–25 only',
                   market_weight=0,calibration='identity; no post-hoc probability remapping',recommendations_enabled=False)
    write_json(output/'selection-lock.json',selection)
    print('Selection locked before final test: '+str(selection),flush=True)
    train=[r for r in tr if r['season']<20252026]; test=[r for r in tr if r['season']==20252026]
    tg=[g for g in games if g['season']<20252026]; fg=[g for g in games if g['season']==20252026]
    pt=[r for r in pr if r['season']<20252026]; pv=[r for r in pr if r['season']==20252026]
    report['final']['training']=window(train); report['final']['test']=window(test)
    report['final']['team']={}; report['final']['player']={}
    with gzip.open(output/'team-predictions.jsonl.gz','wt') as save:
        for name in dict.fromkeys(['rate',selected_team]):
            m=TeamModel(name).fit(train,tg)
            report['final']['team'][name]=team_metrics(m,test,fg,save)
    with gzip.open(output/'player-predictions.jsonl.gz','wt') as save:
        for name in dict.fromkeys(['rate_poisson',selected_shots,selected_scoring]):
            m=PlayerModel(name).fit(pt)
            report['final']['player'][name]=player_metrics(m,pv,save)
    report['selection']=selection
    report['betting_evaluation']=dict(status='blocked_missing_timestamped_historical_odds',bets=0,turnover=0,
        net_units=None,roi=None,drawdown=None,odds_distribution=None,execution_sensitivity=None,clv=None)
    # Same selected algorithms; refit on all now-completed seasons for live use only.
    bundle=dict(team=TeamModel(selected_team).fit(tr,games),shots=PlayerModel(selected_shots).fit(pr),
                scoring=PlayerModel(selected_scoring).fit(pr))
    model_dir=Path(model_dir); model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,model_dir/'models.joblib',compress=3)
    import hashlib
    model_sha=hashlib.sha256((model_dir/'models.joblib').read_bytes()).hexdigest()
    card=dict(version=VERSION,feature_schema=FEATURE_SCHEMA,selection=selection,artifact_sha256=model_sha,
              trained_through=max(g['game_date'] for g in games),training_window=window(games),
              validation_status='experimental_predictive_evaluation_only',history_sha256=report['history_sha256'],
              environment=report['environment'],created_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    # Frozen completed-game history permits deterministic offline inference and cold starts.
    with gzip.GzipFile(filename=str(model_dir/'history.json.gz'),mode='wb',mtime=0) as f:
        f.write(json.dumps(dict(games=games,players=players),separators=(',',':')).encode())
    card['history_archive_sha256']=hashlib.sha256((model_dir/'history.json.gz').read_bytes()).hexdigest()
    write_json(model_dir/'manifest.json',card); write_json(output/'evaluation.json',report)
    write_json(output/'grading-summary.json',dict(team_games=len(fg),player_appearances=len(pv),
        grading='Official full-game scores; player statistics exclude shootouts. No betting ledger mutated.',betting=report['betting_evaluation']))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,default=ROOT/'data/nhl/v2/history')
    p.add_argument('--output',type=Path,default=ROOT/'reports/nhl-rebuild')
    p.add_argument('--models',type=Path,default=ROOT/'models/nhl/v2')
    a=p.parse_args(); run(a.history,a.output,a.models)

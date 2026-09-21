"""Train, calibrate and audit MLB models with strictly chronological partitions."""
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys

# Keep CI and laptops from creating hundreds of nested numerical threads.
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from datetime import date, datetime, timedelta, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from nba.pipeline import save_json,iso
from mlb.model_data import load,update
from mlb.models import TARGETS,SUPPORT,LINES,VERSION,dataset,baseline,means,raw_cdf,pmf,joint,game_outcome

MODEL_PATH=ROOT/'data/mlb/models/current.joblib'
REPORT_PATH=ROOT/'docs/mlb/data/validation.json'


def signature():
    source=b''.join((Path(__file__).parent/name).read_bytes() for name in ['models.py','train.py','model_data.py'])
    return hashlib.sha256(source+sklearn.__version__.encode()).hexdigest()[:16]


def fit_model(target,train,cal):
    if len(train)<150 or len(cal)<80:raise ValueError(f'{target}: insufficient chronological training/calibration samples')
    features=sorted(train[0]['x'])
    x=np.array([[r['x'][k] for k in features] for r in train]);y=np.array([r['y'] for r in train])
    estimator=HistGradientBoostingRegressor(loss='squared_error' if target=='pitcher_outs' else 'poisson',
        learning_rate=.05,max_iter=100,max_leaf_nodes=7,min_samples_leaf=60,l2_regularization=20,
        early_stopping=False,random_state=42).fit(x,y)
    model=dict(target=target,features=features,estimator=estimator,kind='boosted',alpha=0,sigma=4)
    yc=np.array([r['y'] for r in cal]);boost=means(model,cal)
    rolling=np.array([baseline(r['x'],target) for r in cal])
    if np.mean((rolling-yc)**2)<=np.mean((boost-yc)**2):model['kind']='rolling'
    mu=means(model,cal)
    model['alpha']=float(np.clip(np.sum((yc-mu)**2-yc)/np.sum(mu**2),0,2))
    model['sigma']=float(max(1,np.sqrt(np.mean((yc-mu)**2))))
    cdf=raw_cdf(mu,model)
    labels=(yc[:,None]<=np.arange(SUPPORT[target])[None,:]).astype(float)
    model['calibrator']=IsotonicRegression(out_of_bounds='clip').fit(cdf.ravel(),labels.ravel())
    counts=np.bincount(np.minimum(y.astype(int),SUPPORT[target]),minlength=SUPPORT[target]+1).astype(float)+.25
    model['reference_pmf']=counts/counts.sum()
    model['train_n']=len(train);model['cal_n']=len(cal)
    return model


def score(prob,actual,reference,game_ids,post=False):
    p=np.asarray(prob);y=np.asarray(actual);b=np.asarray(reference)
    if not len(y):return dict(samples=0,games=0,passed=False)
    brier=float(np.mean((p-y)**2));ref=float(np.mean((b-y)**2));ece=0
    bins=[]
    for low in np.arange(0,1,.1):
        selected=(p>=low)&(p<low+.1 if low<.9 else p<=1)
        if selected.any():
            pred=float(p[selected].mean());observed=float(y[selected].mean());n=int(selected.sum())
            ece+=n/len(y)*abs(pred-observed)
            bins.append(dict(predicted=round(pred,4),observed=round(observed,4),n=n))
    n_games=len(set(game_ids))
    # These are predeclared checks, not proof of profitable betting.
    passed=n_games>=(30 if post else 150) and brier<ref and ece<=(.12 if post else .06)
    return dict(samples=len(y),games=n_games,brier=round(brier,6),reference_brier=round(ref,6),
        brier_skill=round(1-brier/ref,4) if ref else None,ece=round(float(ece),6),
        log_loss=round(float(-np.mean(y*np.log(np.clip(p,1e-6,1-1e-6))+(1-y)*np.log(np.clip(1-p,1e-6,1-1e-6)))),6),
        passed=bool(passed),calibration_bins=bins)


def evaluate(models,samples,post=False):
    report={};games=default_game_pairs(samples['team_runs'])
    for target in TARGETS[1:]:
        rows=samples[target];model=models[target]
        if not rows:report[target]=dict(samples=0,games=0,passed=False);continue
        mu=means(model,rows);mass=pmf(mu,model);prob=[];actual=[];reference=[];ids=[]
        for index,row in enumerate(rows):
            for line in LINES[target]:
                prob.append(float(mass[index,int(line)+1:].sum()));actual.append(int(row['y']>line))
                reference.append(float(model['reference_pmf'][int(line)+1:].sum()));ids.append(row['game_id'])
        report[target]=score(prob,actual,reference,ids,post)
        report[target].update(forecasts=len(rows),mae=round(float(np.mean(np.abs(mu-[r['y'] for r in rows]))),4),
            mean_prediction=round(float(mu.mean()),4),mean_actual=round(float(np.mean([r['y'] for r in rows])),4),
            lines=LINES[target])
    totals={market:dict(p=[],y=[],b=[],ids=[]) for market in ['totals','h2h','spreads']}
    model=models['team_runs'];rows=samples['team_runs'];mass=pmf(means(model,rows),model)
    reference=joint(model['reference_pmf'],model['reference_pmf'])
    for game_id,pair in games.items():
        if set(pair)!= {'home','away'}:continue
        hi,ai=pair['home'],pair['away'];row=rows[hi]
        matrix=joint(mass[hi],mass[ai]);home,away=row['home_score'],row['away_score']
        for market,values in totals.items():
            for line in LINES[market]:
                p,_=game_outcome(matrix,market,line);b,_=game_outcome(reference,market,line)
                y=int(home+away>line) if market=='totals' else int(home>away) if market=='h2h' else int(home-away+line>0)
                values['p'].append(p);values['y'].append(y);values['b'].append(b);values['ids'].append(game_id)
    for market,v in totals.items():report[market]={**score(v['p'],v['y'],v['b'],v['ids'],post),'lines':LINES[market]}
    return report


def default_game_pairs(rows):
    pairs={}
    for i,row in enumerate(rows):pairs.setdefault(row['game_id'],{})[row['side']]=i
    return pairs


def partition(samples,train_end,cal_end,test_end,post_only=False):
    train={};cal={};test={}
    for target,rows in samples.items():
        train[target]=[r for r in rows if r['date']<=train_end]
        cal[target]=[r for r in rows if train_end<r['date']<=cal_end]
        test[target]=[r for r in rows if cal_end<r['date']<=test_end and (r['game_type']!='R' if post_only else r['game_type']=='R')]
    return train,cal,test


def train_models(refresh_history=True):
    if refresh_history:update()
    games,manifest=load()
    if len(games)<1000:raise ValueError('At least 1,000 completed games are required')
    missing=manifest['expected_games']-len(games)
    if missing:raise ValueError(f'{missing} expected game observations are missing; model refresh stopped')
    if MODEL_PATH.exists():
        cached=joblib.load(MODEL_PATH)
        if cached.get('source_signature')==signature() and cached.get('history_fetched_date')==manifest['through_date']:
            save_json(REPORT_PATH,cached['report']);return cached
    print('MLB model: building strictly pre-date features',flush=True)
    samples,state=dataset(games)
    latest=date.fromisoformat(games[-1]['date']);train_end=(latest-timedelta(days=60)).isoformat();cal_end=(latest-timedelta(days=30)).isoformat()
    training,calibration,test=partition(samples,train_end,cal_end,latest.isoformat())
    models={}
    for target in TARGETS:
        print(f'MLB model: {target}; train={len(training[target])}, calibration={len(calibration[target])}, test={len(test[target])}',flush=True)
        models[target]=fit_model(target,training[target],calibration[target])
    regular=evaluate(models,test)
    # Historical postseason audit uses an entirely earlier model, trained BEFORE that postseason.
    prior_year=latest.year-1
    reg_dates=[g['date'] for g in games if g['season']==prior_year and g['game_type']=='R']
    reg_end=max(reg_dates);post_train_end=(date.fromisoformat(reg_end)-timedelta(days=21)).isoformat()
    pt,pc,pe=partition(samples,post_train_end,reg_end,f'{prior_year}-12-31',post_only=True)
    postseason_error=None
    try:
        print('MLB model: independent prior-postseason audit',flush=True)
        old_models={target:fit_model(target,pt[target],pc[target]) for target in TARGETS}
        postseason=evaluate(old_models,pe,post=True)
    except ValueError as error:
        postseason={};postseason_error=str(error)
    version=VERSION+'-'+signature()+'-'+latest.isoformat()
    inputs=json.dumps(games,separators=(',',':')).encode()
    report=dict(version=version,created_at=iso(datetime.now(timezone.utc)),games=len(games),
        inputs_sha256=hashlib.sha256(inputs).hexdigest(),sklearn_version=sklearn.__version__,
        input_through=latest.isoformat(),training_through=train_end,calibration_through=cal_end,
        test_start=(date.fromisoformat(cal_end)+timedelta(days=1)).isoformat(),test_end=latest.isoformat(),
        postseason_year=prior_year,postseason_training_through=post_train_end,postseason_calibration_through=reg_end,
        regular=regular,postseason=postseason,postseason_error=postseason_error,
        models={k:dict(kind=m['kind'],train_n=m['train_n'],calibration_n=m['cal_n'],features=m['features'],
                       dispersion=m['alpha'],outs_sigma=m['sigma'] if k=='pitcher_outs' else None) for k,m in models.items()},
        reference='Empirical outcome distribution in the training partition, independent of sportsbook prices',
        thresholds=dict(regular_min_games=150,regular_max_brier_excess=0,regular_max_ece=.06,
                        postseason_min_games=30,postseason_max_brier_excess=0,postseason_max_ece=.12),
        limitations=['Fixed threshold grids, not archived historical bookmaker lines; no historical ROI claim.',
            'Historical evaluation conditions on actual starters and original batting order; archived pregame announcements are unavailable.',
            'Postseason audit is small and uses an earlier model fitted before that postseason; upcoming playoff accuracy is unproven.',
            'No explicit handedness, weather, injury, umpire or tactical pinch-hit adjustments.',
            'Park estimates are shrunk venue scoring averages and can include home-team effects.',
            'Joint team-run approximation conditions out ties; extra-inning dynamics are not simulated pitch by pitch.'])
    bundle=dict(version=version,source_signature=signature(),history_fetched_date=manifest['through_date'],
        models=models,state=state,report=report)
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(MODEL_PATH.parent/'training_inputs.json.gz','wb') as stream:stream.write(inputs)
    temp=MODEL_PATH.with_suffix('.tmp');joblib.dump(bundle,temp);temp.replace(MODEL_PATH)
    save_json(REPORT_PATH,report)
    print('MLB model: validation report saved',flush=True)
    for market,metrics in regular.items():print(market,metrics.get('brier'),metrics.get('reference_brier'),metrics['passed'],flush=True)
    return bundle


if __name__=='__main__':
    train_models(refresh_history='--cached-history' not in sys.argv)

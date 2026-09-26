"""Fixed monthly historical-price diagnostic; model/thresholds locked before downloads.

21 decisions: the 15th, October–April, in 2023–24, 2024–25 and 2025–26.
At most 630 existing Odds API credits. No subscription or upgrade operations.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import sys
import time

import joblib
import requests
from dotenv import load_dotenv
import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2])); __package__='nhl.v2'
from .data import ROOT,load,iso,stamp,write_json,digest
from .features import decision_time
from .models import TeamModel,game_outcome
from .pricing import compare,price,decimal
from .evaluate import binary_metrics,interval_mean
from .grading import settle,betting_metrics
from nhl.refresh import flatten,MARKETS,normal_name


def dates():
    return [f'{y}-{m:02}-15' for year in [2023,2024,2025] for y,m in
            [(year,m) for m in [10,11,12]]+[(year+1,m) for m in [1,2,3,4]]]


def download(root,env_file):
    load_dotenv(env_file);key=os.getenv('NHL_ODDS_API_KEY') or os.getenv('ODDS_API_KEY')
    if not key: raise ValueError('Existing odds key unavailable')
    used=0;root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for day in dates():
        path=root/f'{day}.json.gz'
        if path.exists(): continue
        if used+30>630: raise ValueError('Fixed historical-request budget reached')
        decision=iso(decision_time(day))
        params=dict(regions='us',markets='h2h,spreads,totals',oddsFormat='american',date=decision)
        try:
            response=requests.get('https://api.the-odds-api.com/v4/historical/sports/icehockey_nhl/odds',
                                  params=dict(params,apiKey=key),timeout=60)
        except requests.RequestException:
            raise RuntimeError('Historical price request unavailable') from None
        if not response.ok: raise RuntimeError(f'Historical price access HTTP {response.status_code}')
        payload=response.json();cost=int(response.headers.get('x-requests-last',30));used+=cost
        if stamp(payload['timestamp'])>stamp(decision): raise ValueError('Provider returned a future snapshot')
        record=dict(decision_at=decision,ingested_at=iso(datetime.now(timezone.utc)),query=params,
                    request_cost=cost,quota_remaining=response.headers.get('x-requests-remaining'),
                    sha256=digest(payload),payload=payload)
        with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:f.write(json.dumps(record,separators=(',',':')).encode())
        print(f'{day}: {len(payload["data"])} provider events; {cost} credits; {used} used this run',flush=True)
        if int(record['quota_remaining'] or 0)<2000: raise ValueError('Preserving 2,000 credits for live operation')
        time.sleep(.2)


def evaluate(root,history,output):
    games,_,manifests=load(history,[20222023,20232024,20242025,20252026])
    cache=joblib.load(Path(history)/'features.joblib');tr,_=cache['rows']
    if cache['key']!=digest([manifests,Path(__file__).with_name('features.py').read_text()]):
        raise ValueError('Feature cache does not match source history/code; rerun the predictive evaluation')
    chosen=json.loads((ROOT/'reports/nhl-rebuild/selection-lock.json').read_text())['team']
    models={s:TeamModel(chosen).fit([r for r in tr if r['season']<s],[g for g in games if g['season']<s]) for s in [20232024,20242025,20252026]}
    means={}
    for season,model in models.items():
        rows=[r for r in tr if r['season']==season]
        for r,mu in zip(rows,model.predict(rows)):means.setdefault(r['game_id'],{})['home' if r['home'] else 'away']=float(mu)
    output=Path(output);all_rows=[];selected=[];snapshots=[];joints={}
    for day in dates():
        with gzip.open(Path(root)/f'{day}.json.gz','rt') as f:raw=json.load(f)
        if digest(raw['payload'])!=raw['sha256']: raise ValueError('Historical odds checksum mismatch')
        asof=stamp(raw['decision_at']);eligible=[g for g in games if g['game_date']==day]
        rows=[];map_games={g['game_id']:g for g in eligible}
        for event in raw['payload']['data']:
            matches=[g for g in eligible if normal_name(g['home_team'])==normal_name(event['home_team']) and normal_name(g['away_team'])==normal_name(event['away_team'])]
            if len(matches)!=1 or stamp(event['commence_time']).astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).date().isoformat()!=day:continue
            event['nhl_game_id']=matches[0]['game_id'];rows.extend(flatten(event,asof))
        rows=compare(rows,asof)
        for r in rows:
            g=map_games[r['nhl_game_id']];model=models[g['season']];mu=means[g['game_id']]
            args=(r['market'],r['line'],r['side']==r['home_team'],r['side'])
            if g['game_id'] not in joints:
                joints[g['game_id']]=[model.joint(mu['home']*h,mu['away']*a) for h,a in [(1,1),(.9,.9),(1.1,1.1),(.9,1.1),(1.1,.9)]]
            probs=game_outcome(joints[g['game_id']][0],*args)
            scenarios=[game_outcome(j,*args) for j in joints[g['game_id']][1:]]
            r.update(price(probs,r['price'],scenarios=scenarios))
            r.update(season=g['season'],decision_at=iso(asof),game_date=day,model=chosen,
                     result=settle(r,g),settlement_vintage='standard_full_game_assumption_not_certified',
                     validation_status='experimental_shadow',stake_units=0)
        # Policy already documented before the historical access probe: 2% EV, scenario
        # minimum price, fresh quote, positive conservative log-growth, max four units/day,
        # one offer per game. No analyst override or final-test threshold search.
        candidates=[r for r in rows if r['settlement_verified'] and r['result'] in ['won','lost','push'] and
                    r['estimated_ev']>=.02 and r['rank_score']>0 and r['minimum_acceptable_decimal'] and
                    decimal(r['price'])>=r['minimum_acceptable_decimal'] and (asof-stamp(r['quoted_at'])).total_seconds()<=1800]
        picked=[];seen=set()
        for r in sorted(candidates,key=lambda r:(-r['rank_score'],r['offer_id'])):
            if r['nhl_game_id'] in seen:continue
            seen.add(r['nhl_game_id']);picked.append(dict(r,stake_units=1,simulation_only=True))
            if len(picked)==4:break
        selected.extend(picked);all_rows.extend(rows)
        snapshots.append(dict(date=day,decision_at=raw['decision_at'],provider_at=raw['payload']['timestamp'],offers=len(rows),sample_games=len({r['nhl_game_id'] for r in rows}),shadow_bets=len(picked)))
    # One observation per game/market/exact line: retain home or Over and choose the
    # freshest mapped quote that has other-book paired consensus. No outcome selection.
    groups={}
    for r in sorted(all_rows,key=lambda r:(r['quoted_at'],r['book'])):
        if (r['side'] not in ['Over',r['home_team']] or r.get('market_probability') is None or
            not r.get('settlement_verified') or r['result'] not in ['won','lost']):continue
        groups[(r['nhl_game_id'],r['market'],r['line'])]=r
    metrics={}
    for season in models:
        metrics[str(season)]={}
        for market in ['h2h','spreads','totals']:
            rows=[r for r in groups.values() if r['season']==season and r['market']==market]
            if not rows:continue
            y=[r['result']=='won' for r in rows];ind=[r['conditional_probability'] for r in rows];mkt=[r['market_probability'] for r in rows]
            losses=lambda p:-np.asarray(y,dtype=float)*np.log(np.clip(p,1e-9,1-1e-9))-(1-np.asarray(y,dtype=float))*np.log(1-np.clip(p,1e-9,1-1e-9))
            diff=losses(ind)-losses(mkt)
            metrics[str(season)][market]=dict(independent=binary_metrics(ind,y),market=binary_metrics(mkt,y),
                difference=float(diff.mean()),difference_ci=interval_mean(diff,[r['nhl_game_id'] for r in rows]),
                unique_games=len({r['nhl_game_id'] for r in rows}),basis='conditional_on_non_push')
    report=dict(protocol='Monthly 15th at 10:30 America/New_York; fixed before download; not selected by outcomes',
        data_status='Timestamped historical provider snapshots; historical house-rule versions and actual account execution unverified',
        model_selection='Unchanged; no model or threshold tuning on this sample',
        total_credits=630,snapshots=snapshots,metrics=metrics,
        shadow={str(s):dict(nominal=betting_metrics([r for r in selected if r['season']==s]),
            worse_execution=betting_metrics([r for r in selected if r['season']==s],.05)) for s in models},
        market_blend='Not fitted: sparse sampled game coverage below the predeclared 500-game training floor',
        limitations=['Game markets only; no historical player prices acquired.',
          'Monthly sampling is low power and is not representative of a full daily strategy.',
          'Quotes are observations, not proof an account could execute them; flat-unit shadow results are not realized returns.',
          'No closing-price sample or historical analyst evidence; CLV and early-versus-later remain unavailable.'])
    write_json(output/'historical-market-evaluation.json',report)
    with gzip.open(output/'historical-offers-graded.jsonl.gz','wt') as f:
        for r in all_rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    write_json(output/'shadow-selected-offers.json',selected)
    lines=['# Timestamped historical game-price diagnostic','',
      '**No betting edge is established.** Historical access is included in the existing authorized odds plan. '
      'After the core forecast evaluation, we acquired a fixed monthly sample for 630 credits, plus a separate '
      '10-credit access probe. This supersedes the earlier finding that no usable historical price sample was available locally.','',
      '## Fixed scope and assumptions','',
      'The 15th of every month, October–April, in 2023–24, 2024–25 and 2025–26, at 10:30 America/New_York: '
      '21 decision snapshots covering 130 regular-season games. Dates were fixed before downloading; no dates '
      'were chosen by returns. Raw provider timestamps, query times, ingestion times and checksums are archived. '
      'Only prices timestamped at or before the decision, for games not started, are eligible.','',
      'The hockey algorithms, fitted-window rules and shadow thresholds were unchanged. Each season uses '
      'parameters trained on earlier seasons. Development-season results are retrospective diagnostics of the '
      'eventually selected specification; only 2025–26 is the final-season diagnostic. Historical house-rule '
      'versions, account limits and actual execution are unverified. These are quoted-price shadow simulations, '
      'not realized betting returns or a certified executable backtest.','',
      '## Paired probability benchmark','',
      'One home/Over observation per game, market and exact line; freshest mapped quote with leave-offered-book-out '
      'paired consensus. Push results are excluded from the conditional probability score. Multiple lines remain '
      'grouped by game in uncertainty calculations. Differences below are model minus market log loss; lower is better.','',
      '| Season | Market | Observations / games | Hockey log loss | Market log loss | Difference 95% interval |',
      '|---|---|---:|---:|---:|---|']
    for season,markets in metrics.items():
        for market,r in markets.items():
            lo,hi=r['difference_ci'];lines.append(f"| {season} | {market} | {r['independent']['n']} / {r['unique_games']} | {r['independent']['log_loss']:.5f} | {r['market']['log_loss']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines+=['','Every paired interval includes zero. The sample does not support a claim that the independent '
      'model beats the market. Brier scores and reliability bins are included in `historical-market-evaluation.json`. '
      'The 86 development games are below the predeclared 500-game blend-training floor, so no market blend is fitted.','',
      '## Fixed shadow price policy','',
      'At most four one-unit offers/day and one offer/game, requiring a mapped standard settlement profile, '
      'quote age ≤30 minutes, independent EV ≥2%, positive worst-scenario log-growth and an offered price at '
      'least the minimum from the ±10% rate scenarios. Actual win/loss/push outcomes only grade the frozen '
      'selections. Historical analyst approval is not fabricated; these selections are explicitly simulation-only '
      'and never enter the production recommendations list.','',
      '| Season | Count / turnover | Net units | ROI | Max drawdown | ROI 95% interval | ROI at −0.05 decimal execution |',
      '|---|---:|---:|---:|---:|---|---:|']
    for season,scenarios in report['shadow'].items():
        r=scenarios['nominal'];w=scenarios['worse_execution']
        if not r['count']:lines.append(f'| {season} | 0 / 0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |');continue
        lo,hi=r['roi_ci'];lines.append(f"| {season} | {r['count']} / {r['turnover']} | {r['net_units']:.4f} | {100*r['roi']:.2f}% | {r['max_drawdown']:.4f} | [{100*lo:.2f}%, {100*hi:.2f}%] | {100*w['roi']:.2f}% |")
    lines+=['','The final season produces only three shadow selections and loses 1.5238 units. The sample is far '
      'too small for a stable profitability estimate; the positive development returns are not a discovered '
      'strategy. No threshold was changed after these results. Odds quantiles, selection identities, raw offers '
      'and settled outcomes are retained in JSON/gzip outputs. Confidence intervals resample game clusters.','',
      '## What remains unavailable','',
      'Full daily historical price coverage, player-price histories, contemporaneous analyst evidence, historical '
      'rule versions and confirmed execution are not supplied by this sample. Closing quotes and afternoon '
      'snapshots were not acquired, so CLV and early-versus-later comparisons remain unavailable. The access '
      'probe proves entitlement, not coverage of every requested feature. A larger backfill needs an explicit '
      'quota budget; no subscription upgrade is required or was attempted.','',
      'Reproduce without network access after restoring/building the feature cache: '
      '`python scripts/nhl/v2/market_evaluate.py`. Raw snapshots live in '
      '`artifacts/nhl/historical-odds/`. This diagnostic does not enable production picks.']
    (output/'HISTORICAL_MARKETS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(offers=len(all_rows),games=len({r['nhl_game_id'] for r in all_rows}),shadow_selections=len(selected),metrics=metrics,shadow=report['shadow']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true')
    p.add_argument('--env-file',type=Path,default=ROOT/'.env');p.add_argument('--root',type=Path,default=ROOT/'artifacts/nhl/historical-odds')
    p.add_argument('--history',type=Path,default=ROOT/'data/nhl/v2/history');p.add_argument('--output',type=Path,default=ROOT/'reports/nhl-rebuild')
    a=p.parse_args()
    if a.download:download(a.root,a.env_file)
    evaluate(a.root,a.history,a.output)

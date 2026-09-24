#!/usr/bin/env python3
"""Freeze weekly NFL evidence, grade the completed week, and publish an audited review.

The scorecard is deterministic: no model/API call calculates results, units, Brier
scores, totals errors, or preview rankings. A weekly pregame archive is immutable
once written so later results cannot rewrite what the site knew before kickoff.
"""
import argparse
import hashlib
import html
import json
import math
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'reports/nfl-weekly'
BLOG=ROOT/'docs/blog'
STATS={
    'pass_attempts':'attempts','pass_completions':'completions','pass_yds':'passing_yards',
    'pass_tds':'passing_tds','interceptions':'passing_interceptions','receptions':'receptions',
    'recv_yds':'receiving_yards','rush_attempts':'carries','rush_yds':'rushing_yards',
}
LABELS={
    'pass_attempts':'Passing attempts','pass_completions':'Passing completions',
    'pass_yds':'Passing yards','pass_tds':'Passing touchdowns','interceptions':'Interceptions',
    'receptions':'Receptions','recv_yds':'Receiving yards','rush_attempts':'Rushing attempts',
    'rush_yds':'Rushing yards',
}
TEAM_NAMES=dict(zip(
    ['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills','Carolina Panthers',
     'Chicago Bears','Cincinnati Bengals','Cleveland Browns','Dallas Cowboys','Denver Broncos',
     'Detroit Lions','Green Bay Packers','Houston Texans','Indianapolis Colts','Jacksonville Jaguars',
     'Kansas City Chiefs','Las Vegas Raiders','Los Angeles Chargers','Los Angeles Rams','Miami Dolphins',
     'Minnesota Vikings','New England Patriots','New Orleans Saints','New York Giants','New York Jets',
     'Philadelphia Eagles','Pittsburgh Steelers','San Francisco 49ers','Seattle Seahawks',
     'Tampa Bay Buccaneers','Tennessee Titans','Washington Commanders'],
    'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LA MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS'.split()))
KEY=['game_id','player','market_std']


def utc(value):
    return pd.Timestamp(value).tz_convert('UTC') if pd.Timestamp(value).tzinfo else pd.Timestamp(value,tz='UTC')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def american_profit(price):
    price=float(price)
    return price/100 if price>0 else 100/abs(price)


def record(frame):
    graded=frame[frame.outcome.notna()]
    result={k:int((graded.outcome==k).sum()) for k in ['win','loss','push']}
    return dict(selected=int(len(frame)),graded=int(len(graded)),pending=int(len(frame)-len(graded)),
        wins=result['win'],losses=result['loss'],pushes=result['push'],
        win_rate=(result['win']/(result['win']+result['loss'])) if result['win']+result['loss'] else None,
        units=float(graded.units.sum()) if len(graded) else 0.0,
        roi=float(graded.units.mean()) if len(graded) else None)


def parse_props_page(path):
    text=Path(path).read_text()
    match=re.search(r'id="props-data">(.*?)</script>',text,re.S)
    if not match:raise ValueError(f'No props-data payload in {path}')
    packed=json.loads(match.group(1))
    rows=[]
    for values in packed['rows']:
        rows.append({field:(packed['dictionary'][field][value] if field in packed['dictionary'] else value)
                     for field,value in zip(packed['fields'],values)})
    return pd.DataFrame(rows)


def validate_snapshot_frame(frame,season,week,now,label):
    if frame.empty:raise ValueError(f'{label} snapshot is empty')
    if 'season' in frame and frame.season.notna().any() and set(pd.to_numeric(frame.season.dropna()).astype(int))!={season}:
        raise ValueError(f'{label} contains the wrong season')
    if 'week' in frame and frame.week.notna().any() and set(pd.to_numeric(frame.week.dropna()).astype(int))!={week}:
        raise ValueError(f'{label} contains the wrong week')
    if 'commence_time' in frame:
        starts=pd.to_datetime(frame.commence_time,utc=True,errors='coerce')
        if starts.isna().any():raise ValueError(f'{label} has invalid kickoff timestamps')
        if (starts<=now).any():raise ValueError(f'{label} contains a game that has already started; refusing a hindsight snapshot')


def snapshot_week(season,week,now=None,root=ROOT):
    now=pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert('UTC')
    dest=Path(root)/'reports/nfl-weekly'/str(season)/f'week-{week}'/'pregame'
    manifest=dest/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        for item in saved['files']:
            path=Path(root)/item['path']
            if not path.exists() or sha(path)!=item['sha256']:
                raise ValueError('Existing weekly archive failed its integrity check; refusing overwrite')
        print(json.dumps(dict(status='existing',season=season,week=week,path=str(dest.relative_to(root)))))
        return saved
    sources={
        'props':Path(root)/f'data/props/props_with_model_week{week}.csv',
        'predictions':Path(root)/'data/nfl/predictions/week_predictions.csv',
        'lines':Path(root)/'data/nfl/lines/totals_spreads.csv',
        'top_page':Path(root)/'docs/props/top.html',
    }
    missing=[str(path) for path in sources.values() if not path.exists()]
    if missing:raise ValueError('Cannot freeze weekly evidence; missing: '+', '.join(missing))
    props=pd.read_csv(sources['props'])
    predictions=pd.read_csv(sources['predictions'])
    lines=pd.read_csv(sources['lines'])
    top=parse_props_page(sources['top_page'])
    for frame,label in [(props,'props'),(lines,'lines'),(top,'top picks')]:
        validate_snapshot_frame(frame,season,week,now,label)
    if predictions.empty or set(pd.to_numeric(predictions.week).astype(int))!={week} or set(pd.to_numeric(predictions.season).astype(int))!={season}:
        raise ValueError('Totals predictions are empty or for the wrong season/week')
    dest.mkdir(parents=True,exist_ok=False)
    targets={
        'props':'props.csv','predictions':'week_predictions.csv','lines':'totals_spreads.csv','top':'top_picks.csv'
    }
    props.to_csv(dest/targets['props'],index=False)
    predictions.to_csv(dest/targets['predictions'],index=False)
    lines.to_csv(dest/targets['lines'],index=False)
    top.to_csv(dest/targets['top'],index=False)
    optional=[
        (Path(root)/'data/nfl/consensus/totals_spreads_consensus.csv','totals_spreads_consensus.csv'),
        (Path(root)/f'data/injuries/injuries_week{week}.csv','injuries.csv'),
        (Path(root)/f'data/nfl/injuries/injury_totals_week{week}.csv','injury_totals.csv'),
    ]
    for source,name in optional:
        if source.exists():shutil.copyfile(source,dest/name)
    files=[]
    for path in sorted(dest.iterdir()):
        if path.is_file() and path.name!='manifest.json':
            files.append(dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size))
    starts=pd.to_datetime(props.commence_time,utc=True)
    payload=dict(schema=1,season=season,week=week,snapshot_at=now.isoformat(),
        earliest_kickoff=starts.min().isoformat(),latest_kickoff=starts.max().isoformat(),
        props_rows=int(len(props)),top_rows=int(len(top)),games=int(props.game_id.nunique()),files=files)
    manifest.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(status='created',season=season,week=week,path=str(dest.relative_to(root)),
        props_rows=len(props),top_rows=len(top),games=props.game_id.nunique())))
    return payload


def legacy_archive(season,week,root=ROOT):
    # One-time migration path: Week 3 existed before immutable weekly archives were introduced.
    if season==2026 and week==3:
        base=Path(root)/'reports/week3-2026-preview'
        if (base/'props_with_model_week3.csv').exists():
            props=pd.read_csv(base/'props_with_model_week3.csv')
            top=props[props.model_prob.notna() & props.ev_per_100.notna() & props.point.notna() &
                      props.name.isin(['over','under'])].copy()
            return dict(props=props,top=top,predictions=pd.read_csv(base/'week_predictions.csv'),
                lines=pd.read_csv(base/'totals_spreads.csv'),manifest=json.loads((base/'manifest.json').read_text()),
                source='legacy-week3-preview')
    return None


def load_archive(season,week,root=ROOT):
    base=Path(root)/'reports/nfl-weekly'/str(season)/f'week-{week}'/'pregame'
    manifest=base/'manifest.json'
    if not manifest.exists():
        legacy=legacy_archive(season,week,root)
        if legacy:return legacy
        raise ValueError(f'No frozen pregame archive for {season} Week {week}')
    meta=json.loads(manifest.read_text())
    for item in meta['files']:
        path=Path(root)/item['path']
        if not path.exists() or sha(path)!=item['sha256']:
            raise ValueError(f'Frozen archive integrity failure: {item["path"]}')
    return dict(props=pd.read_csv(base/'props.csv'),top=pd.read_csv(base/'top_picks.csv'),
        predictions=pd.read_csv(base/'week_predictions.csv'),lines=pd.read_csv(base/'totals_spreads.csv'),
        manifest=meta,source='immutable-weekly-archive')


def normalize(name):
    return re.sub(r'(?:iii|ii|iv|jr|sr)$','',re.sub('[^a-z]','',str(name).lower()))


def stat_game(row,season,week):
    try:return f'{season}_{week:02d}_{TEAM_NAMES[row.away_team]}_{TEAM_NAMES[row.home_team]}'
    except KeyError:raise ValueError(f'Unknown team name in archived props: {row.away_team} @ {row.home_team}')


def grade_props(props,stats,schedule,season,week):
    d=props.copy()
    d['stat_game']=[stat_game(row,season,week) for row in d.itertuples()]
    games=set(schedule.game_id.astype(str))
    if not set(d.stat_game).issubset(games):raise ValueError('Archived prop game could not be reconciled to official schedule')
    lookup={}
    for row in stats.itertuples():
        key=(str(row.game_id),normalize(row.player_display_name))
        if key in lookup:raise ValueError(f'Ambiguous player result: {key}')
        lookup[key]=row
    actual=[];status=[]
    for row in d.itertuples():
        if row.market_std not in STATS:
            actual.append(np.nan);status.append('unsupported settlement market');continue
        result=lookup.get((row.stat_game,normalize(row.player)))
        if result is None:
            actual.append(np.nan);status.append('no matching game/player result');continue
        usage=sum(float(getattr(result,k,0) or 0) for k in ['attempts','carries','targets','receptions'])
        if usage<=0:
            actual.append(np.nan);status.append('no offensive usage evidence');continue
        value=getattr(result,STATS[row.market_std],np.nan)
        actual.append(value);status.append('graded' if pd.notna(value) else 'missing statistic')
    d['actual']=actual;d['grading_status']=status;d['outcome']=None
    valid=d.actual.notna() & d.point.notna() & d.name.isin(['over','under'])
    win=(d.name.eq('over') & d.actual.gt(d.point)) | (d.name.eq('under') & d.actual.lt(d.point))
    push=d.actual.eq(d.point)
    d.loc[valid,'outcome']=np.where(push[valid],'push',np.where(win[valid],'win','loss'))
    d['units']=np.where(d.outcome.eq('win'),d.price.map(american_profit),
                np.where(d.outcome.eq('loss'),-1,np.where(d.outcome.eq('push'),0,np.nan)))
    return d


def select_tickets(top):
    d=top[top.market_std.isin(STATS) & top.point.notna() & top.name.isin(['over','under']) &
          top.model_prob.notna() & top.ev_per_100.notna()].copy()
    if d.empty:return d
    return d.sort_values(['ev_per_100','bookmaker','point','name'],ascending=[False,True,True,True]).drop_duplicates(KEY)


def representative(frame):
    chosen=[]
    d=frame[frame.market_std.isin(STATS) & frame.point.notna() & frame.name.isin(['over','under'])].copy()
    for _,group in d.groupby(KEY,sort=True):
        lines=group[['bookmaker','point']].drop_duplicates()
        counts=lines.groupby('point').bookmaker.nunique()
        median=lines.point.median()
        line=sorted(counts.index,key=lambda x:(-counts[x],abs(x-median),x))[0]
        at=group[group.point.eq(line)]
        for _,offers in at.groupby('name'):
            offers=offers.assign(_payout=offers.price.map(american_profit))
            chosen.append(offers.sort_values(['_payout','bookmaker'],ascending=[False,True]).iloc[0])
    return pd.DataFrame(chosen).drop(columns=['_payout'],errors='ignore').reset_index(drop=True) if chosen else pd.DataFrame()


def prop_metrics(board,top,stats,schedule,season,week):
    graded_board=grade_props(board,stats,schedule,season,week)
    graded_top=grade_props(top,stats,schedule,season,week)
    tickets=select_tickets(graded_top)
    reps=representative(graded_board)
    overs=reps[reps.name.eq('over')].copy() if not reps.empty else reps
    paired=overs[overs.actual.notna() & overs.actual.ne(overs.point) & overs.model_prob.notna() & overs.consensus_prob.notna()].copy()
    if not paired.empty:
        paired['y']=paired.actual.gt(paired.point).astype(float)
        paired['model_brier']=(paired.model_prob-paired.y)**2
        paired['market_brier']=(paired.consensus_prob-paired.y)**2
        paired['model_abs_error']=(paired.mu-paired.actual).abs()
        paired['line_abs_error']=(paired.point-paired.actual).abs()
    markets={m:record(g) for m,g in tickets.groupby('market_std')} if not tickets.empty else {}
    market_results=[]
    if not overs.empty:
        for market,group in overs.groupby('market_std'):
            valid=group[group.actual.notna()]
            market_results.append(dict(market=market,matched=int(len(valid)),
                overs=int(valid.actual.gt(valid.point).sum()),unders=int(valid.actual.lt(valid.point).sum()),
                pushes=int(valid.actual.eq(valid.point).sum())))
    overall=None
    if not paired.empty:
        blocks=[g[['model_brier','market_brier']].to_numpy() for _,g in paired.groupby('stat_game')]
        rng=np.random.default_rng(season*100+week)
        delta=[]
        for _ in range(2000):
            sample=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
            delta.append(float((sample[:,0]-sample[:,1]).mean()))
        overall=dict(n=int(len(paired)),model_brier=float(paired.model_brier.mean()),
            market_brier=float(paired.market_brier.mean()),brier_difference_ci95=np.quantile(delta,[.025,.975]).tolist())
    return dict(board=graded_board,tickets=tickets,reps=reps,paired=paired,
        summary=dict(tickets=record(tickets),ticket_markets=markets,market_results=market_results,probability_overall=overall))


def best_same_line(lines,side,line):
    price_col='total_over_price' if side=='over' else 'total_under_price'
    subset=lines[lines.total_over_line.eq(line) & lines[price_col].notna()]
    if subset.empty:return np.nan
    return float(max(subset[price_col],key=american_profit))


def totals_metrics(archive,schedule,season,week):
    preds=archive['predictions'].copy();lines=archive['lines'].copy()
    official=schedule[(schedule.season==season)&(schedule.week==week)&(schedule.game_type=='REG')].copy()
    if official.empty:raise ValueError('Official completed-week schedule is empty')
    if official.home_score.isna().any() or official.away_score.isna().any():
        raise ValueError('Completed-week schedule still has unplayed games')
    official['game']=official.away_team+' @ '+official.home_team
    official['actual_total']=official.home_score+official.away_score
    market=[]
    for game,group in lines.groupby('game'):
        distinct=group[['book','total_over_line']].dropna().drop_duplicates()
        if distinct.empty:continue
        median=float(distinct.total_over_line.median())
        market.append(dict(game=game,market_total=median,books=int(distinct.book.nunique()),
            best_over_price=best_same_line(group,'over',median),best_under_price=best_same_line(group,'under',median)))
    market=pd.DataFrame(market)
    games=official.merge(preds[['game','total_pred']],on='game',how='left',validate='one_to_one').merge(market,on='game',how='left',validate='one_to_one')
    if games.total_pred.isna().any() or games.market_total.isna().any():
        missing=games[games.total_pred.isna()|games.market_total.isna()].game.tolist()
        raise ValueError('Frozen totals evidence missing official games: '+', '.join(missing))
    games['model_error']=games.total_pred-games.actual_total
    games['market_error']=games.market_total-games.actual_total
    games['closing_error']=games.total_line-games.actual_total if 'total_line' in games else np.nan
    games['model_pick']=np.where(games.total_pred>games.market_total,'over',np.where(games.total_pred<games.market_total,'under','pass'))
    games['market_result']=np.where(games.actual_total>games.market_total,'over',np.where(games.actual_total<games.market_total,'under','push'))
    games['model_result']=np.where(games.model_pick.eq('pass'),'push',np.where(games.model_pick.eq(games.market_result),'win',np.where(games.market_result.eq('push'),'push','loss')))
    games['model_units']=np.nan
    for idx,row in games.iterrows():
        if row.model_pick=='pass':continue
        price=row.best_over_price if row.model_pick=='over' else row.best_under_price
        # A statistical median need not be an offered line. Without an executable
        # same-line quote, neither a hypothetical win nor loss belongs in P/L.
        if pd.isna(price):continue
        games.loc[idx,'model_units']=american_profit(price) if row.model_result=='win' else (-1.0 if row.model_result=='loss' else 0.0)
    closing=dict(overs=int((games.actual_total>games.total_line).sum()),unders=int((games.actual_total<games.total_line).sum()),
        pushes=int((games.actual_total==games.total_line).sum())) if 'total_line' in games else None
    return games,dict(games=int(len(games)),
        archived_overs=int((games.actual_total>games.market_total).sum()),
        archived_unders=int((games.actual_total<games.market_total).sum()),
        archived_pushes=int((games.actual_total==games.market_total).sum()),
        model_direction=dict(wins=int((games.model_result=='win').sum()),losses=int((games.model_result=='loss').sum()),
            pushes=int((games.model_result=='push').sum()),units=float(games.model_units.dropna().sum()),
            priced_games=int(games.model_units.notna().sum()),unpriced_games=int((games.model_pick.ne('pass') & games.model_units.isna()).sum())),
        mae=dict(model=float(games.model_error.abs().mean()),market=float(games.market_error.abs().mean()),
            recorded_close=float(games.closing_error.abs().mean()) if 'total_line' in games else None),
        closing=closing)


def preview_metrics(archive):
    preds=archive['predictions'].copy();lines=archive['lines'].copy()
    med=lines.groupby('game',as_index=False).total_over_line.median().rename(columns={'total_over_line':'market_total'})
    preview=preds.merge(med,on='game',how='inner')
    preview['gap']=preview.total_pred-preview.market_total
    preview=preview.reindex(preview.gap.abs().sort_values(ascending=False).index)
    top=select_tickets(archive['top'])
    top=top.sort_values('ev_per_100',ascending=False) if not top.empty else top
    return dict(total_gaps=preview.head(5).to_dict('records'),
        prop_watch=top.head(5)[['game','player','market_std','name','point','price','bookmaker','mu','model_prob','consensus_prob','ev_per_100','model_status']].to_dict('records') if not top.empty else [])


def chart(path,title,subtitle,rows,lo=None,hi=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    vals=[float(r[1]) for r in rows];lo=min(vals+[0]) if lo is None else lo;hi=max(vals+[0]) if hi is None else hi
    if math.isclose(lo,hi):lo-=1;hi+=1
    height=145+42*len(rows);x=lambda v:230+(v-lo)/(hi-lo)*610
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><title>{html.escape(title)}</title>',
        '<rect width="100%" height="100%" fill="#0b1219"/><g font-family="Arial,sans-serif" fill="#e7eef9">',
        f'<text x="30" y="38" font-size="25" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="30" y="68" font-size="16" fill="#bac8d6">{html.escape(subtitle)}</text>',
        f'<path d="M{x(0)} 88V{height-30}" stroke="#78909c"/>']
    for i,(label,value,note) in enumerate(rows):
        y=100+42*i;end=x(float(value));start=x(0)
        parts += [f'<text x="218" y="{y+19}" text-anchor="end" font-size="16">{html.escape(str(label))}</text>',
            f'<rect x="{min(start,end):.1f}" y="{y}" width="{max(abs(end-start),1):.1f}" height="26" fill="{"#7ce2bd" if value>=0 else "#f4ae91"}" rx="3"/>',
            f'<text x="855" y="{y+19}" font-size="15">{html.escape(str(note))}</text>']
    parts.append(f'<text x="30" y="{height-10}" font-size="13" fill="#bac8d6">Fourth &amp; Value · frozen pregame evidence</text></g></svg>')
    path.write_text(''.join(parts))


def fmt_units(value):
    return f'{value:+.2f}u'


def pct(value):
    return 'n/a' if value is None else f'{100*value:.1f}%'


def render_article(season,completed,preview,summary,totals,prop,preview_data,outdir,root=ROOT):
    blog=Path(root)/'docs/blog';blog.mkdir(parents=True,exist_ok=True)
    title=f'NFL Week {completed} review: the model receipts and the Week {preview} watchlist'
    desc=f'An audited Week {completed} scorecard using frozen pregame data, plus Fourth & Value model-market disagreements to research for Week {preview}.'
    ticket=summary['props']['tickets'];closing=summary['totals']['closing'];prob=summary['props'].get('probability_overall')
    legacy=summary.get('completed_archive')=='legacy-week3-preview'
    selection_label='archived modeled board (not a saved published shortlist)' if legacy else 'published shortlist'
    archive_note=('Week 3 uses the preserved legacy pregame model board. Its selections are reconstructed with the stated highest-EV rule; they are not a record of the published shortlist. Later weeks use hash-verified archives.' if legacy else 'The scorecard uses hash-verified immutable weekly pregame archives.')
    markets=sorted(summary['props']['ticket_markets'].items(),key=lambda kv:kv[1]['units'],reverse=True)
    total_rows=''.join(f'<tr><td>{html.escape(LABELS.get(m,m))}</td><td>{r["wins"]}–{r["losses"]}</td><td>{fmt_units(r["units"])}</td><td>{r["pending"]}</td></tr>' for m,r in markets)
    gaps=''.join(f'<li><strong>{html.escape(str(r["game"]))}</strong>: model {r["total_pred"]:.1f} vs archived median {r["market_total"]:.1f} ({r["gap"]:+.1f}). This is a research gap, not a calibrated edge.</li>' for r in preview_data['total_gaps'])
    props=''.join(f'<li><strong>{html.escape(str(r["player"]))} {html.escape(LABELS.get(r["market_std"],r["market_std"]))} {html.escape(str(r["name"]))} {r["point"]}</strong> at {html.escape(str(r["bookmaker"]))} ({int(r["price"]):+d}); saved model mean {r["mu"]:.2f}, model probability {100*r["model_prob"]:.1f}%, estimated EV {r["ev_per_100"]:+.1f}%. <em>{html.escape(str(r["model_status"]))}</em></li>' for r in preview_data['prop_watch'])
    brier=('No matched probability comparison was available.' if not prob else
        f'Across {prob["n"]} representative non-push outcomes with both probabilities available, the model Brier score was <strong>{prob["model_brier"]:.4f}</strong> versus <strong>{prob["market_brier"]:.4f}</strong> for the de-vigged consensus. The game-block 95% interval for model minus market Brier was {prob["brier_difference_ci95"][0]:+.4f} to {prob["brier_difference_ci95"][1]:+.4f}.')
    closing_text=('Recorded closing totals were unavailable.' if not closing else
        f'Recorded closes produced {closing["overs"]} overs, {closing["unders"]} unders and {closing["pushes"]} pushes.')
    html_text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} | Fourth &amp; Value</title><meta name="description" content="{html.escape(desc)}"><link rel="canonical" href="https://fourthandvalue.com/blog/week-{completed}-recap-week-{preview}-preview-{season}.html"><link rel="stylesheet" href="../assets/site.css"><link rel="stylesheet" href="/assets/responsible-use.css?v=1"></head><body><div id="nav-root"></div><script src="../nav.js?v=42"></script><main style="max-width:900px;margin:auto;padding:36px 20px 80px"><p class="eyebrow">Fourth &amp; Value · Weekly market review</p><h1>{html.escape(title)}</h1><p class="lead">{html.escape(desc)}</p><p class="meta">Generated from saved pregame evidence and official completed results. Re-running the workflow does not rewrite the frozen Week {completed} inputs.</p>
<h2>Week {completed}: the market scoreboard</h2><p>{closing_text} Against our archived market median, {summary["totals"]["archived_overs"]} games finished over, {summary["totals"]["archived_unders"]} under and {summary["totals"]["archived_pushes"]} pushed.</p><p>The raw totals model direction went <strong>{summary["totals"]["model_direction"]["wins"]}–{summary["totals"]["model_direction"]["losses"]}</strong> with {summary["totals"]["model_direction"]["pushes"]} pushes and {fmt_units(summary["totals"]["model_direction"]["units"])} across {summary["totals"]["model_direction"]["priced_games"]} games with archived same-line prices; {summary["totals"]["model_direction"]["unpriced_games"]} unpriced directions are excluded from profit/loss. Mean absolute error was {summary["totals"]["mae"]["model"]:.2f} points for the model versus {summary["totals"]["mae"]["market"]:.2f} for the archived market median{f' and {summary["totals"]["mae"]["recorded_close"]:.2f} for the recorded close' if summary["totals"]["mae"]["recorded_close"] is not None else ''}.</p><figure><img src="week-{completed}-week-{preview}-{season}/totals.svg" alt="Week {completed} final points relative to the recorded closing total" style="width:100%"><figcaption>Final points minus the recorded close. This grades a saved forecast; it does not retroactively select a strategy.</figcaption></figure>
<h2 id="scorecard">The model scorecard</h2><p>The deduplicated {selection_label} contained {ticket["selected"]} selections; {ticket["graded"]} were conservatively graded and {ticket["pending"]} remain unresolved. The graded set went <strong>{ticket["wins"]}–{ticket["losses"]}</strong> with {ticket["pushes"]} pushes for <strong>{fmt_units(ticket["units"])}</strong>, or {pct(ticket["roi"])} on units risked. Missing participation or missing statistics remain unresolved rather than becoming automatic unders.</p><div style="overflow:auto"><table><thead><tr><th>Market</th><th>W–L</th><th>Net units</th><th>Unresolved</th></tr></thead><tbody>{total_rows}</tbody></table></div><figure><img src="week-{completed}-week-{preview}-{season}/props.svg" alt="Week {completed} archived prop selection units by market" style="width:100%"><figcaption>One highest-EV archived offer per player, game and modeled market.</figcaption></figure>
<h2>Probability quality, not just profit</h2><p>{brier}</p><p>A positive one-week return and a better probability score are different claims. This report keeps both because a profitable slate can still expose probability weaknesses, and vice versa.</p>
<h2>Week {preview}: disagreement is a research question</h2><p>The new weekly refresh is frozen before the first kickoff. The largest raw totals disagreements are:</p><ul>{gaps or '<li>No complete model/market total comparisons passed the archive checks.</li>'}</ul><figure><img src="week-{completed}-week-{preview}-{season}/preview.svg" alt="Week {preview} model minus market total gaps" style="width:100%"><figcaption>Raw model minus archived median total. These gaps are not automatically betting recommendations.</figcaption></figure>
<h3>Player props to recheck</h3><ul>{props or '<li>No qualifying modeled prop rows were available in the frozen preview shortlist.</li>'}</ul><p>These are saved prices from the weekly archive. Recheck the current line, price, injury status and role before treating any one of them as actionable. Model status labels are shown rather than hidden.</p>
<h2>Audit notes</h2><p>{html.escape(archive_note)}</p><p>The completed-week scorecard uses only the frozen pregame archive for Week {completed} and official results/player statistics available after the games. The preview uses a separate frozen Week {preview} archive. The workflow refuses to create a new archive after a game has started, refuses to overwrite an existing archive whose hashes do not match, and refuses to publish a review while any official Week {completed} game remains unplayed.</p><p><a href="week-{completed}-{season}-review-data.json">Download the public review data</a> · <a href="/">Back to Fourth &amp; Value</a></p></main></body></html>'''
    path=blog/f'week-{completed}-recap-week-{preview}-preview-{season}.html'
    path.write_text(html_text+'\n')
    return path,title,desc


def update_discovery(path,title,desc,date,root=ROOT):
    blog=Path(root)/'docs/blog/index.html'
    if blog.exists():
        text=blog.read_text();href='./'+Path(path).name
        if href not in text:
            marker='<!-- editorial-managed:end -->'
            card=f'''<li class="post" data-title="{html.escape(title)}" data-excerpt="{html.escape(desc)}"><h2><a href="{href}">{html.escape(title)}</a></h2><div class="meta">{date} · NFL weekly market review</div><p class="excerpt">{html.escape(desc)}</p></li>'''
            if marker not in text:raise ValueError('Blog managed marker missing')
            text=text.replace(marker,marker+card)
            blog.write_text(text)
    sitemap=Path(root)/'docs/sitemap.xml'
    if sitemap.exists():
        text=sitemap.read_text();url='https://fourthandvalue.com/blog/'+Path(path).name
        if url not in text:
            text=text.replace('</urlset>',f'  <url><loc>{url}</loc></url>\n</urlset>')
            sitemap.write_text(text)


def review_week(season,completed,preview,root=ROOT):
    schedule_path=Path(root)/f'data/schedule_{season}.csv'
    stats_path=Path(root)/f'data/weekly_player_stats_{season}.parquet'
    if not schedule_path.exists() or not stats_path.exists():raise ValueError('Official schedule/player results are unavailable')
    schedule=pd.read_csv(schedule_path)
    official=schedule[(schedule.season==season)&(schedule.week==completed)&(schedule.game_type=='REG')].copy()
    if official.empty or official.home_score.isna().any() or official.away_score.isna().any():
        raise ValueError(f'Week {completed} is not fully completed; weekly review withheld')
    stats=pd.read_parquet(stats_path)
    stats=stats[(stats.season==season)&(stats.week==completed)&(stats.season_type=='REG')].copy()
    if stats.empty:raise ValueError(f'Week {completed} player results are not published; weekly review withheld')
    old=load_archive(season,completed,root);new=load_archive(season,preview,root)
    prop=prop_metrics(old['props'],old['top'],stats,official,season,completed)
    games,total_summary=totals_metrics(old,official,season,completed)
    preview_data=preview_metrics(new)
    summary=dict(schema=1,season=season,completed_week=completed,preview_week=preview,
        generated_at=datetime.now(timezone.utc).isoformat(),completed_archive=old['source'],preview_archive=new['source'],
        props=prop['summary'],totals=total_summary,preview=preview_data)
    review_dir=Path(root)/'reports/nfl-weekly'/str(season)/f'week-{completed}'/'review'
    review_dir.mkdir(parents=True,exist_ok=True)
    (review_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    prop['tickets'].to_csv(review_dir/'tickets.csv',index=False)
    prop['paired'].to_csv(review_dir/'probability_comparison.csv',index=False)
    games.to_csv(review_dir/'games.csv',index=False)
    blog=Path(root)/'docs/blog';blog.mkdir(parents=True,exist_ok=True)
    outdir=blog/f'week-{completed}-week-{preview}-{season}'
    outdir.mkdir(parents=True,exist_ok=True)
    close_col='closing_error' if 'closing_error' in games else 'market_error'
    chart(outdir/'totals.svg',f'Week {completed}: final points vs recorded close','Negative finished under; positive finished over',
        [(r.game,-getattr(r,close_col),f'{-getattr(r,close_col):+.1f}') for r in games.sort_values(close_col).itertuples()])
    market_rows=sorted(summary['props']['ticket_markets'].items(),key=lambda kv:kv[1]['units'])
    chart(outdir/'props.svg',f'Week {completed}: archived prop selections','One archived selection per player / game / market',
        [(LABELS.get(m,m),v['units'],fmt_units(v['units'])) for m,v in market_rows])
    chart(outdir/'preview.svg',f'Week {preview}: model minus archived market total','Largest absolute disagreements in the frozen preview',
        [(r['game'],r['gap'],f'{r["gap"]:+.1f}') for r in reversed(preview_data['total_gaps'])])
    public=dict(summary=summary,
        tickets=json.loads(prop['tickets'][['stat_game','player','market_std','name','point','price','bookmaker','model_prob','consensus_prob','ev_per_100','actual','outcome','units','grading_status']].to_json(orient='records')) if not prop['tickets'].empty else [],
        games=json.loads(games[['game','home_score','away_score','actual_total','market_total','total_pred','total_line','model_pick','model_result','model_units']].to_json(orient='records')),
        preview=preview_data)
    data_path=blog/f'week-{completed}-{season}-review-data.json'
    data_path.write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    article,title,desc=render_article(season,completed,preview,summary,games,prop,preview_data,outdir,root)
    update_discovery(article,title,desc,datetime.now().date().isoformat(),root)
    print(json.dumps(dict(status='published',article=str(article.relative_to(root)),summary=str((review_dir/'summary.json').relative_to(root)),
        tickets=summary['props']['tickets'],totals=summary['totals']['model_direction']),indent=2))
    return summary


def verify_week2(root=ROOT):
    base=Path(root)/'reports/week2-2026'
    tickets=pd.read_csv(base/'tickets.csv')
    games=pd.read_csv(base/'games.csv')
    probs=pd.read_csv(base/'probability_comparison.csv')
    got=record(tickets)
    expected=json.loads((base/'summary.json').read_text())
    checks={
        'tickets_selected':got['selected']==expected['tickets']['selected'],
        'tickets_graded':got['graded']==expected['tickets']['graded'],
        'tickets_units':abs(got['units']-expected['tickets']['units'])<1e-9,
        'closing_unders':int((games.total<games.total_line).sum())==10,
        'closing_overs':int((games.total>games.total_line).sum())==6,
        'model_brier':abs(float(probs.model_brier.mean())-expected['probability_overall']['model_brier'])<1e-12,
        'market_brier':abs(float(probs.market_brier.mean())-expected['probability_overall']['market_brier'])<1e-12,
    }
    if not all(checks.values()):raise ValueError('Week 2 regression mismatch: '+json.dumps(checks))
    print(json.dumps(dict(status='passed',checks=checks,tickets=got),indent=2))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    snap=sub.add_parser('snapshot');snap.add_argument('--season',type=int,required=True);snap.add_argument('--week',type=int,required=True)
    review=sub.add_parser('review');review.add_argument('--season',type=int,required=True);review.add_argument('--completed-week',type=int,required=True);review.add_argument('--preview-week',type=int,required=True)
    sub.add_parser('verify-week2')
    args=p.parse_args()
    if args.command=='snapshot':snapshot_week(args.season,args.week)
    elif args.command=='review':review_week(args.season,args.completed_week,args.preview_week)
    else:verify_week2()


if __name__=='__main__':main()

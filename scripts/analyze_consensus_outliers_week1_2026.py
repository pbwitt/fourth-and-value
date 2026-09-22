#!/usr/bin/env python3
"""Compare consensus fades with the same side at the consensus line.

Run review_week1_2026.py first to regenerate the frozen, graded input.
This is a market-only diagnostic, not a production strategy or model refit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_week1_2026 import STATS, payout

KEY = ['game_id', 'player', 'market_std']
OUT = Path('reports/consensus-outliers-week1-2026')


def settlement(actual, point, side):
    if pd.isna(actual):
        return None
    if actual == point:
        return 'push'
    return 'win' if (actual < point if side == 'under' else actual > point) else 'loss'


def units(result, price):
    if result is None:
        return np.nan
    return (price/100 if price > 0 else 100/-price) if result == 'win' else -1.0 if result == 'loss' else 0.0


def summarize(frame):
    d = frame[frame.result.notna()]
    w, l, p = [int(d.result.eq(r).sum()) for r in ['win', 'loss', 'push']]
    return dict(selected=len(frame), graded=len(d), pending=len(frame)-len(d), wins=w, losses=l, pushes=p,
        win_rate=w/(w+l) if w+l else None, units=float(d.units.sum()), roi=float(d.units.mean()) if len(d) else None,
        consensus_wins=int(d.consensus_result.eq('win').sum()),
        consensus_losses=int(d.consensus_result.eq('loss').sum()),
        consensus_pushes=int(d.consensus_result.eq('push').sum()),
        consensus_units=float(d.consensus_units.sum()), consensus_roi=float(d.consensus_units.mean()) if len(d) else None,
        extra_wins=int((d.result.eq('win') & ~d.consensus_result.eq('win')).sum()),
        median_price=float(d.price.median()) if len(d) else None,
        median_gap=float(d.gap.abs().median()) if len(d) else None)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    offers = pd.read_csv('reports/week1-2026/offers.csv')
    d = offers[offers.market_std.isin(STATS) & offers.name.isin(['over','under']) & offers.point.notna()].copy()
    d['side'] = d['name']
    idx = KEY+['bookmaker','point']
    assert not d.duplicated(idx+['side']).any()
    pairs = d.pivot(index=idx,columns='side',values='price').dropna().reset_index()
    meta = d.groupby(idx,as_index=False).agg(actual=('actual','first'),stat_game=('stat_game','first'),
        last_update=('last_update','max'), grading_status=('grading_status','first'))
    pairs = pairs.merge(meta,on=idx,validate='one_to_one')
    # Main-line proxy: the offered O/U pair closest to equal de-vigged probabilities.
    # Alternate lines cannot cast additional votes or masquerade as main-line outliers.
    def implied(prices):
        return np.where(prices>0,100/(prices+100),-prices/(100-prices))
    op,up = implied(pairs.over),implied(pairs.under)
    pairs['fair_over'] = op/(op+up)
    pairs['balance'] = abs(pairs.fair_over-.5)
    pairs['hold'] = op+up-1
    pairs['payout_over'] = payout(pairs.over)
    pairs['payout_under'] = payout(pairs.under)
    central = pairs.sort_values(['balance','hold','point']).drop_duplicates(KEY+['bookmaker'])
    rows=[]
    for _,group in central.groupby(KEY):
        for _,target in group.iterrows():
            other = group[group.bookmaker.ne(target.bookmaker)]
            counts=other.point.value_counts()
            if counts.empty or counts.iloc[0]<4 or counts.iloc[0]<=len(other)/2:
                continue
            consensus=float(counts.index[0]);gap=float(target.point-consensus)
            if gap==0:
                continue
            peers=other[other.point.eq(consensus)]
            times=pd.to_datetime(pd.concat([peers.last_update,pd.Series([target.last_update])]),utc=True)
            if times.isna().any() or (times.max()-times.min()).total_seconds()>1800:
                continue
            side='under' if gap>0 else 'over'
            price=float(target[side]);base=peers.sort_values('payout_'+side,ascending=False).iloc[0]
            actual=target.actual;result=settlement(actual,target.point,side);baseline=settlement(actual,consensus,side)
            rows.append(dict(game_id=target.game_id,stat_game=target.stat_game,player=target.player,
                market_std=target.market_std,book=target.bookmaker,line=float(target.point),price=price,side=side,
                consensus_line=consensus,agreeing_books=len(peers),peer_books=len(other),
                total_books=len(group),consensus_book=base.bookmaker,consensus_price=float(base[side]),
                gap=gap,actual=actual,result=result,units=units(result,price),
                consensus_result=baseline,consensus_units=units(baseline,float(base[side])),
                fair_side=float(1-target.fair_over if side=='under' else target.fair_over),
                grading_status=target.grading_status,
                unanimous_peers=len(peers)==len(other),last_update=target.last_update))
    all_bets=pd.DataFrame(rows)
    assert not all_bets.empty
    # Define one ticket per outcome family without inspecting the result.
    selected=all_bets.assign(abs_gap=all_bets.gap.abs(),payout=payout(all_bets.price)).sort_values(
        ['abs_gap','payout','book'],ascending=[False,False,True]).drop_duplicates(KEY)
    single=all_bets[all_bets.unanimous_peers].copy()
    assert not single.duplicated(KEY).any()
    # The primary book's market probability captures pricing differences between lines.
    # No model-implied EV is used to select or evaluate these tickets.
    cluster=selected[selected.result.notna()].copy()
    rng=np.random.default_rng(20260918)
    groups=[g[['units','consensus_units']].to_numpy() for _,g in cluster.groupby('game_id')]
    boot=[]
    for _ in range(5000):
        a=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        boot.append([float(a[:,0].mean()),float((a[:,0]-a[:,1]).mean())])
    strict_groups=[g[['units','consensus_units']].to_numpy() for _,g in single[single.result.notna()].groupby('game_id')]
    strict_rng=np.random.default_rng(20260918)
    strict_boot=[]
    for _ in range(5000):
        a=np.concatenate([strict_groups[i] for i in strict_rng.integers(0,len(strict_groups),len(strict_groups))])
        strict_boot.append([float(a[:,0].mean()),float((a[:,0]-a[:,1]).mean())])
    summary=dict(source_commit='6d420cf',season=2026,week=1,quote_window='2026-09-09 approximately 22:36 UTC',
        selection='One central O/U pair per book, closest to 50/50 de-vigged; target excluded from consensus; at least four other books and strict majority at one exact line; quote skew <=30 minutes; choose largest gap then best payout per player/game/market.',
        central_pairs=len(central),player_markets=len(central[KEY].drop_duplicates()),
        all_outlier_book_offers=summarize(all_bets),one_per_player_market=summarize(selected),
        unanimous_peer_outliers=summarize(single),
        exact_four_vs_one=summarize(single[single.total_books.eq(5)]),
        unanimous_by_direction={m:summarize(g) for m,g in single.groupby('side')},
        unanimous_by_market={m:summarize(g) for m,g in single.groupby('market_std')},
        unanimous_roi_game_bootstrap_ci95=np.quantile(np.array(strict_boot)[:,0],[.025,.975]).tolist(),
        unanimous_roi_improvement_game_bootstrap_ci95=np.quantile(np.array(strict_boot)[:,1],[.025,.975]).tolist(),
        by_market={m:summarize(g) for m,g in selected.groupby('market_std')},
        by_direction={m:summarize(g) for m,g in selected.groupby('side')},
        by_book={m:summarize(g) for m,g in selected.groupby('book')},
        markets_with_at_least_one_outlier=int(len(selected)),
        roi_game_bootstrap_ci95=np.quantile(np.array(boot)[:,0],[.025,.975]).tolist(),
        roi_improvement_game_bootstrap_ci95=np.quantile(np.array(boot)[:,1],[.025,.975]).tolist())
    central.to_csv(OUT/'central_book_lines.csv',index=False)
    all_bets.to_csv(OUT/'all_outlier_offers.csv',index=False)
    selected.to_csv(OUT/'selected_outliers.csv',index=False)
    output={'summary':summary,'selected':json.loads(selected.to_json(orient='records')),
            'all_outliers':json.loads(all_bets.to_json(orient='records'))}
    (OUT/'audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

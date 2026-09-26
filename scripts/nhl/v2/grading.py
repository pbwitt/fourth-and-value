"""Local prospective grading and execution diagnostics; never writes sportsbook accounts."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2])); __package__='nhl.v2'
from .data import ROOT, load, stamp, write_json
from .pricing import decimal, key


def settle(row,game,player=None,participation=None):
    if game is None: return 'pending'
    if row.get('nhl_game_id')!=game['game_id']: raise ValueError('Game identity mismatch')
    if row.get('settlement_profile','').startswith('unverified'): return 'unresolved_rules'
    market=row['market']
    if market.startswith('player_'):
        if participation is False: return 'void'
        if player is None: return 'unresolved_participation'
        if player['player_id']!=row.get('player_id') or player['game_id']!=game['game_id']:
            raise ValueError('Player identity mismatch')
        stat={'player_shots_on_goal':'shots','player_goals':'goals','player_assists':'assists','player_points':'points'}[market]
        margin=(player[stat]-row['line'])*(1 if row['side']=='Over' else -1)
    elif market=='totals': margin=(game['home_score']+game['away_score']-row['line'])*(1 if row['side']=='Over' else -1)
    else:
        margin=(game['home_score']-game['away_score'])*(1 if row['side']==row['home_team'] else -1)
        if market=='spreads': margin+=row['line']
        elif market!='h2h': raise ValueError('Unknown settlement market')
    return 'won' if margin>0 else 'lost' if margin<0 else 'push'


def select(rows,now,approved=False,max_units=4):
    """Fixed shadow policy: one unit, one offer/game, no forced picks or joint parlays."""
    if not approved: return []
    eligible=[]
    for r in rows:
        if (r.get('validation_status')!='validated_executable' or r.get('analyst_status')!='reviewed'
            or not r.get('settlement_verified') or r.get('estimated_ev') is None
            or r['estimated_ev']<.02 or r.get('rank_score',-1)<=0
            or not r.get('minimum_acceptable_decimal') or decimal(r['price'])<r['minimum_acceptable_decimal']
            or not 0 <= (now-stamp(r['quoted_at'])).total_seconds()<=1800
            or stamp(r['commence_time'])<=now): continue
        eligible.append(r)
    selected=[];seen=set()
    for r in sorted(eligible,key=lambda r:r['rank_score'],reverse=True):
        if r['nhl_game_id'] in seen: continue
        selected.append(dict(r,stake_units=1));seen.add(r['nhl_game_id'])
        if len(selected)>=max_units: break
    return selected


def closing_value(entry,closing):
    # A changed line is a different contract. Never call an odds change at a new line CLV.
    if key(entry)!=key(closing) or entry['side']!=closing['side']:
        return dict(status='line_or_contract_changed',probability_clv=None,price_clv=None)
    t=stamp(closing['quoted_at']);start=stamp(entry['commence_time'])
    if not stamp(entry['quoted_at'])<=t<start or (start-t).total_seconds()>1800:
        return dict(status='no_verified_close',probability_clv=None,price_clv=None)
    fair=closing.get('other_book_probability')
    if fair is None: return dict(status='no_paired_close',probability_clv=None,price_clv=None)
    return dict(status='same_line_conditional_on_non_push',probability_clv=fair-entry['book_probability'],
                price_clv=decimal(entry['price'])*fair-1,closing_quote_at=closing['quoted_at'])


def betting_metrics(graded,worse_execution=0):
    settled=[r for r in graded if r['result'] in ['won','lost','push']]
    if not settled: return dict(count=0,turnover=0,net_units=None,roi=None,max_drawdown=None,roi_ci=None)
    by_game=defaultdict(list);profits=[];prices=[]
    for r in sorted(settled,key=lambda r:(r['commence_time'],r['offer_id'])):
        d=max(1.01,decimal(r['price'])-worse_execution)
        profit=d-1 if r['result']=='won' else -1 if r['result']=='lost' else 0
        profits.append(profit);prices.append(d);by_game[r['nhl_game_id']].append(profit)
    # Game clusters for uncertainty and chronological drawdown.
    sums=np.array([sum(v) for v in by_game.values()]);counts=np.array([len(v) for v in by_game.values()])
    rng=np.random.default_rng(48);idx=rng.integers(0,len(sums),(1000,len(sums)))
    roi=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    curve=np.r_[0,np.cumsum(sums)]
    return dict(count=len(settled),turnover=len(settled),net_units=float(sum(profits)),roi=float(np.mean(profits)),
        max_drawdown=float(np.max(np.maximum.accumulate(curve)-curve)),roi_ci=np.quantile(roi,[.025,.975]).tolist(),
        decimal_odds_quantiles=np.quantile(prices,[0,.25,.5,.75,1]).tolist(),
        game_clusters=len(by_game),staking='1 flat unit; no compounding',worse_decimal_execution=worse_execution)


def grade_snapshots(paths,games,players):
    by_game={g['game_id']:g for g in games};by_player={(r['game_id'],r['player_id']):r for r in players}
    seen=set();out=[]
    for path in paths:
        with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f: snapshot=json.load(f)
        for r in snapshot.get('rows',[]):
            if not r.get('offer_id') or r['offer_id'] in seen: continue
            seen.add(r['offer_id']);game=by_game.get(r.get('nhl_game_id'))
            # Missing appearance is unresolved, unless an explicit participation record exists.
            player=by_player.get((r.get('nhl_game_id'),r.get('player_id')))
            result=settle(r,game,player)
            out.append(dict(r,result=result,snapshot_id=snapshot.get('snapshot_id'),
                            decision_session=snapshot.get('decision_session'),stake_units=0))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,default=ROOT/'data/nhl/snapshots')
    p.add_argument('--history',type=Path,default=ROOT/'data/nhl/v2/history');p.add_argument('--output',type=Path,default=ROOT/'data/nhl/v2/grading.json')
    a=p.parse_args();g,r,_=load(a.history);graded=grade_snapshots(sorted(a.snapshots.glob('*.json*')),g,r)
    write_json(a.output,dict(rows=graded,offers=len(graded),recommendations_placed=0,
        note='Forecast grading only. No bet was placed or inferred from an observation.'))

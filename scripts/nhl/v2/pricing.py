"""Exact-offer comparisons; conditional market probabilities and unconditional EV."""
from collections import defaultdict
from datetime import timedelta
import math
from statistics import median

from .data import digest, stamp

PAIR_WINDOW = timedelta(minutes=5)
CONSENSUS_WINDOW = timedelta(minutes=15)


def decimal(american):
    if not math.isfinite(float(american)) or abs(float(american))<100:
        raise ValueError('Invalid American odds')
    return 1+american/100 if american>0 else 1+100/abs(american)


def american(dec):
    if dec<=1 or not math.isfinite(dec): return None
    return 100*(dec-1) if dec>=2 else -100/(dec-1)


def key(row):
    line=row['line']
    if row['market']=='spreads' and row['side']==row['away_team']: line=-line
    return (row['event_id'],row.get('player_id') or row['player'],row['market'],line,
            row.get('settlement_profile','unverified'),row['commence_time'],row.get('nhl_game_id'))


def push_capable(row):
    return row['market']!='h2h' and float(row['line']).is_integer()


def compare(rows,now=None):
    unique,conflicts={},set()
    for source in rows:
        row=dict(source)
        try:
            quote=stamp(row['quoted_at']); dec=decimal(row['price'])
        except (ValueError,TypeError,KeyError): continue
        if now and (quote>now or now-quote>timedelta(hours=24) or stamp(row['commence_time'])<=now): continue
        row['book_probability']=1/dec
        row.setdefault('settlement_profile',f"unverified:{row['book']}:{row['market']}")
        row.setdefault('settlement_verified',False)
        identity=(*key(row),row['book'],row['side'])
        if identity in unique:
            prev=unique[identity]
            # Newer observations replace older ones; contradictory same-time quotes fail closed.
            if quote<stamp(prev['quoted_at']): continue
            if quote==stamp(prev['quoted_at']) and row['price']!=prev['price']: conflicts.add(identity)
        unique[identity]=row
    rows=[r for k,r in unique.items() if k not in conflicts]
    pairs=defaultdict(list)
    for r in rows: pairs[(*key(r),r['book'])].append(r)
    for pair in pairs.values():
        expected={pair[0]['home_team'],pair[0]['away_team']} if pair[0]['market'] in ['h2h','spreads'] else {'Over','Under'}
        times=[stamp(r['quoted_at']) for r in pair]
        valid=len(pair)==2 and {r['side'] for r in pair}==expected and max(times)-min(times)<=PAIR_WINDOW
        total=sum(r['book_probability'] for r in pair)
        for r in pair:
            r.update(fair_probability=r['book_probability']/total if valid else None,
                     devig_method='multiplicative',market_probability_basis='conditional_on_non_push',
                     devig_sensitivity=abs(r['book_probability']/total-(r['book_probability']-(total-1)/2)) if valid else None,
                     paired_at_min=min(times).isoformat(),paired_at_max=max(times).isoformat())
    groups=defaultdict(list)
    for r in rows: groups[(*key(r),r['side'])].append(r)
    for group in groups.values():
        for r in group:
            t=stamp(r['quoted_at'])
            # Both prices of each reference pair must be in the offer's decision window.
            refs=[q for q in group if q['fair_probability'] is not None and
                  max(abs(stamp(q['paired_at_min'])-t),abs(stamp(q['paired_at_max'])-t))<=CONSENSUS_WINDOW]
            vals=[q['fair_probability'] for q in refs]
            other=[q for q in refs if q['book']!=r['book']]
            otherp=median(q['fair_probability'] for q in other) if other else None
            advantage=100*(otherp/r['book_probability']-1) if len(other)>=3 else None
            r.update(consensus_probability=median(vals) if vals else None,paired_books=len(vals),
                best_price=r['book_probability']==min(q['book_probability'] for q in group),
                other_book_probability=otherp,other_books=len(other),market_probability=otherp,
                consensus_ev=advantage if not push_capable(r) else None,
                conditional_price_advantage=advantage,comparison_settlement_verified=all(q['settlement_verified'] for q in other) and r['settlement_verified'],
                consensus_quote_times=[q['quoted_at'] for q in other],
                market_disagreement=(max(vals)-min(vals)) if vals else None)
            r['offer_id']=digest([*key(r),r['book'],r['side'],r['price'],r['quoted_at']])[:24]
    return rows


def price(probabilities,offered,minimum_ev=.02,lower_win=None,scenarios=None):
    win,push,loss=(float(probabilities[k]) for k in ['win','push','loss'])
    if min(win,push,loss)<-1e-10 or abs(win+push+loss-1)>1e-7:
        raise ValueError('Invalid settlement probabilities')
    dec=decimal(offered)
    fair=(1-push)/win if win>0 else None
    lower=win if lower_win is None else lower_win
    if not 0<=lower<=win+1e-9: raise ValueError('Invalid lower sensitivity probability')
    minimum=(1-push+minimum_ev)/lower if lower>0 else None
    rank=lower*math.log1p(.0025*(dec-1))+(1-push-lower)*math.log1p(-.0025)
    if scenarios:
        checked=[probabilities,*scenarios]
        for p in checked:
            if min(p.values())<0 or abs(sum(p.values())-1)>1e-7: raise ValueError('Invalid scenario probabilities')
        minimum=max((1-p['push']+minimum_ev)/p['win'] for p in checked) if all(p['win']>0 for p in checked) else None
        rank=min(p['win']*math.log1p(.0025*(dec-1))+p['loss']*math.log1p(-.0025) for p in checked)
    return dict(model_probability=win,independent_probability=win,final_probability=win,
        push_probability=push,loss_probability=loss,conditional_probability=win/(1-push) if push<1 else None,
        fair_decimal=fair,fair_odds=american(fair) if fair else None,
        estimated_ev=win*(dec-1)-loss,minimum_acceptable_decimal=minimum,
        minimum_acceptable_odds=american(minimum) if minimum else None,minimum_ev_target=minimum_ev,
        sensitivity_lower_win=lower,probability_basis='unconditional_win_given_action',
        rank_score=rank)


def signal(row):
    model=row.get('conditional_probability'); market=row.get('market_probability')
    if model is None: return 'market_only_observation'
    if market is None: return 'independent_forecast_only'
    hockey=model-market; shopping=market-row['book_probability']
    row['independent_market_difference']=hockey
    row['price_shopping_difference']=shopping
    if hockey>0 and shopping>0: return 'combined_signal_unvalidated'
    if hockey>0: return 'hockey_disagreement_unvalidated'
    return 'price_shopping_only' if shopping>0 else 'no_positive_signal'


def fit_blend(records):
    """For future timestamped, non-push outcomes; separate train/validation by caller.

    Returns no artifact for absent/insufficient data. Never fit on quote probabilities
    as labels. The caller must keep games grouped and verify past-only metadata.
    """
    if len({r['game_id'] for r in records})<500: return None
    for r in records:
        if not r.get('outcome_available_at') or not r.get('decision_at') or r.get('offer_book') in r.get('reference_books',[]):
            raise ValueError('Invalid blend training provenance')
        if stamp(r['outcome_available_at'])<=stamp(r['decision_at']): raise ValueError('Invalid outcome time')
    candidates=[]
    for i in range(21):
        w=i/20
        loss=0
        for r in records:
            p=max(1e-9,min(1-1e-9,(1-w)*r['independent_probability']+w*r['market_probability']))
            loss-=r['outcome']*math.log(p)+(1-r['outcome'])*math.log(1-p)
        candidates.append((loss,w))
    return dict(market_weight=min(candidates)[1],training_games=len({r['game_id'] for r in records}),
                outcome_cutoff=max(r['outcome_available_at'] for r in records),status='requires_later_validation')

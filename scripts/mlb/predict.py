"""Attach independent, pregame forecasts and conservative pick eligibility."""
from collections import Counter, defaultdict
from datetime import timedelta
import math
import os
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from nba.pipeline import iso, normal_name, timestamp
from mlb.models import LINES, means, pmf, joint, outcome, game_outcome, expected_return
from mlb.train import MODEL_PATH, signature

ET = ZoneInfo('America/New_York')
POLICY = 'mlb-picks-v1'


def published_lineups(box, game):
    """Accept a full published original order; never infer one from season stats."""
    result = {}
    for side in ['home', 'away']:
        team = box.get('teams', {}).get(side, {})
        if team.get('team', {}).get('id') != game[side+'_team_id']:
            continue
        stats = team.get('teamStats', {})
        if stats.get('batting', {}).get('plateAppearances', 0) or stats.get('pitching', {}).get('outs', 0):
            continue
        order = []
        for player in team.get('players', {}).values():
            value = int(player.get('battingOrder') or 0)
            person = player.get('person', {})
            if value and value % 100 == 0 and 1 <= value//100 <= 9 and person.get('id') and person.get('fullName'):
                order.append(dict(id=person['id'], name=person['fullName'], slot=value//100, side=side))
        if ({p['slot'] for p in order} == set(range(1, 10)) and len(order) == 9
            and len({p['id'] for p in order}) == 9):
            result[side] = order
    return result


def load_bundle(now):
    if os.getenv('MLB_TRAINING_FAILED')=='1':
        raise ValueError('The latest model training failed; forecasts are withheld')
    if not MODEL_PATH.exists():
        raise ValueError('Model training has not completed')
    # This is a locally produced artifact, never a downloaded untrusted pickle.
    bundle = joblib.load(MODEL_PATH)
    expected = (now.astimezone(ET).date()-timedelta(days=1)).isoformat()
    if bundle.get('source_signature') != signature():
        raise ValueError('Model code changed; retraining is required')
    if bundle.get('history_fetched_date') != expected:
        raise ValueError('Model history needs its daily refresh')
    return bundle


def pick_reason(row, report, now):
    market = row['market']
    audit = report.get('regular' if row['game_type']=='R' else 'postseason', {}).get(market, {})
    if not audit.get('passed'):
        return 'Research forecast: this market did not pass the applicable validation checks'
    line = row['line']
    if market != 'h2h' and not min(LINES[market]) <= line <= max(LINES[market]):
        return 'Research forecast: line outside the validation range'
    quote, start = timestamp(row.get('quoted_at')), timestamp(row.get('commence_time'))
    if not quote or not timedelta(minutes=-5) <= now-quote <= timedelta(minutes=90):
        return 'Research forecast: quote is more than 90 minutes old'
    if not start or not now < start <= now+timedelta(hours=24):
        return 'Research forecast: outside the next 24 hours'
    if row.get('fair_probability') is None or row.get('paired_books', 0) < 2:
        return 'Research forecast: fewer than two paired books at this line'
    if row.get('model_ev_pct', 0) > 30:
        return 'Research forecast: unusually large discrepancy needs manual review'
    if row.get('model_ev_pct', 0) < 3 or row.get('model_edge_pp', 0) < 3:
        return 'No pick: estimated edge is below the 3% EV / 3-point threshold'
    if not row.get('best_price'):
        return 'No pick: a better price is available at the same line'
    return None


def attach(state, now, fetch_box, bundle=None):
    rows = state['rows']
    for row in rows:
        row.update(model_probability=None, model_push_probability=None, model_mean=None,
                   model_ev_pct=None, model_edge_pp=None, is_model_pick=False,
                   model_status='Forecast unavailable')
    try:
        bundle = bundle or load_bundle(now)
    except (ValueError, OSError, EOFError) as error:
        state.update(model_status=str(error), model_summary=dict(forecasts=0, picks=0, error=str(error)))
        for row in rows:row['model_status']=str(error)
        return state
    models, history, report = bundle['models'], bundle['state'], bundle['report']
    prepared = {}
    for game in state['events']:
        if not any(r['mlb_game_id']==game['mlb_game_id'] for r in rows):continue
        try:
            if not all(game.get(s+'_pitcher', {}).get('id') if game.get(s+'_pitcher') else False for s in ['home','away']):
                raise ValueError('Waiting for both probable starters')
            if not game.get('venue_id'):raise ValueError('Official venue ID is unavailable')
            data = dict(date=timestamp(game['commence_time']).astimezone(ET).date().isoformat(),
                game_type=game['game_type'], venue=game['venue_id'],
                **{s+'_id':game[s+'_team_id'] for s in ['home','away']},
                **{s+'_starter':game[s+'_pitcher']['id'] for s in ['home','away']})
            features = {s:history.features(data,s) for s in ['home','away']}
            if any(x['team_team_games']<10 or x['opp_team_games']<10 for x in features.values()):
                raise ValueError('Insufficient recent team history')
            # Both starters are important even for game lines and opposing hitter props.
            if any(x['starter_starts']<3 for x in features.values()):
                raise ValueError('Insufficient starting-pitcher history (minimum three starts)')
            distributions = {s:pmf(means(models['team_runs'],[{'x':features[s]}]),models['team_runs'])[0] for s in features}
            matrix = joint(distributions['home'],distributions['away'])
            lineups = {}
            if timestamp(game['commence_time']) <= now+timedelta(hours=24):
                try:lineups = published_lineups(fetch_box(f"game/{game['mlb_game_id']}/boxscore"),game)
                except (RuntimeError, ValueError, KeyError, TypeError):pass
            game['lineup_status'] = 'Both batting orders published' if len(lineups)==2 else 'One batting order published' if lineups else 'Awaiting published batting orders'
            prepared[game['mlb_game_id']] = dict(game=game,data=data,x=features,joint=matrix,lineups=lineups,cache={})
        except (ValueError, KeyError) as error:
            prepared[game['mlb_game_id']] = dict(error=str(error))
    for row in rows:
        prepared_game = prepared.get(row['mlb_game_id'], {'error':'Scheduled game is unavailable'})
        try:
            if prepared_game.get('error'):raise ValueError(prepared_game['error'])
            g=prepared_game;game=g['game'];market=row['market'];inputs={}
            row['lineup_status']=game['lineup_status']
            if market in ['h2h','spreads','totals']:
                win,push=game_outcome(g['joint'],market,row['line'],row['side']==row['home_team'],row['side'])
                h,a=np.indices(g['joint'].shape)
                mean=float((g['joint']*(h+a if market=='totals' else h-a)).sum())
                inputs={f'{side} recent runs/game':round(g['x'][side]['team_runs'],2) for side in ['home','away']}
                inputs.update({f'{side} starter recent RA/9':round(g['x'][side]['starter_ra9'],2) for side in ['home','away']})
                row['model_mean_label']='Projected total runs' if market=='totals' else 'Projected home run margin'
            else:
                if market.startswith('pitcher_'):
                    players=[dict(id=game[s+'_pitcher']['id'],name=game[s+'_pitcher']['fullName'],side=s) for s in ['home','away']]
                else:
                    players=[p for lineup in g['lineups'].values() for p in lineup]
                found=[p for p in players if normal_name(p['name'])==normal_name(row['player'])]
                if len(found)!=1:
                    raise ValueError('Player is not uniquely matched to a probable starter' if market.startswith('pitcher_') else 'Awaiting this player in a published starting batting order')
                player=found[0];side=player['side'];key=(market,player['id'])
                if key not in g['cache']:
                    x=g['x'][side] if market.startswith('pitcher_') else history.features(g['data'],side,player['id'],player['slot'])
                    if market.startswith('batter_') and x['batter_pa']<50:raise ValueError('Insufficient batter history (minimum 50 plate appearances)')
                    model=models[market];mass=pmf(means(model,[{'x':x}]),model)[0]
                    g['cache'][key]=(x,mass)
                x,mass=g['cache'][key];win,push=outcome(mass,row['line'],row['side']);mean=float(mass@np.arange(len(mass)))
                row['model_player_id']=player['id'];row['model_mean_label']='Projected '+row['market_label'].lower()
                inputs=dict(zip(['Recent starter outings','Starter outs/start, last five','Starter strikeouts per batter faced','Opponent strikeouts per PA'],
                    [x['starter_starts'],round(x['starter_outs5'],2),round(x['starter_k_rate'],3),round(x['opp_k_rate'],3)])) if market.startswith('pitcher_') else {
                    'Published batting slot':player['slot'],'Recent plate appearances':x['batter_pa'],
                    'Expected PA input':round(x['projected_pa'],2),'Recent hits per PA':round(x['batter_hit_rate'],3),
                    'Opponent starter RA/9':round(x['opp_starter_ra9'],2)}
            conditional=win/(1-push)
            if not all(math.isfinite(v) for v in [win,push,mean,conditional]) or not 0<conditional<1:
                raise ValueError('Forecast distribution could not be priced')
            ev=100*expected_return(win,push,row['price']);edge=100*(conditional-row['book_probability'])
            fair=-100*conditional/(1-conditional) if conditional>=.5 else 100*(1-conditional)/conditional
            row.update(model_probability=win,model_push_probability=push,model_conditional_probability=conditional,
                model_mean=mean,model_ev_pct=ev,model_edge_pp=edge,model_fair_price=round(fair),
                model_inputs=inputs,model_version=bundle['version'],model_policy=POLICY,
                model_input_through=report['input_through'])
            reason=pick_reason(row,report,now)
            row['model_status']=reason or 'Eligible experimental model pick'
            row['is_model_pick']=reason is None
        except (ValueError, KeyError) as error:row['model_status']=str(error)
    # One offer per player/market/game, avoiding opposing or alternate-line picks.
    best={}
    for row in sorted(rows,key=lambda r:r.get('model_ev_pct') or -1e9,reverse=True):
        if row['is_model_pick']:
            key=(row['mlb_game_id'],row['player'],row['market'])
            if key in best:row.update(is_model_pick=False,model_status='Another offer is selected for this player and market')
            else:best[key]=row
    coverage=Counter(r['model_status'] for r in rows if r['model_probability'] is None)
    state.update(model_status='Independent MLB forecasts available',model_version=bundle['version'],model_checked_at=iso(now),
        model_validation=report,model_summary=dict(forecasts=sum(r['model_probability'] is not None for r in rows),
            picks=sum(r['is_model_pick'] for r in rows),unavailable_reasons=dict(coverage),
            history_through=report['input_through'],weights_trained_through=report['training_through'],policy=POLICY))
    return state

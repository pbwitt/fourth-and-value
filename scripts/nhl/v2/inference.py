"""Fail-closed adapter from frozen NHL models to the existing public quote contract."""
from datetime import datetime,timedelta,timezone
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from nba.pipeline import normal_name
from . import VERSION, FEATURE_SCHEMA
from .data import ROOT, load, stamp, iso, digest, write_json
from .features import history_at, weighted
from .models import outcome, game_outcome
from .pricing import compare, price, signal
from .review import apply_review, validate_review
from player_context import describe, nhl_context, versus, opposing

MARKETS=['player_shots_on_goal','player_goals','player_assists','player_points']
MODEL_DIR=ROOT/'models/nhl/v2'


def bundle(model_dir=MODEL_DIR):
    manifest=json.loads((model_dir/'manifest.json').read_text())
    if manifest['feature_schema']!=FEATURE_SCHEMA or manifest['version']!=VERSION:
        raise ValueError('Model/schema version mismatch')
    if hashlib.sha256((model_dir/'models.joblib').read_bytes()).hexdigest()!=manifest['artifact_sha256']:
        raise ValueError('Model checksum mismatch')
    # Only repository-controlled artifacts with manifest checksums, never remote pickles.
    return joblib.load(model_dir/'models.joblib'),manifest


def live_history(now):
    root=ROOT/'data/nhl/v2/history'
    checked_path=root/'live-check.json'
    from .data import collect
    from nhl.refresh import season_for
    last=json.loads(checked_path.read_text()) if checked_path.exists() else {}
    checked=stamp(last['checked_at']) if last.get('checked_at') else None
    if not checked or not timedelta(0)<=now-checked<timedelta(hours=12):
        collect([season_for(now)],root,now.date().isoformat())
        checked=datetime.now(timezone.utc)
        write_json(checked_path,dict(checked_at=iso(checked),season=season_for(now)))
    archive=MODEL_DIR/'history.json.gz'
    with gzip.open(archive,'rt') as f: past=json.load(f)
    manifest=json.loads((MODEL_DIR/'manifest.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['history_archive_sha256']:
        raise ValueError('Model history checksum mismatch')
    games,players,manifests=load(root,[season_for(now)])
    # Current season supersedes any same-season bootstrap history; never duplicate records.
    return ([g for g in past['games'] if g['season']!=season_for(now)]+games,
            [r for r in past['players'] if r['season']!=season_for(now)]+players,iso(checked))


def annotate(rows,games,players,events,models,manifest,now,history_checked_at,roster=None,reviews=None):
    if manifest['trained_through']>=now.date().isoformat(): raise ValueError('Model training window is not pre-decision')
    if (now.date()-datetime.fromisoformat(manifest['trained_through']).date()).days>400:
        raise ValueError('Model artifact requires annual retraining')
    if not timedelta(0)<=now-stamp(history_checked_at)<timedelta(hours=36): raise ValueError('Stale independent-model inputs')
    state=history_at(games,players,now)
    event_map={g['nhl_game_id']:g for g in events}
    team_ids={}
    for g in games:
        for side in ['home','away']: team_ids.setdefault(normal_name(g[side+'_team']),set()).add(g[side+'_id'])
    # Display only: team abbreviations label the opponent in a player's recent-game log.
    sides={x.get('game_id'):(x.get('home_id'),x.get('away_id')) for x in games}
    player_ids,abbrev={},{}
    for r in players:
        player_ids.setdefault(normal_name(r['player']),set()).add(r['player_id'])
        ids=sides.get(r.get('game_id'))
        if ids and r.get('team_abbrev'): abbrev[ids[0] if r.get('home') else ids[1]]=r['team_abbrev']
    def opponent(r):
        ids=sides.get(r.get('game_id'))
        return versus(r.get('home'),abbrev.get(ids[1] if r.get('home') else ids[0])) if ids else None
    defenses={}
    def matchup(g,records):
        """Display only: the opponent's recency-weighted shots and goals allowed, as the game-line model sees them."""
        last=records[-1]
        # His team from his previous appearance, when that team is in this game.
        if last.get('team_id') not in (g['home_id'],g['away_id']):
            return None
        rival=g['away_id'] if last['team_id']==g['home_id'] else g['home_id']
        if g['game_date'] not in defenses:
            day=datetime.fromisoformat(g['game_date']).date()
            active={tid:list(rows) for tid,rows in state.teams.items()
                    if rows and (day-datetime.fromisoformat(rows[-1]['game_date']).date()).days<=250}
            defenses[g['game_date']]={tid:dict(zip(['sa','ga'],map(float,weighted(rows,g['game_date'],['sa','ga'],[30,3]))))
                                      for tid,rows in active.items()}
        table=defenses[g['game_date']]
        if rival not in table:
            return None
        return dict(team=abbrev.get(rival),label='Opposing defense',items=[
            opposing(table,rival,'sa','Shots allowed per game','',1,'most',1),
            opposing(table,rival,'ga','Regulation goals allowed per game','',1,'most',2)])
    cache={}
    for row in rows:
        row.update(model_probability=None,independent_probability=None,final_probability=None,
            player_context=None,model_inputs=None,
            decision_at=iso(now),
            model_version=VERSION,feature_schema=FEATURE_SCHEMA,model_data_checked_at=history_checked_at,
            validation_status=manifest['validation_status'],analyst_status='unreviewed',
            recommendation=False,model_status='Independent model unavailable',market_weight=0,
            signal_type='market_only_observation',key_drivers=[],uncertainties=[],
            invalidation_conditions=['Price or line changes','Quote expires','Game rescheduled','New goalie, lineup or injury information'],
            goalie_assumption='Unconfirmed; team-level defense includes historical goalie mixture')
        official=event_map.get(row.get('nhl_game_id'))
        if not official:
            row['model_status']='Official game identity unavailable'; continue
        ids=[]
        for side in ['home','away']:
            options=team_ids.get(normal_name(row[side+'_team']),set())
            tid=official.get(side+'_id') or (next(iter(options)) if len(options)==1 else None)
            ids.append(tid)
        if None in ids:
            row['model_status']='Stable team identity unavailable'; continue
        g=dict(game_id=official['nhl_game_id'],game_date=stamp(row['commence_time']).astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).date().isoformat(),home_id=ids[0],away_id=ids[1])
        if stamp(row['commence_time'])-now>timedelta(days=2):
            row['model_status']='Independent forecast withheld beyond 48 hours'; continue
        if row['market'] in MARKETS:
            identity=player_ids.get(normal_name(row['player']),set())
            if len(identity)!=1:
                row['model_status']='Player identity or history unavailable'; continue
            pid=next(iter(identity)); records=state.players[pid]
            if not records:
                row['model_status']='No pre-decision player history'; continue
            # Roster is optional sourced live context; old team membership is never asserted current.
            if roster is not None and not any(pid in roster.get(t,[]) for t in ids):
                row['model_status']='Player not matched to this game’s current roster'; continue
            row['player_id']=pid
            # Last recorded team, kept only when it is one of this game's teams.
            row['player_team_id']=records[-1].get('team_id') if records[-1].get('team_id') in ids else None
            k=(g['game_id'],pid)
            if k not in cache:
                features=state.player_features(pid,records[-1]['position'],g['game_date'],now)
                cache[k]=(features,models['shots'].pmfs(features)[0],models['scoring'].pmfs(features))
            f,shots,scoring=cache[k]; j=MARKETS.index(row['market']); pmf=shots if j==0 else scoring[j]
            row['model_inputs']=f
            row['player_context']=describe(nhl_context,records,f,j,models['shots' if j==0 else 'scoring'].kind,VERSION,opponent,
                                           g['game_date'],pmf,describe(matchup,g,records))
            probs=outcome(pmf,row['line'],row['side'])
            # Scenario bounds, not confidence intervals or evidence of a learned injury effect.
            scenario=[]
            for multiplier in [.9,1.1]:
                changed={**f,'base_means':[x*multiplier for x in f['base_means']],
                         'opportunity_means':[x*multiplier for x in f['opportunity_means']]}
                pm=models['shots' if j==0 else 'scoring'].pmfs(changed)[j]
                scenario.append(outcome(pm,row['line'],row['side']))
            row.update(projected_mean=float(np.dot(np.arange(len(pmf)),pmf)),projected_toi=f['projected_toi'],
                history_games=f['history_games'],conditional_on_participation=True,participation_probability=None,
                lineup_assumption='Participation required for action; current role and active lineup unconfirmed',
                key_drivers=[f"{f['history_games']} prior appearances; last game {f['last_game']}",f"Projected ice time {f['projected_toi']:.1f} minutes; separately shrunk production rate"],
                uncertainties=['Line and power-play assignment not verified','Injury/return risk not quantified','Rookie estimates use position priors'],
                sensitivity=dict(assumption='Production/opportunity means ±10%; not a confidence interval',win_min=min(p['win'] for p in scenario),win_max=max(p['win'] for p in scenario)))
        else:
            k=g['game_id']
            if k not in cache:
                f=state.team_features(g,now); means=models['team'].predict(f)
                cache[k]=(means,models['team'].joint(*means),f)
            means,joint,f=cache[k]
            row['model_inputs']=f
            args=(row['market'],row['line'],row['side']==row['home_team'],row['side'])
            probs=game_outcome(joint,*args)
            scenario=[game_outcome(models['team'].joint(means[0]*h,means[1]*a),*args) for h,a in [(.9,.9),(1.1,1.1),(.9,1.1),(1.1,.9)]]
            row.update(projected_home_reg_goals=float(means[0]),projected_away_reg_goals=float(means[1]),
                key_drivers=[f"Regulation goal rates: home {means[0]:.2f}, away {means[1]:.2f}",f"Lagged home attack {f[0]['attack']:.2f}; away defense {f[0]['defense']:.2f}"],
                uncertainties=['Starting goalie and lineup unconfirmed','Empty-net effects represented only in aggregate','No shot-quality or travel-distance feature'],
                sensitivity=dict(assumption='Team rates ±10%; not a confidence interval',win_min=min(p['win'] for p in scenario),win_max=max(p['win'] for p in scenario)))
        row.update(price(probs,row['price'],lower_win=min(probs['win'],min(p['win'] for p in scenario)),scenarios=scenario))
        row['pricing_status']='standard_rules_mapped' if row.get('settlement_verified') else 'unverified_settlement'
        if not row.get('settlement_verified'):
            row.update(estimated_ev=None,minimum_acceptable_odds=None,minimum_acceptable_decimal=None)
            row['uncertainties'].append('Settlement rules unverified; EV and minimum price withheld')
        row['model_status']='Experimental independent forecast; no validated betting edge'
        row['signal_type']=signal(row)
        if row.get('conditional_price_advantage') is not None:
            row['consensus_ev']=row['conditional_price_advantage']*(1-row['push_probability'])
            row['consensus_ev_basis']='Market conditional probability with independent-model push estimate'
        row['forecast_id']=digest([row['offer_id'],manifest['artifact_sha256'],history_checked_at,probs])[:24]
    for i,row in enumerate(rows):
        for review in reviews or []:
            if review.get('offer_id')==row.get('offer_id') and review.get('forecast_id')==row.get('forecast_id'):
                rows[i]=apply_review(row,validate_review(review,now))
    return rows


def enrich(state,now,offline_inputs=None):
    """A modeling failure cannot hide an otherwise healthy market snapshot."""
    state['schema_version']=2
    state['recommendations']=[]
    state['model_version']=VERSION
    try:
        models,manifest=bundle()
        games,players,checked=offline_inputs or live_history(now)
        # The prediction decision follows ingestion; a cached history check keeps its actual age.
        decision_now=now if offline_inputs else datetime.now(timezone.utc)
        ledger=ROOT/'artifacts/nhl/reviews.jsonl'
        reviews=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        state['rows']=annotate(state['rows'],games,players,state['events'],models,manifest,decision_now,checked,reviews=reviews)
        state['model_status']='Experimental independent forecasts; market blend and recommendations disabled'
        state['model_manifest']=manifest
        # Count-distribution parameters, archived so cross-market checks can be reproduced.
        # A failure here withholds those checks only, never the independent forecasts.
        try:
            from .coherence import distribution
            state['model_distribution']=distribution(models,manifest)
        except Exception:
            state['model_distribution']=None
        state['model_data_checked_at']=checked
        state['model_prediction_at']=iso(decision_now)
        state['model_error']=None
    except Exception as error:
        # No exception URL or credentials; no stale model fallback.
        state['model_error']=f'Independent model unavailable ({type(error).__name__})'
        state['model_distribution']=None
        state['model_status']=state['model_error']
        for row in state['rows']:
            for field in ['forecast_id','estimated_ev','fair_odds','fair_decimal','minimum_acceptable_odds',
                          'minimum_acceptable_decimal','push_probability','loss_probability','conditional_probability',
                          'rank_score','analyst_probability','independent_market_difference','model_inputs','player_context','player_team_id']:
                row[field]=None
            row.update(model_probability=None,independent_probability=None,final_probability=None,
                       model_status=state['model_error'],validation_status='unavailable',recommendation=False,
                       signal_type='market_only_observation')
    return state

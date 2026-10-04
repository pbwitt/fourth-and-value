"""One feature builder shared by reconstructed evaluation and daily inference."""
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .data import iso, stamp

TEAM_FEATURES = ['home', 'attack', 'defense', 'shots_for', 'shots_against',
                 'save_rate', 'opp_save_rate', 'pp', 'opp_pk', 'rest', 'opp_rest', 'b2b', 'opp_b2b']
CORE_FEATURES = ['home', 'attack', 'defense']
PLAYER_STATS = ['shots', 'goals', 'assists', 'points']
# Player history is aged by the player's own appearances, newest = 0, so an offseason, injury or
# break does not erase it. Half-lives are in games, chosen on the 2023-24 and 2024-25 validation
# folds only (feature schema nhl-pit-2); position priors and prior strengths are unchanged.
PLAYER_HALF_LIVES = dict(base=82.5, toi=14, rate=110)


def decision_time(day, hour=10, minute=30):
    return datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=ZoneInfo('America/New_York'))


def weighted(records, day, fields, priors, strength=12, half_life=90, ages=None):
    """Recency-weighted mean shrunk toward `priors`; ages default to calendar days before `day`."""
    if not records:
        return np.asarray(priors, dtype=float)
    if ages is None:
        target = datetime.fromisoformat(day)
        ages = np.array([(target-datetime.fromisoformat(r['game_date'])).days for r in records])
    weights = np.exp2(-np.asarray(ages, dtype=float) / half_life)
    values = np.array([[r[f] for f in fields] for r in records])
    return (weights @ values + strength * np.asarray(priors)) / (weights.sum()+strength)


class History:
    def __init__(self):
        self.teams = defaultdict(lambda:deque(maxlen=164))
        self.players = defaultdict(lambda:deque(maxlen=164))
        self.max_available = None

    def add_game(self, g):
        for side, opp in [('home','away'),('away','home')]:
            r = dict(game_date=g['game_date'], available_at=g['available_at'],
                     gf=g[side+'_reg_goals'], ga=g[opp+'_reg_goals'],
                     sf=g[side+'_shots'], sa=g[opp+'_shots'],
                     pp=g[side+'_pp_pct'] if g[side+'_pp_pct'] is not None else .20,
                     pk=g[side+'_pk_pct'] if g[side+'_pk_pct'] is not None else .80)
            self.teams[g[side+'_id']].append(r)
        self.max_available = max(self.max_available or g['available_at'],g['available_at'])

    def add_player(self, r):
        self.players[r['player_id']].append(r)

    def team_features(self, g, asof):
        if self.max_available and stamp(self.max_available) > asof:
            raise ValueError('Future history in feature state')
        day = g['game_date']
        records = [self.teams[g[s+'_id']] for s in ['home','away']]
        means = [weighted(r,day,['gf','ga','sf','sa','pp','pk'],[3,3,30,30,.2,.8]) for r in records]
        rest = [min(7,max(0,(datetime.fromisoformat(day)-datetime.fromisoformat(r[-1]['game_date'])).days-1)) if r else 7 for r in records]
        out = []
        for i in range(2):
            own,opp = means[i],means[1-i]
            out.append(dict(zip(TEAM_FEATURES,[int(i==0),own[0],opp[1],own[2],opp[3],
                1-own[1]/max(own[3],1),1-opp[1]/max(opp[3],1),own[4],opp[5],
                rest[i],rest[1-i],int(rest[i]==0),int(rest[1-i]==0)])))
        return out

    def player_features(self, pid, position, day, asof):
        records = self.players[pid]
        if records and stamp(records[-1]['available_at']) > asof:
            raise ValueError('Future player history')
        # Fixed cold-start priors; these are never season-final summaries.
        priors = [1.6,.12,.30] if position == 'D' else [1.9,.25,.34]
        # Records are in arrival order, so this counts the player's later appearances.
        games = np.arange(len(records))[::-1]
        means = weighted(records,day,['shots','goals','assists'],priors,half_life=PLAYER_HALF_LIVES['base'],ages=games)
        toi = weighted(records,day,['toi'],[18 if position=='D' else 15],strength=5,
                       half_life=PLAYER_HALF_LIVES['toi'],ages=games)[0]
        # Shrink per-minute production separately from projected opportunity.
        records_rate = [{**r, **{s:r[s]/r['toi'] for s in PLAYER_STATS[:3]}} for r in records]
        rates = weighted(records_rate,day,PLAYER_STATS[:3],np.asarray(priors)/(18 if position=='D' else 15),
                         half_life=PLAYER_HALF_LIVES['rate'],ages=games)
        return dict(player_id=pid, history_games=len(records), projected_toi=float(toi),
                    last_game=records[-1]['game_date'] if records else None,
                    base_means=[*means,float(means[1]+means[2])],
                    opportunity_means=[*(toi*rates),float(toi*(rates[1]+rates[2]))],
                    feature_cutoff=records[-1]['available_at'] if records else None)


def build(games, players):
    """Every game on a calendar date sees only results available by that morning."""
    history = History()
    by_game = defaultdict(list)
    for r in players:
        by_game[r['game_id']].append(r)
    arrivals = sorted(games,key=lambda g:(g['available_at'],g['game_id']))
    cursor, team_rows, player_rows = 0, [], []
    for g in sorted(games,key=lambda g:(g['game_date'],g['game_id'])):
        asof = decision_time(g['game_date'])
        while cursor < len(arrivals) and stamp(arrivals[cursor]['available_at']) <= asof:
            past = arrivals[cursor]
            history.add_game(past)
            for r in by_game[past['game_id']]: history.add_player(r)
            cursor += 1
        features = history.team_features(g,asof)
        for i,side in enumerate(['home','away']):
            team_rows.append(dict(**features[i],game_id=g['game_id'],season=g['season'],
                game_date=g['game_date'], decision_at=iso(asof), feature_cutoff=history.max_available,
                team_id=g[side+'_id'], opponent_id=g[('away' if side=='home' else 'home')+'_id'],
                target=g[side+'_reg_goals'], final_score=g[side+'_score']))
        for r in by_game[g['game_id']]:
            # A target-game position is retrospective context too. Use last observed position;
            # cold-start players receive the fixed unknown-position prior.
            prior = history.players[r['player_id']]
            position = prior[-1]['position'] if prior else 'U'
            f = history.player_features(r['player_id'],position,g['game_date'],asof)
            player_rows.append(dict(**f,game_id=g['game_id'],season=g['season'], game_date=g['game_date'],
                decision_at=iso(asof), player=r['player'],position=position,
                targets=[r[s] for s in PLAYER_STATS],actual_toi=r['toi']))
    return team_rows,player_rows


def history_at(games,players,asof):
    history = History()
    eligible = {g['game_id'] for g in games if stamp(g['available_at']) <= asof}
    for g in sorted(games,key=lambda g:(g['game_date'],g['game_id'])):
        if g['game_id'] in eligible: history.add_game(g)
    for r in sorted(players,key=lambda r:(r['game_date'],r['game_id'])):
        if r['game_id'] in eligible and stamp(r['available_at']) <= asof: history.add_player(r)
    return history

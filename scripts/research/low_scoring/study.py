"""Situational low-scoring study (Technical Report FV-2026-04): factor tests against closing totals.

Run from the repository root: python3 scripts/research/low_scoring/study.py
Reads the game tables in reports/low-scoring-2026/data/ (built by fetch.py in Actions) and writes
reports/low-scoring-2026/results.json. Hypotheses, periods and pass rules are fixed in
reports/low-scoring-2026/PREREGISTRATION.md; nothing here is tuned on outcomes.
"""
import gzip
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get('LOW_SCORING_DATA', ROOT / 'reports/low-scoring-2026/data'))
OUT = Path(os.environ.get('LOW_SCORING_OUT', ROOT / 'reports/low-scoring-2026/results.json'))
BREAK_EVEN = 110 / 210          # win rate needed at -110
MIN_GAMES = 100
OFFICIAL_MIN_GAMES = 30


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def side_summary(r):
    """Under/over record on residuals r (pushes excluded)."""
    r = np.asarray(r, float)
    under, over = int((r < 0).sum()), int((r > 0).sum())
    n = under + over
    lo, hi = wilson(under, n)
    return dict(under=under, over=over, push=int((r == 0).sum()), under_rate=under / n if n else None,
                under_lo=lo, under_hi=hi)


def compare(sub, flag):
    """Factor games (flag True) against the rest of the same universe."""
    f, o = sub[flag], sub[~flag]
    out = dict(n=int(len(f)), n_other=int(len(o)))
    if len(f) < 2 or len(o) < 2:
        return out
    rf, ro = f.residual.to_numpy(float), o.residual.to_numpy(float)
    t = stats.ttest_ind(rf, ro, equal_var=False)
    se = math.sqrt(rf.var(ddof=1) / len(rf) + ro.var(ddof=1) / len(ro))
    diff = rf.mean() - ro.mean()
    out.update(
        actual=float(f.actual.mean()), actual_other=float(o.actual.mean()),
        effect=float(f.actual.mean() - o.actual.mean()),
        line=float(f.line.mean()), line_other=float(o.line.mean()),
        priced=float(f.line.mean() - o.line.mean()),
        residual=float(rf.mean()), residual_other=float(ro.mean()),
        residual_lo=float(rf.mean() - 1.96 * rf.std(ddof=1) / math.sqrt(len(rf))),
        residual_hi=float(rf.mean() + 1.96 * rf.std(ddof=1) / math.sqrt(len(rf))),
        excess=float(diff), excess_lo=float(diff - 1.96 * se), excess_hi=float(diff + 1.96 * se),
        p_two=float(t.pvalue), t=float(t.statistic), **side_summary(rf))
    out['over_rate'] = None if out['under_rate'] is None else 1 - out['under_rate']
    return out


def price_roi(sub, side):
    """Return per 1-unit bet at the recorded closing price (NFL only; pushes return the stake)."""
    col = f'{side}_odds'
    if col not in sub or sub[col].isna().all():
        return None
    x = sub[sub[col].notna()]
    odds = x[col].astype(float)
    win_ = (x.residual < 0) if side == 'under' else (x.residual > 0)
    push = x.residual == 0
    profit = np.where(push, 0.0, np.where(win_, np.where(odds > 0, odds / 100, 100 / odds.abs()), -1.0))
    return dict(n=int(len(x)), roi=float(profit.mean()))


def one_sided(p_two, t, sign):
    if p_two is None or t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    return p_two / 2 if np.sign(t) == sign else 1 - p_two / 2


def bh(pvals):
    """Benjamini-Hochberg q-values for a dict id -> p."""
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, q, prev = len(items), {}, 1.0
    for rank in range(m, 0, -1):
        p, k = items[rank - 1]
        prev = min(prev, p * m / rank)
        q[k] = prev
    return q


def roi(win_rate):
    return None if win_rate is None else win_rate * (100 / 110) - (1 - win_rate)


# ---------------------------------------------------------------- shared schedule helpers

def team_games(df):
    """One row per team per game, sorted by date, with home flag and venue key."""
    a = df[['gid', 'date', 'season', 'away', 'home', 'venue_tz', 'game_type']].rename(columns={'away': 'team', 'home': 'opp'})
    a['is_home'] = False
    h = df[['gid', 'date', 'season', 'home', 'away', 'venue_tz', 'game_type']].rename(columns={'home': 'team', 'away': 'opp'})
    h['is_home'] = True
    t = pd.concat([a, h]).sort_values(['team', 'date', 'gid']).reset_index(drop=True)
    t['day'] = pd.to_datetime(t.date)
    g = t.groupby(['team', 'season'], sort=False)
    t['prev_day'] = g.day.shift(1)
    t['rest'] = (t.day - t.prev_day).dt.days - 1          # 0 = back-to-back
    t['prev_tz'] = g.venue_tz.shift(1)
    t['prev2_day'] = g.day.shift(2)
    t['third_in_four'] = (t.day - t.prev2_day).dt.days <= 3
    # consecutive road games including this one; consecutive road games just before a home game
    streak, run = [], 0
    for team_season, rows in t.groupby(['team', 'season'], sort=False):
        run = 0
        for is_home in rows.is_home:
            run = 0 if is_home else run + 1
            streak.append(run)
    t['road_streak'] = streak
    t['prev_road_streak'] = t.groupby(['team', 'season'], sort=False).road_streak.shift(1).fillna(0)
    return t


def attach(df, t, cols):
    """Attach team-game columns as away_*/home_* on the game table."""
    a = t[~t.is_home].set_index('gid')[cols].add_prefix('away_')
    h = t[t.is_home].set_index('gid')[cols].add_prefix('home_')
    return df.join(a, on='gid').join(h, on='gid')


# ---------------------------------------------------------------- per-sport factor definitions

PACIFIC = {'SEA', 'SF', 'OAK', 'LV', 'SD', 'LAC', 'LA', 'LAR'}
EASTERN = {'ATL', 'BAL', 'BUF', 'CAR', 'CIN', 'CLE', 'DET', 'IND', 'JAX', 'MIA', 'NE', 'NYG', 'NYJ', 'PHI', 'PIT', 'TB', 'WAS'}


def nfl(df):
    df = df.copy()
    df['outdoor'] = df.roof.isin(['outdoors', 'open'])
    df['week'] = df.week.astype(int)
    playoffs = df.game_type != 'REG'
    # A team's home roof: its most common roof at home games that season.
    home_roof = df[df.location == 'Home'].groupby(['season', 'home']).roof.agg(lambda s: s.mode().iat[0])
    away_roof = [home_roof.get((s, a)) for s, a in zip(df.season, df.away)]
    df['away_dome_team'] = [r in ('dome', 'closed') for r in away_roof]
    w = df.weather_text.fillna('').str.lower()
    precip = w.str.contains(r'rain|snow|shower|drizzle|sleet|flurr', regex=True) & ~w.str.contains(r'no rain|0% chance|chance of rain: 0', regex=True)
    snow = w.str.contains(r'snow|flurr', regex=True)
    # Starting-QB change: today's starter differs from last game's, who had started 3 of the last 4.
    rows = []
    for side in ('away', 'home'):
        x = df[['gid', 'date', 'season', side, f'{side}_qb_id']].rename(columns={side: 'team', f'{side}_qb_id': 'qb'})
        rows.append(x)
    q = pd.concat(rows).sort_values(['team', 'date', 'gid'])
    new_qb = {}
    for _, grp in q.groupby('team', sort=False):
        hist = []
        for gid, qb in zip(grp.gid, grp.qb):
            flag = bool(hist) and isinstance(qb, str) and qb != hist[-1] and sum(h == hist[-1] for h in hist[-4:]) >= 3
            new_qb[gid] = new_qb.get(gid, False) or flag
            hist.append(qb)
    df['new_qb'] = df.gid.map(lambda g: bool(new_qb.get(g, False)))
    # Win rate entering the game, by team and season (regular season only).
    rec = []
    for side, other in (('away', 'home'), ('home', 'away')):
        x = df[['gid', 'date', 'season', 'game_type', side, f'{side}_score', f'{other}_score']].copy()
        x.columns = ['gid', 'date', 'season', 'game_type', 'team', 'pf', 'pa']
        x['side'] = side
        rec.append(x)
    r = pd.concat(rec).sort_values(['team', 'date'])
    r['w'] = np.where(r.pf > r.pa, 1.0, np.where(r.pf == r.pa, .5, 0.0))
    r = r[r.game_type == 'REG']
    g = r.groupby(['team', 'season'])
    r['wins_before'] = g.w.cumsum() - r.w
    r['games_before'] = g.cumcount()
    r['pct_before'] = r.wins_before / r.games_before.replace(0, np.nan)
    pct = r.set_index(['gid', 'side']).pct_before
    df['away_pct'] = [pct.get((g_, 'away')) for g_ in df.gid]
    df['home_pct'] = [pct.get((g_, 'home')) for g_ in df.gid]
    # Second divisional meeting in a season.
    pair = df.apply(lambda x: (x.season, tuple(sorted((x.away, x.home)))), axis=1)
    df['meeting'] = df.assign(pair=pair).sort_values('date').groupby('pair').cumcount().reindex(df.index) + 1
    tod = df.gametime.fillna('00:00')
    known_weather = df.outdoor & df.weather_text.notna()
    f = {
        'NFL-1': dict(flag=df.wind >= 15, universe=df.outdoor & df.wind.notna(), name='Strong wind (15+ mph)', dir='U', group='Weather'),
        'NFL-2': dict(flag=df.temp <= 32, universe=df.outdoor & df.temp.notna(), name='Freezing (32°F or colder)', dir='U', group='Weather'),
        'NFL-3': dict(flag=precip, universe=known_weather, name='Rain or snow', dir='U', group='Weather'),
        'NFL-4': dict(flag=snow, universe=known_weather, name='Snow', dir='U', group='Weather'),
        'NFL-5': dict(flag=df.away_dome_team & df.outdoor & (df.temp <= 40), universe=pd.Series(True, index=df.index), name='Dome team in the cold', dir='U', group='Weather'),
        'NFL-6': dict(flag=(df.weekday == 'Thursday') & (df.week > 1) & (df.away_rest <= 4) & (df.home_rest <= 4), name='Thursday night, short rest', dir='U', group='Schedule'),
        'NFL-7': dict(flag=(df.weekday == 'Sunday') & ((df.away_rest == 6) | (df.home_rest == 6)), name='Short week after Monday night', dir='U', group='Schedule'),
        'NFL-8': dict(flag=~playoffs & ((df.away_rest >= 13) | (df.home_rest >= 13)), universe=~playoffs, name='Off a bye week', dir='2', group='Schedule'),
        'NFL-9': dict(flag=tod >= '20:00', name='Prime time (8 p.m. ET or later)', dir='U', group='Schedule'),
        'NFL-10': dict(flag=df.div_game == 1, name='Divisional game', dir='U', group='Familiarity'),
        'NFL-11': dict(flag=(df.div_game == 1) & (df.meeting == 2) & (df.week >= 10) & ~playoffs, name='Late divisional rematch', dir='U', group='Familiarity'),
        'NFL-12': dict(official='referee', name='Low-scoring referee', dir='U', group='Officials'),
        'NFL-13': dict(flag=df.away.isin(PACIFIC) & df.home.isin(EASTERN) & (df.location == 'Home') & (tod < '14:00') & (tod > '00:00'), name='West Coast team, 1 p.m. ET kickoff', dir='U', group='Travel'),
        'NFL-14': dict(flag=~playoffs & (df.week >= 15) & ((df.away_pct <= .3) | (df.home_pct <= .3)), universe=~playoffs, name='Late season, a bad team', dir='2', group='Stakes'),
        'NFL-15': dict(flag=df.new_qb, name='New starting quarterback', dir='U', group='Personnel'),
        'NFL-16': dict(flag=(df.home == 'DEN') & (df.location == 'Home'), name='Altitude (Denver)', dir='2', group='Travel'),
        'NFL-17': dict(flag=playoffs, name='Playoffs', dir='U', group='Stakes'),
        'NFL-18': dict(flag=(df.location == 'Neutral') & (df.game_type != 'SB'), name='Neutral or international site', dir='2', group='Travel'),
    }
    return df, f



NHL_DIVISIONS = {
    'old': {'Atlantic': 'NJD NYI NYR PHI PIT', 'Northeast': 'BOS BUF MTL OTT TOR', 'Southeast': 'ATL WPG CAR FLA TBL WSH',
            'Central': 'CHI CBJ DET NSH STL', 'Northwest': 'CGY COL EDM MIN VAN', 'Pacific': 'ANA DAL LAK PHX ARI SJS'},
    'four': {'Atlantic': 'BOS BUF DET FLA MTL OTT TBL TOR', 'Metropolitan': 'CAR CBJ NJD NYI NYR PHI PIT WSH',
             'Central': 'CHI COL DAL MIN NSH STL WPG', 'Pacific': 'ANA ARI PHX CGY EDM LAK SJS VAN VGK'},
    'covid': {'North': 'CGY EDM MTL OTT TOR VAN WPG', 'East': 'BOS BUF NJD NYI NYR PHI PIT WSH',
              'Central': 'CAR CHI CBJ DAL DET FLA NSH TBL', 'West': 'ANA ARI COL LAK MIN SJS STL VGK'},
    'now': {'Atlantic': 'BOS BUF DET FLA MTL OTT TBL TOR', 'Metropolitan': 'CAR CBJ NJD NYI NYR PHI PIT WSH',
            'Central': 'ARI UTA CHI COL DAL MIN NSH STL WPG', 'Pacific': 'ANA CGY EDM LAK SJS SEA VAN VGK'},
}


def nhl_division(season, team):
    era = 'old' if season <= 2012 else 'four' if season <= 2019 else 'covid' if season == 2020 else 'now'
    for name, teams in NHL_DIVISIONS[era].items():
        if team in teams.split():
            return name
    return None


OUTDOOR = re.compile(r'stadium|field|park|bowl|tahoe|lansdowne|td place', re.I)
INDOOR_WORDS = re.compile(r'arena|centre|center|garden|coliseum|forum|dome|place\b', re.I)


def offset_hours(v):
    if not isinstance(v, str) or ':' not in v:
        return np.nan
    sign = -1 if v.startswith('-') else 1
    h, m = v.lstrip('+-').split(':')
    return sign * (int(h) + int(m) / 60)


def last_two_weeks(df):
    reg = df[df.reg]
    last = reg.groupby('season').date.max().map(lambda d: (pd.Timestamp(d) - pd.Timedelta(days=13)).date().isoformat())
    return df.reg & (df.date >= df.season.map(last))


def first_days(df, days):
    reg = df[df.reg]
    first = reg.groupby('season').date.min().map(lambda d: (pd.Timestamp(d) + pd.Timedelta(days=days - 1)).date().isoformat())
    return df.reg & (df.date <= df.season.map(first))


def schedule_features(df):
    t = team_games(df)
    t['tz_jump'] = (t.venue_tz - t.prev_tz).abs()
    t['first_of_season'] = t.prev_day.isna()
    df = attach(df, t, ['rest', 'third_in_four', 'road_streak', 'prev_road_streak', 'tz_jump', 'first_of_season'])
    return df, t


def nhl(df):
    df = df.copy()
    df['reg'] = df.game_type == 2
    df['venue_tz'] = df.venue_offset.map(offset_hours)
    df, t = schedule_features(df)
    b2b_a, b2b_h = df.away_rest == 0, df.home_rest == 0
    # Backup goalie: today's starter is not the team's season-to-date leader in starts (10+ team games).
    starts = []
    for side in ('away', 'home'):
        x = pd.DataFrame({'gid': df.gid, 'date': df.date, 'season': df.season, 'team': df[side],
                          'pid': df[f'{side}_goalie'].map(lambda g: g[0] if isinstance(g, list) else None), 'side': side})
        starts.append(x)
    s_ = pd.concat(starts).sort_values(['team', 'season', 'date', 'gid'])
    backup = {}
    for _, grp in s_.groupby(['team', 'season'], sort=False):
        counts, games = {}, 0
        for gid, pid, side in zip(grp.gid, grp.pid, grp.side):
            if games >= 10 and pid is not None and counts:
                top = max(counts.values())
                backup[(gid, side)] = counts.get(pid, 0) < top
            if pid is not None:
                counts[pid] = counts.get(pid, 0) + 1
            games += 1
    df['away_backup'] = [backup.get((g, 'away')) for g in df.gid]
    df['home_backup'] = [backup.get((g, 'home')) for g in df.gid]
    known_goalies = df.away_backup.notna() & df.home_backup.notna()
    div = [nhl_division(s, a) is not None and nhl_division(s, a) == nhl_division(s, h) for s, a, h in zip(df.season, df.away, df.home)]
    venue = df.venue.fillna('')
    outdoor = venue.str.contains(OUTDOOR) & ~venue.str.contains(INDOOR_WORDS)
    f = {
        'NHL-1': dict(flag=b2b_a | b2b_h, name='Back-to-back (either team)', dir='O', group='Schedule'),
        'NHL-2': dict(flag=b2b_a & b2b_h, name='Both teams on a back-to-back', dir='2', group='Schedule'),
        'NHL-3': dict(flag=(b2b_a & (df.home_rest >= 2)) | (b2b_h & (df.away_rest >= 2)), name='Rest mismatch', dir='O', group='Schedule'),
        'NHL-4': dict(flag=(df.away_backup == True) | (df.home_backup == True), universe=known_goalies, name='Backup goalie starts', dir='O', group='Personnel'),  # noqa: E712
        'NHL-5': dict(flag=(df.away_third_in_four == True) | (df.home_third_in_four == True), name='Third game in four nights', dir='2', group='Schedule'),  # noqa: E712
        'NHL-6': dict(flag=df.away_road_streak >= 4, name='Long road trip (4th+ straight road game)', dir='2', group='Travel'),
        'NHL-7': dict(flag=df.home_prev_road_streak >= 4, name='Homecoming after a long trip', dir='2', group='Travel'),
        'NHL-8': dict(flag=(df.away_tz_jump >= 2) & (df.away_rest <= 1), name='Road team crossed 2+ time zones', dir='2', group='Travel'),
        'NHL-9': dict(flag=pd.Series(div, index=df.index), name='Divisional game', dir='U', group='Familiarity'),
        'NHL-10': dict(flag=~df.reg, name='Playoffs', dir='U', group='Stakes'),
        'NHL-11': dict(official='referees', name='Low-scoring referee', dir='U', group='Officials'),
        'NHL-12': dict(flag=outdoor, name='Outdoor game', dir='U', group='Weather'),
        'NHL-13': dict(flag=df.reg & ((df.away_rest >= 4) | (df.home_rest >= 4)), universe=df.reg, name='Rust after 4+ days off', dir='U', group='Schedule'),
        'NHL-14': dict(flag=df.reg & (pd.to_datetime(df.date).dt.month == 10), universe=df.reg, name='October', dir='2', group='Calendar'),
        'NHL-15': dict(flag=last_two_weeks(df), universe=df.reg, name="Season's final two weeks", dir='2', group='Stakes'),
    }
    df.attrs['outdoor_venues'] = sorted(set(venue[outdoor]))
    return df, f


NBA_TZ = {**{t: -5 for t in 'ATL BOS BKN NJ CHA CLE DET IND MIA NY ORL PHI TOR WSH'.split()},
          **{t: -6 for t in 'CHI DAL HOU MEM MIL MIN NO OKC SA'.split()},
          **{t: -7 for t in 'DEN UTAH PHX'.split()},
          **{t: -8 for t in 'GS LAC LAL POR SAC SEA'.split()}}


def all_star_return(df):
    """Each season's first regular-season day after the longest mid-season gap in the league schedule."""
    out = {}
    reg = df[df.reg]
    for season, grp in reg.groupby('season'):
        days = sorted(pd.to_datetime(grp.date.unique()))
        lo, hi = pd.Timestamp(f'{season + 1}-01-20'), pd.Timestamp(f'{season + 1}-03-31')
        best = None
        for a, b in zip(days, days[1:]):
            if lo <= a <= hi and (best is None or (b - a) > (best[1] - best[0])):
                best = (a, b)
        if best:
            out[season] = best[1].date().isoformat()
    return out


def nba(df):
    df = df.copy()
    df['reg'] = df.game_type == 2
    df['venue_tz'] = df.home.map(NBA_TZ)
    df, t = schedule_features(df)
    b2b_a, b2b_h = df.away_rest == 0, df.home_rest == 0
    ret = all_star_return(df)
    t2 = t.copy()
    t2['ret'] = t2.season.map(ret)
    t2 = t2[t2.game_type == 2]
    first_back = t2[t2.date >= t2.ret].groupby(['team', 'season']).gid.first()
    fb = set(first_back.values)
    df['chief'] = df.officials.map(lambda o: o[0] if isinstance(o, list) and o else None)
    rest_a, rest_h = df.away_rest.clip(upper=7), df.home_rest.clip(upper=7)
    f = {
        'NBA-1': dict(flag=b2b_a | b2b_h, name='Back-to-back (either team)', dir='U', group='Schedule'),
        'NBA-2': dict(flag=b2b_a & b2b_h, name='Both teams on a back-to-back', dir='U', group='Schedule'),
        'NBA-3': dict(flag=b2b_a & (df.home_rest >= 1), name='Tired visitor (road back-to-back)', dir='U', group='Schedule'),
        'NBA-4': dict(flag=(df.away_third_in_four == True) | (df.home_third_in_four == True), name='Third game in four nights', dir='U', group='Schedule'),  # noqa: E712
        'NBA-5': dict(flag=df.home.isin(['DEN', 'UTAH']) & (df.neutral != True), name='Altitude (Denver, Utah)', dir='2', group='Travel'),  # noqa: E712
        'NBA-6': dict(flag=(df.away_tz_jump >= 2) & (df.away_rest <= 1), name='Road team crossed 2+ time zones', dir='U', group='Travel'),
        'NBA-7': dict(flag=df.away_road_streak >= 5, name='Long road trip (5th+ straight road game)', dir='U', group='Travel'),
        'NBA-8': dict(flag=df.gid.isin(fb), universe=df.reg, name='First game after the All-Star break', dir='2', group='Calendar'),
        'NBA-9': dict(flag=last_two_weeks(df), universe=df.reg, name="Season's final two weeks", dir='2', group='Stakes'),
        'NBA-10': dict(flag=~df.reg, name='Playoffs', dir='U', group='Stakes'),
        'NBA-11': dict(official='chief', name='Low-scoring crew chief', dir='U', group='Officials'),
        'NBA-12': dict(flag=first_days(df, 14), universe=df.reg, name='Opening two weeks', dir='2', group='Calendar'),
        'NBA-13': dict(flag=df.reg & ((rest_a - rest_h).abs() >= 2) & (df.away_first_of_season == False) & (df.home_first_of_season == False), universe=df.reg, name='Rest advantage (2+ days)', dir='2', group='Schedule'),  # noqa: E712
    }
    return df, f


def mlb(df):
    df = df.copy()
    df['reg'] = df.game_type == 'R'
    cond = df.condition.fillna('')
    df['outdoor'] = ~cond.isin(['Dome', 'Roof Closed']) & (cond != '')
    wdir = df.wind_dir.fillna('')
    wind_in = df.outdoor & (df.wind_mph >= 10) & wdir.str.startswith('In')
    wind_out = df.outdoor & (df.wind_mph >= 10) & wdir.str.startswith('Out')
    wind_known = df.outdoor & df.wind_mph.notna()
    wrigley = df.venue == 'Wrigley Field'
    local = [pd.Timestamp(s).tz_convert(z) if isinstance(s, str) and isinstance(z, str) else pd.NaT for s, z in zip(df.start, df.tz)]
    minutes = pd.Series([x.hour * 60 + x.minute if x is not pd.NaT else np.nan for x in local], index=df.index)
    # Team schedule: previous and next game per team.
    rows = []
    for side in ('away', 'home'):
        rows.append(pd.DataFrame({'gid': df.gid, 'date': pd.to_datetime(df.date), 'start': df.start, 'team': df[side],
                                  'venue': df.venue_id, 'night': df.day_night == 'night', 'side': side}))
    t = pd.concat(rows).sort_values(['team', 'date', 'start', 'gid'])
    g = t.groupby('team', sort=False)
    t['prev_date'], t['prev_night'] = g.date.shift(1), g.night.shift(1)
    t['next_date'], t['next_venue'] = g.date.shift(-1), g.venue.shift(-1)
    t['after_night'] = (t.prev_date == t.date - pd.Timedelta(days=1)) & (t.prev_night == True)  # noqa: E712
    t['getaway'] = (t.next_date == t.date + pd.Timedelta(days=1)) & (t.next_venue != t.venue)
    ti = t.set_index(['gid', 'side'])
    after_night = pd.Series([bool(ti.after_night.get((x, 'away'))) or bool(ti.after_night.get((x, 'home'))) for x in df.gid], index=df.index)
    getaway = pd.Series([bool(ti.getaway.get((x, 'away'))) or bool(ti.getaway.get((x, 'home'))) for x in df.gid], index=df.index)
    f = {
        'MLB-1': dict(flag=df.temp <= 50, universe=df.outdoor & df.temp.notna(), name='Cold (50°F or colder)', dir='U', group='Weather'),
        'MLB-2': dict(flag=df.temp >= 85, universe=df.outdoor & df.temp.notna(), name='Heat (85°F or hotter)', dir='O', group='Weather'),
        'MLB-3': dict(flag=wind_in, universe=wind_known, name='Wind blowing in (10+ mph)', dir='U', group='Weather'),
        'MLB-4': dict(flag=wind_out, universe=wind_known, name='Wind blowing out (10+ mph)', dir='O', group='Weather'),
        'MLB-5': dict(flag=wind_in, universe=wind_known & wrigley, name='Wrigley Field, wind in', dir='U', group='Weather'),
        'MLB-6': dict(flag=wind_out, universe=wind_known & wrigley, name='Wrigley Field, wind out', dir='O', group='Weather'),
        'MLB-7': dict(flag=cond.isin(['Dome', 'Roof Closed']), universe=cond != '', name='Roof closed or dome', dir='2', group='Weather'),
        'MLB-8': dict(flag=cond.str.contains('Rain|Drizzle', regex=True), universe=df.outdoor, name='Rain or drizzle at first pitch', dir='U', group='Weather'),
        'MLB-9': dict(flag=(df.day_night == 'day') & after_night, name='Day game after a night game', dir='U', group='Schedule'),
        'MLB-10': dict(flag=getaway, name='Getaway day', dir='U', group='Schedule'),
        'MLB-11': dict(flag=df.dh.isin(['Y', 'S']), name='Doubleheader', dir='2', group='Schedule'),
        'MLB-12': dict(official='hp_umpire', name='Low-scoring plate umpire', dir='U', group='Officials'),
        'MLB-13': dict(flag=minutes.between(16 * 60, 18 * 60 + 30), universe=minutes.notna(), name='Twilight start (4–6:30 p.m. local)', dir='U', group='Schedule'),
        'MLB-14': dict(flag=df.reg & (pd.to_datetime(df.date).dt.month >= 9), universe=df.reg, name='September and later', dir='2', group='Stakes'),
    }
    return df, f


# ---------------------------------------------------------------- engine

def evaluate(sport, df, factors, periods, extra_period=None):
    rows = {}
    for fid, spec in factors.items():
        sign = {'U': -1, 'O': 1, '2': 0}[spec['dir']]
        res = dict(id=fid, sport=sport, name=spec['name'], dir=spec['dir'], group=spec['group'])
        if 'official' in spec:
            res.update(evaluate_official(df, spec, periods))
        else:
            universe = spec.get('universe', pd.Series(True, index=df.index)).fillna(False).astype(bool)
            flag = spec['flag'].fillna(False).astype(bool)
            for label, (lo, hi) in list(periods.items()) + ([('fresh', extra_period)] if extra_period else []):
                m = universe & df.season.between(lo, hi)
                res[label] = compare(df[m], flag[m])
                if 'under_odds' in df:
                    res[label]['price_roi'] = {side: price_roi(df[m & flag], side) for side in ('under', 'over')}
            m = universe & df.season.between(periods['discovery'][0], periods['replication'][1])
            res['full'] = compare(df[m], flag[m])
        decide(res, sign)
        rows[fid] = res
    return rows


def evaluate_official(df, spec, periods):
    col = spec['official']
    (d0, d1), (r0, r1) = periods['discovery'], periods['replication']
    disc = df[df.season.between(d0, d1)]
    names = disc[col].explode() if disc[col].map(lambda v: isinstance(v, list)).any() else disc[col]
    per = pd.DataFrame({'name': names.values, 'residual': disc.loc[names.index, 'residual'].values}).dropna()
    agg = per.groupby('name').residual.agg(['mean', 'size'])
    agg = agg[agg['size'] >= OFFICIAL_MIN_GAMES]
    low = set(agg[agg['mean'] <= agg['mean'].quantile(1 / 3)].index)
    rated = set(agg.index)

    def has(values, pool):
        values = values if isinstance(values, list) else [values]
        return any(v in pool for v in values)

    out = dict(officials_rated=len(rated), officials_low=len(low))
    for label, (lo, hi) in periods.items():
        m = df.season.between(lo, hi) & df[col].map(lambda v: has(v, rated))
        sub = df[m]
        out[label] = compare(sub, sub[col].map(lambda v: has(v, low)))
        if label == 'discovery':
            out[label]['in_sample'] = True
    out['full'] = dict(out['replication'])
    # Stability: officials' mean residual in discovery vs replication.
    rep = df[df.season.between(r0, r1)]
    rnames = rep[col].explode() if rep[col].map(lambda v: isinstance(v, list)).any() else rep[col]
    rper = pd.DataFrame({'name': rnames.values, 'residual': rep.loc[rnames.index, 'residual'].values,
                         'actual': rep.loc[rnames.index, 'actual'].values}).dropna()
    ragg = rper.groupby('name').residual.agg(['mean', 'size'])
    both = agg.join(ragg[ragg['size'] >= OFFICIAL_MIN_GAMES], rsuffix='_rep', how='inner')
    if len(both) >= 5:
        rho = stats.spearmanr(both['mean'], both['mean_rep'])
        out['stability'] = dict(officials=int(len(both)), spearman=float(rho.statistic), p=float(rho.pvalue))
    return out


def decide(res, sign):
    d, r = res.get('discovery', {}), res.get('replication', {})
    enough = d.get('n', 0) >= MIN_GAMES and r.get('n', 0) >= MIN_GAMES
    if 'excess' not in d or 'excess' not in r:
        res.update(replicates=False, bettable=False, verdict='Too few games', descriptive=True)
        return
    if sign == 0:
        sign_use = int(np.sign(d['excess'])) or 1
        same = np.sign(r['excess']) == sign_use
        p_rep = r['p_two']
    else:
        sign_use = sign
        same = np.sign(d['excess']) == sign and np.sign(r['excess']) == sign
        p_rep = one_sided(r['p_two'], r['t'], sign)
    if res.get('group') == 'Officials':          # discovery is the selection period by construction
        same = np.sign(r['excess']) == sign_use
    replicates = bool(enough and same and p_rep is not None and p_rep < .05)
    win = r.get('under_rate') if sign_use < 0 else r.get('over_rate')
    bettable = bool(replicates and win is not None and win >= BREAK_EVEN)
    res['replication_price_roi'] = (r.get('price_roi') or {}).get('under' if sign_use < 0 else 'over')
    res.update(tested_side='under' if sign_use < 0 else 'over', p_replication=p_rep, replicates=replicates,
               bettable=bettable, replication_win_rate=win, replication_roi=roi(win), descriptive=not enough)
    res['verdict'] = ('Bettable at -110' if bettable else 'Replicates, but not enough to beat the vig' if replicates
                      else 'Too few games' if not enough else 'Does not replicate')


def by_season(df, spec):
    """Season-by-season record for a factor (post-hoc stability view, not part of the pass rule)."""
    universe = spec.get('universe', pd.Series(True, index=df.index)).fillna(False).astype(bool)
    flag = spec['flag'].fillna(False).astype(bool)
    out = {}
    for season, idx in df[universe].groupby('season').groups.items():
        sub = df.loc[idx]
        c = compare(sub, flag.loc[idx])
        if c.get('n', 0) >= 10 and 'excess' in c:
            out[int(season)] = {k: c[k] for k in ('n', 'under_rate', 'excess', 'residual')}
    return out


def thresholds(df, periods, value, universe, cuts, side):
    """Post-hoc sensitivity of a weather factor to its cut-off."""
    out = []
    for cut in cuts:
        row = dict(cut=cut)
        for label, (lo, hi) in periods.items():
            m = universe & df.season.between(lo, hi)
            flag = (value >= cut) & side
            row[label] = {k: v for k, v in compare(df[m], flag[m]).items() if k in ('n', 'under_rate', 'excess', 'p_two')}
        out.append(row)
    return out


def sensitivity(sport, df, periods):
    if sport == 'nfl':
        outdoor = df.roof.isin(['outdoors', 'open']) & df.wind.notna()
        return dict(wind=thresholds(df, periods, df.wind, outdoor, [10, 12, 14, 15, 16, 18, 20], pd.Series(True, index=df.index)))
    if sport == 'mlb':
        known = df.outdoor & df.wind_mph.notna()
        wdir = df.wind_dir.fillna('')
        return dict(wind_in=thresholds(df, periods, df.wind_mph, known, [5, 8, 10, 12, 15, 20], wdir.str.startswith('In')),
                    wind_out=thresholds(df, periods, df.wind_mph, known, [5, 8, 10, 12, 15, 20], wdir.str.startswith('Out')))
    return {}


def load(name):
    path = DATA / f'{name}.jsonl.gz'
    with gzip.open(path, 'rt') as fh:
        rows = [json.loads(line) for line in fh]
    return pd.DataFrame(rows), hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results, hashes, meta = {}, {}, {}
    for sport, builder, periods, fresh in SPORTS:
        path = DATA / f'{sport}.jsonl.gz'
        if not path.exists():
            print('missing', path, file=sys.stderr)
            continue
        df, digest = load(sport)
        hashes[f'{sport}.jsonl.gz'] = digest
        df = df[df.actual.notna() & df.line.notna()].copy()
        df['residual'] = df.actual - df.line
        df, factors = builder(df)
        rows = evaluate(sport.upper(), df, factors, periods, fresh)
        for fid, res in rows.items():
            if 'flag' in factors[fid]:
                res['seasons'] = by_season(df, factors[fid])
        results.update(rows)
        span = df.season.between(periods['discovery'][0], periods['replication'][1])
        base = df[span]
        meta[sport.upper()] = dict(
            games=int(span.sum()), periods=periods, fresh=fresh, seasons=sorted(int(x) for x in base.season.unique()),
            mean_actual=float(base.actual.mean()), mean_line=float(base.line.mean()), mean_residual=float(base.residual.mean()),
            residual_sd=float(base.residual.std()), mae=float(base.residual.abs().mean()), **side_summary(base.residual),
            line_sources={str(k): int(v) for k, v in base.get('line_source', pd.Series('nflverse', index=base.index)).fillna('none').value_counts().items()},
            sensitivity=sensitivity(sport, df, periods), notes=df.attrs)
    tests = {k: (v.get('full') or {}).get('p_two') for k, v in results.items()}
    for k, q in bh(tests).items():
        results[k]['q_full'] = q
    OUT.write_text(json.dumps(dict(report='FV-2026-04', inputs=hashes, sports=meta, factors=results),
                              indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o) + '\n')
    print('wrote', OUT, len(results), 'factors')


SPORTS = [
    ('nfl', nfl, {'discovery': (2006, 2015), 'replication': (2016, 2025)}, (2026, 2026)),
]

if __name__ == '__main__':
    main()

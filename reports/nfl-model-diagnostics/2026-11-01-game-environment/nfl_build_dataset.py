"""Join Fourth & Value NFL pregame prop projections (weeks 3-7, 2026) to actual
stats, market game lines and the site's own game-total predictions.

One row per (week, player, market): the model mean is the same across books.
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[3])
WEEKLY = f'{REPO}/reports/nfl-weekly/2026'
# nflverse stats_player release file, e.g. stats_player_week_2026.csv.gz
STATS = sys.argv[1] if len(sys.argv) > 1 else 'stats_player_week_2026.csv.gz'
OUT = str(Path(__file__).resolve().parent / 'results/nfl_dataset.csv')
WEEKS = [3, 4, 5, 6, 7]

sys.path.insert(0, f'{REPO}/scripts')
TEAM_TO_ABBREV = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'}

STAT_COL = {'pass_attempts': 'attempts', 'pass_completions': 'completions', 'pass_yds': 'passing_yards',
            'pass_tds': 'passing_tds', 'rush_attempts': 'carries', 'rush_yds': 'rushing_yards',
            'receptions': 'receptions', 'recv_yds': 'receiving_yards'}
SUFFIX = re.compile(r'\b(jr|sr|ii|iii|iv|v)\b\.?')


def key(name):
    n = SUFFIX.sub('', str(name).lower().replace('.', ' '))
    return re.sub(r'[^a-z]', '', n)


def player_home(diag):
    try:
        return json.loads(diag).get('home')
    except (TypeError, ValueError):
        return None


def scores():
    """Team points by week, weeks 1-7."""
    sched = pd.read_csv(f'{REPO}/reports/week2-2026/source/schedule_2026.csv')
    frames = [sched[sched.week <= 2]]
    for w in WEEKS:
        frames.append(pd.read_csv(f'{WEEKLY}/week-{w}/review/games.csv'))
    g = pd.concat(frames)[['week', 'home_team', 'away_team', 'home_score', 'away_score']].dropna()
    rows = [dict(week=r.week, team=r.home_team, pts=r.home_score) for r in g.itertuples()]
    rows += [dict(week=r.week, team=r.away_team, pts=r.away_score) for r in g.itertuples()]
    return pd.DataFrame(rows)


def main():
    stats = pd.read_csv(STATS, low_memory=False)
    stats = stats[stats.season_type == 'REG']
    stats['pkey'] = stats.player_display_name.map(key)
    pts = scores()

    out = []
    for w in WEEKS:
        props = pd.read_csv(f'{WEEKLY}/week-{w}/pregame/props.csv', low_memory=False)
        props = props[props.market_std.isin(STAT_COL) & props.mu.notna()].copy()
        props['home_abbr'] = props.home_team.map(TEAM_TO_ABBREV)
        props['away_abbr'] = props.away_team.map(TEAM_TO_ABBREV)
        if 'player_position' not in props.columns:
            props['player_position'] = np.nan
        # consensus line for the main O/U (median across books, both sides share it)
        grp = props.groupby(['player', 'market_std'])
        one = grp.agg(mu=('mu', 'first'), mu_n=('mu', 'nunique'), sigma=('sigma', 'first'), lam=('lam', 'first'),
                      consensus_line=('consensus_line', 'median'), median_point=('point', 'median'),
                      home_abbr=('home_abbr', 'first'), away_abbr=('away_abbr', 'first'),
                      position=('player_position', 'first'), books=('bookmaker', 'nunique'),
                      commence=('commence_time', 'first')).reset_index()
        one['week'] = w
        # Market view at the consensus line: median de-vigged over probability and median decimal prices.
        sub = props[(props.point == props.consensus_line)].copy()
        sub['dec'] = np.where(sub.price > 0, 1 + sub.price / 100, 1 + 100 / sub.price.abs())
        side = sub.name.astype(str).str.lower()
        ov = sub[side == 'over'].groupby(['player', 'market_std']).agg(
            mkt_p_over=('prob_devig', 'median'), over_dec=('dec', 'median'))
        un = sub[side == 'under'].groupby(['player', 'market_std']).agg(under_dec=('dec', 'median'))
        one = one.merge(ov, on=['player', 'market_std'], how='left').merge(un, on=['player', 'market_std'], how='left')
        # player's team = whichever team in this game he recorded stats for this week
        s = stats[stats.week == w]
        s = s[['pkey', 'team'] + list(set(STAT_COL.values()))]
        one['pkey'] = one.player.map(key)
        one = one.merge(s, on='pkey', how='left')
        one['matched'] = (one.team == one.home_abbr) | (one.team == one.away_abbr)
        one = one[one.matched | one.team.isna()].drop_duplicates(['player', 'market_std'])
        one['matched'] = one.matched.fillna(False).astype(bool)
        one['home'] = one.team == one.home_abbr
        one['opp'] = np.where(one.home, one.away_abbr, one.home_abbr)
        one['actual'] = [getattr(r, STAT_COL[r.market_std]) if r.matched else np.nan
                         for r in one.itertuples()]
        one = one.drop(columns=list(set(STAT_COL.values())))

        # The consensus file also carries next week's early lines, so key by the (home, away) pair.
        cons = pd.read_csv(f'{WEEKLY}/week-{w}/pregame/totals_spreads_consensus.csv')
        g = cons.pivot_table(index=['home_team', 'away_team'], columns='market', values='consensus_line',
                             aggfunc='first').reset_index().rename(
            columns={'home_team': 'home_abbr', 'away_team': 'away_abbr', 'total': 'T', 'spread': 'S'})
        pred = pd.read_csv(f'{WEEKLY}/week-{w}/pregame/week_predictions.csv').rename(
            columns={'home_team': 'home_abbr', 'away_team': 'away_abbr'})[['home_abbr', 'away_abbr', 'home_pred', 'away_pred']]
        one = one.merge(g, on=['home_abbr', 'away_abbr'], how='left').merge(pred, on=['home_abbr', 'away_abbr'], how='left')
        h = one.home
        one['mkt_total'] = one['T']
        one['mkt_spread'] = np.where(h, one.S, -one.S)            # team's own line; + = underdog
        one['mkt_team_total'] = (one['T'] - one['mkt_spread']) / 2
        one['opp_mkt_team_total'] = (one['T'] + one['mkt_spread']) / 2
        one['our_team_total'] = np.where(h, one.home_pred, one.away_pred)
        one['our_opp_total'] = np.where(h, one.away_pred, one.home_pred)
        one = one.drop(columns=['T', 'S', 'home_pred', 'away_pred'])

        recent = (pts[pts.week < w].sort_values('week').groupby('team').tail(4)
                  .groupby('team').pts.mean().rename('team_recent_pts'))
        one = one.merge(recent, left_on='team', right_index=True, how='left')
        out.append(one)

    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT, index=False)
    print('rows', len(df), 'matched', int(df.matched.sum()), f'({df.matched.mean():.1%})')
    print('mu varies across books for', int((df.mu_n > 1).sum()), 'player-markets')
    print('missing market lines', int(df.mkt_total.isna().sum()), '| missing our totals', int(df.our_team_total.isna().sum()))
    print(df[df.matched].groupby('market_std').size())
    print(df[~df.matched].groupby('position').size().sort_values(ascending=False).head())


if __name__ == '__main__':
    main()

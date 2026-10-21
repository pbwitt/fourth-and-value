"""Grade the published NFL model mean (mu) against what happened.

For each event, player and market, the decision snapshot is the last archived props board
committed before kickoff (extract_boards.py). Actuals come from nflverse box scores; a player
with offensive snaps but no box-score line counts as zero, and a player with neither is treated
as not playing (books void those props). The mean is compared with the outcome, with the
market's implied mean (backed out of the main line and its no-vig price) and with the simple
recency-weighted average from shape_backtest.py. Intervals resample whole games.

usage: production_mean_test.py BOARDS_CSV NFLVERSE_DIR [--max-week N]
"""
import argparse
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import gamma, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape_backtest as sb  # noqa: E402

STAT = {'recv_yds': 'receiving_yards', 'rush_yds': 'rushing_yards', 'pass_yds': 'passing_yards',
        'receptions': 'receptions', 'rush_attempts': 'carries', 'pass_attempts': 'attempts',
        'pass_completions': 'completions'}
SKEWED = {'recv_yds', 'rush_yds'}  # shape_backtest.py found these right-skewed; the rest stay Normal
HALF_LIVES = [0.76, 2, 4, 6, 10]   # 0.76 games is the production L4 decay (alpha 0.4)
TEAM = {'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
        'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
        'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
        'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
        'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'Seattle Seahawks': 'SEA',
        'San Francisco 49ers': 'SF', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN',
        'Washington Commanders': 'WAS'}


def key(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


def decision_rows(path):
    b = pd.read_csv(path, low_memory=False)
    b = b[b.market_std.isin(STAT) & b.mu.notna()].copy()
    b['committed'] = pd.to_datetime(b.committed_at, utc=True)
    b['kickoff'] = pd.to_datetime(b.commence_time, utc=True)
    b = b[b.committed < b.kickoff]
    ident = ['game_id', 'player', 'market_std']
    b = b[b.committed == b.groupby(ident).committed.transform('max')]
    out = []
    for (event, player, market), g in b.groupby(ident):
        overs = g[(g.name == 'over') & g.consensus_prob.notna() & g.point.notna()]
        main = None
        if len(overs):
            by_line = overs.groupby('point').agg(books=('book_count', 'max'), p=('consensus_prob', 'median'),
                                                 model_p=('model_prob', 'median')).reset_index()
            by_line['even'] = (by_line.p - .5).abs()
            main = by_line.sort_values(['books', 'even', 'point'], ascending=[False, True, True]).iloc[0]
        first = g.iloc[0]
        out.append(dict(event=event, player=player, market=market, home=first.home_team, away=first.away_team,
                        kickoff=first.kickoff, committed=first.committed, mu=g.mu.median(), sigma=g.sigma.median(),
                        line=main.point if main is not None else g.consensus_line.median(),
                        over_fair=main.p if main is not None else np.nan,
                        model_over=main.model_p if main is not None else np.nan,
                        **{c: g[c].median() for c in g if c.startswith('stage_')}))
    return pd.DataFrame(out)


def attach_actuals(d, src, max_week):
    s = pd.read_csv(src / 'stats_player_week_2026.csv.gz', low_memory=False,
                    usecols=['game_id', 'week', 'player_display_name', *sorted(set(STAT.values()))])
    snaps = pd.read_csv(src / 'snap_counts_2026.csv')
    if max_week:
        s, snaps = s[s.week <= max_week], snaps[snaps.week <= max_week]
    games = {}
    for gid in s.game_id.unique():
        _, week, away, home = gid.split('_')
        games[(away, home)] = (gid, int(week))
    found = [games.get((TEAM[a], TEAM[h]), (None, None)) for a, h in zip(d.away, d.home)]
    d['nfl_game'], d['week'] = [f[0] for f in found], [f[1] for f in found]
    d = d[d.nfl_game.notna()].copy()
    d['k'] = d.player.map(key)
    s = s.assign(k=s.player_display_name.map(key))
    snaps = snaps.assign(k=snaps.player.map(key))
    played = set(zip(snaps.loc[snaps.offense_snaps > 0, 'game_id'], snaps.loc[snaps.offense_snaps > 0, 'k']))
    stats = s.set_index(['game_id', 'k'])
    stats = stats[~stats.index.duplicated()]
    actual, how = [], []
    for g, k, m in zip(d.nfl_game, d.k, d.market):
        if (g, k) in stats.index:
            actual.append(float(stats.at[(g, k), STAT[m]] or 0)); how.append('box score')
        elif (g, k) in played:
            actual.append(0.0); how.append('snaps, no line')
        else:
            actual.append(np.nan); how.append('did not play')
    d['actual'], d['source'] = actual, how
    return d


def simple_means(d, src, half_life):
    """The shape backtest's recency-weighted average, from games before each one."""
    sb.SRC, sb.HALF_LIFE = src, half_life
    data = sb.load()
    out = []
    for m in ['recv_yds', 'rush_yds', 'pass_yds', 'receptions']:
        col, pos, _ = sb.MARKETS[m]
        f = sb.forecasts(data, col, pos, 0)
        f = f[f.season == 2026].assign(k=lambda x: x.player.map(key), market=m)
        out.append(f[['game_id', 'k', 'market', 'mean']].rename(columns={'game_id': 'nfl_game', 'mean': 'simple'}))
    return d.merge(pd.concat(out), on=['nfl_game', 'k', 'market'], how='left')


def implied_mean(line, p_over, sd_ratio, skewed):
    """Mean whose distribution (sd = sd_ratio * mean) puts p_over above the line."""
    if not (0 < p_over < 1) or not math.isfinite(line):
        return np.nan

    def gap(m):
        sd = sd_ratio * m
        sf = gamma.sf(line, (m / sd) ** 2, scale=sd * sd / m) if skewed else norm.sf(line, m, sd)
        return sf - p_over
    try:
        return brentq(gap, max(line * 0.2, 0.05), line * 4 + 10)
    except ValueError:
        return np.nan


def boot(values, groups, n=2000, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'v': values, 'g': groups}).groupby('g').v.agg(['sum', 'size'])
    s, c = df['sum'].to_numpy(), df['size'].to_numpy()
    idx = rng.integers(0, len(df), (n, len(df)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def compare(g, other):
    """RMSE of mu and of another mean on the same rows; mean MSE difference and its interval."""
    s = g.dropna(subset=['mu', other])
    a, b = (s.actual - s.mu) ** 2, (s.actual - s[other]) ** 2
    lo, hi = boot((b - a).to_numpy(), s.nfl_game.to_numpy())
    return len(s), math.sqrt(a.mean()), math.sqrt(b.mean()), (b - a).mean(), lo, hi


def fmt(c, other):
    n, mu, alt, diff, lo, hi = c
    return f'n={n:4d}  mu {mu:6.2f} vs {other} {alt:6.2f}  MSE diff {diff:+9.2f} [{lo:+.2f}, {hi:+.2f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('boards', type=Path)
    ap.add_argument('nflverse', type=Path)
    ap.add_argument('--max-week', type=int, default=None, help='last 2026 week to grade')
    a = ap.parse_args()
    d = attach_actuals(decision_rows(a.boards), a.nflverse, a.max_week)
    print('rows by actual source:', d.source.value_counts().to_dict())
    d = d[d.actual.notna()].copy()
    d = simple_means(d, a.nflverse, sb.HALF_LIFE)
    print(f'graded player-markets: {len(d)} across {d.nfl_game.nunique()} games, weeks {int(d.week.min())}-'
          f'{int(d.week.max())}; snapshot lead time median {(d.kickoff - d.committed).median()}')
    pd.set_option('display.width', 250)

    rows = []
    for m, g in d.groupby('market'):
        err = g.actual - g.mu
        lo, hi = boot(err.to_numpy(), g.nfl_game.to_numpy())
        o = g[g.over_fair.notna() & (g.line % 1 != 0)]
        below = o[o.mu < o.line]
        rows.append(dict(market=m, n=len(g), games=g.nfl_game.nunique(), mean_mu=g.mu.mean(),
                         mean_actual=g.actual.mean(), bias=err.mean(), bias_lo=lo, bias_hi=hi, median_err=err.median(),
                         share_mu_below_line=(g.mu < g.line).mean(), mu_minus_line=(g.mu - g.line).median(),
                         over_rate=(o.actual > o.line).mean(), fair_over=o.over_fair.mean(),
                         under_hit_when_mu_below=(below.actual < below.line).mean(), n_mu_below=len(below)))
    r = pd.DataFrame(rows).set_index('market')
    print('\nLevel: bias = actual - mu (95% interval resampling games)')
    print(r[['n', 'games', 'mean_mu', 'mean_actual', 'bias', 'bias_lo', 'bias_hi', 'median_err']].round(2).to_string())
    print('\nAgainst the main market line (most books, then most even price)')
    print(r[['share_mu_below_line', 'mu_minus_line', 'over_rate', 'fair_over', 'n_mu_below',
             'under_hit_when_mu_below']].round(3).to_string())

    print('\nAccuracy as a mean forecast, same rows (RMSE; negative MSE diff = the other mean is better)')
    for m, g in d.groupby('market'):
        g = g.copy()
        ratio = (g.sigma / g.mu).median()
        g['market_mean'] = [implied_mean(L, p, ratio, m in SKEWED) for L, p in zip(g.line, g.over_fair)]
        print(f'{m:16s} market-implied mean  {fmt(compare(g, "market_mean"), "market")}')
        if g.simple.notna().any():
            print(f'{m:16s} simple average       {fmt(compare(g, "simple"), "simple")}')

    print('\nAdjustment stages against the published mean (boards that record them)')
    for m, g in d.groupby('market'):
        for stage in ['stage_before_adjustments', 'stage_after_defense']:
            if g[stage].notna().any():
                print(f'{m:16s} {stage[6:]:19s}  {fmt(compare(g, stage), "stage")}  games={g.dropna(subset=[stage]).nfl_game.nunique()}')

    print('\nSimple average by recency half-life against mu (same rows per market)')
    for hl in HALF_LIVES:
        alt = simple_means(d.drop(columns='simple'), a.nflverse, hl)
        parts = []
        for m in ['recv_yds', 'receptions', 'rush_yds', 'pass_yds']:
            _, mu, simple, _, lo, hi = compare(alt[alt.market == m], 'simple')
            parts.append(f'{m} {simple:.2f} vs mu {mu:.2f} [{lo:+.1f}, {hi:+.1f}]')
        print(f'half-life {hl:>4} games: ' + ' | '.join(parts))

    print('\nBias by week (actual - mu)')
    print(d.assign(err=d.actual - d.mu).groupby(['market', 'week']).err.mean().round(1).unstack('week').to_string())


if __name__ == '__main__':
    main()

"""Out-of-sample test of prop distribution shape: Normal vs skewed, same mean and spread.

Every forecast uses only games played before it. Variance curves are fitted on 2022-2023
forecasts and frozen; scoring is on 2024-2025 (grid of lines) and on the real 2026 lines
from our graded tickets (weeks 1-5). Both curves get the same mean and spread, so only the
shape differs. The forecasts are a simple recency-weighted average, not the production mean.

usage: shape_backtest.py NFLVERSE_DIR REPO_ROOT [--half-life GAMES] [--prior-k GAMES]
"""
import argparse
import glob
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, lognorm, nbinom, norm, poisson

SRC = REPO = None  # set by main()
MARKETS = {  # stat column, positions, minimum forecast mean for a book to post it
    'recv_yds': ('receiving_yards', {'WR', 'TE', 'RB'}, 20),
    'rush_yds': ('rushing_yards', {'RB', 'QB'}, 20),
    'pass_yds': ('passing_yards', {'QB'}, 150),
    'receptions': ('receptions', {'WR', 'TE', 'RB'}, 2),
}
HALF_LIFE, PRIOR_K, SD_K, MIN_GAMES = 6, 2, 4, 4
TRAIN, TEST = {2022, 2023}, {2024, 2025}
MULTS = [0.6, 0.8, 1.0, 1.2, 1.4]


def load():
    frames = [pd.read_csv(p, low_memory=False) for p in sorted(SRC.glob('stats_player_week_*.csv.gz'))]
    d = pd.concat(frames)
    d = d[(d.season_type == 'REG') & d.player_id.notna()].copy()
    return d.sort_values(['player_id', 'season', 'week'])


def forecasts(d, col, positions, floor):
    d = d[d.position.isin(positions)][['player_id', 'player_display_name', 'position', 'team', 'season', 'week',
                                       'game_id', col]].dropna(subset=[col])
    prior = d[d.season.isin({2021, 2022})].groupby('position')[col].mean().to_dict()
    out = []
    for pid, g in d.groupby('player_id', sort=False):
        ys, seasons = g[col].to_numpy(float), g.season.to_numpy()
        for i in range(MIN_GAMES, len(g)):
            keep = seasons[:i] >= seasons[i] - 2
            hist = ys[:i][keep]
            if len(hist) < MIN_GAMES:
                continue
            w = 0.5 ** (np.arange(len(hist))[::-1] / HALF_LIFE)
            p = prior.get(g.position.iat[i], hist.mean())
            m = (w @ hist + PRIOR_K * p) / (w.sum() + PRIOR_K)
            if m < floor:
                continue
            own_var = w @ (hist - m) ** 2 / w.sum()
            out.append((pid, g.player_display_name.iat[i], g.team.iat[i], g.season.iat[i], g.week.iat[i],
                        g.game_id.iat[i], m, own_var, w.sum(), ys[i]))
    return pd.DataFrame(out, columns=['player_id', 'player', 'team', 'season', 'week', 'game_id', 'mean', 'own_var',
                                      'wsum', 'y'])


def fit_variance(f):
    """var = a*m + b*m^2 by least squares on squared forecast errors (training seasons only)."""
    t = f[f.season.isin(TRAIN)]
    X = np.column_stack([t['mean'], t['mean'] ** 2])
    coef, *_ = np.linalg.lstsq(X, (t.y - t['mean']) ** 2, rcond=None)
    return coef


def p_over(shape, m, v, line, discrete):
    m, v, line = (np.asarray(x, float) for x in (m, v, line))
    sd = np.sqrt(v)
    if shape == 'normal':
        return norm.sf(line, m, sd)
    if discrete:  # half lines: P(X >= ceil(line)); negative binomial when overdispersed, else Poisson
        k = np.floor(line)
        over = v > m * 1.0001
        r = np.where(over, m * m / np.maximum(v - m, 1e-9), 1.0)
        return np.where(over, nbinom.sf(k, r, r / (r + m)), poisson.sf(k, m))
    if shape == 'gamma':
        return gamma.sf(line, (m / sd) ** 2, scale=v / m)
    s2 = np.log1p(v / m ** 2)
    return lognorm.sf(line, np.sqrt(s2), scale=np.exp(np.log(m) - s2 / 2))


def shapes(discrete):
    return ['normal', 'negbin'] if discrete else ['normal', 'gamma', 'lognormal']


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(a, b, games, n=1000, seed=7):
    """95% interval of mean(a-b), resampling whole games."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'d': a - b, 'g': games}).groupby('g')['d'].agg(['sum', 'size'])
    s, c = df['sum'].to_numpy(), df['size'].to_numpy()
    idx = rng.integers(0, len(df), (n, len(df)))
    vals = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(vals, [2.5, 97.5])


def variance_for(f, coef, kind):
    fit = np.maximum(coef[0] * f['mean'] + coef[1] * f['mean'] ** 2, 1e-6)
    if kind == 'pooled':
        return fit
    return (f.wsum * f.own_var + SD_K * fit) / (f.wsum + SD_K)  # player's own spread, shrunk


def grid_test(name, f, coef, discrete):
    t = f[f.season.isin(TEST)].reset_index(drop=True)
    rows = []
    for kind in ['pooled', 'own']:
        v = variance_for(t, coef, kind).to_numpy()
        for mult in MULTS:
            lines = np.floor(t['mean'].to_numpy() * mult) + 0.5
            y = (t.y.to_numpy() > lines).astype(float)
            ps = {s: p_over(s, t['mean'], v, lines, discrete) for s in shapes(discrete)}
            for s, p in ps.items():
                rows.append(dict(market=name, sd=kind, mult=mult, shape=s, n=len(y), over_rate=y.mean(),
                                 predicted=p.mean(), logloss=ll(p, y).mean(), brier=((p - y) ** 2).mean(),
                                 _ll=ll(p, y), _games=t.game_id.to_numpy()))
    return rows


def norm_name(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


def tickets():
    seen, out = set(), []
    for path in sorted(glob.glob(str(REPO / 'docs/blog/week-*-review-data.json'))):
        for r in json.loads(Path(path).read_text()).get('tickets', []):
            if r.get('outcome') not in ('win', 'loss') or r.get('point') is None or float(r['point']).is_integer():
                continue
            k = (r['stat_game'], r['player'], r['market_std'], float(r['point']))
            if k in seen:
                continue
            seen.add(k)
            over = r['name'] == 'over'
            won = r['outcome'] == 'win'
            out.append(dict(game_id=r['stat_game'], key=norm_name(r['player']), market=r['market_std'],
                            line=float(r['point']), y=float(won == over),
                            production=r['model_prob'] if over else 1 - r['model_prob'],
                            market_p=(r['consensus_prob'] if over else 1 - r['consensus_prob'])
                            if r.get('consensus_prob') is not None else np.nan))
    return pd.DataFrame(out)


def main():
    global SRC, REPO, HALF_LIFE, PRIOR_K
    ap = argparse.ArgumentParser()
    ap.add_argument('nflverse', type=Path, help='directory of stats_player_week_<season>.csv.gz files')
    ap.add_argument('repo', type=Path, help='repository root (for the graded ticket files)')
    ap.add_argument('--half-life', type=float, default=HALF_LIFE, help='recency half-life in games')
    ap.add_argument('--prior-k', type=float, default=PRIOR_K, help='position-prior weight in games')
    a = ap.parse_args()
    SRC, REPO, HALF_LIFE, PRIOR_K = a.nflverse, a.repo, a.half_life, a.prior_k
    print(f'half-life {HALF_LIFE} games, prior weight {PRIOR_K} games')
    d = load()
    tk = tickets()
    grid, real = [], []
    for name, (col, pos, floor) in MARKETS.items():
        discrete = name == 'receptions'
        f_all = forecasts(d, col, pos, 0)
        f = f_all[f_all['mean'] >= floor]
        coef = fit_variance(f)
        print(f'{name}: {len(f)} forecasts, variance = {coef[0]:.3f}*m + {coef[1]:.4f}*m^2', flush=True)
        grid += grid_test(name, f, coef, discrete)
        # Real 2026 lines (weeks 1-5): same forecasts, no floor, joined on game and player.
        f26 = f_all[f_all.season == 2026].assign(key=lambda x: x.player.map(norm_name))
        j = tk[tk.market == name].merge(f26[['game_id', 'key', 'mean', 'own_var', 'wsum']], on=['game_id', 'key'])
        if j.empty:
            continue
        for kind in ['pooled', 'own']:
            v = variance_for(j, coef, kind).to_numpy()
            for s in shapes(discrete):
                j[f'{s}_{kind}'] = p_over(s, j['mean'], v, j.line, discrete)
        j['market'] = name
        real.append(j)
    g = pd.DataFrame(grid)

    print('\n== 2024-2025 out of sample, lines at forecast mean x multiplier ==')
    for (m, kind), sub in g.groupby(['market', 'sd'], sort=False):
        print(f'\n{m} (spread: {kind})')
        print(sub.pivot_table(index='mult', columns='shape', values='predicted').round(3)
              .assign(actual_over=sub.groupby('mult').over_rate.first().round(3)).to_string())
        tot = sub.groupby('shape').apply(lambda x: pd.Series({'logloss': np.concatenate(x._ll.to_list()).mean(),
                                                               'brier': x.brier.mean()}), include_groups=False)
        print(tot.round(5).to_string())
        base = np.concatenate(sub[sub['shape'] == 'normal']._ll.to_list())
        games = np.concatenate(sub[sub['shape'] == 'normal']._games.to_list())
        for s in sub['shape'].unique():
            if s == 'normal':
                continue
            alt = np.concatenate(sub[sub['shape'] == s]._ll.to_list())
            lo, hi = boot_diff(alt, base, games)
            print(f'  {s} minus normal log loss: {(alt - base).mean():+.5f}  95% [{lo:+.5f}, {hi:+.5f}]')

    print('\n== Real 2026 lines from graded tickets (weeks 1-5), P(over) scored on outcome ==')
    r = pd.concat(real)
    for m, sub in r.groupby('market', sort=False):
        cols = [c for c in sub.columns if c.endswith('_pooled') or c.endswith('_own')] + ['production', 'market_p']
        sub = sub.dropna(subset=['market_p'])
        res = {c: ll(sub[c], sub.y).mean() for c in cols}
        print(f'\n{m}: {len(sub)} lines, {sub.game_id.nunique()} games, over rate {sub.y.mean():.3f}')
        print('  mean P(over): ' + ', '.join(f'{c} {sub[c].mean():.3f}' for c in cols))
        print('  log loss:     ' + ', '.join(f'{c} {v:.4f}' for c, v in res.items()))
        alt = 'negbin_own' if m == 'receptions' else 'gamma_own'
        lo, hi = boot_diff(ll(sub[alt], sub.y), ll(sub['normal_own'], sub.y), sub.game_id.to_numpy())
        print(f'  {alt} minus normal_own: {(ll(sub[alt], sub.y) - ll(sub["normal_own"], sub.y)).mean():+.4f} '
              f'95% [{lo:+.4f}, {hi:+.4f}]')
    unmatched = len(tk[tk.market.isin(MARKETS)]) - len(r)
    print(f'\nticket lines not matched to a forecast: {unmatched}')


if __name__ == '__main__':
    main()

"""Does adding the game environment (implied team total, spread) to the NFL prop
means improve them? Weeks 3-7 of 2026, leave-one-week-out.

Adjusted mean: mu' = mu * (1 + a + b . x), fit by OLS of (actual - mu) on mu*[1, x]
within each market group. The comparison that matters is level-only (a) versus
level + environment (a + b.x), so a general level bias cannot masquerade as signal.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

df = pd.read_csv(str(__import__('pathlib').Path(__file__).resolve().parent / 'results/nfl_dataset.csv'))
df = df[df.matched & df.actual.notna() & df.mu.notna() & (df.mu > 0)].copy()
GROUP = {'pass_attempts': 'pass', 'pass_completions': 'pass', 'pass_yds': 'pass', 'pass_tds': 'pass',
         'rush_attempts': 'rush', 'rush_yds': 'rush', 'receptions': 'recv', 'recv_yds': 'recv'}
df['group'] = df.market_std.map(GROUP)
df['game'] = df.week.astype(str) + '_' + df.home_abbr
df['tt_rel'] = df.mkt_team_total - df.team_recent_pts
df['mkt_minus_ours'] = df.mkt_team_total - df.our_team_total
df['our_spread'] = df.our_opp_total - df.our_team_total   # + = we project them to trail

VARIANTS = {
    'level only': [],
    'mkt team total': ['mkt_team_total'],
    'mkt team total + spread': ['mkt_team_total', 'mkt_spread'],
    'spread only': ['mkt_spread'],
    'our spread only': ['our_spread'],
    'our total + our spread': ['our_team_total', 'our_spread'],
    'our team total': ['our_team_total'],
    'our team total + mkt spread': ['our_team_total', 'mkt_spread'],
    'mkt total vs recent pts': ['tt_rel'],
}


def fit(train, feats):
    means = {f: train[f].mean() for f in feats}
    X = np.column_stack([train.mu.values] + [train.mu.values * (train[f].values - means[f]) for f in feats])
    y = (train.actual - train.mu).values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, means


def apply(test, feats, coef, means):
    mult = 1 + coef[0] + sum(coef[i + 1] * (test[f].values - means[f]) for i, f in enumerate(feats))
    return test.mu.values * np.clip(mult, 0.3, 2.0)


def p_over(mu, row_sigma, line, market):
    if market == 'pass_tds':
        return 1 - poisson.cdf(np.floor(line), np.maximum(mu, 1e-6))
    return 1 - norm.cdf(line, loc=mu, scale=row_sigma)


def cluster_t(sub, feats):
    """Full-sample slopes with game-clustered standard errors (diagnostic only)."""
    means = {f: sub[f].mean() for f in feats}
    X = np.column_stack([sub.mu.values] + [sub.mu.values * (sub[f].values - means[f]) for f in feats])
    y = (sub.actual - sub.mu).values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[idx].T @ e[idx], X[idx].T @ e[idx]) for idx in sub.groupby('game').indices.values())
    G = sub.game.nunique()
    se = np.sqrt(np.diag(bread @ meat @ bread) * G / (G - 1))
    return coef, se


def evaluate(scope):
    rows = []
    for name, feats in VARIANTS.items():
        preds = pd.Series(np.nan, index=scope.index)
        for g, gdf in scope.groupby('group'):
            for w in sorted(gdf.week.unique()):
                tr, te = gdf[gdf.week != w], gdf[gdf.week == w]
                coef, means = fit(tr, feats)
                preds.loc[te.index] = apply(te, feats, coef, means)
        d = scope.assign(mu_adj=preds)
        d['p'] = [p_over(r.mu_adj, r.sigma, r.consensus_line, r.market_std) for r in d.itertuples()]
        d = d[d.actual != d.consensus_line]
        y = (d.actual > d.consensus_line).astype(float)
        p = d.p.clip(0.01, 0.99)
        # flat-bet sim vs market no-vig at median price, 3-point edge threshold
        m = d.dropna(subset=['mkt_p_over', 'over_dec', 'under_dec'])
        ym = (m.actual > m.consensus_line)
        e_over, e_under = m.p - m.mkt_p_over, (1 - m.p) - (1 - m.mkt_p_over)
        bet_over, bet_under = e_over >= 0.03, e_under >= 0.03
        units = np.where(bet_over, np.where(ym, m.over_dec - 1, -1), 0) + \
            np.where(bet_under, np.where(~ym, m.under_dec - 1, -1), 0)
        n_bets = int(bet_over.sum() + bet_under.sum())
        wins = int((bet_over & ym).sum() + (bet_under & ~ym).sum())
        rows.append(dict(variant=name, n=len(d),
                         mae=np.mean(np.abs(d.actual - d.mu_adj)), rmse=np.sqrt(np.mean((d.actual - d.mu_adj) ** 2)),
                         logloss=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                         brier=np.mean((p - y) ** 2), side_hit=np.mean((p > 0.5) == (y == 1)),
                         bets=n_bets, bet_win=wins / n_bets if n_bets else np.nan,
                         roi=units.sum() / n_bets if n_bets else np.nan))
    raw = scope[scope.actual != scope.consensus_line]
    yr = (raw.actual > raw.consensus_line).astype(float)
    pr = pd.Series([p_over(r.mu, r.sigma, r.consensus_line, r.market_std) for r in raw.itertuples()], index=raw.index).clip(.01, .99)
    mk = raw.dropna(subset=['mkt_p_over'])
    ymk = (mk.actual > mk.consensus_line).astype(float)
    pm = mk.mkt_p_over.clip(.01, .99)
    base = dict(variant='(current model, no adj)', n=len(raw), mae=np.mean(np.abs(raw.actual - raw.mu)),
                rmse=np.sqrt(np.mean((raw.actual - raw.mu) ** 2)),
                logloss=-np.mean(yr * np.log(pr) + (1 - yr) * np.log(1 - pr)), brier=np.mean((pr - yr) ** 2),
                side_hit=np.mean((pr > .5) == (yr == 1)))
    mkt = dict(variant='(market no-vig, reference)', n=len(mk),
               logloss=-np.mean(ymk * np.log(pm) + (1 - ymk) * np.log(1 - pm)), brier=np.mean((pm - ymk) ** 2))
    return pd.DataFrame([base] + rows + [mkt])


pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')
print(f'Sample: {len(df)} player-markets, {df.game.nunique()} games, weeks {sorted(df.week.unique())}\n')
print(df.groupby('market_std').size().to_string(), '\n')

print('=== Diagnostic: full-sample slope, % change in stat per +1 pt (game-clustered t) ===')
diag = []
for mkt, sub in df.groupby('market_std'):
    for feats in (['mkt_team_total'], ['mkt_spread'], ['our_team_total'], ['mkt_minus_ours'], ['mkt_team_total', 'mkt_spread'], ['our_spread'], ['mkt_spread', 'our_spread']):
        coef, se = cluster_t(sub, feats)
        for i, f in enumerate(feats):
            diag.append(dict(market=mkt, model='+'.join(feats), feature=f, pct_per_pt=100 * coef[i + 1],
                             t=coef[i + 1] / se[i + 1], level_bias_pct=100 * coef[0], n=len(sub)))
diag = pd.DataFrame(diag)
print(diag[diag.model.isin(['mkt_team_total', 'mkt_spread', 'our_team_total', 'mkt_minus_ours'])]
      .pivot_table(index='market', columns='feature', values=['pct_per_pt', 't']).round(2).to_string())
print('\nOur spread alone / market spread + our spread jointly:')
print(diag[diag.model.isin(['our_spread','mkt_spread+our_spread'])].pivot_table(index='market', columns=['model','feature'], values=['pct_per_pt','t']).round(2).to_string())
print('\nJoint (team total + spread):')
print(diag[diag.model == 'mkt_team_total+mkt_spread'].pivot_table(index='market', columns='feature',
      values=['pct_per_pt', 't']).round(2).to_string())

for label, scope in [('ALL MARKETS', df)] + [(g.upper(), df[df.group == g]) for g in ['pass', 'rush', 'recv']] + \
        [('VOLUME ONLY (attempts, completions, carries, receptions)',
          df[df.market_std.isin(['pass_attempts', 'pass_completions', 'rush_attempts', 'receptions'])])]:
    print(f'\n=== Leave-one-week-out: {label} ===')
    print(evaluate(scope).to_string(index=False))

df.to_csv(str(__import__('pathlib').Path(__file__).resolve().parent / 'results/nfl_dataset_scored.csv'), index=False)

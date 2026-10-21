"""Shared loaders and scoring for the real-line curve and pick tests (2026 Weeks 1-6 NFL boards).

Inputs come from the two earlier reports: the archived boards (shape-and-level/extract_boards.py),
the graded decision rows (shape-and-level/production_mean_test.py --out) and the replay with the
shipped settings (pass-volume-recency/replay_means.py ... starts_recent17_hl6).
"""
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom, norm, poisson

MARKETS = ['receptions', 'recv_yds', 'rush_yds', 'rush_attempts', 'pass_yds', 'pass_attempts', 'pass_completions']
COUNT = {'receptions', 'rush_attempts', 'pass_attempts', 'pass_completions'}


def key(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


def logit(p):
    return math.log(p / (1 - p))


def decimal(american):
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def game_interval(values, games, n=2000, seed=3):
    """95% interval of a mean, resampling whole games."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'v': values, 'g': games}).groupby('g').v.agg(['sum', 'size'])
    s, c = df['sum'].to_numpy(), df['size'].to_numpy()
    idx = rng.integers(0, len(df), (n, len(df)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def p_over(shape, mu, var, line, count):
    """P(X > line) at a half-point line. Skewed: negative binomial for counts (Poisson if not
    overdispersed), gamma for yards; both keep the given mean and variance."""
    mu, var = np.maximum(np.asarray(mu, float), 1e-6), np.maximum(np.asarray(var, float), 1e-6)
    if shape == 'normal':
        return norm.sf(line, mu, np.sqrt(var))
    if count:
        k, over = np.floor(line), var > mu * 1.0001
        r = np.where(over, mu * mu / np.maximum(var - mu, 1e-9), 1.0)
        return np.where(over, nbinom.sf(k, r, r / (r + mu)), poisson.sf(k, mu))
    return gamma.sf(line, mu * mu / var, scale=var / mu)


def predictive_variance(replay):
    """var = a*m + b*m^2 per market, fitted on 2024-25 forecast errors where a prop is plausible."""
    t = replay[replay.season.isin([2024, 2025]) & (replay.mu > 0)]
    floor = np.where(t.market == 'receptions', 2, np.where(t.market == 'recv_yds', 15,
                     np.where(t.market == 'rush_attempts', 6, 25)))
    t = t[np.where(t.market.str.startswith('pass'), t.qb_start, t.mu > floor)]
    return {m: np.linalg.lstsq(np.column_stack([g.mu, g.mu ** 2]), (g.actual - g.mu) ** 2, rcond=None)[0]
            for m, g in t.groupby('market')}


def decision_boards(boards_csv):
    """Board rows from the last snapshot committed before each player-market's kickoff."""
    b = pd.read_csv(boards_csv, low_memory=False)
    b = b[b.market_std.isin(MARKETS) & b.point.notna() & (b.point % 1 != 0)].copy()
    b['committed'] = pd.to_datetime(b.committed_at, utc=True)
    b = b[b.committed < pd.to_datetime(b.commence_time, utc=True)]
    return b[b.committed == b.groupby(['game_id', 'player', 'market_std']).committed.transform('max')]


def offers(boards_csv, graded_csv):
    """Best price per game, player, market, side and line, graded, with the blend inputs."""
    b = decision_boards(boards_csv)
    b = b[b.model_prob.notna() & b.consensus_prob.notna() & b.price.notna() & b.name.isin(['over', 'under'])].copy()
    b['dec'] = b.price.map(decimal)
    b = b.sort_values('dec').groupby(['game_id', 'player', 'market_std', 'name', 'point']).tail(1)
    g = pd.read_csv(graded_csv)[['event', 'player', 'market', 'nfl_game', 'actual']]
    b = b.merge(g.rename(columns={'event': 'game_id', 'market': 'market_std'}), on=['game_id', 'player', 'market_std'])
    b['y'] = np.where(b.name == 'over', b.actual > b.point, b.actual < b.point).astype(float)
    b['profit'] = np.where(b.y == 1, b.dec - 1, -1.0)
    b['model'], b['q'] = b.model_prob.clip(1e-4, 1 - 1e-4), b.consensus_prob.clip(1e-4, 1 - 1e-4)
    b['gap'] = [logit(x) - logit(y) for x, y in zip(b.model, b.q)]
    return b

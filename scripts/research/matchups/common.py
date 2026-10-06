"""Shared pieces for the matchup backtests (research only; nothing here feeds a live forecast).

Every adjustment is an empirically shrunk ratio of what a player actually produced to what the
baseline expected, over games strictly before the one being forecast:

    ratio = (actual + k * prior) / (expected + k)

k is in the stat's own units (expected shots, yards, hits ...). A large k means the history is
ignored; a small k trusts it. The prior is 1 for a player's overall ratio and the player's overall
ratio for a split (home/away, one opponent), so a split only moves the forecast by how much the
player differs from himself, not from the league.
"""
from collections import defaultdict
import math

import numpy as np

INF = float('inf')


class Ledger:
    """Running actual/expected sums per key, read before the day's games are added."""

    def __init__(self):
        self.a = defaultdict(float)
        self.e = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, key, actual, expected):
        self.a[key] += actual
        self.e[key] += expected
        self.n[key] += 1

    def get(self, key):
        return self.a.get(key, 0.), self.e.get(key, 0.), self.n.get(key, 0)


def shrunk(actual, expected, k, prior=1.):
    if k == INF or expected + k <= 0:
        return prior
    return (actual + k * prior) / (expected + k)


def cluster_ci(diff, groups, draws=1000, seed=7):
    """Mean of a per-row difference and a 95% interval from resampling whole groups (games/dates)."""
    diff = np.asarray(diff, float)
    keys, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inverse, diff, len(keys))
    counts = np.bincount(inverse, None, len(keys))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), (draws, len(keys)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return dict(mean=float(diff.mean()), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)), n=int(len(diff)))


def brier(p, y):
    p = np.asarray(p, float)
    return (p - np.asarray(y, float)) ** 2


def fmt_k(k):
    return 'none' if k == INF else (str(int(k)) if float(k).is_integer() else str(k))


def weight_after(expected, k):
    """Share of a split's own evidence in the forecast after `expected` units of history."""
    return 0. if k == INF else expected / (expected + k)


def finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)

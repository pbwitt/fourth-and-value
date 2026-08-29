#!/usr/bin/env python3
"""
career_baseline.py

Hierarchical (position pool -> player career -> current-season recent
form) baseline estimation, replacing "estimate everything from this
player's last 4 games in isolation."

Why: the backtest in backtest_calibration.py showed the old approach was
badly overconfident, and the root cause traced to sigma (and sometimes mu)
being computed from as few as 4 data points per player with no shrinkage.
A player with a genuinely short current-season sample (rookies, Week 1-3,
a new starter) gets no benefit from anyone else's data under that scheme.

This module computes, for a given volume stat (e.g. rush attempts) and
efficiency ratio (e.g. yards per carry = rushing_yards / carries):

  1. Position-level pools: mean/std of the per-game stat across every
     player at that position, across multiple seasons. This is the prior
     for a player with no history of their own (rookies).
  2. Player career baseline: each player's own multi-season history,
     exponentially weighted so recent seasons count more, shrunk toward
     the position pool in proportion to how much career data they have.
  3. Blend with current-season recent form (still computed the existing
     way, last 4 games of the CURRENT season only) in proportion to how
     many current-season games exist - 0 games this season (Week 1) means
     100% career baseline; the weight shifts to recent form as the season
     goes on.

This is deliberately simple (linear shrinkage / empirical-Bayes-style
blending, not a fitted hierarchical Bayesian model) so it's auditable and
fast, and it's validated the same way the calibration fix was: by
backtesting against real historical outcomes (see backtest_new_baseline.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CAREER_SEASON_DECAY = 0.75   # weight multiplier per season further in the past
K_CAREER_POS = 40            # career games needed to fully trust player over position pool
K_RECENT_MU = 4              # current-season games needed to fully trust recent over career
K_RECENT_SIGMA = 8           # current-season games needed to fully trust recent sigma


def _clean_weekly(df: pd.DataFrame) -> pd.DataFrame:
    from common_markets import strip_generational_suffix
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "player_display_name" in df.columns:
        df["player"] = df["player_display_name"].fillna("")
    elif "player_name" in df.columns:
        df["player"] = df["player_name"].fillna("")
    # nflverse includes Jr./Sr./II/III; odds feeds consistently don't -
    # strip so this player's own career history actually matches their
    # props (e.g. "Deebo Samuel Sr." here vs. "Deebo Samuel" in odds data).
    df["player"] = df["player"].map(strip_generational_suffix)
    return df


def load_career_logs(seasons: list[int], data_dir: str = "data") -> pd.DataFrame:
    """Load and concatenate weekly logs for the given seasons. Skips any
    season whose parquet hasn't been fetched yet (no hard failure - a
    shorter career window just means less shrinkage benefit for players
    who've been in the league longer than what's cached)."""
    frames = []
    for season in seasons:
        p = Path(data_dir) / f"weekly_player_stats_{season}.parquet"
        if not p.exists():
            continue
        frames.append(_clean_weekly(pd.read_parquet(p)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _per_game_ratio(df: pd.DataFrame, num_col: str, den_col: Optional[str] = None) -> pd.Series:
    if den_col is None:
        return df[num_col]
    return df[num_col] / df[den_col].replace(0, np.nan)


def compute_position_pool(
    career_df: pd.DataFrame, position: str, num_col: str, den_col: Optional[str] = None
) -> tuple[float, float]:
    """Mean/std of a per-game stat (or ratio) across all players at a
    position, across every season in career_df. This is the prior for a
    player with no career data of their own."""
    sub = career_df[career_df.get("position") == position]
    vals = _per_game_ratio(sub, num_col, den_col).dropna()
    if len(vals) < 20:
        return (np.nan, np.nan)
    return (float(vals.mean()), float(vals.std()))


def player_career_baseline(
    player_logs: pd.DataFrame,
    current_season: int,
    num_col: str,
    den_col: Optional[str],
    position_mu: float,
    position_sigma: float,
) -> tuple[float, float, int]:
    """
    Season-recency-weighted career mu/sigma for one player's logs (all
    seasons up to but not including current_season), shrunk toward the
    position pool based on career sample size.

    Returns (career_mu, career_sigma, n_career_games).
    """
    if player_logs.empty or pd.isna(position_mu):
        return (position_mu, position_sigma, 0)

    vals = _per_game_ratio(player_logs, num_col, den_col)
    seasons = player_logs["season"]
    valid = vals.notna()
    vals, seasons = vals[valid], seasons[valid]
    if len(vals) == 0:
        return (position_mu, position_sigma, 0)

    age = (current_season - seasons).clip(lower=0)
    weights = CAREER_SEASON_DECAY ** age
    weights = weights / weights.sum()

    own_mu = float(np.average(vals, weights=weights))
    if len(vals) > 1:
        own_var = float(np.average((vals - own_mu) ** 2, weights=weights))
        own_sigma = float(np.sqrt(own_var))
    else:
        own_sigma = position_sigma

    n = len(vals)
    w_own = n / (n + K_CAREER_POS)
    career_mu = w_own * own_mu + (1 - w_own) * position_mu
    career_sigma = w_own * own_sigma + (1 - w_own) * position_sigma
    return (career_mu, career_sigma, n)


def blend_recent_with_career(
    recent_mu: float, recent_sigma: float, n_recent: int,
    career_mu: float, career_sigma: float,
) -> tuple[float, float]:
    """Final mu/sigma: blend current-season recent form with the career
    baseline, trusting recent form more as more current-season games
    accumulate. Sigma needs more games than mu before it's trusted (a
    variance estimate is noisier than a mean estimate at the same n)."""
    if pd.isna(career_mu):
        return (recent_mu, recent_sigma)
    if pd.isna(recent_mu) or n_recent == 0:
        return (career_mu, career_sigma)

    w_mu = n_recent / (n_recent + K_RECENT_MU)
    final_mu = w_mu * recent_mu + (1 - w_mu) * career_mu

    if pd.isna(recent_sigma):
        final_sigma = career_sigma
    else:
        w_sigma = n_recent / (n_recent + K_RECENT_SIGMA)
        final_sigma = w_sigma * recent_sigma + (1 - w_sigma) * career_sigma

    return (final_mu, final_sigma)

#!/usr/bin/env python3
"""
backtest_new_baseline.py

Fair, no-leakage comparison of the OLD (recent-4-games-only) player
baseline vs. the NEW hierarchical (position pool -> career -> recent form)
baseline from career_baseline.py, on the same historical weeks and the
same real outcomes used in backtest_calibration.py.

For each historical week W in season 2025, "current-season logs" are
truncated to week < W before computing latents - data/weekly_player_stats_2025.parquet
contains the whole (now-complete) season, so without this truncation the
model would see future games, which is not something it ever has access
to in production and would make the comparison meaningless.

Both variants get graded through the exact same probability math
(compute_model_prob_row) and the exact same real-outcome grading
(backtest_calibration.grade), so the only thing that differs is where
mu/sigma/lam came from.

Usage:
  python scripts/backtest_new_baseline.py
"""
import argparse
import json
import sys
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.dirname(__file__))
import career_baseline as cb
from backtest_calibration import latest_snapshot_per_week, load_outcomes, grade, MARKET_TO_STAT, brier_score
from make_props_edges import compute_model_prob_row
from make_player_prop_params import (
    estimate_rush_latents, estimate_receive_latents, estimate_pass_latents,
    derive_market_from_latents, name_std_str, slugify,
)

MIN_MARKET_N = 500
CALIBRATION_OUT = "models/nfl_prop_calibration.json"

SEASON = 2025
TEST_WEEKS = [4, 5, 6, 7, 9, 10, 11, 12, 14]


def load_truncated_logs(season: int, before_week: int) -> pd.DataFrame:
    """Same cleaning as make_player_prop_params._load_season_weekly, but
    truncated to week < before_week to avoid leaking future games."""
    df = pd.read_parquet(f"data/weekly_player_stats_{season}.parquet")
    df.columns = [c.lower() for c in df.columns]
    df = df[df["season"] == season].copy()
    df = df[df["week"] < before_week]

    if "carries" in df.columns and "rushing_attempts" not in df.columns:
        df["rushing_attempts"] = df["carries"]
    if "team" in df.columns and "recent_team" not in df.columns:
        df["recent_team"] = df["team"]
    if "player_display_name" in df.columns:
        df["player"] = df["player_display_name"].fillna("")
    from common_markets import strip_generational_suffix
    df["player"] = df["player"].map(strip_generational_suffix)

    tail = (
        df.sort_values(["player", "season", "week"])
        .groupby("player", group_keys=False)
        .apply(lambda g: g.tail(17))
        .reset_index(drop=True)
    )
    return tail


def build_mu_sigma(logs: pd.DataFrame, player_idx: pd.Index, season: int, career_df) -> dict:
    """Returns {market_std: (mu_series, sigma_series)} using whatever
    career_df is passed (None -> old recent-only behavior)."""
    latents = {
        "rush": estimate_rush_latents(logs, player_idx, alpha=0.4, career_df=career_df, season=season),
        "receive": estimate_receive_latents(logs, player_idx, alpha=0.4, career_df=career_df, season=season),
        "pass": estimate_pass_latents(logs, player_idx, alpha=0.4, career_df=career_df, season=season),
    }
    out = {}
    for market in ["rush_yds", "rush_attempts", "recv_yds", "receptions",
                   "pass_yds", "pass_attempts", "pass_completions"]:
        try:
            mu, sigma = derive_market_from_latents(market, latents, player_idx)
            out[market] = (mu, sigma)
        except Exception:
            continue
    return out


def score_week(preds_week: pd.DataFrame, mu_sigma: dict) -> pd.DataFrame:
    from common_markets import strip_generational_suffix
    df = preds_week.copy()
    # Must match player_idx's key format exactly (raw display name, suffix-
    # stripped) - this used to be name_std_str, which never once matched
    # career_grp/logs.groupby("player")'s raw keys. See the module docstring.
    df["player_key2"] = df["player"].astype(str).map(strip_generational_suffix)
    rows = []
    for market, (mu_s, sigma_s) in mu_sigma.items():
        sub = df[df["market_std"] == market].copy()
        if sub.empty:
            continue
        sub["mu"] = sub["player_key2"].map(mu_s.to_dict())
        sub["sigma"] = sub["player_key2"].map(sigma_s.to_dict())
        sub["lam"] = np.nan
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["mu", "sigma"])
    out["model_prob"] = out.apply(compute_model_prob_row, axis=1)
    return out


def main():
    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    all_preds = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    all_preds = all_preds[all_preds["season"] == SEASON]
    outcomes = load_outcomes(SEASON)

    full_career_df = cb.load_career_logs(list(range(SEASON - 6, SEASON + 1)))
    print(f"[bt] career pool: {len(full_career_df):,} rows, seasons {SEASON-6}-{SEASON}")

    old_all, new_all = [], []

    for wk in TEST_WEEKS:
        preds_wk = all_preds[all_preds["week"] == wk]
        preds_wk = preds_wk[preds_wk["market_std"].isin(MARKET_TO_STAT.keys())]
        preds_wk = preds_wk[preds_wk["point"].notna()]
        if preds_wk.empty:
            continue

        logs_trunc = load_truncated_logs(SEASON, before_week=wk)
        from common_markets import strip_generational_suffix
        player_idx = pd.Index(preds_wk["player"].astype(str).map(strip_generational_suffix).unique())

        # OLD: recent-4-games-only (no career_df)
        old_mu_sigma = build_mu_sigma(logs_trunc, player_idx, SEASON, career_df=None)
        old_scored = score_week(preds_wk, old_mu_sigma)
        old_scored["week"] = wk
        old_all.append(old_scored)

        # NEW: hierarchical (career_df spans seasons < SEASON only, by construction
        # inside player_career_baseline's "season < season" filter)
        new_mu_sigma = build_mu_sigma(logs_trunc, player_idx, SEASON, career_df=full_career_df)
        new_scored = score_week(preds_wk, new_mu_sigma)
        new_scored["week"] = wk
        new_all.append(new_scored)

        print(f"[bt] week {wk}: {len(old_scored)} old-scored, {len(new_scored)} new-scored")

    old_df = pd.concat(old_all, ignore_index=True)
    new_df = pd.concat(new_all, ignore_index=True)

    old_graded = grade(old_df, outcomes).dropna(subset=["model_prob", "outcome"])
    new_graded = grade(new_df, outcomes).dropna(subset=["model_prob", "outcome"])

    print("\n" + "=" * 70)
    print(f"OLD (recent-4-only):  n={len(old_graded):5d}  Brier={brier_score(old_graded,'model_prob'):.4f}"
          f"  corr={np.corrcoef(old_graded.model_prob, old_graded.outcome)[0,1]:+.3f}")
    print(f"NEW (hierarchical):   n={len(new_graded):5d}  Brier={brier_score(new_graded,'model_prob'):.4f}"
          f"  corr={np.corrcoef(new_graded.model_prob, new_graded.outcome)[0,1]:+.3f}")
    print("=" * 70)

    print("\nBy market:")
    for market in MARKET_TO_STAT:
        o = old_graded[old_graded.market_std == market]
        n = new_graded[new_graded.market_std == market]
        if len(o) < 30 or len(n) < 30:
            continue
        co = np.corrcoef(o.model_prob, o.outcome)[0, 1]
        cn = np.corrcoef(n.model_prob, n.outcome)[0, 1]
        print(f"  {market:20s} old: n={len(o):4d} brier={brier_score(o,'model_prob'):.4f} corr={co:+.3f}"
              f"   |   new: n={len(n):4d} brier={brier_score(n,'model_prob'):.4f} corr={cn:+.3f}")

    if args.refit_calibration:
        refit_calibration(new_graded)


def fit_curve(x, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(x, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def refit_calibration(new_graded: pd.DataFrame):
    """Refit models/nfl_prop_calibration.json against the NEW hierarchical
    baseline's outputs, since the old curve was fit against the old
    recent-4-only raw probabilities - a different underlying distribution
    that the old curve's mapping no longer matches correctly."""
    pooled = fit_curve(new_graded["model_prob"].values, new_graded["outcome"].values)
    result = {
        "_meta": {
            "fitted_on_season": SEASON,
            "fitted_on_weeks": TEST_WEEKS,
            "n_graded_picks": int(len(new_graded)),
            "note": "Fit against the hierarchical (career_baseline.py) model, not the old recent-4-only model.",
        },
        "_pooled_fallback": pooled,
        "_eligible_markets": list(MARKET_TO_STAT.keys()),
        "markets": {},
    }
    for market in MARKET_TO_STAT:
        sub = new_graded[new_graded.market_std == market]
        if len(sub) < MIN_MARKET_N:
            print(f"  refit: {market:20s} n={len(sub):5d} -> too small, using pooled fallback")
            continue
        result["markets"][market] = fit_curve(sub["model_prob"].values, sub["outcome"].values)
        print(f"  refit: {market:20s} n={len(sub):5d} -> fitted own curve")

    os.makedirs(os.path.dirname(CALIBRATION_OUT), exist_ok=True)
    with open(CALIBRATION_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[bt] wrote {CALIBRATION_OUT} (refit against new hierarchical baseline)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-calibration", action="store_true",
                     help="Overwrite models/nfl_prop_calibration.json using the new baseline's outputs.")
    args = ap.parse_args()
    main()

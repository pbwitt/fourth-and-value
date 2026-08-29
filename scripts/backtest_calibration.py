#!/usr/bin/env python3
"""
backtest_calibration.py

Grades historical model predictions (data/preds_historical/*.csv) against
real outcomes (data/weekly_player_stats_{season}.parquet) to answer one
question: when the model says a side has X% probability, does it actually
win X% of the time? Same question for the market's own de-vigged consensus
probability, for comparison.

This does NOT change any production model behavior. It's a read-only
diagnostic to quantify calibration before deciding how much (if any)
shrinkage to apply toward market consensus.

Usage:
  python scripts/backtest_calibration.py
  python scripts/backtest_calibration.py --season 2025
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from common_markets import std_player_name

MARKET_TO_STAT = {
    "rush_yds": "rushing_yards",
    "recv_yds": "receiving_yards",
    "pass_yds": "passing_yards",
    "receptions": "receptions",
    "rush_attempts": "carries",
    "pass_attempts": "attempts",
    "pass_completions": "completions",
    "pass_tds": "passing_tds",
}


def latest_snapshot_per_week(pattern: str) -> list[str]:
    """Return one file per (season inferred from week label) - the most
    recent snapshot for that week, skipping the current in-progress season
    (no real outcomes exist yet to grade against)."""
    by_week = defaultdict(list)
    for f in glob.glob(pattern):
        m = re.search(r"week(\d+)_(\d{8})_(\d{6})", os.path.basename(f))
        if not m:
            continue
        wk, d, t = int(m.group(1)), m.group(2), m.group(3)
        if d.startswith("2026"):
            continue  # current season in progress, no outcomes to grade yet
        by_week[wk].append((d + t, f))
    return [sorted(v)[-1][1] for v in by_week.values()]


def load_outcomes(season: int) -> pd.DataFrame:
    path = f"data/weekly_player_stats_{season}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run: python scripts/fetch_weekly_player_stats.py --season {season}"
        )
    df = pd.read_parquet(path)
    df = df[df["player_display_name"].notna()].copy()
    df["name_std"] = df["player_display_name"].map(std_player_name)
    return df


def grade(preds: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    preds = preds[preds["market_std"].isin(MARKET_TO_STAT.keys())].copy()
    preds = preds[preds["model_prob"].notna() & preds["point"].notna()].copy()
    preds["name_std"] = preds["player"].astype(str).map(std_player_name)

    preds = preds.drop_duplicates(subset=["season", "week", "name_std", "market_std", "point", "name"])

    rows = []
    for market_std, stat_col in MARKET_TO_STAT.items():
        sub = preds[preds["market_std"] == market_std]
        if sub.empty:
            continue
        merged = sub.merge(
            outcomes[["name_std", "season", "week", stat_col]],
            on=["name_std", "season", "week"],
            how="inner",
        )
        merged = merged.dropna(subset=[stat_col])
        merged["actual"] = merged[stat_col]
        merged["side"] = merged["name"].str.lower()
        merged = merged[merged["side"].isin(["over", "under"])]

        is_push = merged["actual"] == merged["point"]
        merged = merged[~is_push]

        won_over = (merged["side"] == "over") & (merged["actual"] > merged["point"])
        won_under = (merged["side"] == "under") & (merged["actual"] < merged["point"])
        merged["outcome"] = (won_over | won_under).astype(int)
        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def calibration_table(df: pd.DataFrame, prob_col: str, bins=None) -> pd.DataFrame:
    if bins is None:
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    d = df[[prob_col, "outcome"]].dropna().copy()
    d["bucket"] = pd.cut(d[prob_col], bins=bins, include_lowest=True)
    g = d.groupby("bucket", observed=True).agg(
        n=("outcome", "size"),
        predicted=(prob_col, "mean"),
        actual=("outcome", "mean"),
    )
    g["gap_pp"] = (g["actual"] - g["predicted"]) * 100
    return g


def brier_score(df: pd.DataFrame, prob_col: str) -> float:
    d = df[[prob_col, "outcome"]].dropna()
    return float(np.mean((d[prob_col] - d["outcome"]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    print(f"[backtest] using {len(files)} weekly snapshots (one per week, most recent)")

    preds = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    preds = preds[preds["season"] == args.season]

    outcomes = load_outcomes(args.season)
    graded = grade(preds, outcomes)

    if graded.empty:
        print("[backtest] no gradeable rows found - check name matching / column names")
        return

    print(f"\n[backtest] {len(graded):,} graded (player, market, line, side) picks, season {args.season}\n")

    print("=" * 78)
    print("MODEL calibration (model_prob vs. actual hit rate)")
    print("=" * 78)
    print(calibration_table(graded, "model_prob").to_string())
    print(f"\nModel Brier score: {brier_score(graded, 'model_prob'):.4f}  (lower is better; 0.25 = coin-flip baseline)")

    if "consensus_prob" in graded.columns and graded["consensus_prob"].notna().any():
        print("\n" + "=" * 78)
        print("MARKET calibration (de-vigged consensus_prob vs. actual hit rate)")
        print("=" * 78)
        print(calibration_table(graded, "consensus_prob").to_string())
        print(f"\nMarket Brier score: {brier_score(graded, 'consensus_prob'):.4f}")

    print("\n" + "=" * 78)
    print("BY MARKET: model calibration gap (actual - predicted) in high-confidence bucket (>=90%)")
    print("=" * 78)
    for market_std in MARKET_TO_STAT:
        sub = graded[graded["market_std"] == market_std]
        high = sub[sub["model_prob"] >= 0.90]
        if len(high) < 10:
            continue
        print(f"  {market_std:20s} n={len(high):5d}  predicted={high['model_prob'].mean():.3f}  actual={high['outcome'].mean():.3f}")


if __name__ == "__main__":
    main()

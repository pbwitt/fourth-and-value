#!/usr/bin/env python3
"""
backtest_early_vs_late.py

Tests the actual strategy described: bet an outlier book's line as soon as
it's posted (using the EARLIEST available snapshot each week), rather than
whatever the last/only snapshot happened to be - which is what every other
backtest tonight actually used, since latest_snapshot_per_week() always
picks the LAST snapshot. That's a real gap: for 6 of the 9 backtested
weeks there was only one snapshot (or a few minutes of spread) at all, so
"early vs late" was never actually testable for them. Only weeks 3, 4, 5,
6 have enough real time spread (44-166 hours) to test this for real.

For each of those weeks:
  - Compute consensus_line ourselves from the EARLIEST snapshot (it wasn't
    saved in the raw historical file at that point in the project), using
    every book present at that early moment.
  - Identify early outlier books (line differs from early consensus by
    >= OUTLIER_THRESHOLD).
  - Fade the outlier using the EARLY price (this is the price you'd have
    actually gotten hammering it Tuesday/Wednesday) and the model_prob
    that was actually live in production at that time (this is what a
    real bettor would have seen - not a model rebuilt with hindsight).
  - Grade against real final outcomes.
  - Separately: quantify how often the early outlier's line actually
    moved toward consensus by the late snapshot (validates or refutes the
    "it's expected to correct" assumption directly).

Usage:
  python scripts/backtest_early_vs_late.py
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from backtest_calibration import load_outcomes, MARKET_TO_STAT
from backtest_topbets_roi import american_payout
from common_markets import std_player_name

WEEKS_WITH_SPREAD = [3, 4, 5, 6]
SEASON = 2025
OUTLIER_THRESHOLD = 1.0
MODEL_AGREEMENT_TOL = 1.5


def snapshots_for_week(wk):
    out = []
    for f in os.listdir("data/preds_historical"):
        m = re.search(r"props_with_model_week(\d+)_(\d{8})_(\d{6})", f)
        if not m or int(m.group(1)) != wk:
            continue
        out.append((m.group(2) + m.group(3), f"data/preds_historical/{f}"))
    return sorted(out)


def compute_consensus(df):
    cons = (
        df.groupby(["player", "market_std", "name"], as_index=False)
        .agg(consensus_line=("point", "median"))
    )
    return df.merge(cons, on=["player", "market_std", "name"], how="left")


def resolve_bets(df, outcomes, price_col="price"):
    df = df[df["market_std"].isin(MARKET_TO_STAT.keys())].dropna(subset=["mu", "consensus_line", "point", price_col])
    df["name_std"] = df["player"].astype(str).map(std_player_name)
    stat_lookup = outcomes.set_index(["name_std", "season", "week"])

    bets = []
    for keys, g in df.groupby(["player", "market_std", "bookmaker"], dropna=False):
        r = g.iloc[0]
        line_gap = r["point"] - r["consensus_line"]
        if abs(line_gap) < OUTLIER_THRESHOLD:
            continue
        if abs(r["mu"] - r["consensus_line"]) > MODEL_AGREEMENT_TOL:
            continue
        fade_side = "under" if line_gap > 0 else "over"
        side_row = g[g["name"].str.lower() == fade_side]
        if side_row.empty:
            continue
        price = side_row.iloc[0][price_col]

        key = (r["name_std"], SEASON, r["week"])
        if key not in stat_lookup.index:
            continue
        stat_col = MARKET_TO_STAT[r["market_std"]]
        actual = stat_lookup.loc[key, stat_col]
        if isinstance(actual, pd.Series):
            actual = actual.iloc[0]
        if pd.isna(actual) or actual == r["point"]:
            continue
        won = (actual > r["point"]) if fade_side == "over" else (actual < r["point"])
        pnl = american_payout(100, price) if won else -100
        bets.append({
            "week": r["week"], "player": r["player"], "market_std": r["market_std"],
            "book": r["bookmaker"], "book_line": r["point"], "consensus_line": r["consensus_line"],
            "model_mu": r["mu"], "fade_side": fade_side, "price": price, "actual": actual,
            "won": int(won), "pnl": pnl,
        })
    return pd.DataFrame(bets)


def main():
    outcomes = load_outcomes(SEASON)
    early_all, late_all = [], []
    movement_stats = []

    for wk in WEEKS_WITH_SPREAD:
        snaps = snapshots_for_week(wk)
        if len(snaps) < 2:
            continue
        early_path, late_path = snaps[0][1], snaps[-1][1]

        early_df = pd.read_csv(early_path, low_memory=False)
        late_df = pd.read_csv(late_path, low_memory=False)
        early_df["week"] = wk
        late_df["week"] = wk

        early_df = compute_consensus(early_df)
        if "consensus_line" not in late_df.columns or late_df["consensus_line"].isna().all():
            late_df = compute_consensus(late_df)

        # Movement check: for early outliers, did the SAME book's line move
        # toward the early consensus by the late snapshot?
        early_outliers = early_df.copy()
        early_outliers["line_gap"] = early_outliers["point"] - early_outliers["consensus_line"]
        early_outliers = early_outliers[early_outliers["line_gap"].abs() >= OUTLIER_THRESHOLD]
        key = ["player", "market_std", "bookmaker", "name"]
        cmp = early_outliers[key + ["point", "consensus_line", "line_gap"]].merge(
            late_df[key + ["point"]], on=key, suffixes=("_early", "_late")
        )
        if not cmp.empty:
            cmp["moved_toward_consensus"] = (
                (cmp["point_early"] - cmp["consensus_line"]).abs() >
                (cmp["point_late"] - cmp["consensus_line"]).abs()
            )
            movement_stats.append({
                "week": wk, "n_outliers": len(cmp),
                "pct_moved_toward_consensus": cmp["moved_toward_consensus"].mean() * 100,
                "pct_unchanged": (cmp["point_early"] == cmp["point_late"]).mean() * 100,
            })

        early_bets = resolve_bets(early_df, outcomes)
        late_bets = resolve_bets(late_df, outcomes)
        early_all.append(early_bets)
        late_all.append(late_bets)

    early_bets = pd.concat(early_all, ignore_index=True)
    late_bets = pd.concat(late_all, ignore_index=True)

    print("=" * 78)
    print("Does the early outlier line move TOWARD consensus by the late snapshot?")
    print("=" * 78)
    print(pd.DataFrame(movement_stats).to_string(index=False))

    print("\n" + "=" * 78)
    print("BET EARLY (at the price/line available when first posted)")
    print("=" * 78)
    if len(early_bets):
        print(f"n={len(early_bets)}  win_rate={early_bets['won'].mean()*100:.1f}%  "
              f"ROI={early_bets['pnl'].sum()/(len(early_bets)*100)*100:+.1f}%")
    else:
        print("no qualifying bets")

    print("\n" + "=" * 78)
    print("BET LATE (only using the final/last available snapshot - what every other backtest tonight used)")
    print("=" * 78)
    if len(late_bets):
        print(f"n={len(late_bets)}  win_rate={late_bets['won'].mean()*100:.1f}%  "
              f"ROI={late_bets['pnl'].sum()/(len(late_bets)*100)*100:+.1f}%")
    else:
        print("no qualifying bets")

    early_bets.to_csv("data/qc/early_bets_backtest.csv", index=False)
    late_bets.to_csv("data/qc/late_bets_backtest_4weeks.csv", index=False)
    print("\nWrote data/qc/early_bets_backtest.csv and late_bets_backtest_4weeks.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_market_behavior_2025.py

Market-wide (not model-specific) analysis of NFL player prop behavior across
the 2025 season, for a season-preview blog post. Answers three questions
using real lines and real outcomes, independent of our own model:

  1. Do overs or unders hit more often, by market? (and is that difference
     big enough to survive real vig, i.e. is "always bet the under" +EV?)
  2. Which markets are more volatile (spikier game-to-game) than others?
  3. What general, market-level suggestions fall out of (1) and (2)?

Uses the same real snapshot archive and real 2025 outcomes as every other
backtest in this project - one closing-ish snapshot per available week
(1,3,4,5,6,7,9,10,11,12,14), not walk-forward (this is about market
behavior, not about our model, so no leakage concern applies).

Usage:
  python scripts/analyze_market_behavior_2025.py
"""
import glob
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

SEASON = 2025
PRETTY = {
    "rush_yds": "Rushing Yards", "recv_yds": "Receiving Yards", "pass_yds": "Passing Yards",
    "receptions": "Receptions", "rush_attempts": "Rush Attempts", "pass_attempts": "Pass Attempts",
    "pass_completions": "Pass Completions", "pass_tds": "Passing TDs",
}


def latest_snapshot_per_week(pattern):
    by_week = defaultdict(list)
    for f in glob.glob(pattern):
        m = re.search(r"week(\d+)_(\d{8})_(\d{6})", os.path.basename(f))
        if not m:
            continue
        wk, d, t = int(m.group(1)), m.group(2), m.group(3)
        if d.startswith("2026"):
            continue
        by_week[wk].append((d + t, f))
    return [sorted(v)[-1][1] for v in by_week.values()]


def load_all_props():
    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df = df[df["season"] == SEASON]
    df = df[df["market_std"].isin(MARKET_TO_STAT.keys())]
    df = df.dropna(subset=["point", "price"])
    df["name_std"] = df["player"].astype(str).map(std_player_name)
    df["side"] = df["name"].astype(str).str.lower()
    df = df[df["side"].isin(["over", "under"])]
    return df


def part1_over_under_bias(df, outcomes):
    """Market-wide: does the actual result land over or under the
    consensus number more often, by market? And is fading that bias with
    real vig-adjusted prices actually profitable?"""
    stat_lookup = outcomes.set_index(["name_std", "season", "week"])

    # One row per (player, market, week): consensus line = median book line
    cons = (
        df.groupby(["name_std", "market_std", "week"], as_index=False)
        .agg(consensus_line=("point", "median"), num_books=("bookmaker", "nunique"))
    )
    cons["season"] = SEASON

    rows = []
    for _, r in cons.iterrows():
        key = (r["name_std"], SEASON, r["week"])
        if key not in stat_lookup.index:
            continue
        stat_col = MARKET_TO_STAT[r["market_std"]]
        actual = stat_lookup.loc[key, stat_col]
        if isinstance(actual, pd.Series):
            actual = actual.iloc[0]
        if pd.isna(actual) or actual == r["consensus_line"]:
            continue
        rows.append({
            "market_std": r["market_std"], "week": r["week"], "name_std": r["name_std"],
            "consensus_line": r["consensus_line"], "actual": actual,
            "over_hit": int(actual > r["consensus_line"]),
        })
    hits = pd.DataFrame(rows)

    print("=" * 90)
    print("PART 1: Over/under hit rate vs. CONSENSUS line, by market (n = player-weeks, pushes excluded)")
    print("=" * 90)
    seg = hits.groupby("market_std").agg(n=("over_hit", "size"), over_hit_rate=("over_hit", "mean"))
    seg["under_hit_rate"] = 1 - seg["over_hit_rate"]
    seg["market"] = seg.index.map(lambda m: PRETTY.get(m, m))
    seg = seg[["market", "n", "over_hit_rate", "under_hit_rate"]].sort_values("under_hit_rate", ascending=False)
    print(seg.to_string(index=False, formatters={"over_hit_rate": "{:.1%}".format, "under_hit_rate": "{:.1%}".format}))

    # Real-money version: always bet the under (or over), one bet per real
    # (book, player, market, week, line) row - own price AND own line kept
    # together (mixing "median line" with "median price across books" would
    # silently blend different books' different lines into one fake bet).
    print("\n" + "=" * 90)
    print("PART 1b: Real ROI of a naive 'always bet the under' / 'always bet the over' strategy, by market")
    print("(flat $100/bet, EVERY real book/line row graded against its own line - no cross-book blending)")
    print("=" * 90)

    stat_lookup = outcomes.set_index(["name_std", "season", "week"])
    bet_rows = []
    for _, r in df.iterrows():
        key = (r["name_std"], SEASON, r["week"])
        if key not in stat_lookup.index:
            continue
        stat_col = MARKET_TO_STAT[r["market_std"]]
        actual = stat_lookup.loc[key, stat_col]
        if isinstance(actual, pd.Series):
            actual = actual.iloc[0]
        if pd.isna(actual) or actual == r["point"] or r["price"] == 0:
            continue
        won = (actual < r["point"]) if r["side"] == "under" else (actual > r["point"])
        pnl = american_payout(100, r["price"]) if won else -100
        bet_rows.append({"market_std": r["market_std"], "side": r["side"], "won": int(won), "pnl": pnl})
    bets = pd.DataFrame(bet_rows)

    for side in ["under", "over"]:
        sub = bets[bets["side"] == side]
        seg2 = sub.groupby("market_std").agg(n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum"))
        seg2["roi_pct"] = seg2["pnl"] / (seg2["n"] * 100) * 100
        seg2["market"] = seg2.index.map(lambda m: PRETTY.get(m, m))
        seg2 = seg2[["market", "n", "win_rate", "roi_pct"]].sort_values("roi_pct", ascending=False)
        print(f"\n--- Always bet the {side.upper()} (every book/line, {len(sub)} total bets) ---")
        print(seg2.to_string(index=False, formatters={"win_rate": "{:.1%}".format, "roi_pct": "{:+.1f}%".format}))

    return hits


def part2_volatility(outcomes):
    """Which stat swings the most game-to-game, relative to its own mean,
    for the players who actually get lines posted on them? Uses full-season
    game logs (not just prop-line weeks) for every player who appears in the
    props snapshots at all, so the volatility number reflects real players
    getting bet, not the whole league including non-relevant depth players."""
    props = load_all_props()
    active_players = props["name_std"].unique()

    print("\n" + "=" * 90)
    print("PART 2: Game-to-game volatility by market (players who had real props posted on them, >=4 games)")
    print("=" * 90)

    rows = []
    for market_std, stat_col in MARKET_TO_STAT.items():
        sub = outcomes[outcomes["name_std"].isin(active_players)][["name_std", "week", stat_col]].dropna()
        grp = sub.groupby("name_std")[stat_col]
        counts = grp.transform("count")
        sub = sub[counts >= 4]
        stats = sub.groupby("name_std")[stat_col].agg(["mean", "std"])
        stats = stats[stats["mean"] > 0.5]  # drop near-zero-usage players (CV explodes meaninglessly)
        stats["cv"] = stats["std"] / stats["mean"]

        # "Blowup rate": fraction of individual games where a player's actual
        # deviates from THEIR OWN season mean by more than 40% - an intuitive
        # measure of how often this market produces a game far from normal.
        merged = sub.merge(stats[["mean"]], on="name_std", suffixes=("", "_playermean"))
        merged["pct_dev"] = (merged[stat_col] - merged["mean"]).abs() / merged["mean"]
        blowup_rate = (merged["pct_dev"] >= 0.4).mean()

        rows.append({
            "market_std": market_std, "market": PRETTY.get(market_std, market_std),
            "n_players": len(stats), "median_cv": stats["cv"].median(),
            "blowup_rate_40pct": blowup_rate,
        })

    vol = pd.DataFrame(rows).sort_values("median_cv", ascending=False)
    print(vol[["market", "n_players", "median_cv", "blowup_rate_40pct"]].to_string(
        index=False, formatters={"median_cv": "{:.2f}".format, "blowup_rate_40pct": "{:.1%}".format}
    ))
    return vol


def main():
    outcomes = load_outcomes(SEASON)
    df = load_all_props()
    print(f"Loaded {len(df)} book/line rows across weeks {sorted(df['week'].unique())}\n")

    hits = part1_over_under_bias(df, outcomes)
    vol = part2_volatility(outcomes)

    os.makedirs("data/qc", exist_ok=True)
    hits.to_csv("data/qc/market_behavior_over_under_2025.csv", index=False)
    vol.to_csv("data/qc/market_behavior_volatility_2025.csv", index=False)
    print("\nWrote data/qc/market_behavior_over_under_2025.csv and market_behavior_volatility_2025.csv")


if __name__ == "__main__":
    main()

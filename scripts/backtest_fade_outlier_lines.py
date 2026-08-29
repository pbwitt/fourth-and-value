#!/usr/bin/env python3
"""
backtest_fade_outlier_lines.py

Tests a different strategy than "bet the model's biggest edge_bps" (which
backtest_topbets_roi.py showed loses money): fade a single book's line when
it disagrees with the multi-book consensus line AND the model's own
estimate agrees with that consensus. Rationale: consensus across ~6-7 books
is a much more robust "true value" estimate than any one book, and an
outlier book is more likely stale/wrong than the market as a whole; model
agreement with consensus is an independent check that consensus itself
isn't the odd one out. Real prices, no price filtering (bet it regardless
of the odds offered), same real 2025 outcomes as every other backtest in
this project.

Usage:
  python scripts/backtest_fade_outlier_lines.py
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from backtest_calibration import latest_snapshot_per_week, load_outcomes, MARKET_TO_STAT
from common_markets import std_player_name, strip_generational_suffix
from backtest_topbets_roi import american_payout
from backtest_new_baseline import load_truncated_logs, build_mu_sigma, TEST_WEEKS
import career_baseline as cb

SEASON = 2025
MODEL_AGREEMENT_TOL = 1.5   # |mu - consensus_line| <= this counts as "model agrees with consensus"
OUTLIER_THRESHOLD = 1.0     # |book_line - consensus_line| >= this counts as an outlier book

NORMAL_MARKETS = ["rush_yds", "recv_yds", "pass_yds", "receptions",
                   "rush_attempts", "pass_attempts", "pass_completions"]


def load_all():
    """Loads historical props with whatever mu was saved in production at
    the time (mixed code versions across the season - see
    load_all_consistent_mu for the more rigorous version)."""
    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df = df[df["season"] == SEASON]
    df = df[df["market_std"].isin(MARKET_TO_STAT.keys())]
    df = df.dropna(subset=["mu", "consensus_line", "point", "price"])
    df["name_std"] = df["player"].astype(str).map(std_player_name)
    return df


def load_all_consistent_mu():
    """Recomputes mu with ONE consistent model (the corrected hierarchical
    estimator, same as backtest_new_baseline.py / the Research paper)
    across every week, instead of trusting whatever mu happened to be
    saved in production at the time. This is the more rigorous version -
    an apples-to-apples test of the strategy against one fixed model."""
    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    all_preds = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    all_preds = all_preds[all_preds["season"] == SEASON]
    all_preds = all_preds[all_preds["market_std"].isin(NORMAL_MARKETS)]
    all_preds = all_preds.dropna(subset=["consensus_line", "point", "price"])

    full_career_df = cb.load_career_logs(list(range(SEASON - 6, SEASON + 1)))

    out = []
    for wk in TEST_WEEKS:
        preds_wk = all_preds[all_preds["week"] == wk].copy()
        if preds_wk.empty:
            continue
        logs_trunc = load_truncated_logs(SEASON, before_week=wk)
        preds_wk["player"] = preds_wk["player"].astype(str).map(strip_generational_suffix)
        player_idx = pd.Index(preds_wk["player"].unique())
        mu_sigma = build_mu_sigma(logs_trunc, player_idx, SEASON, career_df=full_career_df)

        for mkt in NORMAL_MARKETS:
            if mkt not in mu_sigma:
                continue
            mu_s, _ = mu_sigma[mkt]
            sub = preds_wk[preds_wk["market_std"] == mkt].copy()
            sub["mu"] = sub["player"].map(mu_s.to_dict())
            out.append(sub)

    df = pd.concat(out, ignore_index=True)
    df = df.dropna(subset=["mu"])
    df["name_std"] = df["player"].astype(str).map(std_player_name)
    return df


def grade_row(actual, point, side):
    if actual == point:
        return None  # push
    if side == "over":
        return int(actual > point)
    if side == "under":
        return int(actual < point)
    return None


def run(df, outcomes, require_model_agreement: bool):
    """One bet per (week, player, market, book): the raw data has a
    separate row per side (over AND under) for the same book/line, but a
    "fade this book's line" decision is a single wager on one side using
    that side's own price - iterating over both side-rows independently
    would double-count every qualifying book/line and could even grab the
    wrong side's price when over/under prices differ."""
    stat_lookup = outcomes.set_index(["name_std", "season", "week"])

    # "book" exists as a column but is entirely empty in this data; the
    # real bookmaker identifier is "bookmaker". Check for actual non-null
    # data, not just column presence, or this silently collapses every
    # book together into one group per player/market/week.
    book_col = "book" if ("book" in df.columns and df["book"].notna().any()) else "bookmaker"

    bets = []
    for keys, g in df.groupby(["week", "player", "market_std", book_col], dropna=False):
        r = g.iloc[0]  # point/consensus_line/mu are identical across the group's side-rows
        line_gap = r["point"] - r["consensus_line"]
        if abs(line_gap) < OUTLIER_THRESHOLD:
            continue  # this book isn't a meaningful outlier, skip

        if require_model_agreement:
            if abs(r["mu"] - r["consensus_line"]) > MODEL_AGREEMENT_TOL:
                continue  # model itself disagrees with consensus - sit out

        # Fade the outlier: if book's line is HIGH vs consensus, bet under; if LOW, bet over
        fade_side = "under" if line_gap > 0 else "over"

        side_row = g[g["name"].str.lower() == fade_side]
        if side_row.empty:
            continue  # this book didn't actually offer the side we'd need to fade it
        price = side_row.iloc[0]["price"]

        lookup_key = (r["name_std"], r["season"], r["week"])
        if lookup_key not in stat_lookup.index:
            continue
        stat_col = MARKET_TO_STAT[r["market_std"]]
        actual = stat_lookup.loc[lookup_key, stat_col]
        if isinstance(actual, pd.Series):
            actual = actual.iloc[0]
        if pd.isna(actual):
            continue

        outcome = grade_row(actual, r["point"], fade_side)
        if outcome is None:
            continue  # push

        pnl = american_payout(100, price) if outcome == 1 else -100
        bets.append({
            "week": r["week"], "player": r["player"], "market_std": r["market_std"],
            "book": r.get("bookmaker", r.get("book")), "book_line": r["point"],
            "consensus_line": r["consensus_line"], "model_mu": r["mu"],
            "fade_side": fade_side, "price": price, "actual": actual,
            "won": outcome, "pnl": pnl,
        })
    return pd.DataFrame(bets)


def summarize(bets: pd.DataFrame, label: str):
    if bets.empty:
        print(f"{label}: no qualifying bets")
        return
    n = len(bets)
    win_rate = bets["won"].mean()
    roi = bets["pnl"].sum() / (n * 100) * 100
    print(f"{label}: n={n}  win_rate={win_rate*100:.1f}%  total_pnl=${bets['pnl'].sum():,.2f}  ROI={roi:+.1f}%")


def main():
    print("Building consistent-model dataset (one hierarchical estimator across all weeks)...")
    df = load_all_consistent_mu()
    outcomes = load_outcomes(SEASON)

    print("=" * 78)
    print(f"Fade single-book outlier lines (>= {OUTLIER_THRESHOLD} pt from consensus), real 2025 data")
    print("Using ONE consistent model (not mixed historical production versions)")
    print("=" * 78)

    bets_naive = run(df, outcomes, require_model_agreement=False)
    summarize(bets_naive, "Fade ANY outlier book (no model filter)")

    bets_filtered = run(df, outcomes, require_model_agreement=True)
    summarize(bets_filtered, f"Fade outlier book, ONLY when model agrees w/ consensus (tol={MODEL_AGREEMENT_TOL})")

    print(f"\n{'='*78}\nROW-LEVEL SAMPLE (first 30 qualifying bets, model-agreement version)\n{'='*78}")
    cols = ["week", "player", "market_std", "book", "book_line", "consensus_line",
            "model_mu", "fade_side", "price", "actual", "won", "pnl"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(bets_filtered[cols].head(30).to_string(index=False))

    print("\nBy market (model-agreement version):")
    if not bets_filtered.empty:
        seg = bets_filtered.groupby("market_std").agg(
            n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum")
        )
        seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
        print(seg.to_string())

    print("\nBy week (model-agreement version):")
    if not bets_filtered.empty:
        seg = bets_filtered.groupby("week").agg(
            n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum")
        )
        seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
        print(seg.to_string())

    bets_filtered.to_csv("data/qc/fade_outlier_lines_backtest.csv", index=False)
    print("\nWrote data/qc/fade_outlier_lines_backtest.csv")


if __name__ == "__main__":
    main()

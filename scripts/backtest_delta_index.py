#!/usr/bin/env python3
"""
backtest_delta_index.py

Deeper segmentation of the "fade the outlier book" idea: instead of a
binary "model agrees with consensus" filter, compute the actual gaps
between all three reference points - model mu, multi-book consensus
line, and one specific book's own line - and see whether the SIZE of
these gaps, by market, indicates a stronger or weaker real signal.

For every (book, player, market, week) candidate:
  book_dev  = book_line - consensus_line   (how far this book strays from the market)
  model_dev = model_mu  - consensus_line   (how far our model strays from the market)
  signal    = model_mu  - book_line        (model vs this specific book - the bet itself)

Bet direction is simply "toward the model": over if model_mu > book_line,
under otherwise, at that book's own real price. Bucketed by |signal| per
market to look for where (if anywhere) a real, non-overfit edge lives.

Uses the same consistent hierarchical model and real 2025 outcomes as
every other backtest in this project. Reuses the row-grouping fix from
backtest_fade_outlier_lines.py (the "book" column is empty; "bookmaker"
is the real one - grouping on the wrong one silently merges books).

Usage:
  python scripts/backtest_delta_index.py
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from backtest_calibration import load_outcomes, MARKET_TO_STAT
from backtest_fade_outlier_lines import load_all_consistent_mu, grade_row
from backtest_topbets_roi import american_payout

SEASON = 2025


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (week, player, market, book): resolves the correct
    price for whichever side the model actually favors (model_mu vs that
    book's own line), instead of processing both side-rows independently."""
    rows = []
    for keys, g in df.groupby(["week", "player", "market_std", "bookmaker"], dropna=False):
        r = g.iloc[0]
        if pd.isna(r["consensus_line"]) or pd.isna(r["mu"]):
            continue

        bet_side = "over" if r["mu"] > r["point"] else "under"
        side_row = g[g["name"].str.lower() == bet_side]
        if side_row.empty:
            continue
        price = side_row.iloc[0]["price"]

        rows.append({
            "week": r["week"], "player": r["player"], "market_std": r["market_std"],
            "book": r["bookmaker"], "book_line": r["point"], "consensus_line": r["consensus_line"],
            "model_mu": r["mu"], "bet_side": bet_side, "price": price,
            "book_dev": r["point"] - r["consensus_line"],
            "model_dev": r["mu"] - r["consensus_line"],
            "signal": r["mu"] - r["point"],
            "name_std": r["name_std"], "season": r["season"],
        })
    return pd.DataFrame(rows)


def grade_candidates(cands: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    stat_lookup = outcomes.set_index(["name_std", "season", "week"])
    out = []
    for _, r in cands.iterrows():
        key = (r["name_std"], r["season"], r["week"])
        if key not in stat_lookup.index:
            continue
        stat_col = MARKET_TO_STAT[r["market_std"]]
        actual = stat_lookup.loc[key, stat_col]
        if isinstance(actual, pd.Series):
            actual = actual.iloc[0]
        if pd.isna(actual):
            continue
        outcome = grade_row(actual, r["book_line"], r["bet_side"])
        if outcome is None:
            continue  # push
        pnl = american_payout(100, r["price"]) if outcome == 1 else -100
        rec = r.to_dict()
        rec["actual"] = actual
        rec["won"] = outcome
        rec["pnl"] = pnl
        out.append(rec)
    return pd.DataFrame(out)


def main():
    print("Loading consistent-model dataset...")
    df = load_all_consistent_mu()
    outcomes = load_outcomes(SEASON)

    cands = build_candidates(df)
    graded = grade_candidates(cands, outcomes)
    print(f"Total graded candidates: {len(graded)}\n")

    graded["abs_signal"] = graded["signal"].abs()
    graded["aligned"] = np.sign(graded["book_dev"]) != np.sign(graded["model_dev"])  # True = "clean fade" shape (book and model pull opposite ways from consensus)

    bins = [0, 2, 4, 6, 10, 1000]
    labels = ["0-2", "2-4", "4-6", "6-10", "10+"]
    graded["signal_bucket"] = pd.cut(graded["abs_signal"], bins=bins, labels=labels, include_lowest=True)

    print("=" * 100)
    print("PER-MARKET INDEX: win rate / ROI by |model - book| signal size")
    print("=" * 100)
    for mkt in sorted(graded["market_std"].unique()):
        sub = graded[graded["market_std"] == mkt]
        print(f"\n--- {mkt} (n={len(sub)}) ---")
        seg = sub.groupby("signal_bucket", observed=True).agg(
            n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum")
        )
        seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
        print(seg.to_string())

    print("\n" + "=" * 100)
    print("Overall by signal bucket (all markets pooled)")
    print("=" * 100)
    seg = graded.groupby("signal_bucket", observed=True).agg(
        n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum")
    )
    seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
    print(seg.to_string())

    print("\n" + "=" * 100)
    print('"Aligned" (clean fade: model and book point opposite ways from consensus) vs not, by market')
    print("=" * 100)
    seg = graded.groupby(["market_std", "aligned"], observed=True).agg(
        n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum")
    )
    seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
    print(seg.to_string())

    graded.to_csv("data/qc/delta_index_backtest.csv", index=False)
    print("\nWrote data/qc/delta_index_backtest.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute edges for NHL props by joining odds + consensus + baseline model.

For Phase A MVP, uses market consensus as model baseline (market-implied predictions).
Future phases will add trained ML models for SOG, Goals, Assists, Points.

Inputs:
  - data/nhl/processed/odds_props_{date}.csv
  - data/nhl/consensus/consensus_props_{date}.csv

Outputs:
  - data/nhl/props/props_with_model_{date}.csv

Usage:
  python3 scripts/nhl/make_nhl_edges.py --date 2025-10-08
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    try:
        o = float(odds)
        if np.isnan(o):
            return np.nan
        if o > 0:
            return 100.0 / (o + 100.0)
        else:
            return (-o) / ((-o) + 100.0)
    except (ValueError, TypeError):
        return np.nan


def prob_to_american(prob: float) -> float:
    """Convert probability to American odds."""
    try:
        p = float(prob)
        if np.isnan(p) or p <= 0 or p >= 1:
            return np.nan
        if p >= 0.5:
            return -100 * p / (1 - p)
        else:
            return 100 * (1 - p) / p
    except (ValueError, TypeError):
        return np.nan


def normalize_player_name(name: str) -> str:
    """Normalize player name for joining."""
    if not name or not isinstance(name, str):
        return ""
    return " ".join(name.strip().lower().split())


def compute_baseline_model_prob(consensus_prob: float, side: str) -> float:
    """
    Phase A baseline: Use market consensus as model probability.

    For Over/Under markets, consensus_prob represents the market's implied
    probability. We use this as our baseline model prediction.

    Future phases will replace this with trained ML models.
    """
    # For Phase A MVP, market consensus IS our model
    return consensus_prob


def main():
    parser = argparse.ArgumentParser(description="Compute NHL prop edges")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print(f"[make_nhl_edges] Computing edges for {date_str}...")

    # Load odds data
    props_path = Path(f"data/nhl/processed/odds_props_{date_str}.csv")
    if not props_path.exists():
        print(f"[ERROR] Props file not found: {props_path}", file=sys.stderr)
        sys.exit(1)

    props_df = pd.read_csv(props_path)
    print(f"[make_nhl_edges] Loaded {len(props_df)} prop rows")

    # Load consensus
    consensus_path = Path(f"data/nhl/consensus/consensus_props_{date_str}.csv")
    if not consensus_path.exists():
        print(f"[ERROR] Consensus file not found: {consensus_path}", file=sys.stderr)
        sys.exit(1)

    consensus_df = pd.read_csv(consensus_path)
    print(f"[make_nhl_edges] Loaded {len(consensus_df)} consensus groups")

    # Normalize player names for joining
    props_df["name_std"] = props_df["player"].apply(normalize_player_name)

    # Convert odds to probabilities
    props_df["mkt_prob"] = props_df["price"].apply(american_to_prob)

    # Rename "name" column to "side" for joining (Over/Under)
    props_df = props_df.rename(columns={"name": "side"})

    # Join props with consensus on (name_std, market_std, side)
    merged = props_df.merge(
        consensus_df,
        on=["name_std", "market_std", "side"],
        how="left",
        suffixes=("", "_consensus"),
    )

    print(f"[make_nhl_edges] Joined {len(merged)} prop rows with consensus")

    # Compute baseline model probability (Phase A: use consensus as model)
    merged["model_prob"] = merged.apply(
        lambda row: compute_baseline_model_prob(row["consensus_prob"], row["side"]),
        axis=1,
    )

    # Compute fair odds from model probability
    merged["fair_odds"] = merged["model_prob"].apply(prob_to_american)

    # Compute edge in basis points
    merged["edge_bps"] = (merged["model_prob"] - merged["mkt_prob"]) * 10000

    # Add display columns
    merged["model_pct"] = (merged["model_prob"] * 100).round(1)
    merged["consensus_pct"] = (merged["consensus_prob"] * 100).round(1)

    # Model line (for Phase A, use consensus line)
    merged["model_line"] = merged["consensus_line"]

    # Select final columns for output
    output_cols = [
        "sport",
        "game_id",
        "commence_time",
        "home_team",
        "away_team",
        "game",
        "bookmaker",
        "bookmaker_title",
        "market",
        "market_std",
        "player",
        "name_std",
        "side",
        "price",
        "point",
        "mkt_prob",
        "consensus_line",
        "consensus_prob",
        "consensus_pct",
        "book_count",
        "model_line",
        "model_prob",
        "model_pct",
        "fair_odds",
        "edge_bps",
    ]

    # Filter to columns that exist
    output_cols = [c for c in output_cols if c in merged.columns]
    output_df = merged[output_cols].copy()

    # Sort by absolute edge (biggest edges first)
    output_df["abs_edge"] = output_df["edge_bps"].abs()
    output_df = output_df.sort_values("abs_edge", ascending=False).drop(columns=["abs_edge"])

    # Write output
    props_dir = Path("data/nhl/props")
    props_dir.mkdir(parents=True, exist_ok=True)

    output_path = props_dir / f"props_with_model_{date_str}.csv"
    output_df.to_csv(output_path, index=False)

    print(f"[make_nhl_edges] Wrote {len(output_df)} rows to {output_path}")

    # Summary stats
    print(f"\n[make_nhl_edges] Edge Summary:")
    print(f"  Mean edge: {output_df['edge_bps'].mean():.1f} bps")
    print(f"  Median edge: {output_df['edge_bps'].median():.1f} bps")
    print(f"  P95 edge: {output_df['edge_bps'].quantile(0.95):.1f} bps")
    print(f"  Max edge: {output_df['edge_bps'].max():.1f} bps")
    print(f"  Min edge: {output_df['edge_bps'].min():.1f} bps")

    # Coverage check
    missing_consensus = merged["consensus_prob"].isna().sum()
    if missing_consensus > 0:
        pct_missing = 100.0 * missing_consensus / len(merged)
        print(f"\n[WARN] {missing_consensus} props ({pct_missing:.1f}%) missing consensus data", file=sys.stderr)

    print("\n[make_nhl_edges] Done!")


if __name__ == "__main__":
    main()

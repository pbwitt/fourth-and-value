#!/usr/bin/env python3
"""
Compute consensus lines and probabilities for NHL props and game lines.

Groups odds by (player, market, side) for props or (game, market, side) for game lines,
then computes median line and median implied probability across all books.

Inputs:
  - data/nhl/processed/odds_props_{date}.csv
  - data/nhl/processed/odds_games_{date}.csv

Outputs:
  - data/nhl/consensus/consensus_props_{date}.csv
  - data/nhl/consensus/consensus_games_{date}.csv

Usage:
  python3 scripts/nhl/make_nhl_consensus.py --date 2025-10-08
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (no vig)."""
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


def normalize_player_name(name: str) -> str:
    """Normalize player name for grouping."""
    if not name or not isinstance(name, str):
        return ""
    # Simple normalization: lowercase, strip, collapse spaces
    return " ".join(name.strip().lower().split())


def compute_consensus_props(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consensus for player props.

    Groups by (player, market_std, side) and computes:
      - consensus_line: median of point
      - consensus_prob: median of implied probabilities
      - book_count: number of books offering this prop
    """
    if props_df.empty:
        return pd.DataFrame()

    # Normalize player names for grouping
    props_df = props_df.copy()
    props_df["name_std"] = props_df["player"].apply(normalize_player_name)

    # Convert odds to probabilities
    props_df["mkt_prob"] = props_df["price"].apply(american_to_prob)

    # Filter out rows with invalid data
    valid = props_df["name_std"].notna() & props_df["market_std"].notna() & props_df["name"].notna()
    props_df = props_df[valid].copy()

    if props_df.empty:
        return pd.DataFrame()

    # Group by player + market + side
    grouped = props_df.groupby(["name_std", "market_std", "name"], dropna=False).agg({
        "point": "median",
        "mkt_prob": "median",
        "bookmaker": "count",  # book count
    }).reset_index()

    # Rename columns
    grouped = grouped.rename(columns={
        "name": "side",
        "point": "consensus_line",
        "mkt_prob": "consensus_prob",
        "bookmaker": "book_count",
    })

    return grouped


def compute_consensus_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consensus for game lines (moneyline, spreads, totals).

    Groups by (game_id, market_std, side) and computes:
      - consensus_price: median American odds
      - consensus_line: median spread/total (if applicable)
      - consensus_prob: median implied probability
      - book_count: number of books
    """
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df.copy()
    games_df["mkt_prob"] = games_df["price"].apply(american_to_prob)

    # Filter valid rows
    valid = games_df["game_id"].notna() & games_df["market_std"].notna() & games_df["side"].notna()
    games_df = games_df[valid].copy()

    if games_df.empty:
        return pd.DataFrame()

    # Group by game + market + side
    grouped = games_df.groupby(["game_id", "market_std", "side"], dropna=False).agg({
        "price": "median",
        "point": "median",  # spread/total line
        "mkt_prob": "median",
        "bookmaker": "count",
    }).reset_index()

    # Rename columns
    grouped = grouped.rename(columns={
        "price": "consensus_price",
        "point": "consensus_line",
        "mkt_prob": "consensus_prob",
        "bookmaker": "book_count",
    })

    return grouped


def main():
    parser = argparse.ArgumentParser(description="Compute NHL consensus odds")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print(f"[make_nhl_consensus] Computing consensus for {date_str}...")

    # Load odds data
    props_path = Path(f"data/nhl/processed/odds_props_{date_str}.csv")
    games_path = Path(f"data/nhl/processed/odds_games_{date_str}.csv")

    if not props_path.exists():
        print(f"[ERROR] Props file not found: {props_path}", file=sys.stderr)
        sys.exit(1)

    props_df = pd.read_csv(props_path)
    print(f"[make_nhl_consensus] Loaded {len(props_df)} prop rows")

    games_df = pd.DataFrame()
    if games_path.exists():
        games_df = pd.read_csv(games_path)
        print(f"[make_nhl_consensus] Loaded {len(games_df)} game line rows")

    # Compute consensus
    consensus_props = compute_consensus_props(props_df)
    consensus_games = compute_consensus_games(games_df) if not games_df.empty else pd.DataFrame()

    print(f"[make_nhl_consensus] Computed consensus for {len(consensus_props)} prop groups")
    if not consensus_games.empty:
        print(f"[make_nhl_consensus] Computed consensus for {len(consensus_games)} game line groups")

    # Write outputs
    consensus_dir = Path("data/nhl/consensus")
    consensus_dir.mkdir(parents=True, exist_ok=True)

    if not consensus_props.empty:
        props_out = consensus_dir / f"consensus_props_{date_str}.csv"
        consensus_props.to_csv(props_out, index=False)
        print(f"[make_nhl_consensus] Wrote {len(consensus_props)} rows to {props_out}")

    if not consensus_games.empty:
        games_out = consensus_dir / f"consensus_games_{date_str}.csv"
        consensus_games.to_csv(games_out, index=False)
        print(f"[make_nhl_consensus] Wrote {len(consensus_games)} rows to {games_out}")

    print("[make_nhl_consensus] Done!")


if __name__ == "__main__":
    main()

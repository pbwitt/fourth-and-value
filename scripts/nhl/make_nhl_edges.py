#!/usr/bin/env python3
"""
Compute edges for NHL props by joining odds + consensus + ML models.

Phase B: Uses trained models for SOG (Goals/Assists/Points fall back to consensus).

Inputs:
  - data/nhl/processed/odds_props_{date}.csv
  - data/nhl/consensus/consensus_props_{date}.csv
  - data/nhl/models/sog_model_latest.pkl (if available)

Outputs:
  - data/nhl/props/props_with_model_{date}.csv

Usage:
  python3 scripts/nhl/make_nhl_edges.py --date 2025-10-08
"""

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# Import shared model classes
sys.path.insert(0, str(Path(__file__).parent))
from nhl_models import SimpleSOGModel, SimpleScoringModel


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


def load_models():
    """Load trained models if available."""
    models = {}

    # Load SOG model
    sog_model_path = Path("data/nhl/models/sog_model_latest.pkl")
    if sog_model_path.exists():
        with open(sog_model_path, "rb") as f:
            models["sog"] = pickle.load(f)
        print(f"[make_nhl_edges] Loaded SOG model from {sog_model_path}")
    else:
        print(f"[make_nhl_edges] SOG model not found, using consensus baseline")

    # Load scoring models (goals, assists, points)
    for stat in ["goals", "assists", "points"]:
        model_path = Path(f"data/nhl/models/{stat}_model_latest.pkl")
        if model_path.exists():
            with open(model_path, "rb") as f:
                models[stat] = pickle.load(f)
            print(f"[make_nhl_edges] Loaded {stat} model from {model_path}")
        else:
            print(f"[make_nhl_edges] {stat.capitalize()} model not found, using consensus baseline")

    return models


def compute_model_line(row, models):
    """
    Compute model's expected value (prediction).

    Returns the model's point estimate for the stat.
    """
    market_std = row.get("market_std")
    player = row.get("player")

    # Check if we have a model for this market
    if market_std in models:
        model = models[market_std]
        try:
            # Get the model's expected value
            if hasattr(model, 'predict_expected_value'):
                return model.predict_expected_value(player)
            elif hasattr(model, 'skater_stats') and model.skater_stats is not None:
                # Fallback: get mean from model stats
                player_norm = " ".join(player.strip().lower().split())
                if player_norm in model.skater_stats.index:
                    stats = model.skater_stats.loc[player_norm]
                    if market_std == "sog":
                        return stats.get("shots_per_game", np.nan)
                    else:
                        return stats.get(f"{market_std}_per_game", np.nan)
        except Exception:
            pass

    # Fallback: use consensus line
    return row.get("consensus_line", np.nan)


def compute_model_prob(row, models, consensus_prob):
    """
    Compute model probability for a prop.

    Phase C: Use trained models for all markets (SOG, goals, assists, points).
    Falls back to consensus if model not available.
    """
    market_std = row.get("market_std")
    player = row.get("player")
    side = row.get("side")
    line = row.get("point")

    # Check if we have a model for this market
    if market_std in models:
        model = models[market_std]
        try:
            # Use UNCALIBRATED probabilities for edge calculation
            # We want our raw model opinion vs market, not calibrated-to-match-market
            if side == "Over":
                prob = model.predict_prob_over(player, line, calibrated=False)
            else:  # Under
                prob = 1 - model.predict_prob_over(player, line, calibrated=False)
            return prob
        except Exception as e:
            # Fallback to consensus on error
            print(f"[warn] Model prediction failed for {player} {market_std}: {e}", file=sys.stderr)
            return consensus_prob

    # Fallback to consensus if no model available
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

    # Load trained models
    models = load_models()

    # Compute model lines (expected values)
    merged["model_line"] = merged.apply(
        lambda row: compute_model_line(row, models),
        axis=1,
    )

    # Compute model probabilities (Phase B: use models where available)
    merged["model_prob"] = merged.apply(
        lambda row: compute_model_prob(row, models, row["consensus_prob"]),
        axis=1,
    )

    # Compute fair odds from model probability
    merged["fair_odds"] = merged["model_prob"].apply(prob_to_american)

    # Compute edge in basis points
    merged["edge_bps"] = (merged["model_prob"] - merged["mkt_prob"]) * 10000

    # Add display columns
    merged["model_pct"] = (merged["model_prob"] * 100).round(1)
    merged["consensus_pct"] = (merged["consensus_prob"] * 100).round(1)

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

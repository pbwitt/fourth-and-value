#!/usr/bin/env python3
"""
Train SOG (Shots on Goal) model for NHL props.

Phase B approach:
- Uses season summary stats from NHL API (TOI, shots per game)
- Simple Poisson/Normal distribution baseline
- Future: Add game logs, rolling averages, opponent defense metrics

Features:
- TOI per game (proxy for ice time)
- Season shots per game average
- Position (C/W/D)
- Team offensive rating (future)

Outputs:
- data/nhl/models/sog_model_{date}.pkl
- data/nhl/models/sog_model_latest.pkl (symlink)

Usage:
  python3 scripts/nhl/train_sog_model.py --date 2025-10-08
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
from nhl_models import SimpleSOGModel


def main():
    parser = argparse.ArgumentParser(description="Train NHL SOG model")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print(f"[train_sog_model] Training SOG model for {date_str}...")

    # Load skater stats
    skater_path = Path(f"data/nhl/processed/skater_logs_{date_str}.parquet")
    if not skater_path.exists():
        print(f"[ERROR] Skater logs not found: {skater_path}", file=sys.stderr)
        sys.exit(1)

    skater_df = pd.read_parquet(skater_path)
    print(f"[train_sog_model] Loaded {len(skater_df)} skaters")

    # Normalize player names for matching
    skater_df["player"] = skater_df["player"].apply(
        lambda x: " ".join(x.strip().lower().split())
    )

    # Train model
    model = SimpleSOGModel()
    model.fit(skater_df)

    print(f"[train_sog_model] Model fitted on {len(model.skater_stats)} players")

    # Save model
    models_dir = Path("data/nhl/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"sog_model_{date_str}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[train_sog_model] Saved model to {model_path}")

    # Create symlink to latest
    latest_path = models_dir / "sog_model_latest.pkl"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(model_path.name)

    print(f"[train_sog_model] Created symlink: {latest_path} -> {model_path.name}")

    # Test predictions on a few players
    print("\n[train_sog_model] Sample predictions:")
    test_players = skater_df.head(5)["player"].tolist()
    for player in test_players:
        prob_over_3 = model.predict_prob_over(player, 3.5)
        print(f"  {player:30s} P(SOG > 3.5) = {prob_over_3:.3f}")

    print("\n[train_sog_model] Done!")


if __name__ == "__main__":
    main()

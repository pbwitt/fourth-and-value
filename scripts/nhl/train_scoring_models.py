#!/usr/bin/env python3
"""
Train Goals/Assists/Points models for NHL props.

Uses Poisson distribution based on season rates:
- Goals: Low-frequency events (0.5 line common)
- Assists: Similar to goals
- Points: Sum of goals + assists

Outputs:
- data/nhl/models/goals_model_{date}.pkl
- data/nhl/models/assists_model_{date}.pkl
- data/nhl/models/points_model_{date}.pkl
- Symlinks to *_latest.pkl

Usage:
  python3 scripts/nhl/train_scoring_models.py --date 2025-10-08
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
from nhl_models import SimpleScoringModel


def main():
    parser = argparse.ArgumentParser(description="Train NHL scoring models")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print(f"[train_scoring_models] Training scoring models for {date_str}...")

    # Load skater stats
    skater_path = Path(f"data/nhl/processed/skater_logs_{date_str}.parquet")
    if not skater_path.exists():
        print(f"[ERROR] Skater logs not found: {skater_path}", file=sys.stderr)
        sys.exit(1)

    skater_df = pd.read_parquet(skater_path)
    print(f"[train_scoring_models] Loaded {len(skater_df)} skaters")

    # Normalize player names for matching
    skater_df["player"] = skater_df["player"].apply(
        lambda x: " ".join(x.strip().lower().split())
    )

    # Ensure directories exist
    models_dir = Path("data/nhl/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train models for each stat
    for stat_name in ["goals", "assists", "points"]:
        print(f"\n[train_scoring_models] Training {stat_name} model...")

        model = SimpleScoringModel(stat_name=stat_name)
        model.fit(skater_df)

        print(f"[train_scoring_models] {stat_name.capitalize()} model fitted on {len(model.skater_stats)} players")

        # Save model
        model_path = models_dir / f"{stat_name}_model_{date_str}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"[train_scoring_models] Saved to {model_path}")

        # Create symlink to latest
        latest_path = models_dir / f"{stat_name}_model_latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(model_path.name)

        print(f"[train_scoring_models] Created symlink: {latest_path} -> {model_path.name}")

        # Test predictions
        test_players = skater_df.head(3)["player"].tolist()
        print(f"[train_scoring_models] Sample predictions for {stat_name}:")
        for player in test_players:
            prob_over_05 = model.predict_prob_over(player, 0.5)
            print(f"  {player:30s} P({stat_name} > 0.5) = {prob_over_05:.3f}")

    print("\n[train_scoring_models] All models trained!")


if __name__ == "__main__":
    main()

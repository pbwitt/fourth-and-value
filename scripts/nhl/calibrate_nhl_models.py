#!/usr/bin/env python3
"""
Calibrate NHL models using isotonic regression.

Uses consensus probabilities from market odds as calibration targets.
Fits isotonic regression: calibrated_prob = f(model_prob)

This improves probability estimates by aligning model outputs with market consensus.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import pickle
import pandas as pd
from datetime import datetime

from nhl_models import SimpleSOGModel, SimpleScoringModel


def load_consensus_props(date_str: str) -> pd.DataFrame:
    """Load consensus props with market probabilities."""
    props_path = Path(f"data/nhl/consensus/consensus_props_{date_str}.csv")
    if not props_path.exists():
        raise FileNotFoundError(f"Consensus props not found: {props_path}")

    props_df = pd.read_csv(props_path)

    # Rename columns to match model expectations
    props_df = props_df.rename(columns={
        "name_std": "player",
        "consensus_line": "point",
    })

    # Normalize player names for matching
    props_df["player"] = props_df["player"].apply(
        lambda x: " ".join(x.strip().lower().split())
    )

    # Filter to markets we have models for
    valid_markets = ["sog", "goals", "assists", "points"]
    props_df = props_df[props_df["market_std"].isin(valid_markets)].copy()

    print(f"Loaded {len(props_df)} consensus props for calibration")
    print(f"Markets: {props_df['market_std'].value_counts().to_dict()}")

    return props_df


def calibrate_model(model, props_df: pd.DataFrame, market_std: str) -> int:
    """
    Calibrate a single model using props for its market.

    Args:
        model: Model to calibrate (SimpleSOGModel or SimpleScoringModel)
        props_df: All consensus props
        market_std: Market standard name ("sog", "goals", etc.)

    Returns:
        Number of props used for calibration
    """
    # Filter to this model's market
    market_props = props_df[props_df["market_std"] == market_std].copy()

    if len(market_props) == 0:
        print(f"  WARNING: No props found for {market_std}")
        return 0

    # Calibrate model
    model.calibrate(market_props)

    print(f"  Calibrated {market_std} model on {len(market_props)} props")

    return len(market_props)


def main():
    parser = argparse.ArgumentParser(description="Calibrate NHL models")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date string (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print(f"Calibrating NHL models for {date_str}")

    # Paths
    models_dir = Path("data/nhl/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load consensus props
    props_df = load_consensus_props(date_str)

    # Calibrate SOG model
    print("\n=== SOG Model ===")
    sog_model_path = models_dir / "sog_model_latest.pkl"
    if not sog_model_path.exists():
        print(f"ERROR: SOG model not found at {sog_model_path}")
        print("Run train_sog_model.py first")
        sys.exit(1)

    with open(sog_model_path, "rb") as f:
        sog_model = pickle.load(f)

    n_sog = calibrate_model(sog_model, props_df, "sog")

    # Save calibrated SOG model
    sog_output_path = models_dir / f"sog_model_{date_str}.pkl"
    with open(sog_output_path, "wb") as f:
        pickle.dump(sog_model, f)

    # Update symlink
    sog_latest_path = models_dir / "sog_model_latest.pkl"
    if sog_latest_path.exists():
        sog_latest_path.unlink()
    sog_latest_path.symlink_to(sog_output_path.name)

    # Calibrate scoring models
    for stat_name in ["goals", "assists", "points"]:
        print(f"\n=== {stat_name.title()} Model ===")

        model_path = models_dir / f"{stat_name}_model_latest.pkl"
        if not model_path.exists():
            print(f"ERROR: {stat_name} model not found at {model_path}")
            print("Run train_scoring_models.py first")
            sys.exit(1)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        n_props = calibrate_model(model, props_df, stat_name)

        # Save calibrated model
        output_path = models_dir / f"{stat_name}_model_{date_str}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(model, f)

        # Update symlink
        latest_path = models_dir / f"{stat_name}_model_latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(output_path.name)

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print(f"Calibrated models saved to {models_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

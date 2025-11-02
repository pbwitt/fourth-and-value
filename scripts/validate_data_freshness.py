#!/usr/bin/env python3
"""
Validate that props data is fresh (games haven't already happened).

This prevents publishing stale data to production.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd


def validate_props_freshness(props_csv: str, max_age_hours: int = 48) -> bool:
    """
    Check if props data is fresh.

    Returns:
        True if data is fresh, False if stale
    """
    props_path = Path(props_csv)

    if not props_path.exists():
        print(f"❌ ERROR: Props file not found: {props_csv}", file=sys.stderr)
        return False

    # Load props
    df = pd.read_csv(props_path)

    if 'commence_time' not in df.columns:
        print(f"❌ ERROR: No 'commence_time' column in props data", file=sys.stderr)
        return False

    # Parse game times
    df['commence_dt'] = pd.to_datetime(df['commence_time'])
    now = datetime.now(timezone.utc)

    earliest_game = df['commence_dt'].min()
    latest_game = df['commence_dt'].max()

    days_until_earliest = (earliest_game - now).total_seconds() / 86400
    days_until_latest = (latest_game - now).total_seconds() / 86400

    print("\n" + "="*70)
    print("DATA FRESHNESS VALIDATION")
    print("="*70)
    print(f"Current time:     {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Earliest game:    {earliest_game.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Latest game:      {latest_game.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Days until games: {days_until_earliest:.1f} to {days_until_latest:.1f}")

    # Check if any games have already happened
    if days_until_earliest < -0.5:  # Allow 12 hours grace period
        print("\n🚨 CRITICAL ERROR: Props contain games that already happened!")
        print(f"   Earliest game was {abs(days_until_earliest):.1f} days ago")
        print("   This data is STALE and cannot be published.")
        print("="*70 + "\n")
        return False

    # Check if games are too far in the future (wrong week)
    if days_until_earliest > 7:
        print(f"\n⚠️  WARNING: Earliest game is {days_until_earliest:.1f} days away")
        print("   This might be the wrong week.")
        print("="*70 + "\n")
        return False

    # Check if data is within the time window
    if days_until_latest < 0:
        print(f"\n⚠️  WARNING: Latest game has passed")
    elif days_until_earliest < 1:
        print(f"\n✅ DATA IS FRESH: Games start in {days_until_earliest*24:.1f} hours")
    else:
        print(f"\n✅ DATA IS FRESH: Games in {days_until_earliest:.1f} to {days_until_latest:.1f} days")

    print("="*70 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate props data freshness")
    parser.add_argument("--props", default="data/props/latest_all_props.csv",
                        help="Props CSV file to validate")
    parser.add_argument("--max-age-hours", type=int, default=48,
                        help="Maximum age of data in hours")
    args = parser.parse_args()

    is_fresh = validate_props_freshness(args.props, args.max_age_hours)

    if not is_fresh:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

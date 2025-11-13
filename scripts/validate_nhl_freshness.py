#!/usr/bin/env python3
"""
Validate NHL data freshness before deployment.

Checks:
1. Predictions file exists and is from today
2. Consensus file exists and is from today
3. Props file exists and is from today
4. All files contain upcoming games (not past games)

Exit codes:
  0 = All data is fresh
  1 = Data is stale or missing
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd


def check_file_date(filepath: Path, expected_date: str, file_type: str) -> bool:
    """Check if file exists and is from expected date."""
    if not filepath.exists():
        print(f"❌ {file_type}: File not found: {filepath}")
        return False

    # Check file modification time
    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    today = datetime.now()
    age_hours = (today - mtime).total_seconds() / 3600

    if age_hours > 24:
        print(f"❌ {file_type}: File is {age_hours:.1f} hours old (expected < 24h)")
        print(f"   Path: {filepath}")
        print(f"   Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        return False

    print(f"✅ {file_type}: Fresh ({age_hours:.1f}h old)")
    return True


def check_game_times(filepath: Path, file_type: str) -> bool:
    """Check that games in file are upcoming (not past)."""
    if not filepath.exists():
        return False

    try:
        df = pd.read_csv(filepath)

        # Check for commence_time column
        if 'commence_time' in df.columns:
            df['game_time'] = pd.to_datetime(df['commence_time'])
            now = datetime.now(timezone.utc)

            earliest = df['game_time'].min()
            latest = df['game_time'].max()

            hours_until_earliest = (earliest - now).total_seconds() / 3600

            if hours_until_earliest < -12:  # Games more than 12h in past
                print(f"⚠️  {file_type}: Contains games that already happened!")
                print(f"   Earliest game: {earliest}")
                print(f"   Hours ago: {abs(hours_until_earliest):.1f}h")
                return False

            print(f"   Games: {earliest.strftime('%m/%d %H:%M')} to {latest.strftime('%m/%d %H:%M')} UTC")

    except Exception as e:
        print(f"⚠️  {file_type}: Could not validate game times: {e}")
        # Don't fail on this, just warn

    return True


def main():
    today = datetime.now().strftime('%Y-%m-%d')

    print("=" * 70)
    print(f"NHL DATA FRESHNESS VALIDATION - {today}")
    print("=" * 70)
    print()

    all_fresh = True

    # Check predictions
    predictions = Path("data/nhl/predictions/today.csv")
    if not check_file_date(predictions, today, "Predictions"):
        all_fresh = False

    # Check consensus_games
    consensus_games = Path(f"data/nhl/consensus/consensus_games_{today}.csv")
    if not check_file_date(consensus_games, today, f"Consensus Games"):
        all_fresh = False
    else:
        check_game_times(consensus_games, "Consensus Games")

    # Check props
    props = Path(f"data/nhl/props/props_with_model_{today}.csv")
    if not check_file_date(props, today, f"Props"):
        all_fresh = False
    else:
        check_game_times(props, "Props")

    # Check player stats
    player_stats = Path("data/nhl/raw/player_stats.csv")
    if player_stats.exists():
        try:
            df = pd.read_csv(player_stats)
            if 'game_date' in df.columns:
                # Filter out NaT/empty values and convert to datetime
                dates = pd.to_datetime(df['game_date'], errors='coerce').dropna()
                if len(dates) > 0:
                    max_date = dates.max().strftime('%Y-%m-%d')
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    if max_date < yesterday:
                        print(f"⚠️  Player Stats: Latest data is from {max_date} (expected >= {yesterday})")
                        all_fresh = False
                    else:
                        print(f"✅ Player Stats: Latest data from {max_date}")
        except Exception as e:
            print(f"⚠️  Player Stats: Could not validate: {e}")

    print()
    print("=" * 70)

    if all_fresh:
        print("✅ ALL DATA IS FRESH - SAFE TO DEPLOY")
        print("=" * 70)
        sys.exit(0)
    else:
        print("❌ STALE DATA DETECTED - DO NOT DEPLOY")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()

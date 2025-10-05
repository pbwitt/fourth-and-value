#!/usr/bin/env python3
"""
fetch_defensive_stats.py

Fetch and compute opponent defensive strength metrics for adjusting player prop predictions.

Outputs CSV with columns:
    - team: Defensive team
    - pass_def_rating: How tough vs passing (lower = easier matchup, scale 0-2)
    - rush_def_rating: How tough vs rushing (lower = easier matchup, scale 0-2)
    - recv_def_rating: How tough vs receiving (lower = easier matchup, scale 0-2)

Usage:
    python scripts/fetch_defensive_stats.py --season 2025 --week 5 --out data/defensive_stats_2025.csv
"""

import argparse
import pandas as pd
import numpy as np

try:
    import nfl_data_py as nfl
    NFL_OK = True
except Exception:
    NFL_OK = False


def fetch_defensive_stats(season: int, week: int) -> pd.DataFrame:
    """
    Fetch defensive stats through specified week and compute difficulty ratings.

    Returns DataFrame with team-level defensive ratings (0-2 scale, 1.0 = league average)
    """
    if not NFL_OK:
        print("[WARN] nfl_data_py not available, using neutral defensive ratings")
        return pd.DataFrame({
            'team': [],
            'pass_def_rating': [],
            'rush_def_rating': [],
            'recv_def_rating': []
        })

    # Fetch weekly data for the season up to specified week
    try:
        df = nfl.import_weekly_data([season], ['QB', 'RB', 'WR', 'TE'])
    except Exception as e:
        print(f"[WARN] Could not fetch NFL data: {e}")
        return pd.DataFrame({
            'team': [],
            'pass_def_rating': [],
            'rush_def_rating': [],
            'recv_def_rating': []
        })

    # Filter to completed weeks only
    df = df[df['week'] < week].copy()

    if df.empty:
        print(f"[WARN] No data available for season {season} before week {week}")
        return pd.DataFrame({
            'team': [],
            'pass_def_rating': [],
            'rush_def_rating': [],
            'recv_def_rating': []
        })

    # Group by opponent team (defensive team) and calculate yards/attempts allowed
    defense_stats = []

    for team in df['opponent_team'].unique():
        if pd.isna(team):
            continue

        opp_df = df[df['opponent_team'] == team].copy()

        # Passing defense: yards per attempt allowed
        pass_attempts = opp_df['attempts'].sum()
        pass_yards = opp_df['passing_yards'].sum()
        pass_ypa = pass_yards / pass_attempts if pass_attempts > 0 else np.nan

        # Rushing defense: yards per carry allowed
        rush_attempts = opp_df['carries'].sum() if 'carries' in opp_df.columns else 0
        rush_yards = opp_df['rushing_yards'].sum()
        rush_ypc = rush_yards / rush_attempts if rush_attempts > 0 else np.nan

        # Receiving defense: yards per reception allowed
        receptions = opp_df['receptions'].sum()
        recv_yards = opp_df['receiving_yards'].sum()
        recv_ypr = recv_yards / receptions if receptions > 0 else np.nan

        defense_stats.append({
            'team': team,
            'pass_ypa_allowed': pass_ypa,
            'rush_ypc_allowed': rush_ypc,
            'recv_ypr_allowed': recv_ypr,
        })

    def_df = pd.DataFrame(defense_stats)

    # Convert to difficulty ratings (relative to league average)
    # Higher yards allowed = easier matchup = lower rating
    # Scale so 1.0 = league average, 0.5 = easiest, 2.0 = toughest

    def normalize_to_rating(series):
        """Convert stat to 0-2 scale where 1.0 = average, higher = tougher defense"""
        if series.isna().all() or len(series) == 0:
            return pd.Series(1.0, index=series.index)

        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            return pd.Series(1.0, index=series.index)

        # Invert: higher yards allowed = easier = lower rating
        z_scores = -(series - mean) / std  # negative so high yards = low rating

        # Scale to 0.5-2.0 range (3 std = full range)
        ratings = 1.0 + (z_scores / 3.0) * 0.75
        return ratings.clip(0.5, 2.0)

    def_df['pass_def_rating'] = normalize_to_rating(def_df['pass_ypa_allowed'])
    def_df['rush_def_rating'] = normalize_to_rating(def_df['rush_ypc_allowed'])
    def_df['recv_def_rating'] = normalize_to_rating(def_df['recv_ypr_allowed'])

    # Keep only team and ratings
    result = def_df[['team', 'pass_def_rating', 'rush_def_rating', 'recv_def_rating']].copy()

    print(f"[defensive stats] Computed ratings for {len(result)} teams")
    print(f"[defensive stats] Pass def range: {result['pass_def_rating'].min():.2f} - {result['pass_def_rating'].max():.2f}")
    print(f"[defensive stats] Rush def range: {result['rush_def_rating'].min():.2f} - {result['rush_def_rating'].max():.2f}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default="data/defensive_stats.csv")
    args = ap.parse_args()

    df = fetch_defensive_stats(args.season, args.week)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()

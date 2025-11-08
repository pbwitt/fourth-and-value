"""
Build team-level features for NFL totals from play-by-play data
Similar to NHL team features but using EPA and other NFL-specific metrics
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

def build_team_features(pbp_path='data/pbp/pbp_2025.parquet', output_path='data/nfl/processed/team_features.csv'):
    """
    Aggregate play-by-play data to team-level features
    """
    print("Loading play-by-play data...")
    pbp = pd.read_parquet(pbp_path)

    # Get game-level summary first
    games = pbp.groupby(['game_id', 'home_team', 'away_team', 'week', 'game_date']).agg({
        'total': 'first',
        'total_line': 'first',
        'home_score': 'last',
        'away_score': 'last'
    }).reset_index()

    print(f"Found {len(games)} games")

    # Build offensive features for each team
    team_games = []

    for _, game in games.iterrows():
        game_pbp = pbp[pbp['game_id'] == game['game_id']]

        # Home team offense
        home_off = game_pbp[game_pbp['posteam'] == game['home_team']]
        home_features = {
            'game_id': game['game_id'],
            'team': game['home_team'],
            'opponent': game['away_team'],
            'week': game['week'],
            'game_date': game['game_date'],
            'home_away': 'home',
            'points_scored': game['home_score'],
            'points_allowed': game['away_score'],
            'total': game['total'],
            'total_line': game['total_line'],

            # Offensive metrics
            'off_plays': len(home_off),
            'off_epa_per_play': home_off['epa'].mean() if len(home_off) > 0 else 0,
            'off_success_rate': home_off['success'].mean() if len(home_off) > 0 else 0,
            'off_pass_epa': home_off[home_off['pass'] == 1]['epa'].mean() if len(home_off[home_off['pass'] == 1]) > 0 else 0,
            'off_rush_epa': home_off[home_off['rush'] == 1]['epa'].mean() if len(home_off[home_off['rush'] == 1]) > 0 else 0,
            'off_explosive_play_rate': (home_off['epa'] > 1.0).mean() if len(home_off) > 0 else 0,
            'off_third_down_conv': home_off[home_off['down'] == 3]['third_down_converted'].mean() if len(home_off[home_off['down'] == 3]) > 0 else 0,
            'off_red_zone_td_rate': home_off[(home_off['yardline_100'] <= 20) & (home_off['touchdown'] == 1)].shape[0] / max(1, home_off[home_off['yardline_100'] <= 20].shape[0]) if len(home_off[home_off['yardline_100'] <= 20]) > 0 else 0,
        }

        # Defensive metrics (when this team is on defense)
        home_def = game_pbp[game_pbp['defteam'] == game['home_team']]
        home_features.update({
            'def_epa_per_play': home_def['epa'].mean() if len(home_def) > 0 else 0,
            'def_success_rate': home_def['success'].mean() if len(home_def) > 0 else 0,
            'def_sacks': home_def['sack'].sum(),
            'def_turnovers': (home_def['interception'].sum() + home_def['fumble_lost'].sum()),
        })

        team_games.append(home_features)

        # Away team offense
        away_off = game_pbp[game_pbp['posteam'] == game['away_team']]
        away_features = {
            'game_id': game['game_id'],
            'team': game['away_team'],
            'opponent': game['home_team'],
            'week': game['week'],
            'game_date': game['game_date'],
            'home_away': 'away',
            'points_scored': game['away_score'],
            'points_allowed': game['home_score'],
            'total': game['total'],
            'total_line': game['total_line'],

            # Offensive metrics
            'off_plays': len(away_off),
            'off_epa_per_play': away_off['epa'].mean() if len(away_off) > 0 else 0,
            'off_success_rate': away_off['success'].mean() if len(away_off) > 0 else 0,
            'off_pass_epa': away_off[away_off['pass'] == 1]['epa'].mean() if len(away_off[away_off['pass'] == 1]) > 0 else 0,
            'off_rush_epa': away_off[away_off['rush'] == 1]['epa'].mean() if len(away_off[away_off['rush'] == 1]) > 0 else 0,
            'off_explosive_play_rate': (away_off['epa'] > 1.0).mean() if len(away_off) > 0 else 0,
            'off_third_down_conv': away_off[away_off['down'] == 3]['third_down_converted'].mean() if len(away_off[away_off['down'] == 3]) > 0 else 0,
            'off_red_zone_td_rate': away_off[(away_off['yardline_100'] <= 20) & (away_off['touchdown'] == 1)].shape[0] / max(1, away_off[away_off['yardline_100'] <= 20].shape[0]) if len(away_off[away_off['yardline_100'] <= 20]) > 0 else 0,
        }

        # Defensive metrics
        away_def = game_pbp[game_pbp['defteam'] == game['away_team']]
        away_features.update({
            'def_epa_per_play': away_def['epa'].mean() if len(away_def) > 0 else 0,
            'def_success_rate': away_def['success'].mean() if len(away_def) > 0 else 0,
            'def_sacks': away_def['sack'].sum(),
            'def_turnovers': (away_def['interception'].sum() + away_def['fumble_lost'].sum()),
        })

        team_games.append(away_features)

    df = pd.DataFrame(team_games)
    df = df.sort_values(['team', 'week'])

    print(f"\nBuilt features for {len(df)} team-game records")

    # Add rolling averages (L3, L5)
    print("Adding rolling features...")
    df = add_rolling_features(df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    return df


def add_rolling_features(df, windows=[3, 5]):
    """
    Add rolling window features for each team
    """
    feature_cols = [
        'off_epa_per_play', 'off_success_rate', 'off_pass_epa', 'off_rush_epa',
        'off_explosive_play_rate', 'off_third_down_conv', 'off_red_zone_td_rate',
        'def_epa_per_play', 'def_success_rate', 'points_scored', 'points_allowed'
    ]

    for team in df['team'].unique():
        team_mask = df['team'] == team
        team_df = df[team_mask].copy().sort_values('week')

        for window in windows:
            for col in feature_cols:
                # Rolling mean (excluding current game)
                df.loc[team_mask, f'{col}_L{window}'] = team_df[col].shift(1).rolling(window, min_periods=1).mean().values

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build NFL team features from PBP')
    parser.add_argument('--pbp', default='data/pbp/pbp_2025.parquet', help='PBP parquet file')
    parser.add_argument('--output', default='data/nfl/processed/team_features.csv', help='Output CSV')

    args = parser.parse_args()

    build_team_features(args.pbp, args.output)

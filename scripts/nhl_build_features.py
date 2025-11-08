"""
Build team-level features from player stats for NHL totals modeling
This is the KEY script - aggregates player stats properly for team total predictions
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os


def aggregate_player_stats_to_team(player_stats_df):
    """
    Aggregate player-level stats to team-level for each game
    Creates ONE row per team per game with aggregated features
    """
    # Separate by position type
    forwards = player_stats_df[player_stats_df['position_type'] == 'F'].copy()
    defense = player_stats_df[player_stats_df['position_type'] == 'D'].copy()
    goalies = player_stats_df[player_stats_df['position_type'] == 'G'].copy()

    # Aggregate forwards
    forward_agg = forwards.groupby(['game_id', 'game_date', 'team', 'opponent', 'home_away']).agg({
        'goals': ['sum', 'mean', 'max'],  # Total goals, avg per forward, top scorer
        'assists': ['sum', 'mean'],
        'points': ['sum', 'mean', 'max'],
        'shots': ['sum', 'mean'],
        'hits': 'sum',
        'pp_goals': 'sum',
        'toi_seconds': 'mean',
        'faceoff_pct': 'mean',
        'takeaways': 'sum',
        'giveaways': 'sum',
        'blocked_shots': 'sum'
    })
    forward_agg.columns = ['_'.join(col).strip() for col in forward_agg.columns.values]
    forward_agg = forward_agg.add_prefix('fwd_')
    forward_agg = forward_agg.reset_index()

    # Aggregate defense
    defense_agg = defense.groupby(['game_id', 'game_date', 'team', 'opponent', 'home_away']).agg({
        'goals': 'sum',
        'assists': 'sum',
        'points': 'sum',
        'shots': 'sum',
        'hits': 'sum',
        'blocked_shots': 'sum',
        'toi_seconds': 'mean',
        'plus_minus': 'mean'
    })
    defense_agg.columns = ['_'.join(col).strip() for col in defense_agg.columns.values]
    defense_agg = defense_agg.add_prefix('def_')
    defense_agg = defense_agg.reset_index()

    # Get starting goalie stats
    starters = goalies[goalies['starter'] == True].copy()
    goalie_agg = starters.groupby(['game_id', 'game_date', 'team', 'opponent', 'home_away']).agg({
        'saves': 'first',
        'shots_against': 'first',
        'goals_against': 'first',
        'save_pct': 'first',
        'decision': 'first'
    })
    goalie_agg.columns = ['_'.join(col).strip() for col in goalie_agg.columns.values]
    goalie_agg = goalie_agg.add_prefix('goalie_')
    goalie_agg = goalie_agg.reset_index()

    # Merge all together
    team_features = forward_agg.merge(
        defense_agg, on=['game_id', 'game_date', 'team', 'opponent', 'home_away'], how='left'
    ).merge(
        goalie_agg, on=['game_id', 'game_date', 'team', 'opponent', 'home_away'], how='left'
    )

    # Calculate total team goals
    team_features['goals_for'] = team_features['fwd_goals_sum'] + team_features['def_goals_sum']

    return team_features


def add_rolling_features(team_df, windows=[5, 10]):
    """
    Add rolling window features (like your NFL model)
    Exponential weighting for recent games
    """
    team_df = team_df.sort_values(['team', 'game_date']).copy()

    for window in windows:
        # Goals rolling average
        team_df[f'goals_l{window}'] = team_df.groupby('team')['goals_for'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

        # Shots rolling average
        team_df[f'shots_l{window}'] = team_df.groupby('team')['fwd_shots_sum'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

        # PP goals rolling average
        team_df[f'pp_goals_l{window}'] = team_df.groupby('team')['fwd_pp_goals_sum'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

        # Save pct rolling average
        team_df[f'save_pct_l{window}'] = team_df.groupby('team')['goalie_save_pct_first'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

    # Exponential weighted moving average (more weight on recent games)
    team_df['goals_ewm'] = team_df.groupby('team')['goals_for'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean().shift(1)
    )

    team_df['shots_ewm'] = team_df.groupby('team')['fwd_shots_sum'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean().shift(1)
    )

    return team_df


def add_opponent_features(team_df):
    """
    Add opponent defensive stats
    """
    # Create defensive metrics
    team_df['goals_against_per_game'] = team_df['goalie_goals_against_first']

    # Calculate opponent's defensive strength
    opp_defense = team_df.groupby('team').agg({
        'goals_against_per_game': 'mean',
        'goalie_save_pct_first': 'mean',
        'def_blocked_shots_sum': 'mean'
    }).add_prefix('opp_allows_')

    opp_defense = opp_defense.reset_index().rename(columns={'team': 'opponent'})

    # Merge opponent stats
    team_df = team_df.merge(opp_defense, on='opponent', how='left')

    return team_df


def add_home_away_splits(team_df):
    """
    Calculate home vs away performance
    """
    # Home stats
    home_stats = team_df[team_df['home_away'] == 'home'].groupby('team').agg({
        'goals_for': 'mean'
    }).add_prefix('home_avg_')
    home_stats = home_stats.reset_index()

    # Away stats
    away_stats = team_df[team_df['home_away'] == 'away'].groupby('team').agg({
        'goals_for': 'mean'
    }).add_prefix('away_avg_')
    away_stats = away_stats.reset_index()

    # Merge splits
    team_df = team_df.merge(home_stats, on='team', how='left')
    team_df = team_df.merge(away_stats, on='team', how='left')

    # Fill NaN for teams with no home/away games yet
    team_df['home_avg_goals_for'] = team_df['home_avg_goals_for'].fillna(team_df['goals_for'].mean())
    team_df['away_avg_goals_for'] = team_df['away_avg_goals_for'].fillna(team_df['goals_for'].mean())

    return team_df


def build_training_data(player_stats_path, output_path='data/nhl/processed/team_features.csv'):
    """
    Main pipeline: player stats → team features with rolling windows
    """
    print(f"Loading player stats from {player_stats_path}...")
    player_df = pd.read_csv(player_stats_path)

    print(f"Loaded {len(player_df)} player-game records")

    print("\n1. Aggregating player stats to team level...")
    team_df = aggregate_player_stats_to_team(player_df)
    print(f"Created {len(team_df)} team-game records")

    print("\n2. Adding rolling window features...")
    team_df = add_rolling_features(team_df, windows=[5, 10])

    print("\n3. Adding opponent features...")
    team_df = add_opponent_features(team_df)

    print("\n4. Adding home/away splits...")
    team_df = add_home_away_splits(team_df)

    print("\n5. Creating binary home/away indicator...")
    team_df['is_home'] = (team_df['home_away'] == 'home').astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    team_df.to_csv(output_path, index=False)

    print(f"\nSaved team features to {output_path}")
    print(f"Shape: {team_df.shape}")
    print(f"\nColumns: {team_df.columns.tolist()}")

    return team_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build NHL team features from player stats')
    parser.add_argument('--input', default='data/nhl/raw/player_stats.csv', help='Input player stats CSV')
    parser.add_argument('--output', default='data/nhl/processed/team_features.csv', help='Output file')

    args = parser.parse_args()

    build_training_data(args.input, args.output)

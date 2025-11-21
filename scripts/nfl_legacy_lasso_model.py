"""
Recreate the legacy Lasso regression model for NFL totals
Uses individual player stats aggregated to team level
Based on PDubbs Over-Under Predictions-9-25-21.ipynb
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import nfl_data_py as nfl
from datetime import datetime

print("Fetching historical player data from nfl_data_py...")
print("This may take a few minutes...")

# Fetch weekly player stats (2019-2024)
# 2025 doesn't exist in nflverse yet, we'll add it from PBP
years = [2019, 2020, 2021, 2022, 2023, 2024]
weekly_stats = nfl.import_weekly_data(years, downcast=True)

print(f"Loaded {len(weekly_stats)} player-week records")
print("Columns available:", weekly_stats.columns.tolist()[:20])

# Add 2025 data from our PBP
print("\nAdding 2025 data from PBP...")
pbp_2025 = pd.read_parquet('data/pbp/pbp_2025_updated.parquet')

# Aggregate 2025 PBP to weekly player stats
# Passing stats
passers = pbp_2025.groupby(['week', 'passer_player_id', 'passer_player_name', 'posteam']).agg({
    'complete_pass': 'sum',
    'pass_attempt': 'sum',
    'passing_yards': 'sum',
    'pass_touchdown': 'sum',
    'interception': 'sum',
    'sack': 'sum',
}).rename(columns={
    'complete_pass': 'completions',
    'pass_attempt': 'attempts',
    'pass_touchdown': 'passing_tds',
    'interception': 'interceptions',
    'sack': 'sacks'
}).reset_index()
passers['season'] = 2025
passers = passers.rename(columns={'passer_player_id': 'player_id', 'passer_player_name': 'player_name', 'posteam': 'recent_team'})
passers['position'] = 'QB'

# Rushing stats
rushers = pbp_2025.groupby(['week', 'rusher_player_id', 'rusher_player_name', 'posteam']).agg({
    'rush_attempt': 'sum',
    'rushing_yards': 'sum',
    'rush_touchdown': 'sum'
}).rename(columns={
    'rush_attempt': 'carries',
    'rush_touchdown': 'rushing_tds'
}).reset_index()
rushers['season'] = 2025
rushers = rushers.rename(columns={'rusher_player_id': 'player_id', 'rusher_player_name': 'player_name', 'posteam': 'recent_team'})
rushers['position'] = 'RB'

# Receiving stats
receivers = pbp_2025.groupby(['week', 'receiver_player_id', 'receiver_player_name', 'posteam']).agg({
    'complete_pass': 'sum',
    'pass_attempt': 'sum',
    'receiving_yards': 'sum',
    'pass_touchdown': 'sum'
}).rename(columns={
    'complete_pass': 'receptions',
    'pass_attempt': 'targets',
    'pass_touchdown': 'receiving_tds'
}).reset_index()
receivers['season'] = 2025
receivers = receivers.rename(columns={'receiver_player_id': 'player_id', 'receiver_player_name': 'player_name', 'posteam': 'recent_team'})
receivers['position'] = 'WR'

# Combine 2025 data
weekly_2025 = pd.concat([passers, rushers, receivers], ignore_index=True)
weekly_2025 = weekly_2025.dropna(subset=['player_id'])

# Calculate fantasy points (simple version)
weekly_2025['fantasy_points_ppr'] = (
    weekly_2025.get('passing_yards', 0) * 0.04 +
    weekly_2025.get('passing_tds', 0) * 4 +
    weekly_2025.get('interceptions', 0) * -2 +
    weekly_2025.get('rushing_yards', 0) * 0.1 +
    weekly_2025.get('rushing_tds', 0) * 6 +
    weekly_2025.get('receptions', 0) * 1 +
    weekly_2025.get('receiving_yards', 0) * 0.1 +
    weekly_2025.get('receiving_tds', 0) * 6
)
weekly_2025['fantasy_points'] = weekly_2025['fantasy_points_ppr'] - weekly_2025.get('receptions', 0)

print(f"Created {len(weekly_2025)} 2025 player-week records")

# Append to historical data
weekly_stats = pd.concat([weekly_stats, weekly_2025], ignore_index=True)
print(f"Total after adding 2025: {len(weekly_stats)} records")

# Fetch schedule to get game-level info
years_schedule = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
schedule = nfl.import_schedules(years_schedule)
print(f"Loaded {len(schedule)} games")

# Merge player stats with schedule to get home/away context
# Match on team and week
merged = weekly_stats.merge(
    schedule[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'temp', 'wind', 'roof', 'surface']],
    how='left',
    left_on=['season', 'week', 'recent_team'],
    right_on=['season', 'week', 'home_team']
)

# Also merge for away teams
merged_away = weekly_stats.merge(
    schedule[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'temp', 'wind', 'roof', 'surface']],
    how='left',
    left_on=['season', 'week', 'recent_team'],
    right_on=['season', 'week', 'away_team']
)

# Combine
merged = pd.concat([merged, merged_away], ignore_index=True).dropna(subset=['home_team'])

print(f"After merge: {len(merged)} records")

# Keep only key offensive stats (similar to old model)
feature_cols = [
    'season', 'week', 'recent_team', 'opponent_team', 'position',
    'home_team', 'away_team', 'home_score', 'away_score',
    # Passing
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
    # Rushing
    'carries', 'rushing_yards', 'rushing_tds',
    # Receiving
    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
    # Fantasy points (proxy for overall value)
    'fantasy_points', 'fantasy_points_ppr',
    # Environmental
    'temp', 'wind', 'roof', 'surface'
]

# Filter to available columns
feature_cols = [c for c in feature_cols if c in merged.columns]
data = merged[feature_cols].copy()

# Fill NaN with 0 for stats
stat_cols = [c for c in feature_cols if c not in ['season', 'week', 'recent_team', 'opponent_team', 'position', 'home_team', 'away_team', 'roof', 'surface']]
data[stat_cols] = data[stat_cols].fillna(0)

print(f"Final dataset: {len(data)} records with {len(feature_cols)} features")

# Save for inspection
data.to_csv('data/nfl/processed/player_stats_historical.csv', index=False)
print("Saved to data/nfl/processed/player_stats_historical.csv")

print("\n=== Sample of data ===")
print(data.head())
print("\n=== Stats for Week 10, 2025 ===")
week10 = data[(data['season'] == 2025) & (data['week'] == 10)]
print(f"Week 10 records: {len(week10)}")
print("Teams:", week10['recent_team'].unique())

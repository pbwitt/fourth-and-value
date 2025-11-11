"""
Train simple box score model using basic team stats
Uses: points scored/allowed, yards, TDs, turnovers, time of possession
Similar to old approach but with 2025 data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os

print("Loading PBP data...")
pbp = pd.read_parquet('data/pbp/pbp_2025_updated.parquet')

# Aggregate to team-game level with box score stats
print("Building box score features...")

games = pbp.groupby(['game_id', 'week', 'posteam', 'defteam', 'home_team', 'away_team']).agg({
    'home_score': 'max',
    'away_score': 'max',
    'total': 'first',
    'play_id': 'count',  # total plays
    'yards_gained': 'sum',
    'pass_touchdown': 'sum',
    'rush_touchdown': 'sum',
    'interception': 'sum',
    'fumble_lost': 'sum',
    'third_down_converted': 'sum',
    'third_down_failed': 'sum',
    'game_seconds_remaining': lambda x: x.max() - x.min()  # time of possession proxy
}).reset_index()

games.columns = ['game_id', 'week', 'team', 'opponent', 'home_team', 'away_team',
                 'home_score', 'away_score', 'total', 'plays', 'yards',
                 'pass_tds', 'rush_tds', 'ints', 'fumbles', 'third_conv', 'third_fail', 'time_poss']

# Calculate derived stats
games['total_tds'] = games['pass_tds'] + games['rush_tds']
games['turnovers'] = games['ints'] + games['fumbles']
games['third_pct'] = games['third_conv'] / (games['third_conv'] + games['third_fail'] + 0.001)
games['yards_per_play'] = games['yards'] / (games['plays'] + 0.001)

# Split into home and away
home_games = games[games['team'] == games['home_team']].copy()
away_games = games[games['team'] == games['away_team']].copy()

# Rename for home team
home_games = home_games.rename(columns={
    'team': 'home_team_name',
    'plays': 'home_plays',
    'yards': 'home_yards',
    'total_tds': 'home_tds',
    'turnovers': 'home_turnovers',
    'third_pct': 'home_third_pct',
    'yards_per_play': 'home_ypp'
})

# Rename for away team
away_games = away_games.rename(columns={
    'team': 'away_team_name',
    'plays': 'away_plays',
    'yards': 'away_yards',
    'total_tds': 'away_tds',
    'turnovers': 'away_turnovers',
    'third_pct': 'away_third_pct',
    'yards_per_play': 'away_ypp'
})

# Merge to get both teams in one row
merged = home_games[['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'total',
                     'home_plays', 'home_yards', 'home_tds', 'home_turnovers', 'home_third_pct', 'home_ypp']].merge(
    away_games[['game_id', 'away_plays', 'away_yards', 'away_tds', 'away_turnovers', 'away_third_pct', 'away_ypp']],
    on='game_id'
)

print(f"Built {len(merged)} game records")

# Calculate rolling averages (L3, L5) for each team
def add_rolling_features(df, team_col, feature_cols, windows=[3, 5]):
    for team in df[team_col].unique():
        team_mask = df[team_col] == team
        team_df = df[team_mask].copy().sort_values('week')

        for window in windows:
            for col in feature_cols:
                df.loc[team_mask, f'{col}_L{window}'] = team_df[col].shift(1).rolling(window, min_periods=1).mean().values

    return df

# Home team rolling features
home_features = ['home_plays', 'home_yards', 'home_tds', 'home_turnovers', 'home_third_pct', 'home_ypp']
merged = add_rolling_features(merged, 'home_team', home_features)

# Away team rolling features
away_features = ['away_plays', 'away_yards', 'away_tds', 'away_turnovers', 'away_third_pct', 'away_ypp']
merged = add_rolling_features(merged, 'away_team', away_features)

print("Rolling features added")

# Prepare training data
feature_cols = [c for c in merged.columns if c.endswith('_L3') or c.endswith('_L5')]
print(f"Features: {len(feature_cols)}")

# Filter to games with valid scores and features
train_data = merged[(merged['home_score'] > 0) & (merged['week'] >= 2)].dropna(subset=feature_cols).copy()

print(f"Training samples: {len(train_data)}")

X = train_data[feature_cols]
y_home = train_data['home_score']
y_away = train_data['away_score']
y_total = train_data['total']

# Train with time series CV
tscv = TimeSeriesSplit(n_splits=5)

# Home score model
print("\nTraining home score model...")
model_home = Ridge(alpha=1.0)
home_scores = []
for train_idx, val_idx in tscv.split(X):
    model_home.fit(X.iloc[train_idx], y_home.iloc[train_idx])
    preds = model_home.predict(X.iloc[val_idx])
    errors = np.abs(preds - y_home.iloc[val_idx])
    home_scores.append(errors.mean())

print(f"Home CV MAE: {np.mean(home_scores):.2f}")

# Away score model
print("Training away score model...")
model_away = Ridge(alpha=1.0)
away_scores = []
for train_idx, val_idx in tscv.split(X):
    model_away.fit(X.iloc[train_idx], y_away.iloc[train_idx])
    preds = model_away.predict(X.iloc[val_idx])
    errors = np.abs(preds - y_away.iloc[val_idx])
    away_scores.append(errors.mean())

print(f"Away CV MAE: {np.mean(away_scores):.2f}")

# Retrain on all data
model_home.fit(X, y_home)
model_away.fit(X, y_away)

# Save models
os.makedirs('data/nfl/models', exist_ok=True)

model_data = {
    'model_home': model_home,
    'model_away': model_away,
    'features': feature_cols,
    'cv_mae_home': np.mean(home_scores),
    'cv_mae_away': np.mean(away_scores),
    'cv_mae_total': (np.mean(home_scores) + np.mean(away_scores)) / 2
}

with open('data/nfl/models/boxscore_totals.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Saved box score model to data/nfl/models/boxscore_totals.pkl")
print(f"  CV MAE (total): {model_data['cv_mae_total']:.2f}")

# Save team features for prediction
merged.to_csv('data/nfl/processed/team_boxscore_features.csv', index=False)
print(f"✓ Saved team features to data/nfl/processed/team_boxscore_features.csv")

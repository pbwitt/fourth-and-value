"""
Train pace-based model
Theory: Total points = (Team A plays × Team A yards/play × scoring efficiency) +
                        (Team B plays × Team B yards/play × scoring efficiency)
Simplified: Focus on pace (plays per game) and efficiency (points per play)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os

print("Loading PBP data...")
pbp = pd.read_parquet('data/pbp/pbp_2025_updated.parquet')

# Calculate pace metrics per team per game
print("Building pace features...")

pace_data = pbp.groupby(['game_id', 'week', 'posteam', 'home_team', 'away_team']).agg({
    'play_id': 'count',  # total plays
    'yards_gained': 'sum',
    'home_score': 'max',
    'away_score': 'max',
    'total': 'first',
    'game_seconds_remaining': lambda x: (x.max() - x.min()) / 60  # time of possession in minutes
}).reset_index()

pace_data.columns = ['game_id', 'week', 'team', 'home_team', 'away_team',
                     'plays', 'yards', 'home_score', 'away_score', 'total', 'time_poss']

# Calculate efficiency metrics
pace_data['yards_per_play'] = pace_data['yards'] / (pace_data['plays'] + 0.001)
pace_data['plays_per_min'] = pace_data['plays'] / (pace_data['time_poss'] + 0.001)

# Determine if team was home or away
pace_data['is_home'] = (pace_data['team'] == pace_data['home_team']).astype(int)

# Calculate team scoring
pace_data['points_scored'] = np.where(pace_data['is_home'] == 1,
                                      pace_data['home_score'],
                                      pace_data['away_score'])
pace_data['points_allowed'] = np.where(pace_data['is_home'] == 1,
                                       pace_data['away_score'],
                                       pace_data['home_score'])

pace_data['points_per_play'] = pace_data['points_scored'] / (pace_data['plays'] + 0.001)

print(f"Built {len(pace_data)} team-game records")

# Add rolling averages for pace metrics
def add_rolling_features(df, team_col, feature_cols, windows=[3, 5]):
    for team in df[team_col].unique():
        team_mask = df[team_col] == team
        team_df = df[team_mask].copy().sort_values('week')

        for window in windows:
            for col in feature_cols:
                df.loc[team_mask, f'{col}_L{window}'] = team_df[col].shift(1).rolling(window, min_periods=1).mean().values

    return df

pace_features = ['plays', 'yards_per_play', 'plays_per_min', 'points_per_play', 'points_scored', 'points_allowed']
pace_data = add_rolling_features(pace_data, 'team', pace_features)

print("Rolling features added")

# Pivot to get home and away teams in same row
home_pace = pace_data[pace_data['is_home'] == 1].copy()
away_pace = pace_data[pace_data['is_home'] == 0].copy()

# Rename columns
home_cols_map = {c: f'home_{c}' for c in pace_features if c in pace_data.columns}
home_cols_map.update({c: f'home_{c}' for c in pace_data.columns if c.endswith('_L3') or c.endswith('_L5')})

away_cols_map = {c: f'away_{c}' for c in pace_features if c in pace_data.columns}
away_cols_map.update({c: f'away_{c}' for c in pace_data.columns if c.endswith('_L3') or c.endswith('_L5')})

home_pace = home_pace.rename(columns=home_cols_map)
away_pace = away_pace.rename(columns=away_cols_map)

# Merge
merged = home_pace[['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'total'] +
                   [c for c in home_pace.columns if c.startswith('home_') and ('_L3' in c or '_L5' in c)]].merge(
    away_pace[['game_id'] + [c for c in away_pace.columns if c.startswith('away_') and ('_L3' in c or '_L5' in c)]],
    on='game_id'
)

print(f"Merged to {len(merged)} games")

# Prepare training data
feature_cols = [c for c in merged.columns if ('_L3' in c or '_L5' in c)]
print(f"Features: {len(feature_cols)}")

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

with open('data/nfl/models/pace_totals.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Saved pace model to data/nfl/models/pace_totals.pkl")
print(f"  CV MAE (total): {model_data['cv_mae_total']:.2f}")

# Save team features for prediction
merged.to_csv('data/nfl/processed/team_pace_features.csv', index=False)
print(f"✓ Saved team features to data/nfl/processed/team_pace_features.csv")

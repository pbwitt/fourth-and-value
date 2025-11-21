"""
Calibrate ensemble predictions using historical data
Trains a correction model to fix systematic biases
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

print("="*80)
print("CALIBRATING ENSEMBLE MODEL")
print("="*80)

# Load historical predictions and actuals
print("\nGenerating historical predictions...")

# Load all three models
with open('data/nfl/models/ridge_totals.pkl', 'rb') as f:
    epa_model = pickle.load(f)

with open('data/nfl/models/boxscore_totals.pkl', 'rb') as f:
    boxscore_model = pickle.load(f)

with open('data/nfl/models/pace_totals.pkl', 'rb') as f:
    pace_model = pickle.load(f)

# Load feature data
epa_features = pd.read_csv('data/nfl/processed/team_features.csv')
boxscore_features = pd.read_csv('data/nfl/processed/team_boxscore_features.csv')
pace_features = pd.read_csv('data/nfl/processed/team_pace_features.csv')

# Get actual scores from PBP
pbp = pd.read_parquet('data/pbp/pbp_2025_updated.parquet')
actuals = pbp.groupby(['game_id', 'week', 'home_team', 'away_team']).agg({
    'home_score': 'max',
    'away_score': 'max',
    'total': 'first'
}).reset_index()

print("Loaded all models and features")

# Generate predictions for all historical games
print("\nGenerating historical ensemble predictions...")

historical = []

# Filter to games with valid scores (Weeks 2-10)
valid_games = actuals[(actuals['home_score'] > 0) & (actuals['week'] >= 2) & (actuals['week'] <= 10)].copy()

print(f"Processing {len(valid_games)} historical games...")

for idx, game in valid_games.iterrows():
    game_id = game['game_id']
    week = game['week']
    home_team = game['home_team']
    away_team = game['away_team']
    actual_total = game['total']

    # EPA prediction
    game_epa = epa_features[epa_features['game_id'] == game_id]
    if len(game_epa) > 0:
        features = game_epa[epa_model['features']].fillna(0)
        epa_pred = (
            epa_model['model_home'].predict(features)[0] +
            epa_model['model_away'].predict(features)[0]
        )
    else:
        epa_pred = np.nan

    # Box score prediction
    game_box = boxscore_features[
        (boxscore_features['home_team'] == home_team) &
        (boxscore_features['away_team'] == away_team) &
        (boxscore_features['week'] == week)
    ]
    if len(game_box) > 0:
        features = game_box[boxscore_model['features']].fillna(0)
        boxscore_pred = (
            boxscore_model['model_home'].predict(features)[0] +
            boxscore_model['model_away'].predict(features)[0]
        )
    else:
        boxscore_pred = np.nan

    # Pace prediction
    game_pace = pace_features[
        (pace_features['home_team'] == home_team) &
        (pace_features['away_team'] == away_team) &
        (pace_features['week'] == week)
    ]
    if len(game_pace) > 0:
        features = game_pace[pace_model['features']].fillna(0)
        pace_pred = (
            pace_model['model_home'].predict(features)[0] +
            pace_model['model_away'].predict(features)[0]
        )
    else:
        pace_pred = np.nan

    # Ensemble (simple average)
    ensemble_pred = np.nanmean([epa_pred, boxscore_pred, pace_pred])
    model_std = np.nanstd([epa_pred, boxscore_pred, pace_pred])
    num_models = np.sum(~np.isnan([epa_pred, boxscore_pred, pace_pred]))

    if not np.isnan(ensemble_pred):
        historical.append({
            'game_id': game_id,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'actual_total': actual_total,
            'epa_pred': epa_pred,
            'boxscore_pred': boxscore_pred,
            'pace_pred': pace_pred,
            'ensemble_pred': ensemble_pred,
            'model_std': model_std,
            'num_models': num_models,
            'error': ensemble_pred - actual_total
        })

hist_df = pd.DataFrame(historical)

print(f"\nGenerated {len(hist_df)} historical predictions")
print(f"  Mean error: {hist_df['error'].mean():.2f} points")
print(f"  MAE: {hist_df['error'].abs().mean():.2f} points")
print(f"  Std: {hist_df['error'].std():.2f} points")

# Identify systematic biases
print("\n" + "="*80)
print("IDENTIFYING BIASES")
print("="*80)

# Bias by model agreement
print("\nBias by model agreement:")
for num in sorted(hist_df['num_models'].unique()):
    subset = hist_df[hist_df['num_models'] == num]
    print(f"  {int(num)} models: {subset['error'].mean():+.2f} pts (n={len(subset)})")

# Bias by ensemble value (over/under predictions)
print("\nBias by predicted total:")
hist_df['pred_bucket'] = pd.cut(hist_df['ensemble_pred'], bins=[0, 40, 45, 50, 55, 100], labels=['<40', '40-45', '45-50', '50-55', '55+'])
for bucket in hist_df['pred_bucket'].cat.categories:
    subset = hist_df[hist_df['pred_bucket'] == bucket]
    if len(subset) > 0:
        print(f"  {bucket}: {subset['error'].mean():+.2f} pts (n={len(subset)})")

# Bias by model disagreement
print("\nBias by model disagreement (std):")
hist_df['std_bucket'] = pd.cut(hist_df['model_std'], bins=[0, 2, 4, 6, 100], labels=['<2', '2-4', '4-6', '6+'])
for bucket in hist_df['std_bucket'].cat.categories:
    subset = hist_df[hist_df['std_bucket'] == bucket]
    if len(subset) > 0:
        print(f"  {bucket}: {subset['error'].mean():+.2f} pts, MAE: {subset['error'].abs().mean():.2f} (n={len(subset)})")

# Train calibration model
print("\n" + "="*80)
print("TRAINING CALIBRATION MODEL")
print("="*80)

# Features for calibration
calib_features = ['ensemble_pred', 'model_std', 'num_models', 'epa_pred', 'boxscore_pred', 'pace_pred']

# Filter to rows with all features
calib_data = hist_df.dropna(subset=calib_features).copy()

print(f"Training on {len(calib_data)} games...")

X_calib = calib_data[calib_features]
y_error = calib_data['error']  # What we want to correct

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
calib_model = Ridge(alpha=0.5)

cv_scores = []
for train_idx, val_idx in tscv.split(X_calib):
    calib_model.fit(X_calib.iloc[train_idx], y_error.iloc[train_idx])
    preds = calib_model.predict(X_calib.iloc[val_idx])
    # MAE of the calibrated predictions
    calibrated_ensemble = calib_data.iloc[val_idx]['ensemble_pred'] - preds
    errors = np.abs(calibrated_ensemble - calib_data.iloc[val_idx]['actual_total'])
    cv_scores.append(errors.mean())

print(f"\nCalibration CV Results:")
print(f"  Original MAE: {hist_df['error'].abs().mean():.2f} points")
print(f"  Calibrated MAE: {np.mean(cv_scores):.2f} points")
print(f"  Improvement: {hist_df['error'].abs().mean() - np.mean(cv_scores):.2f} points")

# Retrain on all data
calib_model.fit(X_calib, y_error)

# Analyze corrections
predictions_correction = calib_model.predict(X_calib)
print(f"\nCalibration statistics:")
print(f"  Mean correction: {predictions_correction.mean():.2f} points")
print(f"  Correction range: [{predictions_correction.min():.2f}, {predictions_correction.max():.2f}]")

# Save calibration model
calibration_data = {
    'model': calib_model,
    'features': calib_features,
    'original_mae': hist_df['error'].abs().mean(),
    'calibrated_mae': np.mean(cv_scores),
    'improvement': hist_df['error'].abs().mean() - np.mean(cv_scores)
}

with open('data/nfl/models/ensemble_calibration.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)

print(f"\n✓ Saved calibration model to data/nfl/models/ensemble_calibration.pkl")

# Save historical predictions for analysis
hist_df['calibrated_pred'] = hist_df['ensemble_pred'] - calib_model.predict(hist_df[calib_features].fillna(0))
hist_df['calibrated_error'] = hist_df['calibrated_pred'] - hist_df['actual_total']

hist_df.to_csv('data/nfl/processed/historical_predictions.csv', index=False)
print(f"✓ Saved historical predictions to data/nfl/processed/historical_predictions.csv")

print("\n" + "="*80)
print("CALIBRATION COMPLETE")
print("="*80)

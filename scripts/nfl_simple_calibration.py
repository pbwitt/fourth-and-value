"""
Simple calibration using historical actuals vs predictions from saved data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pickle

print("="*80)
print("SIMPLE ENSEMBLE CALIBRATION")
print("="*80)

# Load PBP to get actuals
pbp = pd.read_parquet('data/pbp/pbp_2025_updated.parquet')
actuals = pbp.groupby(['game_id', 'week', 'home_team', 'away_team']).agg({
    'home_score': 'max',
    'away_score': 'max',
    'total': 'first',
    'total_line': 'first'
}).reset_index()

# Filter to weeks 2-10
actuals = actuals[(actuals['week'] >= 2) & (actuals['week'] <= 10) & (actuals['home_score'] > 0)].copy()

print(f"Loaded {len(actuals)} actual results (weeks 2-10)")

# Load team features to generate predictions
epa_features = pd.read_csv('data/nfl/processed/team_features.csv')
boxscore_features = pd.read_csv('data/nfl/processed/team_boxscore_features.csv')
pace_features = pd.read_csv('data/nfl/processed/team_pace_features.csv')

# Load models
with open('data/nfl/models/ridge_totals.pkl', 'rb') as f:
    epa_model = pickle.load(f)

with open('data/nfl/models/boxscore_totals.pkl', 'rb') as f:
    boxscore_model = pickle.load(f)

with open('data/nfl/models/pace_totals.pkl', 'rb') as f:
    pace_model = pickle.load(f)

print("Loaded models")

# Generate predictions for each game
historical = []

for _, actual in actuals.iterrows():
    game_id = actual['game_id']
    week = actual['week']
    home = actual['home_team']
    away = actual['away_team']
    actual_total = actual['total']

    # EPA prediction
    epa_game = epa_features[epa_features['game_id'] == game_id]
    if len(epa_game) > 0 and all(c in epa_game.columns for c in epa_model['features']):
        try:
            X = epa_game[epa_model['features']].fillna(0)
            epa_pred = epa_model['model_home'].predict(X)[0] + epa_model['model_away'].predict(X)[0]
        except:
            epa_pred = np.nan
    else:
        epa_pred = np.nan

    # Box score prediction
    box_game = boxscore_features[(boxscore_features['home_team'] == home) &
                                  (boxscore_features['away_team'] == away) &
                                  (boxscore_features['week'] == week)]
    if len(box_game) > 0:
        try:
            X = box_game[boxscore_model['features']].fillna(0)
            box_pred = boxscore_model['model_home'].predict(X)[0] + boxscore_model['model_away'].predict(X)[0]
        except:
            box_pred = np.nan
    else:
        box_pred = np.nan

    # Pace prediction
    pace_game = pace_features[(pace_features['home_team'] == home) &
                              (pace_features['away_team'] == away) &
                              (pace_features['week'] == week)]
    if len(pace_game) > 0:
        try:
            X = pace_game[pace_model['features']].fillna(0)
            pace_pred = pace_model['model_home'].predict(X)[0] + pace_model['model_away'].predict(X)[0]
        except:
            pace_pred = np.nan
    else:
        pace_pred = np.nan

    # Calculate ensemble
    preds = [epa_pred, box_pred, pace_pred]
    preds_valid = [p for p in preds if not np.isnan(p)]

    if len(preds_valid) > 0:
        ensemble_pred = np.mean(preds_valid)
        model_std = np.std(preds_valid) if len(preds_valid) > 1 else 0
        num_models = len(preds_valid)

        historical.append({
            'game_id': game_id,
            'week': week,
            'actual_total': actual_total,
            'market_line': actual['total_line'],
            'epa_pred': epa_pred,
            'box_pred': box_pred,
            'pace_pred': pace_pred,
            'ensemble_pred': ensemble_pred,
            'model_std': model_std,
            'num_models': num_models,
            'error': ensemble_pred - actual_total
        })

hist_df = pd.DataFrame(historical)

print(f"\nGenerated {len(hist_df)} historical predictions")
print(f"  Mean error: {hist_df['error'].mean():.2f} points (positive = overpredicting)")
print(f"  MAE: {hist_df['error'].abs().mean():.2f} points")
print(f"  Std: {hist_df['error'].std():.2f} points")

# Analyze biases
print("\n" + "="*80)
print("SYSTEMATIC BIASES")
print("="*80)

print("\nBy predicted total:")
hist_df['pred_bucket'] = pd.cut(hist_df['ensemble_pred'], bins=[0, 42, 46, 50, 100], labels=['<42', '42-46', '46-50', '50+'])
for bucket in hist_df['pred_bucket'].cat.categories:
    subset = hist_df[hist_df['pred_bucket'] == bucket]
    if len(subset) > 0:
        print(f"  {bucket}: {subset['error'].mean():+.2f} pts (MAE: {subset['error'].abs().mean():.2f}, n={len(subset)})")

print("\nBy model agreement (std dev):")
hist_df['std_bucket'] = pd.cut(hist_df['model_std'], bins=[0, 2, 4, 100], labels=['<2', '2-4', '4+'])
for bucket in hist_df['std_bucket'].cat.categories:
    subset = hist_df[hist_df['std_bucket'] == bucket]
    if len(subset) > 0:
        print(f"  {bucket}: {subset['error'].mean():+.2f} pts (MAE: {subset['error'].abs().mean():.2f}, n={len(subset)})")

# Simple calibration model
print("\n" + "="*80)
print("TRAINING CALIBRATION")
print("="*80)

# Features: ensemble prediction, model std, num models
calib_features = ['ensemble_pred', 'model_std', 'num_models']
X_calib = hist_df[calib_features].fillna(0)
y_error = hist_df['error']

# Train Ridge regression to predict error
calib_model = Ridge(alpha=1.0)
calib_model.fit(X_calib, y_error)

# Cross-validate
cv_scores = cross_val_score(calib_model, X_calib, y_error, cv=5, scoring='neg_mean_absolute_error')

print(f"\nCalibration Results:")
print(f"  Original MAE: {hist_df['error'].abs().mean():.2f} points")
print(f"  Calibrated MAE (CV): {-cv_scores.mean():.2f} points")
print(f"  Improvement: {hist_df['error'].abs().mean() + cv_scores.mean():.2f} points")

# Apply corrections to see distribution
corrections = calib_model.predict(X_calib)
hist_df['calibrated_pred'] = hist_df['ensemble_pred'] - corrections
hist_df['calibrated_error'] = hist_df['calibrated_pred'] - hist_df['actual_total']

print(f"\nCorrection statistics:")
print(f"  Mean correction: {corrections.mean():.2f} points")
print(f"  Range: [{corrections.min():.2f}, {corrections.max():.2f}]")
print(f"  Calibrated MAE: {hist_df['calibrated_error'].abs().mean():.2f} points")

# Save calibration model
calibration_data = {
    'model': calib_model,
    'features': calib_features,
    'original_mae': hist_df['error'].abs().mean(),
    'calibrated_mae': hist_df['calibrated_error'].abs().mean(),
    'improvement': hist_df['error'].abs().mean() - hist_df['calibrated_error'].abs().mean(),
    'mean_correction': corrections.mean()
}

with open('data/nfl/models/ensemble_calibration.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)

print(f"\n✓ Saved calibration model")

# Save historical data
hist_df.to_csv('data/nfl/processed/historical_predictions.csv', index=False)
print(f"✓ Saved historical predictions")

print("\n" + "="*80)
print("DONE")
print("="*80)

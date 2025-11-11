"""
Final ensemble with all 3 models + calibration
"""

import pandas as pd
import numpy as np
import pickle
import subprocess

print("="*80)
print("NFL TOTALS - FINAL CALIBRATED ENSEMBLE")
print("="*80)

# Step 1: Generate pace predictions
print("\n[1/3] Generating pace predictions...")
result = subprocess.run(['python3', 'scripts/nfl_predict_pace.py'], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)

# Step 2: Calibrate ensemble
print("\n[2/3] Calibrating ensemble...")
result = subprocess.run(['python3', 'scripts/nfl_simple_calibration.py'], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)

# Step 3: Generate final predictions with calibration
print("\n[3/3] Generating final calibrated predictions...")

# Load calibration model
with open('data/nfl/models/ensemble_calibration.pkl', 'rb') as f:
    calibration = pickle.load(f)

# Load all Week 11 predictions
epa_preds = pd.read_csv('data/nfl/predictions/week_predictions.csv')
epa_preds = epa_preds[['game', 'home_team', 'away_team', 'total_pred']].rename(columns={'total_pred': 'epa_pred'})

try:
    boxscore_preds = pd.read_csv('data/nfl/predictions/week11_boxscore.csv')
    boxscore_preds = boxscore_preds[['game', 'total_pred']].rename(columns={'total_pred': 'boxscore_pred'})
except:
    boxscore_preds = pd.DataFrame({'game': epa_preds['game'], 'boxscore_pred': np.nan})

try:
    pace_preds = pd.read_csv('data/nfl/predictions/week11_pace.csv')
    pace_preds = pace_preds[['game', 'total_pred']].rename(columns={'total_pred': 'pace_pred'})
except:
    pace_preds = pd.DataFrame({'game': epa_preds['game'], 'pace_pred': np.nan})

# Merge
ensemble = epa_preds.merge(boxscore_preds, on='game', how='left')
ensemble = ensemble.merge(pace_preds, on='game', how='left')

# Calculate ensemble
ensemble['ensemble_pred'] = ensemble[['epa_pred', 'boxscore_pred', 'pace_pred']].mean(axis=1, skipna=True)
ensemble['model_std'] = ensemble[['epa_pred', 'boxscore_pred', 'pace_pred']].std(axis=1, skipna=True)
ensemble['num_models'] = ensemble[['epa_pred', 'boxscore_pred', 'pace_pred']].notna().sum(axis=1)

# Apply calibration
X_calib = ensemble[calibration['features']].fillna(0)
corrections = calibration['model'].predict(X_calib)

ensemble['calibrated_pred'] = ensemble['ensemble_pred'] - corrections
ensemble['correction_applied'] = corrections

# Add market lines
market_lines = {
    'NYJ @ NE': 43.5, 'SF @ ARI': 48.0, 'CAR @ ATL': 48.5, 'TB @ BUF': 50.5,
    'BAL @ CLE': 43.5, 'KC @ DEN': 44.5, 'LAC @ JAX': 44.5, 'SEA @ LA': 48.5,
    'WAS @ MIA': 47.5, 'CHI @ MIN': 46.5, 'GB @ NYG': 44.5, 'DET @ PHI': 49.0,
    'CIN @ PIT': 49.5, 'HOU @ TEN': 38.0, 'DAL @ LV': 50.5
}

ensemble['market_total'] = ensemble['game'].map(market_lines)
ensemble['edge_uncalibrated'] = ensemble['ensemble_pred'] - ensemble['market_total']
ensemble['edge_calibrated'] = ensemble['calibrated_pred'] - ensemble['market_total']

# Identify high-confidence bets
ensemble['high_confidence'] = (
    (ensemble['num_models'] == 3) &
    (ensemble['model_std'] < 3.0) &
    (ensemble['edge_calibrated'].abs() >= 3.0)
)

ensemble['medium_confidence'] = (
    (ensemble['num_models'] >= 2) &
    (ensemble['model_std'] < 4.0) &
    (ensemble['edge_calibrated'].abs() >= 3.0) &
    (~ensemble['high_confidence'])
)

print("\n" + "="*80)
print("WEEK 11 BETTING RECOMMENDATIONS")
print("="*80)

high_conf = ensemble[ensemble['high_confidence']].sort_values('edge_calibrated', key=abs, ascending=False)
med_conf = ensemble[ensemble['medium_confidence']].sort_values('edge_calibrated', key=abs, ascending=False)

if len(high_conf) > 0:
    print(f"\n{'='*80}")
    print(f"HIGH CONFIDENCE BETS ({len(high_conf)} games)")
    print(f"{'='*80}\n")

    for _, row in high_conf.iterrows():
        print(f"Game: {row['game']}")
        print(f"  Models: EPA={row['epa_pred']:.1f}, Box={row['boxscore_pred']:.1f}, Pace={row['pace_pred']:.1f}")
        print(f"  Agreement: {row['model_std']:.1f} std dev (HIGH)" if row['model_std'] < 2 else f"  Agreement: {row['model_std']:.1f} std dev (MODERATE)")
        print(f"  Ensemble (raw): {row['ensemble_pred']:.1f}")
        print(f"  Calibrated: {row['calibrated_pred']:.1f} (correction: {row['correction_applied']:+.1f})")
        print(f"  Market: {row['market_total']:.1f}")
        print(f"  EDGE: {row['edge_calibrated']:+.1f} points")
        if row['edge_calibrated'] > 0:
            print(f"  → BET OVER {row['market_total']}")
        else:
            print(f"  → BET UNDER {row['market_total']}")
        print()

if len(med_conf) > 0:
    print(f"\n{'='*80}")
    print(f"MEDIUM CONFIDENCE BETS ({len(med_conf)} games)")
    print(f"{'='*80}\n")

    for _, row in med_conf.iterrows():
        print(f"Game: {row['game']}")
        print(f"  Calibrated: {row['calibrated_pred']:.1f} | Market: {row['market_total']:.1f} | Edge: {row['edge_calibrated']:+.1f}")
        print(f"  Models: {int(row['num_models'])} | Std: {row['model_std']:.1f}")
        if row['edge_calibrated'] > 0:
            print(f"  → BET OVER {row['market_total']}")
        else:
            print(f"  → BET UNDER {row['market_total']}")
        print()

if len(high_conf) == 0 and len(med_conf) == 0:
    print("\nNo high or medium confidence bets this week.")
    print("\nAll games with edges:")
    all_edges = ensemble[ensemble['edge_calibrated'].abs() >= 2.0].sort_values('edge_calibrated', key=abs, ascending=False)

    for _, row in all_edges.iterrows():
        conf = "LOW" if row['model_std'] > 4 else "MODERATE"
        print(f"{row['game']:20} | Cal: {row['calibrated_pred']:5.1f} | Mkt: {row['market_total']:5.1f} | Edge: {row['edge_calibrated']:+5.1f} | Conf: {conf}")

print(f"\n{'='*80}")
print("FULL PREDICTIONS (with calibration)")
print(f"{'='*80}\n")

display_cols = ['game', 'epa_pred', 'boxscore_pred', 'pace_pred', 'ensemble_pred',
                'calibrated_pred', 'market_total', 'edge_calibrated', 'model_std']
print(ensemble[display_cols].to_string(index=False))

print(f"\n{'='*80}")
print("CALIBRATION IMPACT")
print(f"{'='*80}")
print(f"\nAverage correction applied: {ensemble['correction_applied'].mean():.2f} points")
print(f"Correction range: [{ensemble['correction_applied'].min():.2f}, {ensemble['correction_applied'].max():.2f}]")
print(f"\nOriginal model MAE (training): {calibration['original_mae']:.2f} points")
print(f"Calibrated MAE (training): {calibration['calibrated_mae']:.2f} points")
print(f"Improvement: {calibration['improvement']:.2f} points")

# Save
ensemble.to_csv('data/nfl/predictions/week11_final_calibrated.csv', index=False)
print(f"\n✓ Saved to data/nfl/predictions/week11_final_calibrated.csv")

print("\n" + "="*80)
print("STAY AWAY GAMES (Models match market)")
print("="*80)

stay_away = ensemble[ensemble['edge_calibrated'].abs() < 2.0].copy()
print(f"\n{len(stay_away)} games where calibrated prediction is within 2 points of market:")
for _, row in stay_away.iterrows():
    print(f"  {row['game']:20} Cal: {row['calibrated_pred']:.1f} vs Mkt: {row['market_total']:.1f} (diff: {row['edge_calibrated']:+.1f})")

print("\n✓ Complete!")

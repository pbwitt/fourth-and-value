"""
Ensemble predictions from EPA, Box Score, and Pace models
Add calibration layer
Compare to market consensus
Flag high-confidence bets where models agree AND disagree with Vegas
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

print("="*80)
print("NFL TOTALS ENSEMBLE MODEL")
print("="*80)

# Load schedule
schedule = pd.read_csv('data/schedule_2025.csv')
week_11 = schedule[schedule['week'] == 11].copy()

print(f"\nWeek 11 Schedule: {len(week_11)} games")

# ===========================
# 1. TRAIN ALL THREE MODELS
# ===========================

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

# Box Score Model
print("\n[1/3] Training Box Score Model...")
import subprocess
result = subprocess.run(['python3', 'scripts/nfl_train_boxscore_model.py'],
                       capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR:", result.stderr)
else:
    print(result.stdout.split('\n')[-5:])

# Pace Model
print("\n[2/3] Training Pace Model...")
result = subprocess.run(['python3', 'scripts/nfl_train_pace_model.py'],
                       capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR:", result.stderr)
else:
    print(result.stdout.split('\n')[-5:])

# EPA Model (already trained)
print("\n[3/3] EPA Model (already trained)")
with open('data/nfl/models/ridge_totals.pkl', 'rb') as f:
    epa_model_data = pickle.load(f)
print(f"  CV MAE: {epa_model_data['cv_mae']:.2f}")

# ===========================
# 2. GENERATE PREDICTIONS
# ===========================

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

# Run prediction scripts
print("\n[1/3] EPA predictions...")
result = subprocess.run(['python3', 'scripts/nfl_predict_totals.py'],
                       capture_output=True, text=True)

print("\n[2/3] Box Score predictions...")
# Create box score prediction script
boxscore_pred_script = """
import pandas as pd
import pickle

# Load model
with open('data/nfl/models/boxscore_totals.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Load features
features_df = pd.read_csv('data/nfl/processed/team_boxscore_features.csv')

# Get latest week data for each team
latest = features_df.groupby('home_team')['week'].max().reset_index()
latest = latest.merge(features_df, on=['home_team', 'week'])

latest_away = features_df.groupby('away_team')['week'].max().reset_index()
latest_away = latest_away.merge(features_df, on=['away_team', 'week'])

# Load schedule
schedule = pd.read_csv('data/schedule_2025.csv')
week_11 = schedule[schedule['week'] == 11]

predictions = []
for _, game in week_11.iterrows():
    home = game['home_team']
    away = game['away_team']

    # Get features for each team
    home_features = latest[latest['home_team'] == home][model_data['features']].fillna(0)
    away_features = latest_away[latest_away['away_team'] == away][model_data['features']].fillna(0)

    if len(home_features) > 0 and len(away_features) > 0:
        home_pred = model_data['model_home'].predict(home_features)[0]
        away_pred = model_data['model_away'].predict(away_features)[0]

        predictions.append({
            'game': f"{away} @ {home}",
            'home_team': home,
            'away_team': away,
            'home_pred': home_pred,
            'away_pred': away_pred,
            'total_pred': home_pred + away_pred
        })

pd.DataFrame(predictions).to_csv('data/nfl/predictions/week11_boxscore.csv', index=False)
print(f"Saved {len(predictions)} box score predictions")
"""

with open('/tmp/boxscore_predict.py', 'w') as f:
    f.write(boxscore_pred_script)

result = subprocess.run(['python3', '/tmp/boxscore_predict.py'], capture_output=True, text=True)
print(result.stdout)

print("\n[3/3] Pace predictions...")
pace_pred_script = """
import pandas as pd
import pickle

# Load model
with open('data/nfl/models/pace_totals.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Load features
features_df = pd.read_csv('data/nfl/processed/team_pace_features.csv')

# Load schedule
schedule = pd.read_csv('data/schedule_2025.csv')
week_11 = schedule[schedule['week'] == 11]

# Get latest data
latest = features_df[features_df['week'] == features_df['week'].max()]

predictions = []
for _, game in week_11.iterrows():
    home = game['home_team']
    away = game['away_team']

    game_data = latest[(latest['home_team'] == home) & (latest['away_team'] == away)]

    if len(game_data) == 0:
        # Try to construct from team histories
        continue

    features = game_data[model_data['features']].fillna(0)

    if len(features) > 0:
        home_pred = model_data['model_home'].predict(features)[0]
        away_pred = model_data['model_away'].predict(features)[0]

        predictions.append({
            'game': f"{away} @ {home}",
            'home_team': home,
            'away_team': away,
            'home_pred': home_pred,
            'away_pred': away_pred,
            'total_pred': home_pred + away_pred
        })

pd.DataFrame(predictions).to_csv('data/nfl/predictions/week11_pace.csv', index=False)
print(f"Saved {len(predictions)} pace predictions")
"""

with open('/tmp/pace_predict.py', 'w') as f:
    f.write(pace_pred_script)

result = subprocess.run(['python3', '/tmp/pace_predict.py'], capture_output=True, text=True)
print(result.stdout)

# ===========================
# 3. LOAD ALL PREDICTIONS
# ===========================

print("\n" + "="*80)
print("LOADING PREDICTIONS")
print("="*80)

epa_preds = pd.read_csv('data/nfl/predictions/week_predictions.csv')
epa_preds = epa_preds[['game', 'home_team', 'away_team', 'total_pred']].rename(columns={'total_pred': 'epa_total'})

try:
    boxscore_preds = pd.read_csv('data/nfl/predictions/week11_boxscore.csv')
    boxscore_preds = boxscore_preds[['game', 'total_pred']].rename(columns={'total_pred': 'boxscore_total'})
except:
    print("WARNING: Box score predictions not found")
    boxscore_preds = pd.DataFrame({'game': epa_preds['game'], 'boxscore_total': np.nan})

try:
    pace_preds = pd.read_csv('data/nfl/predictions/week11_pace.csv')
    pace_preds = pace_preds[['game', 'total_pred']].rename(columns={'total_pred': 'pace_total'})
except:
    print("WARNING: Pace predictions not found")
    pace_preds = pd.DataFrame({'game': epa_preds['game'], 'pace_total': np.nan})

# Merge all predictions
ensemble = epa_preds.merge(boxscore_preds, on='game', how='left')
ensemble = ensemble.merge(pace_preds, on='game', how='left')

print(f"\nLoaded predictions for {len(ensemble)} games")
print(f"  EPA: {ensemble['epa_total'].notna().sum()} games")
print(f"  Box Score: {ensemble['boxscore_total'].notna().sum()} games")
print(f"  Pace: {ensemble['pace_total'].notna().sum()} games")

# ===========================
# 4. SIMPLE ENSEMBLE (AVERAGE)
# ===========================

print("\n" + "="*80)
print("CREATING ENSEMBLE")
print("="*80)

# Simple average where available
ensemble['ensemble_simple'] = ensemble[['epa_total', 'boxscore_total', 'pace_total']].mean(axis=1, skipna=True)

# Count how many models contributed
ensemble['num_models'] = ensemble[['epa_total', 'boxscore_total', 'pace_total']].notna().sum(axis=1)

# Calculate standard deviation across models (measure of agreement)
ensemble['model_std'] = ensemble[['epa_total', 'boxscore_total', 'pace_total']].std(axis=1, skipna=True)

print(f"\nEnsemble created:")
print(f"  All 3 models: {(ensemble['num_models'] == 3).sum()} games")
print(f"  2 models: {(ensemble['num_models'] == 2).sum()} games")
print(f"  1 model: {(ensemble['num_models'] == 1).sum()} games")

# ===========================
# 5. ADD MARKET LINES
# ===========================

print("\n" + "="*80)
print("COMPARING TO MARKET")
print("="*80)

# Add market lines (from your earlier data)
market_lines = {
    'NYJ @ NE': 43.5, 'SF @ ARI': 48.0, 'CAR @ ATL': 48.5, 'TB @ BUF': 50.5,
    'BAL @ CLE': 43.5, 'KC @ DEN': 44.5, 'LAC @ JAX': 44.5, 'SEA @ LA': 48.5,
    'WAS @ MIA': 47.5, 'CHI @ MIN': 46.5, 'GB @ NYG': 44.5, 'DET @ PHI': 49.0,
    'CIN @ PIT': 49.5, 'HOU @ TEN': 38.0, 'DAL @ LV': 50.5
}

ensemble['market_total'] = ensemble['game'].map(market_lines)
ensemble['diff_vs_market'] = ensemble['ensemble_simple'] - ensemble['market_total']

# ===========================
# 6. IDENTIFY HIGH-CONFIDENCE BETS
# ===========================

print("\n" + "="*80)
print("HIGH-CONFIDENCE BET IDENTIFICATION")
print("="*80)

# Criteria for high-confidence bet:
# 1. All 3 models agree (num_models == 3)
# 2. Models have low disagreement (std < 3 points)
# 3. Ensemble differs from market by 3+ points
# 4. Models mostly agree on direction vs market

ensemble['high_confidence'] = (
    (ensemble['num_models'] == 3) &
    (ensemble['model_std'] < 3.0) &
    (ensemble['diff_vs_market'].abs() >= 3.0)
)

high_conf = ensemble[ensemble['high_confidence']].copy()

print(f"\n{'='*80}")
print(f"WEEK 11 BETTING OPPORTUNITIES")
print(f"{'='*80}\n")

if len(high_conf) > 0:
    print(f"Found {len(high_conf)} high-confidence opportunities:\n")

    for _, row in high_conf.iterrows():
        print(f"{'='*80}")
        print(f"Game: {row['game']}")
        print(f"-"*80)
        print(f"  EPA Model:       {row['epa_total']:.1f}")
        print(f"  Box Score Model: {row['boxscore_total']:.1f}")
        print(f"  Pace Model:      {row['pace_total']:.1f}")
        print(f"  Model Std Dev:   {row['model_std']:.1f} (agreement: {'HIGH' if row['model_std'] < 2 else 'MODERATE'})")
        print(f"-"*80)
        print(f"  Ensemble:        {row['ensemble_simple']:.1f}")
        print(f"  Market:          {row['market_total']:.1f}")
        print(f"  Edge:            {row['diff_vs_market']:+.1f} points")
        print(f"-"*80)
        if row['diff_vs_market'] > 0:
            print(f"  RECOMMENDATION:  BET OVER {row['market_total']}")
        else:
            print(f"  RECOMMENDATION:  BET UNDER {row['market_total']}")
        print()
else:
    print("No high-confidence bets found this week.")
    print("\nGames with 3+ point edges (lower confidence):\n")

    med_conf = ensemble[(ensemble['diff_vs_market'].abs() >= 3.0)]

    for _, row in med_conf.iterrows():
        print(f"{row['game']:20} | Ensemble: {row['ensemble_simple']:5.1f} | Market: {row['market_total']:5.1f} | Edge: {row['diff_vs_market']:+5.1f} | Models: {int(row['num_models'])} | Std: {row['model_std']:.1f}")

print(f"\n{'='*80}")
print("FULL PREDICTIONS")
print(f"{'='*80}\n")

display_cols = ['game', 'epa_total', 'boxscore_total', 'pace_total', 'ensemble_simple', 'market_total', 'diff_vs_market', 'model_std']
print(ensemble[display_cols].to_string(index=False))

# Save
ensemble.to_csv('data/nfl/predictions/week11_ensemble.csv', index=False)
print(f"\n✓ Saved to data/nfl/predictions/week11_ensemble.csv")

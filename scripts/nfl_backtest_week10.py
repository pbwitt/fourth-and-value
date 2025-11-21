"""
Backtest Week 10 using only data through Week 9
Proper out-of-sample test
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import pickle

print("="*80)
print("WEEK 10 BACKTEST - TRAINING ON WEEKS 2-9 ONLY")
print("="*80)

# Load PBP data
pbp = pd.read_parquet('data/pbp/pbp_2025_updated.parquet')

# Split into training (weeks 2-9) and test (week 10)
pbp_train = pbp[pbp['week'].between(2, 9)].copy()
pbp_test = pbp[pbp['week'] == 10].copy()

print(f"\nTraining data: Weeks 2-9")
print(f"Test data: Week 10")

# Get Week 10 actuals
week10_actuals = pbp_test.groupby(['game_id', 'home_team', 'away_team']).agg({
    'home_score': 'max',
    'away_score': 'max',
    'total': 'first',
    'total_line': 'first'
}).reset_index()

print(f"Week 10 games: {len(week10_actuals)}")

# ============================================================================
# 1. TRAIN EPA MODEL (Weeks 2-9)
# ============================================================================

print("\n" + "="*80)
print("[1/3] TRAINING EPA MODEL (Weeks 2-9)")
print("="*80)

# Build EPA features from training data
from sklearn.model_selection import TimeSeriesSplit

def build_epa_features(pbp_data):
    """Build EPA-based team features"""
    games = pbp_data.groupby(['game_id', 'week', 'posteam', 'defteam', 'home_team', 'away_team']).agg({
        'home_score': 'max',
        'away_score': 'max',
        'total': 'first',
        'epa': 'mean',
        'success': 'mean'
    }).reset_index()

    games.columns = ['game_id', 'week', 'team', 'opponent', 'home_team', 'away_team',
                     'home_score', 'away_score', 'total', 'off_epa', 'off_success']

    # Split home/away
    home_games = games[games['team'] == games['home_team']].copy()
    away_games = games[games['team'] == games['away_team']].copy()

    # Rolling features
    def add_rolling(df, team_col, features, windows=[3, 5]):
        for team in df[team_col].unique():
            mask = df[team_col] == team
            team_df = df[mask].copy().sort_values('week')
            for window in windows:
                for feat in features:
                    df.loc[mask, f'{feat}_L{window}'] = team_df[feat].shift(1).rolling(window, min_periods=1).mean().values
        return df

    home_games = add_rolling(home_games, 'home_team', ['off_epa', 'off_success'])
    away_games = add_rolling(away_games, 'away_team', ['off_epa', 'off_success'])

    # Merge
    merged = home_games[['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'total'] +
                        [c for c in home_games.columns if '_L3' in c or '_L5' in c]].merge(
        away_games[['game_id'] + [c for c in away_games.columns if '_L3' in c or '_L5' in c]],
        on='game_id', suffixes=('_home', '_away')
    )

    return merged

epa_train_data = build_epa_features(pbp_train)
epa_test_data = build_epa_features(pbp_test)

feature_cols = [c for c in epa_train_data.columns if '_L3' in c or '_L5' in c]
X_train = epa_train_data[feature_cols].fillna(0)
y_home_train = epa_train_data['home_score']
y_away_train = epa_train_data['away_score']

model_epa_home = Ridge(alpha=1.0)
model_epa_away = Ridge(alpha=1.0)
model_epa_home.fit(X_train, y_home_train)
model_epa_away.fit(X_train, y_away_train)

print(f"✓ EPA model trained on {len(X_train)} games")

# ============================================================================
# 2. TRAIN BOX SCORE MODEL (Weeks 2-9)
# ============================================================================

print("\n" + "="*80)
print("[2/3] TRAINING BOX SCORE MODEL (Weeks 2-9)")
print("="*80)

def build_boxscore_features(pbp_data):
    """Build box score features"""
    games = pbp_data.groupby(['game_id', 'week', 'posteam', 'home_team', 'away_team']).agg({
        'home_score': 'max',
        'away_score': 'max',
        'total': 'first',
        'yards_gained': 'sum',
        'pass_touchdown': 'sum',
        'rush_touchdown': 'sum',
        'interception': 'sum',
        'fumble_lost': 'sum'
    }).reset_index()

    games['total_tds'] = games['pass_touchdown'] + games['rush_touchdown']
    games['turnovers'] = games['interception'] + games['fumble_lost']

    home_games = games[games['posteam'] == games['home_team']].copy()
    away_games = games[games['posteam'] == games['away_team']].copy()

    # Rolling features
    def add_rolling(df, team_col, features):
        for team in df[team_col].unique():
            mask = df[team_col] == team
            team_df = df[mask].copy().sort_values('week')
            for feat in features:
                df.loc[mask, f'{feat}_L3'] = team_df[feat].shift(1).rolling(3, min_periods=1).mean().values
        return df

    home_games = add_rolling(home_games, 'home_team', ['yards_gained', 'total_tds', 'turnovers'])
    away_games = add_rolling(away_games, 'away_team', ['yards_gained', 'total_tds', 'turnovers'])

    # Merge
    merged = home_games[['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score'] +
                        [c for c in home_games.columns if '_L3' in c]].merge(
        away_games[['game_id'] + [c for c in away_games.columns if '_L3' in c]],
        on='game_id', suffixes=('_home', '_away')
    )

    return merged

box_train_data = build_boxscore_features(pbp_train)
box_test_data = build_boxscore_features(pbp_test)

feature_cols_box = [c for c in box_train_data.columns if '_L3' in c]
X_train_box = box_train_data[feature_cols_box].fillna(0)
y_home_train_box = box_train_data['home_score']
y_away_train_box = box_train_data['away_score']

model_box_home = Ridge(alpha=1.0)
model_box_away = Ridge(alpha=1.0)
model_box_home.fit(X_train_box, y_home_train_box)
model_box_away.fit(X_train_box, y_away_train_box)

print(f"✓ Box score model trained on {len(X_train_box)} games")

# ============================================================================
# 3. TRAIN PACE MODEL (Weeks 2-9)
# ============================================================================

print("\n" + "="*80)
print("[3/3] TRAINING PACE MODEL (Weeks 2-9)")
print("="*80)

def build_pace_features(pbp_data):
    """Build pace features"""
    games = pbp_data.groupby(['game_id', 'week', 'posteam', 'home_team', 'away_team']).agg({
        'home_score': 'max',
        'away_score': 'max',
        'total': 'first',
        'play_id': 'count',
        'yards_gained': 'sum'
    }).reset_index()

    games['yards_per_play'] = games['yards_gained'] / (games['play_id'] + 0.001)

    home_games = games[games['posteam'] == games['home_team']].copy()
    away_games = games[games['posteam'] == games['away_team']].copy()

    # Rolling features
    def add_rolling(df, team_col, features):
        for team in df[team_col].unique():
            mask = df[team_col] == team
            team_df = df[mask].copy().sort_values('week')
            for feat in features:
                df.loc[mask, f'{feat}_L3'] = team_df[feat].shift(1).rolling(3, min_periods=1).mean().values
        return df

    home_games = add_rolling(home_games, 'home_team', ['play_id', 'yards_per_play'])
    away_games = add_rolling(away_games, 'away_team', ['play_id', 'yards_per_play'])

    # Merge
    merged = home_games[['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score'] +
                        [c for c in home_games.columns if '_L3' in c]].merge(
        away_games[['game_id'] + [c for c in away_games.columns if '_L3' in c]],
        on='game_id', suffixes=('_home', '_away')
    )

    return merged

pace_train_data = build_pace_features(pbp_train)
pace_test_data = build_pace_features(pbp_test)

feature_cols_pace = [c for c in pace_train_data.columns if '_L3' in c]
X_train_pace = pace_train_data[feature_cols_pace].fillna(0)
y_home_train_pace = pace_train_data['home_score']
y_away_train_pace = pace_train_data['away_score']

model_pace_home = Ridge(alpha=1.0)
model_pace_away = Ridge(alpha=1.0)
model_pace_home.fit(X_train_pace, y_home_train_pace)
model_pace_away.fit(X_train_pace, y_away_train_pace)

print(f"✓ Pace model trained on {len(X_train_pace)} games")

# ============================================================================
# 4. GENERATE WEEK 10 PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING WEEK 10 PREDICTIONS")
print("="*80)

predictions = []

for _, actual in week10_actuals.iterrows():
    game_id = actual['game_id']
    home = actual['home_team']
    away = actual['away_team']

    # EPA prediction
    epa_game = epa_test_data[epa_test_data['game_id'] == game_id]
    if len(epa_game) > 0:
        X = epa_game[feature_cols].fillna(0)
        epa_pred = model_epa_home.predict(X)[0] + model_epa_away.predict(X)[0]
    else:
        epa_pred = np.nan

    # Box score prediction
    box_game = box_test_data[box_test_data['game_id'] == game_id]
    if len(box_game) > 0:
        X = box_game[feature_cols_box].fillna(0)
        box_pred = model_box_home.predict(X)[0] + model_box_away.predict(X)[0]
    else:
        box_pred = np.nan

    # Pace prediction
    pace_game = pace_test_data[pace_test_data['game_id'] == game_id]
    if len(pace_game) > 0:
        X = pace_game[feature_cols_pace].fillna(0)
        pace_pred = model_pace_home.predict(X)[0] + model_pace_away.predict(X)[0]
    else:
        pace_pred = np.nan

    # Ensemble
    preds = [epa_pred, box_pred, pace_pred]
    preds_valid = [p for p in preds if not np.isnan(p)]

    if len(preds_valid) > 0:
        ensemble_pred = np.mean(preds_valid)
        model_std = np.std(preds_valid) if len(preds_valid) > 1 else 0

        predictions.append({
            'game_id': game_id,
            'home_team': home,
            'away_team': away,
            'epa_pred': epa_pred,
            'box_pred': box_pred,
            'pace_pred': pace_pred,
            'ensemble_pred': ensemble_pred,
            'model_std': model_std,
            'num_models': len(preds_valid),
            'actual_total': actual['total'],
            'market_line': actual['total_line']
        })

results = pd.DataFrame(predictions)

# Apply simple calibration (subtract mean bias from training)
train_preds = []
for _, game in epa_train_data.iterrows():
    if game['home_score'] > 0:
        X_epa = game[feature_cols].values.reshape(1, -1)
        epa_p = model_epa_home.predict(X_epa)[0] + model_epa_away.predict(X_epa)[0]
        train_preds.append(epa_p - game['total'])

mean_bias = np.mean(train_preds)
results['calibrated_pred'] = results['ensemble_pred'] - mean_bias

print(f"\nGenerated predictions for {len(results)} games")
print(f"Mean training bias: {mean_bias:.2f} points")

# ============================================================================
# 5. EVALUATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("WEEK 10 BACKTEST RESULTS")
print("="*80)

results['error'] = results['ensemble_pred'] - results['actual_total']
results['calib_error'] = results['calibrated_pred'] - results['actual_total']
results['market_error'] = results['market_line'] - results['actual_total']

print(f"\nModel Performance:")
print(f"  Ensemble MAE: {results['error'].abs().mean():.2f} points")
print(f"  Calibrated MAE: {results['calib_error'].abs().mean():.2f} points")
print(f"  Market MAE: {results['market_error'].abs().mean():.2f} points")

# ============================================================================
# 6. SIMULATE BETTING
# ============================================================================

print("\n" + "="*80)
print("BETTING SIMULATION")
print("="*80)

# Identify bets where we have 3+ point edge
results['edge'] = results['calibrated_pred'] - results['market_line']
results['bet'] = (results['edge'].abs() >= 3.0) & (results['num_models'] == 3) & (results['model_std'] < 4.0)

bets = results[results['bet']].copy()

print(f"\nBet Criteria:")
print(f"  - Edge: 3+ points")
print(f"  - All 3 models agree")
print(f"  - Model std < 4.0")
print(f"\nFound {len(bets)} qualifying bets")

if len(bets) > 0:
    # Determine bet outcome
    bets['bet_side'] = np.where(bets['edge'] > 0, 'OVER', 'UNDER')

    # Win if:
    # - Bet OVER and actual > market line
    # - Bet UNDER and actual < market line
    bets['bet_won'] = np.where(
        bets['bet_side'] == 'OVER',
        bets['actual_total'] > bets['market_line'],
        bets['actual_total'] < bets['market_line']
    )

    # Calculate profit (assuming -110 odds)
    bets['profit'] = np.where(bets['bet_won'], 100, -110)

    print(f"\n{'='*80}")
    print("BET RESULTS")
    print(f"{'='*80}\n")

    for _, bet in bets.iterrows():
        game = f"{bet['away_team']} @ {bet['home_team']}"
        result = "✓ WIN" if bet['bet_won'] else "✗ LOSS"
        print(f"{game:20} | {bet['bet_side']:5} {bet['market_line']:.1f} | Actual: {bet['actual_total']:.0f} | {result} | {bet['profit']:+4.0f}")

    wins = bets['bet_won'].sum()
    losses = len(bets) - wins
    win_pct = wins / len(bets) * 100
    total_profit = bets['profit'].sum()
    roi = (total_profit / (len(bets) * 110)) * 100

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Record: {wins}-{losses} ({win_pct:.1f}%)")
    print(f"  Profit/Loss: ${total_profit:+.0f}")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Breakeven needed: 52.38%")

    if win_pct > 52.38:
        print(f"  ✓ PROFITABLE (+{win_pct - 52.38:.1f}% above breakeven)")
    else:
        print(f"  ✗ UNPROFITABLE ({52.38 - win_pct:.1f}% below breakeven)")

else:
    print("\nNo bets met the criteria for Week 10")

# Show all games
print(f"\n{'='*80}")
print("ALL WEEK 10 GAMES")
print(f"{'='*80}\n")

for _, row in results.iterrows():
    game = f"{row['away_team']} @ {row['home_team']}"
    bet_flag = "📍 BET" if row['bet'] else ""
    print(f"{game:20} | Pred: {row['calibrated_pred']:5.1f} | Mkt: {row['market_line']:5.1f} | Act: {row['actual_total']:5.0f} | Edge: {row['edge']:+5.1f} | {bet_flag}")

print("\n✓ Backtest complete!")

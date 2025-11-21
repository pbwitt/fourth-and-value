"""
Run Week 11 predictions using legacy Lasso model approach
Mimics PDubbs Over-Under Predictions-9-25-21.ipynb logic
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Loading historical player stats...")
data = pd.read_csv('data/nfl/processed/player_stats_historical.csv')

# Week 11 schedule
schedule = pd.DataFrame({
    'home_team': ['PHI', 'CHI', 'CLE', 'JAX', 'DET', 'IND', 'NO', 'MIN', 'LV', 'LAR', 'KC', 'SEA', 'GB', 'HOU', 'BAL'],
    'away_team': ['WAS', 'GB', 'PIT', 'DET', 'JAX', 'NYJ', 'CLE', 'TEN', 'MIA', 'NE', 'BUF', 'SF', 'CHI', 'DAL', 'PIT']
})

print(f"\nWeek 11 Schedule:")
print(schedule)

# Get Week 10 stats for blending with historical
week10_data = data[(data['season'] == 2025) & (data['week'] == 10)].copy()

results = []

for idx, game in schedule.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']

    print(f"\n{'='*60}")
    print(f"Predicting: {away_team} @ {home_team}")
    print(f"{'='*60}")

    # Get historical matchups: home team at home vs this opponent
    # OR home team playing this opponent in general
    historical = data[
        ((data['home_team'] == home_team) & (data['away_team'] == away_team)) |
        ((data['recent_team'] == home_team) & (data['opponent_team'] == away_team))
    ].copy()

    if len(historical) < 20:
        print(f"  Limited history ({len(historical)} records), using home team history...")
        historical = data[
            (data['recent_team'] == home_team) & (data['season'] >= 2022)
        ].copy()

    # Get Week 10 players who will likely play in Week 11
    week10_home = week10_data[week10_data['recent_team'] == home_team].copy()
    week10_away = week10_data[week10_data['recent_team'] == away_team].copy()

    # Combine historical + Week 10 (blend recent with history)
    combined = pd.concat([historical, week10_home, week10_away], ignore_index=True)

    print(f"  Historical records: {len(historical)}")
    print(f"  Week 10 home players: {len(week10_home)}")
    print(f"  Week 10 away players: {len(week10_away)}")
    print(f"  Combined: {len(combined)}")

    # Aggregate to team level (sum player stats)
    # Group by game (season, week, recent_team) and sum
    team_agg = combined.groupby(['season', 'week', 'recent_team', 'home_team', 'away_team', 'home_score', 'away_score']).agg({
        'completions': 'sum',
        'attempts': 'sum',
        'passing_yards': 'sum',
        'passing_tds': 'sum',
        'interceptions': 'sum',
        'sacks': 'sum',
        'carries': 'sum',
        'rushing_yards': 'sum',
        'rushing_tds': 'sum',
        'receptions': 'sum',
        'targets': 'sum',
        'receiving_yards': 'sum',
        'receiving_tds': 'sum',
        'fantasy_points': 'sum',
        'fantasy_points_ppr': 'sum'
    }).reset_index()

    # Create feature matrix
    feature_cols = [
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
        'carries', 'rushing_yards', 'rushing_tds',
        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
        'fantasy_points', 'fantasy_points_ppr'
    ]

    # Split into training (historical with scores) and prediction (Week 10)
    train_data = team_agg[(team_agg['home_score'] > 0)].copy()

    if len(train_data) < 10:
        print(f"  WARNING: Only {len(train_data)} training samples - predictions may be unreliable")

    # Prepare training data
    X_train = train_data[feature_cols].fillna(0)
    y_home = train_data['home_score']
    y_away = train_data['away_score']

    # Create prediction data (average of Week 10 for each team)
    week10_home_agg = week10_home[feature_cols].fillna(0).mean().to_frame().T
    week10_away_agg = week10_away[feature_cols].fillna(0).mean().to_frame().T

    # Train home score model
    try:
        model_home = linear_model.Lasso(alpha=1.0)
        model_home.fit(X_train, y_home)

        # Train away score model
        model_away = linear_model.Lasso(alpha=1.0)
        model_away.fit(X_train, y_away)

        # Predict using Week 10 averages
        home_pred = model_home.predict(week10_home_agg)[0]
        away_pred = model_away.predict(week10_away_agg)[0]

        total_pred = home_pred + away_pred

        print(f"\n  PREDICTIONS:")
        print(f"    {home_team} (home): {home_pred:.1f}")
        print(f"    {away_team} (away): {away_pred:.1f}")
        print(f"    Total: {total_pred:.1f}")

        results.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_pred': home_pred,
            'away_pred': away_pred,
            'total_pred': total_pred,
            'training_samples': len(train_data)
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_pred': np.nan,
            'away_pred': np.nan,
            'total_pred': np.nan,
            'training_samples': len(train_data)
        })

# Create results dataframe
results_df = pd.DataFrame(results)

print(f"\n{'='*60}")
print("WEEK 11 PREDICTIONS SUMMARY")
print(f"{'='*60}\n")
print(results_df.to_string(index=False))

# Save
results_df.to_csv('data/nfl/predictions/week11_legacy_lasso.csv', index=False)
print(f"\nSaved to: data/nfl/predictions/week11_legacy_lasso.csv")

# Load consensus lines
print(f"\nComparing to market lines...")
try:
    # Read from temp files you showed earlier
    import glob
    week10_files = glob.glob('/tmp/week10*.csv')
    week11_files = glob.glob('/tmp/week11*.csv')

    if week11_files:
        consensus = pd.read_csv(week11_files[0])
        print(f"\nFound Week 11 consensus file: {week11_files[0]}")
        print(consensus.head())
    else:
        print("No Week 11 consensus file found in /tmp/")

except Exception as e:
    print(f"Could not load consensus: {e}")

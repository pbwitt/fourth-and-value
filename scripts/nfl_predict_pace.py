"""
Generate pace-based predictions for upcoming week
"""

import pandas as pd
import pickle
import numpy as np

print("Loading pace model...")
with open('data/nfl/models/pace_totals.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("Loading pace features...")
pace_df = pd.read_csv('data/nfl/processed/team_pace_features.csv')

# Load schedule
schedule = pd.read_csv('data/schedule_2025.csv')
week_11 = schedule[schedule['week'] == 11].copy()

print(f"Predicting {len(week_11)} games...")

predictions = []

for _, game in week_11.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']

    # Get most recent features for each team
    home_data = pace_df[pace_df['home_team'] == home_team].sort_values('week', ascending=False).head(1)
    away_data = pace_df[pace_df['away_team'] == away_team].sort_values('week', ascending=False).head(1)

    # If no exact match, try to build from team-level data
    if len(home_data) == 0 or len(away_data) == 0:
        # Get any game where team played
        home_any = pace_df[
            (pace_df['home_team'] == home_team) | (pace_df['away_team'] == home_team)
        ].sort_values('week', ascending=False).head(1)

        away_any = pace_df[
            (pace_df['home_team'] == away_team) | (pace_df['away_team'] == away_team)
        ].sort_values('week', ascending=False).head(1)

        if len(home_any) > 0 and len(away_any) > 0:
            # Extract home and away features
            home_features = {col: home_any.iloc[0][col] for col in model_data['features'] if col.startswith('home_')}
            away_features = {col: away_any.iloc[0][col] for col in model_data['features'] if col.startswith('away_')}

            # Combine into prediction row
            pred_row = pd.DataFrame([{**home_features, **away_features}])
            pred_row = pred_row[model_data['features']].fillna(0)

            if len(pred_row) > 0:
                home_pred = model_data['model_home'].predict(pred_row)[0]
                away_pred = model_data['model_away'].predict(pred_row)[0]

                predictions.append({
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_pred': home_pred,
                    'away_pred': away_pred,
                    'total_pred': home_pred + away_pred
                })
                continue

    # Use matched data
    if len(home_data) > 0 and len(away_data) > 0:
        # Try to find a game between these teams or construct features
        combined = pace_df[
            ((pace_df['home_team'] == home_team) & (pace_df['away_team'] == away_team)) |
            ((pace_df['home_team'] == away_team) & (pace_df['away_team'] == home_team))
        ].sort_values('week', ascending=False).head(1)

        if len(combined) > 0:
            features = combined[model_data['features']].fillna(0)
        else:
            # Build synthetic features from most recent team data
            features_dict = {}

            for feat in model_data['features']:
                if feat.startswith('home_'):
                    if feat in home_data.columns:
                        features_dict[feat] = home_data.iloc[0][feat]
                    else:
                        features_dict[feat] = 0
                elif feat.startswith('away_'):
                    if feat in away_data.columns:
                        features_dict[feat] = away_data.iloc[0][feat]
                    else:
                        features_dict[feat] = 0

            features = pd.DataFrame([features_dict])[model_data['features']].fillna(0)

        if len(features) > 0:
            home_pred = model_data['model_home'].predict(features)[0]
            away_pred = model_data['model_away'].predict(features)[0]

            predictions.append({
                'game': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'home_pred': home_pred,
                'away_pred': away_pred,
                'total_pred': home_pred + away_pred
            })

if len(predictions) == 0:
    print("WARNING: No predictions generated")
else:
    results_df = pd.DataFrame(predictions)
    results_df.to_csv('data/nfl/predictions/week11_pace.csv', index=False)
    print(f"\n✓ Saved {len(predictions)} pace predictions to data/nfl/predictions/week11_pace.csv")
    print("\nSample predictions:")
    print(results_df[['game', 'total_pred']].head())

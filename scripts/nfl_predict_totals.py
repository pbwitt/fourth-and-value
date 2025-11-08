"""
Generate NFL totals predictions for upcoming week
"""
import pandas as pd
import pickle
import os
from datetime import datetime

def generate_predictions(model_path='data/nfl/models/ridge_totals.pkl',
                        team_features_path='data/nfl/processed/team_features.csv',
                        week=None,
                        output_path='data/nfl/predictions/week_predictions.csv'):
    """
    Generate predictions for upcoming week
    """
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['features']

    print(f"Model CV MAE: {model_data['cv_mae']:.2f} points")

    # Load team features
    print("\nLoading team features...")
    df = pd.read_csv(team_features_path)

    # If week not specified, use next week
    if week is None:
        week = df['week'].max() + 1

    print(f"Generating predictions for Week {week}...")

    # Get latest features for each team
    latest = df.groupby('team').apply(lambda x: x.sort_values('week').iloc[-1]).reset_index(drop=True)

    # Prepare for predictions (we'll need to specify matchups)
    # For now, just show what each team would contribute
    predictions = []

    # Get this week's schedule if available
    try:
        schedule = pd.read_csv('data/schedule_2025.csv')
        week_games = schedule[schedule['week'] == week]

        for _, game in week_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']

            if home_team not in latest['team'].values or away_team not in latest['team'].values:
                print(f"  Warning: Missing data for {away_team} @ {home_team}, skipping")
                continue

            home_row = latest[latest['team'] == home_team]
            away_row = latest[latest['team'] == away_team]

            # Build feature vectors (remove 'is_home' from feature_cols if it exists, add it manually)
            base_features = [f for f in feature_cols if f != 'is_home']

            home_features = home_row[base_features].values[0].tolist() + [1]  # is_home=1
            away_features = away_row[base_features].values[0].tolist() + [0]  # is_home=0

            # Predict home and away separately
            home_pred = model.predict([home_features])[0]
            away_pred = model.predict([away_features])[0]

            # Total prediction
            total_pred = home_pred + away_pred

            predictions.append({
                'game': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'week': week,
                'home_pred': home_pred,
                'away_pred': away_pred,
                'total_pred': total_pred
            })

    except FileNotFoundError:
        print("  Warning: No schedule file found, showing team totals only")

        for _, team_row in latest.iterrows():
            team_features = team_row[feature_cols].values
            pred = model.predict([team_features.reshape(1, -1)])[0]

            predictions.append({
                'team': team_row['team'],
                'predicted_contribution': pred
            })

    # Save predictions
    preds_df = pd.DataFrame(predictions)

    if len(preds_df) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        preds_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(preds_df)} predictions to {output_path}")
        print("\nPredictions:")
        print(preds_df.to_string(index=False))
    else:
        print("\n⚠ No predictions generated")

    return preds_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate NFL totals predictions')
    parser.add_argument('--model', default='data/nfl/models/ridge_totals.pkl', help='Model pickle file')
    parser.add_argument('--team-features', default='data/nfl/processed/team_features.csv', help='Team features CSV')
    parser.add_argument('--week', type=int, help='Week to predict (default: next week)')
    parser.add_argument('--output', default='data/nfl/predictions/week_predictions.csv', help='Output CSV')

    args = parser.parse_args()

    generate_predictions(args.model, args.team_features, args.week, args.output)

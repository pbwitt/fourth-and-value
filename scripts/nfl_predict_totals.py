"""
Generate NFL totals predictions for upcoming week
"""
import pandas as pd
import pickle
import os
from datetime import datetime

def load_schedule(season):
    """
    Season schedule with kickoff dates. Cached per season so a rebuild does not
    depend on the network, and so the season is always explicit rather than
    inherited from whichever schedule file happened to be on disk.
    """
    cache = f'data/schedule_{season}.csv'
    if os.path.exists(cache):
        return pd.read_csv(cache)

    import nfl_data_py as nfl
    schedule = nfl.import_schedules([season])
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    schedule.to_csv(cache, index=False)
    return schedule


def generate_predictions(model_path='data/nfl/models/ridge_totals.pkl',
                        team_features_path='data/nfl/processed/team_features.csv',
                        week=None,
                        output_path='data/nfl/predictions/week_predictions.csv',
                        season=None):
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

    if season is None:
        season = int(df['season'].max()) if 'season' in df.columns else None
    if season is None:
        raise ValueError('season is required (team features carry no season column)')

    # If week not specified, use next week within the target season
    if week is None:
        season_weeks = df.loc[df['season'] == season, 'week']
        week = int(season_weeks.max()) + 1 if len(season_weeks) else 1

    print(f"Generating predictions for {season} Week {week}...")

    predictions = []

    schedule = load_schedule(season)
    week_games = schedule[schedule['week'] == week]
    if len(week_games) == 0:
        raise ValueError(f'No games found in the {season} schedule for week {week}')

    # Rolling form is carried across the season break, so Week 1 is predicted
    # from the prior season's closing form. Only games that kicked off before
    # this one may inform it.
    cutoff = str(week_games['gameday'].min())
    history = df[df['game_date'].astype(str) < cutoff]
    if len(history) == 0:
        raise ValueError(f'No team-game history before {cutoff}; cannot build features')

    latest = (history.sort_values(['game_date', 'game_id'])
                     .groupby('team', as_index=False)
                     .tail(1)
                     .reset_index(drop=True))
    print(f"Carrying form from games before {cutoff} "
          f"({latest['game_date'].min()} to {latest['game_date'].max()}, {len(latest)} teams)")

    if True:
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
                'season': season,
                'week': week,
                'gameday': game['gameday'],
                'home_pred': home_pred,
                'away_pred': away_pred,
                'total_pred': total_pred
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
    parser.add_argument('--season', type=int, required=True, help='Season to predict (e.g. 2026)')
    parser.add_argument('--output', default='data/nfl/predictions/week_predictions.csv', help='Output CSV')

    args = parser.parse_args()

    generate_predictions(args.model, args.team_features, args.week, args.output, args.season)

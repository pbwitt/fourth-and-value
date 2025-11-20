"""
Generate predictions for today's NHL games
Integrates with odds data
"""
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import os
import sys


def fetch_todays_odds(api_key):
    """
    Fetch NHL totals odds from The Odds API
    """
    SPORT = 'icehockey_nhl'
    REGIONS = 'us'
    MARKETS = 'totals'
    ODDS_FORMAT = 'american'

    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'

    params = {
        'api_key': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching odds: {e}")

    return []


def parse_odds(odds_json):
    """
    Parse odds JSON into dataframe
    """
    odds_data = []

    for game in odds_json:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']

        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker['key']

            for market in bookmaker.get('markets', []):
                if market['key'] == 'totals':
                    for outcome in market['outcomes']:
                        odds_data.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'bookmaker': book_name,
                            'name': outcome['name'],
                            'point': outcome.get('point'),
                            'price': outcome.get('price')
                        })

    return pd.DataFrame(odds_data)


def calculate_devigged_prob(over_price, under_price):
    """
    Remove vig to get fair probability
    """
    def price_to_prob(price):
        if price > 0:
            return 100 / (price + 100)
        else:
            return abs(price) / (abs(price) + 100)

    over_prob = price_to_prob(over_price)
    under_prob = price_to_prob(under_price)

    # Remove vig
    total = over_prob + under_prob
    over_fair = over_prob / total
    under_fair = under_prob / total

    return over_fair, under_fair


def load_recent_team_stats(team_features_path, team_name, n_games=5):
    """
    Load recent stats for a team to generate features for prediction
    """
    team_df = pd.read_csv(team_features_path)

    # Filter to team
    team_data = team_df[team_df['team'] == team_name].copy()
    team_data = team_data.sort_values('game_date', ascending=False)

    if len(team_data) == 0:
        print(f"Warning: No data found for team {team_name}")
        return None

    # Get most recent game for features
    latest = team_data.iloc[0]

    return latest


def generate_predictions(model_path, feature_list_path, team_features_path, todays_games):
    """
    Generate predictions for today's games
    """
    # Load model
    model = joblib.load(model_path)

    # Load feature list
    with open(feature_list_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    predictions = []

    for _, game in todays_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']

        # Load recent stats for both teams
        home_stats = load_recent_team_stats(team_features_path, home_team)
        away_stats = load_recent_team_stats(team_features_path, away_team)

        if home_stats is None or away_stats is None:
            print(f"Skipping {away_team} @ {home_team} - missing data")
            continue

        # Prepare features for home team
        home_features = home_stats[feature_cols].to_frame().T
        home_features['is_home'] = 1  # Override to home

        # Prepare features for away team
        away_features = away_stats[feature_cols].to_frame().T
        away_features['is_home'] = 0  # Override to away

        # Check for missing data and warn
        home_missing = home_features.isna().sum()
        away_missing = away_features.isna().sum()

        if home_missing.sum() > 0:
            print(f"  ⚠️  WARNING: {home_team} has {home_missing.sum()} missing features, using column means as fallback")
            print(f"     Missing columns: {list(home_missing[home_missing > 0].index)}")

        if away_missing.sum() > 0:
            print(f"  ⚠️  WARNING: {away_team} has {away_missing.sum()} missing features, using column means as fallback")
            print(f"     Missing columns: {list(away_missing[away_missing > 0].index)}")

        # Fill NaN with column means (fallback)
        home_features = home_features.fillna(home_features.mean())
        away_features = away_features.fillna(away_features.mean())

        # Predict
        home_pred = model.predict(home_features)[0]
        away_pred = model.predict(away_features)[0]

        total_pred = home_pred + away_pred

        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_pred': round(home_pred, 2),
            'away_pred': round(away_pred, 2),
            'total_pred': round(total_pred, 2)
        })

    return pd.DataFrame(predictions)


def main():
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description='Predict NHL totals for today')
    parser.add_argument('--model', default='data/nhl/models/ridge_team_totals.pkl', help='Model path')
    parser.add_argument('--features', default='data/nhl/models/feature_list.txt', help='Feature list')
    parser.add_argument('--team-features', default='data/nhl/processed/team_features.csv', help='Team features CSV')
    parser.add_argument('--output', default='data/nhl/predictions/today.csv', help='Output file')

    args = parser.parse_args()

    # Get today's schedule
    print("Fetching today's NHL schedule...")
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://api-web.nhle.com/v1/schedule/{today}"

    try:
        response = requests.get(url)
        games_today = []

        if response.status_code == 200:
            data = response.json()

            for week in data.get('gameWeek', []):
                for date_obj in week.get('games', []):
                    for game in [date_obj]:
                        if isinstance(game, dict):
                            games_today.append({
                                'game_id': game.get('id'),
                                'home_team': game.get('homeTeam', {}).get('abbrev'),
                                'away_team': game.get('awayTeam', {}).get('abbrev'),
                                'start_time': game.get('startTimeUTC')
                            })

        todays_games = pd.DataFrame(games_today)
        print(f"Found {len(todays_games)} games today")

        if len(todays_games) == 0:
            print("No games today")
            return

        # Generate predictions
        print("\nGenerating predictions...")
        preds = generate_predictions(args.model, args.features, args.team_features, todays_games)

        # Fetch odds
        api_key = os.getenv('ODDS_API_KEY')
        if api_key:
            print("\nFetching odds...")
            odds_json = fetch_todays_odds(api_key)
            odds_df = parse_odds(odds_json)

            # Get consensus line
            if len(odds_df) > 0:
                consensus = odds_df.groupby(['home_team', 'away_team']).agg({
                    'point': 'mean'
                }).reset_index().rename(columns={'point': 'market_total'})

                preds = preds.merge(consensus, on=['home_team', 'away_team'], how='left')
                preds['edge'] = preds['total_pred'] - preds['market_total']

        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        preds.to_csv(args.output, index=False)

        print(f"\nPredictions:")
        print(preds.to_string(index=False))
        print(f"\nSaved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

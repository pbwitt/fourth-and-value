"""
Fetch NHL game data from new NHL API
Similar to fetch_games.py for NFL
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

def get_todays_games():
    """Fetch today's NHL games"""
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://api-web.nhle.com/v1/schedule/{today}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            games = []

            for week in data.get('gameWeek', []):
                for date_obj in week.get('games', []):
                    for game in [date_obj]:
                        if isinstance(game, dict):
                            games.append({
                                'game_id': game.get('id'),
                                'season': game.get('season'),
                                'game_date': game.get('gameDate'),
                                'start_time': game.get('startTimeUTC'),
                                'home_team': game.get('homeTeam', {}).get('abbrev'),
                                'away_team': game.get('awayTeam', {}).get('abbrev'),
                                'venue': game.get('venue', {}).get('default')
                            })

            return pd.DataFrame(games)
    except Exception as e:
        print(f"Error fetching schedule: {e}")

    return pd.DataFrame()


def get_season_games(season='20242025', start_date=None, end_date=None):
    """
    Fetch historical games for a season
    Args:
        season: YYYYYYYY format (e.g., '20242025' for 2024-25 season)
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
    """
    if not start_date:
        # Default to start of season (October 10)
        year = int(season[:4])
        start_date = f"{year}-10-01"

    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    all_games = []

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        url = f"https://api-web.nhle.com/v1/schedule/{date_str}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()

                for week in data.get('gameWeek', []):
                    for date_obj in week.get('games', []):
                        for game in [date_obj]:
                            if isinstance(game, dict) and game.get('season') == int(season):
                                all_games.append({
                                    'game_id': game.get('id'),
                                    'season': game.get('season'),
                                    'game_type': game.get('gameType'),
                                    'game_date': game.get('gameDate'),
                                    'start_time': game.get('startTimeUTC'),
                                    'home_team': game.get('homeTeam', {}).get('abbrev'),
                                    'away_team': game.get('awayTeam', {}).get('abbrev'),
                                    'venue': game.get('venue', {}).get('default'),
                                    'home_score': game.get('homeTeam', {}).get('score'),
                                    'away_score': game.get('awayTeam', {}).get('score'),
                                    'game_state': game.get('gameState')
                                })

            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {date_str}: {e}")

        current += timedelta(days=1)

    return pd.DataFrame(all_games)


def get_game_boxscore(game_id):
    """Fetch detailed boxscore for a game"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")

    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch NHL games')
    parser.add_argument('--season', default='20242025', help='Season in YYYYYYYY format')
    parser.add_argument('--start-date', help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', help='End date YYYY-MM-DD')
    parser.add_argument('--output', default='data/nhl/raw/games.csv', help='Output file')

    args = parser.parse_args()

    print(f"Fetching NHL games for season {args.season}...")
    games_df = get_season_games(args.season, args.start_date, args.end_date)

    print(f"Found {len(games_df)} games")

    # Save to output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    games_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

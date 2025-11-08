"""
Fetch player-level stats from NHL games
Aggregates into player-game records for feature engineering
"""
import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime


def toi_to_seconds(toi_str):
    """Convert TOI string (MM:SS) to seconds"""
    if not toi_str or toi_str == '0:00':
        return 0
    try:
        parts = str(toi_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return 0


def parse_player_stats(boxscore_data, game_id, game_date):
    """
    Extract player stats from boxscore into flat structure
    """
    if not boxscore_data or 'playerByGameStats' not in boxscore_data:
        return pd.DataFrame()

    all_players = []

    for team_type in ['homeTeam', 'awayTeam']:
        team_data = boxscore_data['playerByGameStats'].get(team_type, {})
        team_abbrev = boxscore_data.get(team_type, {}).get('abbrev', '')
        opp_team = boxscore_data.get('awayTeam' if team_type == 'homeTeam' else 'homeTeam', {}).get('abbrev', '')

        # Process forwards
        for player in team_data.get('forwards', []):
            all_players.append({
                'game_id': game_id,
                'game_date': game_date,
                'player_id': player.get('playerId'),
                'player_name': player.get('name', {}).get('default'),
                'team': team_abbrev,
                'opponent': opp_team,
                'home_away': 'home' if team_type == 'homeTeam' else 'away',
                'position': player.get('position'),
                'position_type': 'F',
                'goals': player.get('goals', 0),
                'assists': player.get('assists', 0),
                'points': player.get('points', 0),
                'plus_minus': player.get('plusMinus', 0),
                'pim': player.get('pim', 0),
                'hits': player.get('hits', 0),
                'shots': player.get('sog', 0),
                'blocked_shots': player.get('blockedShots', 0),
                'faceoff_pct': player.get('faceoffWinningPctg', 0) if player.get('faceoffWinningPctg') else 0,
                'toi_seconds': toi_to_seconds(player.get('toi', '0:00')),
                'pp_goals': player.get('powerPlayGoals', 0),
                'giveaways': player.get('giveaways', 0),
                'takeaways': player.get('takeaways', 0),
                'shifts': player.get('shifts', 0)
            })

        # Process defense
        for player in team_data.get('defense', []):
            all_players.append({
                'game_id': game_id,
                'game_date': game_date,
                'player_id': player.get('playerId'),
                'player_name': player.get('name', {}).get('default'),
                'team': team_abbrev,
                'opponent': opp_team,
                'home_away': 'home' if team_type == 'homeTeam' else 'away',
                'position': 'D',
                'position_type': 'D',
                'goals': player.get('goals', 0),
                'assists': player.get('assists', 0),
                'points': player.get('points', 0),
                'plus_minus': player.get('plusMinus', 0),
                'pim': player.get('pim', 0),
                'hits': player.get('hits', 0),
                'shots': player.get('sog', 0),
                'blocked_shots': player.get('blockedShots', 0),
                'faceoff_pct': 0,
                'toi_seconds': toi_to_seconds(player.get('toi', '0:00')),
                'pp_goals': player.get('powerPlayGoals', 0),
                'giveaways': player.get('giveaways', 0),
                'takeaways': player.get('takeaways', 0),
                'shifts': player.get('shifts', 0)
            })

        # Process goalies
        for player in team_data.get('goalies', []):
            all_players.append({
                'game_id': game_id,
                'game_date': game_date,
                'player_id': player.get('playerId'),
                'player_name': player.get('name', {}).get('default'),
                'team': team_abbrev,
                'opponent': opp_team,
                'home_away': 'home' if team_type == 'homeTeam' else 'away',
                'position': 'G',
                'position_type': 'G',
                'goals': 0,
                'assists': 0,
                'points': 0,
                'toi_seconds': toi_to_seconds(player.get('toi', '0:00')),
                'starter': player.get('starter', False),
                'decision': player.get('decision', ''),
                'saves': player.get('saves', 0),
                'shots_against': player.get('shotsAgainst', 0),
                'goals_against': player.get('goalsAgainst', 0),
                'save_pct': player.get('savePctg', 0) if player.get('savePctg') else 0
            })

    return pd.DataFrame(all_players)


def fetch_player_stats_for_games(games_csv, output_path='data/nhl/raw/player_stats.csv'):
    """
    Fetch player stats for all games in the schedule
    """
    games_df = pd.read_csv(games_csv)

    # Only fetch completed games
    completed = games_df[games_df['game_state'].isin(['OFF', 'FINAL'])].copy()

    print(f"Fetching player stats for {len(completed)} completed games...")

    all_player_stats = []
    total_games = len(completed)

    for i, (idx, game) in enumerate(completed.iterrows(), 1):
        game_id = game['game_id']
        game_date = game['game_date']

        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                boxscore = response.json()
                player_stats = parse_player_stats(boxscore, game_id, game_date)

                if len(player_stats) > 0:
                    all_player_stats.append(player_stats)

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"\nError fetching game {game_id}: {e}")
            continue

        # Progress bar
        pct = int(100 * i / total_games)
        bar_length = 50
        filled = int(bar_length * i / total_games)
        bar = '█' * filled + '░' * (bar_length - filled)
        total_records = sum(len(df) for df in all_player_stats)
        print(f'\r[{bar}] {pct}% ({i}/{total_games} games, {total_records} player records)', end='', flush=True)

    if len(all_player_stats) > 0:
        final_df = pd.concat(all_player_stats, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(final_df)} player-game records to {output_path}")
        return final_df
    else:
        print("No player stats fetched")
        return pd.DataFrame()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch NHL player stats')
    parser.add_argument('--games', default='data/nhl/raw/games.csv', help='Input games CSV')
    parser.add_argument('--output', default='data/nhl/raw/player_stats.csv', help='Output file')

    args = parser.parse_args()

    fetch_player_stats_for_games(args.games, args.output)

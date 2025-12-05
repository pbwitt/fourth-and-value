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


def fetch_player_stats_for_games(games_csv, output_path='data/nhl/raw/player_stats.csv', incremental=True, days_back=2):
    """
    Fetch player stats for all games in the schedule

    Args:
        games_csv: Path to games CSV
        output_path: Where to save player stats
        incremental: If True, only fetch recent games and append to existing data
        days_back: When incremental=True, fetch games from this many days back
    """
    games_df = pd.read_csv(games_csv)

    # Only fetch completed games
    completed = games_df[games_df['game_state'].isin(['OFF', 'FINAL'])].copy()

    # INCREMENTAL MODE: Only fetch recent games and append
    if incremental and os.path.exists(output_path):
        # Load existing stats
        existing_stats = pd.read_csv(output_path)
        existing_game_ids = set(existing_stats['game_id'].unique())

        # Filter to only NEW completed games from last N days
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        # Ensure game_date is string type for comparison
        completed['game_date'] = completed['game_date'].astype(str)
        completed = completed[
            (completed['game_date'] >= cutoff_date) &
            (~completed['game_id'].isin(existing_game_ids))
        ].copy()

        print(f"INCREMENTAL MODE: Fetching {len(completed)} NEW completed games since {cutoff_date}...")
    else:
        print(f"FULL MODE: Fetching player stats for {len(completed)} completed games...")
        existing_stats = None

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
        new_df = pd.concat(all_player_stats, ignore_index=True)

        # If incremental, append to existing data
        if incremental and existing_stats is not None:
            final_df = pd.concat([existing_stats, new_df], ignore_index=True)
            # Deduplicate (in case we refetched some games)
            final_df = final_df.drop_duplicates(subset=['game_id', 'player_id'], keep='last')
            print(f"\nAppended {len(new_df)} new records to existing {len(existing_stats)} records")
            print(f"Total: {len(final_df)} player-game records")
        else:
            final_df = new_df
            print(f"\nSaved {len(final_df)} player-game records")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Written to {output_path}")
        return final_df
    else:
        if incremental and existing_stats is not None:
            print("\nNo new games to fetch - keeping existing data")
            return existing_stats
        else:
            print("No player stats fetched")
            return pd.DataFrame()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch NHL player stats')
    parser.add_argument('--games', default='data/nhl/raw/games.csv', help='Input games CSV')
    parser.add_argument('--output', default='data/nhl/raw/player_stats.csv', help='Output file')
    parser.add_argument('--full', action='store_true', help='Fetch all games (not incremental)')
    parser.add_argument('--days-back', type=int, default=2, help='Days back to fetch (incremental mode)')

    args = parser.parse_args()

    fetch_player_stats_for_games(args.games, args.output, incremental=not args.full, days_back=args.days_back)

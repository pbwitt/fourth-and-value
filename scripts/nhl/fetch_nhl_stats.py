#!/usr/bin/env python3
"""
Fetch NHL stats from NHL Stats API for modeling.

Outputs:
  - data/nhl/processed/skater_logs_{date}.parquet (last 30 days game logs)
  - data/nhl/processed/goalie_logs_{date}.parquet (last 30 days game logs)
  - data/nhl/processed/schedule_{date}.csv (schedule with rest/b2b computed)
  - data/nhl/raw/nhl_api_skaters_{date}.json (raw API response)
  - data/nhl/raw/nhl_api_goalies_{date}.json (raw API response)

Features for SOG model:
  - TOI (EV/PP), shots, goals, assists, points
  - Opponent, home/away, rest days, b2b flag

Usage:
  python3 scripts/nhl/fetch_nhl_stats.py --date 2025-10-08
  python3 scripts/nhl/fetch_nhl_stats.py  # defaults to today
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

# NHL Stats API endpoints
NHL_API_BASE = "https://api.nhle.com/stats/rest/en"
SCHEDULE_API = "https://api-web.nhle.com/v1/schedule"

# Team code mapping (NHL API uses 3-letter codes)
TEAM_CODES = {
    "ANA": "Anaheim Ducks",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montréal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}


def req_json(url: str, timeout: float = 20.0, retry_429: bool = True) -> Any:
    """Make GET request with retry on 429."""
    time.sleep(0.5)  # Rate limiting
    r = requests.get(url, timeout=timeout)
    if r.status_code == 429 and retry_429:
        print(f"[warn] Rate limited, sleeping 2s...", file=sys.stderr)
        time.sleep(2.0)
        r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_season_stats(season: str = "20252026") -> pd.DataFrame:
    """
    Fetch season aggregate stats for all players across all teams.

    Returns DataFrame with columns:
        - player: Full name (e.g., "Brad Marchand")
        - team: Team code
        - position: Position code (L/R/C/D)
        - games_played: Games played
        - shots: Total shots
        - goals: Total goals
        - assists: Total assists
        - points: Total points
        - toi_per_game: Average TOI per game (seconds)
    """
    print(f"[fetch_nhl_stats] Fetching season stats for {season}...", file=sys.stderr)

    all_skaters = []

    for team_code in TEAM_CODES.keys():
        try:
            url = f"https://api-web.nhle.com/v1/club-stats/{team_code}/{season}/2"
            data = req_json(url)

            for skater in data.get("skaters", []):
                first_name = skater.get("firstName", {}).get("default", "")
                last_name = skater.get("lastName", {}).get("default", "")
                player_name = f"{first_name} {last_name}".strip()

                if not player_name:
                    continue

                all_skaters.append({
                    "player": player_name,
                    "team": team_code,
                    "position": skater.get("positionCode", "F"),
                    "games_played": skater.get("gamesPlayed", 0),
                    "shots": skater.get("shots", 0),
                    "goals": skater.get("goals", 0),
                    "assists": skater.get("assists", 0),
                    "points": skater.get("points", 0),
                    "toi_per_game": skater.get("avgTimeOnIcePerGame", 0),
                })

        except Exception as e:
            print(f"[warn] Failed to fetch season stats for {team_code}: {e}", file=sys.stderr)
            continue

    print(f"[fetch_nhl_stats] Fetched season stats for {len(all_skaters)} players", file=sys.stderr)
    return pd.DataFrame(all_skaters)


def fetch_skater_logs(season: str = "20252026", days_back: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch skater game logs from NHL API.

    Season format: "20252026" for 2025-26 season.
    """
    # Use summary endpoint for recent games
    url = f"{NHL_API_BASE}/skater/summary?cayenneExp=seasonId={season}"
    print(f"[fetch_nhl_stats] Fetching skater logs from {url}...")
    data = req_json(url)

    if not isinstance(data, dict) or "data" not in data:
        print(f"[warn] Unexpected skater response: {type(data)}", file=sys.stderr)
        return []

    return data.get("data", [])


def fetch_goalie_logs(season: str = "20252026") -> List[Dict[str, Any]]:
    """Fetch goalie logs from NHL API."""
    url = f"{NHL_API_BASE}/goalie/summary?cayenneExp=seasonId={season}"
    print(f"[fetch_nhl_stats] Fetching goalie logs from {url}...")
    data = req_json(url)

    if not isinstance(data, dict) or "data" not in data:
        print(f"[warn] Unexpected goalie response: {type(data)}", file=sys.stderr)
        return []

    return data.get("data", [])


def fetch_schedule(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch schedule from NHL API.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        List of games with home/away teams, date, venue
    """
    # NHL API requires single date calls, not ranges
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    games = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{SCHEDULE_API}/{date_str}"
        print(f"[fetch_nhl_stats] Fetching schedule for {date_str}...")

        try:
            data = req_json(url)
            for game_week in data.get("gameWeek", []):
                for game in game_week.get("games", []):
                    games.append({
                        "game_id": game.get("id"),
                        "game_date": game.get("gameDate") or date_str,
                        "start_time_utc": game.get("startTimeUTC"),
                        "home_team": game.get("homeTeam", {}).get("abbrev"),
                        "away_team": game.get("awayTeam", {}).get("abbrev"),
                        "venue": game.get("venue", {}).get("default"),
                        "game_state": game.get("gameState"),
                    })
        except Exception as e:
            print(f"[warn] Failed to fetch schedule for {date_str}: {e}", file=sys.stderr)

        current += timedelta(days=1)

    # Deduplicate by game_id (NHL API returns game week, not just single date)
    seen_game_ids = set()
    unique_games = []
    for game in games:
        if game["game_id"] not in seen_game_ids:
            seen_game_ids.add(game["game_id"])
            unique_games.append(game)

    return unique_games


def fetch_game_boxscore(game_id: int) -> Dict[str, Any]:
    """
    Fetch boxscore for a single game from NHL API.

    Returns player-level stats for all skaters in the game.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    data = req_json(url)
    return data


def fetch_player_name(player_id: int) -> str:
    """
    Fetch full player name from NHL API using player ID.

    Returns full name like "Sam Bennett" or empty string on error.
    """
    try:
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        data = req_json(url)

        # Extract first and last name
        first_name = data.get("firstName", {}).get("default", "")
        last_name = data.get("lastName", {}).get("default", "")

        if first_name and last_name:
            return f"{first_name} {last_name}"
        return ""

    except Exception as e:
        print(f"[warn] Failed to fetch player name for ID {player_id}: {e}", file=sys.stderr)
        return ""


def build_player_name_map(player_ids: List[int]) -> Dict[int, str]:
    """
    Build mapping of player_id -> full_name by fetching from player API.

    The boxscore API returns abbreviated names like "S. Bennett",
    but we need full names like "Sam Bennett" for matching with odds/props data.

    Args:
        player_ids: List of unique player IDs to resolve

    Returns:
        Dict mapping player_id to full name
    """
    print(f"[fetch_nhl_stats] Resolving {len(player_ids)} player names...", file=sys.stderr)

    name_map = {}
    for i, player_id in enumerate(player_ids):
        if (i + 1) % 50 == 0:
            print(f"[fetch_nhl_stats]   Resolved {i+1}/{len(player_ids)} players...", file=sys.stderr)

        full_name = fetch_player_name(player_id)
        if full_name:
            name_map[player_id] = full_name

    print(f"[fetch_nhl_stats] Resolved {len(name_map)}/{len(player_ids)} player names", file=sys.stderr)
    return name_map


def extract_skater_stats_from_boxscore(
    boxscore: Dict[str, Any],
    game_info: Dict[str, Any],
    player_name_map: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Extract skater stats from a boxscore response.

    Returns list of player stat dictionaries (one per player in the game).

    Args:
        boxscore: Boxscore data from NHL API
        game_info: Game metadata (must include start_time_utc for accurate date)
        player_name_map: Mapping of player_id -> full_name for name resolution
    """
    skater_rows = []

    # Get player stats container
    player_stats = boxscore.get("playerByGameStats", {})
    if not player_stats:
        return []

    # Parse actual game date from start_time_utc (ISO format)
    # Convert UTC timestamp to date string in local time
    try:
        from datetime import datetime
        game_datetime = datetime.fromisoformat(game_info["start_time_utc"].replace("Z", "+00:00"))
        game_date_str = game_datetime.strftime("%Y-%m-%d")
    except Exception:
        # Fallback to game_date if parsing fails
        game_date_str = game_info.get("game_date", "")

    # Extract home and away team data
    for team_key in ["homeTeam", "awayTeam"]:
        team_data = player_stats.get(team_key, {})
        team_abbrev = boxscore.get(team_key, {}).get("abbrev")
        is_home = (team_key == "homeTeam")

        # Loop through forwards and defense
        for position_group in ["forwards", "defense"]:
            players = team_data.get(position_group, [])
            for player in players:
                player_id = player.get("playerId")

                # Resolve full name from player_id map, fallback to abbreviated name
                player_name = player_name_map.get(
                    player_id,
                    player.get("name", {}).get("default", "")
                )

                skater_rows.append({
                    "game_id": game_info["game_id"],
                    "game_date": game_date_str,
                    "player_id": player_id,
                    "player": player_name,
                    "team": team_abbrev,
                    "opponent": game_info["away_team"] if is_home else game_info["home_team"],
                    "is_home": is_home,
                    "position": player.get("position"),
                    "goals": player.get("goals", 0),
                    "assists": player.get("assists", 0),
                    "points": player.get("points", 0),
                    "shots": player.get("sog", 0),  # shots on goal
                    "plus_minus": player.get("plusMinus", 0),
                    "pim": player.get("pim", 0),  # penalty minutes
                    "toi": player.get("toi", "0:00"),  # time on ice MM:SS
                    "pp_goals": player.get("powerPlayGoals", 0),
                    "pp_points": player.get("powerPlayPoints", 0),
                    "sh_goals": player.get("shorthandedGoals", 0),
                    "faceoff_pct": player.get("faceoffWinningPctg", 0.0),
                })

    return skater_rows


def fetch_game_logs(games: List[Dict[str, Any]], season: str = "20252026") -> pd.DataFrame:
    """
    Fetch game-by-game stats for all skaters across multiple games.

    Args:
        games: List of game dicts from fetch_schedule()
        season: NHL season ID (unused - kept for compatibility)

    Returns:
        DataFrame with one row per player per game
    """
    all_skater_rows = []
    all_player_ids = set()

    print(f"[fetch_nhl_stats] Fetching boxscores for {len(games)} games...")

    # First pass: fetch all boxscores with abbreviated names and collect player IDs
    temp_name_map = {}  # Empty map for first pass
    for i, game in enumerate(games):
        game_id = game["game_id"]
        game_state = game.get("game_state")

        # Only fetch completed games (OFF = final)
        if game_state not in ["OFF", "FINAL"]:
            continue

        try:
            print(f"[fetch_nhl_stats]   Game {i+1}/{len(games)}: {game_id} ({game['home_team']} vs {game['away_team']})")
            boxscore = fetch_game_boxscore(game_id)
            skater_stats = extract_skater_stats_from_boxscore(boxscore, game, temp_name_map)
            all_skater_rows.extend(skater_stats)

            # Collect unique player IDs
            for row in skater_stats:
                if row.get("player_id"):
                    all_player_ids.add(row["player_id"])

        except Exception as e:
            print(f"[warn] Failed to fetch game {game_id}: {e}", file=sys.stderr)
            continue

    if not all_skater_rows:
        print(f"[warn] No game logs fetched", file=sys.stderr)
        return pd.DataFrame()

    # Build player name map from collected IDs
    player_name_map = build_player_name_map(list(all_player_ids))

    # Second pass: replace abbreviated names with full names
    for row in all_skater_rows:
        player_id = row.get("player_id")
        if player_id and player_id in player_name_map:
            row["player"] = player_name_map[player_id]

    df = pd.DataFrame(all_skater_rows)

    # Deduplicate: NHL API sometimes returns same player twice in one game
    # Keep row with max stats (shots, goals, etc) in case of duplicates
    if not df.empty and 'player_id' in df.columns and 'game_id' in df.columns:
        original_count = len(df)

        # Sort by shots DESC so we keep the row with more shots
        if 'shots' in df.columns:
            df = df.sort_values('shots', ascending=False)

        # Drop duplicates based on (player_id, game_id), keeping first (highest shots)
        df = df.drop_duplicates(subset=['player_id', 'game_id'], keep='first')

        deduped_count = original_count - len(df)
        if deduped_count > 0:
            print(f"[fetch_nhl_stats] Removed {deduped_count} duplicate player-game rows", file=sys.stderr)

    print(f"[fetch_nhl_stats] Fetched {len(df)} player-game rows from {len(games)} games")

    return df


def normalize_skater_logs(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize skater logs to canonical schema.

    Expected fields from NHL API:
      - playerId, skaterFullName, teamAbbrevs, positionCode
      - gamesPlayed, timeOnIcePerGame, evTimeOnIcePerGame, ppTimeOnIcePerGame
      - shots, goals, assists, points
    """
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    # Rename to canonical columns
    rename_map = {
        "playerId": "player_id",
        "skaterFullName": "player",
        "teamAbbrevs": "team",
        "positionCode": "position",
        "gamesPlayed": "games_played",
        "timeOnIcePerGame": "toi_per_game",
        "evTimeOnIcePerGame": "toi_ev_per_game",
        "ppTimeOnIcePerGame": "toi_pp_per_game",
        "shots": "shots",
        "goals": "goals",
        "assists": "assists",
        "points": "points",
        "plusMinus": "plus_minus",
        "ppGoals": "pp_goals",
        "ppPoints": "pp_points",
        "shPoints": "sh_points",
        "shootingPct": "shooting_pct",
        "faceoffWinPct": "faceoff_win_pct",
    }

    # Apply renames for columns that exist
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert TOI from MM:SS to minutes (float)
    for col in ["toi_per_game", "toi_ev_per_game", "toi_pp_per_game"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_toi_to_minutes)

    # Add sport identifier
    df["sport"] = "nhl"

    return df


def normalize_goalie_logs(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normalize goalie logs to canonical schema."""
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    rename_map = {
        "playerId": "player_id",
        "goalieFullName": "player",
        "teamAbbrevs": "team",
        "gamesPlayed": "games_played",
        "gamesStarted": "games_started",
        "wins": "wins",
        "losses": "losses",
        "otLosses": "ot_losses",
        "savePct": "save_pct",
        "goalsAgainstAverage": "gaa",
        "shotsAgainst": "shots_against",
        "saves": "saves",
        "goalsAgainst": "goals_against",
        "shutouts": "shutouts",
        "timeOnIce": "toi",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["sport"] = "nhl"

    return df


def parse_toi_to_minutes(toi_str: str) -> float:
    """Convert MM:SS to minutes (float)."""
    if not toi_str or not isinstance(toi_str, str):
        return 0.0
    try:
        if ":" in toi_str:
            parts = toi_str.split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(toi_str)
    except Exception:
        return 0.0


def compute_rest_and_b2b(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest_days and b2b_flag for each team in each game.

    Returns schedule with additional columns:
      - rest_days_home, rest_days_away
      - b2b_home, b2b_away
    """
    if schedule_df.empty:
        return schedule_df

    schedule_df = schedule_df.sort_values("game_date").copy()
    schedule_df["rest_days_home"] = 7  # default high rest
    schedule_df["rest_days_away"] = 7
    schedule_df["b2b_home"] = False
    schedule_df["b2b_away"] = False

    # Track last game date per team
    last_game = {}

    for idx, row in schedule_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        game_date = pd.to_datetime(row["game_date"]).date()

        # Compute rest for home team
        if home in last_game:
            rest_days = (game_date - last_game[home]).days
            schedule_df.at[idx, "rest_days_home"] = rest_days
            schedule_df.at[idx, "b2b_home"] = (rest_days == 1)

        # Compute rest for away team
        if away in last_game:
            rest_days = (game_date - last_game[away]).days
            schedule_df.at[idx, "rest_days_away"] = rest_days
            schedule_df.at[idx, "b2b_away"] = (rest_days == 1)

        # Update last game dates
        last_game[home] = game_date
        last_game[away] = game_date

    return schedule_df


def main():
    parser = argparse.ArgumentParser(description="Fetch NHL stats for modeling")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date to fetch (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="20252026",
        help="NHL season ID (e.g., 20252026 for 2025-26)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Fetch schedule going back N days",
    )
    args = parser.parse_args()

    date_str = args.date
    season = args.season

    print(f"[fetch_nhl_stats] Fetching NHL stats for {date_str}, season {season}...")

    # Fetch schedule (from season start: Oct 7, 2025)
    # 2025-26 season started Oct 7, so don't go back further
    now = datetime.now()
    season_start = datetime(2025, 10, 7)

    # Use max of (season_start, now - days_back) to avoid fetching offseason
    earliest_date = max(season_start, now - timedelta(days=args.days_back))
    start_date = earliest_date.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    try:
        schedule_data = fetch_schedule(start_date, end_date)
        print(f"[fetch_nhl_stats] Fetched {len(schedule_data)} scheduled games")
    except Exception as e:
        print(f"[error] Schedule fetch failed: {e}", file=sys.stderr)
        schedule_data = []

    # Fetch game-by-game logs from boxscores
    skater_df = fetch_game_logs(schedule_data, season=season)

    # Fetch season aggregate stats (fallback for players without game logs)
    season_stats_df = fetch_season_stats(season=season)

    # Skip goalies for now (not needed for skater prop modeling)
    goalie_df = pd.DataFrame()

    schedule_df = pd.DataFrame(schedule_data)

    # Compute rest/b2b
    if not schedule_df.empty:
        schedule_df = compute_rest_and_b2b(schedule_df)

    # Write outputs
    raw_dir = Path("data/nhl/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = Path("data/nhl/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Write raw JSON (schedule data)
    with open(raw_dir / f"nhl_schedule_{date_str}.json", "w") as f:
        json.dump(schedule_data, f, indent=2)

    # Write normalized parquet (game logs)
    if not skater_df.empty:
        skater_path = proc_dir / f"skater_logs_{date_str}.parquet"
        skater_df.to_parquet(skater_path, index=False)
        print(f"[fetch_nhl_stats] Wrote {len(skater_df)} skater game-log rows to {skater_path}")

    if not goalie_df.empty:
        goalie_path = proc_dir / f"goalie_logs_{date_str}.parquet"
        goalie_df.to_parquet(goalie_path, index=False)
        print(f"[fetch_nhl_stats] Wrote {len(goalie_df)} goalies to {goalie_path}")

    if not schedule_df.empty:
        schedule_path = proc_dir / f"schedule_{date_str}.csv"
        schedule_df.to_csv(schedule_path, index=False)
        print(f"[fetch_nhl_stats] Wrote {len(schedule_df)} games to {schedule_path}")

    # Write season stats (fallback data)
    if not season_stats_df.empty:
        season_stats_path = proc_dir / f"season_stats_{date_str}.parquet"
        season_stats_df.to_parquet(season_stats_path, index=False)
        print(f"[fetch_nhl_stats] Wrote season stats for {len(season_stats_df)} players to {season_stats_path}")

    print("[fetch_nhl_stats] Done!")


if __name__ == "__main__":
    main()

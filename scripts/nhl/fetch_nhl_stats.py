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
    url = f"{SCHEDULE_API}/{start_date}/{end_date}"
    print(f"[fetch_nhl_stats] Fetching schedule from {url}...")
    data = req_json(url)

    games = []
    for game_week in data.get("gameWeek", []):
        for game in game_week.get("games", []):
            games.append({
                "game_id": game.get("id"),
                "game_date": game.get("gameDate"),
                "start_time_utc": game.get("startTimeUTC"),
                "home_team": game.get("homeTeam", {}).get("abbrev"),
                "away_team": game.get("awayTeam", {}).get("abbrev"),
                "venue": game.get("venue", {}).get("default"),
                "game_state": game.get("gameState"),
            })

    return games


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

    # Fetch skater and goalie logs
    skater_data = fetch_skater_logs(season=season)
    goalie_data = fetch_goalie_logs(season=season)

    print(f"[fetch_nhl_stats] Fetched {len(skater_data)} skater records, {len(goalie_data)} goalie records")

    # Fetch schedule (last 30 days + next 7 days)
    # Use real current date for API, not simulated date
    now = datetime.now()
    start_date = (now - timedelta(days=args.days_back)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        schedule_data = fetch_schedule(start_date, end_date)
    except Exception as e:
        print(f"[warn] Schedule fetch failed: {e}. Continuing without schedule.", file=sys.stderr)
        schedule_data = []

    print(f"[fetch_nhl_stats] Fetched {len(schedule_data)} scheduled games")

    # Normalize
    skater_df = normalize_skater_logs(skater_data)
    goalie_df = normalize_goalie_logs(goalie_data)
    schedule_df = pd.DataFrame(schedule_data)

    # Compute rest/b2b
    if not schedule_df.empty:
        schedule_df = compute_rest_and_b2b(schedule_df)

    # Write outputs
    raw_dir = Path("data/nhl/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = Path("data/nhl/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Write raw JSON
    with open(raw_dir / f"nhl_api_skaters_{date_str}.json", "w") as f:
        json.dump(skater_data, f, indent=2)

    with open(raw_dir / f"nhl_api_goalies_{date_str}.json", "w") as f:
        json.dump(goalie_data, f, indent=2)

    # Write normalized parquet
    if not skater_df.empty:
        skater_path = proc_dir / f"skater_logs_{date_str}.parquet"
        skater_df.to_parquet(skater_path, index=False)
        print(f"[fetch_nhl_stats] Wrote {len(skater_df)} skaters to {skater_path}")

    if not goalie_df.empty:
        goalie_path = proc_dir / f"goalie_logs_{date_str}.parquet"
        goalie_df.to_parquet(goalie_path, index=False)
        print(f"[fetch_nhl_stats] Wrote {len(goalie_df)} goalies to {goalie_path}")

    if not schedule_df.empty:
        schedule_path = proc_dir / f"schedule_{date_str}.csv"
        schedule_df.to_csv(schedule_path, index=False)
        print(f"[fetch_nhl_stats] Wrote {len(schedule_df)} games to {schedule_path}")

    print("[fetch_nhl_stats] Done!")


if __name__ == "__main__":
    main()

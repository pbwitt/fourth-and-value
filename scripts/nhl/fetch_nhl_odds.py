#!/usr/bin/env python3
"""
Fetch NHL player prop odds and game lines from The Odds API.

Outputs:
  - data/nhl/raw/odds_{date}.csv (raw API responses flattened)
  - data/nhl/processed/odds_props_{date}.csv (normalized player props)
  - data/nhl/processed/odds_games_{date}.csv (normalized game lines)

Usage:
  python3 scripts/nhl/fetch_nhl_odds.py --date 2025-10-08
  python3 scripts/nhl/fetch_nhl_odds.py  # defaults to today
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# NHL player prop markets (Odds API keys)
PLAYER_PROP_MARKETS = [
    "player_shots_on_goal",
    "player_goals",
    "player_assists",
    "player_points",
]

# Game line markets
GAME_LINE_MARKETS = [
    "h2h",        # moneyline
    "spreads",    # puckline
    "totals",     # over/under total goals
]

# Market normalization (Odds API key → canonical market_std)
MARKET_STD_MAP = {
    "player_shots_on_goal": "sog",
    "player_goals": "goals",
    "player_assists": "assists",
    "player_points": "points",
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
}

PLAYERISH_SIDES = {"Over", "Under", "Yes", "No"}


def get_api_key() -> str:
    """Load ODDS_API_KEY from environment."""
    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        print("ERROR: ODDS_API_KEY not set (source .env first).", file=sys.stderr)
        sys.exit(2)
    return api_key


def req_json(url: str, timeout: float = 20.0, retry_429: bool = True) -> Any:
    """Make GET request with retry on 429."""
    r = requests.get(url, timeout=timeout)
    if r.status_code == 422:
        return {"__422__": True, "body": r.text}
    if r.status_code == 429 and retry_429:
        print(f"[warn] Rate limited, sleeping 2s...", file=sys.stderr)
        time.sleep(2.0)
        r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_events(api_key: str, sport_key: str = "icehockey_nhl") -> List[Dict[str, Any]]:
    """Fetch NHL events (games) from Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?dateFormat=iso&apiKey={api_key}"
    data = req_json(url)
    if isinstance(data, dict) and data.get("__422__"):
        print(f"[warn] events 422: {data.get('body')}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        print(f"[warn] events response unexpected: {type(data)}", file=sys.stderr)
        return []
    return data


def fetch_event_odds(
    api_key: str,
    sport_key: str,
    event_id: str,
    markets: List[str],
    regions: str = "us",
    sleep_sec: float = 0.5,
) -> Dict[str, Any]:
    """Fetch odds for a single event with multiple markets."""
    markets_param = ",".join(markets)
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
        f"?regions={regions}&oddsFormat=american&markets={markets_param}&apiKey={api_key}"
    )
    time.sleep(sleep_sec)  # Rate limiting
    data = req_json(url)
    if isinstance(data, dict) and data.get("__422__"):
        print(f"[warn] event {event_id} → 422 {data.get('body')}", file=sys.stderr)
        return {}
    return data


def split_player_and_side(outcome: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (player, side) from outcome robustly."""
    name = (outcome.get("name") or "").strip()
    desc = (outcome.get("description") or "").strip()

    if name in PLAYERISH_SIDES and desc:
        return desc, name
    if desc in PLAYERISH_SIDES and name:
        return name, desc

    # Heuristic: whichever has a space and isn't a side
    player = name if (" " in name and name not in PLAYERISH_SIDES) else desc
    side = desc if player == name else name
    return player, side


def flatten_props(event_odds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten player prop odds to rows."""
    rows = []
    event_id = event_odds.get("id", "")
    commence_time = event_odds.get("commence_time", "")
    home_team = (event_odds.get("home_team") or "").strip()
    away_team = (event_odds.get("away_team") or "").strip()
    game_label = f"{away_team} @ {home_team}" if (home_team and away_team) else ""

    for bookmaker in event_odds.get("bookmakers", []):
        bk_key = bookmaker.get("key", "")
        bk_title = bookmaker.get("title", "")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            market_std = MARKET_STD_MAP.get(market_key, market_key)

            # Skip non-player-prop markets
            if market_key not in PLAYER_PROP_MARKETS:
                continue

            for outcome in market.get("outcomes", []):
                player, side = split_player_and_side(outcome)
                price = outcome.get("price")
                point = outcome.get("point")

                rows.append(
                    {
                        "sport": "nhl",
                        "game_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "game": game_label,
                        "bookmaker": bk_key,
                        "bookmaker_title": bk_title,
                        "market": market_key,
                        "market_std": market_std,
                        "player": player,
                        "name": side,  # Over/Under/Yes/No
                        "price": price,
                        "point": point if point is not None else "",
                    }
                )
    return rows


def flatten_game_lines(event_odds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten game line odds (moneyline, spread, totals) to rows."""
    rows = []
    event_id = event_odds.get("id", "")
    commence_time = event_odds.get("commence_time", "")
    home_team = (event_odds.get("home_team") or "").strip()
    away_team = (event_odds.get("away_team") or "").strip()

    for bookmaker in event_odds.get("bookmakers", []):
        bk_key = bookmaker.get("key", "")
        bk_title = bookmaker.get("title", "")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            market_std = MARKET_STD_MAP.get(market_key, market_key)

            # Skip player props
            if market_key not in GAME_LINE_MARKETS:
                continue

            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "").strip()  # team name or Over/Under
                price = outcome.get("price")
                point = outcome.get("point")  # spread/total line

                # Determine side
                if market_key == "h2h":
                    side = "home" if name == home_team else "away"
                elif market_key in ["spreads", "totals"]:
                    side = name.lower() if name.lower() in ["over", "under"] else name

                rows.append(
                    {
                        "sport": "nhl",
                        "game_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bk_key,
                        "bookmaker_title": bk_title,
                        "market": market_key,
                        "market_std": market_std,
                        "side": side,
                        "team_name": name,  # actual team or Over/Under
                        "price": price,
                        "point": point if point is not None else "",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch NHL odds from The Odds API")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date to fetch (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument("--sport", type=str, default="icehockey_nhl", help="Sport key")
    parser.add_argument("--regions", type=str, default="us", help="Regions (us, uk, eu)")
    args = parser.parse_args()

    api_key = get_api_key()
    date_str = args.date
    sport_key = args.sport

    print(f"[fetch_nhl_odds] Fetching NHL odds for {date_str}...")

    # Fetch events
    events = fetch_events(api_key, sport_key)
    print(f"[fetch_nhl_odds] Found {len(events)} NHL events")

    if not events:
        print("[fetch_nhl_odds] No events found, exiting.")
        sys.exit(0)

    # Fetch odds for each event
    all_props = []
    all_games = []

    all_markets = PLAYER_PROP_MARKETS + GAME_LINE_MARKETS

    for i, event in enumerate(events, 1):
        event_id = event.get("id")
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        print(f"[{i}/{len(events)}] Fetching {away} @ {home} ({event_id})...")

        odds = fetch_event_odds(api_key, sport_key, event_id, all_markets, args.regions)
        if not odds:
            continue

        # Flatten props and games
        props = flatten_props(odds)
        games = flatten_game_lines(odds)

        all_props.extend(props)
        all_games.extend(games)

    print(f"[fetch_nhl_odds] Collected {len(all_props)} prop rows, {len(all_games)} game line rows")

    # Create output directories
    raw_dir = Path("data/nhl/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Write normalized props
    proc_dir = Path("data/nhl/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    props_path = proc_dir / f"odds_props_{date_str}.csv"

    if all_props:
        with open(props_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_props[0].keys()))
            writer.writeheader()
            writer.writerows(all_props)
        print(f"[fetch_nhl_odds] Wrote {len(all_props)} props to {props_path}")

    # Write normalized game lines
    games_path = proc_dir / f"odds_games_{date_str}.csv"
    if all_games:
        with open(games_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_games[0].keys()))
            writer.writeheader()
            writer.writerows(all_games)
        print(f"[fetch_nhl_odds] Wrote {len(all_games)} game lines to {games_path}")

    print("[fetch_nhl_odds] Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fetch NFL totals and spreads from The Odds API across multiple books.
Outputs CSV with all book lines for consensus calculation.

Output columns:
  game, commence_time, home_team, away_team, book,
  total_over_line, total_over_price, total_under_price,
  spread_home_line, spread_home_price, spread_away_price

Usage:
  python3 scripts/nfl_fetch_totals_spreads.py --output data/nfl/lines/totals_spreads.csv
"""
import argparse
import os
import sys
import json
import urllib.parse
import urllib.request
from datetime import datetime
import pandas as pd


def fetch_json(url: str, timeout: int = 30):
    """Fetch JSON from URL"""
    req = urllib.request.Request(url, headers={"User-Agent": "fourth-and-value/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8", errors="ignore")

    # Strip any info lines
    lines = [ln for ln in data.splitlines() if not ln.lstrip().startswith("[info]")]
    txt = "\n".join(lines).strip()

    if not txt or txt[0] != "[":
        raise SystemExit("Odds API response is not a JSON array. First chars: " + txt[:80])

    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse Odds API JSON: {e}")


def parse_totals_spreads(events):
    """
    Parse totals and spreads from events array.
    Returns list of dicts with one row per game per book.
    """
    # Team abbreviation mapping
    team_abbrev = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
    }

    rows = []

    for event in events:
        game_id = event.get("id")
        commence_time = event.get("commence_time")
        home_team_full = event.get("home_team")
        away_team_full = event.get("away_team")

        # Convert to abbreviations
        home_team = team_abbrev.get(home_team_full, home_team_full)
        away_team = team_abbrev.get(away_team_full, away_team_full)
        game = f"{away_team} @ {home_team}"

        bookmakers = event.get("bookmakers", [])

        for book in bookmakers:
            book_name = book.get("key") or book.get("title")
            markets = book.get("markets", [])

            # Extract totals
            total_over_line = None
            total_over_price = None
            total_under_price = None

            for market in markets:
                if market.get("key") == "totals":
                    outcomes = market.get("outcomes", [])
                    for outcome in outcomes:
                        if outcome.get("name") == "Over":
                            total_over_line = outcome.get("point")
                            total_over_price = outcome.get("price")
                        elif outcome.get("name") == "Under":
                            total_under_price = outcome.get("price")

            # Extract spreads
            spread_home_line = None
            spread_home_price = None
            spread_away_price = None

            for market in markets:
                if market.get("key") == "spreads":
                    outcomes = market.get("outcomes", [])
                    for outcome in outcomes:
                        outcome_team = outcome.get("name")
                        if outcome_team == home_team_full:
                            spread_home_line = outcome.get("point")
                            spread_home_price = outcome.get("price")
                        elif outcome_team == away_team_full:
                            spread_away_price = outcome.get("price")

            # Only add row if we have at least totals or spreads
            if total_over_line or spread_home_line:
                rows.append({
                    "game": game,
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book": book_name,
                    "total_over_line": total_over_line,
                    "total_over_price": total_over_price,
                    "total_under_price": total_under_price,
                    "spread_home_line": spread_home_line,
                    "spread_home_price": spread_home_price,
                    "spread_away_price": spread_away_price,
                })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch NFL totals and spreads")
    parser.add_argument("--sport-key", default="americanfootball_nfl", help="Sport key for API")
    parser.add_argument("--regions", default="us", help="Regions (us, uk, etc)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ERROR: ODDS_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    # Build URL
    params = {
        "regions": args.regions,
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "apiKey": api_key,
    }
    base_url = "https://api.the-odds-api.com/v4"
    url = f"{base_url}/sports/{urllib.parse.quote(args.sport_key)}/odds?{urllib.parse.urlencode(params)}"

    print(f"Fetching NFL totals and spreads from The Odds API...", file=sys.stderr)
    events = fetch_json(url)
    print(f"✓ Fetched {len(events)} events", file=sys.stderr)

    # Parse into flat rows
    rows = parse_totals_spreads(events)

    if not rows:
        print("No totals or spreads found", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"✓ Saved {len(df)} book lines to {args.output}", file=sys.stderr)
    print(f"  Games: {df['game'].nunique()}", file=sys.stderr)
    print(f"  Books: {df['book'].nunique()}", file=sys.stderr)


if __name__ == "__main__":
    main()

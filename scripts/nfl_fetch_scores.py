#!/usr/bin/env python3
"""
Fetch NFL game scores from The Odds API.
Outputs CSV with final scores for completed games.

Output columns:
  game_date, game, home_team, away_team, home_score, away_score, completed

Usage:
  python3 scripts/nfl_fetch_scores.py --output data/nfl/scores/scores_latest.csv
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


def parse_scores(events):
    """
    Parse scores from events array.
    Returns list of dicts with one row per game.
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
        commence_time = event.get("commence_time")
        completed = event.get("completed", False)

        home_team_full = event.get("home_team")
        away_team_full = event.get("away_team")

        # Convert to abbreviations
        home_team = team_abbrev.get(home_team_full, home_team_full)
        away_team = team_abbrev.get(away_team_full, away_team_full)
        game = f"{away_team} @ {home_team}"

        # Extract scores
        scores = event.get("scores")
        home_score = None
        away_score = None

        if scores:
            for score_entry in scores:
                team_name = score_entry.get("name")
                score = score_entry.get("score")

                if team_name == home_team_full:
                    home_score = score
                elif team_name == away_team_full:
                    away_score = score

        # Parse game date from commence_time
        game_date = None
        if commence_time:
            try:
                dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                game_date = dt.date().isoformat()
            except:
                pass

        rows.append({
            "game_date": game_date,
            "game": game,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "completed": completed,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch NFL game scores")
    parser.add_argument("--sport-key", default="americanfootball_nfl", help="Sport key for API")
    parser.add_argument("--days-from", type=int, default=3, help="Days back to fetch scores")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ERROR: ODDS_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    # Build URL
    params = {
        "daysFrom": args.days_from,
        "apiKey": api_key,
    }
    base_url = "https://api.the-odds-api.com/v4"
    url = f"{base_url}/sports/{urllib.parse.quote(args.sport_key)}/scores?{urllib.parse.urlencode(params)}"

    print(f"Fetching NFL scores from The Odds API...", file=sys.stderr)
    events = fetch_json(url)
    print(f"✓ Fetched {len(events)} events", file=sys.stderr)

    # Parse into flat rows
    rows = parse_scores(events)

    if not rows:
        print("No scores found", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"✓ Saved {len(df)} game scores to {args.output}", file=sys.stderr)
    print(f"  Completed games: {df['completed'].sum()}", file=sys.stderr)

    # Show completed games with scores
    completed = df[df['completed'] == True]
    if len(completed) > 0:
        print("\nCompleted games:", file=sys.stderr)
        for _, row in completed.iterrows():
            print(f"  {row['game']}: {row['away_team']} {row['away_score']}, {row['home_team']} {row['home_score']}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fetch NFL odds from The Odds API and OUTPUT a FLAT CSV of games to STDOUT.

Columns:
  game_id, commence_time, home_team, away_team

Usage (example):
  python3 scripts/fetch_odds.py \
    --sport_key americanfootball_nfl \
    --markets h2h,spreads,totals \
    --regions us \
    --odds_format american > data/odds/latest.csv
"""
import argparse, os, sys, json, urllib.parse, urllib.request
import pandas as pd

def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-2025/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8", errors="ignore")
    # Some plans prepend info lines; strip them out here
    lines = [ln for ln in data.splitlines() if not ln.lstrip().startswith("[info]")]
    txt = "\n".join(lines).strip()
    # Expect a JSON array
    if not txt or txt[0] != "[":
        raise SystemExit("Odds API response is not a JSON array. First chars: " + txt[:80])
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse Odds API JSON: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport_key", default="americanfootball_nfl")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--markets", default="h2h,spreads,totals")
    ap.add_argument("--odds_format", default="american")
    ap.add_argument("--api_base", default="https://api.the-odds-api.com/v4")
    args = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ERROR: ODDS_API_KEY not set in environment (.env)", file=sys.stderr)
        sys.exit(1)

    # Build URL
    q = {
        "regions": args.regions,
        "markets": args.markets,
        "oddsFormat": args.odds_format,
        "apiKey": api_key,
    }
    url = f"{args.api_base}/sports/{urllib.parse.quote(args.sport_key)}/odds?{urllib.parse.urlencode(q)}"

    arr = fetch_json(url)

    # Flatten to one row per game
    rows = []
    for g in arr:
        rows.append({
            "game_id": g.get("id") or g.get("event_id"),
            "commence_time": g.get("commence_time"),
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
        })
    # Deduplicate by game_id just in case
    df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"], keep="first")

    # Output FLAT CSV to STDOUT
    df.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    main()

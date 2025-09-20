#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        from nfl_data_py import import_schedules
    except ImportError:
        raise SystemExit("pip install nfl_data_py")

    df = import_schedules([args.season])

    # Normalize expected columns
    # Try to pick a reliable UTC-ish kickoff; fallbacks included.
    # Columns we want: game_id, season, week, home_team, away_team, start_time (UTC)
    colmap = {
        "game_id": "game_id",
        "season": "season",
        "week": "week",
        "home_team": "home_team",
        "away_team": "away_team",
    }
    # Choose a datetime column in priority order
    dt_cols = ["start_time", "game_time_utc", "gameday", "game_datetime"]
    dt = None
    for c in dt_cols:
        if c in df.columns:
            dt = c
            break
    if dt is None:
        raise SystemExit("Could not find a kickoff time column in schedules.")

    out = df.rename(columns=colmap).copy()
    # Standardize names
    needed = ["game_id","season","week","home_team","away_team"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise SystemExit(f"Missing schedule columns: {missing}")

    # Build start_time (UTC) as ISO8601
    # nfl_data_py already tends to provide tz-aware; coerce and convert to UTC.
    out["start_time"] = pd.to_datetime(df[dt], errors="coerce", utc=True)
    if out["start_time"].isna().all():
        # If the chosen col had dates only, still fine—kept as midnight UTC
        pass

    out = out[["game_id","season","week","home_team","away_team","start_time"]].dropna(subset=["start_time"])
    out = out.sort_values(["week","start_time","home_team"]).reset_index(drop=True)

    path = args.out or f"data/schedule_{args.season}.csv"
    out.to_csv(path, index=False)
    print(f"[write] {path}  rows={len(out)}")

if __name__ == "__main__":
    main()

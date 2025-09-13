#!/usr/bin/env python3
"""
Fetch player prop odds from The Odds API and flatten to CSV.

- Requests explicit player-prop market keys (NOT "player_props") to avoid 422s.
- Guarantees a `player` column and a `name` column (Over/Under/Yes/No).
- ODDS_API_KEY read from env (source your .env first).
- Writes data/props/latest_all_props.csv (one row per outcome).

CLI is flexible: --season/--week are OPTIONAL (kept for Makefile compatibility).
"""

import argparse, csv, os, sys, time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests

DEFAULT_MARKETS = [
    "player_anytime_td","player_1st_td","player_last_td",
    "player_pass_yds","player_pass_tds","player_pass_attempts","player_pass_completions","player_pass_interceptions",
    "player_rush_yds","player_rush_tds","player_rush_attempts","player_rush_longest",
    "player_receptions","player_reception_yds","player_reception_tds","player_reception_longest",
]
PLAYERISH_SIDES = {"Over","Under","Yes","No"}

def chunks(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def split_player_and_side(outcome: Dict[str, Any]) -> Tuple[str, str]:
    """Return (player, side) from an outcome dict robustly."""
    name = (outcome.get("name") or "").strip()
    desc = (outcome.get("description") or "").strip()
    if name in PLAYERISH_SIDES and desc:
        return desc, name
    if desc in PLAYERISH_SIDES and name:
        return name, desc
    # heuristic: whichever looks like a person (contains space and not a side)
    player = name if (" " in name and name not in PLAYERISH_SIDES) else desc
    side   = desc if player == name else name
    return player, side

def get_api_key(override: str = None) -> str:
    api = override or os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not api:
        print("ERROR: ODDS_API_KEY not set (source .env first).", file=sys.stderr)
        sys.exit(2)
    return api

def req_json(url: str, timeout: float = 20.0) -> Any:
    r = requests.get(url, timeout=timeout)
    if r.status_code == 422:
        try:
            return {"__422__": True, "body": r.json()}
        except Exception:
            return {"__422__": True, "body": r.text}
    if r.status_code == 429:
        time.sleep(2.0)
        r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_events(api_key: str, sport_key: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?dateFormat=iso&apiKey={api_key}"
    data = req_json(url)
    if isinstance(data, dict) and data.get("__422__"):
        print(f"[warn] events 422: {data.get('body')}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        print(f"[warn] events response unexpected: {type(data)}", file=sys.stderr)
        return []
    return data

def fetch_event_props(api_key: str, sport_key: str, event_id: str,
                      markets: List[str], regions: str, chunk: int, sleep: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mchunk in chunks(markets, chunk):
        markets_param = ",".join(mchunk)
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
            f"?regions={regions}&oddsFormat=american&markets={markets_param}&apiKey={api_key}"
        )
        data = req_json(url)
        if isinstance(data, dict) and data.get("__422__"):
            print(f"[warn] event {event_id} {mchunk} → 422 {data.get('body')}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"[warn] event {event_id} unexpected type {type(data)}", file=sys.stderr)
            continue

        event = {"id": data.get("id"), "commence_time": data.get("commence_time")}
        home = (data.get("home_team") or "").strip() if isinstance(data.get("home_team"), str) else ""
        away = (data.get("away_team") or "").strip() if isinstance(data.get("away_team"), str) else ""
        game_label = f"{away} @ {home}" if (home and away) else ""

        for bk in data.get("bookmakers", []):
            bk_key = bk.get("key")
            bk_title = bk.get("title")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                for oc in m.get("outcomes", []):
                    player, side = split_player_and_side(oc)
                    rows.append({
                        "game_id": event["id"],
                        "commence_time": event.get("commence_time"),
                        "home_team": home, "away_team": away, "game": game_label,
                        "bookmaker": bk_key, "bookmaker_title": bk_title,
                        "market": mkey, "market_std": mkey,
                        "player": player,             # REQUIRED downstream
                        "name": side,                 # Over/Under/Yes/No
                        "price": oc.get("price"), "point": oc.get("point"),
                    })
        if sleep > 0:
            time.sleep(sleep)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Optional; kept for Makefile compatibility.")
    ap.add_argument("--week",   type=int, default=None, help="Optional; kept for Makefile compatibility.")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--sport_key", default="americanfootball_nfl")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    ap.add_argument("--chunk", type=int, default=8, help="Markets per request (6–10 is good).")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between market chunks.")
    ap.add_argument("--out", default="data/props/latest_all_props.csv")
    args = ap.parse_args()

    api_key = get_api_key(args.api_key)
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    print(f"[info] season={args.season} week={args.week} sport={args.sport_key} regions={args.regions} "
          f"markets={len(markets)} out={args.out}")

    events = fetch_events(api_key, args.sport_key)
    if not events:
        print("[warn] No events returned; writing header-only CSV.", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []
    for i, ev in enumerate(events, 1):
        ev_id = ev.get("id")
        if not ev_id:
            continue
        rows = fetch_event_props(api_key, args.sport_key, ev_id, markets, args.regions, args.chunk, args.sleep)
        if rows:
            all_rows.extend(rows)
        if i % 10 == 0:
            print(f"[info] processed {i}/{len(events)} events ...")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_id","commence_time","home_team","away_team","game",
        "bookmaker","bookmaker_title","market","market_std",
        "player","name","price","point"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)

    print(f"Wrote {out_path} with {len(all_rows):,} rows across {len(events):,} events")

if __name__ == "__main__":
    main()

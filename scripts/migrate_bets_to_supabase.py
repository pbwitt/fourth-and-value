#!/usr/bin/env python3
"""
One-time migration: import the existing docs/data/bets/bets.csv history into
the new Supabase bets table, under a single real user account.

Run this once, after:
  1. You've created the Supabase project and run supabase/schema.sql
  2. You've signed up for your OWN account at fourthandvalue.com/tracking/
     (so there's a real auth.users row for your email to attach these to)
  3. SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set (source .env)

Safe to re-run: rows are upserted on legacy_bet_id, so running this twice
won't create duplicates.

Usage:
  python scripts/migrate_bets_to_supabase.py --email you@example.com
  python scripts/migrate_bets_to_supabase.py --email you@example.com --csv data/bets/bets.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from supabase_client import get_user_id_by_email, insert_bets


def _num_or_none(v):
    v = (v or "").strip()
    return float(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="Email of the account to attach these bets to")
    ap.add_argument("--csv", default="docs/data/bets/bets.csv")
    args = ap.parse_args()

    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / args.csv
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    user_id = get_user_id_by_email(args.email)
    if not user_id:
        print(f"No Supabase account found for {args.email} - sign up at fourthandvalue.com/tracking/ first.")
        return
    print(f"Found user {args.email} -> {user_id}")

    with open(csv_path, "r", encoding="latin-1") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} bets from {csv_path}")

    payload = []
    for r in rows:
        payload.append({
            "legacy_bet_id": r.get("bet_id"),
            "user_id": user_id,
            "timestamp": r.get("timestamp") or None,
            "league": r.get("league"),
            "game_date": r.get("game_date") or None,
            "team_home": r.get("team_home") or None,
            "team_away": r.get("team_away") or None,
            "player": r.get("player") or None,
            "market_type": r.get("market_type") or None,
            "side": r.get("side") or None,
            "line": _num_or_none(r.get("line")),
            "book": r.get("book") or None,
            "odds": _num_or_none(r.get("odds")),
            "stake_dollars": _num_or_none(r.get("stake_dollars")) or 0,
            "status": r.get("status") or "pending",
            "actual_result": _num_or_none(r.get("actual_result")),
            "payout": _num_or_none(r.get("payout")),
            "graded_timestamp": r.get("graded_timestamp") or None,
            "model_prob": _num_or_none(r.get("model_prob")),
            "edge_bps": _num_or_none(r.get("edge_bps")) if r.get("edge_bps") not in (None, "undefined") else None,
        })

    insert_bets(payload)
    print(f"Migrated {len(payload)} bets to Supabase for {args.email}.")


if __name__ == "__main__":
    main()

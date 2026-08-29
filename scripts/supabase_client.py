#!/usr/bin/env python3
"""
Minimal REST client for the Supabase-backed bet tracker, used by the
grading scripts (grade_bets_nfl.py, grade_bets_nhl.py) and the one-time
CSV migration script.

Uses the service_role key, which bypasses row-level security - that's what
lets one scheduled job grade every user's pending bets in a single pass
instead of needing each user to be logged in. This key must only ever be
used here (local .env or GitHub Actions secrets) - never in docs/, which is
served publicly as the static site.
"""
import os

import requests


def _base_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_URL not set (source .env first).")
    return url.rstrip("/")


def _headers() -> dict:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not set (source .env first).")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def fetch_pending_bets(league: str) -> list[dict]:
    """All pending bets for a league, across every user."""
    resp = requests.get(
        f"{_base_url()}/rest/v1/bets",
        headers=_headers(),
        params={"league": f"eq.{league}", "status": "eq.pending", "select": "*"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def update_bet(bet_id: str, fields: dict) -> None:
    resp = requests.patch(
        f"{_base_url()}/rest/v1/bets",
        headers=_headers(),
        params={"id": f"eq.{bet_id}"},
        json=fields,
        timeout=30,
    )
    resp.raise_for_status()


def insert_bets(rows: list[dict]) -> None:
    """Bulk insert, used by the one-time CSV migration. Rows should include
    legacy_bet_id so re-running the migration doesn't create duplicates."""
    headers = dict(_headers())
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    resp = requests.post(
        f"{_base_url()}/rest/v1/bets",
        headers=headers,
        params={"on_conflict": "legacy_bet_id"},
        json=rows,
        timeout=60,
    )
    resp.raise_for_status()


def get_user_id_by_email(email: str) -> str | None:
    """Admin API lookup - only needed for the one-time CSV migration, to
    turn 'this is your email' into the user_id every bet row must carry."""
    resp = requests.get(
        f"{_base_url()}/auth/v1/admin/users",
        headers=_headers(),
        params={"email": email},
        timeout=30,
    )
    resp.raise_for_status()
    users = resp.json().get("users", [])
    return users[0]["id"] if users else None

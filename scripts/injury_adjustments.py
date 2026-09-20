#!/usr/bin/env python3
"""Shared, conservative injury-report normalization for NFL forecasts.

The raw nflverse report is a weekly designation, not a live transaction feed.
This module deliberately keeps the evidence and the adjustment separate:
``availability`` is an input to the forecast, while ``source_status`` and
``retrieved_at`` remain available for audit and page copy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json

import pandas as pd

from common_markets import std_player_name


STATUS_AVAILABILITY = {
    "out": 0.0,
    "ir": 0.0,
    "injured reserve": 0.0,
    "doubtful": 0.15,
    "questionable": 0.65,
    "probable": 0.90,
}

# Keep adjustment scope explicit. A status is not evidence that every market
# should move by the same amount, but availability is a safe first-order input.
MODELED_MARKETS = {
    "rush_yds", "recv_yds", "receptions", "pass_yds", "rush_attempts",
    "pass_attempts", "pass_completions", "pass_tds", "pass_interceptions",
    "interceptions", "anytime_td",
}


def _clean_status(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def availability_for_status(status: str, practice_status: str = "") -> Optional[float]:
    """Return a conservative availability probability, or None if ungraded."""
    s = _clean_status(status)
    if s in STATUS_AVAILABILITY:
        return STATUS_AVAILABILITY[s]
    # Practice reports are useful supporting evidence but should not turn a
    # player into an automatic scratch without an official game designation.
    p = _clean_status(practice_status)
    if p == "did not participate in practice":
        return 0.85
    if p == "limited participation in practice":
        return 0.95
    return None


def normalize_injury_report(raw: pd.DataFrame, season: int, week: int,
                            retrieved_at: Optional[str] = None) -> pd.DataFrame:
    """Normalize one season's nflverse injury file to target-week rows."""
    required = {"season", "week", "team", "full_name", "position",
                "practice_status"}
    if "source_status" not in raw.columns:
        required.add("report_status")
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"injury report missing columns: {sorted(missing)}")
    d = raw.copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d[(d["season"] == season) & (d["week"] == week)].copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "season", "week", "team", "full_name", "name_std", "position",
            "source_status", "practice_status", "availability", "retrieved_at",
        ])
    source_col = "source_status" if "source_status" in d.columns else "report_status"
    d["source_status"] = d[source_col].fillna("").astype(str).str.strip()
    d["practice_status"] = d["practice_status"].fillna("").astype(str).str.strip()
    d["availability"] = [availability_for_status(a, b)
                          for a, b in zip(d["source_status"], d["practice_status"])]
    d = d[d["availability"].notna()].copy()
    d["name_std"] = d["full_name"].map(std_player_name)
    d["retrieved_at"] = retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Keep the most restrictive designation if a feed contains duplicates.
    d = (d.sort_values(["name_std", "availability"])
           .drop_duplicates(["team", "name_std"], keep="first"))
    return d[["season", "week", "team", "full_name", "name_std", "position",
              "source_status", "practice_status", "availability", "retrieved_at"]]


def load_week_injuries(path: str | Path, season: int, week: int) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return normalize_injury_report(pd.read_csv(p, low_memory=False), season, week)


def write_adjustment_audit(injuries: pd.DataFrame, out: str | Path) -> None:
    """Write a small human-readable JSON audit beside the normalized CSV."""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = injuries.to_dict(orient="records") if not injuries.empty else []
    p.write_text(json.dumps({"rows": records}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = normalize_injury_report(pd.read_csv(args.input, low_memory=False), args.season, args.week)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    write_adjustment_audit(out, Path(args.output).with_suffix(".json"))
    print(f"[injuries] wrote {len(out):,} actionable Week {args.week} rows to {args.output}")

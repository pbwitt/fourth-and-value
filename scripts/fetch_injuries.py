#!/usr/bin/env python3
"""Fetch and normalize nflverse weekly injury reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd

from injury_adjustments import normalize_injury_report, write_adjustment_audit


URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"


def fetch(season: int, week: int, raw_out: Path, normalized_out: Path) -> pd.DataFrame:
    r = requests.get(URL.format(season=season), timeout=90)
    r.raise_for_status()
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_bytes(r.content)
    raw = pd.read_csv(raw_out, low_memory=False)
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    normalized = normalize_injury_report(raw, season, week, retrieved)
    normalized.to_csv(normalized_out, index=False)
    write_adjustment_audit(normalized, normalized_out.with_suffix(".json"))
    print(f"[injuries] fetched {len(raw):,} rows; {len(normalized):,} actionable Week {week} rows")
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out-dir", default="data/injuries")
    args = ap.parse_args()
    root = Path(args.out_dir)
    fetch(args.season, args.week,
          root / f"injuries_{args.season}.csv",
          root / f"injuries_week{args.week}.csv")


if __name__ == "__main__":
    main()

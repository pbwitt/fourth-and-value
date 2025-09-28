#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_player_prop_params.py — clean, fail‑fast drop‑in
-----------------------------------------------------
Builds per‑player parameter estimates for player‑prop markets using nflverse
weekly stats up to (season, week-1) and the current props dump. Designed to
maximize coverage *without* relying on game_id in stats and to avoid silent
fallbacks.

Outputs one row per (player × market_std) *present in props* with:
  - mu, sigma for Normal markets (yards, receptions, attempts, completions)
  - lam for Poisson markets (TDs, interceptions, anytime_td)
  - n_games used, plus basic identifiers

Markets supported (canonical keys):
  rush_yds, recv_yds, pass_yds, receptions,
  rush_attempts, pass_attempts, pass_completions,
  pass_tds, pass_interceptions, anytime_td

Usage (example):
  python -m make_player_prop_params \

    --season 2025 --week 4 \

    --props_csv data/props/latest_all_props.csv \

    --stats_parquet data/nflverse/stats_player_week_2025.parquet \

    --out data/props/params_week4.csv \

    --zero_prior_report data/props/zero_prior_week4.csv

Notes
-----
* No reliance on game_id from stats; we only trust game_id from props.
* No silent fallbacks; if a required stats column is missing for a market
  that appears in props, we fail fast with a clear error message.
* Name joining: prefer player_key when available; otherwise std_name.
"""
import argparse
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd


# -----------------------------
# Canonical market mapping
# -----------------------------

ALIAS_MARKET: Dict[str, str] = {
    # yards
    "player_rushing_yards": "rush_yds",
    "rush_yds": "rush_yds",
    "rushing_yards": "rush_yds",
    "player_receiving_yards": "recv_yds",
    "reception_yds": "recv_yds",
    "receiving_yards": "recv_yds",
    "recv_yds": "recv_yds",
    "player_passing_yards": "pass_yds",
    "passing_yards": "pass_yds",
    "pass_yds": "pass_yds",

    # receptions
    "player_receptions": "receptions",
    "receptions": "receptions",

    # attempts / completions
    "player_rushing_attempts": "rush_attempts",
    "rushing_attempts": "rush_attempts",
    "rush_attempts": "rush_attempts",
    "carries": "rush_attempts",

    "player_passing_attempts": "pass_attempts",
    "passing_attempts": "pass_attempts",
    "attempts": "pass_attempts",
    "pass_attempts": "pass_attempts",

    "player_passing_completions": "pass_completions",
    "passing_completions": "pass_completions",
    "completions": "pass_completions",
    "pass_completions": "pass_completions",

    # scoring / turnovers
    "player_passing_tds": "pass_tds",
    "passing_tds": "pass_tds",
    "pass_tds": "pass_tds",

    "player_interceptions": "pass_interceptions",
    "interceptions": "pass_interceptions",
    "passing_interceptions": "pass_interceptions",
    "pass_interceptions": "pass_interceptions",

    "player_anytime_td": "anytime_td",
    "anytime_td": "anytime_td",
    # yards (short forms)
    "player_rush_yds": "rush_yds",
    "player_reception_yds": "recv_yds",
    "player_pass_yds": "pass_yds",

    # attempts / completions (short forms)
    "player_rush_attempts": "rush_attempts",
    "player_pass_attempts": "pass_attempts",
    "player_pass_completions": "pass_completions",

    # TDs / interceptions (short forms)
    "player_pass_tds": "pass_tds",
    "player_pass_interceptions": "pass_interceptions",

    # normalize but (for now) unmodeled
    "player_1st_td": "first_td",
    "player_last_td": "last_td",
    "player_reception_longest": "reception_longest",
    "player_rush_longest": "rush_longest",







}

NORMAL_MARKETS = {
    "rush_yds", "recv_yds", "pass_yds",
    "receptions",
    "rush_attempts", "pass_attempts", "pass_completions",
}
POISSON_MARKETS = {"pass_tds", "pass_interceptions", "anytime_td"}
SUPPORTED_MARKETS = NORMAL_MARKETS | POISSON_MARKETS


# -----------------------------
# Helpers
# -----------------------------

def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[info] {msg}")


def std_name(s: Optional[str]) -> str:
    s = (str(s) if s is not None else "").strip().lower()
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    s = " ".join(s.split())
    return s


def name_loose(s: Optional[str]) -> str:
    """Loose key: first initial + last name (for emergency joins)."""
    t = std_name(s)
    if not t:
        return ""
    parts = t.split()
    if not parts:
        return ""
    first_initial = parts[0][0] if parts[0] else ""
    last = parts[-1] if len(parts) > 1 else ""
    return (first_initial + last).strip()


def canon_market(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower().replace(" ", "_")
    return ALIAS_MARKET.get(s, s)


def choose_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Required stats columns by canonical market
STATS_COLS: Dict[str, List[str]] = {
    "rush_yds": ["rushing_yards"],
    "recv_yds": ["receiving_yards"],
    "pass_yds": ["passing_yards"],
    "receptions": ["receptions"],
    "rush_attempts": ["rushing_attempts", "carries"],
    "pass_attempts": ["passing_attempts", "attempts"],
    "pass_completions": ["passing_completions", "completions"],
    "pass_tds": ["passing_tds"],
    "pass_interceptions": ["interceptions", "passing_interceptions"],
    # anytime_td = rushing_tds + receiving_tds (needs both)
    "anytime_td": ["rushing_tds", "receiving_tds"],
}


# -----------------------------
# Core logic
# -----------------------------

def load_props(path: str) -> pd.DataFrame:
    info(f"Loading props from {path}")
    df = pd.read_csv(path, low_memory=False)
    if not any(c in df.columns for c in ["player", "name"]):
        fail("props missing a player name column (expected 'player' or 'name').")
    if "market" not in df.columns:
        fail("props missing 'market' column.")

    df["market_std"] = df["market"].apply(canon_market)
    # normalize name fields
    name_src = "player" if "player" in df.columns else "name"
    df["player_display_name"] = df[name_src].astype(str)
    df["name_std"] = df["player_display_name"].map(std_name)
    df["name_loose"] = df["player_display_name"].map(name_loose)

    # carry player_key if present
    if "player_key" not in df.columns:
        df["player_key"] = pd.NA

    return df


def slice_stats(stats_path: str, season: int, week: int) -> pd.DataFrame:
    info(f"Loading weekly stats from {stats_path}")
    stats = pd.read_parquet(stats_path) if stats_path.lower().endswith(".parquet") else pd.read_csv(stats_path, low_memory=False)

    # Required columns
    for c in ["season", "week"]:
        if c not in stats.columns:
            fail(f"weekly stats missing required column '{c}'")

    # Use completed weeks only: weeks 1 .. (week-1)
    if week <= 1:
        fail(f"weekly stats empty for season={season}, weeks < {week} (no prior weeks).")
    stats = stats.loc[(stats["season"] == season) & (stats["week"] >= 1) & (stats["week"] < week)].copy()
    if stats.empty:
        fail(f"weekly stats empty for season={season}, weeks < {week}")

    # Visibility: show counts by week
    by_week = (stats.groupby("week").size().rename("rows").reset_index())
    info("[stats] rows by week after filter:\n" + by_week.to_string(index=False))

    # Name columns
    name_src = choose_first_present(stats, ["player_display_name", "player_name", "name"])
    if not name_src:
        fail("weekly stats missing player name col (player_display_name/player_name/name)")
    stats["player_display_name"] = stats[name_src].astype(str)
    stats["name_std"]   = stats["player_display_name"].map(std_name)
    stats["name_loose"] = stats["player_display_name"].map(name_loose)

    # Ensure some ID if available
    if "player_id" not in stats.columns:
        stats["player_id"] = pd.NA

    return stats



def build_params(props: pd.DataFrame, stats: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    # Determine which markets we actually need to compute
    needed_markets = sorted(set(props["market_std"]).intersection(SUPPORTED_MARKETS))
    if not needed_markets:
        fail("No supported markets found in props after canonicalization.")

    # Verify stats columns for all markets present in props
    # (anytime_td needs both rushing_tds and receiving_tds)
    for m in needed_markets:
        cols = STATS_COLS[m]
        if m == "anytime_td":
            missing = [c for c in cols if c not in stats.columns]
            if missing:
                fail(f"weekly stats missing columns for anytime_td: {missing}")
        else:
            if choose_first_present(stats, cols) is None:
                fail(f"weekly stats missing any of columns {cols} for market '{m}'")

    # Candidate players = unique names/ids present in props
    cand = (props[["player_key", "player_display_name", "name_std", "name_loose"]]
            .drop_duplicates()
            .reset_index(drop=True))

    rows = []
    # For each market, get the per-game series from stats and aggregate per player
    for m in needed_markets:
        if m == "anytime_td":
            if "rushing_tds" not in stats.columns or "receiving_tds" not in stats.columns:
                fail("weekly stats missing rushing_tds or receiving_tds needed for anytime_td")
            series = stats["rushing_tds"].fillna(0) + stats["receiving_tds"].fillna(0)
        else:
            col = choose_first_present(stats, STATS_COLS[m])
            if col is None:
                fail(f"weekly stats missing columns for {m}")
            series = stats[col]

        temp = stats[["player_id", "player_display_name", "name_std", "name_loose"]].copy()
        temp["market_std"] = m
        temp["value"] = series.astype(float)

        agg = (
            temp.groupby(["player_id", "name_std", "name_loose", "market_std"], dropna=False)["value"]
                .agg(n_games="count", mean="mean", var=lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan)
                .reset_index()
        )
        rows.append(agg)

    long_agg = pd.concat(rows, ignore_index=True)

    # Join to candidates (props players) on name_std (primary)
    merged = cand.merge(long_agg, on=["name_std"], how="left", suffixes=("", "_stats"))

    # Compute mu/sigma/lam by market type
    merged["mu"] = np.where(merged["market_std"].isin(list(NORMAL_MARKETS)),
                            merged["mean"], np.nan)
    # sigma requires at least 2 games
    merged["sigma"] = np.where(merged["market_std"].isin(list(NORMAL_MARKETS)),
                               np.where(merged["n_games"] > 1, np.sqrt(merged["var"]), np.nan),
                               np.nan)
    merged["lam"] = np.where(merged["market_std"].isin(list(POISSON_MARKETS)),
                             merged["mean"], np.nan)

    merged["season"] = season
    merged["week"] = week
    merged["built_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    out_cols = [
        "player_key", "player_display_name", "name_std", "name_loose",
        "market_std", "n_games", "mu", "sigma", "lam",
        "season", "week", "built_at",
    ]
    out = merged[out_cols].drop_duplicates().reset_index(drop=True)
    # Keep only rows where we computed at least one parameter
    out = out.loc[out[["mu", "lam"]].notna().any(axis=1)].copy()
    return out


def main():
    ap = argparse.ArgumentParser(description="Build per‑player params from weekly stats + props (fail‑fast, no game_id dependency in stats).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--props_csv", type=str, required=True)
    ap.add_argument("--stats_parquet", type=str, required=True, help="Path to nflverse weekly stats parquet (preferred) or CSV.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--zero_prior_report", type=str, default=None,
                    help="Optional CSV path: list players from props with zero prior‑week history this season.")
    args = ap.parse_args()

    # --- parse + train window ---
    season = int(args.season)
    week   = int(args.week)

    # We model Week N using completed weeks only (1..N-1)
    if week <= 1:
        fail(f"Week {week} has no prior weeks; pass week >= 2.")
    train_to_week = week  # note: slice_stats uses "< week" so passing W=4 yields weeks 1..3

    info(f"[run] season={season} week={week} (training on weeks < {train_to_week})")

    # --- load inputs ---
    props = load_props(args.props_csv)
    stats = slice_stats(args.stats_parquet, season, train_to_week)

    info(f"[props] rows={len(props)}; columns={list(props.columns)[:8]}...")
    info(f"[stats] rows={len(stats)} after filter")







    # Optional: report players with zero prior weeks (by name_std)
    if args.zero_prior_report:
        present_names = set(stats["name_std"].unique())
        zero_prior = (
            props.loc[~props["name_std"].isin(present_names), ["player_key", "player_display_name", "name_std"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        zero_prior.to_csv(args.zero_prior_report, index=False)
        info(f"Wrote zero‑prior report: {args.zero_prior_report} (rows={len(zero_prior)})")

        # --- build params ---
    out = build_params(props, stats, season, week)

    # --- Guardrails: MU clamp + SIGMA floor (before write) ---

    # 1) μ clamp: no negative means for OU markets
    if "mu" in out.columns:
        out["mu"] = pd.to_numeric(out["mu"], errors="coerce").fillna(0.0)
        out["mu"] = out["mu"].clip(lower=0.0)
        assert (out["mu"] >= 0).all(), "[params] mu clamp failed: found mu<0 after clamp"
        info(f"[params] mu clamp applied; negatives now {(out['mu']<0).sum()} rows")

    # 2) σ floor: only for Normal markets (market-aware floors)
    if "sigma" in out.columns and "market_std" in out.columns:
        NORMAL_MARKETS = {
            "rush_yds", "recv_yds", "pass_yds",
            "receptions",
            "rush_attempts", "pass_attempts", "pass_completions",
        }

        # conservative per-market minimums; tweak later as needed
        sigma_min_map = {
            "rush_yds":         3.0,
            "recv_yds":         4.0,
            "pass_yds":        12.0,
            "receptions":       1.2,
            "rush_attempts":    1.8,
            "pass_attempts":    4.0,
            "pass_completions": 3.0,
        }
        GLOBAL_SIGMA_FLOOR = 2.0  # fallback if a Normal market slips through unmapped
        EPS = 1e-6                # numerical guard

        out["sigma"] = pd.to_numeric(out["sigma"], errors="coerce").fillna(0.0)
        is_normal = out["market_std"].isin(NORMAL_MARKETS)
        floors = out["market_std"].map(sigma_min_map).fillna(GLOBAL_SIGMA_FLOOR)

        before_bad = int((is_normal & (out["sigma"] <= 0)).sum())
        out.loc[is_normal, "sigma"] = np.where(
            out.loc[is_normal, "sigma"] > 0,
            out.loc[is_normal, "sigma"],
            floors.loc[is_normal]
        )
        out.loc[is_normal, "sigma"] = out.loc[is_normal, "sigma"].clip(lower=EPS)
        after_bad = int((is_normal & (out["sigma"] <= 0)).sum())
        info(f"[params] sigma floor applied; fixed {before_bad - after_bad} rows in Normal markets")
        assert after_bad == 0, "[params] sigma floor failed: found sigma<=0 after floor in Normal markets"

    # --- write ---
    out.to_csv(args.out, index=False)
    info(f"Wrote params: {args.out} (rows={len(out)})")



if __name__ == "__main__":
    main()

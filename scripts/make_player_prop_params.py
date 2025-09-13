#!/usr/bin/env python3
"""
make_player_prop_params.py

Build per-player, per-market distribution parameters for edges calculation.

Outputs:
  data/props/params_week{WEEK}.csv with columns:
    - season, week
    - player, player_key, name_std
    - team (if available), position (best-effort), gsis_id (if resolved)
    - market_std
    - mu, sigma, lam  (mu/sigma for Normal OU markets; lam for Poisson markets)
    - dist            ("normal" or "poisson")

Key guarantees (coverage):
  - rush_attempts: always provide mu & sigma (fallback from carries/stats; final fallback defaults)
  - anytime_td:    always provide lam = rushing_tds + receiving_tds (fallback defaults)

Dependencies:
  - pandas, numpy
  - nfl_data_py (optional but strongly recommended; falls back to priors if unavailable)

Usage:
  python scripts/make_player_prop_params.py --season 2025 --week 2
  # optional: --props_csv data/props/latest_all_props.csv --out data/props/params_week2.csv
"""
from __future__ import annotations

from common_markets import standardize_input, apply_priors_if_missing




def load_weekly_logs_or_fallback(season: int, week: int):
    """
    Return per-player weekly logs for (season, weeks < week).
    1) Try nfl.import_weekly_data (official weekly parquet)
    2) If missing/empty, aggregate from PBP.

    Output columns include (when available):
      season, week, player, recent_team,
      passing_yards, attempts, completions, passing_tds, interceptions,
      rushing_yards, rush_attempts, rushing_tds,
      receiving_yards, receptions, receiving_tds,
      name_std, _source in {"weekly","pbp_agg"}
    """
    import pandas as pd
    import numpy as np
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    from common_markets import std_player_name

    def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # numeric week + filter to current season and weeks < current week
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df = df[(df.get("season") == season) & (df.get("week") < week)].copy()
        if len(df) == 0:
            return df
        # ensure a team column
        if "recent_team" not in df.columns:
            if "posteam" in df.columns:
                df["recent_team"] = df["posteam"]
            elif "team" in df.columns:
                df["recent_team"] = df["team"]
        # normalize name key
        name_col = "player" if "player" in df.columns else ("player_name" if "player_name" in df.columns else None)
        if name_col:
            df["name_std"] = df[name_col].map(std_player_name)
        df["_source"] = source
        return df

    # ---- 1) Weekly parquet path
    if nfl is not None:
        try:
            weekly = nfl.import_weekly_data([season], downcast=True)
            weekly = _finalize(weekly, "weekly")
            if len(weekly) > 0:
                return weekly
        except Exception:
            pass  # fall back to PBP

    # ---- 2) PBP aggregate fallback
    if nfl is None:
        raise RuntimeError("nfl_data_py unavailable; cannot load logs.")

    pbp = nfl.import_pbp_data([season], downcast=True)
    pbp["week"] = pd.to_numeric(pbp["week"], errors="coerce")
    pbp = pbp[pbp["week"] < week].copy()
    if len(pbp) == 0:
        return pd.DataFrame()

    # Flags (handle naming diffs across versions)
    pass_attempt   = pbp["pass_attempt"] if "pass_attempt" in pbp.columns else pbp.get("pass", 0)
    complete_pass  = pbp["complete_pass"] if "complete_pass" in pbp.columns else 0
    rush_attempt   = pbp["rush_attempt"] if "rush_attempt" in pbp.columns else pbp.get("rush", 0)
    yards_gained   = pbp["yards_gained"] if "yards_gained" in pbp.columns else 0

    # Yard totals if not provided
    if "passing_yards" not in pbp.columns:
        pbp["passing_yards"] = yards_gained.where(pass_attempt == 1, 0)
    if "rushing_yards" not in pbp.columns:
        pbp["rushing_yards"] = yards_gained.where(rush_attempt == 1, 0)
    if "receiving_yards" not in pbp.columns:
        pbp["receiving_yards"] = yards_gained.where(pass_attempt == 1, 0)

    # TDs / INTs (robust to column names)
    pbp["passing_tds"]   = pbp.get("pass_touchdown",    0)
    pbp["interceptions"] = pbp.get("interception",      0)
    pbp["rushing_tds"]   = pbp.get("rush_touchdown",    0)
    pbp["receiving_tds"] = pbp.get("rec_touchdown", pbp.get("receiving_touchdown", 0))

    frames = []

    # Passers
    if "passer_player_name" in pbp.columns:
        t = pbp[["season","week","posteam","passer_player_name",
                 "passing_yards","passing_tds","interceptions"]].copy()
        t["attempts"]    = pass_attempt
        t["completions"] = complete_pass
        t = t.rename(columns={"posteam":"recent_team","passer_player_name":"player"})
        frames.append(t)

    # Rushers
    if "rusher_player_name" in pbp.columns:
        t = pbp[["season","week","posteam","rusher_player_name",
                 "rushing_yards","rushing_tds"]].copy()
        t["rush_attempts"] = rush_attempt
        t = t.rename(columns={"posteam":"recent_team","rusher_player_name":"player"})
        frames.append(t)

    # Receivers (receptions = completed passes credited to a receiver)
    if "receiver_player_name" in pbp.columns:
        t = pbp[["season","week","posteam","receiver_player_name",
                 "receiving_yards","receiving_tds"]].copy()
        t["receptions"] = complete_pass.where(pbp["receiver_player_name"].notna(), 0)
        t = t.rename(columns={"posteam":"recent_team","receiver_player_name":"player"})
        frames.append(t)

    if not frames:
        return pd.DataFrame()

    weekly = pd.concat(frames, ignore_index=True)

    agg_cols = [c for c in [
        "passing_yards","attempts","completions","passing_tds","interceptions",
        "rushing_yards","rush_attempts","rushing_tds",
        "receiving_yards","receptions","receiving_tds"
    ] if c in weekly.columns]

    weekly = (weekly
              .groupby(["season","week","player","recent_team"], as_index=False)
              .agg({c: "sum" for c in agg_cols}))

    # finalize + tag source
    weekly["name_std"] = weekly["player"].map(std_player_name)
    weekly["_source"]  = "pbp_agg"
    return _finalize(weekly, "pbp_agg")


def load_props(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

import os, sys, numpy as np, pandas as pd
sys.path.append(os.path.dirname(__file__))
from common_markets import (
    standardize_input, ensure_param_schema, apply_priors_if_missing, PRIORS
)





import argparse, logging, os, sys, math
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# --- import market normalizer (works whether you're running from repo root or scripts/)
# nfl_data_py is preferred; if missing, we'll degrade gracefully
try:
    import nfl_data_py as nfl
    NFL_OK = True
except Exception:
    NFL_OK = False


# -----------------------------
# Config / Priors
# -----------------------------
NORMAL_MARKETS = {
    "rush_yds",
    "recv_yds",
    "receptions",
    "pass_yds",
    "rush_attempts",
}

POISSON_MARKETS = {
    "anytime_td",
    "pass_tds",
    "pass_interceptions",
}

# conservative league-wide fallback priors (per game)
PRIORS = {
    "rush_yds":          {"mu": 30.0, "sigma": 22.0},
    "recv_yds":          {"mu": 35.0, "sigma": 25.0},
    "receptions":        {"mu": 2.8,  "sigma": 2.0},
    "pass_yds":          {"mu": 225.0,"sigma": 55.0},
    "rush_attempts":     {"mu": 8.0,  "sigma": 4.0},
    "anytime_td":        {"lam": 0.25},
    "pass_tds":          {"lam": 1.4},
    "pass_interceptions":{"lam": 0.8},
}

# minimal sample size before trusting player-specific numbers
MIN_GAMES_STRICT = 4       # prefer at least this many games for strong trust
MIN_GAMES_LOOSE  = 2       # if <strict but >=loose, still use but add shrinkage
WINDOW_GAMES     = 17      # lookback window (last N games) across seasons


# -----------------------------
# Helpers
# -----------------------------
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    # cheap, safe slug (ASCII-ish)
    out = []
    prev_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "unknown"

def name_std_text(s: str) -> str:
    s = (s or "").lower()
    return "".join(ch if ch.isalnum() or ch == " " else " " for ch in s).split()
    # returns list of tokens; we'll join with single space:
def name_std_str(s: str) -> str:
    return " ".join(name_std_text(s))

def ensure_cols(df: pd.DataFrame, cols: Dict[str, float]) -> pd.DataFrame:
    for c, val in cols.items():
        if c not in df.columns:
            df[c] = val
    return df

def robust_std(s: pd.Series) -> float:
    # avoid zero sigma; use IQR-based fallback if needed
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    st = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
    if st <= 1e-6:
        q75, q25 = np.percentile(s, [75, 25])
        iqr = q75 - q25
        st = float(iqr / 1.349) if iqr > 0 else 0.0
    return max(st, 0.5)  # floor for sanity


# -----------------------------
# Data loading
# -----------------------------
def load_props_candidates(props_csv: str) -> pd.DataFrame:
    """
    Read the fetched props (latest_all_props.csv or equivalent) and return
    unique (player, market_std) rows we need to parameterize.
    """
    if not os.path.exists(props_csv):
        raise FileNotFoundError(f"props_csv not found: {props_csv}")

    raw = pd.read_csv(props_csv, low_memory=False)
    raw.columns = [c.strip().lower() for c in raw.columns]

    # normalize possible column names
    player_col = "player"
    if "player_name" in raw.columns and "player" not in raw.columns:
        raw["player"] = raw["player_name"]

    if "market" not in raw.columns:
        raise ValueError("Expected a 'market' column in props CSV.")

    # standardize markets


    # use only markets we know how to parameterize
    mask = raw["market_std"].isin(NORMAL_MARKETS | POISSON_MARKETS)
    df = raw.loc[mask, [player_col, "market_std"]].dropna().copy()
    df[player_col] = df[player_col].astype(str)

    # dedupe
    df = df.drop_duplicates(ignore_index=True)
    df["player_key"] = df[player_col].map(slugify)
    df["name_std"]   = df[player_col].map(name_std_str)

    # keep a 'player' column name stable
    df.rename(columns={player_col: "player"}, inplace=True)

    return df


def fetch_recent_game_logs(season: int) -> Optional[pd.DataFrame]:
    """
    Load weekly player logs for `season` from local nflverse parquet fetched earlier.
    Fail-fast if missing instead of silently falling back.
    """
    from pathlib import Path
    p = Path(f"data/weekly_player_stats_{season}.parquet")
    if not p.exists():
        logging.error(f"Missing weekly parquet for season {season}: {p} (run weekly_stats first)")
        return None

    weekly = pd.read_parquet(p)
    weekly = weekly.loc[weekly.get("season").astype(int) == int(season)].copy()

    weekly.columns = [c.lower() for c in weekly.columns]

    needed = [
        "player_name","recent_team","position","gsis_id","season","week",
        "rushing_yards","rushing_attempts","rushing_tds",
        "receptions","receiving_yards","receiving_tds",
        "passing_yards","passing_tds","interceptions"
    ]
    for c in needed:
        if c not in weekly.columns:
            weekly[c] = np.nan

    weekly["player"] = weekly["player_name"].fillna("")

    keep = list(dict.fromkeys(["player","season","week"] + needed))
    have = [c for c in keep if c in weekly.columns]
    weekly = weekly[have]

    tail = (
        weekly.sort_values(["player","season","week"])
              .groupby("player", group_keys=False)[have]
              .apply(lambda g: g.tail(WINDOW_GAMES))
              .reset_index(drop=True)
    )

    tail["name_std"] = tail["player"].astype(str).map(name_std_str)
    tail["player_key"] = tail["player"].astype(str).map(slugify)
    return tail





# -----------------------------
# Parameter building
# -----------------------------
def _agg_mu_sigma(s: pd.Series) -> Tuple[float, float]:
    mu = float(pd.to_numeric(s, errors="coerce").dropna().mean()) if len(s.dropna()) else np.nan
    sg = robust_std(s)
    return mu, sg

def build_params(cands, logs, season, week):
    """
    Board-driven param builder.
    - Emits a row for every (player, market_std) present on the props board.
    - Normal markets -> dist='normal', fills mu/sigma from logs if available, else PRIORS.
    - Poisson markets -> dist='poisson', fills lam from logs if available, else PRIORS (and mirrors to mu).
    - Adds used_logs=1 per row only if that player has any logs; else 0.
    - Includes team/position/gsis_id if we can infer them from logs; else NaN.
    """
    import numpy as np
    import pandas as pd

    # Expect these to exist at module scope in this file
    # NORMAL_MARKETS, POISSON_MARKETS, PRIORS

    # --- player universe from the board (not from logs) ---
    player_idx = pd.Index(cands["player"].dropna().unique(), name="player")

    # columns to carry through from the board
    keep_cols = [c for c in ["season","week","player","player_key","name_std","market_std"] if c in cands.columns]

    # --- helpers ---
    def const_series(val):
        return pd.Series(val, index=player_idx, dtype="float64")

    def pick_col(df, candidates):
        if df is None or getattr(df, "empty", True):
            return None
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # infer team/position/gsis_id and per-player log presence
    team_map = {}
    pos_map  = {}
    gsis_map = {}
    used_logs_map = {}
    if logs is not None and not logs.empty:
        grp = logs.groupby("player", dropna=False)
        # choose best-available columns
        team_col = pick_col(logs, ["recent_team", "team", "posteam", "recent_team_x"])
        pos_col  = pick_col(logs, ["position", "pos"])
        gsis_col = pick_col(logs, ["gsis_id", "player_id", "player_gsis_id", "id"])
        if team_col:
            team_map = grp[team_col].last().to_dict()
        if pos_col:
            pos_map = grp[pos_col].last().to_dict()
        if gsis_col:
            gsis_map = grp[gsis_col].last().to_dict()
        used_logs_map = grp.size().to_dict()
    # fallback lambdas
    g_used = lambda p: 1 if used_logs_map.get(p, 0) > 0 else 0
    g_team = lambda p: team_map.get(p, np.nan)
    g_pos  = lambda p: pos_map.get(p,  np.nan)
    g_gsis = lambda p: gsis_map.get(p, np.nan)

    # --- which log columns feed each market ---
    LOG_COLS_NORMAL = {
        "rush_yds":         ["rushing_yards"],
        "recv_yds":         ["receiving_yards"],
        "receptions":       ["receptions"],
        "pass_yds":         ["passing_yards"],
        "pass_attempts":    ["passing_attempts", "attempts"],
        "pass_completions": ["passing_completions", "completions"],
        "rush_attempts":    ["rushing_attempts"],
    }
    LOG_COLS_POISSON = {
        "pass_tds":            ["passing_tds"],
        "pass_interceptions":  ["interceptions"],
        # anytime_td handled via rushing_tds + receiving_tds
    }

    # --- precompute per-player μ/σ for Normal markets ---
    mu_map, sg_map = {}, {}
    if logs is not None and not logs.empty:
        grp = logs.groupby("player", dropna=False)

    for mkt, cand_cols in LOG_COLS_NORMAL.items():
        mu0 = float(PRIORS.get(mkt, {}).get("mu", np.nan))
        sg0 = float(PRIORS.get(mkt, {}).get("sigma", np.nan))
        if logs is None or getattr(logs, "empty", True):
            mu, sg = const_series(mu0), const_series(sg0)
        else:
            col = pick_col(logs, cand_cols)
            if col is None:
                mu, sg = const_series(mu0), const_series(sg0)
            else:
                g = grp[col]
                mu = g.mean().reindex(player_idx).astype("float64").fillna(mu0)
                sd = g.std(ddof=1).reindex(player_idx).astype("float64")
                sg = sd.fillna(sg0)
        mu_map[mkt], sg_map[mkt] = mu, sg

    # --- precompute per-player λ for Poisson markets ---
    def lam_from(cols, prior_key, default_val=None):
        base = PRIORS.get(prior_key, {}).get("lam", np.nan) if default_val is None else float(default_val)
        if logs is None or getattr(logs, "empty", True):
            return const_series(base)
        col = pick_col(logs, cols)
        if col is None:
            return const_series(base)
        s = grp[col].mean().reindex(player_idx).astype("float64")
        return s.fillna(base)

    lam_any = float(PRIORS.get("anytime_td", {}).get("lam", 0.5))
    rtd  = lam_from(["rushing_tds"],  "anytime_td", default_val=lam_any/2.0)
    rctd = lam_from(["receiving_tds"],"anytime_td", default_val=lam_any/2.0)
    ptd  = lam_from(LOG_COLS_POISSON["pass_tds"], "pass_tds")
    ints = lam_from(LOG_COLS_POISSON["pass_interceptions"], "pass_interceptions")

    def lam_for(mkt):
        if mkt == "anytime_td":         return (rtd + rctd).clip(lower=0.01, upper=5.0)
        if mkt == "pass_tds":           return ptd.clip(lower=0.01, upper=5.0)
        if mkt == "pass_interceptions": return ints.clip(lower=0.01, upper=5.0)
        return const_series(np.nan)

    # --- assemble rows per market from the board (safe: .assign + concat) ---
    rows = []
    present = sorted(cands["market_std"].dropna().unique())
    for mkt in present:
        want = cands.loc[cands["market_std"] == mkt, keep_cols].drop_duplicates()
        if want.empty:
            continue

        # attach per-player meta if requested by your file's schema
        want = want.assign(
            team     = want["player"].map(g_team)  if "team"     in getattr(cands, "columns", []) or True else np.nan,
            position = want["player"].map(g_pos)   if "position" in getattr(cands, "columns", []) or True else np.nan,
            gsis_id  = want["player"].map(g_gsis)  if "gsis_id"  in getattr(cands, "columns", []) or True else np.nan,
        )

        if mkt in LOG_COLS_NORMAL:
            mu = mu_map[mkt]
            sg = sg_map[mkt]
            rows.append(
                want.assign(
                    dist      = "normal",
                    mu        = want["player"].map(mu),
                    sigma     = want["player"].map(sg),
                    lam       = np.nan,
                    used_logs = want["player"].map(lambda p: g_used(p)),
                )
            )
        elif mkt in POISSON_MARKETS:
            lam = lam_for(mkt)
            rows.append(
                want.assign(
                    dist      = "poisson",
                    mu        = want["player"].map(lam),  # legacy compat
                    sigma     = np.nan,
                    lam       = want["player"].map(lam),
                    used_logs = want["player"].map(lambda p: g_used(p)),
                )
            )
        else:
            # unmodeled (longest, 1st/last TD) — skip
            continue

    params = (
        pd.concat(rows, ignore_index=True)
        if rows else pd.DataFrame(columns=keep_cols + ["team","position","gsis_id","dist","mu","sigma","lam","used_logs"])
    )

    # back-compat: some downstream code expects 'market'
    if "market" not in params.columns and "market_std" in params.columns:
        params["market"] = params["market_std"]

    # ensure types
    for col in ["mu","sigma","lam"]:
        if col in params.columns:
            params[col] = pd.to_numeric(params[col], errors="coerce")

    # attach season/week if missing
    if "season" not in params.columns:
        params["season"] = season
    if "week" not in params.columns:
        params["week"] = week

    return params





# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build player prop params (μ/σ/λ) by market.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--props_csv", type=str, default="data/props/latest_all_props.csv",
                   help="Fetched props CSV used to enumerate (player, market) pairs.")
    p.add_argument("--out", type=str, default=None,
                   help="Output CSV path (default: data/props/params_week{WEEK}.csv)")
    p.add_argument("--loglevel", type=str, default="INFO")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="%(levelname)s:%(message)s")

    season, week = args.season, args.week
    out_path = args.out or f"data/props/params_week{week}.csv"

    logging.info(f"Building params for season={season}, week={week}")






    cands = load_props_candidates(args.props_csv)

    # Ensure we have a props path
    if not getattr(args, "props_csv", None):
        args.props_csv = "data/props/latest_all_props.csv"

    logging.info(f"Loading props from {args.props_csv}")
    props = load_props(args.props_csv)

    # 1) Normalize market labels on the board to canonical names
    from common_markets import std_market, MODELED_MARKETS
    pcol = "market_std" if "market_std" in props.columns else "market"
    props["market_std"] = props[pcol].map(std_market)

    # 2) Filter to modeled markets only (drop longest/1st/last TD, etc.)
    props = props[props["market_std"].isin(MODELED_MARKETS)].copy()
    if props.empty:
        raise SystemExit("No modeled props found after normalization/filtering.")

    # 3) Build the candidate set from the (now-normalized) board
    keep_cols = [c for c in ["player","player_key","name_std","market_std"] if c in props.columns]
    cands = (
        props.loc[:, keep_cols]
             .dropna(subset=["player","market_std"])
             .drop_duplicates()
             .copy()
    )
    cands["season"] = args.season
    cands["week"]   = args.week
    logging.info(f"Candidates: {cands[['player','market_std']].drop_duplicates().shape[0]} "
                 "unique (player, market_std) pairs.")





    logging.info(f"Candidates: {len(cands)} unique (player, market_std) pairs.")

    logs = fetch_recent_game_logs(season)



        # --- PROPS → cands (board-driven) ---
    props_csv = getattr(args, "props_csv", None) or "data/props/latest_all_props.csv"
    logging.info(f"Loading props from {props_csv}")
    props = load_props(props_csv)

    from common_markets import std_market, MODELED_MARKETS

    # normalize book labels → canonical (e.g., player_reception_yds → recv_yds)
    pcol = "market_std" if "market_std" in props.columns else "market"
    props["market_std"] = props[pcol].map(std_market)

    # keep only modeled markets (drop longest / 1st / last TD, etc.)
    props = props[props["market_std"].isin(MODELED_MARKETS)].copy()
    if props.empty:
        raise SystemExit("No modeled props found after normalization/filtering.")

    keep_cols = [c for c in ["player","player_key","name_std","market_std"] if c in props.columns]
    cands = (
        props.loc[:, keep_cols]
             .dropna(subset=["player","market_std"])
             .drop_duplicates()
             .copy()
    )
    cands["season"] = args.season
    cands["week"]   = args.week
    logging.info(f"Candidates: {cands[['player','market_std']].drop_duplicates().shape[0]} "
                 "unique (player, market_std) pairs.")

    params = build_params(cands, logs, args.season, args.week)


    # tidy columns order
    from common_markets import ensure_param_schema, apply_priors_if_missing
    params = ensure_param_schema(params)
    params = apply_priors_if_missing(params)

    # Back-compat: some downstream code expects 'market'
    if "market" not in params.columns:
        params["market"] = params["market_std"]

    # Final tidy + write
    cols = ["season","week","player","player_key","name_std","market_std","market",
        "dist","mu","sigma","lam","used_logs"]
    params = params[[c for c in cols if c in params.columns]].copy()
    params = standardize_input(params)           # adds market_std/name/point/name_std
    params = apply_priors_if_missing(params)     # fills missing mu/sigma/lam using PRIORS


    out_csv = args.out or f"data/props/params_week{args.week}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # --- ensure join keys exist in params (permanent fix) ---
    from common_markets import std_player_name

    df = params  # or: df = out   <-- use whatever your params DataFrame is named

    # Ensure 'player' exists (fallbacks if your builder used a different column)
    if 'player' not in df.columns:
        for c in ('player_name','athlete','athlete_name','name'):
            if c in df.columns:
                df['player'] = df[c]
                break

    # name_std from player text
    if 'name_std' not in df.columns:
        df['name_std'] = df['player'].map(std_player_name)
    else:
        df['name_std'] = df['name_std'].fillna(df['player'].map(std_player_name))

    # player_key = existing or fallback to name_std
    if 'player_key' not in df.columns:
        df['player_key'] = df['name_std']
    else:
        df['player_key'] = df['player_key'].fillna(df['name_std'])

    # put it back if you used another variable name
    params = df  # or: out = df

    params.to_csv(out_csv, index=False)
    logging.info("params written: %s", out_csv)





if __name__ == "__main__":
    main()

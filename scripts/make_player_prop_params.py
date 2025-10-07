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

from common_markets import standardize_input, apply_priors_if_missing, std_player_name

# --- add near the top of the script (imports) ---
from pathlib import Path
from datetime import datetime, timezone
import os

# absolute base for historical snapshots
HIST_DIR = Path("/Users/pwitt/fourth-and-value/data/preds_historical")

# one run timestamp for the whole script
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def snapshot_props_with_model(df, season, week, *, base_dir=HIST_DIR, run_ts=RUN_TS, fmt="csv"):
    """
    Write an immutable, timestamped backup of the final predictions.
    - Adds a `snapshot_ts` column (UTC ISO-like).
    - Writes atomically (tmp -> rename).
    - Updates a convenience 'latest' symlink.
    """
    out_dir = base_dir / str(int(season)) / f"week{int(week):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    df2["snapshot_ts"] = run_ts  # keep the run time inside the file

    if fmt == "parquet":
        tmp = out_dir / f"props_with_model_week{int(week):02d}_{run_ts}.parquet.tmp"
        final = out_dir / f"props_with_model_week{int(week):02d}_{run_ts}.parquet"
        df2.to_parquet(tmp, index=False)
        latest = out_dir / "latest.parquet"
    else:
        tmp = out_dir / f"props_with_model_week{int(week):02d}_{run_ts}.csv.tmp"
        final = out_dir / f"props_with_model_week{int(week):02d}_{run_ts}.csv"
        df2.to_csv(tmp, index=False)
        latest = out_dir / "latest.csv"

    os.replace(tmp, final)  # atomic
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        # relative symlink so moving the folder preserves linkage
        os.symlink(final.name, latest)
    except Exception:
        # symlink can fail on some filesystems — ignore
        pass

    print(f"[snapshot] wrote {final}")








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

    # Alias column names to match expected schema
    if "carries" in weekly.columns and "rushing_attempts" not in weekly.columns:
        weekly["rushing_attempts"] = weekly["carries"]
    if "team" in weekly.columns and "recent_team" not in weekly.columns:
        weekly["recent_team"] = weekly["team"]
    if "player_id" in weekly.columns and "gsis_id" not in weekly.columns:
        weekly["gsis_id"] = weekly["player_id"]
    if "passing_interceptions" in weekly.columns and "interceptions" not in weekly.columns:
        weekly["interceptions"] = weekly["passing_interceptions"]

    # Use player_display_name for matching (full names) instead of player_name (initials)
    if "player_display_name" in weekly.columns:
        weekly["player"] = weekly["player_display_name"].fillna("")
    elif "player_name" in weekly.columns:
        weekly["player"] = weekly["player_name"].fillna("")
    else:
        weekly["player"] = ""

    needed = [
        "recent_team","position","gsis_id","season","week",
        "rushing_yards","rushing_attempts","rushing_tds","carries",
        "receptions","receiving_yards","receiving_tds","targets",  # Added targets
        "passing_yards","passing_tds","interceptions",
        "attempts","completions"
    ]
    for c in needed:
        if c not in weekly.columns:
            weekly[c] = np.nan

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


def exponential_weighted_mean(values: pd.Series, alpha: float = 0.3) -> float:
    """
    Calculate exponentially weighted mean where recent games have more weight.

    Args:
        values: Series of historical values (ordered by time, oldest first)
        alpha: Decay factor (0-1). Higher = more weight to recent games.
               0.3 means most recent game gets ~30% more weight than oldest.

    Returns:
        Weighted mean
    """
    if len(values) == 0 or values.isna().all():
        return np.nan

    values_clean = values.dropna()
    if len(values_clean) == 0:
        return np.nan

    # Create exponential weights: more recent = higher weight
    n = len(values_clean)
    weights = np.array([alpha ** (n - i - 1) for i in range(n)])
    weights = weights / weights.sum()  # normalize to sum to 1

    return float(np.average(values_clean.values, weights=weights))


def exponential_weighted_std(values: pd.Series, alpha: float = 0.3) -> float:
    """
    Calculate exponentially weighted standard deviation.

    Args:
        values: Series of historical values (ordered by time, oldest first)
        alpha: Decay factor (0-1)

    Returns:
        Weighted standard deviation
    """
    if len(values) <= 1 or values.isna().all():
        return np.nan

    values_clean = values.dropna()
    if len(values_clean) <= 1:
        return np.nan

    # Calculate weighted mean first
    wmean = exponential_weighted_mean(values_clean, alpha)
    if np.isnan(wmean):
        return np.nan

    # Create exponential weights
    n = len(values_clean)
    weights = np.array([alpha ** (n - i - 1) for i in range(n)])
    weights = weights / weights.sum()

    # Weighted variance
    variance = np.average((values_clean.values - wmean) ** 2, weights=weights)
    return float(np.sqrt(variance))


def rolling_window_stats(values: pd.Series, window: int = 4) -> Tuple[float, float]:
    """
    Calculate mean and std using only the most recent N games.

    Args:
        values: Series of historical values (ordered by time, oldest first)
        window: Number of most recent games to use (default 4 = L4)

    Returns:
        (mean, std) of last N games
    """
    if len(values) == 0 or values.isna().all():
        return np.nan, np.nan

    values_clean = values.dropna()
    if len(values_clean) == 0:
        return np.nan, np.nan

    # Take last N games
    recent = values_clean.tail(window)

    if len(recent) == 0:
        return np.nan, np.nan

    mu = float(recent.mean())
    sigma = float(recent.std(ddof=1)) if len(recent) > 1 else np.nan

    return mu, sigma


def apply_defensive_adjustment(mu: float, market_std: str, def_rating: float) -> float:
    """
    Adjust player expectation based on opponent defensive strength.

    Args:
        mu: Base player expectation (mean)
        market_std: Market type (e.g., 'pass_yds', 'rush_yds', 'receptions')
        def_rating: Defensive rating (0.5-2.0, where 1.0 = average, 2.0 = toughest)

    Returns:
        Adjusted mu

    Logic:
        - def_rating = 0.5 (weakest defense) → multiply mu by 1.15 (+15%)
        - def_rating = 1.0 (average defense) → multiply mu by 1.00 (no change)
        - def_rating = 2.0 (toughest defense) → multiply mu by 0.85 (-15%)
    """
    if pd.isna(mu) or pd.isna(def_rating):
        return mu

    # Map def_rating (0.5-2.0) to adjustment factor (1.15-0.85)
    # Linear interpolation: rating 0.5 → 1.15x, rating 1.0 → 1.0x, rating 2.0 → 0.85x
    adjustment = 1.15 - (def_rating - 0.5) * 0.2  # slope = -0.2 per rating point

    return mu * adjustment


def calculate_defensive_ratings(season: int, week: int) -> pd.DataFrame:
    """
    Calculate defensive ratings from yards allowed per game.

    Returns DataFrame with columns:
        - team: Team abbreviation
        - pass_def_rating: 0.5-2.0 (for receiving markets)
        - rush_def_rating: 0.5-2.0 (for rushing markets)

    Scale:
        - 2.0 = Toughest defense (fewest yards) → -15% to offense
        - 1.0 = Average defense → no adjustment
        - 0.5 = Easiest defense (most yards) → +15% to offense
    """
    from pathlib import Path

    # Load current season weekly data
    parquet_path = Path(f"data/weekly_player_stats_{season}.parquet")
    if not parquet_path.exists():
        logging.warning(f"No defensive data available: {parquet_path}")
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    df.columns = [c.lower() for c in df.columns]

    # Filter to weeks before current week
    df = df[df['week'] < week].copy()

    if len(df) == 0:
        logging.warning(f"No defensive data for weeks before {week}")
        return pd.DataFrame()

    # Calculate yards allowed per team
    teams = df['team'].unique()
    def_data = []

    for team in teams:
        # Get all players who played AGAINST this team
        against_team = df[df['opponent_team'] == team]

        # Sum yards gained against this team
        pass_yds_allowed = against_team['receiving_yards'].sum()
        rush_yds_allowed = against_team['rushing_yards'].sum()

        # Count games
        games = len(df[df['team'] == team]['week'].unique())

        if games > 0:
            def_data.append({
                'team': team,
                'pass_yds_per_game': pass_yds_allowed / games,
                'rush_yds_per_game': rush_yds_allowed / games,
                'games': games
            })

    def_df = pd.DataFrame(def_data)

    if len(def_df) == 0:
        return pd.DataFrame()

    # Normalize to 0.5-2.0 scale using z-scores
    # Fewer yards allowed = tougher defense = higher rating
    for stat_type in ['pass_yds_per_game', 'rush_yds_per_game']:
        mean = def_df[stat_type].mean()
        std = def_df[stat_type].std()

        if std > 0:
            z_scores = (def_df[stat_type] - mean) / std
            # Invert: fewer yards = higher rating
            rating = 1.0 - (z_scores * 0.25)
            rating = rating.clip(0.5, 2.0)
        else:
            rating = 1.0  # All average if no variation

        rating_col = stat_type.replace('_yds_per_game', '_def_rating')
        def_df[rating_col] = rating

    logging.info(f"[defense] Calculated ratings for {len(def_df)} teams")
    return def_df[['team', 'pass_def_rating', 'rush_def_rating', 'games']].set_index('team')


def create_opponent_map(props: pd.DataFrame, logs: pd.DataFrame) -> dict:
    """
    Create player → opponent team mapping.

    Returns:
        dict: {player_name: opponent_team_abbrev}

    Logic:
        1. Get player's team from game logs (most recent team)
        2. Parse game string to find opponent
    """
    # Team name to abbreviation mapping
    team_to_abbrev = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }

    opponent_map = {}

    # Get player → team mapping from logs (most recent team)
    player_team_map = {}
    if logs is not None and not logs.empty and 'player' in logs.columns and 'recent_team' in logs.columns:
        # Use most recent team per player
        player_teams = logs.groupby('player')['recent_team'].last()
        player_team_map = player_teams.to_dict()

    # Process props to find opponents
    for _, row in props.iterrows():
        player = row.get('player')
        if pd.isna(player):
            continue

        home = row.get('home_team', '')
        away = row.get('away_team', '')

        # Get player's team
        player_team = player_team_map.get(player)

        if player_team:
            # Determine opponent
            if player_team == home:
                opponent_map[player] = team_to_abbrev.get(away, away)
            elif player_team == away:
                opponent_map[player] = team_to_abbrev.get(home, home)
            else:
                # Team mismatch - player might have been traded
                # Try to infer from game string
                game = row.get('game', '')
                if home and away:
                    # Default to home team as opponent if player is away, vice versa
                    # This is a fallback heuristic
                    opponent_map[player] = team_to_abbrev.get(home, home)  # Conservative default

    logging.info(f"[opponent] Mapped opponents for {len(opponent_map)} players")
    return opponent_map


def load_player_adjustments() -> dict:
    """
    Load manual player adjustments from JSON config file.

    Returns:
        Dict mapping {name_std: {market_std: {mu_multiplier, reason}}}
    """
    import json
    from pathlib import Path

    adj_file = Path(__file__).parent.parent / "data" / "player_adjustments.json"

    if not adj_file.exists():
        return {}

    try:
        with open(adj_file, 'r') as f:
            config = json.load(f)
            return config.get("adjustments", {})
    except Exception as e:
        print(f"[WARN] Could not load player adjustments: {e}")
        return {}


def apply_player_adjustment(mu: float, name_std: str, market_std: str, adjustments: dict) -> float:
    """
    Apply manual player adjustment (injury, role change, etc).

    Args:
        mu: Base player expectation
        name_std: Standardized player name
        market_std: Market type
        adjustments: Dict from load_player_adjustments()

    Returns:
        Adjusted mu

    Example adjustment:
        {"christianmccaffrey": {"rush_yds": {"mu_multiplier": 0.0, "reason": "Out - injury"}}}
    """
    if not adjustments or pd.isna(mu):
        return mu

    player_adj = adjustments.get(name_std, {})
    market_adj = player_adj.get(market_std, {})

    multiplier = market_adj.get("mu_multiplier")
    if multiplier is not None:
        reason = market_adj.get("reason", "manual adjustment")
        adjusted = mu * multiplier
        if adjusted != mu:
            print(f"[adjustment] {name_std} {market_std}: {mu:.2f} → {adjusted:.2f} ({reason})")
        return adjusted

    return mu


# ========================================
# FAMILY-BASED MODELING (Multi-Market Coherence)
# ========================================
"""
Fourth & Value — Multi-Market "Family" Modeling

Instead of modeling each market independently, we model shared latent factors:
  - Volume (V): attempts, targets, carries
  - Efficiency (E): yards per carry, catch rate, yards per reception
  - Scoring rate (S): TD probability per touch

Each market is DERIVED from these latents with family-specific noise.
This ensures coherence (e.g., realistic YPC) and surfaces value when books
move markets independently.

Families:
  1. Rush: rush_attempts, rush_yds (derived from V_rush × YPC)
  2. Receive: targets, receptions, recv_yds (derived from V_rec × CR × YPR)
  3. Pass: pass_attempts, pass_completions, pass_yds (derived from V_pass × comp% × Y/C)
  4. Score: anytime_td (derived from rush/rec scoring rates)
"""

# Sanity bounds for efficiency metrics
YPC_MIN, YPC_MAX = 2.5, 6.5      # yards per carry (RB)
YPR_MIN, YPR_MAX = 6.0, 18.0     # yards per reception
COMP_PCT_MIN, COMP_PCT_MAX = 0.50, 0.75  # completion percentage
YPC_PASS_MIN, YPC_PASS_MAX = 5.0, 12.0   # yards per completion (QB)
CR_MIN, CR_MAX = 0.50, 0.95      # catch rate (receptions/targets)


def estimate_rush_latents(logs: pd.DataFrame, player_idx: pd.Index, alpha: float = 0.4) -> dict:
    """
    Estimate RUSH family latents: Volume (carries) and Efficiency (YPC).

    Returns dict with:
      - volume_mu: Series of expected carries per player
      - volume_sigma: Series of carry variance
      - ypc_mu: Series of expected yards per carry
      - ypc_sigma: Series of YPC variance
    """
    if logs is None or logs.empty:
        # Fall back to priors
        return {
            "volume_mu": pd.Series(PRIORS["rush_attempts"]["mu"], index=player_idx),
            "volume_sigma": pd.Series(PRIORS["rush_attempts"]["sigma"], index=player_idx),
            "ypc_mu": pd.Series(4.2, index=player_idx),  # league average
            "ypc_sigma": pd.Series(0.8, index=player_idx),
        }

    grp = logs.groupby("player", dropna=False)

    # Pick columns
    carries_col = None
    yards_col = None
    for c in ["carries", "rushing_attempts", "rush_attempts"]:
        if c in logs.columns:
            carries_col = c
            break
    for c in ["rushing_yards", "rush_yds"]:
        if c in logs.columns:
            yards_col = c
            break

    if not carries_col or not yards_col:
        # Fallback
        return {
            "volume_mu": pd.Series(PRIORS["rush_attempts"]["mu"], index=player_idx),
            "volume_sigma": pd.Series(PRIORS["rush_attempts"]["sigma"], index=player_idx),
            "ypc_mu": pd.Series(4.2, index=player_idx),
            "ypc_sigma": pd.Series(0.8, index=player_idx),
        }

    volume_mu_vals = {}
    volume_sigma_vals = {}
    ypc_mu_vals = {}
    ypc_sigma_vals = {}

    for player in player_idx:
        if player in grp.groups:
            player_data = grp.get_group(player).sort_index()
            recent = player_data.tail(4)

            if len(recent) > 0:
                # Volume (carries)
                volume_mu_vals[player] = exponential_weighted_mean(recent[carries_col], alpha)
                volume_sigma_vals[player] = exponential_weighted_std(recent[carries_col], alpha)

                # Efficiency (YPC) - compute per-game YPC then average
                ypc_series = recent[yards_col] / recent[carries_col].replace(0, np.nan)
                ypc_mu_vals[player] = exponential_weighted_mean(ypc_series.dropna(), alpha)
                ypc_sigma_vals[player] = exponential_weighted_std(ypc_series.dropna(), alpha)
            else:
                volume_mu_vals[player] = PRIORS["rush_attempts"]["mu"]
                volume_sigma_vals[player] = PRIORS["rush_attempts"]["sigma"]
                ypc_mu_vals[player] = 4.2
                ypc_sigma_vals[player] = 0.8
        else:
            volume_mu_vals[player] = PRIORS["rush_attempts"]["mu"]
            volume_sigma_vals[player] = PRIORS["rush_attempts"]["sigma"]
            ypc_mu_vals[player] = 4.2
            ypc_sigma_vals[player] = 0.8

    # Apply sanity bounds to YPC
    ypc_mu_series = pd.Series(ypc_mu_vals, index=player_idx).clip(YPC_MIN, YPC_MAX).fillna(4.2)
    ypc_sigma_series = pd.Series(ypc_sigma_vals, index=player_idx).clip(0.3, 2.0).fillna(0.8)

    return {
        "volume_mu": pd.Series(volume_mu_vals, index=player_idx).fillna(PRIORS["rush_attempts"]["mu"]),
        "volume_sigma": pd.Series(volume_sigma_vals, index=player_idx).fillna(PRIORS["rush_attempts"]["sigma"]),
        "ypc_mu": ypc_mu_series,
        "ypc_sigma": ypc_sigma_series,
    }


def estimate_receive_latents(logs: pd.DataFrame, player_idx: pd.Index, alpha: float = 0.4) -> dict:
    """
    Estimate RECEIVE family latents: Volume (targets), Catch Rate, YPR.

    Returns dict with:
      - volume_mu: targets
      - volume_sigma: target variance
      - cr_mu: catch rate (receptions/targets)
      - ypr_mu: yards per reception
      - ypr_sigma: YPR variance
    """
    if logs is None or logs.empty:
        return {
            "volume_mu": pd.Series(4.0, index=player_idx),
            "volume_sigma": pd.Series(2.0, index=player_idx),
            "cr_mu": pd.Series(0.65, index=player_idx),
            "ypr_mu": pd.Series(11.0, index=player_idx),
            "ypr_sigma": pd.Series(3.0, index=player_idx),
        }

    grp = logs.groupby("player", dropna=False)

    # Pick columns
    targets_col = "targets" if "targets" in logs.columns else None
    rec_col = "receptions" if "receptions" in logs.columns else None
    yards_col = None
    for c in ["receiving_yards", "recv_yds"]:
        if c in logs.columns:
            yards_col = c
            break

    if not rec_col or not yards_col:
        return {
            "volume_mu": pd.Series(4.0, index=player_idx),
            "volume_sigma": pd.Series(2.0, index=player_idx),
            "cr_mu": pd.Series(0.65, index=player_idx),
            "ypr_mu": pd.Series(11.0, index=player_idx),
            "ypr_sigma": pd.Series(3.0, index=player_idx),
        }

    volume_mu_vals = {}
    volume_sigma_vals = {}
    cr_mu_vals = {}
    ypr_mu_vals = {}
    ypr_sigma_vals = {}

    for player in player_idx:
        if player in grp.groups:
            player_data = grp.get_group(player).sort_index()
            recent = player_data.tail(4)

            if len(recent) > 0:
                # If we have targets, use them; otherwise estimate from receptions
                if targets_col and targets_col in recent.columns:
                    volume_mu_vals[player] = exponential_weighted_mean(recent[targets_col], alpha)
                    volume_sigma_vals[player] = exponential_weighted_std(recent[targets_col], alpha)

                    # Catch rate
                    cr_series = recent[rec_col] / recent[targets_col].replace(0, np.nan)
                    cr_mu_vals[player] = exponential_weighted_mean(cr_series.dropna(), alpha)
                else:
                    # No targets - estimate volume from receptions assuming 65% catch rate
                    rec_mu = exponential_weighted_mean(recent[rec_col], alpha)
                    volume_mu_vals[player] = rec_mu / 0.65
                    volume_sigma_vals[player] = exponential_weighted_std(recent[rec_col], alpha) / 0.65
                    cr_mu_vals[player] = 0.65

                # YPR
                ypr_series = recent[yards_col] / recent[rec_col].replace(0, np.nan)
                ypr_mu_vals[player] = exponential_weighted_mean(ypr_series.dropna(), alpha)
                ypr_sigma_vals[player] = exponential_weighted_std(ypr_series.dropna(), alpha)
            else:
                volume_mu_vals[player] = 4.0
                volume_sigma_vals[player] = 2.0
                cr_mu_vals[player] = 0.65
                ypr_mu_vals[player] = 11.0
                ypr_sigma_vals[player] = 3.0
        else:
            volume_mu_vals[player] = 4.0
            volume_sigma_vals[player] = 2.0
            cr_mu_vals[player] = 0.65
            ypr_mu_vals[player] = 11.0
            ypr_sigma_vals[player] = 3.0

    # Apply sanity bounds
    cr_mu_series = pd.Series(cr_mu_vals, index=player_idx).clip(CR_MIN, CR_MAX).fillna(0.65)
    ypr_mu_series = pd.Series(ypr_mu_vals, index=player_idx).clip(YPR_MIN, YPR_MAX).fillna(11.0)
    ypr_sigma_series = pd.Series(ypr_sigma_vals, index=player_idx).clip(1.0, 6.0).fillna(3.0)

    return {
        "volume_mu": pd.Series(volume_mu_vals, index=player_idx).fillna(4.0),
        "volume_sigma": pd.Series(volume_sigma_vals, index=player_idx).fillna(2.0),
        "cr_mu": cr_mu_series,
        "ypr_mu": ypr_mu_series,
        "ypr_sigma": ypr_sigma_series,
    }


def estimate_pass_latents(logs: pd.DataFrame, player_idx: pd.Index, alpha: float = 0.4) -> dict:
    """
    Estimate PASS family latents: Volume (attempts), Comp%, Yards/Completion.

    Returns dict with:
      - volume_mu: pass attempts
      - volume_sigma: attempt variance
      - comp_pct_mu: completion percentage
      - ypc_mu: yards per completion
      - ypc_sigma: Y/C variance
    """
    if logs is None or logs.empty:
        return {
            "volume_mu": pd.Series(32.0, index=player_idx),
            "volume_sigma": pd.Series(6.0, index=player_idx),
            "comp_pct_mu": pd.Series(0.63, index=player_idx),
            "ypc_mu": pd.Series(7.2, index=player_idx),
            "ypc_sigma": pd.Series(1.5, index=player_idx),
        }

    grp = logs.groupby("player", dropna=False)

    # Pick columns
    att_col = None
    comp_col = None
    yards_col = None
    for c in ["attempts", "passing_attempts", "pass_attempts"]:
        if c in logs.columns:
            att_col = c
            break
    for c in ["completions", "passing_completions", "pass_completions"]:
        if c in logs.columns:
            comp_col = c
            break
    for c in ["passing_yards", "pass_yds"]:
        if c in logs.columns:
            yards_col = c
            break

    if not att_col or not comp_col or not yards_col:
        return {
            "volume_mu": pd.Series(32.0, index=player_idx),
            "volume_sigma": pd.Series(6.0, index=player_idx),
            "comp_pct_mu": pd.Series(0.63, index=player_idx),
            "ypc_mu": pd.Series(7.2, index=player_idx),
            "ypc_sigma": pd.Series(1.5, index=player_idx),
        }

    volume_mu_vals = {}
    volume_sigma_vals = {}
    comp_pct_vals = {}
    ypc_mu_vals = {}
    ypc_sigma_vals = {}

    for player in player_idx:
        if player in grp.groups:
            player_data = grp.get_group(player).sort_index()
            recent = player_data.tail(4)

            if len(recent) > 0:
                # Volume (attempts)
                volume_mu_vals[player] = exponential_weighted_mean(recent[att_col], alpha)
                volume_sigma_vals[player] = exponential_weighted_std(recent[att_col], alpha)

                # Comp %
                comp_pct_series = recent[comp_col] / recent[att_col].replace(0, np.nan)
                comp_pct_vals[player] = exponential_weighted_mean(comp_pct_series.dropna(), alpha)

                # Yards per completion
                ypc_series = recent[yards_col] / recent[comp_col].replace(0, np.nan)
                ypc_mu_vals[player] = exponential_weighted_mean(ypc_series.dropna(), alpha)
                ypc_sigma_vals[player] = exponential_weighted_std(ypc_series.dropna(), alpha)
            else:
                volume_mu_vals[player] = 32.0
                volume_sigma_vals[player] = 6.0
                comp_pct_vals[player] = 0.63
                ypc_mu_vals[player] = 7.2
                ypc_sigma_vals[player] = 1.5
        else:
            volume_mu_vals[player] = 32.0
            volume_sigma_vals[player] = 6.0
            comp_pct_vals[player] = 0.63
            ypc_mu_vals[player] = 7.2
            ypc_sigma_vals[player] = 1.5

    # Apply sanity bounds
    comp_pct_series = pd.Series(comp_pct_vals, index=player_idx).clip(COMP_PCT_MIN, COMP_PCT_MAX).fillna(0.63)
    ypc_mu_series = pd.Series(ypc_mu_vals, index=player_idx).clip(YPC_PASS_MIN, YPC_PASS_MAX).fillna(7.2)
    ypc_sigma_series = pd.Series(ypc_sigma_vals, index=player_idx).clip(0.5, 3.0).fillna(1.5)

    return {
        "volume_mu": pd.Series(volume_mu_vals, index=player_idx).fillna(32.0),
        "volume_sigma": pd.Series(volume_sigma_vals, index=player_idx).fillna(6.0),
        "comp_pct_mu": comp_pct_series,
        "ypc_mu": ypc_mu_series,
        "ypc_sigma": ypc_sigma_series,
    }


def derive_market_from_latents(market_std: str, latents: dict, player_idx: pd.Index) -> tuple:
    """
    Derive (mu, sigma) for a specific market from family latents.

    Returns: (mu_series, sigma_series)
    """
    # Rush family
    if market_std == "rush_attempts":
        return latents["rush"]["volume_mu"], latents["rush"]["volume_sigma"]

    elif market_std == "rush_yds":
        # rush_yds = rush_attempts × YPC
        volume_mu = latents["rush"]["volume_mu"]
        ypc_mu = latents["rush"]["ypc_mu"]
        mu = volume_mu * ypc_mu

        # Variance: Var(X*Y) ≈ E[X]²Var(Y) + E[Y]²Var(X) for independent X,Y
        # Simplified: sigma ≈ sqrt((volume_mu * ypc_sigma)² + (ypc_mu * volume_sigma)²)
        volume_sigma = latents["rush"]["volume_sigma"]
        ypc_sigma = latents["rush"]["ypc_sigma"]
        sigma = np.sqrt((volume_mu * ypc_sigma)**2 + (ypc_mu * volume_sigma)**2)

        return mu, sigma.clip(lower=5.0)  # floor sigma for stability

    # Receive family
    elif market_std == "receptions":
        # receptions = targets × catch_rate
        volume_mu = latents["receive"]["volume_mu"]
        cr_mu = latents["receive"]["cr_mu"]
        mu = volume_mu * cr_mu

        # Simplified variance
        volume_sigma = latents["receive"]["volume_sigma"]
        sigma = cr_mu * volume_sigma  # approximate

        return mu, sigma.clip(lower=0.5)

    elif market_std == "recv_yds":
        # recv_yds = receptions × YPR = (targets × CR) × YPR
        volume_mu = latents["receive"]["volume_mu"]
        cr_mu = latents["receive"]["cr_mu"]
        ypr_mu = latents["receive"]["ypr_mu"]
        mu = volume_mu * cr_mu * ypr_mu

        # Variance approximation
        volume_sigma = latents["receive"]["volume_sigma"]
        ypr_sigma = latents["receive"]["ypr_sigma"]
        sigma = np.sqrt((volume_mu * cr_mu * ypr_sigma)**2 + (ypr_mu * cr_mu * volume_sigma)**2)

        return mu, sigma.clip(lower=5.0)

    # Pass family
    elif market_std == "pass_attempts":
        return latents["pass"]["volume_mu"], latents["pass"]["volume_sigma"]

    elif market_std == "pass_completions":
        # completions = attempts × comp%
        volume_mu = latents["pass"]["volume_mu"]
        comp_pct = latents["pass"]["comp_pct_mu"]
        mu = volume_mu * comp_pct

        volume_sigma = latents["pass"]["volume_sigma"]
        sigma = comp_pct * volume_sigma

        return mu, sigma.clip(lower=1.0)

    elif market_std == "pass_yds":
        # pass_yds = completions × Y/C = (attempts × comp%) × Y/C
        volume_mu = latents["pass"]["volume_mu"]
        comp_pct = latents["pass"]["comp_pct_mu"]
        ypc_mu = latents["pass"]["ypc_mu"]
        mu = volume_mu * comp_pct * ypc_mu

        volume_sigma = latents["pass"]["volume_sigma"]
        ypc_sigma = latents["pass"]["ypc_sigma"]
        sigma = np.sqrt((volume_mu * comp_pct * ypc_sigma)**2 + (ypc_mu * comp_pct * volume_sigma)**2)

        return mu, sigma.clip(lower=10.0)

    # Fallback for markets not in families (shouldn't happen if called correctly)
    else:
        prior = PRIORS.get(market_std, {})
        return (
            pd.Series(prior.get("mu", 0.0), index=player_idx),
            pd.Series(prior.get("sigma", 1.0), index=player_idx)
        )


def build_params(cands, logs, season, week, defensive_ratings=None, opponent_map=None):
    """
    Board-driven param builder.
    - Emits a row for every (player, market_std) present on the props board.
    - Normal markets -> dist='normal', fills mu/sigma from logs if available, else PRIORS.
    - Poisson markets -> dist='poisson', fills lam from logs if available, else PRIORS (and mirrors to mu).
    - Adds used_logs=1 per row only if that player has any logs; else 0.
    - Includes team/position/gsis_id if we can infer them from logs; else NaN.
    - Applies defensive adjustments if defensive_ratings and opponent_map provided.
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
        "pass_attempts":    ["attempts", "passing_attempts"],
        "pass_completions": ["completions", "passing_completions"],
        "rush_attempts":    ["carries", "rushing_attempts"],
    }
    LOG_COLS_POISSON = {
        "pass_tds":            ["passing_tds"],
        "pass_interceptions":  ["interceptions"],
        # anytime_td handled via rushing_tds + receiving_tds
    }

    # ========================================
    # FAMILY-BASED LATENT ESTIMATION
    # ========================================
    print("[family] Estimating latent parameters for Rush, Receive, Pass families...")

    # Estimate latent parameters for each family
    rush_latents = estimate_rush_latents(logs, player_idx, alpha=0.4)
    receive_latents = estimate_receive_latents(logs, player_idx, alpha=0.4)
    pass_latents = estimate_pass_latents(logs, player_idx, alpha=0.4)

    # Package into dict for derive_market_from_latents
    latents = {
        "rush": rush_latents,
        "receive": receive_latents,
        "pass": pass_latents,
    }

    # Derive market params from latents (ensures coherence)
    mu_map, sg_map = {}, {}
    family_markets = {
        "rush_attempts", "rush_yds",
        "receptions", "recv_yds",
        "pass_attempts", "pass_completions", "pass_yds",
    }

    for mkt in family_markets:
        mu, sg = derive_market_from_latents(mkt, latents, player_idx)
        mu_map[mkt] = mu
        sg_map[mkt] = sg

    print(f"[family] Derived params for {len(mu_map)} Normal markets from latents")

    # ========================================
    # APPLY DEFENSIVE ADJUSTMENTS
    # ========================================
    if defensive_ratings is not None and opponent_map:
        print(f"[defense] Applying defensive adjustments...")
        adjustments_applied = 0

        for mkt in family_markets:
            # Map market to defense type
            if mkt in ["receptions", "recv_yds"]:
                def_type = "pass_def_rating"
            elif mkt in ["rush_yds", "rush_attempts"]:
                def_type = "rush_def_rating"
            elif mkt in ["pass_yds", "pass_attempts", "pass_completions"]:
                def_type = "pass_def_rating"  # QB facing pass defense
            else:
                continue

            # Apply adjustment per player
            for player in player_idx:
                opponent = opponent_map.get(player)
                if opponent and opponent in defensive_ratings.index:
                    def_rating = defensive_ratings.loc[opponent, def_type]
                    old_mu = mu_map[mkt][player]
                    mu_map[mkt][player] = apply_defensive_adjustment(
                        old_mu,
                        mkt,
                        def_rating
                    )
                    adjustments_applied += 1

        print(f"[defense] Applied {adjustments_applied} defensive adjustments")

    # ========================================
    # GROUP-BASED LOGIC (backward compat for any markets not in families)
    # ========================================
    if logs is not None and not logs.empty:
        grp = logs.groupby("player", dropna=False)

    # --- precompute per-player λ for Poisson markets ---
    def lam_from(cols, prior_key, default_val=None):
        base = PRIORS.get(prior_key, {}).get("lam", np.nan) if default_val is None else float(default_val)
        if logs is None or getattr(logs, "empty", True):
            raise ValueError(f"FATAL: No weekly stats loaded for Poisson market {prior_key}. Cannot build params without player data.")

        col = pick_col(logs, cols)
        if col is None:
            available_cols = [c for c in logs.columns if 'td' in c.lower() or 'int' in c.lower()]
            raise ValueError(
                f"FATAL: Poisson market {prior_key} - no matching column from {cols}.\n"
                f"Available TD/INT columns: {available_cols[:20]}\n"
                f"Fix LOG_COLS_POISSON mapping or update weekly stats schema."
            )

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

    # --- Load manual player adjustments ---
    player_adjustments = load_player_adjustments()
    if player_adjustments:
        print(f"[adjustments] Loaded manual adjustments for {len(player_adjustments)} players")

    # Build player → name_std mapping for adjustments
    player_to_name_std = {}
    if "name_std" in cands.columns and "player" in cands.columns:
        player_to_name_std = dict(zip(cands["player"], cands["name_std"]))

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

            # Apply manual adjustments if present
            if player_adjustments and player_to_name_std:
                mu_adjusted = {}
                for player in mu.index:
                    # Get standardized name for lookup
                    name_std_val = player_to_name_std.get(player, player)
                    original_mu = mu[player]
                    adjusted_mu = apply_player_adjustment(original_mu, name_std_val, mkt, player_adjustments)
                    mu_adjusted[player] = adjusted_mu
                mu = pd.Series(mu_adjusted, index=mu.index)

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

    # ========================================
    # ADD DIAGNOSTIC COLUMNS (Family Coherence Checks)
    # ========================================
    print("[diagnostics] Adding family coherence diagnostic columns...")

    # Create diagnostic columns for coherence checking
    params["implied_ypc"] = np.nan
    params["implied_ypr"] = np.nan
    params["implied_comp_pct"] = np.nan
    params["implied_ypc_pass"] = np.nan
    params["implied_cr"] = np.nan

    # Map player to latent metrics using vectorized operations
    player_to_ypc = rush_latents["ypc_mu"].to_dict()
    player_to_ypr = receive_latents["ypr_mu"].to_dict()
    player_to_cr = receive_latents["cr_mu"].to_dict()
    player_to_comp_pct = pass_latents["comp_pct_mu"].to_dict()
    player_to_ypc_pass = pass_latents["ypc_mu"].to_dict()

    # Apply diagnostics (vectorized for performance)
    rush_mask = params["market_std"].isin(["rush_attempts", "rush_yds"])
    params.loc[rush_mask, "implied_ypc"] = params.loc[rush_mask, "player"].map(player_to_ypc)

    rec_mask = params["market_std"].isin(["receptions", "recv_yds"])
    params.loc[rec_mask, "implied_ypr"] = params.loc[rec_mask, "player"].map(player_to_ypr)
    params.loc[rec_mask, "implied_cr"] = params.loc[rec_mask, "player"].map(player_to_cr)

    pass_mask = params["market_std"].isin(["pass_attempts", "pass_completions", "pass_yds"])
    params.loc[pass_mask, "implied_comp_pct"] = params.loc[pass_mask, "player"].map(player_to_comp_pct)

    pass_yds_mask = params["market_std"].isin(["pass_completions", "pass_yds"])
    params.loc[pass_yds_mask, "implied_ypc_pass"] = params.loc[pass_yds_mask, "player"].map(player_to_ypc_pass)

    print("[diagnostics] Added implied_ypc, implied_ypr, implied_cr, implied_comp_pct, implied_ypc_pass")

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

    # 0) Add name_std for player name standardization (needed for adjustments)
    if "name_std" not in props.columns and "player" in props.columns:
        props["name_std"] = props["player"].map(std_player_name)

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

    # Always use current season for L4 (last 4 games) methodology
    # This ensures we're using the most recent games from THIS season
    logs = fetch_recent_game_logs(season)

    # Calculate defensive ratings from current season
    logging.info(f"Calculating defensive ratings for season={season}, week={week}")
    defensive_ratings = calculate_defensive_ratings(season, week)
    logging.info(f"Calculated defensive ratings for {len(defensive_ratings)} teams")

    # Create opponent lookup
    logging.info("Creating opponent map from props data")
    opponent_map = create_opponent_map(props, logs)
    logging.info(f"Mapped {len(opponent_map)} players to opponents")

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

    params = build_params(cands, logs, args.season, args.week,
                         defensive_ratings=defensive_ratings,
                         opponent_map=opponent_map)


    # tidy columns order
    from common_markets import ensure_param_schema, apply_priors_if_missing
    params = ensure_param_schema(params)
    params = apply_priors_if_missing(params)

    # Back-compat: some downstream code expects 'market'
    if "market" not in params.columns:
        params["market"] = params["market_std"]

    # Final tidy + write
    cols = ["season","week","player","player_key","name_std","market_std","market",
        "dist","mu","sigma","lam","used_logs",
        "implied_ypc","implied_ypr","implied_comp_pct","implied_ypc_pass","implied_cr"]
    params = params[[c for c in cols if c in params.columns]].copy()
    params = standardize_input(params)           # adds market_std/name/point/name_std
    params = apply_priors_if_missing(params)     # fills missing mu/sigma/lam using PRIORS


    out_csv = args.out or f"data/props/params_week{args.week}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # --- ensure join keys exist in params (permanent fix) ---
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

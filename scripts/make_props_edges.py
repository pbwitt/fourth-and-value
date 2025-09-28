#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_props_edges.py
Merge sportsbook props with model params to compute model_prob, mkt_prob, and edge_bps.

Surgical goals:
- Normalize both inputs with common_markets.standardize_input(...)
- Build a canonical point_key on BOTH frames
- Primary merge on (player_key, market_std, point_key)
- Fallback by (name_std, market_std, point_key) with an ALIGNED mask (no misaligned boolean Series)
- Compute mkt_prob from American odds; compute model_prob from Normal / Poisson params
- Write a single CSV for downstream builders

Usage:
  python3 scripts/make_props_edges.py \
    --season 2025 --week 2 \
    --props_csv  data/props/latest_all_props.csv \
    --params_csv data/props/params_week2.csv \
    --out       data/props/props_with_model_week2.csv
"""


import argparse
import logging
import math
import re
from typing import Iterable, List

import numpy as np
import pandas as pd  # Shared normalizer (you added this module)
# --- market alias normalization (local fallback if common_markets missing) ---
import re
# === Market normalization (must be defined before use) =======================
# Try to use the shared canonical normalizer; fall back to a local mapper.
# === Market normalization (single source of truth) ==========================
try:
    from scripts.common_markets import std_market as _std_market  # preferred
except Exception:
    _std_market = None

_MARKET_ALIAS = {
    "player_receiving_yds":"recv_yds","receiving_yards":"recv_yds","reception_yds":"recv_yds",
    "player_rushing_yds":"rush_yds","rushing_yards":"rush_yds",
    "player_passing_yds":"pass_yds","passing_yards":"pass_yds",
    "player_receptions":"receptions","receptions":"receptions",
    "rushing_attempts":"rush_attempts","rush_att":"rush_attempts",
    "pass_attempts":"pass_attempts","completions":"pass_completions","pass_cmp":"pass_completions",
    "player_passing_tds":"pass_tds","passing_tds":"pass_tds",
    "player_interceptions":"pass_interceptions","interceptions":"pass_interceptions",
    "player_anytime_td":"anytime_td","anytime_touchdown":"anytime_td","anytime_td":"anytime_td",# yards (short forms)
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


def _key_str(x):
    # Normalize merge keys to string-or-None (never float)
    import math
    if x is None:
        return None
    try:
        # pandas NA/NaN handling
        if x != x:  # NaN check without importing numpy
            return None
    except Exception:
        pass
    # Integers/floats → canonical string (avoid "12.0")
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if float(x).is_integer() else str(x)
    s = str(x).strip()
    return s or None


def _std_local_market(x):
    s = str(x or "").strip().lower()
    if not s: return s
    s = s.replace("player_", "").replace(" ", "_").replace("-", "_").replace("__", "_")
    return _MARKET_ALIAS.get(s, s)

def canon_market(x):
    """Canonicalize market names; use shared normalizer if present, else local map."""
    if _std_market:
        try:
            return _std_market(x)
        except Exception:
            pass
    return _std_local_market(x)
# ===========================================================================>


def _dead_std_market(x):
    """
    Canonicalize market names across sources.
    Prefers scripts.common_markets.std_market; falls back to the local mapper.
    """
    if _std_market:
        try:
            return _std_market(x)
        except Exception:
            # fall through to local if shared impl errors on a weird token
            pass
    return _std_local_market(x)
# ============================================================================#

# ---------- NORMALIZERS ----------
def std_name(s: str) -> str:
    import re
    s = (str(s) if s is not None else "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def name_slug(s: str) -> str:
    s = std_name(s)
    parts = s.split(" ")
    return (parts[0][0] + parts[-1]) if parts else ""


def norm_side(side: str, market_std: str) -> str:
    s = (str(side) if side is not None else "").strip().lower()
    if market_std == "anytime_td":
        return "yes" if s in ("yes", "y", "1", "true") else "no" if s in ("no", "n", "0", "false") else "yes"
    # O/U
    if s.startswith("o"): return "over"
    if s.startswith("u"): return "under"
    return s or "over"

def derive_side_row(row) -> str:
    # Try explicit fields first; fall back to inferring from the odds row text
    s = row.get("side") or row.get("bet") or row.get("outcome_name")
    s = (str(s) if s is not None else "")
    m = row.get("market_std")
    return norm_side(s, m)

# ---------- PRICE / PROB HELPERS ----------
def price_to_prob_american(price) -> float:
    try:
        p = float(price)
    except Exception:
        return float("nan")
    if p > 0:
        return 100.0 / (p + 100.0)
    else:
        return -p / (-p + 100.0)

def prob_to_american(p) -> float:
    if p is None or p != p or p <= 0 or p >= 1:
        return float("nan")
    return (100.0 * p / (1.0 - p)) if p >= 0.5 else (-100.0 * (1.0 - p) / p)

def edge_bps(model_p: float, market_p: float) -> float:
    if any((model_p is None, market_p is None)): return float("nan")
    if (model_p != model_p) or (market_p != market_p): return float("nan")
    return (model_p - market_p) * 1e4

# ---------- DISTRIBUTIONS ----------
from math import erf, sqrt, exp



def poisson_cdf(k, lam):
    if lam is None or lam < 0: return float("nan")
    # CDF up to k inclusive; for “Under” we’ll use floor(point), etc.
    # simple summation (k is tiny: TDs, INTs)
    k = int(k)
    s = 0.0
    term = exp(-lam)
    s += term
    for i in range(1, k + 1):
        term *= lam / i
        s += term
    return s

def ou_prob(point, mu, sigma, side: str) -> float:
    if side == "over":
        # P(X > point) = 1 - CDF(point)
        return 1.0 - normal_cdf(point, mu, sigma)
    else:
        # P(X < point) = CDF(point)
        return normal_cdf(point, mu, sigma)

def poisson_ou_prob(point, lam, side: str) -> float:
    # Treat “Under point” as P(X < point) ~= P(X <= floor(point - 1e-9))
    # and “Over” as 1 - CDF(floor(point))
    from math import floor
    k = floor(point + 1e-9)
    c = poisson_cdf(k, lam)
    return (1.0 - c) if side == "over" else c

# ---------- CORE MERGE ----------
def prepare_and_merge(props: "pd.DataFrame", params: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    # Normalize keys on both frames
    for df, src in ((props, "props"), (params, "params")):
        if "market_std" in df:
            df["market_std"] = df["market_std"].apply(canon_market)
        elif "market" in df:
            df["market_std"] = df["market"].apply(canon_market)
        else:
            raise RuntimeError(f"[{src}] missing market/market_std")

        # Names
        name_col = "player" if "player" in df else ("name" if "name" in df else None)
        if name_col is None:
            raise RuntimeError(f"[{src}] missing player/name column")
        df["name"] = df[name_col]
        df["name_std"] = df["name"].map(std_name)
        df["name_slug"] = df["name"].map(name_slug)

        # Coerce player_key to string-or-None on BOTH frames to avoid float/object merge errors
        # ensure column exists on both frames
        for df in (props, params):
            if "player_key" not in df.columns:
                df["player_key"] = pd.NA

        # coerce to uniform string-or-None to avoid float/object merge errors
        props["player_key"]  = props["player_key"].apply(_key_str)
        params["player_key"] = params["player_key"].apply(_key_str)



        # Player key if available
        if "player_key" not in df:
            df["player_key"] = pd.NA

    # Side normalization (props only)
    if "side" not in props:
        props["side"] = props.apply(derive_side_row, axis=1)
    props["side"] = props.apply(lambda r: norm_side(r["side"], r["market_std"]), axis=1)

    # Never join on point/line. Use stable keys with fallbacks.
    def _merge_on(keys):
        cand = props.merge(params, how="left", on=keys, suffixes=("", "_param"), indicator=True)
        return cand

    # Try strictest first
    attempts = [
        (["player_key", "market_std"], "player_key+market"),
        (["name_std",  "market_std"], "name_std+market"),
        (["name_slug", "market_std"], "name_slug+market"),
    ]

    merged = None
    for keys, label in attempts:
        c = _merge_on(keys)
        matched = (c["_merge"] == "both").mean()
        if matched >= 0.90 or merged is None:  # keep best so far; accept once we’re ≥90%
            merged = c
        if matched >= 0.98:
            break

    # --- after merged = ( ... your merge attempts ... ) and before QC ---

# Remove obvious team/defense props that we don't model
def _is_player_like(n):
    s = str(n).lower()
    if not s or s == "nan":
        return False
    bad = (" d/st", " defense", " special teams", " team", " dst")
    return not any(k in s for k in bad)


def _model_prob_row(r):
    m = r["market_std"]; side = r["side"]
    if m in ("rush_yds","recv_yds","pass_yds","receptions","rush_attempts","pass_attempts","pass_completions"):
        return ou_prob(r["point"], r["mu"], r["sigma"], side)
    if m in ("pass_tds","pass_interceptions","anytime_td"):
        # point default ~0.5 for >0 events
        pnt = r["point"] if pd.notna(r["point"]) else 0.5
        return poisson_ou_prob(pnt, r["lam"], side)
    return float("nan")

def _model_line_row(r):
    m = r["market_std"]
    if m in ("rush_yds","recv_yds","pass_yds","receptions","rush_attempts","pass_attempts","pass_completions"):
        return r["mu"]
    if m in ("pass_tds","pass_interceptions","anytime_td"):
        return r["lam"]
    return float("nan")

    # ---- Modeling (after merge + side + normalization) ----
    # Identify rows that actually have parameters
    mask = (merged["lam"].notna()) | (merged["mu"].notna() & merged["sigma"].notna())
    merged["_has_params"] = mask

    # Compute model probability & model line ONLY where params exist
    merged.loc[mask, "model_prob"] = merged.loc[mask].apply(_model_prob_row, axis=1)
    merged.loc[mask, "model_line"] = merged.loc[mask].apply(_model_line_row, axis=1)

    # Book side (use the canonical name expected by pages/QC)
    merged["mkt_prob"] = merged["price"].apply(price_to_prob_american)

    # Fair odds from model_prob — single assignment, NaN-safe
    def _prob_to_american_safe(p):
        if isinstance(p, (int, float)) and 0 < p < 1:
            return 100.0 * p / (1.0 - p) if p >= 0.5 else -100.0 * (1.0 - p) / p
        return np.nan

    merged["fair_odds"] = merged["model_prob"].map(_prob_to_american_safe)

    # Edge (basis points) — requires both model_prob and mkt_prob
    merged["edge_bps"] = (merged["model_prob"] - merged["mkt_prob"]) * 10000.0

    # ---- QC (non-blocking) ----
    publishable = merged.loc[merged["_has_params"]].copy()
    modeled_share = publishable["model_prob"].notna().mean() if not publishable.empty else 0.0
    print(f"[model] modeled_share among rows with params: {modeled_share:.1%}")

    # Anti-join report BEFORE dropping _merge
    anti = merged.loc[merged["_merge"] == "left_only", ["name", "name_std", "market_std"]].copy() \
           if "_merge" in merged.columns else pd.DataFrame()
    if len(anti) > 0:
        print(f"WARNING: {len(anti)} props rows failed to find params; sample:\n{anti.head(10).to_string(index=False)}")

    merged.drop(columns=["_merge"], inplace=True, errors="ignore")

    by_mkt = (
        merged.assign(_had_params = merged["lam"].notna() | (merged["mu"].notna() & merged["sigma"].notna()))
              .groupby("market_std")["_had_params"].mean()
              .sort_values(ascending=True)
    )
    print("[coverage] share with params by market:\n", by_mkt.to_string())

    # Ensure expected columns exist (don’t hard-fail; pages will hide blanks)
    for col in ("mu", "sigma", "lam", "point", "price"):
        if col not in merged:
            merged[col] = pd.NA


    # Compute model_prob / fair_odds / edge
    # Ensure columns
    for col in ("mu", "sigma", "lam"):
        if col not in merged:
            merged[col] = pd.NA

    # Point, price
    if "point" not in merged:
        merged["point"] = pd.NA
    if "price" not in merged:
        merged["price"] = pd.NA

    # Model probability by market family
    def _model_prob_row(r):
        m = r["market_std"]
        side = r["side"]
        if m in ("rush_yds", "recv_yds", "pass_yds", "receptions", "rush_attempts", "pass_attempts", "pass_completions"):
            return ou_prob(r["point"], r["mu"], r["sigma"], side)
        if m in ("pass_tds", "pass_interceptions", "anytime_td"):
            return poisson_ou_prob(r["point"] if pd.notna(r["point"]) else 0.5, r["lam"], side)
        return float("nan")

    merged["model_prob"] = merged.apply(_model_prob_row, axis=1)

    # Model line (50th percentile) for OU normals; for Poisson show λ as "model_line" when usable
    def _model_line_row(r):
        m = r["market_std"]
        if m in ("rush_yds", "recv_yds", "pass_yds", "receptions", "rush_attempts", "pass_attempts", "pass_completions"):
            return r["mu"]
        if m in ("pass_tds", "pass_interceptions", "anytime_td"):
            return r["lam"]
        return float("nan")

    merged["model_line"] = merged.apply(_model_line_row, axis=1)

    # Market implied prob from American odds
    merged["market_prob"] = merged["price"].apply(price_to_prob_american)

    # Fair odds from model prob
    merged["fair_odds"] = merged["model_prob"].apply(prob_to_american)

    # Edge in bps
    merged["edge_bps"] = merged.apply(lambda r: edge_bps(r["model_prob"], r["market_prob"]), axis=1)

    # Display helpers
    def _fmt_american(x):
        try:
            v = float(x)
        except Exception:
            return "—"
        if v != v: return "—"
        return f"{int(v) if abs(v) >= 1 else round(v, 2)}"

    merged["book_price_disp"] = merged["price"].apply(_fmt_american)

    # Required output columns union
    desired_cols = [
        "game", "game_id", "commence_time", "home_team", "away_team",
        "bookmaker", "bookmaker_title", "market_std", "market", "player", "name",
        "side", "point", "price", "book_price_disp",
        "model_prob", "market_prob", "edge_bps", "fair_odds", "model_line",
        "mu", "sigma", "lam", "name_std", "name_slug", "player_key",
    ]
    for dc in desired_cols:
        if dc not in merged:
            merged[dc] = pd.NA

    # QC: hard checks (fail fast)
    import numpy as np
    if merged["market_std"].isna().mean() > 0.05:
        raise RuntimeError("props: >5% markets unmapped to canonical set")
    if (merged["model_prob"].notna().mean() < 0.4):  # at least 40% modeled initially
        raise RuntimeError("merge/model: too few rows have model_prob — check params join and market aliases")

    # Optional de-dup (same book/game/market/player/point rows)
    dedupe_keys = ["bookmaker", "game_id", "market_std", "name_std", "point", "side"]
    before = len(merged)
    merged = (merged
              .sort_values(by=["edge_bps"], ascending=False)
              .drop_duplicates(subset=dedupe_keys, keep="first"))
    after = len(merged)
    if before != after:
        rate = (before - after) / max(before, 1) * 100
        print(f"[dedupe] removed {before - after} duplicates ({rate:.2f}%) on {dedupe_keys}")

    return merged





try:
    from scripts.common_markets import std_market  # preferred, if available
except Exception:
    ALIAS = {
        "reception_yds": "recv_yds",
        "receiving_yards": "recv_yds",
        "rush_att": "rush_attempts",
        "interceptions": "pass_interceptions",  # fixes 'interceptions' bucket
    }
    def _dead_std_market(s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"\s+", "_", s)
        return ALIAS.get(s, s)

# --- imports near the top ---
# bad (remove this):
# from common_markets import standardize_input, std_market, std_name

# good (handles both `python -m scripts.make_props_edges` and `python scripts/make_props_edges.py`)
try:
    from scripts.common_markets import standardize_input, std_market
except ModuleNotFoundError:
    try:
        from common_markets import standardize_input, std_market  # fallback if run directly
    except ModuleNotFoundError:
        import sys, os
        sys.path.append(os.path.dirname(__file__))  # last-ditch: add scripts/ to path
        from common_markets import standardize_input, std_market





def slug_no_space(s: str) -> str:
    # compressed key for last-ditch joins, e.g. "kyler murray" -> "kylermurray"
    return re.sub(r"\s+", "", std_name(s))


def normalize_side(row: pd.Series) -> str:
    mkt = str(row.get("market_std", ""))
    for col in ("side", "selection", "outcome", "label", "bet", "pick"):
        v = str(row.get(col, "")).strip().lower()
        if v in ("over", "under", "yes", "no"):
            if "anytime_td" in mkt:
                return (
                    "Yes"
                    if v == "yes"
                    else "No" if v == "no" else "Over" if v == "over" else "Under"
                )
            return "Over" if v == "over" else ("Under" if v == "under" else ("Yes" if v == "yes" else "No"))
    p = row.get("model_prob", np.nan)
    if isinstance(p, (int, float)) and not pd.isna(p) and "anytime_td" not in mkt:
        return "Over" if float(p) >= 0.5 else "Under"
    if "anytime_td" in mkt:
        return "Yes"
    return ""


def _get_series(df: pd.DataFrame, *columns: str) -> pd.Series:
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series("", index=df.index)


from pathlib import Path
from datetime import datetime, timezone
import os

HIST_DIR = Path("/Users/pwitt/fourth-and-value/data/preds_historical")
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



# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--props_csv", required=True)
    ap.add_argument("--params_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loglevel", default="INFO")
    return ap.parse_args()


# ---------------------- Helpers ----------------------
def american_to_prob(price) -> float:
    """Convert American odds to implied probability (no de-vig)."""
    try:
        p = float(price)
    except Exception:
        return np.nan
    if p > 0:
        return 100.0 / (p + 100.0)
    elif p < 0:
        return (-p) / ((-p) + 100.0)
    return np.nan


def _point_key(v) -> str:
    """Canonicalize O/U thresholds so 65.0 == '65'."""
    try:
        f = float(v)
        return str(int(f)) if math.isfinite(f) and float(f).is_integer() else str(f)
    except Exception:
        return "" if pd.isna(v) else str(v)


def normal_cdf(x, mu, sigma) -> float:
    """Φ((x-mu)/σ) via erf. Returns NaN when σ<=0."""
    try:
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            return np.nan
        z = (x - mu) / (sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))
    except Exception:
        return np.nan


def poisson_cdf(k, lam) -> float:
    """P(X<=k) for X~Poisson(lam)."""
    try:
        if lam is None or lam < 0 or not np.isfinite(lam):
            return np.nan
        k = int(math.floor(float(k)))
        s = 0.0
        for i in range(0, k + 1):
            # use log to avoid overflow: exp(i*log(lam) - lam - log(i!))
            s += math.exp(i * math.log(lam) - lam - math.lgamma(i + 1))
        return min(max(s, 0.0), 1.0)
    except Exception:
        return np.nan


def modeled_market_kind(market_std: str) -> str:
    """
    Classify into 'normal_ou', 'poisson_ou', 'binary_anytime', or 'other'.
    This matches the params you already produce (mu/sigma or lam).
    """
    m = str(market_std or "").lower()
    # Normal O/U style markets
    if m in {
        "recv_yds", "receptions", "rush_yds", "rush_attempts",
        "pass_yds", "pass_attempts", "pass_completions"
    }:
        return "normal_ou"
    # Poisson O/U (counts) commonly modeled
    if m in {"pass_tds", "interceptions"}:
        return "poisson_ou"
    # Binary anytime TD
    if m in {"anytime_td"}:
        return "binary_anytime"
    return "other"


def compute_model_prob_row(row) -> float:
    """
    Compute model probability for a single row given:
      - row['market_std'], row['name'] (side), row['point']
      - row['mu'], row['sigma'], row['lam']
    """
    side = str(row.get("name", "")).strip().lower()  # Over/Under/Yes/No
    market = str(row.get("market_std", "")).strip().lower()
    point = row.get("point", np.nan)
    mu = row.get("mu", np.nan)
    sigma = row.get("sigma", np.nan)
    lam = row.get("lam", np.nan)

    kind = modeled_market_kind(market)

    # Normal OU: P(X > point) or P(X < point)
    if kind == "normal_ou":
        # Convert to float when possible
        try:
            x = float(point)
        except Exception:
            return np.nan
        cdf = normal_cdf(x, mu, sigma)
        if not np.isfinite(cdf):
            return np.nan
        if side == "over":
            # strictly > threshold; continuity correction not applied
            return 1.0 - cdf
        elif side == "under":
            return cdf
        else:
            return np.nan

    # Poisson OU: counts (e.g., pass_tds, interceptions)
    if kind == "poisson_ou":
        try:
            x = float(point)
        except Exception:
            return np.nan
        # Most props are half-integers (e.g., 0.5, 1.5) → use floor for <=,
        # Over 1.5 ⇒ P(X >= 2) = 1 - P(X <= 1)
        if side == "over":
            return 1.0 - poisson_cdf(math.floor(x), lam)
        elif side == "under":
            # Under 1.5 ⇒ P(X <= 1)
            return poisson_cdf(math.floor(x), lam)
        else:
            return np.nan

    # Binary anytime TD: P(X >= 1) = 1 - exp(-lam)
    if kind == "binary_anytime":
        p_yes = 1.0 - math.exp(-lam) if (lam is not None and lam >= 0 and math.isfinite(lam)) else np.nan
        if side in ("yes", "over"):     # sometimes books label as Over on 0.5
            return p_yes
        elif side in ("no", "under"):
            return 1.0 - p_yes
        else:
            return np.nan

    # Unknown / not modeled
    return np.nan


def ensure_cols(df: pd.DataFrame, cols: Iterable[str], default=np.nan) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = default


# ---------------------- Loaders ----------------------
def load_props(props_csv: str) -> pd.DataFrame:
    df = pd.read_csv(props_csv, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = standardize_input(df)  # adds market_std, name (OU/Yes/No), point (float), name_std, player_key, etc.

    if "price" in df.columns:
        # one price column (american odds)
        pass
    else:
        # try common alternates
        for cand in ("american", "odds", "offer_price"):
            if cand in df.columns:
                df["price"] = df[cand]
                break
        if "price" not in df.columns:
            df["price"] = np.nan

    ensure_cols(df, ["point"])
    df["point_key"] = df["point"].map(_point_key)
    return df


def load_params(params_csv: str) -> pd.DataFrame:
    df = pd.read_csv(params_csv, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = standardize_input(df)  # ensures market_std/name_std/point exist consistently

    ensure_cols(df, ["point"])
    df["point_key"] = df["point"].map(_point_key)

    # required model fields
    need = ["player_key", "name_std", "market_std", "mu", "sigma", "lam"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Params CSV missing columns: {missing}")

    if "used_logs" not in df.columns:
        df["used_logs"] = 0
    else:
        df["used_logs"] = df["used_logs"].fillna(0).astype(int)

    return df


# ---------------------- Main ----------------------
def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.loglevel).upper(), logging.INFO),
        format="%(levelname)s:%(message)s",
    )

    logging.info(f"[merge] loading props from {args.props_csv}")
    props  = standardize_input(load_props(args.props_csv))

    logging.info(f"[merge] loading params from {args.params_csv}")
    params = standardize_input(load_params(args.params_csv))

    # --- Normalize merge keys --------------------------------------------------
    merged = prepare_and_merge(props, params)


        # --- Normalize market + side consistently on BOTH frames -------------------
    try:
        # if you have the shared mapper, use it
        from scripts.common_markets import std_market
    except Exception:
        ALIAS = {
            "reception_yds": "recv_yds",
            "receiving_yards": "recv_yds",
            "rush_att": "rush_attempts",
            "pass_tds": "pass_tds",         # keep stable even if sources vary
            "pass_interceptions": "pass_ints",
        }
        def _dead_std_market(s: str) -> str:
            s = str(s or "").strip().lower().replace(" ", "_")
            return ALIAS.get(s, s)

    def norm_side(s):
        if s is None: return ""
        s = str(s).strip().lower()
        return {"o":"Over","over":"Over","u":"Under","under":"Under","y":"Yes","yes":"Yes","n":"No","no":"No"}.get(s, s.title())

    for df in (props, params):
        if "market_std" not in df.columns and "market" in df.columns:
            df["market_std"] = df["market"].map(canon_market)
        else:
            df["market_std"] = df["market_std"].map(canon_market)
        if "side" in df.columns:
            df["side"] = df["side"].map(norm_side)


    # ---------------- Merge params ----------------
    # === Begin params merge (no line in keys) ==================================
# We intentionally do NOT join on line/point_key. Params are per (player, market).
    key_sets = []

    # Strongest keys first
    if all(k in props.columns and k in params.columns for k in ("player_key", "market_std")):
        key_sets.append(["player_key", "market_std"])

    # Fallbacks (names) — still without line
    if all(k in props.columns and k in params.columns for k in ("name_std", "market_std")):
        key_sets.append(["name_std", "market_std"])
    if all(k in props.columns and k in params.columns for k in ("name_slug", "market_std")):
        key_sets.append(["name_slug", "market_std"])


    best = None
    best_hr = -1.0


    tmp = best if best is not None else props.copy()
    merge_hit_rate = best_hr if best is not None else 0.0
    #logging.info(f"[merge] param merge hit rate: {merge_hit_rate:.2%}")


# === End params merge =======================================================


    for col in ("mu", "sigma", "lam", "model_prob", "model_line"):
        param_col = f"{col}_param"
        if col in tmp.columns and param_col in tmp.columns:
            tmp[col] = np.where(tmp[col].isna(), tmp[param_col], tmp[col])

    drop_cols = [c for c in tmp.columns if c.endswith("_param")]
    if "_merge" in tmp.columns:
        drop_cols.append("_merge")
    merged = tmp.drop(columns=drop_cols, errors="ignore")

    # ---- normalize side + point BEFORE computing model_prob ----
# many feeds put "Over 65.5" / "UNDER" / "Yes" in various columns; normalize to {over, under, yes, no}
    if "side" in merged.columns:
        merged["side"] = merged["side"].astype(str).str.title()

    merged["name"] = merged.get("name", "").astype(str).str.strip().str.lower()
    if "side" in merged.columns:
        empty = merged["name"].isin(["", "nan", "none", "null"])
        merged.loc[empty, "name"] = merged.loc[empty, "side"].astype(str).str.strip().str.lower()

    # if there are alternate side columns, use them to fill missing
    for alt in ("selection", "bet_selection", "bet_side", "over_under"):
        if alt in merged.columns:
            empty = merged["name"].isin(["", "nan", "none", "null"])
            merged.loc[empty, "name"] = merged.loc[empty, alt].astype(str).str.strip().str.lower()

    # collapse strings that *contain* over/under/yes/no into canonical tokens
    merged.loc[merged["name"].str.contains(r"\bover\b",  na=False), "name"] = "over"
    merged.loc[merged["name"].str.contains(r"\bunder\b", na=False), "name"] = "under"
    merged.loc[merged["name"].str.contains(r"\byes\b",   na=False), "name"] = "yes"
    merged.loc[merged["name"].str.contains(r"\bno\b",    na=False), "name"] = "no"

    # ensure numeric threshold for O/U markets
    merged["point"] = pd.to_numeric(merged.get("point", np.nan), errors="coerce")
    # ---------------- Probabilities & edges ----------------
    # Market implied prob from American price
    merged["mkt_prob"] = merged["price"].map(american_to_prob)

    # Model probability
    merged["model_prob"] = merged.apply(compute_model_prob_row, axis=1)

    # Edge in basis points
    merged["edge_bps"] = (merged["model_prob"] - merged["mkt_prob"]) * 10000.0

    # Attach season/week if not present
    if "season" not in merged.columns:
        merged["season"] = args.season
    if "week" not in merged.columns:
        merged["week"] = args.week

    # Tidy & write
    # Keep original columns and add our metrics; do not drop unknowns to stay compatible with builders
    out_cols_priority: List[str] = [
        "season", "week", "game", "player", "name_std", "player_key",
        "book", "market_std", "name", "point", "point_key", "price",
        "mkt_prob", "model_prob", "edge_bps",
        "mu", "sigma", "lam",'fair_odds'
    ]
    # maintain any others too
    out_cols = [c for c in out_cols_priority if c in merged.columns] + \
               [c for c in merged.columns if c not in out_cols_priority]

    merged.to_csv(args.out, index=False)

    # ensure folder exists
    out_dir = "/Users/pwitt/fourth-and-value/data/preds_historical"
    os.makedirs(out_dir, exist_ok=True)

    # pick up week from argparse/env/df (in that order)
    week = (getattr(args, "week", None)
            or os.getenv("WEEK")
            or (str(int(merged["week"].max())) if "week" in merged.columns and merged["week"].notna().any() else "unknown"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{out_dir}/props_with_model_week{week}_{stamp}.csv"









if __name__ == "__main__":
    main()

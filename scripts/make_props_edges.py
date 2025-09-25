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


def std_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s


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
    props["market_std"] = _get_series(props, "market_std", "market").apply(std_market)
    params["market_std"] = _get_series(params, "market_std", "market").apply(std_market)

    props_name_src = _get_series(props, "player", "name")
    props["name_std"] = props_name_src.fillna("").apply(std_name)
    props["name_slug"] = props["name_std"].apply(slug_no_space)

    params_name_src = _get_series(params, "player", "name")
    params["name_std"] = params_name_src.fillna("").apply(std_name)
    params["name_slug"] = params["name_std"].apply(slug_no_space)

    props["side"] = props.apply(normalize_side, axis=1)
    props["side"] = props["side"].astype(str).str.title().replace(
        {"Yes": "Yes", "No": "No", "Over": "Over", "Under": "Under"}
    )

    if "side" in params.columns:
        params["side"] = params["side"].astype(str).str.title()
    else:
        params["side"] = ""

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
        def std_market(s: str) -> str:
            s = str(s or "").strip().lower().replace(" ", "_")
            return ALIAS.get(s, s)

    def norm_side(s):
        if s is None: return ""
        s = str(s).strip().lower()
        return {"o":"Over","over":"Over","u":"Under","under":"Under","y":"Yes","yes":"Yes","n":"No","no":"No"}.get(s, s.title())

    for df in (props, params):
        if "market_std" not in df.columns and "market" in df.columns:
            df["market_std"] = df["market"].map(std_market)
        else:
            df["market_std"] = df["market_std"].map(std_market)
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

    def _merge_on(keys):
        return props.merge(params, how="left", on=keys, suffixes=("", "_param"), indicator=True)

    best = None
    best_hr = -1.0
    for keys in key_sets:
        cand = _merge_on(keys)
        hr = cand["_merge"].eq("both").mean() if "_merge" in cand.columns else 0.0
        if hr > best_hr:
            best, best_hr = cand, hr

    tmp = best if best is not None else props.copy()
    merge_hit_rate = best_hr if best is not None else 0.0
    logging.info(f"[merge] param merge hit rate: {merge_hit_rate:.2%}")


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
        "mu", "sigma", "lam",
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

    merged.to_csv(out_path, index=False)
    print(f"[backup] wrote {out_path}")




    logging.info(f"[merge] wrote {args.out} with {len(merged):,} rows")


if __name__ == "__main__":
    main()

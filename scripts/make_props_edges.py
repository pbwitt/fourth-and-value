#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge sportsbook props with model parameters and compute edges."""

import argparse
import logging
import os
from datetime import datetime, timezone
from math import erf, sqrt, exp
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.common_markets import std_market as _std_market, standardize_input  # type: ignore
except Exception:  # noqa: BLE001
    _std_market = None

    def standardize_input(df, *_, **__):  # type: ignore
        return df


MARKET_ALIAS = {
    "player_passing_yards": "pass_yds",
    "player_pass_yds": "pass_yds",
    "passing_yards": "pass_yds",
    "player_rushing_yards": "rush_yds",
    "player_rush_yds": "rush_yds",
    "rushing_yards": "rush_yds",
    "player_receiving_yards": "recv_yds",
    "player_reception_yds": "recv_yds",
    "receiving_yards": "recv_yds",
    "reception_yds": "recv_yds",
    "player_receptions": "receptions",
    "player_rushing_attempts": "rush_attempts",
    "player_rush_attempts": "rush_attempts",
    "rush_attempts": "rush_attempts",
    "rush_att": "rush_attempts",
    "rushing_attempts": "rush_attempts",
    "carries": "rush_attempts",
    "player_passing_attempts": "pass_attempts",
    "player_pass_attempts": "pass_attempts",
    "passing_attempts": "pass_attempts",
    "attempts": "pass_attempts",
    "player_passing_completions": "pass_completions",
    "player_pass_completions": "pass_completions",
    "passing_completions": "pass_completions",
    "completions": "pass_completions",
    "player_passing_tds": "pass_tds",
    "player_pass_tds": "pass_tds",
    "passing_tds": "pass_tds",
    "player_pass_interceptions": "pass_interceptions",
    "player_interceptions": "pass_interceptions",
    "pass_interception": "pass_interceptions",
    "passing_interceptions": "pass_interceptions",
    "interceptions": "pass_interceptions",
    "player_anytime_td": "anytime_td",
    "player_1st_td": "first_td",
    "player_first_td": "first_td",
    "player_last_td": "last_td",
    "player_reception_longest": "recv_longest",
    "reception_longest": "recv_longest",
    "player_rush_longest": "rush_longest",
    "rush_longest": "rush_longest",
}

SIDE_CANDIDATES = (
    "side",
    "name",          # Odds API: Over/Under/Yes/No often lives here
    "selection",
    "outcome",
    "outcome_name",
    "label",
    "bet",
    "bet_type",
    "pick",
)


HIST_DIR = Path("/Users/pwitt/fourth-and-value/data/preds_historical")
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canon_market(raw: Optional[str]) -> str:
    """Canonicalize market identifiers using shared helper when available."""
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    s = s.replace("player_", "").replace("__", "_")
    if _std_market is not None:
        try:
            return _std_market(s)
        except Exception:  # noqa: BLE001
            pass
    return MARKET_ALIAS.get(s, s)


def std_name(raw: Optional[str]) -> str:
    """Lower-case alphanumeric slug (e.g., "Austin Ekeler" -> "austin ekeler")."""
    s = (str(raw) if raw is not None else "").strip().lower()
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


def normalize_side_value(raw: Optional[str], market_std: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if market_std == "anytime_td":
        if s in {"yes", "y", "1", "true"}:
            return "yes"
        if s in {"no", "n", "0", "false"}:
            return "no"
        return "yes"
    if s in {"over", "under"}:
        return s
    if s.startswith("o"):
        return "over"
    if s.startswith("u"):
        return "under"
    return ""


def choose_first(df: pd.DataFrame, columns: Sequence[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None

# --- helpers ---

# American odds → implied probability
def price_to_prob_american(price):
    try:
        p = float(price)
    except Exception:  # noqa: BLE001
        return float("nan")
    if p > 0:
        return 100.0 / (p + 100.0)
    elif p < 0:
        return (-p) / ((-p) + 100.0)
    return float("nan")


# probability → American odds
def prob_to_american(prob):
    try:
        p = float(prob)
    except Exception:  # noqa: BLE001
        return float("nan")
    if not (0 < p < 1):
        return float("nan")
    # +odds for underdog (p<0.5), -odds for favorite
    if p < 0.5:
        return round((100.0 * (1 - p)) / p)
    else:
        return round(-(100.0 * p) / (1 - p))


# edge in basis points; NaN-safe
def edge_bps(model_p, market_p):
    try:
        mp = float(model_p)
        qp = float(market_p)
    except Exception:  # noqa: BLE001
        return float("nan")
    if any([(mp != mp), (qp != qp)]):  # NaN check
        return float("nan")
    return round((mp - qp) * 10000.0, 1)


# Determine if a row is modelable (has μ/σ for Normal, λ for Poisson) and a valid side
NORMAL_MKTS   = {"rush_yds","recv_yds","pass_yds","receptions",
                 "rush_attempts","pass_attempts","pass_completions"}
POISSON_MKTS  = {"anytime_td","pass_tds","pass_interceptions"}
VALID_SIDES   = {"over","under","yes","no"}  # we’ll only model over/under/yes


def _has_params(r):
    m = str(r.get("market_std", ""))
    if m in NORMAL_MKTS:
        return (r.get("mu") is not None) and (r.get("sigma") is not None)
    if m in POISSON_MKTS:
        return (r.get("lam") is not None)
    return False


def _side_ok(r):
    s = str(r.get("side", "")).strip().lower()
    return s in VALID_SIDES


# Model probability by row (vectorized apply uses this)
def _norm_cdf(x, mu, sigma):
    if sigma is None or sigma <= 0:
        return float("nan")
    z = (x - mu) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def _poisson_ccdf(k, lam):
    # P(X >= k) for Poisson(λ) (for “Over” on integer k-0.5 approximation)
    if lam is None or lam < 0:
        return float("nan")
    # use a simple tail approximation via survival sum until it’s small (k is often small in props)
    k = int(round(k))
    # cap to avoid infinite loops if k huge
    max_terms = 200
    term = exp(-lam)
    s = term  # P(X=0)
    for i in range(1, k):
        if i > max_terms:
            break
        term *= lam / i
        s += term
    # P(X >= k) = 1 - P(X <= k-1)
    return 1.0 - s


def _model_prob_row(r):
    m = str(r["market_std"])
    s = str(r.get("side", "")).strip().lower()
    line = r.get("point")
    if m in NORMAL_MKTS:
        mu, sigma = r.get("mu"), r.get("sigma")
        if mu is None or sigma is None or line is None:
            return float("nan")
        # Over = 1 - CDF(line), Under = CDF(line)
        cdf = _norm_cdf(float(line), float(mu), float(sigma))
        if s == "over":
            return 1.0 - cdf
        if s == "under":
            return cdf
        return float("nan")
    elif m in POISSON_MKTS:
        lam = r.get("lam")
        if lam is None:
            return float("nan")

        # Anytime TD does NOT use a numeric line
        if m == "anytime_td":
            if s == "yes":   # P(X >= 1)
                return _poisson_ccdf(1, float(lam))
            if s == "no":    # P(X = 0) = 1 - P(X >= 1)
                return 1.0 - _poisson_ccdf(1, float(lam))
            return float("nan")

        # Other Poisson O/U markets require a valid numeric line
        line = r.get("point")
        try:
            L = float(line)
            if np.isnan(L):
                return float("nan")
        except Exception:
            return float("nan")

        from math import ceil
        k = ceil(L)  # typical 0.5 lines → ceil(0.5)=1

        if s == "over":
            return _poisson_ccdf(k, float(lam))
        if s == "under":
            # Under = P(X <= k-1) = 1 - P(X >= k)
            return 1.0 - _poisson_ccdf(k, float(lam))
        return float("nan")



def _model_line_row(r):
    m = str(r["market_std"])
    if m in NORMAL_MKTS:
        return r.get("mu")
    if m in POISSON_MKTS:
        return r.get("lam")
    return float("nan")


def determine_side(row: pd.Series) -> str:
    market_std = row.get("market_std", "")
    for col in SIDE_CANDIDATES:
        if col in row and row[col] is not None:
            normalized = normalize_side_value(row[col], market_std)
            if normalized:
                return normalized
    return ""


def merge_props_with_params_strict(props: pd.DataFrame, params: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    # Normalize keys used for the join; DO NOT touch names
    for df in (props, params):
        if "market_std" in df:
            df["market_std"] = df["market_std"].astype(str).map(canon_market)
        if "player_key" in df:
            df["player_key"] = df["player_key"].astype(str).str.strip().str.lower()

    # strict key join (no line/point in keys, no name fallbacks)
    merged = props.merge(
        params,
        how="left",
        on=["player_key", "market_std"],
        suffixes=("", "_param"),
        indicator=True,
    )

    left_only = merged["_merge"].eq("left_only").sum()
    hit_rate = 1.0 - (left_only / len(merged) if len(merged) else 0.0)
    logging.info(f"[merge] strict hit rate: {hit_rate:.2%}  (left_only={left_only:,}/{len(merged):,})")

    # Write a small anti-join sample for debugging if any misses
    if left_only > 0:
        out_path = Path("data/qc") / f"anti_join_week{week or 0}.csv"
        cols = [
            "player_key",
            "market_std",
            "name",
            "player",
            "bookmaker",
            "game_id",
            "point",
            "price",
        ]
        cols = [c for c in cols if c in merged.columns]
        sample = merged[merged["_merge"].eq("left_only")].head(200)
        if cols:
            sample = sample[cols].copy()
        else:
            sample = sample.copy()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out_path, index=False)
        logging.warning(f"[merge] wrote anti-join sample → {out_path}")

    # Enforce minimum hit-rate (no silent fallbacks)
    if hit_rate < 0.95:
        raise RuntimeError(f"[merge] hit rate {hit_rate:.2%} < 95% — fix keys in inputs; refusing name-based fallbacks")

    merged = merged.drop(columns=["_merge"], errors="ignore")

    # Ensure clean display name for UI
    if "player_display_name" in merged:
        merged["name"] = merged["player_display_name"].astype(str)
    elif "name" in merged:
        merged["name"] = merged["name"].astype(str)

    # Keep only player-like rows (avoid “Over/Under” snags if present)
    def _is_player_like(x):
        x = str(x or "")
        return (any(ch.isalpha() for ch in x) and " over" not in x and " under" not in x)

    merged = merged[merged["name"].map(_is_player_like)].copy()

    return merged


def load_props(path: str) -> pd.DataFrame:
    logging.info("[load] props: %s", path)
    df = pd.read_csv(path, low_memory=False)
    market_col = choose_first(df, ("market_std", "market", "market_name", "market_title"))
    if market_col is None:
        raise ValueError("props missing a market column")
    df["market_std"] = df[market_col].apply(canon_market)

    name_col = choose_first(df, ("name_std", "player", "player_name", "player_display_name", ))
    if name_col is None:
        raise ValueError("props missing a player name column")
    df["player_display_name"] = df[name_col].astype(str)
    df["name_std"] = df["player_display_name"].map(std_name)

    point_col = choose_first(df, ("point", "line", "points"))
    if point_col is not None:
        df["point"] = pd.to_numeric(df[point_col], errors="coerce")
    else:
        df["point"] = np.nan

    price_col = choose_first(df, ("price", "american_odds", "odds"))
    if price_col is None:
        raise ValueError("props missing a price column")
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")

    df["side"] = df.apply(determine_side, axis=1)

    if "player_key" not in df.columns:
        df["player_key"] = pd.NA

    return df


def load_params(path: str) -> pd.DataFrame:
    logging.info("[load] params: %s", path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["player_key", "market_std", "mu", "sigma", "lam"])

    market_col = choose_first(df, ("market_std", "market"))
    if market_col is None:
        raise ValueError("params missing a market column")
    df["market_std"] = df[market_col].apply(canon_market)

    if "player_key" in df.columns:
        df["player_key"] = df["player_key"].astype("string").str.strip().str.lower()
    else:
        name_col = choose_first(df, ("name_std", "player", "player_name", "player_display_name", "name"))
        if name_col is None:
            raise ValueError("params missing a player identifier column to derive player_key")
        # derive a slug-style key to align with props player_key normalization
        df["player_key"] = (
            df[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.replace(r"\s+", "-", regex=True)
        )

    for col, lower in (("mu", 0.0), ("lam", 0.0)):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lower)
        else:
            df[col] = np.nan
    if "sigma" in df.columns:
        df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce").clip(lower=1e-6)
    else:
        df["sigma"] = np.nan

    keep_cols = ["player_key", "market_std", "mu", "sigma", "lam"]
    return df[keep_cols].drop_duplicates().reset_index(drop=True)

def snapshot_props_with_model(df: pd.DataFrame, season: int, week: int, *, base_dir: Path = HIST_DIR, run_ts: str = RUN_TS) -> None:
    out_dir = base_dir / str(int(season)) / f"week{int(week):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_copy = df.copy()
    df_copy["snapshot_ts"] = run_ts

    tmp = out_dir / f"props_with_model_week{int(week):02d}_{run_ts}.csv.tmp"
    final = out_dir / f"props_with_model_week{int(week):02d}_{run_ts}.csv"

    df_copy.to_csv(tmp, index=False)
    os.replace(tmp, final)

    latest = out_dir / "latest.csv"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        os.symlink(final.name, latest)
    except Exception:  # noqa: BLE001
        pass

    logging.info("[snapshot] wrote %s", final)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge props with model params and compute edges.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--props_csv", required=True)
    parser.add_argument("--params_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--loglevel", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(levelname)s %(message)s")

    logging.info(f"[merge] loading props from {args.props_csv}")
    props = standardize_input(load_props(args.props_csv))
    print(props.columns)

    logging.info(f"[merge] loading params from {args.params_csv}")
    params = load_params(args.params_csv)

    # --- BEGIN: Paul-simple-join block -----------------------------------------
    import pandas as pd

    # Re-read params RAW so we control the columns we need (keeps your original 'params' intact if used elsewhere)
    params_raw = pd.read_csv(args.params_csv, low_memory=False)

    # 1) Ensure params has 'player'
    if "player" not in params_raw.columns and "player_display_name" in params_raw.columns:
        params_raw = params_raw.rename(columns={"player_display_name": "player"})

    # 2) Canonicalize markets the same way as your notebook
    CANON_MAP = {
        "player_rushing_yards":      "rush_yds",
        "player_reception_yds":      "recv_yds",
        "player_receiving_yds":      "recv_yds",
        "player_pass_yds":           "pass_yds",
        "player_receptions":         "receptions",
        "player_rush_attempts":      "rush_attempts",
        "player_passing_attempts":   "pass_attempts",
        "player_passing_completions":"pass_completions",
        "player_passing_tds":        "pass_tds",
        "player_interceptions":      "pass_interceptions",
        "player_anytime_td":         "anytime_td",
        "player_1st_td":             "first_td",
        "player_last_td":            "last_td",
        "player_rush_longest":       "rush_longest",
        "player_reception_longest":  "reception_longest",
    }
    if "market_std" in props.columns:
        props["market_std"] = props["market_std"].map(CANON_MAP).fillna(props["market_std"])
    if "market_std" in params_raw.columns:
        params_raw["market_std"] = params_raw["market_std"].map(CANON_MAP).fillna(params_raw["market_std"])

    # 3) Keep only what we need from params for modeling
    need_cols = [c for c in ("player","market_std","mu","sigma","lam") if c in params_raw.columns]
    params_sub = params_raw[need_cols].drop_duplicates()

    # 4) The exact simple LEFT JOIN you asked for
    merged = props.merge(params_sub, how="left", on=["player","market_std"])

    # (quick visibility)
    print("[merge] rows:", len(merged), "| mu% non-null:", merged.get("mu").notna().mean() if "mu" in merged else "n/a")
    # --- END: Paul-simple-join block -------------------------------------------


    # --- Canonicalize markets and do the simple left join (player × market_std) ---

    # side normalization (assume upstream populated; just normalize case)
    if "side" in merged:
        merged["side"] = merged["side"].astype(str).str.strip().str.lower()
    else:
        merged["side"] = ""

    # 1) modelable mask
    modelable = merged.apply(lambda r: _has_params(r) and _side_ok(r), axis=1)

    # 2) model_prob & model_line only on modelable rows
    merged["model_prob"] = float("nan")
    merged.loc[modelable, "model_prob"] = merged.loc[modelable].apply(_model_prob_row, axis=1)

    merged["model_line"] = float("nan")
    merged.loc[modelable, "model_line"] = merged.loc[modelable].apply(_model_line_row, axis=1)

    # 3) market implied probability (American odds)
    merged["mkt_prob"] = merged["price"].apply(price_to_prob_american)

    # 4) fair odds from model_prob (NaN-safe)
    merged["fair_odds"] = merged["model_prob"].apply(prob_to_american)

    # 5) edge (bps) = (model_prob - mkt_prob) * 10,000
    merged["edge_bps"] = merged.apply(lambda r: edge_bps(r["model_prob"], r["mkt_prob"]), axis=1)

    # QC: null edges by market

    # Define what counts as an "edge-ready" row for coverage summaries.
# Minimal: model_prob exists. (You can tighten this later.)
    if "has_edge" not in merged.columns:
        merged["has_edge"] = merged.get("model_prob").notna()
        # If you prefer stricter:
        # merged["has_edge"] = merged["model_prob"].notna() & merged["price"].notna()
        # or: merged["has_edge"] = merged["edge_bps"].notna()

    by_mkt = (
    merged.groupby("market_std", as_index=False)
          .agg(
              rows=("market_std", "size"),
              edge_rows=("has_edge", "sum"),
              edge_rate=("has_edge", "mean"),
          )
          .sort_values("edge_rate", ascending=False)
          )

    logging.info("[summary] by market (rows/edge_rows/edge_rate):\n%s", by_mkt.to_string(index=False))


    if all(col in merged.columns for col in ["bookmaker", "game_id", "market_std", "name_std", "point", "side"]):
        before = len(merged)
        merged = merged.drop_duplicates(subset=["bookmaker", "game_id", "market_std", "name_std", "point", "side"]).copy()
        removed = before - len(merged)
        if removed:
            logging.info("[dedupe] removed %d duplicate rows", removed)

    if "season" in merged.columns:
        merged["season"] = merged["season"].fillna(args.season)
    else:
        merged["season"] = args.season

    if "week" in merged.columns:
        merged["week"] = merged["week"].fillna(args.week)
    else:
        merged["week"] = args.week

    if merged is None or merged.empty:
        raise RuntimeError("[out] merged is empty; refusing to write")

    need = {
        "game_id",
        "bookmaker",
        "bookmaker_title",
        "market",
        "market_std",
        "name",
        "player_key",
        "price",
        "point",
        "side",
        "model_prob",
        "model_line",
        "mkt_prob",
        "fair_odds",
        "edge_bps",
        "commence_time",
    }
    missing = need - set(merged.columns)
    if missing:
        logging.warning(f"[out] missing expected columns (ok if some not applicable): {sorted(missing)}")

    merged.to_csv(args.out, index=False)
    logging.info(f"[out] wrote: {args.out}  rows={len(merged):,}")

    try:
        snapshot_props_with_model(merged, args.season, args.week)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[snapshot] skipped: %s", exc)


if __name__ == "__main__":
    main()

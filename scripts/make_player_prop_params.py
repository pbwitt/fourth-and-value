#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build player-level parameters for prop markets from weekly stats."""

import argparse
import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

MARKET_ALIAS: Dict[str, str] = {
    "player_rushing_yards": "rush_yds",
    "player_rush_yds": "rush_yds",
    "rushing_yards": "rush_yds",
    "rush_yards": "rush_yds",
    "player_receiving_yards": "recv_yds",
    "player_reception_yds": "recv_yds",
    "receiving_yards": "recv_yds",
    "reception_yds": "recv_yds",
    "player_passing_yards": "pass_yds",
    "player_pass_yds": "pass_yds",
    "passing_yards": "pass_yds",
    "pass_yards": "pass_yds",
    "player_receptions": "receptions",
    "receptions": "receptions",
    "player_rushing_attempts": "rush_attempts",
    "player_rush_attempts": "rush_attempts",
    "rushing_attempts": "rush_attempts",
    "rush_attempts": "rush_attempts",
    "carries": "rush_attempts",
    "player_passing_attempts": "pass_attempts",
    "player_pass_attempts": "pass_attempts",
    "passing_attempts": "pass_attempts",
    "attempts": "pass_attempts",
    "pass_attempts": "pass_attempts",
    "player_passing_completions": "pass_completions",
    "player_pass_completions": "pass_completions",
    "passing_completions": "pass_completions",
    "completions": "pass_completions",
    "player_passing_tds": "pass_tds",
    "player_pass_tds": "pass_tds",
    "passing_tds": "pass_tds",
    "pass_tds": "pass_tds",
    "player_pass_interceptions": "pass_interceptions",
    "player_interceptions": "pass_interceptions",
    "passing_interceptions": "pass_interceptions",
    "interceptions": "pass_interceptions",
    "player_anytime_td": "anytime_td",
    "anytime_td": "anytime_td",
}

NORMAL_MARKETS = {
    "rush_yds",
    "recv_yds",
    "pass_yds",
    "receptions",
    "rush_attempts",
    "pass_attempts",
    "pass_completions",
}
POISSON_MARKETS = {"pass_tds", "pass_interceptions", "anytime_td"}
SUPPORTED_MARKETS = NORMAL_MARKETS | POISSON_MARKETS

STATS_COLS: Dict[str, List[str]] = {
    "rush_yds": ["rushing_yards", "rush_yds"],
    "recv_yds": ["receiving_yards", "recv_yds"],
    "pass_yds": ["passing_yards", "pass_yds"],
    "receptions": ["receptions"],
    "rush_attempts": ["rushing_attempts", "carries", "rush_attempts"],
    "pass_attempts": ["passing_attempts", "attempts", "pass_attempts"],
    "pass_completions": ["passing_completions", "completions", "pass_completions"],
    "pass_tds": ["passing_tds", "pass_tds"],
    "pass_interceptions": ["interceptions", "passing_interceptions", "pass_interceptions"],
    "anytime_td": ["rushing_tds", "receiving_tds"],
}


def std_name(raw: Optional[str]) -> str:
    s = (str(raw) if raw is not None else "").strip().lower()
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


_std = std_name


def canon_market(raw: Optional[str]) -> str:
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    s = s.replace("player_", "").replace("__", "_")
    return MARKET_ALIAS.get(s, s)


def choose_first(df: pd.DataFrame, columns: Sequence[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None


def load_props(path: str) -> pd.DataFrame:
    logging.info("[load] props: %s", path)
    props = pd.read_csv(path, low_memory=False)
    market_col = choose_first(props, ("market_std", "market", "market_name", "market_title"))
    if market_col is None:
        raise ValueError("props missing market column")
    props["market_std"] = props[market_col].apply(canon_market)

    name_col = choose_first(props, ("name_std", "player", "player_name", "player_display_name", "name"))
    if name_col is None:
        raise ValueError("props missing player name column")
    props["player_display_name"] = props[name_col].astype(str)
    props["name_std"] = props["player_display_name"].map(_std)

    needed = props[props["market_std"].isin(SUPPORTED_MARKETS)].copy()
    keep_cols = [c for c in ["player_key", "player_display_name", "name_std", "market_std"] if c in needed.columns]
    return needed[keep_cols].drop_duplicates().reset_index(drop=True)


def load_stats(path: str, season: int, week_upper: int) -> pd.DataFrame:
    logging.info("[load] weekly stats: %s", path)
    if path.lower().endswith(".parquet"):
        stats = pd.read_parquet(path)
    else:
        stats = pd.read_csv(path, low_memory=False)

    for required in ("season", "week"):
        if required not in stats.columns:
            raise ValueError(f"weekly stats missing '{required}' column")

    stats = stats.loc[(stats["season"] == season) & (stats["week"] >= 1) & (stats["week"] < week_upper)].copy()
    if stats.empty:
        raise ValueError(f"weekly stats empty for season={season} weeks< {week_upper}")

    name_col = choose_first(stats, ("player_display_name", "player_name", "name"))
    if name_col is None:
        raise ValueError("weekly stats missing player name column")
    stats["player_display_name"] = stats[name_col].astype(str)
    stats["name_std"] = stats["player_display_name"].map(_std)

    logging.info("[stats] rows after filter: %d", len(stats))
    return stats


def pick_stats_series(stats: pd.DataFrame, market: str) -> pd.Series:
    if market == "anytime_td":
        missing = [c for c in ("rushing_tds", "receiving_tds") if c not in stats.columns]
        if missing:
            raise ValueError(f"weekly stats missing columns for anytime_td: {missing}")
        return stats["rushing_tds"].fillna(0) + stats["receiving_tds"].fillna(0)

    candidates = STATS_COLS.get(market, [])
    for col in candidates:
        if col in stats.columns:
            return stats[col]
    raise ValueError(f"weekly stats missing columns for market '{market}' (expected one of {candidates})")


def aggregate_stats(stats: pd.DataFrame, markets: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for market in markets:
        series = pick_stats_series(stats, market).astype(float)
        temp = stats[["name_std"]].copy()
        temp["market_std"] = market
        temp["value"] = series

        agg = (
            temp.groupby(["name_std", "market_std"], dropna=False)["value"]
            .agg(
                n_games="count",
                mean="mean",
                var=lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else float("nan"),
            )
            .reset_index()
        )
        frames.append(agg)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_params(props: pd.DataFrame, stats: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    needed_markets = sorted(props["market_std"].unique())
    if not needed_markets:
        raise ValueError("no supported markets present in props")

    stats_agg = aggregate_stats(stats, needed_markets)
    if stats_agg.empty:
        logging.warning("[params] stats aggregation empty; returning empty frame")
        return pd.DataFrame(columns=["name_std", "market_std", "mu", "sigma", "lam", "n_games", "season", "week", "built_at"])

    merged = props.merge(stats_agg, on=["name_std", "market_std"], how="left", suffixes=("", "_stats"))

    merged["mu"] = np.where(
        merged["market_std"].isin(list(NORMAL_MARKETS)), merged["mean"], np.nan
    )
    merged["sigma"] = np.where(
        merged["market_std"].isin(list(NORMAL_MARKETS)),
        np.sqrt(merged["var"]),
        np.nan,
    )
    merged.loc[merged["sigma"] <= 0, "sigma"] = np.nan

    merged["lam"] = np.where(
        merged["market_std"].isin(list(POISSON_MARKETS)), merged["mean"], np.nan
    )

    merged["season"] = season
    merged["week"] = week
    merged["built_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    cols = [c for c in [
        "player_key",
        "player_display_name",
        "name_std",
        "market_std",
        "n_games",
        "mu",
        "sigma",
        "lam",
        "season",
        "week",
        "built_at",
    ] if c in merged.columns]

    out = merged[cols].drop_duplicates().reset_index(drop=True)

    out["mu"] = pd.to_numeric(out["mu"], errors="coerce").clip(lower=0.0)
    if "sigma" in out.columns:
        out["sigma"] = pd.to_numeric(out["sigma"], errors="coerce").clip(lower=1e-6)
    out["lam"] = pd.to_numeric(out["lam"], errors="coerce").clip(lower=0.0)

    if "n_games" in out.columns:
        out["n_games"] = pd.to_numeric(out["n_games"], errors="coerce").round().astype("Int64")

    return out


def log_market_coverage(params: pd.DataFrame) -> None:
    if params.empty:
        logging.info("[coverage] no params to report")
        return

    rows = []
    for market, group in params.groupby("market_std"):
        if market in NORMAL_MARKETS:
            valid = (group["mu"] > 0) & (group["sigma"] >= 1e-6)
        else:
            valid = group["lam"] > 0
        share = float(valid.mean()) if len(group) else 0.0
        rows.append((market, share))
    rows.sort()
    report = "\n".join(f"  {market}: {share:.2%}" for market, share in rows)
    logging.info("[coverage] share with positive params by market:\n%s", report)


def write_zero_prior_report(props: pd.DataFrame, stats: pd.DataFrame, path: str) -> None:
    present = set(stats["name_std"].dropna().unique())
    missing = props.loc[~props["name_std"].isin(present), [c for c in props.columns if c in {"player_key", "player_display_name", "name_std"}]]
    missing = missing.drop_duplicates().reset_index(drop=True)
    missing.to_csv(path, index=False)
    logging.info("[zero-prior] wrote %s (rows=%d)", path, len(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-player prop parameters from weekly stats.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--props_csv", required=True)
    parser.add_argument("--stats_parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zero_prior_report")
    parser.add_argument("--loglevel", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if args.week <= 1:
        raise ValueError("week must be >= 2 to have prior games")

    props = load_props(args.props_csv)
    stats = load_stats(args.stats_parquet, args.season, args.week)

    if args.zero_prior_report:
        write_zero_prior_report(props, stats, args.zero_prior_report)

    params = build_params(props, stats, args.season, args.week)
    log_market_coverage(params)

    params.to_csv(args.out, index=False)
    logging.info("[out] wrote %s (rows=%d)", args.out, len(params))


if __name__ == "__main__":
    main()

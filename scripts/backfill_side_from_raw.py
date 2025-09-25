#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, re
from pathlib import Path

SIDE_RX = re.compile(r'(?i)\b(over|under|yes|no|o|u)\b')

def std_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def extract_side(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.extract(SIDE_RX, expand=False).str.lower()
    return out.map({'o':'over','u':'under'}).astype("string")

def best_book_col(df):
    for c in ("book","bookmaker_title","bookmaker"):
        if c in df.columns: return c
    raise KeyError("No book column found")

def best_player_col(df):
    for c in ("player","player_name","name_player","participant","selection_name"):
        if c in df.columns: return c
    raise KeyError("No player column found")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True, help="data/props/props_with_model_week3.csv")
    ap.add_argument("--raw_csv",    required=True, help="data/props/latest_all_props.csv")
    ap.add_argument("--out_csv",    required=True, help="data/props/props_with_model_week3_sidefix.csv")
    args = ap.parse_args()

    merged = pd.read_csv(args.merged_csv)
    raw    = pd.read_csv(args.raw_csv)

    # --- normalize keys ---
    # merged
    m_book = best_book_col(merged)
    merged["book"] = merged[m_book].astype(str)
    merged["line"] = pd.to_numeric(merged.get("line", merged.get("point")), errors="coerce")
    if "name_std" not in merged.columns:
        merged["name_std"] = merged.get("player", merged.get("name","")).map(std_name)
    # raw
    r_book = best_book_col(raw)
    raw["book"] = raw[r_book].astype(str)
    raw["line"] = pd.to_numeric(raw.get("line", raw.get("point")), errors="coerce")
    pcol = best_player_col(raw)
    raw["name_std"] = raw[pcol].map(std_name)

    # normalize market_std if present; otherwise fall back to 'market'
    km = "market_std" if "market_std" in merged.columns else "market"
    kr = "market_std" if "market_std" in raw.columns    else "market"
    if km not in merged.columns or kr not in raw.columns:
        raise KeyError("Need market_std or market in both files for a sane join.")

    # --- derive side from raw ---
    side_sources = [c for c in ["side","selection","outcome_name","outcome","label","bet","pick","description","header","name"] if c in raw.columns]
    if not side_sources:
        raise KeyError("Raw file missing any side-like columns (side/selection/outcome/name/...).")

    side_series = pd.Series(pd.NA, index=raw.index, dtype="string")
    for c in side_sources:
        side_series = side_series.fillna(extract_side(raw[c]))
    # default yes for TD markets
    td_mask = raw[kr].astype(str).isin(["anytime_td","first_td","last_td"])
    side_series = side_series.mask(side_series.isna() & td_mask, "yes")

    raw["side_raw"] = side_series

    # --- build side map keyed by (name_std, market_std, line, book) ---
    key = ["name_std", kr, "line", "book"]
    side_map = (raw.dropna(subset=["side_raw"])
                  .drop_duplicates(subset=key + ["side_raw"])
                  [key + ["side_raw"]])

    # --- join into merged where side is missing ---
    merged["side"] = (merged["side"].astype("string") if "side" in merged.columns
                      else pd.Series(pd.NA, index=merged.index, dtype="string"))

    mkey = ["name_std", km, "line", "book"]
    out = merged.merge(side_map.rename(columns={kr: km}), on=mkey, how="left")
    out["side"] = out["side"].fillna(out["side_raw"])
    out.drop(columns=["side_raw"], inplace=True)

    # extra default for TD markets in merged too
    td_mask_m = out[km].astype(str).isin(["anytime_td","first_td","last_td"])
    out["side"] = out["side"].mask(out["side"].isna() & td_mask_m, "yes").astype("string")

    # report + write
    miss_rate = float(out["side"].isna().mean())
    print(f"[backfill] missing side % AFTER: {miss_rate:.3%}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[write]", args.out_csv)

if __name__ == "__main__":
    main()

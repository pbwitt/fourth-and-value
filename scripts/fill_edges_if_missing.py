#!/usr/bin/env python3
import argparse, re
import numpy as np
import pandas as pd

# broadened name patterns
MODEL_PROB_CANDS = [
    "model_prob","model_probability","pred_prob","our_prob","p_model","p_pred","prob_model","model_pct"
]
MKT_PROB_CANDS = [
    "mkt_prob","market_prob","consensus_prob","implied_prob","book_prob","p_market","consensus_implied_prob","market_implied"
]
AMERICAN_CANDS = [
    "mkt_odds","market_odds","odds","price","american","american_odds","american_line",
    "price_over","price_under","odds_over","odds_under","price_yes","price_no"
]
DECIMAL_CANDS = ["decimal","decimal_odds","mkt_decimal","odds_decimal","price_decimal"]

def to_num(s): return pd.to_numeric(s, errors="coerce")

def normalize_prob_col(series: pd.Series) -> pd.Series:
    x = to_num(series)
    if x.notna().sum()==0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    # if looks like 0..100, scale
    v = x.dropna()
    if (v.ge(1.001).mean() > 0.8) and (v.le(100.0).mean() > 0.8):
        x = x / 100.0
    return x

def american_to_prob(o: pd.Series) -> pd.Series:
    o = to_num(o)
    p = pd.Series(np.nan, index=o.index, dtype=float)
    pos = o > 0
    neg = o < 0
    p.loc[pos] = 100.0 / (o.loc[pos] + 100.0)
    p.loc[neg] = (-o.loc[neg]) / ((-o.loc[neg]) + 100.0)
    return p

def decimal_to_prob(d: pd.Series) -> pd.Series:
    d = to_num(d)
    p = pd.Series(np.nan, index=d.index, dtype=float)
    p.loc[d > 1.0001] = 1.0 / d.loc[d > 1.0001]
    return p

def first_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def main():
    ap = argparse.ArgumentParser(description="Fill missing edge_bps using model_prob and market odds")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- locate / normalize model_prob ---
    model_col = first_col(df, MODEL_PROB_CANDS)
    if model_col is None:
        df["model_prob"] = np.nan
    else:
        df["model_prob"] = normalize_prob_col(df[model_col])

    # --- locate / compute mkt_prob ---
    mkt_col = first_col(df, MKT_PROB_CANDS)
    if mkt_col is not None:
        df["mkt_prob"] = normalize_prob_col(df[mkt_col])
    else:
        amer_col = first_col(df, AMERICAN_CANDS)
        dec_col  = first_col(df, DECIMAL_CANDS)
        if amer_col is not None:
            df["mkt_prob"] = american_to_prob(df[amer_col])
        elif dec_col is not None:
            df["mkt_prob"] = decimal_to_prob(df[dec_col])
        else:
            df["mkt_prob"] = np.nan

    # ensure edge_bps exists as numeric
    if "edge_bps" not in df.columns:
        df["edge_bps"] = np.nan
    df["edge_bps"] = to_num(df["edge_bps"])

    # fill where possible
    can_fill = df["edge_bps"].isna() & df["model_prob"].notna() & df["mkt_prob"].notna()
    df.loc[can_fill, "edge_bps"] = (df.loc[can_fill, "model_prob"] - df.loc[can_fill, "mkt_prob"]) * 10000.0

    # coverage print
    coverage = (df.assign(has_edge=df["edge_bps"].notna())
                  .groupby(df.get("market_std", df.get("market")))
                  ["has_edge"].agg(n="size", coverage="mean").reset_index()
                  .sort_values("coverage", ascending=False))
    print("Edge coverage by market (after fill):")
    with pd.option_context("display.max_rows", 100, "display.width", 140):
        print(coverage.to_string(index=False))

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

if __name__ == "__main__":
    main()

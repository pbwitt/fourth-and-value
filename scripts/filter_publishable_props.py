#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser(description="Keep rows with model_prob & mkt_prob & finite edge_bps")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, low_memory=False)
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    for col in ["model_prob","mkt_prob","edge_bps"]:
        if col not in df.columns:
            df[col] = np.nan

    keep = df["model_prob"].notna() & df["mkt_prob"].notna() & np.isfinite(pd.to_numeric(df["edge_bps"], errors="coerce"))
    out = df.loc[keep].copy()
    out.to_csv(args.out_csv, index=False)
    print(f"Filtered {len(df)-len(out)} rows; kept {len(out)} → {args.out_csv}")

if __name__ == "__main__":
    main()

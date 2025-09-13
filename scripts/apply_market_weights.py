#!/usr/bin/env python3
import argparse, json, pandas as pd
import numpy as np

ALIAS = {
  "player_rush_yds":"rush_yds","player_receiving_yds":"recv_yds",
  "player_receptions":"receptions","player_passing_yds":"pass_yds",
  "player_passing_tds":"pass_tds","player_interceptions":"pass_interceptions",
  "player_anytime_td":"anytime_td",
}
def norm_market(m):
    m = str(m).strip().lower().replace(" ","_")
    return ALIAS.get(m, m)

def main():
    ap = argparse.ArgumentParser(description="Apply market weights to edge_bps")
    ap.add_argument("--merged_csv", required=True)   # data/props/props_with_model_week1.csv
    ap.add_argument("--weights_json", required=True) # data/eval/market_weights_week1.json
    ap.add_argument("--out", required=True)          # data/props/props_with_model_week1_weighted.csv
    ap.add_argument("--floor", type=float, default=0.6, help="min multiplier")
    ap.add_argument("--span", type=float, default=0.8, help="range added by weight (0..1)")
    args = ap.parse_args()

    df = pd.read_csv(args.merged_csv, low_memory=False)
    with open(args.weights_json) as f:
        w = json.load(f)["weights"]
    def w_of(m):
        m = norm_market(m)
        return float(w.get(m, {}).get("market_weight", 0.0))
    df["market_std"] = df.get("market_std", df.get("market","")).map(norm_market)

    # multiplier in [floor, floor+span] → default [0.6, 1.4]
    df["market_weight"] = df["market_std"].map(w_of).fillna(0.0)
    df["edge_mult"] = args.floor + args.span * df["market_weight"].clip(0,1)

    # keep original for display; new column for ranking
    if "edge_bps" not in df.columns:
        df["edge_bps"] = np.nan
    df["edge_bps_adj"] = pd.to_numeric(df["edge_bps"], errors="coerce") * df["edge_mult"]

    # (optional) replace edge_bps for downstream sorters that don't know about _adj
    df["edge_bps_orig"] = df["edge_bps"]
    df["edge_bps"] = df["edge_bps_adj"]

    df.to_csv(args.out, index=False)
    print(f"Wrote weighted CSV → {args.out} (multiplier=floor+span*weight)")
    print(df[["market_std","edge_bps_orig","edge_bps_adj","edge_mult"]].head(8).to_string(index=False))

if __name__ == "__main__":
    main()

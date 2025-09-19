#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

# --- helpers ---
def american_to_prob(o):
    """Convert American odds to implied probability."""
    if pd.isna(o):
        return None
    try:
        o = float(o)
    except Exception:
        return None
    if o > 0:
        return 100 / (o + 100)
    else:
        return -o / (-o + 100)

def prob_to_american(p):
    """Convert probability back to American odds."""
    if pd.isna(p) or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return -round((p / (1 - p)) * 100, 0)
    else:
        return round(((1 - p) / p) * 100, 0)

def extract_side(val):
    """Derive side (Over/Under/Yes/No) from the name column if possible."""
    if not isinstance(val, str):
        return None
    low = val.strip().lower()
    if low in ["over", "under", "yes", "no"]:
        return val.strip().title()  # Normalize casing
    return None

# --- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", required=True,
                        help="Input: latest_all_props.csv from fetch step")
    parser.add_argument("--out_csv", required=True,
                        help="Output: consensus_weekX.csv")
    parser.add_argument("--week", type=int, required=True,
                        help="NFL week number")
    args = parser.parse_args()

    # --- load data ---
    df = pd.read_csv(args.merged_csv)

    # strict schema check
    required_cols = ["player", "market_std", "name", "price", "point", "bookmaker"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns {missing} in {args.merged_csv}. "
            f"Got: {list(df.columns)}"
        )

    # derive side from `name`
    df["side"] = df["name"].apply(extract_side)

    # implied probability from American odds
    df["mkt_prob"] = df["price"].apply(american_to_prob)

    # --- group across books ---
    grouped = (
        df.groupby(["player", "market_std", "side"], dropna=False)
        .agg(
            book_count=("bookmaker", "nunique"),
            consensus_prob=("mkt_prob", "mean"),
            consensus_line=("point", "mean"),
        )
        .reset_index()
    )

    # convert back to consensus odds
    grouped["consensus_odds"] = grouped["consensus_prob"].apply(prob_to_american)

    # attach week
    grouped["week"] = args.week

    # --- write out ---
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)

    print(f"[consensus] wrote {out_path} with {len(grouped)} rows")

if __name__ == "__main__":
    main()

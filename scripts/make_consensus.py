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
    # implied probability from American odds already computed as df["mkt_prob"]

# For "best book": choose the book with the most favorable price for the bettor.
# We use the *most positive* edge vs consensus prob. To compute that, we need consensus first,
# then scan books inside each group.

    base = (
        df.groupby(["player", "market_std", "side"], dropna=False)
        .agg(
            book_count=("bookmaker", "nunique"),
            consensus_prob=("mkt_prob", "mean"),
            consensus_line=("point", "mean"),
            # helpful for display
            game=("game", "first"),
            commence_time=("commence_time", "first"),
        )
        .reset_index()
    )

# join back per-book rows so we can pick the best book against consensus
    cols_keep = [
        "player", "market_std", "side",
        "bookmaker", "bookmaker_title", "price", "mkt_prob", "point"
    ]
    per_book = df[cols_keep].copy()

    merged = per_book.merge(
        base[["player", "market_std", "side", "consensus_prob"]],
        on=["player", "market_std", "side"],
        how="left",
    )

    # compute edge vs consensus at the per-book level
    merged["edge_bps_tmp"] = (merged["mkt_prob"] - merged["consensus_prob"]) * 10_000

    # pick the single best row per (player, market_std, side)
    idx = (
        merged.groupby(["player", "market_std", "side"], dropna=False)["edge_bps_tmp"]
        .idxmax()
    )
    best = merged.loc[idx, [
        "player", "market_std", "side",
        "bookmaker", "bookmaker_title", "price", "mkt_prob", "edge_bps_tmp"
    ]].rename(columns={
        "bookmaker": "best_bookmaker",
        "bookmaker_title": "best_bookmaker_title",
        "price": "best_price",
        "mkt_prob": "best_prob",
        "edge_bps_tmp": "edge_bps",
    })

    # final table
    grouped = base.merge(best, on=["player", "market_std", "side"], how="left")

    # back-convert consensus prob → American odds
    grouped["consensus_odds"] = grouped["consensus_prob"].apply(prob_to_american)

    # attach week
    grouped["week"] = args.week

    # tidy column order (nice for downstream rendering)
    out_cols = [
        "player", "market_std", "side",
        "game", "commence_time",
        "book_count",
        "consensus_line", "consensus_prob", "consensus_odds",
        "best_bookmaker_title", "best_bookmaker", "best_price", "best_prob", "edge_bps",
        "week",
    ]
    # keep only those that exist (some markets may lack lines)
    grouped = grouped[[c for c in out_cols if c in grouped.columns]]


    # --- write out ---
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)

    print(f"[consensus] wrote {out_path} with {len(grouped)} rows")

if __name__ == "__main__":
    main()

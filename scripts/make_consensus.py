# scripts/make_consensus.py
import argparse
import pandas as pd

def price_to_prob_american(odds: float) -> float:
    if pd.isna(odds):
        return float("nan")
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_csv",  required=True, help="data/props/props_with_model_week5.csv")
    ap.add_argument("--out", dest="out_csv", required=True, help="data/props/consensus_week5.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, low_memory=False)

    # Create 'side' from 'name' column if missing (yes/over/under/no)
    if "side" not in df.columns and "name" in df.columns:
        df["side"] = df["name"]

    # --- light guards ---
    need = {"market_std","side","point","price","week"}
    if "bookmaker_title" not in df.columns and "bookmaker" in df.columns:
        df["bookmaker_title"] = df["bookmaker"]

    if not need.issubset(df.columns):
        missing = sorted(need - set(df.columns))
        raise SystemExit(f"Missing required columns: {missing}")

    if "name_std" not in df.columns and "player" not in df.columns:
        raise SystemExit("Need one of: name_std or player")

    # Clean join keys (whitespace only - no casing changes)
    if "name_std" not in df.columns:
        if "player" in df.columns:
            df["name_std"] = df["player"]
        else:
            raise SystemExit("Need name_std column")

    df["name_std"] = df["name_std"].astype(str).str.strip()
    df["market_std"] = df["market_std"].astype(str).str.strip()
    df["side"] = df["side"].astype(str).str.strip()

    # Per-row implied probability (reuse mkt_prob if present)
    df["prob_raw"] = df["mkt_prob"] if "mkt_prob" in df.columns else df["price"].apply(price_to_prob_american)

    # --- MANDATORY DE-VIG (by book × player × market) ---
    # If both Over & Under exist in that group, scale so Over+Under ≈ 1.
    def _devig_group(g: pd.DataFrame) -> pd.DataFrame:
        sides = set(g["side"].dropna().unique().tolist())
        # Match raw side values (no case normalization)
        if {"over","under"}.issubset(sides):
            tot = g.loc[g["side"].isin(["over","under"]), "prob_raw"].sum()
            if tot and tot > 0:
                g["prob_devig"] = g["prob_raw"] / tot
            else:
                g["prob_devig"] = g["prob_raw"]
        else:
            # e.g., Yes-only markets → leave raw
            g["prob_devig"] = g["prob_raw"]
        return g

    df = (df.groupby(["bookmaker_title","name_std","market_std"], group_keys=False)
            .apply(_devig_group))

    # === Your simple pattern: groupby → agg → merge ===
    keys = ["name_std","market_std","side"]

    # 1) consensus line (median)
    line_count = (
        df.groupby(keys, as_index=False)
          .agg({"point": "median"})
          .rename(columns={"point": "consensus_line"})
    )

    # 2) book count (distinct books)
    book_count = (
        df.groupby(keys, as_index=False)
          .agg({"bookmaker_title": pd.Series.nunique})
          .rename(columns={"bookmaker_title": "book_count"})
    )

    # 3) consensus probability (median of de-vigged probs)
    prob_med = (
        df.groupby(keys, as_index=False)
          .agg({"prob_devig": "median"})
          .rename(columns={"prob_devig": "consensus_prob"})
    )

    # Merge your two frames, then add consensus_prob (keeps the "very easy" style)
    consensus = (line_count
                 .merge(book_count, on=keys, how="left")
                 .merge(prob_med,  on=keys, how="left"))

    # Carry week (mode)
    wk = df["week"].dropna().astype(int)
    consensus["week"] = int(wk.mode().iat[0]) if not wk.mode().empty else None

    # Ensure exact output columns
    consensus = consensus[["name_std", "market_std", "side", "consensus_line", "consensus_prob", "book_count", "week"]]

    consensus.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(consensus):,} rows → {args.out_csv}")

if __name__ == "__main__":
    main()

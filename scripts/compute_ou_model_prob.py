#!/usr/bin/env python3
import argparse, math
import pandas as pd
import numpy as np

# Markets we treat as O/U with Normal:
NORMAL_OU = {
    "rush_yds","recv_yds","pass_yds",
    "receptions","pass_attempts","pass_completions","rush_attempts"
}

# Markets we treat as O/U with Poisson:
POISSON_OU = {"pass_tds","pass_interceptions"}

def norm_cdf(x):
    # Φ(x)
    return 0.5*(1.0 + math.erf(x / math.sqrt(2.0)))

def poisson_cdf(k, lam):
    # P(X <= k) for Poisson(lam), k >= 0 integer
    if lam < 0: return np.nan
    if k < 0: return 0.0
    # Stable iterative sum
    term = math.exp(-lam)
    s = term
    for i in range(1, int(k)+1):
        term *= lam/i
        s += term
    return min(1.0, max(0.0, s))

def poisson_median(lam):
    # Good closed-form approximation
    if lam <= 0: return 0.0
    return math.floor(lam + 1.0/3.0 - 0.02/lam)

def model_prob_row(market_std, side, point, mu, sigma, lam):
    from market_math import outcome_probabilities
    p, _ = outcome_probabilities(market_std, side, point, mu, sigma, lam)
    line = poisson_median(lam) if market_std in {"pass_tds", "pass_interceptions", "interceptions"} and pd.notna(lam) else mu
    return p, line

def prob_to_american(p):
    if not (0 < p < 1): return ""
    return int(round(-100*p/(1-p))) if p>=0.5 else int(round(100*(1-p)/p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input merged CSV (props_with_model_weekX.csv)")
    ap.add_argument("--out", required=True, help="output CSV (can be same as --inp for in-place)")
    ap.add_argument("--force", action="store_true", help="recompute even if model_prob already present")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, low_memory=False)
    cols = {c:c for c in df.columns}
    # normalize required column names
    for need in ["market_std","name","point","mu","sigma","lam","model_prob"]:
        if need not in df.columns:
            # be permissive (e.g., Point/LINE/Model_Prob)
            for c in df.columns:
                if c.lower()==need:
                    cols[c] = need
                    break
    df = df.rename(columns={k:v for k,v in cols.items() if k!=v})

    # ensure the columns exist
    for c in ["market_std","name","point"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    if "model_prob" not in df.columns:
        df["model_prob"] = np.nan
    if "mu" not in df.columns: df["mu"] = np.nan
    if "sigma" not in df.columns: df["sigma"] = np.nan
    if "lam" not in df.columns: df["lam"] = np.nan

    # compute only where missing or forced
    mask = df["model_prob"].isna() | args.force
    sub = df.loc[mask, ["market_std","name","point","mu","sigma","lam"]].copy()

    out_p = []
    out_line = []
    out_push = []
    from market_math import outcome_probabilities, implied_probability, expected_profit
    for m, side, pt, mu, sig, lam in sub.itertuples(index=False, name=None):
        p, model_line = model_prob_row(
            str(m).strip().lower(), str(side).strip(), float(pt),
            float(mu) if pd.notna(mu) else np.nan,
            float(sig) if pd.notna(sig) else np.nan,
            float(lam) if pd.notna(lam) else np.nan
        )
        out_push.append(outcome_probabilities(str(m).strip().lower(), str(side).strip(), float(pt), mu, sig, lam)[1])
        out_p.append(p)
        out_line.append(model_line)

    df.loc[mask, "model_prob"] = out_p
    df.loc[mask, "model_prob_raw"] = out_p
    df.loc[mask, "model_status"] = "Uncalibrated"
    if "push_prob" not in df.columns: df["push_prob"] = np.nan
    df.loc[mask, "push_prob"] = out_push
    if "price" in df.columns:
        df["mkt_prob"] = df["price"].map(implied_probability)
        df["edge_bps"] = 10000 * (df["model_prob"] - df["mkt_prob"])
        df["ev_per_100"] = df.apply(lambda r: expected_profit(r["model_prob"], r["price"], r["push_prob"]), axis=1)
    if "model_line" not in df.columns: df["model_line"] = np.nan
    df.loc[mask, "model_line"] = out_line

    # optional: model_american from model_prob
    df["model_american"] = df["model_prob"].apply(prob_to_american)

    # tiny sanity
    filled = int(np.isfinite(df.loc[mask, "model_prob"]).sum())
    total  = int(mask.sum())
    print(f"Computed model_prob for {filled}/{total} rows.")

    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

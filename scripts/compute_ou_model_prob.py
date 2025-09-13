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
    # Returns (p, model_line)
    if market_std in NORMAL_OU and pd.notna(mu) and pd.notna(sigma) and sigma > 0:
        z = (point - mu)/sigma
        p_under = norm_cdf(z)
        p_over  = 1.0 - p_under
        p = p_over if side.lower()=="over" else p_under
        model_line = float(mu)  # median ~ mean for Normal; good enough for our "model line"
        return p, model_line

    if market_std in POISSON_OU and pd.notna(lam) and lam >= 0:
        # integer counts; lines are typically x.5
        # Over x.5 => P(X >= ceil(x.5)) ; Under x.5 => P(X <= floor(x.5))
        k_over = int(math.ceil(point - 1e-12))   # e.g., 0.5 -> 1
        k_under = int(math.floor(point + 1e-12)) # e.g., 0.5 -> 0
        if side.lower() == "over":
            p = 1.0 - poisson_cdf(k_over - 1, lam)
        else:
            p = poisson_cdf(k_under, lam)
        model_line = float(poisson_median(lam))
        return p, model_line

    # no model
    return np.nan, np.nan

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
    for m, side, pt, mu, sig, lam in sub.itertuples(index=False, name=None):
        p, model_line = model_prob_row(
            str(m).strip().lower(), str(side).strip(), float(pt),
            float(mu) if pd.notna(mu) else np.nan,
            float(sig) if pd.notna(sig) else np.nan,
            float(lam) if pd.notna(lam) else np.nan
        )
        out_p.append(p)
        out_line.append(model_line)

    df.loc[mask, "model_prob"] = out_p
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

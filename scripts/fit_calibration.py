#!/usr/bin/env python3
"""
fit_calibration.py

Fits a probability calibration curve for each NFL prop market from real
historical outcomes (see backtest_calibration.py for the diagnostic this
is based on), and writes it to models/nfl_prop_calibration.json as plain
monotonic (x, y) breakpoints - no pickled objects, no sklearn dependency
at apply time, just numpy.interp.

Why isotonic regression: it fits a monotonic step function mapping raw
model_prob -> empirical win rate, with no assumption about the shape of
the miscalibration (unlike Platt/logistic scaling). That matters here
because the backtest showed the raw model collapsing toward ~50% at
every confidence level rather than a clean linear over/under-confidence
pattern.

Markets with fewer than MIN_MARKET_N graded picks fall back to a pooled
curve fit across all markets, since a market-specific isotonic fit on a
small sample is itself unreliable.

Run this again whenever data/preds_historical/ has accumulated more
completed, real-outcome weeks (recommended: monthly during the season),
and commit the updated JSON - it's small and it's the only artifact that
needs to survive into the GitHub Actions checkout (data/ is gitignored).

Usage:
  python scripts/fit_calibration.py --season 2025
"""
import argparse
import json
import sys
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.dirname(__file__))
from backtest_calibration import latest_snapshot_per_week, load_outcomes, grade, MARKET_TO_STAT

MIN_MARKET_N = 500
OUT_PATH = "models/nfl_prop_calibration.json"


def fit_curve(x: np.ndarray, y: np.ndarray) -> dict:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(x, y)
    return {
        "x": iso.X_thresholds_.tolist(),
        "y": iso.y_thresholds_.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    preds = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    preds = preds[preds["season"] == args.season]
    outcomes = load_outcomes(args.season)
    graded = grade(preds, outcomes).dropna(subset=["model_prob", "outcome"])

    print(f"[fit] {len(graded):,} graded picks across weeks {sorted(graded['week'].unique())}")

    pooled_curve = fit_curve(graded["model_prob"].values, graded["outcome"].values)

    result = {
        "_meta": {
            "fitted_on_season": args.season,
            "fitted_on_weeks": sorted(int(w) for w in graded["week"].unique()),
            "n_graded_picks": int(len(graded)),
            "note": "Refit periodically as more real in-season data accumulates; see script docstring.",
        },
        "_pooled_fallback": pooled_curve,
        # Only these markets were actually backtested (see MARKET_TO_STAT).
        # Poisson/yes-no markets (anytime_td, first_td, ...) and longest-play
        # props were never validated and must NOT receive the pooled curve -
        # applying an O/U-fit curve to a differently-shaped distribution
        # made things worse in testing (anytime_td edges nearly tripled).
        "_eligible_markets": list(MARKET_TO_STAT.keys()),
        "markets": {},
    }

    for market_std in MARKET_TO_STAT:
        sub = graded[graded["market_std"] == market_std]
        if len(sub) < MIN_MARKET_N:
            print(f"  {market_std:20s} n={len(sub):5d}  -> too small, using pooled fallback")
            continue
        curve = fit_curve(sub["model_prob"].values, sub["outcome"].values)
        result["markets"][market_std] = curve
        print(f"  {market_std:20s} n={len(sub):5d}  -> fitted own curve")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[fit] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

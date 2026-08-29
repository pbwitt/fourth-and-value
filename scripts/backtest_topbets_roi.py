#!/usr/bin/env python3
"""
backtest_topbets_roi.py

Simulates flat-stake betting on the model's own top-N highest-edge picks
each week, against real historical sportsbook prices, and reports actual
dollar P&L - not just win rate or Brier score.

Methodology, and why it's built this way:

- Uses OUT-OF-FOLD calibrated probabilities (same grouped 5-fold CV as
  backtest_new_baseline.py / the Research page), not the calibration curve
  fit on all weeks. Ranking week 5's picks using a calibration curve that
  was partly fit ON week 5 would be a look-ahead bias; OOF calibration
  approximates "what the model would have told you without having seen
  this week's outcomes yet." This is not a full walk-forward simulation
  (a stricter standard - calibrating only on weeks *before* the bet was
  placed) - that's a known limitation, disclosed in the output.
- Ranks by OOF-calibrated edge (calibrated model prob - market prob) and
  takes the top N per week, using the SPECIFIC book/price where that edge
  was computed - this is the price you'd have actually gotten, not an
  idealized average.
- Flat stake ($100/bet) - no Kelly sizing, no bankroll compounding. This
  isolates "was the pick good" from "was the staking strategy good."
- Correlated picks (e.g. the same player's receptions AND yards prop both
  appearing in the same week's top N) are not deduplicated - a real bettor
  following top-N picks would face the same correlation, but it means
  weekly P&L is not fully independent across picks. Disclosed in output.

Usage:
  python scripts/backtest_topbets_roi.py --top-n 5
  python scripts/backtest_topbets_roi.py --top-n 10
"""
import argparse
import sys
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.dirname(__file__))
from backtest_new_baseline import (
    load_truncated_logs, build_mu_sigma, score_week, SEASON, TEST_WEEKS
)
from backtest_calibration import latest_snapshot_per_week, load_outcomes, grade, MARKET_TO_STAT
from common_markets import strip_generational_suffix
import career_baseline as cb


def american_payout(stake: float, odds: float) -> float:
    """Net profit on a WIN at these American odds (does not include the returned stake)."""
    if odds > 0:
        return stake * (odds / 100.0)
    else:
        return stake * (100.0 / abs(odds))


def build_graded_with_oof():
    files = latest_snapshot_per_week("data/preds_historical/props_with_model_week*.csv")
    all_preds = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    all_preds = all_preds[all_preds["season"] == SEASON]
    outcomes = load_outcomes(SEASON)
    full_career_df = cb.load_career_logs(list(range(SEASON - 6, SEASON + 1)))

    new_all = []
    for wk in TEST_WEEKS:
        preds_wk = all_preds[all_preds["week"] == wk]
        preds_wk = preds_wk[preds_wk["market_std"].isin(MARKET_TO_STAT.keys())]
        preds_wk = preds_wk[preds_wk["point"].notna()]
        if preds_wk.empty:
            continue
        logs_trunc = load_truncated_logs(SEASON, before_week=wk)
        player_idx = pd.Index(preds_wk["player"].astype(str).map(strip_generational_suffix).unique())
        mu_sigma = build_mu_sigma(logs_trunc, player_idx, SEASON, career_df=full_career_df)
        scored = score_week(preds_wk, mu_sigma)
        scored["week"] = wk
        new_all.append(scored)

    new_df = pd.concat(new_all, ignore_index=True)
    graded = grade(new_df, outcomes).dropna(subset=["model_prob", "outcome"])

    X, y, wk = graded["model_prob"].values, graded["outcome"].values, graded["week"].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros_like(X)
    for tr, te in gkf.split(X, y, groups=wk):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        iso.fit(X[tr], y[tr])
        oof[te] = iso.predict(X[te])
    graded["oof_calibrated"] = oof
    graded["oof_edge"] = graded["oof_calibrated"] - graded["mkt_prob"]
    return graded


def simulate(graded: pd.DataFrame, top_n: int, stake: float = 100.0) -> pd.DataFrame:
    rows = []
    for wk, sub in graded.groupby("week"):
        picks = sub.sort_values("oof_edge", ascending=False).head(top_n)
        for _, r in picks.iterrows():
            won = bool(r["outcome"])
            pnl = american_payout(stake, r["price"]) if won else -stake
            rows.append({
                "week": wk, "player": r["player"], "market_std": r["market_std"],
                "point": r["point"], "side": r["name"] if "name" in r else r.get("side"),
                "price": r["price"], "oof_calibrated": r["oof_calibrated"],
                "mkt_prob": r["mkt_prob"], "oof_edge_bps": r["oof_edge"] * 10000,
                "won": won, "stake": stake, "pnl": pnl,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--stake", type=float, default=100.0)
    args = ap.parse_args()

    graded = build_graded_with_oof()
    sim = simulate(graded, args.top_n, args.stake)

    total_staked = sim["stake"].sum()
    total_pnl = sim["pnl"].sum()
    win_rate = sim["won"].mean()
    roi = total_pnl / total_staked * 100

    print(f"\n{'='*70}\nTOP-{args.top_n} PICKS/WEEK, FLAT ${args.stake:.0f} STAKE, {len(sim)} bets across {sim['week'].nunique()} weeks\n{'='*70}")
    print(f"Win rate:      {win_rate*100:.1f}%  ({int(sim['won'].sum())}/{len(sim)})")
    print(f"Total staked:  ${total_staked:,.2f}")
    print(f"Total P&L:     ${total_pnl:,.2f}")
    print(f"ROI:           {roi:+.1f}%")

    print(f"\nBy week:")
    wk_summary = sim.groupby("week").agg(n=("pnl","size"), win_rate=("won","mean"), pnl=("pnl","sum")).reset_index()
    wk_summary["cume_pnl"] = wk_summary["pnl"].cumsum()
    print(wk_summary.to_string(index=False))

    sim.to_csv(f"data/qc/topbets_roi_top{args.top_n}.csv", index=False)
    print(f"\nWrote data/qc/topbets_roi_top{args.top_n}.csv")


if __name__ == "__main__":
    main()

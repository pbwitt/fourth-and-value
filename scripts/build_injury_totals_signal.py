#!/usr/bin/env python3
"""Build an auditable injury-vs-total-market reaction screen.

This is deliberately a research signal until calibrated on captured line
movement. It never calls a large market move an edge when an opening snapshot
is missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# Initial priors are intentionally modest and capped. They are placeholders to
# be refit from historical injury announcements and team scoring changes.
POSITION_POINT_IMPACT = {
    "QB": 5.0, "RB": 1.0, "FB": 0.7, "WR": 0.7, "TE": 0.6,
    "T": 0.9, "G": 0.9, "C": 0.9,
}
DEFAULT_POINT_IMPACT = 0.25


def build_signal(injuries: pd.DataFrame, predictions: pd.DataFrame,
                 consensus: pd.DataFrame, movement: pd.DataFrame,
                 lines: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for _, game in predictions.iterrows():
        home, away = game["home_team"], game["away_team"]
        gi = injuries[injuries["team"].isin([home, away])].copy()
        team_impacts = {home: 0.0, away: 0.0}
        counts = {home: 0, away: 0}
        for _, injury in gi.iterrows():
            weight = POSITION_POINT_IMPACT.get(str(injury["position"]).upper(), DEFAULT_POINT_IMPACT)
            impact = -weight * (1.0 - float(injury["availability"]))
            # Keep the team-level prior bounded: an injury screen should not
            # manufacture a double-digit total move from a long report.
            team_impacts[injury["team"]] = max(team_impacts[injury["team"]] + impact, -6.0)
            counts[injury["team"]] += 1
        model_impact = team_impacts[home] + team_impacts[away]
        gname = game.get("game", f"{away} @ {home}")
        c = consensus[(consensus["game"] == gname) & (consensus["market"] == "total")]
        market_line = float(c.iloc[0]["consensus_line"]) if not c.empty else np.nan
        m = movement[movement["game"] == gname] if not movement.empty else pd.DataFrame()
        market_move = float(pd.to_numeric(m.get("total_move"), errors="coerce").median()) if not m.empty else np.nan
        residual = market_move - model_impact if pd.notna(market_move) else np.nan
        gl = lines[lines["game"] == gname] if lines is not None and not lines.empty else pd.DataFrame()
        if not gl.empty:
            over = gl.loc[pd.to_numeric(gl["total_over_price"], errors="coerce").idxmax()]
            under = gl.loc[pd.to_numeric(gl["total_under_price"], errors="coerce").idxmax()]
            best_over = f"{over['book']} {over['total_over_line']:.1f} ({over['total_over_price']:+.0f})"
            best_under = f"{under['book']} {under['total_over_line']:.1f} ({under['total_under_price']:+.0f})"
        else:
            best_over = best_under = ""
        movers = m.loc[pd.to_numeric(m["total_move"], errors="coerce").idxmin(), "book"] if not m.empty else ""
        if pd.isna(market_move):
            status = "insufficient_market_history"
        elif residual <= -1.5 and model_impact < 0:
            status = "possible_downward_overreaction"
        elif residual >= 1.5 and model_impact < 0:
            status = "possible_underreaction"
        else:
            status = "no_clear_reaction_edge"
        rows.append({
            "game": gname, "home_team": home, "away_team": away,
            "baseline_total_model": game.get("total_pred"),
            "injury_adjusted_total_model": game.get("total_pred", np.nan) + model_impact,
            "model_injury_impact_points": model_impact,
            "market_consensus_total": market_line, "market_total_move": market_move,
            "reaction_residual": residual, "home_injury_rows": counts[home],
            "away_injury_rows": counts[away], "largest_downward_move_book": movers,
            "best_over": best_over, "best_under": best_under, "signal_status": status,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injuries", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--consensus", required=True)
    ap.add_argument("--movement", required=True)
    ap.add_argument("--lines", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    lines = pd.read_csv(args.lines) if args.lines else None
    out = build_signal(pd.read_csv(args.injuries), pd.read_csv(args.predictions),
                       pd.read_csv(args.consensus), pd.read_csv(args.movement), lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[injury-totals] wrote {len(out):,} game screens to {args.output}")


if __name__ == "__main__":
    main()

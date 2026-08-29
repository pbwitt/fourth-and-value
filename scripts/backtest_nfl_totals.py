#!/usr/bin/env python3
"""
backtest_nfl_totals.py

Walk-forward backtest of the site's NFL game-totals model (Ridge on trailing
L3/L5 EPA features, same approach as scripts/nfl_train_totals_model.py)
against real closing lines, real over/under odds, and real final scores for
the FULL 2025 regular season (272 games).

Why this is a different shape of test than the player-prop backtests:
  - There is no local historical archive of multi-book/multi-timestamp odds
    for NFL totals/spreads - data/nfl/lines/totals_spreads.csv is a single
    current snapshot (one week), not season-long, and it isn't tracked in
    git (data/ is gitignored) so it was never accumulated over the season
    the way data/preds_historical/ was for props. So "fade the outlier
    book" and "early vs late line movement" literally cannot be tested for
    totals right now - the data doesn't exist anywhere on this machine.
  - What DOES exist, from nflverse (github.com/nflverse/nflverse-data), is
    the real closing total line + real over/under odds + real final score
    for every 2025 game. That's a single line per game (closing, not
    multi-book), but it's real, verifiable, and covers the whole season -
    so the test we CAN run is: does the model's own total prediction beat
    the closing line, for real money, across all 272 games.

No-leakage walk-forward procedure:
  - Team features (data/nfl/processed/team_features_fullseason.csv) use
    only trailing L3/L5 averages via .shift(1).rolling(window).mean() -
    the model never sees a team's own current-game stats.
  - For each week W (starting once every team has >=3 games of history),
    retrain Ridge on ONLY team-games from weeks < W, predict week W's
    games, and grade against that week's real closing line/odds/score.
    This is stricter than the production script (which trains once on all
    data) - it's what a bettor actually had access to in real time.

Usage:
  python scripts/backtest_nfl_totals.py
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.append(os.path.dirname(__file__))
from backtest_topbets_roi import american_payout

SEASON = 2025
MIN_TRAIN_WEEK = 5   # first week with enough trailing history across the league
EDGE_THRESHOLD = 1.0  # points of model-vs-line disagreement required to bet


def load_features():
    df = pd.read_csv("data/nfl/processed/team_features_fullseason.csv")
    feature_cols = [c for c in df.columns if c.endswith("_L3") or c.endswith("_L5")]
    df["is_home"] = (df["home_away"] == "home").astype(int)
    return df, feature_cols + ["is_home"]


def load_games():
    games = pd.read_csv("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
    games = games[(games["season"] == SEASON) & (games["game_type"] == "REG")]
    games = games.dropna(subset=["home_score", "away_score", "total_line", "over_odds", "under_odds"])
    games["actual_total"] = games["home_score"] + games["away_score"]
    return games


def predict_week(train_df, test_df, feature_cols):
    train_df = train_df.dropna(subset=feature_cols + ["points_scored"])
    if len(train_df) < 20:
        return None
    model = Ridge(alpha=1.0)
    model.fit(train_df[feature_cols].values, train_df["points_scored"].values)

    preds = {}
    for _, row in test_df.iterrows():
        if row[feature_cols].isna().any():
            continue
        pred = model.predict([row[feature_cols].values])[0]
        preds[(row["game_id"], row["team"])] = pred
    return preds


def main():
    feats, feature_cols = load_features()
    games = load_games()

    weeks = sorted(feats["week"].unique())
    bets = []
    coverage = []

    for wk in weeks:
        if wk < MIN_TRAIN_WEEK:
            continue
        train = feats[feats["week"] < wk]
        test = feats[feats["week"] == wk]
        preds = predict_week(train, test, feature_cols)
        if not preds:
            continue

        wk_games = games[games["week"] == wk]
        n_total, n_scored = 0, 0
        for _, g in wk_games.iterrows():
            n_total += 1
            home_rows = test[(test["team"] == g["home_team"]) & (test["opponent"] == g["away_team"])]
            away_rows = test[(test["team"] == g["away_team"]) & (test["opponent"] == g["home_team"])]
            if home_rows.empty or away_rows.empty:
                continue
            key_h = (home_rows.iloc[0]["game_id"], g["home_team"])
            key_a = (away_rows.iloc[0]["game_id"], g["away_team"])
            if key_h not in preds or key_a not in preds:
                continue
            n_scored += 1

            model_total = preds[key_h] + preds[key_a]
            line = g["total_line"]
            edge = model_total - line
            if abs(edge) < EDGE_THRESHOLD:
                continue

            side = "over" if edge > 0 else "under"
            price = g["over_odds"] if side == "over" else g["under_odds"]
            actual = g["actual_total"]
            if actual == line:
                continue  # push
            won = (actual > line) if side == "over" else (actual < line)
            pnl = american_payout(100, price) if won else -100

            bets.append({
                "week": wk, "game": f"{g['away_team']} @ {g['home_team']}",
                "model_total": model_total, "line": line, "edge": edge,
                "side": side, "price": price, "actual_total": actual,
                "won": int(won), "pnl": pnl,
            })
        coverage.append({"week": wk, "games": n_total, "matched": n_scored})

    bets_df = pd.DataFrame(bets)
    cov_df = pd.DataFrame(coverage)

    print("=" * 78)
    print("Game-matching coverage (nflverse game_id format vs our pbp-derived id can miss)")
    print("=" * 78)
    print(cov_df.to_string(index=False))
    print(f"Total games: {cov_df['games'].sum()}  matched: {cov_df['matched'].sum()}")

    print("\n" + "=" * 78)
    print(f"WALK-FORWARD MODEL TOTAL vs REAL CLOSING LINE, edge >= {EDGE_THRESHOLD} pts, real odds")
    print("=" * 78)
    if bets_df.empty:
        print("no qualifying bets")
    else:
        n = len(bets_df)
        win_rate = bets_df["won"].mean()
        roi = bets_df["pnl"].sum() / (n * 100) * 100
        print(f"n={n}  win_rate={win_rate*100:.1f}%  total_pnl=${bets_df['pnl'].sum():,.2f}  ROI={roi:+.1f}%")

        print("\nBy side (over vs under):")
        seg = bets_df.groupby("side").agg(n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum"))
        seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
        print(seg.to_string())

        print("\nBy edge size:")
        bets_df["abs_edge"] = bets_df["edge"].abs()
        bins = [1, 2, 4, 6, 100]
        labels = ["1-2", "2-4", "4-6", "6+"]
        bets_df["edge_bucket"] = pd.cut(bets_df["abs_edge"], bins=bins, labels=labels, include_lowest=True)
        seg = bets_df.groupby("edge_bucket", observed=True).agg(n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum"))
        seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
        print(seg.to_string())

        print("\nBy week:")
        seg = bets_df.groupby("week").agg(n=("pnl", "size"), win_rate=("won", "mean"), pnl=("pnl", "sum"))
        seg["roi_pct"] = seg["pnl"] / (seg["n"] * 100) * 100
        seg["cume_pnl"] = seg["pnl"].cumsum()
        print(seg.to_string())

    os.makedirs("data/qc", exist_ok=True)
    bets_df.to_csv("data/qc/nfl_totals_backtest.csv", index=False)
    print("\nWrote data/qc/nfl_totals_backtest.csv")


if __name__ == "__main__":
    main()

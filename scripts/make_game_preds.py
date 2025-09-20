# scripts/make_game_preds.py
# Usage:
#   python3 scripts/make_game_preds.py --season 2025 --week 3
# Output:
#   data/games/model_preds_week{WEEK}.csv  with:
#   game_id, team_home, team_away, model_spread, model_total

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --------------------------
# Helpers
# --------------------------

TEAM_NAME_TO_ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
    "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
}

def _norm_team_name_to_abbr(s: str) -> str:
    if not isinstance(s, str): return "UNK"
    u = s.strip().upper()
    # common shorthand variations
    u = u.replace("SAINTS", "SAINTS").replace("PACKERS","PACKERS")  # no-op example; extend as needed
    return TEAM_NAME_TO_ABBR.get(u, u[:3])  # fallback crude abbr



def _norm_team_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.upper()
         .str.strip()
         .replace({"NAN": "UNK", "NONE": "UNK", "": "UNK"})
    )

def _pick(df: pd.DataFrame, names, default=np.nan):
    for n in names:
        if n in df: return df[n]
    return pd.Series([default] * len(df), index=df.index)

def make_future_week_matchups_from_odds(team_feats: pd.DataFrame, odds_csv: Path, future_week: int) -> pd.DataFrame:
    """
    Build matchup rows for a week not present in the parquet, using odds fixtures for home/away teams.
    Uses latest available rolling features for each team.
    """
    if not odds_csv.exists():
        raise FileNotFoundError(f"Missing odds CSV {odds_csv} for future week fixtures.")
    odds = pd.read_csv(odds_csv)

    # Expect columns like: home_team, away_team (The Odds API default)
    need = {"home_team","away_team"}
    miss = need - set(odds.columns)
    if miss:
        raise RuntimeError(f"Odds CSV missing columns {miss}. Got: {list(odds.columns)[:10]}")

    fixtures = odds[["home_team","away_team"]].drop_duplicates().copy()
    fixtures["team_home"] = fixtures["home_team"].map(_norm_team_name_to_abbr)
    fixtures["team_away"] = fixtures["away_team"].map(_norm_team_name_to_abbr)
    fixtures = fixtures[["team_home","team_away"]].dropna().drop_duplicates()

    # Latest rolling features available per team
    tf = team_feats.sort_values(["team","week"]).copy()
    last_week_avail = int(tf["week"].max())
    latest = (tf.sort_values("week")
                .groupby("team", as_index=False)
                .tail(1)
                .rename(columns={
                    "points_proxy":"points_latest",
                    "pass_yds_r":"pass_yds_r_latest",
                    "rush_yds_r":"rush_yds_r_latest",
                    "to_r":"to_r_latest",
                    "sacks_r":"sacks_r_latest",
                }))

    home = latest.rename(columns={
        "team":"team_home",
        "pass_yds_r_latest":"pass_yds_r_home",
        "rush_yds_r_latest":"rush_yds_r_home",
        "to_r_latest":"to_r_home",
        "sacks_r_latest":"sacks_r_home",
    })[["team_home","pass_yds_r_home","rush_yds_r_home","to_r_home","sacks_r_home"]]

    away = latest.rename(columns={
        "team":"team_away",
        "pass_yds_r_latest":"pass_yds_r_away",
        "rush_yds_r_latest":"rush_yds_r_away",
        "to_r_latest":"to_r_away",
        "sacks_r_latest":"sacks_r_away",
    })[["team_away","pass_yds_r_away","rush_yds_r_away","to_r_away","sacks_r_away"]]

    pairs = (fixtures
             .merge(home, on="team_home", how="left")
             .merge(away, on="team_away", how="left"))

        # ---- Impute missing team features with league medians ----
    med = {
        "pass_yds_r_home":  latest["pass_yds_r_latest"].median(),
        "rush_yds_r_home":  latest["rush_yds_r_latest"].median(),
        "to_r_home":        latest["to_r_latest"].median(),
        "sacks_r_home":     latest["sacks_r_latest"].median(),
        "pass_yds_r_away":  latest["pass_yds_r_latest"].median(),
        "rush_yds_r_away":  latest["rush_yds_r_latest"].median(),
        "to_r_away":        latest["to_r_latest"].median(),
        "sacks_r_away":     latest["sacks_r_latest"].median(),
    }
    pairs = pairs.fillna(value=med)


    # Build features
    pairs["yds_diff_r"]   = (pairs["pass_yds_r_home"] + pairs["rush_yds_r_home"]
                             - (pairs["pass_yds_r_away"] + pairs["rush_yds_r_away"]))
    pairs["to_diff_r"]    = pairs["to_r_home"] - pairs["to_r_away"]
    pairs["sack_diff_r"]  = pairs["sacks_r_home"] - pairs["sacks_r_away"]

    # No actuals for future week
    pairs["spread_actual"] = np.nan
    pairs["total_actual"]  = np.nan

    # Construct a consistent game_id for week N
    pairs["game_key"] = pairs.apply(lambda r: f"{future_week}|{'@'.join(sorted([r['team_home'], r['team_away']]))}", axis=1)
    pairs["week"] = future_week
    return pairs[["game_key","week","team_home","team_away","yds_diff_r","to_diff_r","sack_diff_r","spread_actual","total_actual"]]



# --------------------------
# 1) Player → Team/Game rollup (using your schema)
# --------------------------
def roll_player_to_team(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player weekly stats into team-per-game totals.
    Uses your columns: opponent_team, passing_interceptions, sacks_suffered, etc.
    Builds a TD/2pt-based points proxy because team points aren't present here.
    """
    week      = _pick(stats, ["week", "wk"], default=-1)
    team      = _pick(stats, ["team", "posteam", "team_abbr"], default="UNK")
    opponent  = _pick(stats, ["opponent_team", "opponent", "opp", "defteam"], default="UNK")
    game_id   = _pick(stats, ["game_id", "gameId", "gameid"], default=np.nan)

    # Offense volumes (your names)
    pass_yds  = _pick(stats, ["passing_yards", "pass_yards", "pass_yds"], default=0)
    rush_yds  = _pick(stats, ["rushing_yards", "rush_yards", "rush_yds"], default=0)

    # Turnover & pressure (your names)
    # Interceptions against the offense are 'passing_interceptions'
    ints      = _pick(stats, ["passing_interceptions", "interceptions"], default=0)
    sacks     = _pick(stats, ["sacks_suffered", "sacked", "sacks_taken"], default=0)

    # Fumbles lost (rushing & receiving variants – use what’s available)
    fumbles_rush_l = _pick(stats, ["rushing_fumbles_lost"], default=0)
    fumbles_rec_l  = _pick(stats, ["receiving_fumbles_lost", "receiving_fumbles"], default=0)  # latter is imperfect
    fumbles_l      = fumbles_rush_l.add(fumbles_rec_l, fill_value=0)

    # Touchdowns & 2pt (to build a points proxy)
    pass_tds  = _pick(stats, ["passing_tds"], 0)
    rush_tds  = _pick(stats, ["rushing_tds"], 0)
    recv_tds  = _pick(stats, ["receiving_tds"], 0)
    pass_2pt  = _pick(stats, ["passing_2pt_conversions"], 0)
    rush_2pt  = _pick(stats, ["rushing_2pt_conversions"], 0)
    # Note: no field goals/PATs here → proxy undercounts in FG-heavy games.
    points_proxy = (pass_tds + rush_tds + recv_tds) * 6 + (pass_2pt + rush_2pt) * 2

    mini = pd.DataFrame({
        "week":      week,
        "team":      team,
        "opponent":  opponent,
        "game_id":   game_id,
        "pass_yds":  pass_yds,
        "rush_yds":  rush_yds,
        "ints":      ints,
        "sacks":     sacks,
        "fumbles_lost": fumbles_l,
        "points_proxy": points_proxy,
    })

    # Canonical per-game key if game_id missing
    def make_game_key(r):
        try: w = int(r["week"])
        except Exception: w = -1
        t1 = str(r["team"]) if pd.notna(r["team"]) else "UNK"
        t2 = str(r["opponent"]) if pd.notna(r["opponent"]) else "UNK"
        a, b = sorted([t1, t2])
        return f"{w}|{a}@{b}"

    if mini["game_id"].isna().all():
        mini["game_key"] = mini.apply(make_game_key, axis=1)
    else:
        mini["game_key"] = mini["game_id"].astype(str)

    # Aggregate to team per game (keep opponent via first)
    team_games = (
        mini.groupby(["game_key", "week", "team"], dropna=False)
            .agg(opponent=("opponent","first"),
                 pass_yds=("pass_yds","sum"),
                 rush_yds=("rush_yds","sum"),
                 turnovers=("ints","sum"),
                 sacks=("sacks","sum"),
                 fumbles_lost=("fumbles_lost","sum"),
                 points_proxy=("points_proxy","sum"))
            .reset_index()
    )

    team_games["team"] = _norm_team_series(team_games["team"])
    team_games["opponent"] = _norm_team_series(team_games["opponent"])
    team_games["week"] = pd.to_numeric(team_games["week"], errors="coerce").fillna(-1).astype(int)
    return team_games

# --------------------------
# 2) Rolling team features
# --------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df = df.sort_values(["team", "week"])
    rolled = (
        df.groupby("team", group_keys=False)
          .apply(lambda g: g.assign(
              pass_yds_r = g["pass_yds"].rolling(window, min_periods=1).mean(),
              rush_yds_r = g["rush_yds"].rolling(window, min_periods=1).mean(),
              to_r       = g["turnovers"].rolling(window, min_periods=1).mean(),
              sacks_r    = g["sacks"].rolling(window, min_periods=1).mean(),
              pts_r      = g["points_proxy"].rolling(window, min_periods=1).mean(),
          ))
          .reset_index(drop=True)
    )
    keep = ["game_key","week","team","opponent",
            "pass_yds","rush_yds","turnovers","sacks","points_proxy",
            "pass_yds_r","rush_yds_r","to_r","sacks_r","pts_r"]
    return rolled[keep]

# --------------------------
# 3) Matchups (pair teams per game_key/week)
# --------------------------
def make_matchups(team_feats: pd.DataFrame) -> pd.DataFrame:
    req = {"game_key","week","team","points_proxy","pass_yds_r","rush_yds_r","to_r","sacks_r"}
    missing = req - set(team_feats.columns)
    if missing:
        raise RuntimeError(f"team_feats missing required columns: {missing}")

    tf = team_feats.drop_duplicates(["game_key","week","team"]).copy()
    tf["team"] = _norm_team_series(tf["team"])

    # pick the two teams per game_key
    teams_per_game = (tf.groupby(["game_key","week"])["team"]
                        .agg(lambda x: sorted(list(set(x))))
                        .reset_index())
    teams_per_game = teams_per_game[teams_per_game["team"].map(len) >= 2].copy()
    teams_per_game["team_home"] = teams_per_game["team"].map(lambda lst: lst[0])
    teams_per_game["team_away"] = teams_per_game["team"].map(lambda lst: lst[1])
    teams_per_game = teams_per_game.drop(columns=["team"])

    home = tf.rename(columns={
        "team":"team_home",
        "points_proxy":"points_home",
        "pass_yds_r":"pass_yds_r_home",
        "rush_yds_r":"rush_yds_r_home",
        "to_r":"to_r_home",
        "sacks_r":"sacks_r_home",
    })
    away = tf.rename(columns={
        "team":"team_away",
        "points_proxy":"points_away",
        "pass_yds_r":"pass_yds_r_away",
        "rush_yds_r":"rush_yds_r_away",
        "to_r":"to_r_away",
        "sacks_r":"sacks_r_away",
    })

    pairs = (teams_per_game
             .merge(home, on=["game_key","week","team_home"], how="inner")
             .merge(away, on=["game_key","week","team_away"], how="inner"))

    # Feature diffs (spread) and sums (total)
    pairs["yds_diff_r"]   = (pairs["pass_yds_r_home"] + pairs["rush_yds_r_home"]
                             - (pairs["pass_yds_r_away"] + pairs["rush_yds_r_away"]))
    pairs["to_diff_r"]    = pairs["to_r_home"] - pairs["to_r_away"]
    pairs["sack_diff_r"]  = pairs["sacks_r_home"] - pairs["sacks_r_away"]

    # Targets (proxy) for training/backtest
    pairs["spread_actual"] = pairs["points_home"] - pairs["points_away"]
    pairs["total_actual"]  = pairs["points_home"] + pairs["points_away"]

    out_cols = ["game_key","week","team_home","team_away",
                "yds_diff_r","to_diff_r","sack_diff_r",
                "spread_actual","total_actual"]
    return pairs[out_cols].drop_duplicates(subset=["game_key"]).reset_index(drop=True)

# --------------------------
# 4) Train & Predict
# --------------------------
def _fit_ridge(train_df, target, features):
    X = train_df[features].values
    y = train_df[target].values
    model = Ridge(alpha=1.0).fit(X, y)
    return model





def main(season: int, week: int):
    in_parquet = Path(f"data/weekly_player_stats_{season}.parquet")
    if not in_parquet.exists():
        raise FileNotFoundError(f"Missing {in_parquet}. Fetch nflverse weekly player stats first.")

    stats = pd.read_parquet(in_parquet)
    print(f"[stats] shape={stats.shape}")

    team_games = roll_player_to_team(stats)
    print(f"[team_games] {team_games.shape}")

    team_feats = add_rolling_features(team_games, window=3)
    print(f"[team_feats] {team_feats.shape}")

    matchups = make_matchups(team_feats)
    print(f"[matchups] {matchups.shape}")

    # --- robust split + forward-week fixtures from odds if needed ---
    ODDS_CSV = Path("data/odds/latest.csv")

    weeks_avail = sorted(matchups["week"].unique().tolist())
    if not weeks_avail:
        raise RuntimeError("No weeks found in matchups. Check upstream rollup.")
    latest_avail = weeks_avail[-1]

    if week in weeks_avail:
        # normal in-week prediction
        test_week = week
        train = matchups[matchups["week"] < test_week].copy()
        test  = matchups[matchups["week"] == test_week].copy()
    else:
        # future-week prediction: train on all available, build test from odds fixtures
        test_week = week
        print(f"[info] WEEK {week} not in stats; building fixtures from odds for forward prediction.")
        train = matchups.copy()
        test  = make_future_week_matchups_from_odds(team_feats, ODDS_CSV, future_week=test_week)

    if train.empty or test.empty:
        raise RuntimeError(f"Empty train/test after split (train={train.shape}, test={test.shape}).")

    feat_cols = ["yds_diff_r","to_diff_r","sack_diff_r"]
        # ---- Ensure no NaNs reach the model (use train means; fallback 0) ----
    train_means = train[feat_cols].mean()
    train[feat_cols] = train[feat_cols].fillna(train_means).replace([np.inf, -np.inf], 0)
    test[feat_cols]  = test[feat_cols].fillna(train_means).replace([np.inf, -np.inf], 0)
    # If any column is still all-NaN (edge case), zero it
    for c in feat_cols:
        if train[c].isna().all(): train[c] = 0.0
        if test[c].isna().all():  test[c]  = 0.0

    spread_model = _fit_ridge(train, "spread_actual", feat_cols)
    total_model  = _fit_ridge(train, "total_actual",  feat_cols)

    test = test.copy()
    test["model_spread"] = spread_model.predict(test[feat_cols].values)
    test["model_total"]  = total_model.predict(test[feat_cols].values)

    out = test[["game_key","team_home","team_away","model_spread","model_total"]].rename(
        columns={"game_key":"game_id"}
    )
    out_dir = Path("data/games")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_preds_week{test_week}.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path.as_posix()}  rows={out.shape[0]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()
    main(args.season, args.week)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()
    main(args.season, args.week)

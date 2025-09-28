# --- add/replace near the top of qc_datapull.py ---
import os, json, re, math, pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path

# resolve repo root even if run from scripts/
ROOT = Path(__file__).resolve().parents[1]

SEASON = int(os.getenv("SEASON", 2025))
WEEK   = int(os.getenv("WEEK", 4))

ODDS   = ROOT / "data/props/latest_all_props.csv"
PARAMS = ROOT / f"data/props/params_week{WEEK}.csv"
MERGED = ROOT / f"data/props/props_with_model_week{WEEK}.csv"
STATS  = ROOT / f"data/nflverse/stats_player_week_{SEASON}.parquet"
OUTDIR = ROOT / "data/qc"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / f"run_qc_week{WEEK}.json"

def mtime(path):
    p = Path(path)
    if not p.exists(): return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()

def fsize(path):
    p = Path(path)
    return p.stat().st_size if p.exists() else None


def pct(x):
    try: return round(100*float(x), 2)
    except: return None

def std_name(s):
    s = (str(s) if s is not None else "").strip().lower()
    s = re.sub(r"[^a-z0-9]+"," ",s)
    return re.sub(r"\s+"," ",s)

files = {
    "odds":   {"path": ODDS,   "exists": os.path.exists(ODDS),   "size": fsize(ODDS),   "mtime_utc": mtime(ODDS)},
    "params": {"path": PARAMS, "exists": os.path.exists(PARAMS), "size": fsize(PARAMS), "mtime_utc": mtime(PARAMS)},
    "merged": {"path": MERGED, "exists": os.path.exists(MERGED), "size": fsize(MERGED), "mtime_utc": mtime(MERGED)},
    "stats":  {"path": STATS,  "exists": os.path.exists(STATS),  "size": fsize(STATS),  "mtime_utc": mtime(STATS)},
}

report = {"files": files, "season": SEASON, "week": WEEK}

# --- STATS: training weeks used
stats_ok = files["stats"]["exists"]
if stats_ok:
    s = pd.read_parquet(STATS)
    s = s[(s["season"]==SEASON)]
    weeks_all = sorted(s["week"].dropna().unique().tolist())
    train = sorted(w for w in weeks_all if isinstance(w,(int,np.integer)) and w < WEEK)
    report["stats"] = {
        "weeks_present": weeks_all,
        "trained_on_weeks": train,
        "trained_span": [min(train), max(train)] if train else None,
        "rows_total": int(len(s)),
        "rows_by_week": s.groupby("week").size().to_dict(),
    }

# --- ODDS overview
if files["odds"]["exists"]:
    o = pd.read_csv(ODDS, low_memory=False)
    # basic normalize
    if "name" in o:
        o["name_std"] = o["name"].fillna(o.get("player")).map(std_name)
    elif "player" in o:
        o["name_std"] = o["player"].map(std_name)
    else:
        o["name_std"] = ""

    pulled_markets = sorted(o.get("market_std", o.get("market")).dropna().map(str).str.lower().unique().tolist())
    uniq_games = int(o["game_id"].nunique()) if "game_id" in o else None
    uniq_books = int(o["bookmaker"].nunique()) if "bookmaker" in o else None
    # identity dup rate
    id_keys = [k for k in ["bookmaker","game_id","market_std","player","point"] if k in o.columns]
    dup_rate = 1 - (o.drop_duplicates(id_keys).shape[0] / len(o)) if id_keys else np.nan
    # crude completeness heuristics
    per_game_books = (o.groupby("game_id")["bookmaker"].nunique().median()
                      if "game_id" in o and "bookmaker" in o else None)
    per_market_lines = (o.groupby("market_std")["point"].nunique().median()
                        if "market_std" in o and "point" in o else None)

    report["odds"] = {
        "rows": int(len(o)),
        "unique_games": uniq_games,
        "unique_books": uniq_books,
        "median_books_per_game": float(per_game_books) if per_game_books is not None else None,
        "median_lines_per_market": float(per_market_lines) if per_market_lines is not None else None,
        "duplicate_rate_identity_pct": pct(dup_rate),
        "identity_keys": id_keys,
        "pulled_markets": pulled_markets,
        "pulled_markets_count": len(pulled_markets),
    }

# --- PARAMS overview
if files["params"]["exists"]:
    p = pd.read_csv(PARAMS, low_memory=False)
    POISSON = {"anytime_td","pass_tds","pass_interceptions"}
    # Trust market_std if present
    p["market_std"] = p["market_std"].astype(str).str.lower()
    fam_normal = p[~p["market_std"].isin(POISSON)]
    fam_pois   = p[p["market_std"].isin(POISSON)]

    params_markets = sorted(p["market_std"].dropna().unique().tolist())
    report["params_overview"] = {
        "rows": int(len(p)),
        "markets": params_markets,
        "markets_count": len(params_markets),
        "coverage": {
            "normal_mu_pct": pct(fam_normal["mu"].notna().mean()) if "mu" in fam_normal else None,
            "normal_sigma_pct": pct(fam_normal["sigma"].notna().mean()) if "sigma" in fam_normal else None,
            "poisson_lam_pct": pct(fam_pois["lam"].notna().mean()) if "lam" in fam_pois else None,
        },
        "bad_normal_rows": {
            "mu_lt_0": int(((fam_normal.get("mu",np.nan) < 0).fillna(False)).sum()),
            "sigma_le_0": int(((fam_normal.get("sigma",np.nan) <= 0).fillna(False)).sum()),
        },
    }

# --- MERGED (modeled) overview
if files["merged"]["exists"]:
    m = pd.read_csv(MERGED, low_memory=False)
    m["market_std"] = m["market_std"].astype(str).str.lower()
    modeled_rows = m[m.get("model_prob").notna()] if "model_prob" in m else m.iloc[0:0]
    modeled_markets = sorted(modeled_rows["market_std"].dropna().unique().tolist())

    # overlap vs odds
    pulled = set(report.get("odds",{}).get("pulled_markets", []))
    modeled = set(modeled_markets)
    overlap = sorted(pulled & modeled)
    gaps    = sorted(pulled - modeled)

    report["modeled"] = {
        "rows_total": int(len(m)),
        "rows_modeled": int(len(modeled_rows)),
        "modeled_pct": pct(len(modeled_rows)/len(m) if len(m) else np.nan),
        "markets_modeled": modeled_markets,
        "markets_modeled_count": len(modeled_markets),
        "pulled_vs_modeled_overlap": overlap,
        "overlap_count": len(overlap),
        "pulled_but_unmodeled": gaps,
        "pulled_but_unmodeled_count": len(gaps),
        "model_prob_null_pct": pct(m["model_prob"].isna().mean()) if "model_prob" in m else 100.0,
        "edge_bps_null_pct": pct(m["edge_bps"].isna().mean()) if "edge_bps" in m else 100.0,
    }

# --- Run stamp (simple PASS/WARN/FATAL signals)
signals = {"fatal": [], "warn": [], "info": []}

# file presence
for k,v in files.items():
    if not v["exists"]:
        signals["fatal"].append(f"missing_file:{k}:{v['path']}")

# stale/incomplete heuristic: odds older than params/merged
def newer(a,b):
    return (files[a]["mtime_utc"] or "") >= (files[b]["mtime_utc"] or "")
if files["params"]["exists"] and files["odds"]["exists"] and not newer("params","odds"):
    signals["warn"].append("params_older_than_odds")
if files["merged"]["exists"] and files["params"]["exists"] and not newer("merged","params"):
    signals["warn"].append("merged_older_than_params")

# trained weeks stamp
if "stats" in report and not report["stats"]["trained_on_weeks"]:
    signals["warn"].append("no_training_weeks_found")

# odds duplicate rate
dup = report.get("odds",{}).get("duplicate_rate_identity_pct")
if dup is not None:
    if dup > 10: signals["warn"].append(f"odds_dup_rate_high:{dup}%")
    elif dup > 2: signals["info"].append(f"odds_dup_rate_elevated:{dup}%")

# modeled coverage
mp_null = report.get("modeled",{}).get("model_prob_null_pct")
if mp_null == 100.0:
    signals["fatal"].append("model_prob_all_null")
elif mp_null and mp_null > 50:
    signals["warn"].append(f"model_prob_mostly_null:{mp_null}%")

report["signals"] = signals

# === NEW: verbose drilldowns & CSV artifacts ===
VERBOSE = bool(int(os.getenv("VERBOSE", "1")))

def _write_csv(df, name):
    path = OUTDIR / name
    df.to_csv(path, index=False)
    return str(path)

pulled_markets = report.get("odds", {}).get("pulled_markets", []) or []
modeled_markets = report.get("modeled", {}).get("markets_modeled", []) or []

# --- Summaries from ODDS
if files["odds"]["exists"]:
    o = pd.read_csv(ODDS, low_memory=False)
    if "market_std" not in o and "market" in o:
        o["market_std"] = o["market"].astype(str).str.lower()
    o["name_std"] = (o.get("name", o.get("player", "")).astype(str)
                     .str.lower().str.replace(r"[^a-z0-9]+"," ", regex=True).str.strip())

    # Per-market odds coverage
    mkt_odds = (o.groupby("market_std")
                  .agg(rows=("market_std","size"),
                       uniq_players=("name_std","nunique"),
                       uniq_books=("bookmaker","nunique") if "bookmaker" in o else ("market_std","size"),
                       uniq_points=("point","nunique") if "point" in o else ("market_std","size"))
                  .reset_index()
                  .sort_values("rows", ascending=False))
    report["odds"]["markets_detail_top"] = mkt_odds.head(20).to_dict(orient="records")
    odds_csv = _write_csv(mkt_odds, f"markets_pulled_summary_week{WEEK}.csv")

    # Per-game books coverage (median, p10/p90)
    if "game_id" in o and "bookmaker" in o:
        gb = (o.groupby("game_id")["bookmaker"].nunique().rename("books_per_game"))
        report["odds"]["books_per_game"] = {
            "median": float(gb.median()),
            "p10": float(gb.quantile(0.10)),
            "p90": float(gb.quantile(0.90)),
            "min": int(gb.min()),
            "max": int(gb.max())
        }
        _write_csv(gb.reset_index(), f"odds_books_per_game_week{WEEK}.csv")

    # Side distribution in odds (if present)
    if "side" in o:
        side_odds = (o["side"].str.title().value_counts(dropna=False).reset_index())
        side_odds.columns = ["side","count"]
        report["odds"]["side_distribution"] = side_odds.to_dict(orient="records")
        _write_csv(side_odds, f"odds_side_distribution_week{WEEK}.csv")

# --- Summaries from PARAMS (suspicious zeros where books have lines)
if files["params"]["exists"] and files["odds"]["exists"]:
    p = pd.read_csv(PARAMS, low_memory=False)
    POISSON = {"anytime_td","pass_tds","pass_interceptions"}
    p["market_std"] = p["market_std"].astype(str).str.lower()
    fam_normal = p[~p["market_std"].isin(POISSON)].copy()

    o_avail = pd.read_csv(ODDS, low_memory=False)
    if "market_std" not in o_avail and "market" in o_avail:
        o_avail["market_std"] = o_avail["market"].astype(str).str.lower()
    o_avail["name_std"] = (o_avail.get("name", o_avail.get("player","")).astype(str)
                           .str.lower().str.replace(r"[^a-z0-9]+"," ",regex=True).str.strip())
    avail_keys = o_avail[["name_std","market_std"]].dropna().drop_duplicates()

    fam_normal["mu_zero"] = fam_normal["mu"].fillna(0).eq(0)
    fam_normal["sigma_zero"] = fam_normal["sigma"].fillna(0).eq(0)
    sus = (fam_normal.merge(avail_keys, on=["name_std","market_std"], how="inner")
                     .query("mu_zero or sigma_zero"))

    sus_by_market = (sus.groupby("market_std")
                       .agg(suspicious=("market_std","size"),
                            players=("name_std","nunique"))
                       .reset_index()
                       .sort_values("suspicious", ascending=False))
    _write_csv(sus_by_market, f"suspicious_param_zeros_by_market_week{WEEK}.csv")
    _write_csv(sus[["name_std","market_std","mu","sigma","mu_zero","sigma_zero"]]
                 .sort_values(["market_std","name_std"]),
               f"suspicious_param_zeros_rows_week{WEEK}.csv")
    report["params_overview"]["suspicious_zero_top"] = sus_by_market.head(20).to_dict(orient="records")

# --- Summaries from MERGED (modeled coverage per market, side dist, etc.)
if files["merged"]["exists"]:
    m = pd.read_csv(MERGED, low_memory=False)
    m["market_std"] = m["market_std"].astype(str).str.lower()
    modeled_mask = m["model_prob"].notna() if "model_prob" in m else pd.Series([False]*len(m))

    # Per-market modeled coverage
    mkt_modeled = (m.assign(modeled=modeled_mask)
                     .groupby("market_std")
                     .agg(rows=("market_std","size"),
                          modeled_rows=("modeled","sum"),
                          modeled_pct=("modeled", "mean"),
                          uniq_players=("name_std","nunique"),
                          uniq_books=("bookmaker","nunique") if "bookmaker" in m else ("market_std","size"),
                          uniq_points=("point","nunique") if "point" in m else ("market_std","size"))
                     .reset_index()
                     .sort_values("modeled_pct", ascending=True))
    mkt_modeled["modeled_pct"] = (mkt_modeled["modeled_pct"]*100).round(2)
    modeled_csv = _write_csv(mkt_modeled, f"markets_modeled_summary_week{WEEK}.csv")
    report["modeled"]["markets_detail_bottom"] = mkt_modeled.head(20).to_dict(orient="records")

    # Side distribution in merged (modeled vs all)
    if "side" in m:
        side_all = m["side"].str.title().value_counts(dropna=False).reset_index()
        side_all.columns = ["side","count"]
        _write_csv(side_all, f"merged_side_distribution_all_week{WEEK}.csv")
        if "model_prob" in m:
            side_mod = m.loc[m["model_prob"].notna(),"side"].str.title().value_counts(dropna=False).reset_index()
            side_mod.columns = ["side","count"]
            _write_csv(side_mod, f"merged_side_distribution_modeled_week{WEEK}.csv")
            report["modeled"]["side_distribution_modeled"] = side_mod.to_dict(orient="records")

# --- Pulled vs Modeled overlap table (pretty)
if pulled_markets or modeled_markets:
    pulled_set  = set(pulled_markets)
    modeled_set = set(modeled_markets)
    all_mkts = sorted(pulled_set | modeled_set)
    rows = []
    mkt_modeled_dict = {}
    try:
        mkt_modeled_dict = {r["market_std"]: r for _, r in mkt_modeled.set_index("market_std").to_dict(orient="index").items()}
    except Exception:
        pass
    for mk in all_mkts:
        rows.append({
            "market_std": mk,
            "pulled": mk in pulled_set,
            "modeled": mk in modeled_set,
            "modeled_rows": mkt_modeled_dict.get(mk,{}).get("modeled_rows"),
            "modeled_pct": mkt_modeled_dict.get(mk,{}).get("modeled_pct"),
        })
    pvsm = pd.DataFrame(rows)
    _write_csv(pvsm, f"pulled_vs_modeled_detail_week{WEEK}.csv")
    report["pulled_vs_modeled_detail_top"] = pvsm.sort_values(["modeled","pulled","market_std"], ascending=[True,False,True]).head(20).to_dict(orient="records")

# --- Anti-join diagnostics between PARAMS keys and MERGED keys
if files["params"]["exists"] and files["merged"]["exists"]:
    p = pd.read_csv(PARAMS, low_memory=False)
    m = pd.read_csv(MERGED, low_memory=False)
    for df in (p,m):
        df["market_std"] = df["market_std"].astype(str).str.lower()
        if "side" in df:
            df["side"] = df["side"].astype(str).str.title()
    need = {"name_std","market_std","side"}
    if need.issubset(p.columns) and need.issubset(m.columns):
        pk = p[list(need)].drop_duplicates()
        mk = m[list(need)].drop_duplicates()
        anti = (pk.merge(mk, on=list(need), how="left", indicator=True)
                  .query("_merge=='left_only'")
                  .drop(columns=["_merge"]))
        _write_csv(anti, f"anti_join_params_vs_merged_week{WEEK}.csv")
        anti_by_mkt  = anti["market_std"].value_counts().reset_index().rename(columns={"index":"market_std","market_std":"misses"})
        anti_by_side = anti["side"].value_counts().reset_index().rename(columns={"index":"side","side":"misses"})
        _write_csv(anti_by_mkt,  f"anti_join_by_market_week{WEEK}.csv")
        _write_csv(anti_by_side, f"anti_join_by_side_week{WEEK}.csv")
        report.setdefault("join_audit", {})["anti_join_counts"] = {
            "by_market_top": anti_by_mkt.head(20).to_dict(orient="records"),
            "by_side": anti_by_side.to_dict(orient="records"),
            "total_misses": int(len(anti)),
        }

# --- Console pretty-print
if VERBOSE:
    print("\n=== ODDS: markets pulled (top 20 by rows) ===")
    if "odds" in report and "markets_detail_top" in report["odds"]:
        for r in report["odds"]["markets_detail_top"]:
            print(f"{r['market_std']:<22} rows={r['rows']:<6} players={r['uniq_players']:<4} books={r.get('uniq_books')} points={r.get('uniq_points')}")
    print("\n=== MODELED: markets (lowest modeled% first) ===")
    if "modeled" in report and "markets_detail_bottom" in report["modeled"]:
        for r in report["modeled"]["markets_detail_bottom"]:
            print(f"{r['market_std']:<22} rows={r['rows']:<6} modeled={r['modeled_rows']:<6} modeled%={r['modeled_pct']:<6}")
    print("\n=== PULLED vs MODELED (detail) → data/qc/pulled_vs_modeled_detail_week{WEEK}.csv ===")
    print("\n=== ANTI-JOIN (params→merged) top markets ===")
    aj = report.get("join_audit",{}).get("anti_join_counts",{}).get("by_market_top",[])
    for r in aj[:20]:
        print(f"{r['market_std']:<22} misses={r['misses']}")
    print("\n=== PARAMS suspicious zeros (Normal) top markets ===")
    for r in report.get("params_overview",{}).get("suspicious_zero_top",[])[:20]:
        print(f"{r['market_std']:<22} suspicious_rows={r['suspicious']:<6} players={r['players']}")





with open(OUT, "w") as f:
    json.dump(report, f, indent=2, default=str)


print(f"[QC] wrote {OUT}")
print(f"Trained on weeks: {report.get('stats',{}).get('trained_on_weeks')}")
print(f"Pulled markets ({report.get('odds',{}).get('pulled_markets_count')}): {report.get('odds',{}).get('pulled_markets')}")
print(f"Modeled markets ({report.get('modeled',{}).get('markets_modeled_count')}): {report.get('modeled',{}).get('markets_modeled')}")
print(f"Pulled but unmodeled ({report.get('modeled',{}).get('pulled_but_unmodeled_count')}): {report.get('modeled',{}).get('pulled_but_unmodeled')}")

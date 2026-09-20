#!/usr/bin/env python3
# make_ai_commentary.py
# Produces per-game commentary for Insights. Writes JSON mapping: {game_norm: text}.
# Calls OpenAI if OPENAI_API_KEY + openai SDK are available; otherwise uses a deterministic fallback.

import argparse, json, os, sys, textwrap, math, re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# --- optional OpenAI client ---
def _get_openai_client():
    try:
        from openai import OpenAI  # pip install openai>=1.0
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAI()
    except Exception:
        return None

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") if s else ""

PRETTY_MAP = {
    "recv_yds": "Receiving Yards",
    "rush_yds": "Rushing Yards",
    "pass_yds": "Passing Yards",
    "pass_tds": "Passing TDs",
    "pass_interceptions": "Interceptions",
    "receptions": "Receptions",
    "anytime_td": "Anytime TD",
    "1st_td": "1st TD",
    "last_td": "Last TD",
}
MAX_INSIGHT_EDGE_BPS = 1500

def _pretty_market(m):
    if m is None: return ""
    return PRETTY_MAP.get(str(m), str(m))

def _pick_odds_col(obj):
    if hasattr(obj, "columns"):
        cols = set(obj.columns)
    elif isinstance(obj, pd.Series):
        cols = set(obj.index)
    else:
        try: cols = set(obj.keys())
        except Exception: cols = set()
    for c in ("american_odds","mkt_odds","odds","price"):
        if c in cols: return c
    return None

def _ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    aliases = {
        "game": ["game","matchup"],
        "player": ["player"],
        "market_std": ["market_std","market","bet"],
        "name": ["name","side","pick"],
        "kick_et": ["kick_et","kickoff","kick","kickoff_et"],
        "mkt_prob": ["mkt_prob","market_prob","consensus_prob","book_implied_prob"],
        "model_prob": ["model_prob","model_probability"],
        "edge_bps": ["edge_bps","edge","edge_bps_"],
        "american_odds": ["american_odds","mkt_odds","odds","price"],
        "game_norm": ["game_norm"],
    }
    for std, cands in aliases.items():
        if std not in df.columns:
            for c in cands:
                if c in df.columns:
                    df.rename(columns={c: std}, inplace=True)
                    break

    for p in ("mkt_prob","model_prob"):
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    if "edge_bps" not in df.columns or df["edge_bps"].isna().all():
        if {"model_prob","mkt_prob"}.issubset(df.columns):
            df["edge_bps"] = (df["model_prob"] - df["mkt_prob"]) * 10000

    if "game_norm" not in df.columns or df["game_norm"].isna().all():
        df["game_norm"] = df["game"]

    return df

def _select_top_for_game(df_g: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Missing evidence must not fall back to a market price masquerading as a model.
    prio = df_g[df_g["edge_bps"].gt(0) & df_g["model_prob"].notna() & df_g["mkt_prob"].notna()].copy()
    if "book_count" in prio.columns:
        prio = prio[prio["book_count"].fillna(0).ge(3)]
    prio = prio[prio["edge_bps"].le(MAX_INSIGHT_EDGE_BPS)]
    # One row per player/market/side/line. This keeps an outlier book from
    # dominating the prose when several books quote the same bet.
    keys = [c for c in ["player", "market_std", "name", "point"] if c in prio.columns]
    if keys:
        prio = prio.sort_values("edge_bps", ascending=False).drop_duplicates(keys)
    return prio.sort_values("edge_bps", ascending=False).head(top_n).copy()

import numpy as np
import textwrap
import pandas as pd

WEEKLY_OVERVIEW_TEMPLATE = """You're helping a casual NFL fan understand the betting landscape for Week {week}.

Week {week} Overview:
- Total games: {num_games}
- Top 5 strongest edges: {top_edges_week}
- Market themes: {market_themes}
- High-confidence plays (>80%): {high_conf_count}
- Arbitrage opportunities: {arb_count}

Write a friendly, conversational overview (4-6 sentences) that:
- Highlights the biggest themes this week (e.g., "Lots of tight end unders popping" or "RB props looking soft")
- Mentions 2-3 standout players with the strongest edges across all games
- Notes if there are good arbitrage opportunities worth exploring
- Gives a general confidence vibe for the week (e.g., "Solid week with some confident plays" vs "Tighter lines this week")
- Sounds like you're catching up a friend on what to look for

Avoid generic phrases. Be specific and data-driven but conversational.
Return only the paragraph.
"""

SUMMARY_TEMPLATE = """You're a knowledgeable friend helping a casual NFL fan understand this NFL matchup's modeled player-prop angles.

Game: {matchup}

What our model sees:
- Evidence-backed angles: {top_edges}
- Market directions: {market_skew}
- Price-shopping notes: {best_books}
- Injury context, if present: {injury_notes}

IMPORTANT: Only mention markets that appear in the "Market directions" and "Strongest edges" data above. Do NOT mention or recommend markets like "longest reception", "longest rush", "first TD", "last TD", or any other markets not explicitly shown in the data. Stick strictly to the markets we actually model.

Write a friendly, conversational paragraph (4-6 sentences) that:
- Names up to 2 players and the exact side and line when the evidence supports it.
- Explains the disagreement using the supplied model probability, market probability, edge and book count.
- Mentions the best available book only when it is in the supplied evidence.
- Gives an honest confidence description; these are experimental estimates, not guarantees.
- If no strong angle is supplied, say the matchup is thin rather than inventing a pick.

Avoid:
- Generic phrases like "pops most on our numbers" or "keep stakes modest"
- Overly formal language
- Bullet points or lists
- Mentioning any markets not in the provided data

Return only the conversational paragraph.
"""

def _prompt_for_weekly_overview(df: pd.DataFrame, season: int, week: int) -> str:
    """Generate prompt for week-at-a-glance overview."""
    # Filter to only markets we actually model (have model_prob)
    df = df[df['model_prob'].notna()].copy()

    # Top 5 edges across all games
    if {'player', 'market_std', 'edge_bps'}.issubset(df.columns):
        eligible = df[df.get('book_count', pd.Series(0, index=df.index)).fillna(0).ge(3)
                      & df['edge_bps'].gt(0) & df['edge_bps'].le(MAX_INSIGHT_EDGE_BPS)].copy()
        top = (eligible[eligible['edge_bps'].gt(0)][['player','market_std','edge_bps']]
               .dropna()
               .sort_values('edge_bps', ascending=False)
               .drop_duplicates(['player', 'market_std'])
               .head(5))
        top_edges_week = "; ".join(f"{r.player} {r.market_std} ({int(r.edge_bps)} bps)"
                                    for _, r in top.iterrows()) if len(top) else "n/a"
    else:
        top_edges_week = "n/a"

    # Market themes (which markets have most edges)
    if {'edge_bps', 'market_std'}.issubset(df.columns):
        theme_counts = df[df['edge_bps'].abs() > 200].groupby('market_std').size().sort_values(ascending=False).head(3)
        market_themes = ", ".join(f"{m}: {c} edges" for m, c in theme_counts.items()) if len(theme_counts) else "n/a"
    else:
        market_themes = "n/a"

    # High confidence count
    high_conf_count = len(df[df.get('model_prob', 0) > 0.8]) if 'model_prob' in df.columns else 0

    # Don't mention arbitrage count - the family_arbitrage.csv file
    # contains incoherence warnings, not actual arb opportunities
    arb_count = 0

    num_games = len(df['game_norm'].unique()) if 'game_norm' in df.columns else 0

    prompt = WEEKLY_OVERVIEW_TEMPLATE.format(
        week=week,
        num_games=num_games,
        top_edges_week=top_edges_week,
        market_themes=market_themes,
        high_conf_count=high_conf_count,
        arb_count=arb_count
    )
    return textwrap.dedent(prompt).strip()

def _prompt_for_game(game: str, rows: pd.DataFrame, season: int, week: int) -> str:
    """
    Build a compact, model-led paragraph prompt for GPT-4, using the rows for this matchup.
    """
    df = rows.copy()

    # Filter to only markets we actually model (have model_prob)
    df = df[df['model_prob'].notna()].copy()

    if len(df) == 0:
        return ""

    # --- Market skew by over/under counts per market type
    if {'edge_bps', 'market_std'}.issubset(df.columns):
        skew = (df.assign(dir=df['name'].str.lower())
                  .groupby(['market_std', 'dir']).size()
                  .unstack(fill_value=0))
        parts = []
        for m in skew.index:
            o = int(skew.loc[m].get('over', 0))
            u = int(skew.loc[m].get('under', 0))
            segs = []
            if o: segs.append(f"overs {o}")
            if u: segs.append(f"unders {u}")
            parts.append(f"{m}: {' '.join(segs) if segs else 'balanced'}")
        market_skew = ", ".join(parts) if len(parts) else "n/a"
    else:
        market_skew = "n/a"

    top = _select_top_for_game(df, 3)
    if len(top):
        bits = []
        for _, r in top.iterrows():
            line = "" if pd.isna(r.get("point")) else f" {r.point:g}"
            price = "" if pd.isna(r.get("american_odds")) else f" {r.american_odds:+.0f}"
            books = int(r.book_count) if pd.notna(r.get("book_count")) else 0
            bits.append(f"{r['player']} {r['name']}{line}{price}: model {r['model_prob']:.1%} vs market {r['mkt_prob']:.1%}, +{int(r['edge_bps'])} bps across {books} books")
        top_edges = "; ".join(bits)
    else:
        top_edges = "No consensus-supported positive edge met the evidence threshold."

    # --- Confidence spread (model_conf percentile band)
    conf_range = "experimental; no confidence guarantee"

    # --- Most frequent best-price books
    if 'best_book' in df.columns and df['best_book'].notna().any():
        counts = df['best_book'].value_counts().head(3).to_dict()
        best_books = ", ".join(f"{k}: {v}" for k, v in counts.items())
    else:
        best_books = "n/a"

    if {'player', 'injury_status'}.issubset(df.columns):
        injured = df[df['injury_status'].notna()][['player', 'injury_status']].drop_duplicates()
        injury_notes = "; ".join(f"{r['player']} ({r['injury_status']})" for _, r in injured.head(6).iterrows()) or "none in modeled rows"
    else:
        injury_notes = "none supplied"

    prompt = SUMMARY_TEMPLATE.format(
        matchup=game,
        market_skew=market_skew or "n/a",
        top_edges=top_edges or "n/a",
        conf_range=conf_range,
        best_books=best_books or "n/a",
        injury_notes=injury_notes,
    )
    return textwrap.dedent(prompt).strip()

def _call_llm(client, model, prompt: str) -> str:
    if client is None:
        return ""
    try:
        resp = client.responses.create(
            model=model,
            instructions=("Explain only the supplied NFL data. Distinguish the named bet side from the sign of an edge. "
                          "Do not invent injuries, lineup news, causes, guarantees, or calibrated performance. "
                          "Model estimates are experimental; say so. Return only the requested paragraph."),
            input=prompt,
        )
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return ""



def _fallback_text(game: str, rows: pd.DataFrame) -> str:
    # Deterministic, data-backed copy when the API is unavailable or out of
    # quota. It should still be useful to a casual reader.
    if rows.empty:
        return f"{game}: no consensus-supported positive edge cleared the evidence filter in this snapshot. The board is more useful here for comparing prices than forcing a bet."
    bits = []
    for _, r in rows.head(2).iterrows():
        line = "" if pd.isna(r.get("point")) else f" {r.point:g}"
        price = "" if pd.isna(r.get("american_odds")) else f" at {r.american_odds:+.0f}"
        books = int(r.book_count) if pd.notna(r.get("book_count")) else 0
        bits.append(f"{r['player']}'s {r['name']} {_pretty_market(r['market_std'])}{line}{price} has a model probability of {r['model_prob']:.1%} versus {r['mkt_prob']:.1%} implied by the market across {books} books")
    return f"{game}: " + "; ".join(bits) + ". These are experimental estimates, so the disagreement is a research lead rather than a guarantee; shop the listed price before deciding."

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate per-game LLM commentary for Insights.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--merged_csv", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--model", type=str, default=os.getenv("OPENAI_INSIGHTS_MODEL", "gpt-6"))
    ap.add_argument("--out_json", type=str, required=True, help="Output path for {game_norm: text} JSON")
    ap.add_argument("--force", action="store_true", help="Overwrite existing out_json")
    args = ap.parse_args(argv)

    if os.path.exists(args.out_json) and not args.force:
        print(f"[info] {args.out_json} exists. Use --force to overwrite.", file=sys.stderr)

    df = pd.read_csv(args.merged_csv, low_memory=False)
    # Use the same evidence and quote-age requirements as the NFL shortlist.
    now = pd.Timestamp.now(tz="UTC")
    if {"model_status", "last_update", "commence_time"}.issubset(df.columns):
        age = (now - pd.to_datetime(df["last_update"], utc=True, errors="coerce")).dt.total_seconds()
        df = df[df["model_status"].str.startswith("Calibration fitted", na=False)
                & df["edge_bps"].gt(0) & age.between(0, 48 * 3600)
                & pd.to_datetime(df["commence_time"], utc=True, errors="coerce").gt(now)].copy()
    else:
        df = df.iloc[:0].copy()
    if df.empty:
        out = {"season": args.season, "week": args.week, "generated_at": now.isoformat(),
               "week_overview": "No qualifying model-backed picks in this snapshot. Fresh quote timestamps and supported player estimates are required. Compare the props board for coverage.", "games": []}
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        return
    df = _ensure_core(df)

    # Join the same target-week injury evidence used by the props model. The
    # report is optional so historical or offline builds remain reproducible.
    injury_path = Path(f"data/injuries/injuries_week{args.week}.csv")
    if injury_path.exists() and "name_std" in df.columns:
        inj = pd.read_csv(injury_path, low_memory=False)
        if {"name_std", "source_status", "availability"}.issubset(inj.columns):
            inj = inj[["name_std", "source_status", "availability"]].drop_duplicates("name_std")
            df = df.merge(inj.rename(columns={"source_status": "injury_status",
                                               "availability": "injury_availability"}),
                          on="name_std", how="left")

    games: List[str] = list(pd.unique(df["game_norm"].dropna()))
    if not games:
        print("[error] No games found in merged CSV (missing game_norm).", file=sys.stderr)
        sys.exit(2)

    client = _get_openai_client()
    if client is None:
        print("[warn] OPENAI_API_KEY not found or openai SDK missing; using deterministic fallback.", file=sys.stderr)

    # Generate weekly overview first
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Generating AI Insights for Week {args.week}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"[1/{len(games)+1}] Generating weekly overview...", file=sys.stderr, flush=True)
    weekly_prompt = _prompt_for_weekly_overview(df, args.season, args.week)
    weekly_overview = _call_llm(client, args.model, weekly_prompt) or ""
    if not weekly_overview.strip():
        # Fallback for weekly overview
        top_edge = df.nlargest(1, 'edge_bps').iloc[0] if 'edge_bps' in df.columns and len(df) > 0 else None
        eligible = df[df['edge_bps'].gt(0) & df['edge_bps'].le(MAX_INSIGHT_EDGE_BPS)
                      & df['model_prob'].notna() & df['mkt_prob'].notna()].copy()
        if 'book_count' in eligible.columns:
            eligible = eligible[eligible['book_count'].fillna(0).ge(3)]
        keys = [c for c in ['player', 'market_std', 'name', 'point'] if c in eligible.columns]
        if keys:
            eligible = eligible.sort_values('edge_bps', ascending=False).drop_duplicates(keys)
        top_rows = eligible.head(3)
        if len(top_rows):
            angles = "; ".join(
                f"{r['player']} {r['name']} {r['market_std']} ({int(r['edge_bps'])} bps)"
                for _, r in top_rows.iterrows())
            weekly_overview = f"Week {args.week} has {len(games)} games with consensus-supported model angles. The clearest examples are {angles}. Compare the exact line and price on each game page; these estimates are experimental."
        else:
            weekly_overview = f"Week {args.week} features {len(games)} games. Select a game above to see detailed analysis."
    print(f"    ✓ Weekly overview complete", file=sys.stderr)

    out = {
        "season": args.season, "week": args.week, "generated_at": now.isoformat(),
        "week_overview": weekly_overview.strip(),
        "games": []
    }

    for idx, g in enumerate(games, start=2):
        df_g = df[df["game_norm"] == g]
        top = _select_top_for_game(df_g, args.top_n)

        prompt = _prompt_for_game(g, df_g, args.season, args.week)
        pct = int((idx-1) / len(games) * 100)
        bar_length = 40
        filled = int(bar_length * (idx-1) / len(games))
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{idx}/{len(games)+1}] {bar} {pct}% | {g[:50]}", file=sys.stderr, end='', flush=True)

    # at the top of the run (once)


        # per-game
        text = _call_llm(client, args.model, prompt) or ""
        if not text.strip():
            text = _fallback_text(g, top)

        display_game = (
            df_g["game"].iloc[0] if "game" in df_g.columns and not df_g.empty
            else (df_g["matchup"].iloc[0] if "matchup" in df_g.columns and not df_g.empty
                  else str(g))
        )

        out["games"].append({
            "game": display_game,
            "summary": text.strip(),
        })



    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*70}", file=sys.stderr)
    print(f"✓ Complete! Generated insights for {len(out['games'])} games", file=sys.stderr)
    print(f"  Output: {args.out_json}", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

if __name__ == "__main__":
    main()

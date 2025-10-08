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
    # prefer rows with model + market probs (real edge); if none, use whatever has mkt_prob
    prio = df_g[df_g["edge_bps"].notna() & df_g["mkt_prob"].notna() & df_g["model_prob"].notna()]
    if prio.empty:
        prio = df_g[df_g["mkt_prob"].notna()]
    prio = prio.sort_values(["edge_bps","mkt_prob"], ascending=[False, False])
    return prio.head(top_n).copy()

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

SUMMARY_TEMPLATE = """You're a knowledgeable friend helping a casual NFL fan find good player prop bets for this game.

Game: {matchup}

What our model sees:
- Market directions: {market_skew}
- Strongest edges: {top_edges}
- Confidence range: {conf_range}
- Best books for prices: {best_books}

IMPORTANT: Only mention markets that appear in the "Market directions" and "Strongest edges" data above. Do NOT mention or recommend markets like "longest reception", "longest rush", "first TD", "last TD", or any other markets not explicitly shown in the data. Stick strictly to the markets we actually model.

Write a friendly, conversational paragraph (3-5 sentences) that:
- Talks naturally about what looks interesting in this matchup (e.g., "I really like the RB unders here" or "The receiving props look solid")
- Names 1-3 specific players and why they stand out based ONLY on the markets shown in the data
- Gives honest advice about confidence level (e.g., "This one feels pretty solid" vs "There's more uncertainty here, so maybe go lighter")
- Only mentions shopping around if prices actually vary significantly across books
- Sounds like advice you'd give a friend, not a robot report

Avoid:
- Generic phrases like "pops most on our numbers" or "keep stakes modest"
- Overly formal language
- Bullet points or lists
- Mentioning any markets not in the provided data

Return only the conversational paragraph.
"""

def _prompt_for_weekly_overview(df: pd.DataFrame, season: int, week: int) -> str:
    """Generate prompt for week-at-a-glance overview."""
    # Top 5 edges across all games
    if {'player', 'market_std', 'edge_bps'}.issubset(df.columns):
        top = (df[['player','market_std','edge_bps']]
               .dropna()
               .assign(abs_edge=lambda x: x['edge_bps'].abs())
               .sort_values('abs_edge', ascending=False)
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

    # Check for arbitrage file
    arb_file = Path("data/qc/family_arbitrage.csv")
    arb_count = 0
    if arb_file.exists():
        try:
            arb_df = pd.read_csv(arb_file)
            arb_count = len(arb_df)
        except:
            pass

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
    Build a compact, model-led paragraph prompt for GPT-5, using the rows for this matchup.
    """
    df = rows.copy()

    # --- Market skew by over/under counts per market type
    if {'edge_bps', 'market_std'}.issubset(df.columns):
        skew = (df.assign(dir=np.where(df['edge_bps'] >= 0, 'over', 'under'))
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

    # --- Top absolute edges (player + market)
    if {'player', 'market_std', 'edge_bps'}.issubset(df.columns):
        top = (df[['player','market_std','edge_bps']]
               .dropna()
               .assign(abs_edge=lambda x: x['edge_bps'].abs())
               .sort_values('abs_edge', ascending=False)
               .head(3))
        if len(top):
            top_edges = "; ".join(f"{r.player} {r.market_std} ({int(r.edge_bps)} bps)"
                                  for _, r in top.iterrows())
        else:
            top_edges = "n/a"
    else:
        top_edges = "n/a"

    # --- Confidence spread (model_conf percentile band)
    if 'model_conf' in df.columns and df['model_conf'].notna().any():
        q05, q95 = df['model_conf'].quantile([0.05, 0.95]).tolist()
        conf_range = f"{q05:.0%}–{q95:.0%}"
    else:
        conf_range = "n/a"

    # --- Most frequent best-price books
    if 'best_book' in df.columns and df['best_book'].notna().any():
        counts = df['best_book'].value_counts().head(3).to_dict()
        best_books = ", ".join(f"{k}: {v}" for k, v in counts.items())
    else:
        best_books = "n/a"

    prompt = SUMMARY_TEMPLATE.format(
        matchup=game,
        market_skew=market_skew or "n/a",
        top_edges=top_edges or "n/a",
        conf_range=conf_range,
        best_books=best_books or "n/a",
    )
    return textwrap.dedent(prompt).strip()

def _call_llm(client, model, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,  # e.g., "gpt-5"
            messages=[
                {"role": "system", "content": "You're a knowledgeable friend who follows NFL closely and helps casual fans find smart player prop bets. You're analytical but conversational, honest about uncertainty, and avoid sounding robotic or overly formal."},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return ""



def _fallback_text(game: str, rows: pd.DataFrame) -> str:
    # Simple deterministic line if no LLM is available
    if rows.empty:
        return f"No clear model-backed edges for {game}; shop prices and size stakes modestly."
    lead = rows.iloc[0]
    mk = _pretty_market(lead.get("market_std"))
    side = str(lead.get("name",""))
    return f"{game}: {mk} {side} pops most on our numbers. Shop for best price; keep stakes modest."

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate per-game LLM commentary for Insights.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--merged_csv", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--model", type=str, default="gpt-5")
    ap.add_argument("--out_json", type=str, required=True, help="Output path for {game_norm: text} JSON")
    ap.add_argument("--force", action="store_true", help="Overwrite existing out_json")
    args = ap.parse_args(argv)

    if os.path.exists(args.out_json) and not args.force:
        print(f"[info] {args.out_json} exists. Use --force to overwrite.", file=sys.stderr)

    df = pd.read_csv(args.merged_csv, low_memory=False)
    df = _ensure_core(df)

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
        if top_edge is not None:
            weekly_overview = f"Week {args.week} features {len(games)} games with several interesting edges. Top edge: {top_edge.get('player', 'Unknown')} {top_edge.get('market_std', '')} at {int(top_edge.get('edge_bps', 0))} bps. Check individual games for details."
        else:
            weekly_overview = f"Week {args.week} features {len(games)} games. Select a game above to see detailed analysis."
    print(f"    ✓ Weekly overview complete", file=sys.stderr)

    out = {
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

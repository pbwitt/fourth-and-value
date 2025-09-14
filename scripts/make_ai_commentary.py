#!/usr/bin/env python3
# make_ai_commentary.py
# Produces per-game commentary for Insights. Writes JSON mapping: {game_norm: text}.
# Calls OpenAI if OPENAI_API_KEY + openai SDK are available; otherwise uses a deterministic fallback.

import argparse, json, os, sys, textwrap, math, re
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

SUMMARY_TEMPLATE = """You are producing a concise betting summary for one NFL game.
Write a single paragraph (3–5 sentences). Be specific and grounded in the data below.
Do NOT use bullet points or headings. Avoid generic phrases like "pops most on our numbers".

Game: {matchup}

Signals (model vs market):
- Market directions summary: {market_skew}
- Strongest player edges (abs bps): {top_edges}
- Confidence range (5th–95th pct of model_conf): {conf_range}
- Most frequent best-price books: {best_books}

Guidance:
- State the overall read (e.g., "RB unders dominate", "WR receptions overs cluster").
- Mention 1–3 specific players + markets that drive the read.
- Qualify uncertainty (e.g., "higher uncertainty—keep stakes modest" vs "tighter distribution").
- If prices differ materially by book, suggest shopping; otherwise don’t force it.
Return only the paragraph text.
"""

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
                {"role": "system", "content": "You are a sharp, data-driven NFL betting assistant."},
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

    out = {"games": []}  # <-- THIS is what build_insights_page.py expects

    for g in games:
        df_g = df[df["game_norm"] == g]
        top = _select_top_for_game(df_g, args.top_n)

        prompt = _prompt_for_game(g, df_g, args.season, args.week)
        print(f"[LLM] {g} ...", file=sys.stderr)

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

    print(f"[ok] wrote {args.out_json} with {len(out)} games.")

if __name__ == "__main__":
    main()

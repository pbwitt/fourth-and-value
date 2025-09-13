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

def _prompt_for_game(game: str, rows: pd.DataFrame, season: int, week: int) -> str:
    # Build a compact, model-led prompt the LLM can summarize quickly.
    bullets = []
    for _, r in rows.iterrows():
        player = r.get("player","")
        market = _pretty_market(r.get("market_std"))
        side   = r.get("name","")
        edge   = r.get("edge_bps", np.nan)
        mprob  = r.get("model_prob", np.nan)
        mpct   = f"{float(mprob):.0%}" if pd.notna(mprob) else ""
        if pd.notna(edge):
            bullets.append(f"- {player} — {market} {side} (edge {edge:.0f} bps; model {mpct})")
        else:
            mp = r.get("mkt_prob", np.nan)
            bullets.append(f"- {player} — {market} {side} (implied {float(mp):.0%})" if pd.notna(mp) else f"- {player} — {market} {side}")
    if not bullets:
        bullets = ["- No model-backed edges available; shop around or skip."]

    return textwrap.dedent(f"""\
        You are a sharp, numbers-first betting assistant. In 2–3 short sentences,
        summarize the best player prop opportunities for the NFL game: {game}.
        Mention the general theme (e.g., RB rushing overs), suggest shopping lines,
        and advise modest staking if uncertainty is high. Do not include emojis.

        Season: {season}, Week: {week}
        Candidate picks:
        {chr(10).join(bullets)}
    """).strip()

def _call_llm(client, model: str, prompt: str) -> str:
    """
    Single, modern call: chat.completions with max_completion_tokens.
    (Avoid max_tokens; some newer models reject it.)
    """
    if client is None:
        return ""
    sys_msg = "You write concise, neutral betting summaries."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=180,
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

    out: Dict[str, str] = {}  # <-- THIS is what build_insights_page.py expects

    for g in games:
        df_g = df[df["game_norm"] == g]
        top = _select_top_for_game(df_g, args.top_n)

        prompt = _prompt_for_game(g, top, args.season, args.week)
        print(f"[LLM] {g} ...", file=sys.stderr)

        text = _call_llm(client, args.model, prompt)
        if not text:
            text = _fallback_text(g, top)

        out[g] = text

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {args.out_json} with {len(out)} games.")

if __name__ == "__main__":
    main()

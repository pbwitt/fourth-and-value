#!/usr/bin/env python3
"""
Tweet Workbench API (private)
- /games: list games from merged CSV
- /picks?game=...: top picks for a game (deduped across books)
- /polish: LLM -> 3 tweet variants from chosen picks + prompt
- /queue: append tweets to data/social/tweet_queue.json
- /post: (optional) post immediately to X (Twitter) via tweepy
Guards: simple header X-Admin-Token must match ADMIN_TOKEN (env or .env)

Run:
  uvicorn scripts.tweet_workbench_api:app --reload --port 8787
"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timezone
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security


# ---------- config ----------
MERGED_CSV = os.getenv("TW_MERGED_CSV", "data/props/props_with_model_week2.csv")
QUEUE_PATH = Path(os.getenv("TW_QUEUE", "data/social/tweet_queue.json"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "dev-only-change-me")

# ---------- helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def pct(x) -> str:
    try:
        v = float(x)
        if v > 1: return f"{int(round(v))}%"
        return f"{int(round(v*100))}%"
    except Exception:
        return "NA"

def odds_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)): return ""
    s = str(v).strip()
    return s

def row_first(r: pd.Series, cols, default=None):
    for c in cols:
        if c in r and not pd.isna(r[c]): return r[c]
    return default

def infer_side(r: pd.Series) -> str:
    s = str(row_first(r, ["side","Side","Bet","bet"], "") or "").lower().strip()
    if s.startswith("over"):  return "OVER"
    if s.startswith("under"): return "UNDER"
    try:
        mp = float(row_first(r, ["model_prob","modelp"], float("nan")))
        mk = float(row_first(r, ["mkt_prob","market_prob","consensus_prob"], float("nan")))
        if not pd.isna(mp) and not pd.isna(mk):
            return "OVER" if mp >= mk else "UNDER"
    except Exception:
        pass
    return "OVER"

def pretty_market(s: str) -> str:
    if not isinstance(s, str): return "Market"
    x = s.replace("_"," ").strip()
    return x.title()

def norm_game(s: str) -> str:
    x = (s or "").lower().strip()
    x = x.replace(" vs. "," @ ").replace(" vs "," @ ").replace(" versus "," @ ")
    return " ".join(x.split())

def load_df() -> pd.DataFrame:
    df = pd.read_csv(MERGED_CSV)
    for c in [
        "Game","game","matchup","Matchup","player","Player","market_pretty","market_std","Market","market",
        "line_disp","Line","line","point","book","price","model_prob","mkt_prob","consensus_prob",
        "edge_bps","edge","ev_per_100","fair_odds","side","Bet","bet"
    ]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def dedupe_group(df: pd.DataFrame, book_col="book", price_col="price", max_books=3) -> pd.DataFrame:
    edge_col = "edge_bps" if "edge_bps" in df.columns else ("edge" if "edge" in df.columns else None)
    if edge_col is None:
        raise HTTPException(500, "CSV missing edge/edge_bps")
    if "line_disp" not in df.columns:
        df["line_disp"] = df.get("line", df.get("point", pd.NA))
    if "market_std" not in df.columns:
        df["market_std"] = df.get("market", df.get("Market", pd.NA))
    if "side" not in df.columns:
        df["side"] = df.apply(infer_side, axis=1)
    # one row per player×market×line×side; attach aggregated books
    df["_key"] = df.apply(
        lambda r: (
            str(row_first(r, ["player","Player"], "")),
            str(row_first(r, ["market_std","Market","market"], "")),
            str(row_first(r, ["line_disp","line","point"], "")),
            infer_side(r),
        ),
        axis=1,
    )
    rows = []
    for _, g in df.groupby("_key"):
        g = g.sort_values(edge_col, ascending=False)
        head = g.iloc[0].copy()
        parts = []
        for _, rr in g.iterrows():
            b = rr.get(book_col, pd.NA)
            if pd.isna(b) or not str(b).strip(): continue
            piece = str(b).strip()
            ps = odds_str(rr.get(price_col, pd.NA))
            if ps: piece += f" {ps}"
            parts.append(piece)
            if len(parts) >= max_books: break
        head["_books_joined"] = " | ".join(parts)
        rows.append(head)
    return pd.DataFrame(rows).sort_values(edge_col, ascending=False)

def make_fact(r: pd.Series, include_book=True, book_col="book", price_col="price") -> str:
    player = row_first(r, ["Player","player"], "—")
    market = pretty_market(str(row_first(r, ["market_pretty","market_std","Market","market"], "Market")))
    line   = row_first(r, ["line_disp","Line","line","point"], "NA")
    modelp = r.get("model_prob", pd.NA)
    mktp   = r.get("mkt_prob", pd.NA) if "mkt_prob" in r else pd.NA
    if pd.isna(mktp): mktp = r.get("consensus_prob", pd.NA)
    edge   = r.get("edge_bps", pd.NA) if "edge_bps" in r else r.get("edge", pd.NA)
    fair   = r.get("fair_odds", pd.NA)
    ev100  = row_first(r, ["ev_per_100","ev"], pd.NA)
    side   = infer_side(r)

    bits = [f"{player} — {side} {line} {market}"]
    if not pd.isna(modelp): bits.append(f"model {pct(modelp)}")
    if not pd.isna(mktp):   bits.append(f"market {pct(mktp)}")
    if not pd.isna(edge):
        try: bits.append(f"+{int(round(float(edge)))} bps")
        except Exception: bits.append(f"+{edge} bps")
    if not pd.isna(fair) and str(fair).strip(): bits.append(f"fair {str(fair).strip()}")
    if not pd.isna(ev100):
        try: bits.append(f"EV ${float(ev100):.0f}/$100")
        except Exception: pass
    if include_book:
        agg = str(r.get("_books_joined","") or "").strip()
        if agg: bits.append(f"Best: {agg}")
        else:
            b = r.get(book_col, pd.NA)
            if not pd.isna(b) and str(b).strip():
                piece = f"Best: {str(b).strip()}"
                ps = odds_str(r.get(price_col, pd.NA))
                if ps: piece += f" {ps}"
                bits.append(piece)
    head, *meta = bits
    return head if not meta else f"{head} ({'; '.join(meta)})"

# ---------- auth ----------
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)

def require_admin(x_admin_token: str = Security(api_key_header)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# ---------- models ----------
class PolishReq(BaseModel):
    game: str
    picks: List[str]  # fact lines (already formatted)
    model: str = "gpt-5"
    temperature: float = 0.7
    instruction: Optional[str] = None  # custom user prompt

class QueueReq(BaseModel):
    tweets: List[str]
    game: str

# ---------- app ----------
app = FastAPI(title="Tweet Workbench (private)")

@app.get("/games")
def list_games(auth: bool = Depends(require_admin)):
    df = load_df()
    gcol = "Game" if "Game" in df.columns else "game"
    games = sorted(set(str(x) for x in df[gcol].dropna().unique()))
    return {"games": games}

@app.get("/picks")
def picks_for_game(game: str, top: int = 10, include_book: bool = True,
                   auth: bool = Depends(require_admin)):
    df = load_df()
    gcol = "Game" if "Game" in df.columns else "game"
    sub = df[df[gcol].astype(str).map(norm_game) == norm_game(game)].copy()
    if sub.empty:
        return {"game": game, "picks": []}
    dedup = dedupe_group(sub)
    picks = []
    for _, r in dedup.head(top).iterrows():
        picks.append({
            "player": row_first(r, ["Player","player"], ""),
            "line": row_first(r, ["line_disp","line","point"], ""),
            "market": pretty_market(str(row_first(r, ["market_pretty","market_std","Market","market"], ""))),
            "side": infer_side(r),
            "edge_bps": r.get("edge_bps", r.get("edge", None)),
            "model_prob": r.get("model_prob", None),
            "mkt_prob": r.get("mkt_prob", r.get("consensus_prob", None)),
            "books": str(r.get("_books_joined","")),
            "fact": make_fact(r, include_book=include_book),
        })
    return {"game": game, "picks": picks}

@app.post("/polish")
def polish(req: PolishReq, auth: bool = Depends(require_admin)):
    facts_block = "\n".join("• " + f for f in req.picks)
    base_instr = (
        "Write exactly THREE distinct tweets (<=280 chars each). "
        "Keep ALL numbers/bets EXACT (OVER/UNDER, line, model/market %, edge, book odds). "
        "Start with the game tag (e.g., 'TB @ HOU'). Vary phrasing. "
        "Return ONLY:\n1) <tweet one>\n2) <tweet two>\n3) <tweet three>\n"
    )
    prompt = (req.instruction + "\n" if req.instruction else "") + base_instr + f"\nGame: {req.game}\nFacts:\n{facts_block}\n"
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.responses.create(
            model=req.model,
            input=prompt,
            max_completion_tokens=320,
            temperature=req.temperature,
        )
        text = (getattr(r, "output_text", "") or "").strip()
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    out = []
    for line in text.splitlines():
        line = line.strip()
        if line[:2] in {"1)", "2)", "3)"}:
            out.append(line[2:].strip())
    if not out:
        # fallback
        out = [f"{req.game}: " + facts_block.replace('• ', '').replace('\n', ' ')]
    return {"tweets": out[:3]}

@app.post("/queue")
def queue_tweets(req: QueueReq, auth: bool = Depends(require_admin)):
    q = []
    if QUEUE_PATH.exists():
        try:
            cur = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
            if isinstance(cur, list): q = cur
        except Exception:
            pass
    for t in req.tweets:
        q.append({"text": t, "game": req.game, "created_at": now_iso()})
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "queued": len(req.tweets), "path": str(QUEUE_PATH)}

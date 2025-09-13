#!/usr/bin/env python3
# build_insights_page.py
# Fourth & Value — Insights page builder (Week N)
# - No direct LLM calls here. Use scripts/make_ai_commentary.py to precompute LLM text.
# - Reads merged props CSV, selects top picks, renders summaries + cards with a JS game filter.

import argparse, json, os, re, html, sys
from datetime import datetime
import numpy as np
import pandas as pd

# ---------------------------
# Utilities & normalization
# ---------------------------

def _dedupe_for_bullets(df: pd.DataFrame) -> pd.DataFrame:
    """One bullet per (player, market, side); keep the highest-edge row."""
    if df.empty:
        return df
    # sort so the best edge is kept when dropping dupes
    df = df.sort_values("edge_bps", ascending=False)
    keys = [c for c in ("player", "market_std", "name") if c in df.columns]
    return df.drop_duplicates(subset=keys, keep="first")


def items_for_all_games(df: pd.DataFrame, games: list[str], k_per_game: int = 3) -> list[dict]:
    """Return up to k cards per game so the game filter never shows '1 card' unless that's all we have."""
    out = []
    for g in games:
        gdf = df[df["game_norm"] == g].copy()
        # prefer real edges; else fallback to implied-prob picks
        with_edge = gdf[gdf["edge_bps"].notna() & gdf["mkt_prob"].notna() & gdf["model_prob"].notna()]
        if not with_edge.empty:
            top = with_edge.sort_values("edge_bps", ascending=False).head(k_per_game)
        else:
            implied = gdf[gdf["mkt_prob"].notna()]
            top = implied.sort_values("mkt_prob", ascending=False).head(k_per_game)
        out.extend(top.to_dict(orient="records"))
    return out


def _slug(s: str) -> str:
    if not s: return ""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _is_nan_like(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def _coalesce(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def _pick_odds_col(obj):
    """Return the column name that holds American odds for either a DataFrame, Series(row), or dict."""
    if hasattr(obj, "columns"):
        cols = set(obj.columns)
    elif isinstance(obj, pd.Series):
        cols = set(obj.index)
    else:
        try:
            cols = set(obj.keys())
        except Exception:
            cols = set()
    for c in ("american_odds", "mkt_odds", "odds", "price"):
        if c in cols:
            return c
    return None

def fmt_pct(x):
    try:
        return f"{float(x):.0%}"
    except Exception:
        return ""

def clean_metric_label(row):
    """
    Prefer edge if present; else fall back to price-implied probability; else empty string.
    Works for dicts or Series.
    """
    get = (row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d))
    edge = get("edge_bps", np.nan)
    mkt_prob = get("mkt_prob", np.nan)
    model_prob = get("model_prob", np.nan)

    if pd.notna(edge):
        sign = "+" if edge >= 0 else ""
        return f"Edge: {sign}{edge:.0f} bps"
    if pd.notna(model_prob) and pd.notna(mkt_prob):
        delta = (float(model_prob) - float(mkt_prob)) * 10000
        sign = "+" if delta >= 0 else ""
        return f"Edge: {sign}{delta:.0f} bps"
    if pd.notna(mkt_prob):
        return f"Implied: {fmt_pct(mkt_prob)}"
    return ""

def odds_label(row):
    """Return 'Odds: +150' if American odds exist; else fall back to implied probability; else ''."""
    get = (row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d))
    col = _pick_odds_col(row)
    if col:
        v = get(col, np.nan)
        if pd.notna(v):
            try:
                iv = int(float(str(v)))
                return f"Odds: {iv:+d}"
            except Exception:
                pass
    mp = get("mkt_prob", np.nan)
    if pd.notna(mp):
        return f"Implied: {fmt_pct(mp)}"
    return ""

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
    if m is None:
        return ""
    return PRETTY_MAP.get(str(m), str(m))

def _ensure_probs_and_edge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize snake_case columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # aliases
    ALIASES = {
        "game": ["game", "matchup"],
        "player": ["player"],
        "market_std": ["market_std", "market", "bet"],
        "name": ["name", "side", "pick"],  # Over/Under/Yes/No
        "kick_et": ["kick_et", "kickoff", "kick", "kickoff_et"],
        "mkt_prob": ["mkt_prob", "market_prob", "consensus_prob", "book_implied_prob"],
        "model_prob": ["model_prob", "model_probability"],
        "edge_bps": ["edge_bps", "edge", "edge_bps_"],
        "american_odds": ["american_odds", "mkt_odds", "odds", "price"],
        "title": ["title"],  # optional
        "game_norm": ["game_norm"],  # optional
    }
    for std, cands in ALIASES.items():
        if std not in df.columns:
            for c in cands:
                if c in df.columns:
                    df.rename(columns={c: std}, inplace=True)
                    break

    # ensure probabilities are numeric
    for p in ("mkt_prob", "model_prob"):
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    # compute edge if missing
    if "edge_bps" not in df.columns or df["edge_bps"].isna().all():
        if {"model_prob", "mkt_prob"}.issubset(df.columns):
            df["edge_bps"] = (df["model_prob"] - df["mkt_prob"]) * 10000

    # compute EV per $100 if odds + model_prob exist
    odds_col = _pick_odds_col(df)
    if odds_col and "ev_per_100" not in df.columns:
        o = pd.to_numeric(df[odds_col], errors="coerce")
        payout = np.where(o > 0, o, 10000/(-o))  # $ return for $100 stake
        if "model_prob" in df.columns:
            df["ev_per_100"] = df["model_prob"] * payout - (1 - df["model_prob"]) * 100

    # ensure minimal fields
    for k in ("game", "player", "market_std", "name", "kick_et"):
        if k not in df.columns:
            df[k] = np.nan

    # derive a clean display title + normalized game label
    if "title" not in df.columns or df["title"].isna().all():
        pretty = df["market_std"].map(_pretty_market).fillna(df["market_std"])
        df["title"] = (df["player"].fillna("") + " — " + pretty.astype(str) + " " + df["name"].fillna("")).str.strip()

    if "game_norm" not in df.columns or df["game_norm"].isna().all():
        df["game_norm"] = df["game"]

    # a human kicker string
    if "kick_str" not in df.columns:
        df["kick_str"] = df["kick_et"].astype(str).replace("nan", "")

    return df

def _collapse_books_for_keys(df, keys=("game_norm","player","market_std","name")):
    """
    Keep one row per logical bet across books/dupes. Prefer max edge, then max EV.
    """
    df = df.copy()
    df["_ev_"] = df.get("ev_per_100", pd.Series(index=df.index, dtype=float))
    df = df.sort_values(["edge_bps","_ev_"], ascending=[False, False])
    out = df.groupby(list(keys), dropna=False, as_index=False).head(1)
    return out.drop(columns=["_ev_"], errors="ignore")

def prepare_overall(df, per_game_cap=2, top_n=10):
    """Overall Top picks: require model + market probs; collapse dupes; diversify by game."""
    valid = df[df["edge_bps"].notna() & df["mkt_prob"].notna() & df["model_prob"].notna()].copy()
    if valid.empty:
        return valid
    collapsed = _collapse_books_for_keys(valid)
    collapsed = collapsed.sort_values("edge_bps", ascending=False)
    # diversify across games
    diversified = collapsed.groupby("game_norm", group_keys=False).head(per_game_cap)
    return diversified.head(top_n).reset_index(drop=True)

def select_game_top(df, game, k=10):
    g = df[(df["game_norm"] == game) & df["edge_bps"].notna()]
    g = g.sort_values("edge_bps", ascending=False).head(k)
    return g.reset_index(drop=True)

# ---------------------------
# HTML pieces
# ---------------------------

def _card_html(it: dict) -> str:
    head = " · ".join([s for s in (it.get("kick_str",""), it.get("game_norm","")) if s])

    # derive game strings (escaped + slug)
    game_raw  = it.get("game_norm","") or it.get("game","") or ""
    game_attr = html.escape(game_raw)
    slug      = _slug(game_raw)


    title = html.escape(it.get("title","") or "")

    # Clean meta (no '—')
    meta_txt = " · ".join([x for x in (clean_metric_label(it), odds_label(it)) if x])
    meta_txt = html.escape(meta_txt)

    # Blurb (only include edge text if present)
    player = it.get("player","")
    bet    = _pretty_market(_coalesce(it.get("bet_pretty"), it.get("market_pretty"),
                                      it.get("market_std"), it.get("market")))
    side   = it.get("name","")
    ao     = it.get(_pick_odds_col(it) or "american_odds")
    ao_s   = ""
    if not _is_nan_like(ao):
        s = str(ao).strip()
        if s.lstrip("+-").isdigit():
            ao_s = f" ({int(float(s)):+d})"

    edge_phrase = clean_metric_label(it)
    edge_phrase = f" {edge_phrase}." if edge_phrase else ""
    blurb = f"Our numbers like {player} {bet} {side}{ao_s}.{edge_phrase}"
    blurb = html.escape(blurb)

    return (
        f'<div class="card" data-game="{game_attr}" data-slug="{slug}">'
        f'<div class="kicker">{html.escape(head)}</div>'
        f'<h3 class="title">{title}</h3>'
        f'<div class="meta">{meta_txt}</div>'
        f'<div class="blurb">{blurb}</div>'
        '</div>'
    )

def _compose_summaries(df: pd.DataFrame, games: list, llm_text: dict | None = None, k_per_game=10) -> str:
    parts = ['<section class="summaries"><h2>Summaries (Top 10 per game)</h2>']
    for g in games:
        slug = _slug(g)
        picks = select_game_top(df, g, k=k_per_game)
        picks = _dedupe_for_bullets(picks)

        parts.append(f'<section class="summary-block" data-slug="{slug}">')
        parts.append(f'<h3>{html.escape(g)}</h3>')

        if llm_text and g in llm_text and llm_text[g]:
            parts.append(f"<p class='muted'>{html.escape(llm_text[g].strip())}</p>")
        else:
            parts.append(f"<p class='muted'>Top opportunities for <strong>{html.escape(g)}</strong>: {len(picks)} picks stand out. Shop prices; size modestly.</p>")

        bullets = []
        for _, r in picks.iterrows():
            d = r.to_dict()
            player = _coalesce(d.get("player"))
            bet    = _pretty_market(_coalesce(d.get("bet_pretty"), d.get("market_pretty"), d.get("market_std")))
            side   = _coalesce(d.get("name"))
            meta   = " — ".join([x for x in (clean_metric_label(d), odds_label(d)) if x])
            meta_s = f" <span class='muted'>[{html.escape(meta)}]</span>" if meta else ""
            bullets.append(f"<li><strong>{html.escape(str(player))}</strong> — {html.escape(str(bet))} {html.escape(str(side))}{meta_s}</li>")
        parts.append("<ul>\n" + "\n".join(bullets) + "\n</ul>")
        parts.append("</section>")
    parts.append("</section>")
    return "\n".join(parts)

def _build_cards_grid(items_overall: list[dict]) -> str:
    if not items_overall:
        return "<p>No model-backed edges available yet.</p>"
    cards = [_card_html(it) for it in items_overall]
    return '<section class="cards"><div class="grid">' + "\n".join(cards) + "</div></section>"

def _page_css() -> str:
    return """
<style>
:root { color-scheme: dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; background:#0b1220; color:#e6ebf5; margin:0; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 16px 20px 40px; }
h1 { font-size: 28px; margin: 16px 0 8px; }
h2 { font-size: 18px; margin: 16px 0 8px; color:#cbd5e1; }
h3 { font-size: 16px; margin: 18px 0 6px; }
.muted { color: #93a4bf; }
select { background:#0f172a; color:#e6ebf5; border:1px solid #3b4252; padding:8px 10px; border-radius:10px; }
.cards .grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(300px,1fr)); gap:14px; margin-top: 10px;}
.card { border:1px solid rgba(255,255,255,0.08); background:#0f172a; border-radius:16px; padding:14px; }
.card .kicker { color:#9aa7bd; font-size:13px; }
.card .title { font-size:18px; margin:6px 0 4px; }
.card .meta { color:#9aa7bd; font-size:13px; }
.card .blurb { color:#cbd5e1; font-size:14px; margin-top:6px; }
.topbar { display:flex; align-items:center; gap:10px; border-top:1px solid rgba(255,255,255,0.08); padding-top:12px; margin-top:12px;}
.count { color:#9aa7bd; font-size:13px; }
hr { border:0; border-top:1px solid rgba(255,255,255,0.08); margin:14px 0; }
#nav-root { margin-bottom: 4px; }
</style>
"""

def _page_js() -> str:
    # filters cards by data-slug
    return """
<script>
document.addEventListener('DOMContentLoaded', () => {
  const sel       = document.getElementById('game-filter');
  const cards     = Array.from(document.querySelectorAll('.card[data-slug]'));
  const summaries = Array.from(document.querySelectorAll('.summary-block[data-slug]'));
  const count     = document.getElementById('count');
  const empty     = document.getElementById('empty-state');

  function apply() {
    const v = sel.value;
    let shown = 0;

    for (const c of cards) {
      const show = (v === '__all__') || (c.dataset.slug === v);
      c.style.display = show ? '' : 'none';
      if (show) shown++;
    }
    for (const s of summaries) {
      const show = (v === '__all__') || (s.dataset.slug === v);
      s.style.display = show ? '' : 'none';
    }

    if (count) count.textContent = shown;
    if (empty) empty.style.display = shown ? 'none' : '';
  }

  sel.addEventListener('change', apply);
  apply();
});
</script>

"""

def build_html(title: str, games: list[str], items_overall: list[dict], summaries_html: str) -> str:
    # Build <option>s only for games that actually have cards
    games_in_cards = sorted({
        (it.get("game_norm") or it.get("game") or "").strip()
        for it in items_overall
        if (it.get("game_norm") or it.get("game"))
    })
    opts = ['<option value="__all__">All games</option>'] + [
        f'<option value="{_slug(g)}">{html.escape(g)}</option>' for g in games_in_cards
    ]

    cards_html = _build_cards_grid(items_overall)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
{_page_css()}
</head>
<body>
<div id="nav-root"></div>
<script src="../nav.js?v=6"></script>

<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <p class="muted">AI-assisted commentary on our top edges. We keep it short, factual, and grounded in the numbers.</p>

  <div class="topbar">
    <label for="game-filter" class="muted">Filter:</label>
    <select id="game-filter">{''.join(opts)}</select>
    <span class="count muted"><span id="count">{len(items_overall)}</span> cards</span>
  </div>
  <div id="empty-state" class="muted" style="display:none;margin:8px 0 0;">
    No model-backed picks for that game in this Top list. Try another game, or see the summaries below.
  </div>

  <hr/>

  {summaries_html}

  <hr/>

  {cards_html}
</div>

<script>
document.addEventListener('DOMContentLoaded', () => {{
  const sel   = document.getElementById('game-filter');
  const cards = Array.from(document.querySelectorAll('.card[data-slug]'));
  const count = document.getElementById('count');
  const empty = document.getElementById('empty-state');

  function apply() {{
    const v = sel.value;
    let shown = 0;
    for (const c of cards) {{
      const show = (v === '__all__') || (c.dataset.slug === v);
      c.style.display = show ? '' : 'none';
      if (show) shown++;
    }}
    if (count) count.textContent = shown;
    if (empty) empty.style.display = shown ? 'none' : '';
  }}
  sel.addEventListener('change', apply);
  apply();
}});
</script>
</body>
</html>
"""


# ---------------------------
# Main
# ---------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Build Insights page (no direct LLM calls).")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--merged_csv", type=str,
                   help="Merged props CSV; defaults to data/props/props_with_model_week{week}.csv")
    p.add_argument("--title", type=str, help="Page title (default: Fourth & Value — Insights (Week N))")
    p.add_argument("--out", type=str, default="docs/props/insights.html")
    p.add_argument("--ai_json", type=str,
                   help="Optional JSON cache produced by make_ai_commentary.py (per-game commentary).")
    args = p.parse_args(argv)

    merged_path = args.merged_csv or f"data/props/props_with_model_week{args.week}.csv"
    if not os.path.exists(merged_path):
        print(f"[error] merged CSV not found: {merged_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(merged_path, low_memory=False)
    df = _ensure_probs_and_edge(df)

    # games list (stable order by first appearance)
    games = list(pd.unique(df["game_norm"].dropna()))

    # Overall picks (diversified)
    # BEFORE (remove these two lines)
# df_overall = prepare_overall(df, per_game_cap=2, top_n=10)
# items_overall = [r.to_dict() for _, r in df_overall.iterrows()]

# AFTER (add these)
    games = list(pd.unique(df["game_norm"].dropna()))
    items_overall = items_for_all_games(df, games, k_per_game=3)  # tweak 3 -> 2/4 to taste

    # Optional LLM commentary (precomputed)
    llm_text = None
    src = args.ai_json
    if not src:
        # try a sensible default if present
        cand = f"data/ai/insights_cache_week{args.week}.json"
        if os.path.exists(cand):
            src = cand
    if src and os.path.exists(src):
        try:
            with open(src, "r", encoding="utf-8") as f:
                llm_text = json.load(f)
            print(f"[info] loaded LLM commentary from {src} ({len(llm_text)} games).")
        except Exception as e:
            print(f"[warn] failed to load LLM json: {e}", file=sys.stderr)

    summaries_html = _compose_summaries(df, games, llm_text=llm_text, k_per_game=10)
    title = args.title or f"Fourth & Value — Insights (Week {args.week})"
    page = build_html(title, games, items_overall, summaries_html)

    # Write with nav helper if available
    try:
        from site_common import write_with_nav
        write_with_nav(args.out, page, active="Insights")
        print(f"[ok] wrote {args.out} (with nav).")
    except Exception:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"[ok] wrote {args.out}.")

if __name__ == "__main__":
    main()

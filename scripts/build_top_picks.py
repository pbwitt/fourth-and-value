#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, re, math
from html import escape
from pathlib import Path
from site_common import inject_nav
from pathlib import Path
from site_common import write_with_nav_raw
 # add this
# ---- shared helpers / branding ----
def collapse_best_across_lines_and_books(df):
    """
    ONE card per (game_id, player, market_std, side), ignoring line.
    Choose the bettor-friendliest row by EV, then edge, then price.
    """
    import pandas as pd, re
    d = df.copy()

    # canonical player: first initial + last (handles Cam/Cameron, Jr/Sr/III, etc.)
    def _player_canon(row):
        s = str(row.get("player") or row.get("player_key") or "").lower()
        s = re.sub(r"[^\w\s\.]", " ", s)
        s = re.sub(r"\b(jr|sr|iii|ii|iv)\b\.?", "", s)
        toks = [t for t in s.split() if t]
        if not toks: return ""
        return f"{toks[0][0]}.{toks[-1]}"

    d["player_canon"] = d.apply(_player_canon, axis=1)
    d["name"] = d["name"].astype(str).str.strip().str.lower()

    # prefer game_id; fallback to game_disp
    if "game_id" in d.columns:
        d["_game_key"] = d["game_id"].astype(str).str.strip().str.lower()
    else:
        d["_game_key"] = d["game_disp"].astype(str).str.strip().str.lower()

    # ranking fields
    d["_ev"]    = pd.to_numeric(d.get("ev_per_100"), errors="coerce").fillna(-1e15)
    d["_edge"]  = pd.to_numeric(d.get("edge_bps"),   errors="coerce").fillna(-1e15)
    d["_price"] = pd.to_numeric(d.get("price"),      errors="coerce").fillna(-1e15)

    # best first → drop dups on the line-agnostic key
    d.sort_values(
        ["_game_key","player_canon","market_std","name","_ev","_edge","_price"],
        ascending=[ True,        True,          True,  True,  False,  False,  False],
        inplace=True,
    )
    d = d.drop_duplicates(subset=["_game_key","player_canon","market_std","name"], keep="first")
    return d.drop(columns=["_ev","_edge","_price","_game_key"], errors="ignore")


def _is_numeric_total_market(mkt) -> bool:
    """Markets where 'Over/Under <number>' is expected (not Yes/No)."""
    m = pretty_market(mkt or "").lower()
    if not m: return False
    # broad keywords + common exact names
    if "yards" in m:  # passing/receiving/rushing yards, etc.
        return True
    numeric_exact = {
        "receptions", "rush attempts", "pass attempts",
        "completions", "passing touchdowns", "rushing attempts",
    }
    return m in numeric_exact


import re, math  # ensure both imported

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_numberish(x):
    """Extract a usable float from messy strings like '7,410 bps', '74.1%', ' +150 '."""
    if x is None or (isinstance(x, float) and math.isnan(x)): return np.nan
    s = str(x).replace(",", " ").strip()
    m = NUM_RE.search(s)
    if not m: return np.nan
    v = float(m.group(0))
    # treat percentages like 74.1% → 0.741
    if "%" in s and v > 1: v = v / 100.0
    return v

def _prob01(x):
    """Normalize probs to [0,1] from decimals or percents."""
    v = parse_numberish(x)
    if not isinstance(v, float) or math.isnan(v): return np.nan
    if 0 <= v <= 1: return v
    if 1 < v <= 100: return v / 100.0
    return np.nan

def american_to_prob(o):
    """Implied probability from American odds (no vig)."""
    v = parse_numberish(o)
    if not isinstance(v, float) or math.isnan(v): return np.nan
    return 100.0/(v+100.0) if v > 0 else (-v)/((-v)+100.0)

def _format_pct(x):
    p = _prob01(x)
    if not isinstance(p, float) or math.isnan(p) or p <= 0 or p >= 1: return ""
    return f"{p*100:.1f}%"

def _ev_per_100(prob, american_odds):
    p = _prob01(prob)
    if not isinstance(p, float) or math.isnan(p) or p <= 0 or p >= 1: return ""
    v = parse_numberish(american_odds)
    if not isinstance(v, float) or math.isnan(v): return ""
    win_profit = (v/100.0)*100.0 if v > 0 else (100.0/abs(v))*100.0
    ev = p*win_profit - (1-p)*100.0
    return f"${ev:.2f}" if ev >= 0 else f"-${abs(ev):.2f}"






try:
    from scripts.site_common import nav_html, pretty_market, fmt_odds_american, kickoff_et, BRAND
except Exception:
    from site_common import nav_html, pretty_market, fmt_odds_american, kickoff_et, BRAND  # fallback

# big render cap; UI defaults to Top N=10 so this won't overwhelm the page
CARD_LIMIT = 25000

# columns we’ll scan for the numeric line if line_disp is missing
LINE_CANDIDATES = [
    "line_disp","point","line","market_line","prop_line","number","threshold","total","line_number",
    "handicap","spread","yards","receptions","receiving_yards","rushing_yards","passing_yards","prop_total"
]
GAME_CANDIDATES = ["game","Game","matchup","matchup_name","matchup_display"]

def _num(x):
    with np.errstate(all="ignore"):
        return pd.to_numeric(x, errors="coerce")

def _norm(s: str) -> str:
    """normalize for filtering: lowercase, collapse spaces, trim"""
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _first_nonnull(row, cols):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return row[c]
    return np.nan

def _format_pct(x):
    if x is None or (isinstance(x,float) and (math.isnan(x) or x<0 or x>1)): return ""
    return f"{x*100:.1f}%"

def _ev_per_100(prob, american_odds):
    """Expected profit per $100 stake using prob in [0,1] and American odds."""
    if prob is None or (isinstance(prob,float) and (math.isnan(prob) or prob<=0 or prob>=1)): return ""
    try:
        o = float(american_odds)
    except Exception:
        return ""
    win_profit = (o/100.0)*100.0 if o>0 else (100.0/abs(o))*100.0  # +150→150; -110→90.909...
    ev = prob*win_profit - (1.0-prob)*100.0
    sign = "-" if ev < 0 else ""
    return f"${abs(ev):.2f}"

def read_df(path: str):
    """
    Read the merged props+models CSV and normalize it for Top Picks rendering.
    Keeps prior behaviors: prob-column detection, edge backfill, line/bet/game display,
    helper norm columns, and attrs for renderer.
    """
    import pandas as pd, numpy as np, re, math

    # ---------- 1) Read first ----------
    df = pd.read_csv(path, low_memory=False)

    # After: df = pd.read_csv(path, low_memory=False)  and other normalizations…

# --- kickoff time in ET for the card header ---
    if "kick_et" not in df.columns or df["kick_et"].isna().all():
        if "commence_time" in df.columns:
            t = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
            try:
                df["kick_et"] = t.dt.tz_convert("US/Eastern").dt.strftime("%a %-I:%M %p ET")
            except Exception:
                # Some pandas builds use "America/New_York"
                df["kick_et"] = t.dt.tz_convert("America/New_York").dt.strftime("%a %-I:%M %p ET")
        else:
            df["kick_et"] = ""


    # ---------- 2) Column hygiene ----------
    df.columns = [c.strip() for c in df.columns]

    # Map common variants → expected names (display code expects 'book')
    rename_map = {}
    if "bookmaker" in df.columns and "book" not in df.columns:
        rename_map["bookmaker"] = "book"
    if "sportsbook" in df.columns and "book" not in df.columns:
        rename_map["sportsbook"] = "book"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ---------- 3) Keys & core fields (with fallbacks) ----------
    # player_key
    if "player_key" not in df.columns:
        src = next((c for c in ["player_key","player","player_name","athlete","name"] if c in df.columns), None)
        if not src:
            raise SystemExit(f"{path} missing 'player_key' and no fallback cols; have: {list(df.columns)[:25]}")
        df["player_key"] = df[src].astype(str).str.strip().str.lower()
    else:
        df["player_key"] = df["player_key"].astype(str).str.strip().str.lower()

    # market_std (fallback from 'market')
    if "market_std" not in df.columns and "market" in df.columns:
        MAP = {
            "player_rush_yds": "rush_yds",
            "player_receiving_yds": "recv_yds",
            "player_receptions": "receptions",
            "player_passing_yds": "pass_yds",
            "player_passing_tds": "pass_tds",
            "player_interceptions": "pass_interceptions",
            "player_anytime_td": "anytime_td",
        }
        s = df["market"].astype(str).str.strip().str.lower()
        df["market_std"] = s.map(MAP).fillna(s)

    # side/selection
    if "name" not in df.columns:
        df["name"] = df.get("selection", "")
    df["name"] = df["name"].astype(str).str.strip().str.lower()

    # numerics
    df["point"]        = pd.to_numeric(df.get("point", df.get("line", df.get("points"))), errors="coerce")
    df["price"]        = pd.to_numeric(df.get("price"), errors="coerce")
    df["model_prob"]   = pd.to_numeric(df.get("model_prob"), errors="coerce")
    df["mkt_prob"]     = pd.to_numeric(df.get("mkt_prob", df.get("consensus_prob", df.get("market_prob"))), errors="coerce")
    df["edge_bps"]     = pd.to_numeric(df.get("edge_bps"), errors="coerce")
    df["ev_per_100"]   = pd.to_numeric(df.get("ev_per_100"), errors="coerce")

    # ---------- 4) Prob column detection & edge backfill (if needed) ----------
    prob_cols = {c.lower(): c for c in df.columns if c.lower().endswith("_prob") or c.lower() in ("model_prob","mkt_prob","market_prob","consensus_prob")}
    model_prob_col     = prob_cols.get("model_prob")
    consensus_prob_col = prob_cols.get("mkt_prob") or prob_cols.get("market_prob") or prob_cols.get("consensus_prob")

    if "edge_bps" not in df.columns or df["edge_bps"].isna().all():
        if model_prob_col is not None and consensus_prob_col is not None:
            mp = pd.to_numeric(df[model_prob_col], errors="coerce")
            cp = pd.to_numeric(df[consensus_prob_col], errors="coerce")
            df["edge_bps"] = 10000.0 * (mp - cp)
            df.attrs["_edge_label"] = "Consensus edge"
        elif model_prob_col is not None and "price" in df.columns:
            def _american_to_prob(o):
                if pd.isna(o): return np.nan
                try: o = float(o)
                except Exception: return np.nan
                return 100.0/(o+100.0) if o > 0 else (-o)/((-o)+100.0)
            mp = pd.to_numeric(df[model_prob_col], errors="coerce")
            bp = df["price"].apply(_american_to_prob)
            df["edge_bps"] = 10000.0 * (mp - bp)
            df.attrs["_edge_label"] = "Book edge"

    df.attrs["model_prob_col"] = model_prob_col
    df.attrs["consensus_prob_col"] = consensus_prob_col

    # ---------- 5) Game display/id (for leg keys + UI) ----------
    GAME_CANDIDATES = ["game_disp","game","matchup","game_name"]
    if "game_disp" not in df.columns:
        def _mk_game(row):
            for c in GAME_CANDIDATES:
                if c in df.columns:
                    val = row.get(c)
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()
            away = str(row.get("away_team") or "").strip()
            home = str(row.get("home_team") or "").strip()
            if away and home: return f"{away} @ {home}"
            return away or home or ""
        df["game_disp"] = df.apply(_mk_game, axis=1)

    if "game_id" not in df.columns:
        df["game_id"] = (
            df.get("home_team","").astype(str).str.lower().str.replace(r"\s+","_", regex=True) + "_" +
            df.get("away_team","").astype(str).str.lower().str.replace(r"\s+","_", regex=True) + "_" +
            df.get("commence_time","").astype(str).str[:19]
        )

    # ---------- 6) Line display & bet text ----------
    LINE_CANDIDATES = ["point","line","points","model_line"]
    def _first_nonnull(r, cols):
        for c in cols:
            if c in r and pd.notna(r[c]):
                return r[c]
        return np.nan

    def mk_line_disp(r):
        if "line_disp" in r and str(r["line_disp"]).strip():
            return str(r["line_disp"]).strip()
        raw_line = _first_nonnull(r, LINE_CANDIDATES)
        line_txt = ""
        if pd.notna(raw_line):
            try:    line_txt = f"{float(raw_line):g}"
            except: line_txt = str(raw_line).strip()
        side_raw = str(r.get("name") or r.get("side") or "").strip()
        side = side_raw
        # If numeric-total market and we have a number, normalize Yes/No → Over/Under
        try:
            is_num = _is_numeric_total_market(r.get("market"))
        except Exception:
            is_num = False
        if line_txt and is_num:
            if side_raw.lower() == "yes": side = "Over"
            elif side_raw.lower() == "no": side = "Under"
        return f"{side} {line_txt}".strip()

    if "line_disp" not in df.columns:
        df["line_disp"] = df.apply(mk_line_disp, axis=1)

    def mk_bet(r):
        m = r.get("market") or r.get("market_std")
        try:
            m2 = pretty_market(m) if m is not None else ""
        except Exception:
            m2 = str(m or "")
        return f"{m2} — {r.get('line_disp','').strip()}".strip(" —")
    if "bet" not in df.columns:
        df["bet"] = df.apply(mk_bet, axis=1)

    # ---------- 7) Normalized helper columns ----------
    def _norm(s):
        return re.sub(r"[^a-z0-9]+"," ", str(s or "").lower()).strip()
    if "market" not in df.columns and "market_std" in df.columns:
        df["market"] = df["market_std"]
    if "book" not in df.columns:
        df["book"] = df.get("bookmaker","")
    df["_mkt_norm"]  = df["market"].apply(lambda m: _norm(m))
    df["_game_norm"] = df["game_disp"].apply(_norm)
    df["_book_norm"] = df["book"].apply(_norm)

    # ---------- 8) Ensure presence of renderer-used columns ----------
    for col in ["player","market","game_disp","book","line_disp","price","edge_bps"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ---- rendering ----
def card(row, model_prob_col, consensus_prob_col):
    player = escape(str(row.get("player","")))
    mkt_lbl = pretty_market(row.get("market",""))
    book    = str(row.get("book",""))
    odds    = fmt_odds_american(row.get("price"))
    line_d  = str(row.get("line_disp",""))
    edge    = row.get("edge_bps", np.nan)
    game    = str(row.get("game_disp",""))
    kick    = kickoff_et(row.get("kick_et",""))

    # probs (robust to 0–1 or 0–100 inputs)
    model_prob = row.get(model_prob_col) if model_prob_col else np.nan
    cons_prob  = row.get(consensus_prob_col) if consensus_prob_col else np.nan
    model_prob_txt = _format_pct(model_prob)
    cons_prob_txt  = _format_pct(cons_prob)

    # EV per $100 using model prob & american odds
    ev_txt = _ev_per_100(model_prob if isinstance(model_prob,(int,float,str)) else np.nan, row.get("price"))

    # Optional model line (only if sane)
    model_line = row.get("model_line", np.nan)
    show_model_line = isinstance(model_line, (int, float)) and not math.isnan(model_line) and 0 < float(model_line) < 300
    model_line_txt = f"{float(model_line):g}" if show_model_line else ""

    # normalized attrs for robust filtering
    data_attrs = f'data-market="{escape(_norm(mkt_lbl))}" data-game="{escape(_norm(game))}" data-book="{escape(_norm(book))}"'

    # bet text
    bet_parts = []
    if line_d: bet_parts.append(line_d)                      # "Under 73.5" or "Yes"
    if odds:   bet_parts.append(f"@ {odds}")                 # "@ -110"
    if book:   bet_parts.append(f"on {book}")                # "on BetMGM"
    bet_txt = " ".join(bet_parts).strip()

    edge_txt = "" if (edge is None or (isinstance(edge,float) and math.isnan(edge))) else f"{edge:,.0f} bps"

    return f"""
    <div class="card" {data_attrs} data-edge="{'' if (edge is None or (isinstance(edge,float) and math.isnan(edge))) else f'{float(edge):g}'}">
      <div class="meta">
        <span class="time">{escape(str(kick))}</span>
        <span class="dot">•</span>
        <span class="game">{escape(game)}</span>
      </div>

      <div class="headline">
        <span class="player">{escape(player)}</span>
        <span class="dash">—</span>
        <span class="market">{escape(mkt_lbl)}</span>
      </div>

      <div class="betline">Bet: {escape(bet_txt)}</div>

      <div class="kvgrid">
        <div>Model prob</div><div>{escape(model_prob_txt)}</div>
        <div>Consensus prob</div><div>{escape(cons_prob_txt)}</div>
        <div>Consensus edge</div><div>{escape(edge_txt)}</div>
        <div>EV / $100</div><div>{escape(ev_txt)}</div>
      </div>

      <div class="footer">
        <button class="copy" onclick="copyCard(this)">Copy bet</button>
        <div class="right">
          {"<span class='modelline'>Model: " + escape(model_line_txt) + "</span>" if model_line_txt else ""}
          <span class="bestbook">{"Best book: " + escape(book) if book else ""}</span>
        </div>
      </div>
    </div>
    """


def _opts_from_pairs(pairs):
    out = ['<option value="">All</option>']
    for val, lbl in pairs:
        out.append(f'<option value="{escape(val)}">{escape(lbl)}</option>')
    return "\n".join(out)

def html_page(cards_html, title, market_pairs, game_pairs, book_pairs, week=None, season=None):
    import json
    market_opts = _opts_from_pairs(market_pairs)
    game_opts   = _opts_from_pairs(game_pairs)
    book_opts   = _opts_from_pairs(book_pairs)
    book_opts_js = json.dumps(book_pairs)  # JavaScript array for checkbox population

    # Week header
    week_header = ""
    if week:
        if season:
            week_header = f'<h1 style="margin:0 0 12px;font-size:22px;font-weight:700;letter-spacing:.2px;color:#fff;">Week {week}, {season}</h1>'
        else:
            week_header = f'<h1 style="margin:0 0 12px;font-size:22px;font-weight:700;letter-spacing:.2px;color:#fff;">Week {week}</h1>'

    consensus_note = (
        "Note: <b>market consensus</b> is the aggregated market view "
        "(e.g., average/median de-vig price/line across books). "
        "We surface edges vs that consensus and vs each book."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{escape(title)}</title>
<link rel="icon" href="data:,">
<style>
:root {{ color-scheme: dark }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#0b0b0c; color:#e7e7ea; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Ubuntu,Helvetica,Arial,sans-serif; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 18px 16px 32px; }}
.h1 {{ font-size: clamp(22px,3.5vw,28px); font-weight:900; color:#fff; margin: 4px 0 10px; }}

/* Controls */
.controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 10px; align-items:end; }}
label {{ display:flex; flex-direction:column; gap:4px; font-size:12px; color:#b7b7bb; }}
input,select {{ background:#111113; border:1px solid #232327; color:#e7e7ea; border-radius:10px; padding:8px 10px; min-width:110px; }}
button.badge {{ font-size:12px; background:#2a63ff; border:none; color:#fff; padding:8px 12px; border-radius:10px; cursor:pointer; }}
button.reset {{ background:#1a1a1d; border:1px solid #2a2a2e; color:#e7e7ea; }}
.note {{ margin:6px 0 16px; color:#b7b7bb; font-size:13px; }}

/* Card grid (responsive: 1-up → 2-up → 3-up) */
#list {{
  display: grid;
  grid-template-columns: 1fr;      /* mobile: 1-up */
  gap: 12px;
}}
@media (min-width: 760px) {{
  #list {{ grid-template-columns: repeat(2, 1fr); }}   /* tablet: 2-up */
}}
@media (min-width: 1100px) {{
  #list {{ grid-template-columns: repeat(3, 1fr); }}   /* desktop: 3-up */
}}

/* Card — compact, mobile-first */
.card {{
  background:#111113; border:1px solid #1f1f22; border-radius:16px;
  padding:12px 12px; display:flex; flex-direction:column; gap:6px;
}}
.meta {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; color:#8a8a90; font-size:12px; }}
.meta .dot {{ opacity:.6; }}
.headline {{ display:flex; flex-wrap:wrap; gap:6px; align-items:baseline; }}
.player {{ color:#fff; font-weight:800; font-size:15px; }}
.market {{ color:#c8c8cd; font-size:14px; }}
.betline {{ color:#e3e3e6; font-size:14px; }}

/* Tight stats grid (no big spacing) */
.kvgrid {{
  display: grid;
  grid-template-columns: max-content max-content;
  column-gap: 10px;
  row-gap: 2px;
  align-items: baseline;
  font-size: 13px; color: #d6d6d9;
}}
.kvgrid > div:nth-child(odd) {{ color:#b7b7bb; }}  /* labels */

.footer {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:4px; }}
.copy {{ background:#2a63ff; color:#fff; border:none; border-radius:10px; padding:8px 12px; cursor:pointer; }}
.footer .right {{ display:flex; gap:12px; align-items:center; }}
.bestbook {{ color:#b7b7bb; font-size:12px; }}
.modelline {{ color:#b7b7bb; font-size:12px; }}

/* Wider container on large screens (pairs nicely with 3-up grid) */
@media (min-width: 900px) {{
  .container {{ max-width: 1100px; }}
}}

/* Checkbox dropdown for book filter */
.checkbox-dropdown {{
  position: relative;
  min-width: 180px;
}}
.checkbox-dropdown-button {{
  width: 100%;
  background: #111113;
  color: #e7e7ea;
  border: 1px solid #232327;
  border-radius: 10px;
  padding: 8px 10px;
  outline: none;
  cursor: pointer;
  text-align: left;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
}}
.checkbox-dropdown-button:hover {{
  border-color: #6ee7ff;
}}
.checkbox-dropdown-button::after {{
  content: '▼';
  font-size: 10px;
  opacity: 0.6;
}}
.checkbox-dropdown.open .checkbox-dropdown-button::after {{
  content: '▲';
}}
.checkbox-group {{
  display: none;
  position: absolute;
  z-index: 1000;
  flex-direction: column;
  gap: 6px;
  background: #111113;
  border: 1px solid #232327;
  border-radius: 10px;
  padding: 10px;
  max-height: 250px;
  overflow-y: auto;
  min-width: 100%;
  margin-top: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,.5);
}}
.checkbox-dropdown.open .checkbox-group {{
  display: flex;
}}
.checkbox-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 4px 6px;
  border-radius: 6px;
  transition: background 0.15s;
}}
.checkbox-item:hover {{
  background: rgba(255,255,255,.05);
}}
.checkbox-item input[type="checkbox"] {{
  width: 16px;
  height: 16px;
  cursor: pointer;
  accent-color: #34d399;
}}
.checkbox-item label {{
  cursor: pointer;
  flex: 1;
  font-size: 13px;
  color: #e7e7ea;
  margin: 0;
  white-space: nowrap;
}}
</style>



</head>
<body>

<main class="container">
  {week_header}
  <div class="h1">{escape(title)}</div>

  <div class="controls">
    <label>Min edge (bps)
      <input id="minEdge" type="number" value="0" step="10">
    </label>
    <label>Top N
      <input id="topN" type="number" value="10" step="10">
    </label>
    <label>Market
      <select id="marketFilter">{market_opts}</select>
    </label>
    <label>Game
      <select id="gameFilter">{game_opts}</select>
    </label>
    <label>My Books
      <div id="bookDropdown" class="checkbox-dropdown">
        <button type="button" id="bookButton" class="checkbox-dropdown-button">
          <span id="bookButtonText">Book (All)</span>
        </button>
        <div id="bookFilter" class="checkbox-group"></div>
      </div>
    </label>
    <div style="display:flex; gap:8px;">
      <button class="badge" onclick="applyFilters()">Apply</button>
      <button class="badge reset" onclick="resetFilters()">Reset</button>
    </div>
  </div>
  <div style="margin:8px 0 16px 0;font-size:12px;color:#9aa0a6;">
    <button id="select-all-books" style="padding:6px 12px;margin-right:8px;cursor:pointer;background:#2a2a35;color:#e8eaed;border:1px solid #23232e;border-radius:6px;font-size:12px;">Select All Books</button>
    <button id="clear-books" style="padding:6px 12px;cursor:pointer;background:#2a2a35;color:#e8eaed;border:1px solid #23232e;border-radius:6px;font-size:12px;">Clear All Books</button>
    <span id="book-count" style="margin-left:12px;"></span>
  </div>

  <div class="note">{consensus_note}</div>

  <div id="list">{cards_html}</div>
  <div id="empty" style="display:none; color:#9b9ba1; margin:12px 0;">No results. Try lowering Min edge or clearing filters.</div>
</main>
<script>
function readInt(id, fallback) {{
  const raw = document.getElementById(id)?.value;
  const v = Number.parseInt(raw, 10);
  return Number.isFinite(v) ? v : fallback;
}}
function readFloat(id, fallback) {{
  const raw = document.getElementById(id)?.value;
  const v = Number.parseFloat(raw);
  return Number.isFinite(v) ? v : fallback;
}}
function getEdge(el) {{
  const raw = el.getAttribute('data-edge');
  if (raw === null || raw === '') return NaN;
  const v = Number.parseFloat(raw);
  return Number.isFinite(v) ? v : NaN;
}}
function showCard(el) {{ el.style.removeProperty('display'); }}  // let CSS decide
function hideCard(el) {{ el.style.display = 'none'; }}


function getSelectedBooks() {{
  const bookFilter = document.getElementById('bookFilter');
  const checkboxes = bookFilter.querySelectorAll('input[type="checkbox"]:checked');
  return Array.from(checkboxes).map(cb => cb.value.toLowerCase());
}}

function updateBookCount() {{
  const selected = getSelectedBooks();
  const bookCountEl = document.getElementById('book-count');
  if (bookCountEl) {{
    bookCountEl.textContent = selected.length === 0
      ? 'No books selected'
      : `${{selected.length}} book(s) selected`;
  }}
}}

function applyFilters() {{
  const minEdge = readFloat('minEdge', 0);
  const topN    = readInt('topN', 10);
  const mv      = (document.getElementById('marketFilter')?.value || '');
  const gv      = (document.getElementById('gameFilter')?.value || '');
  const selectedBooks = getSelectedBooks();

  const list  = document.getElementById('list');
  const empty = document.getElementById('empty');
  const cards = Array.from(list.querySelectorAll('.card'));

  // sort by edge desc; NaN edges sort last
  cards.sort((a,b) => {{
    const ea = getEdge(a), eb = getEdge(b);
    const sa = Number.isFinite(ea) ? ea : -1e9;
    const sb = Number.isFinite(eb) ? eb : -1e9;
    return sb - sa;
  }});

  let shown = 0;
  cards.forEach(c => {{
    const e = getEdge(c);
    const okEdge = Number.isFinite(e) ? (e >= minEdge) : (minEdge <= 0);
    const okMkt  = !mv || c.dataset.market === mv;
    const okGame = !gv || c.dataset.game === gv;
    const okBook = selectedBooks.length === 0 || selectedBooks.includes((c.dataset.book || '').toLowerCase());
    const ok = okEdge && okMkt && okGame && okBook;

    if (ok && shown < topN) {{ showCard(c); shown++; }} else {{ hideCard(c); }}
  }});

  empty.style.display = shown ? 'none' : 'block';
  updateBookCount();
}}

function resetFilters() {{
  document.getElementById('minEdge').value = 0;
  document.getElementById('topN').value = 10;
  document.getElementById('marketFilter').selectedIndex = 0;
  document.getElementById('gameFilter').selectedIndex = 0;
  // Select all books by default on reset
  const bookFilter = document.getElementById('bookFilter');
  bookFilter.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
  applyFilters();
}}

function copyCard(btn) {{
  const card = btn.closest('.card');
  const t = (sel) => card.querySelector(sel)?.textContent.trim() || '';
  const player = t('.player');
  const game = t('.game');
  const time = t('.time');
  const bet = t('.betline')?.replace('Bet: ', '') || '';
  const text = [player, game, time, bet].filter(Boolean).join(' | ');
  navigator.clipboard.writeText(text).then(() => {{
    btn.textContent = "Copied!";
    setTimeout(() => btn.textContent = "Copy bet", 900);
  }}).catch(err => {{
    console.error('Copy failed:', err);
    btn.textContent = "Copy failed";
    setTimeout(() => btn.textContent = "Copy bet", 900);
  }});
}}

// Initialize book filter with checkboxes
const bookFilter = document.getElementById('bookFilter');
const bookDropdown = document.getElementById('bookDropdown');
const bookButton = document.getElementById('bookButton');
const bookButtonText = document.getElementById('bookButtonText');
const selectAllBooksBtn = document.getElementById('select-all-books');
const clearBooksBtn = document.getElementById('clear-books');

function updateBookButtonText() {{
  const checkboxes = bookFilter.querySelectorAll('input[type="checkbox"]');
  const checked = bookFilter.querySelectorAll('input[type="checkbox"]:checked');
  const total = checkboxes.length;
  const count = checked.length;
  bookButtonText.textContent = count === total ? 'Book (All)' : count === 0 ? 'Book (None)' : `Book (${{count}})`;
}}

// Populate book checkboxes from book_opts
const bookPairs = {book_opts_js};
bookPairs.forEach(([value, label]) => {{
  const div = document.createElement('div');
  div.className = 'checkbox-item';

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.id = `book-${{value}}`;
  checkbox.value = value;
  checkbox.checked = true;  // All checked by default
  checkbox.addEventListener('change', () => {{
    updateBookButtonText();
    applyFilters();
  }});

  const labelEl = document.createElement('label');
  labelEl.htmlFor = `book-${{value}}`;
  labelEl.textContent = label;

  div.appendChild(checkbox);
  div.appendChild(labelEl);
  bookFilter.appendChild(div);
}});

updateBookButtonText();

// Dropdown toggle
bookButton.addEventListener('click', e => {{
  e.stopPropagation();
  bookDropdown.classList.toggle('open');
}});

// Close dropdown when clicking outside
document.addEventListener('click', e => {{
  if (!bookDropdown.contains(e.target)) {{
    bookDropdown.classList.remove('open');
  }}
}});

// Event listeners for Select All and Clear All buttons
selectAllBooksBtn.addEventListener('click', (e) => {{
  e.preventDefault();
  bookFilter.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
  updateBookButtonText();
  applyFilters();
}});

clearBooksBtn.addEventListener('click', (e) => {{
  e.preventDefault();
  bookFilter.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
  updateBookButtonText();
  applyFilters();
}});

applyFilters();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=f"{BRAND} — Top Picks")
    ap.add_argument("--season", type=int, help="Season year for header")
    ap.add_argument("--week", type=int, help="Week number for header")
    ap.add_argument("--limit", type=int, default=25000)  # render cap
    args = ap.parse_args()




    df_all = read_df(args.merged_csv)

    # One card per leg across all books
    df_all = collapse_best_across_lines_and_books(df_all)



    # Build filter options from FULL CSV (pre-limit) so all choices show
    markets_map, games_map, books_map = {}, {}, {}
    if "market" in df_all.columns:
        for m in df_all["market"].dropna():
            lbl = pretty_market(m); markets_map[_norm(lbl)] = lbl
    if "game_disp" in df_all.columns:
        for g in df_all["game_disp"].dropna():
            s = str(g).strip(); games_map[_norm(s)] = s
    if "book" in df_all.columns:
        for b in df_all["book"].dropna():
            s = str(b).strip(); books_map[_norm(s)] = s
    market_pairs = sorted(markets_map.items(), key=lambda x: x[1].lower())
    game_pairs   = sorted(games_map.items(),    key=lambda x: x[1].lower())
    book_pairs   = sorted(books_map.items(),    key=lambda x: x[1].lower())

    # Sort by edge desc & cap rows to render
    df = df_all.sort_values(by="edge_bps", ascending=False).head(min(CARD_LIMIT, args.limit))
    from pathlib import Path
    from site_common import inject_nav
    # Render cards
    model_prob_col = df_all.attrs.get("model_prob_col")
    consensus_prob_col = df_all.attrs.get("consensus_prob_col")
    cards = "\n".join(card(r, model_prob_col, consensus_prob_col) for _, r in df.iterrows())

    html = html_page(cards, args.title, market_pairs, game_pairs, book_pairs, week=args.week, season=args.season)
# (no manual .replace("__NAV__", ...) needed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)



    write_with_nav_raw(
        args.out,
        (getattr(args, "title", None) or "Fourth & Value — Top Picks"),
        html,
        active="Top Picks",
    )


    # diagnostics stay the same…

    # diagnostics
    nn_edge = int(df_all["edge_bps"].notna().sum()) if "edge_bps" in df_all.columns else 0
    print(f"[top] wrote {args.out}")
    print(f"[top] rows in CSV: {len(df_all)} ; rendered: {len(df)}")
    print(f"[top] non-null edge rows (CSV): {nn_edge}")
    print(f"[top] unique markets: {len(market_pairs)}, games: {len(game_pairs)}, books: {len(book_pairs)}")



    # ... your existing writes to args.out happen earlier ...




if __name__ == "__main__":
    main()

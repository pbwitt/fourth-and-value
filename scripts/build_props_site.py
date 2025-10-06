#!/usr/bin/env python3
import argparse, json
import pandas as pd
import numpy as np
from html import escape
from site_common import nav_html, pretty_market, fmt_odds, fmt_pct, to_kick_et, inject_nav, write_with_nav_raw
from pathlib import Path
# ---------- helpers ----------
def fmt_odds(o):
    if pd.isna(o): return ""
    try:
        o = int(round(float(o)))
        return f"{o:+d}"
    except Exception:
        return str(o)

from site_common import (
    nav_html, pretty_market,
    fmt_odds, to_kick_et
)


def prob_to_american(p):
    if p is None or (isinstance(p,float) and (np.isnan(p) or p<=0 or p>=1)): return ""
    return int(round(-100*p/(1-p))) if p>=0.5 else int(round(100*(1-p)/p))

def unit_for_market_std(market_std: str) -> str:
    if not isinstance(market_std, str): return ""
    ms = market_std.lower()
    if "passing_yards" in ms or "rushing_yards" in ms or "receiving_yards" in ms: return "yds"
    if "receptions"     in ms: return "rec"
    if "completions"    in ms: return "cmp"
    if "attempts"       in ms: return "att"
    if "tackles"        in ms: return "tkl"
    if "assists"        in ms: return "ast"
    return ""

def fmt_line(point, market_std):
    """Format sportsbook threshold from `point` with a unit inferred from market_std."""
    try:
        val = float(point)
    except Exception:
        return ""
    if pd.isna(val): return ""
    unit = unit_for_market_std(market_std or "")
    return f"{val:.1f}{(' ' + unit) if unit else ''}"

def kickoff_et_series(commence_series):
    """Convert ISO8601/Z times to 'Sun 1 p.m' in America/New_York."""
    ts = pd.to_datetime(commence_series.astype(str), utc=True, errors="coerce")
    out = []
    for v in ts:
        if pd.isna(v):
            out.append("")
            continue
        local = v.tz_convert("America/New_York")
        dow = local.strftime("%a")  # Sun, Mon, ...
        hour12 = (local.hour % 12) or 12
        minute = local.minute
        ampm = "a.m" if local.hour < 12 else "p.m"
        time_part = f"{hour12}" if minute == 0 else f"{hour12}:{minute:02d}"
        out.append(f"{dow} {time_part} {ampm}")
    return pd.Series(out, index=commence_series.index)
# ---- Row helpers + renderer (PLACE THIS BELOW TEMPLATE) ----
import math, html
from site_common import to_kick_et  # you already import this at the top

def _fmt_point(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        x = float(v)
        return str(int(x)) if x.is_integer() else f"{x:g}"
    except Exception:
        return str(v) if v is not None else ""

def row_html(r):
    # Game (Away @ Home)
    game = f"{html.escape(str(r.get('away_team','')))} @ {html.escape(str(r.get('home_team','')))}"

    # Bet = market (pretty) + line
    line_raw = r.get("line_disp") if r.get("line_disp") not in (None, "") else r.get("point")
    line_str = _fmt_point(line_raw)
    market_disp = str(r.get("market_disp",""))
    bet = market_disp or str(r.get("market",""))
    market_with_line = f"{bet} {line_str}".strip()

    # Odds, Fair, percents, edge vs market
    mkt_odds = r.get("price_disp", "")
    fair     = r.get("model_price")
    fair_str = "" if fair in (None,"") or (isinstance(fair,float) and math.isnan(fair)) else f"{int(fair):+d}"

    mkt_pct   = r.get("mkt_prob_pct","")
    model_pct = r.get("model_prob_pct","")
    edge_bps  = r.get("edge_bps_mkt")
    edge_str  = "" if edge_bps is None or (isinstance(edge_bps, float) and math.isnan(edge_bps)) else str(int(edge_bps))

    # Kick (ET)
    kick_iso  = r.get("kick_et") or r.get("commence_time") or ""
    kick_disp = to_kick_et(str(kick_iso)) if kick_iso else ""

    return (
        "<tr>"
        f"<td>{game}</td>"
        f"<td>{html.escape(str(r.get('player','')))}</td>"
        f"<td>{html.escape(str(r.get('bookmaker','')))}</td>"
        f"<td>{html.escape(market_with_line)}</td>"
        f"<td>{html.escape(line_str)}</td>"
        f"<td>{mkt_odds}</td>"
        f"<td>{fair_str}</td>"
        f"<td>{html.escape(str(mkt_pct))}</td>"
        f"<td>{html.escape(str(model_pct))}</td>"
        f"<td>{edge_str}</td>"
        f"<td>{html.escape(kick_disp)}</td>"
        "</tr>"
    )

# ---------- main ----------
def main():
    import json
    from html import escape

    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="NFL-2025 — Player Props")
    ap.add_argument("--season", type=int, help="Season year for header")
    ap.add_argument("--week", type=int, help="Week number for header")
    ap.add_argument("--min_prob", type=float, default=0.01, help="Drop rows with model_prob < this (unless --show_unmodeled)")
    ap.add_argument("--limit", type=int, default=3000, help="Max rows to render")
    ap.add_argument("--drop_no_scorer", action="store_true", default=True, help="Hide 'No Scorer' rows")
    ap.add_argument("--show_unmodeled", action="store_true", help="Include rows with missing model_prob")
    args = ap.parse_args()

    # Load
    df0 = pd.read_csv(args.merged_csv, low_memory=False)

    # Add 'side' column from 'name' if missing (BEFORE consensus join)
    if "side" not in df0.columns and "name" in df0.columns:
        df0["side"] = df0["name"]

    # Load consensus CSV and join to get market consensus data
    from pathlib import Path
    consensus_path = Path(args.merged_csv).parent / f"consensus_week{args.week}.csv"
    if consensus_path.exists():
        consensus = pd.read_csv(consensus_path)
        # Strip whitespace only (no case changes)
        for col in ["name_std", "market_std", "side"]:
            if col in df0.columns:
                df0[col] = df0[col].astype(str).str.strip()
            if col in consensus.columns:
                consensus[col] = consensus[col].astype(str).str.strip()

        # Join on name_std, market_std, AND side
        df0 = df0.merge(
            consensus[["name_std", "market_std", "side", "consensus_line", "consensus_prob", "book_count"]],
            on=["name_std", "market_std", "side"],
            how="left",
            suffixes=("", "_cons")
        )

    # Ensure expected cols exist
    for c in ["market_std","player","home_team","away_team","bookmaker",
              "model_prob","model_price","price","commence_time","point","market",
              "consensus_line","consensus_prob","book_count"]:
        if c not in df0.columns:
            df0[c] = np.nan

    # Pretty market label available if you want it later
    df0["market_disp"] = df0["market"].map(pretty_market) if "market" in df0.columns else ""

    # Drop "No Scorer" if requested
    if args.drop_no_scorer and "player" in df0.columns:
        df0 = df0[df0["player"].astype(str).str.lower() != "no scorer"].copy()

    # Game label
    df0["home_team"] = df0["home_team"].fillna("").astype(str).str.strip()
    df0["away_team"] = df0["away_team"].fillna("").astype(str).str.strip()
    df0["game"] = (df0["away_team"] + " @ " + df0["home_team"]).str.strip()

    # Odds display
    df0["mkt_odds"] = df0["price"].map(fmt_odds)
    df0["price_disp"] = df0["mkt_odds"]  # Alias for template compatibility

    # Fair odds (prefer model_price, else from model_prob)
    def _prob_to_american(p):
        try:
            p = float(p)
        except Exception:
            return np.nan
        if not (0 < p < 1):
            return np.nan
        return int(round(-100*p/(1-p))) if p >= 0.5 else int(round(100*(1-p)/p))

    # Fair odds = de-vigged market probability converted to odds
    # Assumes ~4.76% vig (1/(1+0.05))
    from site_common import prob_to_american as prob_to_am_safe
    VIG_FACTOR = 0.9524  # Remove ~5% vig (1/(1+0.05))
    df0["devig_prob"] = (df0["mkt_prob"] * VIG_FACTOR).clip(0.01, 0.99)
    df0["fair_odds"] = df0["devig_prob"].apply(prob_to_am_safe).map(fmt_odds)
    df0["model_price"] = df0["model_prob"].apply(_prob_to_american)  # Fair odds from model

    # Percentages
    def _american_to_prob(o):
        try:
            o = float(o)
        except Exception:
            return np.nan
        return (100.0/(o+100.0)) if o > 0 else (abs(o)/(abs(o)+100.0))

    df0["mkt_prob"]  = df0["price"].apply(_american_to_prob)
    df0["mkt_pct"]   = df0["mkt_prob"].map(fmt_pct)
    df0["model_pct"] = df0["model_prob"].map(fmt_pct)
    df0["mkt_prob_pct"] = df0["mkt_pct"]  # Alias for template compatibility
    df0["model_prob_pct"] = df0["model_pct"]  # Alias for template compatibility

    # Edge vs market implied (bps)
    df0["edge_bps"] = ((df0["model_prob"] - df0["mkt_prob"]) * 1e4).round()
    df0["edge_bps_mkt"] = df0["edge_bps"]  # Alias for template compatibility

    # Line (threshold) from sportsbook `point`
    def _fmt_point(x):
        try:
            if pd.isna(x):
                return ""
            xf = float(x)
            return str(int(xf)) if float(int(xf)) == xf else f"{xf:g}"
        except Exception:
            return str(x) if x is not None else ""
    df0["line_disp"] = df0["point"].apply(_fmt_point)

    # Kickoff, formatted to ET
    df0["kick_et"] = df0["commence_time"].astype(str).map(to_kick_et)

    # Format consensus columns
    df0["consensus_line_disp"] = df0["consensus_line"].apply(_fmt_point)
    df0["consensus_pct"] = df0["consensus_prob"].map(fmt_pct)
    df0["book_count_disp"] = df0["book_count"].apply(lambda x: str(int(x)) if pd.notna(x) else "")

    # Filter modeled if requested
    df = df0.copy()
    if not args.show_unmodeled and "model_prob" in df.columns:
        df = df[df["model_prob"].notna()].copy()
        if args.min_prob is not None:
            df = df[df["model_prob"] >= args.min_prob].copy()

    # Sort & trim
    df = df.sort_values("edge_bps", ascending=False, na_position="last")
    if args.limit:
        df = df.head(args.limit).copy()

    # Keep exactly what the JS table uses
    keep = [
        "game","player","bookmaker",
        # Bet column: use the slug the filters expect (market_std). If you prefer pretty, swap to "market_disp".
        "market_std",
        "line_disp",
        "mkt_odds","fair_odds","mkt_pct","model_pct","edge_bps",
        "consensus_line_disp","consensus_pct","book_count_disp",
        "kick_et"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = ""
    records = json.loads(df[keep].to_json(orient="records"))
    jsdata = json.dumps(records)  # safe literal

    title_html = escape(args.title)

    # Week header (prominent h1)
    week_header = ""
    if args.week:
        if args.season:
            week_header = f'<h1 style="margin:0 0 8px;font-size:22px;font-weight:700;letter-spacing:.2px;color:#fff;">Week {args.week}, {args.season}</h1>'
        else:
            week_header = f'<h1 style="margin:0 0 8px;font-size:22px;font-weight:700;letter-spacing:.2px;color:#fff;">Week {args.week}</h1>'

    # -------- HTML shell (your client-side renderer) --------
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + title_html + """</title>
<style>
:root { --bg:#0b0b10; --card:#14141c; --muted:#9aa0a6; --text:#e8eaed; --border:#23232e }
*{box-sizing:border-box} body{margin:0;padding:24px;background:var(--bg);color:var(--text);
font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
h1{margin:0 0 8px;font-size:22px;font-weight:700;letter-spacing:.2px}
.small{color:var(--muted);font-size:12px;margin-bottom:16px}
.card{background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,0));
border:1px solid var(--border);border-radius:16px;padding:16px;margin-bottom:16px;
box-shadow:0 0 0 1px rgba(255,255,255,.02),0 12px 40px rgba(0,0,0,.35)}
.controls{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
select,input{background:var(--card);color:var(--text);border:1px solid var(--border);
border-radius:10px;padding:10px 12px;outline:none}
.select:focus,input:focus{border-color:#6ee7ff;box-shadow:0 0 0 3px rgba(110,231,255,.15)}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;color:#111;background:#6ee7ff}
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:14px}
table{border-collapse:collapse;width:100%;min-width:1000px}
th,td{padding:10px 12px;border-bottom:1px solid var(--border)}
th{text-align:left;position:sticky;top:0;background:var(--card);z-index:1;font-size:12px;color:var(--muted);letter-spacing:.2px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:rgba(255,255,255,.02)}
footer{color:var(--muted);font-size:12px;margin-top:16px}
a.button{display:inline-block;margin:8px 0;padding:8px 14px;border-radius:10px;text-decoration:none;font-weight:600;color:#111;background:#a78bfa;border:1px solid var(--border)}
a.button:hover{background:#6ee7ff}
.linklike{color:#a78bfa;text-decoration:none;border-bottom:1px dotted #a78bfa}
.checkbox-dropdown{position:relative;min-width:180px}
.checkbox-dropdown-button{width:100%;background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 12px;outline:none;cursor:pointer;text-align:left;display:flex;justify-content:space-between;align-items:center}
.checkbox-dropdown-button:hover{border-color:#6ee7ff}
.checkbox-dropdown-button::after{content:'▼';font-size:10px;opacity:0.6}
.checkbox-dropdown.open .checkbox-dropdown-button::after{content:'▲'}
.checkbox-group{display:none;position:absolute;z-index:1000;flex-direction:column;gap:6px;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px;max-height:250px;overflow-y:auto;min-width:100%;margin-top:4px;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.checkbox-dropdown.open .checkbox-group{display:flex}
.checkbox-item{display:flex;align-items:center;gap:8px;cursor:pointer;padding:4px 6px;border-radius:6px;transition:background .15s}
.checkbox-item:hover{background:rgba(255,255,255,.05)}
.checkbox-item input[type="checkbox"]{width:16px;height:16px;cursor:pointer;accent-color:#34d399}
.checkbox-item label{cursor:pointer;flex:1;font-size:13px;color:var(--text);margin:0;white-space:nowrap}
</style>
</head>
<body>

""" + week_header + """

  <div class="card">
    <h1>""" + title_html + """</h1>
    <div class="small">Select <span class="badge">Bet</span> → Game → Player. Optional: Book & search. Sorted by Edge (bps). Line = sportsbook threshold. <strong>Book Odds</strong> = book's offered price (includes vig). <strong>Fair (De-vig)</strong> = book price with ~5% vig removed. <strong>Book %</strong> = book's implied probability (with vig). <strong>Model %</strong> = our model's probability. <strong>Edge</strong> = Model % − Book % (basis points). <strong>Consensus Line</strong> = median line across books. <strong>Market %</strong> = de-vigged consensus probability across all books. <strong>Books</strong> = number of books offering this prop.</div>
    
    <div class="controls">
      <select id="market"><option value="">Bet (market)</option></select>
      <select id="game"><option value="">Game</option></select>
      <select id="player"><option value="">Player</option></select>
      <div id="bookDropdown" class="checkbox-dropdown">
        <button type="button" id="bookButton" class="checkbox-dropdown-button">
          <span id="bookButtonText">Book</span>
        </button>
        <div id="book" class="checkbox-group"></div>
      </div>
      <input id="q" type="search" placeholder="Search player / team / book…" />
    </div>

    <div class="small" style="margin-top:10px;">
      <span id="count"></span> · Tip: “No Scorer” is hidden.
    </div>
  </div>

  <div class="card table-wrap">
    <table id="tbl">
      <thead>
        <tr>
          <th>Game</th><th>Player</th><th>Book</th><th>Bet</th>
          <th class="num">Line</th>
          <th class="num">Book Odds</th><th class="num">Fair (De-vig)</th>
          <th class="num">Book %</th><th class="num">Model %</th>
          <th class="num">Edge (bps)</th>
          <th class="num">Consensus Line</th><th class="num">Market %</th><th class="num">Books</th>
          <th>Kick (ET)</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <footer>Generated locally. Dark theme, zero dependencies.</footer>

<script>
const DATA = """ + jsdata + """;

function uniq(arr){ return [...new Set(arr.filter(Boolean))].sort((a,b)=>a.localeCompare(b)); }

const state = { market:"", game:"", player:"", books:new Set(), q:"" };
const selMarket = document.getElementById("market");
const selGame   = document.getElementById("game");
const selPlayer = document.getElementById("player");
const bookGroup = document.getElementById("book");
const bookDropdown = document.getElementById("bookDropdown");
const bookButton = document.getElementById("bookButton");
const bookButtonText = document.getElementById("bookButtonText");
const inputQ    = document.getElementById("q");
const tbody     = document.querySelector("#tbl tbody");
const countEl   = document.getElementById("count");

function updateBookButtonText(){
  const total = bookGroup.querySelectorAll('input[type="checkbox"]').length;
  const checked = state.books.size;
  bookButtonText.textContent = checked === total ? 'Book (All)' : checked === 0 ? 'Book (None)' : `Book (${checked})`;
}

function hydrateSelectors(){
  uniq(DATA.map(r=>r.market_std)).forEach(v=>{ const o=document.createElement("option"); o.value=v; o.textContent=v; selMarket.appendChild(o); });

  // Populate book checkboxes
  uniq(DATA.map(r=>r.bookmaker)).forEach(book => {
    const div = document.createElement("div");
    div.className = "checkbox-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `book-${book}`;
    checkbox.value = book;
    checkbox.checked = true;
    state.books.add(book);
    checkbox.addEventListener("change", e => {
      if (e.target.checked) state.books.add(book);
      else state.books.delete(book);
      updateBookButtonText();
      rebuildDependentSelectors();
      render();
    });

    const label = document.createElement("label");
    label.htmlFor = `book-${book}`;
    label.textContent = book;

    div.appendChild(checkbox);
    div.appendChild(label);
    bookGroup.appendChild(div);
  });

  updateBookButtonText();
  rebuildDependentSelectors();
}

// Dropdown toggle
bookButton.addEventListener("click", e => {
  e.stopPropagation();
  bookDropdown.classList.toggle("open");
});

// Close dropdown when clicking outside
document.addEventListener("click", e => {
  if (!bookDropdown.contains(e.target)) {
    bookDropdown.classList.remove("open");
  }
});
function rebuildDependentSelectors(){
  const base = DATA.filter(r => (!state.market || r.market_std===state.market) &&
                                (state.books.size===0 || state.books.has(r.bookmaker)));
  const games = uniq(base.map(r=>r.game));
  selGame.innerHTML = '<option value="">Game</option>' + games.map(g=>`<option value="${g}">${g}</option>`).join("");
  if (games.includes(state.game)) selGame.value = state.game; else state.game = "";

  const base2 = base.filter(r => (!state.game || r.game===state.game));
  const players = uniq(base2.map(r=>r.player));
  selPlayer.innerHTML = '<option value="">Player</option>' + players.map(p=>`<option value="${p}">${p}</option>`).join("");
  if (players.includes(state.player)) selPlayer.value = state.player; else state.player = "";
}

function render(){
  const q = state.q.trim().toLowerCase();
  const rows = DATA.filter(r =>
    (!state.market || r.market_std===state.market) &&
    (!state.game   || r.game===state.game) &&
    (!state.player || r.player===state.player) &&
    (state.books.size===0 || state.books.has(r.bookmaker)) &&
    (!q || (r.player+" "+r.bookmaker+" "+r.game).toLowerCase().includes(q))
  ).sort((a,b)=> (b.edge_bps ?? -1) - (a.edge_bps ?? -1));

  countEl.textContent = rows.length + " rows";

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.game||""}</td>
      <td>${r.player||""}</td>
      <td>${r.bookmaker||""}</td>
      <td>${r.market_std||""}</td>
      <td class="num">${r.line_disp ?? ""}</td>
      <td class="num">${r.mkt_odds ?? ""}</td>
      <td class="num">${r.fair_odds ?? ""}</td>
      <td class="num">${r.mkt_pct ?? ""}</td>
      <td class="num">${r.model_pct ?? ""}</td>
      <td class="num" style="color:${
        (r.edge_bps==null) ? "#9aa0a6" : (r.edge_bps>0 ? "#4ade80" : "#f87171")
      }">${r.edge_bps ?? ""}</td>
      <td class="num">${r.consensus_line_disp ?? ""}</td>
      <td class="num">${r.consensus_pct ?? ""}</td>
      <td class="num">${r.book_count_disp ?? ""}</td>
      <td>${r.kick_et||""}</td>
    </tr>
  `).join("");
}

selMarket.addEventListener("change", e=>{ state.market=e.target.value; rebuildDependentSelectors(); render(); });
selGame  .addEventListener("change", e=>{ state.game  =e.target.value; rebuildDependentSelectors(); render(); });
selPlayer.addEventListener("change", e=>{ state.player=e.target.value; render(); });
inputQ   .addEventListener("input",  e=>{ state.q     =e.target.value; render(); });

hydrateSelectors(); render();
</script>
</body></html>
"""

# ... after building `html`
# (near the top of the file)
    # ---- write page with permanent nav ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_with_nav_raw(
        out_path.as_posix(),
        (getattr(args, "title", None) or f"Fourth & Value — Player Props (Week {args.week})"),
        html,
        active="Props",
    )




    print(f"[props_site] wrote {args.out} with {len(df)} rows (from {len(df0)} source rows)")



if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, re, math
from html import escape
from pathlib import Path
from site_common import inject_nav
from site_common import write_with_nav_raw
 # add this
# shared helpers
from site_common import (
    nav_html, pretty_market, fmt_odds_american, fmt_pct,
    american_to_prob, kickoff_et, BRAND
)

# -------- helpers local to this script --------
# --- 1) Add near your other template helpers (top of file) -------------------
FILTERS_BLOCK = r"""
<style>
  .filters { display:grid; gap:10px; grid-template-columns: repeat(4, minmax(140px, 1fr)) 1fr; align-items:end; margin:12px 0 10px; }
  .filters label { font-size:12px; color:#b7b7bb; display:block; margin-bottom:5px; }
  .filters select, .filters input {
    width:100%; padding:8px 10px; background:#111113; color:#e7e7ea; border:1px solid #1f1f22; border-radius:10px;
  }
  .filters .right { text-align:right; }
  .filters .btn-clear {
    padding:9px 12px; background:#18181b; border:1px solid #2a2a2f; border-radius:10px; color:#d7d7dc; cursor:pointer;
  }
  .filters .btn-clear:hover { background:#202024; }
  .results-note { margin:4px 0 12px; color:#b7b7bb; font-size:12px; }
  @media (max-width: 720px) {
    .filters { grid-template-columns: 1fr 1fr; }
    .filters .right { grid-column: 1 / -1; text-align:left; }
  }
</style>

<div id="consensus-filters" class="filters" hidden>
  <div>
    <label for="f_game">Game</label>
    <select id="f_game"><option value="">All games</option></select>
  </div>
  <div>
    <label for="f_market">Market</label>
    <select id="f_market"><option value="">All markets</option></select>
  </div>
  <div>
    <label for="f_player">Player</label>
    <select id="f_player"><option value="">All players</option></select>
  </div>
  <div>
    <label for="f_book">Book</label>
    <select id="f_book"><option value="">All books</option></select>
  </div>
  <div class="right">
    <button class="btn-clear" id="f_clear">Clear filters</button>
  </div>
</div>
<div id="consensus-results" class="results-note" hidden></div>

<script>
document.addEventListener('DOMContentLoaded',function(){
  const table = document.querySelector('main.container .tablewrap table');
  if (!table) return;

  const tbody = table.tBodies[0];
  const rows  = Array.from(tbody?.rows || []);
  if (!rows.length) return;

  // Column order from the Consensus page you generated:
  // Kick | Game | Player | Market | Best book bet | Market consensus | Edge
  const COL = { game:1, player:2, market:3, bet:4 };

  function extractBook(txt){
    const m = txt.match(/\bon\s+([a-z0-9_]+)\s*$/i);  // "... on fanduel"
    return m ? m[1].toLowerCase() : '';
  }

  const uniq = { game:new Set(), market:new Set(), player:new Set(), book:new Set() };
  const cache = new Map();

  rows.forEach(tr => {
    const game   = tr.cells[COL.game]?.textContent.trim() || '';
    const player = tr.cells[COL.player]?.textContent.trim() || '';
    const market = tr.cells[COL.market]?.textContent.trim() || '';
    const betTxt = tr.cells[COL.bet]?.textContent.trim() || '';
    const book   = extractBook(betTxt);

    cache.set(tr, { game, market, player, book });

    if (game)   uniq.game.add(game);
    if (market) uniq.market.add(market);
    if (player) uniq.player.add(player);
    if (book)   uniq.book.add(book);
  });

  const $wrap   = document.getElementById('consensus-filters');
  const $note   = document.getElementById('consensus-results');
  const $game   = document.getElementById('f_game');
  const $market = document.getElementById('f_market');
  const $player = document.getElementById('f_player');
  const $book   = document.getElementById('f_book');
  const $clear  = document.getElementById('f_clear');

  function fillSelect(sel, items) {
    const sorted = Array.from(items).sort((a,b)=> a.localeCompare(b));
    const frag = document.createDocumentFragment();
    sorted.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      frag.appendChild(opt);
    });
    sel.appendChild(frag);
  }

  fillSelect($game,   uniq.game);
  fillSelect($market, uniq.market);
  fillSelect($player, uniq.player);
  fillSelect($book,   uniq.book);

  function fmt(n){ return n.toLocaleString(); }

  function applyFilters(){
    const f = {
      game:   $game.value,
      market: $market.value,
      player: $player.value,
      book:   $book.value.toLowerCase()
    };

    let shown = 0;
    rows.forEach(tr => {
      const v = cache.get(tr);
      const ok =
        (!f.game   || v.game   === f.game) &&
        (!f.market || v.market === f.market) &&
        (!f.player || v.player === f.player) &&
        (!f.book   || v.book   === f.book);
      tr.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });

    $note.textContent = `${fmt(shown)} result${shown===1?'':'s'} shown` +
                        (f.game||f.market||f.player||f.book ? ' (filters on)' : '');
  }

  [$game, $market, $player, $book].forEach(el => el.addEventListener('change', applyFilters));
  $clear.addEventListener('click', () => {
    $game.value = $market.value = $player.value = $book.value = '';
    applyFilters();
  });

  const tablewrap = table.closest('.tablewrap');
  tablewrap.parentNode.insertBefore($wrap, tablewrap);
  tablewrap.parentNode.insertBefore($note, tablewrap.nextSibling);

  $wrap.hidden = false;
  $note.hidden = false;
  applyFilters();
});
</script>
"""

def inject_filters(html: str) -> str:
    """
    Insert the filter UI before the first .tablewrap.
    If the template contains a '__FILTERS__' marker, use that instead.
    """
    if "__FILTERS__" in html:
        return html.replace("__FILTERS__", FILTERS_BLOCK)
    # Fallback: inject before the first tablewrap div
    needle = '<div class="tablewrap">'
    idx = html.find(needle)
    if idx == -1:
        return html  # nothing to do
    return html[:idx] + FILTERS_BLOCK + html[idx:]


LINE_CANDIDATES = [
    "line_disp","point","line","market_line","prop_line","number","threshold","total","line_number",
    "handicap","spread","yards","receptions","receiving_yards","rushing_yards","passing_yards","prop_total"
]
GAME_CANDIDATES = ["game","Game","matchup","matchup_name","matchup_display"]

def _num(x):
    with np.errstate(all="ignore"):
        return pd.to_numeric(x, errors="coerce")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _first_nonnull(row, cols):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return row[c]
    return np.nan

def _is_numeric_total_market(mkt) -> bool:
    m = pretty_market(mkt or "").lower()
    if not m: return False
    if "yards" in m: return True
    return m in {"receptions","rush attempts","pass attempts","completions","passing touchdowns","rushing attempts"}

def mk_line_disp(r):
    # keep if already nice
    if "line_disp" in r and str(r["line_disp"]).strip():
        return str(r["line_disp"]).strip()
    raw_line = _first_nonnull(r, LINE_CANDIDATES)
    line_txt = ""
    if pd.notna(raw_line):
        try: line_txt = f"{float(raw_line):g}"
        except Exception: line_txt = str(raw_line).strip()
    side_raw = str(r.get("name") or r.get("side") or "").strip()  # Over/Under/Yes/No
    side = side_raw
    # totals: Yes/No → Over/Under
    if line_txt and _is_numeric_total_market(r.get("market")):
        if side_raw.lower() == "yes": side = "Over"
        elif side_raw.lower() == "no": side = "Under"
    # Anytime TD special "No Scorer"
    mkt_lbl = pretty_market(r.get("market","")).lower()
    player  = str(r.get("player","")).strip().lower()
    if ("anytime td" in mkt_lbl or "anytime touchdown" in mkt_lbl) and player in {"no scorer","no td scorer","no touchdown scorer"}:
        return "No Scorer"
    # assemble
    if side and line_txt: return f"{side} {line_txt}"
    return side or line_txt or ""

def kickoff_col(df):
    if "kick_et" in df.columns: return "kick_et"
    if "commence_time" in df.columns: return "commence_time"
    return None

def read_df(path):
    df = pd.read_csv(path)

    # normalize
    if "book" not in df.columns and "bookmaker" in df.columns:
        df["book"] = df["bookmaker"]

    for col in ["price"]:
        if col in df.columns: df[col] = _num(df[col])

    # kickoff
    kc = kickoff_col(df)
    if kc == "commence_time": df["kick_et"] = df["commence_time"]

    # ensure presence
    for col in ["player","market","book","home_team","away_team","name"]:
        if col not in df.columns: df[col] = np.nan
    df["name"] = df["name"].fillna("")

    # line display & game label
    df["line_disp"] = df.apply(mk_line_disp, axis=1)

    def mk_game(row):
        for c in GAME_CANDIDATES:
            if c in df.columns:
                v = row.get(c)
                if pd.notna(v) and str(v).strip(): return str(v).strip()
        away = str(row.get("away_team") or "").strip()
        home = str(row.get("home_team") or "").strip()
        return f"{away} vs {home}" if away and home else (away or home or "")
    df["game_disp"] = df.apply(mk_game, axis=1)

    # implied prob from book price
    df["imp_prob"] = df["price"].apply(american_to_prob) if "price" in df.columns else np.nan

    # consensus prob by (player, market, game, line_disp)
    key = ["player","market","game_disp","line_disp"]
    grp = df.groupby(key, dropna=False)["imp_prob"].median().rename("consensus_prob")
    df = df.merge(grp, on=key, how="left")

    # edge vs consensus (positive if book is better than consensus)
    df["consensus_edge_bps"] = 10000.0 * (df["consensus_prob"] - df["imp_prob"])

    # best book per bet (max edge)
    df["_edge_sort"] = df["consensus_edge_bps"].astype(float).where(df["consensus_edge_bps"].notna(), -1e15)
    df = (df.sort_values(key + ["_edge_sort"], ascending=[True,True,True,True,False])
            .drop_duplicates(subset=key, keep="first")
            .drop(columns=["_edge_sort"])
            .copy())

    # normalized attrs for filters if you add later
    df["_mkt_norm"]  = df["market"].apply(lambda m: _norm(pretty_market(m)))
    df["_game_norm"] = df["game_disp"].apply(_norm)
    df["_book_norm"] = df["book"].apply(_norm)

    return df

# -------- rendering --------
def row_html(r):
    time = kickoff_et(r.get("kick_et",""))
    game = str(r.get("game_disp",""))
    player = str(r.get("player",""))
    market = pretty_market(r.get("market",""))
    bet = []
    ld = str(r.get("line_disp","")).strip()
    if ld: bet.append(ld)
    odds = fmt_odds_american(r.get("price"))
    if odds: bet.append(f"@ {odds}")
    book = str(r.get("book",""))
    if book: bet.append(f"on {book}")
    bet_txt = " ".join(bet)

    cons_p = fmt_pct(r.get("consensus_prob"))
    edge   = r.get("consensus_edge_bps")
    edge_txt = "" if (edge is None or (isinstance(edge,float) and math.isnan(edge))) else f"{edge:,.0f} bps"

    return f"""<tr>
      <td>{escape(str(time))}</td>
      <td class="game">{escape(game)}</td>
      <td class="player">{escape(player)}</td>
      <td>{escape(market)}</td>
      <td class="bet">{escape(bet_txt)}</td>
      <td class="cons">{escape(cons_p)}</td>
      <td class="edge">{escape(edge_txt)}</td>
    </tr>"""

def html_page(rows_html, title):
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
.container {{ max-width: 1200px; margin: 0 auto; padding: 18px 16px 32px; }}
.h1 {{ font-size: clamp(22px,3.5vw,28px); font-weight:900; color:#fff; margin: 4px 0 10px; }}

.tablewrap {{ overflow:auto; border:1px solid #1f1f22; border-radius:14px; }}
table {{ width:100%; border-collapse: collapse; min-width: 900px; }}
thead th {{ text-align:left; font-weight:700; font-size:12px; color:#b7b7bb; padding:10px 12px; background:#111113; position:sticky; top:0; }}
tbody td {{ border-top:1px solid #1f1f22; padding:10px 12px; font-size:13px; }}
tbody tr:hover {{ background:#0f0f11; }}
td.game {{ color:#c8c8cd; }}
td.player {{ font-weight:700; color:#fff; }}
td.bet {{ color:#e3e3e6; }}
td.cons, td.edge {{ white-space:nowrap; }}

.note {{ margin:10px 0 16px; color:#b7b7bb; font-size:13px; }}
</style>
</head>
<body>

<main class="container">
  <div class="h1">{escape(title)}</div>
  <div class="note">Market consensus = median implied probability across books for the same bet. “Edge” shows how favorable the best book is versus that market consensus.</div>
__FILTERS__

  <div class="tablewrap">
    <table>
      <thead>
        <tr><th>Kick</th><th>Game</th><th>Player</th><th>Market</th><th>Best book bet</th><th>Market consensus</th><th>Edge</th></tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
# ... after you finish assembling the final HTML into `html`

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()

    # 1) Build the table
    # 1) Build the table
    df = read_df(args.merged_csv)
    df = df.sort_values(by="consensus_edge_bps", ascending=False).head(args.limit)
    rows = "\n".join(row_html(r) for _, r in df.iterrows())
    default_title = f"{BRAND} — Consensus (Week {args.week})" if args.week else f"{BRAND} — Consensus"
    # after args = ap.parse_args() and after you compute rows
    default_title = f"{BRAND} — Consensus (Week {args.week})" if args.week else f"{BRAND} — Consensus"
    title = args.title or default_title

    page = html_page(rows, title)          # drives <title> and H1
    page = inject_filters(page)            # keep your filter injector

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_with_nav_raw(
        out_path.as_posix(),
        title,                              # pass the SAME title to the writer
        page,
        active="Consensus",
    )





    print(f"[consensus] wrote {args.out} with {len(df)} rows (from {len(read_df(args.merged_csv))} source rows)")

if __name__ == "__main__":
    main()

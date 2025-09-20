#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, re, math
from html import escape
from pathlib import Path

# Robust sibling import for site_common (works as module or script)
try:
    # When run as: python -m scripts.build_consensus_page
    from scripts.site_common import (
        write_with_nav_raw,
        pretty_market, fmt_odds_american, fmt_pct,
        american_to_prob, kickoff_et, BRAND,
    )
except ModuleNotFoundError:
    # When run as: python scripts/build_consensus_page.py
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parent))
    from site_common import (
        write_with_nav_raw,
        pretty_market, fmt_odds_american, fmt_pct,
        american_to_prob, kickoff_et, BRAND,
    )


LINE_CANDIDATES = [
    "line_disp", "point", "line", "market_line", "prop_line", "number", "threshold",
    "total", "line_number", "handicap", "spread", "yards", "receptions",
    "receiving_yards", "rushing_yards", "passing_yards", "prop_total",
]
GAME_CANDIDATES = ["game", "Game", "matchup", "matchup_name", "matchup_display"]

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
    mkt_lbl = pretty_market(r.get("market", "")).lower()
    player  = str(r.get("player", "")).strip().lower()
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
    time = kickoff_et(r.get("kick_et", ""))
    game = str(r.get("game_disp", ""))
    player = str(r.get("player", ""))
    market = pretty_market(r.get("market", ""))
    bet_parts = []
    line_disp = str(r.get("line_disp", "")).strip()
    if line_disp:
        bet_parts.append(line_disp)
    odds = fmt_odds_american(r.get("price"))
    if odds:
        bet_parts.append(f"@ {odds}")
    book = str(r.get("book", "")).strip()
    if book:
        bet_parts.append(f"on {book}")
    bet_txt = " ".join(bet_parts)

    cons_p = fmt_pct(r.get("consensus_prob"))

    attrs = {
        "game": game,
        "player": player,
        "player-search": player.lower(),
        "market": market,
        "book": book,
    }
    attr_html = " ".join(
        f'data-{key}="{escape(value, quote=True)}"'
        for key, value in attrs.items()
    )

    return f"""<tr {attr_html}>
      <td>{escape(str(time))}</td>
      <td class="game">{escape(game)}</td>
      <td class="player">{escape(player)}</td>
      <td>{escape(market)}</td>
      <td class="bet">{escape(bet_txt)}</td>
      <td class="cons">{escape(cons_p)}</td>
    </tr>"""

def table_html(rows_html: str, table_id: str) -> str:
    return f"""  <div class="tablewrap">
    <table id="{table_id}">
      <thead>
        <tr><th>Kick</th><th>Game</th><th>Player</th><th>Market</th><th>Best book bet</th><th>Market consensus</th></tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>"""

def html_page(overview_rows_html: str, value_rows_html: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{escape(title)}</title>
<link rel="icon" href="data:,">
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#0b0b0c; color:#e7e7ea; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Ubuntu,Helvetica,Arial,sans-serif; }}
main.container {{ max-width: 1200px; margin: 0 auto; padding: 18px 16px 32px; }}
.h1 {{ font-size: clamp(22px,3.5vw,28px); font-weight:900; color:#fff; margin: 4px 0 10px; }}
.note {{ margin:10px 0 16px; color:#b7b7bb; font-size:13px; }}

#filters {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: 16px 0 12px; }}
#filters select, #filters input {{ width:100%; padding:8px 12px; background:#111113; color:#e7e7ea; border:1px solid #1f1f22; border-radius:10px; }}

.tabs {{ display:flex; gap:8px; margin: 18px 0 12px; border-bottom:1px solid #1f1f22; }}
.tabs button {{ appearance:none; border:none; background:transparent; color:#b7b7bb; padding:10px 16px; border-radius:12px 12px 0 0; font-weight:600; cursor:pointer; }}
.tabs button.active {{ background:#18181b; color:#fff; border-bottom:2px solid #3b82f6; }}

.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}

.tablewrap {{ overflow:auto; border:1px solid #1f1f22; border-radius:14px; margin-bottom:20px; }}
table {{ width:100%; border-collapse: collapse; min-width: 900px; }}
thead th {{ text-align:left; font-weight:700; font-size:12px; color:#b7b7bb; padding:10px 12px; background:#111113; position:sticky; top:0; }}
tbody td {{ border-top:1px solid #1f1f22; padding:10px 12px; font-size:13px; }}
tbody tr:hover {{ background:#0f0f11; }}
tbody tr[hidden] {{ display:none; }}
td.game {{ color:#c8c8cd; }}
td.player {{ font-weight:700; color:#fff; }}
td.bet {{ color:#e3e3e6; }}
td.cons {{ white-space:nowrap; }}

@media (max-width: 720px) {{
  #filters {{ grid-template-columns: 1fr 1fr; }}
  .tabs {{ flex-wrap:wrap; gap:6px; }}
  table {{ min-width: 0; }}
}}
</style>
</head>
<body>
<main class="container">
  <div class="h1">{escape(title)}</div>
  <div class="note">Market consensus = median implied probability across books for the same bet.</div>
  <div id="filters">
    <select id="filter-game"></select>
    <input id="filter-player" placeholder="Player" />
    <select id="filter-market"></select>
    <select id="filter-book"></select>
  </div>
  <datalist id="player-options"></datalist>
  <div class="tabs" role="tablist">
    <button type="button" class="tab active" data-tab-key="overview">Overview</button>
    <button type="button" class="tab" data-tab-key="value">Value</button>
  </div>
  <section class="tab-panel active" data-tab-key="overview">
{table_html(overview_rows_html, "overview-table")}
  </section>
  <section class="tab-panel" data-tab-key="value">
{table_html(value_rows_html, "value-table")}
  </section>
</main>
<script>
document.addEventListener('DOMContentLoaded', () => {{
  const tabButtons = Array.from(document.querySelectorAll('[data-tab-key]'));
  const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
  function activateTab(key) {{
    tabButtons.forEach(btn => {{
      btn.classList.toggle('active', btn.dataset.tabKey === key);
    }});
    tabPanels.forEach(panel => {{
      panel.classList.toggle('active', panel.dataset.tabKey === key);
    }});
  }}
  if (tabButtons.length) {{
    tabButtons.forEach(btn => {{
      btn.addEventListener('click', () => activateTab(btn.dataset.tabKey));
    }});
    activateTab(tabButtons[0].dataset.tabKey);
  }}

  const filterGame = document.getElementById('filter-game');
  const filterPlayer = document.getElementById('filter-player');
  const filterMarket = document.getElementById('filter-market');
  const filterBook = document.getElementById('filter-book');
  const playerList = document.getElementById('player-options');
  if (filterPlayer && playerList) {{
    filterPlayer.setAttribute('list', 'player-options');
  }}

  const tables = ['overview-table', 'value-table']
    .map(id => document.getElementById(id))
    .filter(Boolean);
  const uniques = {{ game: new Set(), player: new Set(), market: new Set(), book: new Set() }};

  tables.forEach(table => {{
    const tbody = table.tBodies[0];
    if (!tbody) return;
    Array.from(tbody.rows).forEach(row => {{
      const ds = row.dataset;
      if (ds.game) uniques.game.add(ds.game);
      if (ds.player) uniques.player.add(ds.player);
      if (ds.market) uniques.market.add(ds.market);
      if (ds.book) uniques.book.add(ds.book);
    }});
  }});

  function fillSelect(el, values, placeholder) {{
    if (!el) return;
    const frag = document.createDocumentFragment();
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = placeholder;
    frag.appendChild(blank);
    Array.from(values).sort((a, b) => a.localeCompare(b)).forEach(value => {{
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = value;
      frag.appendChild(opt);
    }});
    el.replaceChildren(frag);
  }}

  fillSelect(filterGame, uniques.game, 'All games');
  fillSelect(filterMarket, uniques.market, 'All markets');
  fillSelect(filterBook, uniques.book, 'All books');

  if (playerList) {{
    playerList.replaceChildren();
    Array.from(uniques.player).sort((a, b) => a.localeCompare(b)).forEach(value => {{
      const opt = document.createElement('option');
      opt.value = value;
      playerList.appendChild(opt);
    }});
  }}

  function applyFilters() {{
    const selectedGame = filterGame ? filterGame.value : '';
    const selectedMarket = filterMarket ? filterMarket.value : '';
    const selectedBook = filterBook ? filterBook.value : '';
    const playerQuery = filterPlayer ? filterPlayer.value.trim().toLowerCase() : '';

    tables.forEach(table => {{
      const tbody = table.tBodies[0];
      if (!tbody) return;
      Array.from(tbody.rows).forEach(row => {{
        const ds = row.dataset;
        let visible = true;
        if (selectedGame && ds.game !== selectedGame) visible = false;
        if (visible && selectedMarket && ds.market !== selectedMarket) visible = false;
        if (visible && selectedBook && ds.book !== selectedBook) visible = false;
        if (visible && playerQuery) {{
          const playerSearch = (ds.playerSearch || '').toLowerCase();
          visible = playerSearch.includes(playerQuery);
        }}
        row.hidden = !visible;
      }});
    }});
  }}

  [filterGame, filterMarket, filterBook].forEach(el => {{
    if (el) el.addEventListener('change', applyFilters);
  }});
  if (filterPlayer) {{
    filterPlayer.addEventListener('input', applyFilters);
  }}

  applyFilters();
}});
</script>
</body>
</html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()

    df = read_df(args.merged_csv)
    df = df.sort_values(by="consensus_edge_bps", ascending=False).head(args.limit).copy()

    overview_sort_cols = [c for c in ["kick_et", "game_disp", "player", "market"] if c in df.columns]
    if overview_sort_cols:
        overview_df = df.sort_values(by=overview_sort_cols, ascending=[True] * len(overview_sort_cols))
    else:
        overview_df = df.copy()
    value_df = df.copy()

    overview_rows = "\n".join(row_html(r) for _, r in overview_df.iterrows())
    value_rows = "\n".join(row_html(r) for _, r in value_df.iterrows())

    default_title = f"{BRAND} — Consensus (Week {args.week})" if args.week else f"{BRAND} — Consensus"
    title = args.title or default_title

    page_html = html_page(overview_rows, value_rows, title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_with_nav_raw(
        out_path.as_posix(),
        title,
        page_html,
        active="Consensus",
    )

if __name__ == "__main__":
    main()

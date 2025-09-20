#!/usr/bin/env python3
"""
Build Consensus Page (Overview + Value tabs)

- Overview: consensus lines + probs per player/market
- Value: per-book props with model vs market comparison
- Shared filters: Game, Market, Book, Player (text)
- Hides rows missing model_prob or mkt_prob
"""

import argparse
import pandas as pd
from pathlib import Path


from scripts.site_common import (
    pretty_market,
    write_with_nav_raw,
)

TABS_JS = """
<script>
function showTab(id){
  document.querySelectorAll('.tab-section').forEach(el => el.style.display = 'none');
  const el = document.getElementById(id);
  if (el) el.style.display = '';
  // highlight active tab
  document.querySelectorAll('.fv-tabs a').forEach(a => {
    if (a.getAttribute('href') === '#' + id) a.classList.add('active');
    else a.classList.remove('active');
  });
}
document.addEventListener('DOMContentLoaded', function(){
  // click handlers
  document.querySelectorAll('.fv-tabs a').forEach(a => {
    a.addEventListener('click', function(e){
      e.preventDefault();
      const id = this.getAttribute('href').slice(1);
      history.replaceState(null, '', '#' + id);
      showTab(id);
    });
  });
  // initial tab: hash or overview
  const initial = (location.hash || '#overview').slice(1);
  showTab(initial);
});
</script>
"""



TABS_CSS = """
<style>
/* Tabs header */
.fv-tabs { list-style: none; padding: 0; margin: 0 0 10px 0; display: flex; gap: 14px; }
.fv-tabs li { display: inline; }
.fv-tabs a { text-decoration: none; font-weight: 600; }
.fv-tabs a:hover { text-decoration: underline; }

/* Filter bar */
.fv-filter-bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0 10px; }
.fv-filter-bar label { font-size: 0.95rem; display: inline-flex; align-items: center; gap: 6px; }
.fv-filter-bar select, .fv-filter-bar input { padding: 4px 6px; }

/* Tables */
.consensus-table { border-collapse: collapse; width: 100%; margin: 6px 0 24px; }
.consensus-table th, .consensus-table td { border: 1px solid rgba(255,255,255,0.15); padding: 6px 8px; }
.consensus-table th { text-align: left; }
.consensus-table td.odds_disp,
.consensus-table td.edge_bps_disp,
.consensus-table td.fair_odds_disp,
.consensus-table td.ev_per_100_disp,
.consensus-table td.mkt_prob_disp,
.consensus-table td.model_prob_disp,
.consensus-table td.line_disp,
.consensus-table td.cons_line_disp { text-align: right; white-space: nowrap; }

/* Keep it theme-agnostic: inherit your page colors */
</style>
"""


# === CONSENSUS HELPERS (add once, near imports) ===
import math
import numpy as np
import pandas as pd

PRETTY_MAP = {
    "rush_yds": "Rush Yds",
    "rec_yds": "Recv Yds",
    "reception_yds": "Recv Yds",
    "pass_yds": "Pass Yds",
    "pass_tds": "Pass TDs",
    "pass_ints": "Pass INTs",
    "rush_attempts": "Rush Att",
    "pass_attempts": "Pass Att",
    "pass_completions": "Pass Comp",
    "anytime_td": "Anytime Td",
    "first_td": "Player 1St Td",
}

OU_MARKETS = {
    "rush_yds","rec_yds","reception_yds","pass_yds",
    "pass_tds","pass_ints","rush_attempts","pass_attempts","pass_completions"
}

def american_odds_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)

def prob_to_american_odds(p):
    if pd.isna(p) or p <= 0 or p >= 1:
        return np.nan
    # Fair American odds
    if p >= 0.5:
        # negative odds
        return -int(round(p * 100 / (1 - p)))
    else:
        # positive odds
        return int(round((1 - p) * 100 / p))

def expected_value_per_100(p, odds):
    """EV on a $100 stake (not including stake return)."""
    if pd.isna(p) or pd.isna(odds):
        return np.nan
    o = float(odds)
    # profit if win (per $100)
    win_profit = o if o > 0 else 10000.0 / (-o)
    return p * win_profit - (1 - p) * 100.0

def pretty_market(m):
    if pd.isna(m):
        return m
    return PRETTY_MAP.get(str(m), str(m).replace("_"," ").title())

def derive_side(row):
    if row.get("market_std") in OU_MARKETS:
        p = row.get("model_prob")
        if pd.notna(p):
            return "Over" if float(p) >= 0.5 else "Under"
    if row.get("market_std") == "anytime_td":
        return row.get("side") or "Yes"
    return row.get("side")

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Unify naming: line / price / book / market / side / player
    if "line" not in df.columns and "point" in df.columns:
        df["line"] = df["point"]
    if "odds" not in df.columns and "price" in df.columns:
        df["odds"] = df["price"]
    if "price" not in df.columns and "odds" in df.columns:
        df["price"] = df["odds"]

    if "book" not in df.columns:
        for cand in ["bookmaker_title","bookmaker","book_name"]:
            if cand in df.columns:
                df["book"] = df[cand]
                break

    if "market" not in df.columns and "market_std" in df.columns:
        df["market"] = df["market_std"]

    # prefer player display name
    if "player" not in df.columns and "name" in df.columns:
        df["player"] = df["name"]

    # pretty label
    if "market_disp" not in df.columns:
        df["market_disp"] = df["market"].map(pretty_market) if "market" in df.columns else df.get("market_std", np.nan).map(pretty_market)

    # side
    if "side" not in df.columns:
        df["side"] = np.nan
    df["side"] = df.apply(derive_side, axis=1)

    # kick ET passthrough (if present)
    if "kick_et" not in df.columns and "kickoff" in df.columns:
        df["kick_et"] = df["kickoff"]

    # market prob from price
    if "mkt_prob" not in df.columns:
        df["mkt_prob"] = df["price"].map(american_odds_to_prob) if "price" in df.columns else df["odds"].map(american_odds_to_prob)

    # safe numeric
    for c in ["line","price","odds","model_prob","mkt_prob"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # friendly displays
    df["line_disp"] = df["line"].where(df["line"].notna(), "—") if "line" in df.columns else "—"
    df["mkt_prob_disp"] = (df["mkt_prob"]*100).round(1).map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    if "model_prob" in df.columns:
        df["model_prob_disp"] = (df["model_prob"]*100).round(1).map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    if "odds" in df.columns:
        df["odds"] = df["odds"].round(0)

    return df
def build_overview_df(df: pd.DataFrame) -> pd.DataFrame:
    # group by player × market_std (and game if available)
    keys = [k for k in ["player", "name_std", "market_std"] if k in df.columns]
    if "game" in df.columns:
        keys.append("game")

    if not keys:
        # nothing to group by; return empty frame with expected columns
        return pd.DataFrame(columns=[
            "market_std","game","player","market_disp",
            "mkt_prob","mkt_prob_disp","book_count",
            "cons_line","cons_line_disp",
        ])

    grouped = df.groupby(keys, dropna=False)

    # Build named aggregation dict only for cols that exist
    agg_dict = {}
    if "line" in df.columns:
        agg_dict["cons_line"] = ("line", "median")
    if "mkt_prob" in df.columns:
        agg_dict["mkt_prob"] = ("mkt_prob", "median")
    if "book" in df.columns:
        agg_dict["book_count"] = ("book", "nunique")

    g = grouped.agg(**agg_dict).reset_index()

    # If we couldn't compute book_count (no 'book' col), fallback to group size
    if "book_count" not in g.columns:
        g["book_count"] = grouped.size().values

    # Ensure 'game' column exists for display
    if "game" not in g.columns:
        g["game"] = df.get("game", np.nan)

    # Friendly market label
    if "market_std" in g.columns:
        g["market_disp"] = g["market_std"].map(pretty_market)
    elif "market" in g.columns:
        g["market_disp"] = g["market"].map(pretty_market)
    else:
        g["market_disp"] = np.nan

    # Displays
    if "cons_line" in g.columns:
        g["cons_line_disp"] = g["cons_line"].where(g["cons_line"].notna(), "—")
    else:
        g["cons_line"] = np.nan
        g["cons_line_disp"] = "—"

    if "mkt_prob" in g.columns:
        g["mkt_prob_disp"] = (g["mkt_prob"] * 100).round(1).map(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )
    else:
        g["mkt_prob"] = np.nan
        g["mkt_prob_disp"] = "—"

    # Keep one name column
    if "player" not in g.columns and "name_std" in g.columns:
        g["player"] = g["name_std"]

    cols = [
        "market_std","game","player","market_disp",
        "mkt_prob","mkt_prob_disp","book_count",
        "cons_line","cons_line_disp",
    ]
    return g[[c for c in cols if c in g.columns]].sort_values(
        [c for c in ["game","player","market_std"] if c in g.columns]
    ).reset_index(drop=True)

def build_value_df(df: pd.DataFrame, overview_df: pd.DataFrame) -> pd.DataFrame:
    # join consensus line for context
    on_keys = []
    if "player" in df.columns:
        on_keys.append("player")
    elif "name_std" in df.columns:
        on_keys.append("name_std")
    if "market_std" in df.columns:
        on_keys.append("market_std")

    left = df.copy()
    right_cols = [c for c in ["player","name_std","market_std"] if c in overview_df.columns]
    right = overview_df[right_cols + [c for c in ["cons_line","cons_line_disp"] if c in overview_df.columns]].drop_duplicates()

    merged = left.merge(right, how="left", on=[k for k in on_keys if k in left.columns]) if on_keys else left

    # core calcs (raw)
    if "model_prob" in merged.columns and "mkt_prob" in merged.columns:
        merged["edge_bps"] = (merged["model_prob"] - merged["mkt_prob"]) * 10000.0
    else:
        merged["edge_bps"] = np.nan

    if "model_prob" in merged.columns:
        merged["fair_odds"] = merged["model_prob"].map(prob_to_american_odds)
        merged["ev_per_100"] = merged.apply(
            lambda r: expected_value_per_100(r.get("model_prob"), r.get("odds")), axis=1
        )
    else:
        merged["fair_odds"] = np.nan
        merged["ev_per_100"] = np.nan

    # friendly market label
    if "market_disp" not in merged.columns and "market_std" in merged.columns:
        merged["market_disp"] = merged["market_std"].map(pretty_market)

    # displays (formatted)
    merged["line_disp"] = merged["line"].where(merged["line"].notna(), "—") if "line" in merged.columns else "—"

    def pct_disp(s):
        return s.map(lambda x: f"{round(x*100,1):.1f}%" if pd.notna(x) else "—")

    merged["mkt_prob_disp"]   = pct_disp(merged["mkt_prob"])   if "mkt_prob"   in merged.columns else "—"
    merged["model_prob_disp"] = pct_disp(merged["model_prob"]) if "model_prob" in merged.columns else "—"

    def signed_odds(x):
        if pd.isna(x): return "—"
        x = int(round(x))
        return f"+{x}" if x > 0 else str(x)

    merged["odds_disp"] = merged["odds"].map(signed_odds) if "odds" in merged.columns else "—"
    merged["fair_odds_disp"] = merged["fair_odds"].map(signed_odds)

    def bps_disp(x):
        if pd.isna(x): return "—"
        v = int(round(x))
        return f"{v}"

    merged["edge_bps_disp"] = merged["edge_bps"].map(bps_disp)

    def ev_disp(x):
        if pd.isna(x): return "—"
        return f"${x:,.1f}"

    merged["ev_per_100_disp"] = merged["ev_per_100"].map(ev_disp)

    # tidy order
    cols = [
        "game","player","market_disp","side","book",
        "line_disp","odds_disp","mkt_prob_disp","model_prob_disp","edge_bps_disp",
        "fair_odds_disp","ev_per_100_disp","cons_line_disp","kick_et"
    ]
    return merged[[c for c in cols if c in merged.columns]].sort_values(
        ["game","player","market_disp","book","line_disp"]
    ).reset_index(drop=True)


def html_table(df: pd.DataFrame, columns, table_id: str, klass: str) -> str:
    """Render a simple HTML table with only the selected columns (in order)."""
    cols = [c for c in columns if c in df.columns]
    # header
    th = "".join(f"<th>{c}</th>" for c in cols)
    # body
    rows = []
    for _, r in df.iterrows():
        tds = "".join(f"<td class='{c}'>{'' if pd.isna(r.get(c)) else r.get(c)}</td>" for c in cols)
        rows.append(f"<tr>{tds}</tr>")
    tb = "\n".join(rows)
    return f"""<table id="{table_id}" class="{klass}">
<thead><tr>{th}</tr></thead>
<tbody>
{tb}
</tbody>
</table>"""

FILTER_JS = """
<script>
function buildFilters(tableId){
  const table = document.getElementById(tableId);
  if(!table) return;

  // figure columns by class name
  function uniq(vals){ return [...new Set(vals.filter(v => v && v !== '—'))].sort(); }

  const rows = Array.from(table.querySelectorAll('tbody tr'));
  const games  = uniq(rows.map(tr => tr.querySelector('td.game')?.textContent.trim()));
  const mkts   = uniq(rows.map(tr => tr.querySelector('td.market_disp')?.textContent.trim()));
  const books  = uniq(rows.map(tr => tr.querySelector('td.book')?.textContent.trim()));
  // controls
  const wrap = document.createElement('div');
  wrap.className = 'fv-filter-bar';
  wrap.innerHTML = `
    <label>Game <select id="${tableId}-f-game"><option value="">All</option></select></label>
    <label>Market <select id="${tableId}-f-market"><option value="">All</option></select></label>
    <label>Book <select id="${tableId}-f-book"><option value="">All</option></select></label>
    <label>Player <input id="${tableId}-f-player" placeholder="Search player" /></label>
  `;
  table.parentNode.insertBefore(wrap, table);

  const gSel = wrap.querySelector(`#${tableId}-f-game`);
  const mSel = wrap.querySelector(`#${tableId}-f-market`);
  const bSel = wrap.querySelector(`#${tableId}-f-book`);
  const pInp = wrap.querySelector(`#${tableId}-f-player`);

  games.forEach(v => { const o=document.createElement('option'); o.value=v; o.textContent=v; gSel.appendChild(o); });
  mkts.forEach(v => { const o=document.createElement('option'); o.value=v; o.textContent=v; mSel.appendChild(o); });
  books.forEach(v => { const o=document.createElement('option'); o.value=v; o.textContent=v; bSel.appendChild(o); });

  function apply(){
    const g = gSel.value.toLowerCase();
    const m = mSel.value.toLowerCase();
    const b = bSel.value.toLowerCase();
    const p = pInp.value.toLowerCase();
    rows.forEach(tr => {
      const tgame   = (tr.querySelector('td.game')?.textContent||'').toLowerCase();
      const tmarket = (tr.querySelector('td.market_disp')?.textContent||'').toLowerCase();
      const tbook   = (tr.querySelector('td.book')?.textContent||'').toLowerCase();
      const tplayer = (tr.querySelector('td.player')?.textContent||'').toLowerCase();
      const ok = (!g || tgame===g) &&
                 (!m || tmarket===m) &&
                 (!b || tbook===b) &&
                 (!p || tplayer.includes(p));
      tr.style.display = ok ? '' : 'none';
    });
  }
  gSel.onchange = mSel.onchange = bSel.onchange = pInp.oninput = apply;
}
document.addEventListener('DOMContentLoaded', function(){
  buildFilters('overview-table');
  buildFilters('value-table');
});
</script

<script>
(function(){
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const gameSel   = $('#gameFilter');
  const mktSel    = $('#marketFilter');
  const bookSel   = $('#bookFilter');
  const playerInp = $('#playerFilter');

  const OVERVIEW_ROWS = $$('#overviewTable tbody tr');
  const VALUE_ROWS    = $$('#valueTable tbody tr');

  // Utilities
  const norm = (s) => (s ?? '').toString().toLowerCase();
  const includes = (hay, ndl) => norm(hay).includes(norm(ndl));

  // Rebuild options for a select from a Set<string>, try to preserve selection
  function rebuildOptions(select, values, keepAll=true) {
    const prev = select.value;
    const opts = ['All', ...Array.from(values).sort()];
    select.innerHTML = '';
    for (const v of keepAll ? opts : Array.from(values).sort()) {
      const op = document.createElement('option');
      op.value = v; op.textContent = v;
      select.appendChild(op);
    }
    // Restore if still present; otherwise fall back to first option
    const wanted = values.has(prev) || (prev === 'All' && keepAll) ? prev : (keepAll ? 'All' : Array.from(values)[0] || 'All');
    if (wanted) select.value = wanted;
  }

  // Collect current filter state
  function state() {
    return {
      game:  gameSel ? gameSel.value : 'All',
      market:mktSel  ? mktSel.value  : 'All',
      book:  bookSel ? bookSel.value : 'All',
      player: norm(playerInp ? playerInp.value : '')
    };
  }

  // Core row predicate per tab
  function matchOverviewRow(tr, s) {
    const g = tr.dataset.game || '';
    const m = tr.dataset.market || '';
    const p = tr.dataset.player || '';
    const booksCSV = tr.dataset.books || '';  // comma‐joined lowercased books
    const hasBook = (s.book === 'All') ? true : (','+booksCSV+',').includes(','+norm(s.book)+',');
    return (s.game === 'All'   || g === s.game)
        && (s.market === 'All' || m === s.market)
        && hasBook
        && (s.player === ''     || includes(p, s.player));
  }

  function matchValueRow(tr, s) {
    const g = tr.dataset.game || '';
    const m = tr.dataset.market || '';
    const p = tr.dataset.player || '';
    const b = tr.dataset.book || '';
    return (s.game === 'All'   || g === s.game)
        && (s.market === 'All' || m === s.market)
        && (s.book === 'All'   || b === norm(s.book))
        && (s.player === ''     || includes(p, s.player));
  }

  // Repopulate dependent dropdowns from *currently visible* rows
  function rebuildDependentOptions() {
    const s = state();

    // Use both tabs so options never "disappear" if one tab is empty
    const rows = OVERVIEW_ROWS.concat(VALUE_ROWS);

    const matchingForGame = rows.filter(tr => (s.game === 'All') || (tr.dataset.game === s.game));

    const markets = new Set();
    const books   = new Set();

    for (const tr of matchingForGame) {
      const tab = tr.dataset.tab;
      markets.add(tr.dataset.market || '');
      if (tab === 'overview') {
        const csv = (tr.dataset.books || '').split(',').filter(Boolean);
        csv.forEach(b => books.add(b));
      } else {
        const b = tr.dataset.book || '';
        if (b) books.add(b);
      }
    }

    // Build human labels from your pretty map if you have one server-side.
    // Client side we rebuild with raw keys; your existing <option> labels are fine.

    // Rebuild selects but preserve if possible
    if (mktSel)  rebuildOptions(mktSel,  markets, true);
    if (bookSel) rebuildOptions(bookSel, books,   true);
  }

  // Show/hide rows according to filters
  function applyFilters({rebuild=true} = {}) {
    const s = state();

    if (rebuild) {
      // When Game changes, clear player search to avoid hiding everything
      if (document.activeElement === gameSel && playerInp) playerInp.value = '';
      rebuildDependentOptions();
    }

    let anyOverview = false, anyValue = false;

    for (const tr of OVERVIEW_ROWS) {
      const ok = matchOverviewRow(tr, s);
      tr.style.display = ok ? '' : 'none';
      if (ok) anyOverview = true;
    }

    for (const tr of VALUE_ROWS) {
      const ok = matchValueRow(tr, s);
      tr.style.display = ok ? '' : 'none';
      if (ok) anyValue = true;
    }

    // Optional: gray out tables when empty
    $('#overviewTable')?.classList.toggle('empty', !anyOverview);
    $('#valueTable')?.classList.toggle('empty', !anyValue);
  }

  // Wire up events
  gameSel   && gameSel.addEventListener('change', () => applyFilters({rebuild:true}));
  mktSel    && mktSel.addEventListener('change',  () => applyFilters({rebuild:false}));
  bookSel   && bookSel.addEventListener('change', () => applyFilters({rebuild:false}));
  playerInp && playerInp.addEventListener('input',() => applyFilters({rebuild:false}));

  // Initial build
  rebuildDependentOptions();
  applyFilters({rebuild:false});
})();
</script>

"""
# === END HELPERS ===



def tabs_shell(tabs: dict) -> str:
    """
    Build a tabbed HTML shell.
    tabs = {"Tab Name": "<table html>", ...}
    """
    nav_html = '<ul class="tabs">' + "".join(
        f'<li><a href="#{k.lower()}">{k}</a></li>' for k in tabs
    ) + "</ul>"

    body_html = "".join(
        f'<div id="{k.lower()}" class="tab-content" style="display:none;">{v}</div>'
        for k, v in tabs.items()
    )

    js = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll(".tabs a");
  const contents = document.querySelectorAll(".tab-content");
  function showTab(hash) {
    contents.forEach(c => c.style.display = (c.id === hash.substring(1)) ? "" : "none");
    tabs.forEach(t => t.parentElement.classList.toggle("active", t.getAttribute("href") === hash));
  }
  tabs.forEach(t => {
    t.addEventListener("click", e => {
      e.preventDefault();
      showTab(t.getAttribute("href"));
    });
  });
  if (tabs.length > 0) showTab(tabs[0].getAttribute("href"));
});
</script>
"""
    return nav_html + body_html + js


def load_and_merge(merged_csv: str, week: int):
    df = pd.read_csv(merged_csv)

    # Derive "side" if missing
    if "side" not in df.columns:
        def derive_side(r):
            if r["market_std"] == "anytime_td":
                return "Yes"
            return "Over" if r.get("model_prob", 0.5) >= 0.5 else "Under"
        df["side"] = df.apply(derive_side, axis=1)

    # Drop rows missing key values
    keep = df["model_prob"].notna() & df["mkt_prob"].notna()
    df = df.loc[keep].copy()

    # Add Game column
    if "game" not in df.columns:
        if {"home_team", "away_team"}.issubset(df.columns):
            df["game"] = df["away_team"] + " @ " + df["home_team"]
        else:
            df["game"] = ""

    # Pretty market names
    df["market_disp"] = df["market_std"].apply(pretty_market)

    return df


def make_overview_table(df: pd.DataFrame) -> str:
    """Consensus overview by player × market"""
    agg_dict = {
        "game": ("game", "first"),
        "player": ("player", "first"),
        "market_disp": ("market_disp", "first"),
        "mkt_prob": ("mkt_prob", "median"),
        "book_count": ("bookmaker", "nunique"),
    }

    if "market_cons_line" in df.columns:
        agg_dict["cons_line"] = ("market_cons_line", "first")
    elif "line" in df.columns:
        agg_dict["cons_line"] = ("line", "median")

    sub = df.groupby(["game", "player", "market_std"], as_index=False).agg(**agg_dict)

    # Format
    if "cons_line" in sub.columns:
        sub["cons_line"] = sub["cons_line"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )
    else:
        sub["cons_line"] = "—"

    sub["mkt_prob"] = sub["mkt_prob"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )

    html = sub.to_html(
        classes="sortable filterable",
        escape=False,
        index=False,
        table_id="overview-table",
    )
    return html


def make_value_table(df: pd.DataFrame) -> str:
    """Per-book value view with model vs market"""

    # Define possible columns in order
    cols = [
        "game",
        "player",
        "market_disp",
        "side",
        "line",              # may or may not exist
        "odds",
        "bookmaker_title",
        "model_prob",
        "mkt_prob",
        "edge_bps",
    ]
    # Keep only those actually present
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()

    # Format only if column exists
    if "line" in sub.columns:
        sub["line"] = sub["line"].fillna("—")
    if "odds" in sub.columns:
        sub["odds"] = sub["odds"].fillna("—")
    if "model_prob" in sub.columns:
        sub["model_prob"] = sub["model_prob"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "mkt_prob" in sub.columns:
        sub["mkt_prob"] = sub["mkt_prob"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "edge_bps" in sub.columns:
        sub["edge_bps"] = sub["edge_bps"].map(
            lambda x: f"{x:.0f}" if pd.notna(x) else "—"
        )

    html = sub.to_html(
        classes="sortable filterable",
        escape=False,
        index=False,
        table_id="value-table",
    )
    return html


def inject_filters_js() -> str:
    """Shared filters: Game, Market, Book (dropdowns), Player (text)"""
    return """
<script>
function injectFilters() {
  const controls = `
    <div id="filters" style="margin:10px 0;">
      <label>Game <select id="filter-game"><option value="">All</option></select></label>
      <label>Market <select id="filter-market"><option value="">All</option></select></label>
      <label>Book <select id="filter-book"><option value="">All</option></select></label>
      <label>Player <input type="text" id="filter-player" placeholder="Search player"></label>
    </div>`;
  document.querySelector('#overview-table').insertAdjacentHTML('beforebegin', controls);

  function getUnique(colId) {
    const vals = new Set();
    document.querySelectorAll('#overview-table tbody tr').forEach(tr => {
      const td = tr.querySelector(`td:nth-child(${colId})`);
      if (td) vals.add(td.innerText.trim());
    });
    return Array.from(vals).sort();
  }

  // Populate dropdowns (Game=1st col, Market=3rd, Book ~ Value tab col 7)
  for (const val of getUnique(1)) {
    document.querySelector('#filter-game').insertAdjacentHTML('beforeend', `<option>${val}</option>`);
  }
  for (const val of getUnique(3)) {
    document.querySelector('#filter-market').insertAdjacentHTML('beforeend', `<option>${val}</option>`);
  }

  // Book options from Value table col 7
  const books = new Set();
  document.querySelectorAll('#value-table tbody tr').forEach(tr => {
    const td = tr.querySelector('td:nth-child(7)');
    if (td) books.add(td.innerText.trim());
  });
  for (const val of Array.from(books).sort()) {
    document.querySelector('#filter-book').insertAdjacentHTML('beforeend', `<option>${val}</option>`);
  }

  function applyFilters() {
    const g = document.querySelector('#filter-game').value.toLowerCase();
    const m = document.querySelector('#filter-market').value.toLowerCase();
    const b = document.querySelector('#filter-book').value.toLowerCase();
    const p = document.querySelector('#filter-player').value.toLowerCase();

    function rowMatch(tr, cols) {
      const tds = tr.querySelectorAll('td');
      return (!g || tds[0].innerText.toLowerCase().includes(g)) &&
             (!m || tds[2].innerText.toLowerCase().includes(m)) &&
             (!p || tds[1].innerText.toLowerCase().includes(p)) &&
             (!b || (tds.length >= 7 && tds[6].innerText.toLowerCase().includes(b)));
    }

    document.querySelectorAll('#overview-table tbody tr').forEach(tr => {
      tr.style.display = rowMatch(tr) ? '' : 'none';
    });
    document.querySelectorAll('#value-table tbody tr').forEach(tr => {
      tr.style.display = rowMatch(tr) ? '' : 'none';
    });
  }

  ['#filter-game','#filter-market','#filter-book','#filter-player'].forEach(sel => {
    document.querySelector(sel).addEventListener('input', applyFilters);
    document.querySelector(sel).addEventListener('change', applyFilters);
  });
}
document.addEventListener('DOMContentLoaded', injectFilters);
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    out_path = Path(args.out)

    # Load + harmonize (keep your helper if you like, then normalize)
    try:
        merged = load_and_merge(args.merged_csv, args.week)
    except Exception:
        # Fallback if load_and_merge isn’t available/necessary
        merged = pd.read_csv(args.merged_csv)

    merged = harmonize_columns(merged)

    # Build Overview + Value DataFrames
    overview_df = build_overview_df(merged)
    value_df    = build_value_df(merged, overview_df)

    # Column orders for display
    overview_cols = [
        "market_std","game","player","market_disp",
        "mkt_prob_disp","book_count","cons_line_disp",
    ]

    value_cols = [
    "game","player","market_disp","side","book",
    "line_disp","odds_disp","mkt_prob_disp","model_prob_disp","edge_bps_disp",
    "fair_odds_disp","ev_per_100_disp","cons_line_disp","kick_et",
    ]


    # Render tables with stable ids/classes (for filters + CSS)
    html_overview = html_table(
        overview_df, overview_cols, table_id="overview-table",
        klass="props-table consensus-table"
    )
    html_value = html_table(
        value_df, value_cols, table_id="value-table",
        klass="props-table consensus-table"
    )

    # Simple two-tab shell; IDs match filter JS expectations
    tabs_html = f"""
    <ul class="fv-tabs">
      <li><a href="#overview">Overview</a></li>
      <li><a href="#value">Value</a></li>
    </ul>

    <div id="overview" class="tab-section">
      {html_overview}
    </div>

    <div id="value" class="tab-section" style="display:none">
      {html_value}
    </div>
    """


    # Append inline filter JS that targets both tables by id
    page = TABS_CSS + tabs_html + FILTER_JS + TABS_JS



    title = args.title or f"Fourth & Value — Consensus (Week {args.week}, {args.season})"

    print("[consensus] overview_df:", list(overview_df.columns), "rows:", len(overview_df))
    print("[consensus] value_df:",    list(value_df.columns),    "rows:", len(value_df))

    write_with_nav_raw(out_path.as_posix(), title, page, active="Consensus")


if __name__ == "__main__":
    main()

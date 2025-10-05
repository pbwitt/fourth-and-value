#!/usr/bin/env python3
"""Build the consensus props page with Overview and Value tabs."""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path
from typing import Optional, Sequence
import re

import pandas as pd

from site_common import (
    write_with_nav_raw,
    pretty_market,
    fmt_odds_american,
    fmt_pct,
    kickoff_et,
    american_to_prob,
    prob_to_american, BRAND
)


DISPLAY_DASH = "—"
MODEL_LINE_CANDIDATES: Sequence[str] = (
    "model_line",
    "model_mu",
    "model_p50",
    "mu",
)

TABS_CSS = """
<style>
:root {
  --bg:#0b0b10;
  --card:#14141c;
  --muted:#9aa0a6;
  --text:#e8eaed;
  --border:#23232e;
}

body {
  margin: 0;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}

.small {
  color: var(--muted);
  font-size: 12px;
}

.card {
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: 0 0 0 1px rgba(255,255,255,.02), 0 12px 40px rgba(0,0,0,.35);
}

.fv-filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin: 0 0 16px 0;
}

.fv-filter-bar label {
  font-size: 14px;
  display: inline-flex;
  flex-direction: column;
  gap: 4px;
  color: var(--muted);
}

.fv-filter-bar select, .fv-filter-bar input {
  background: var(--card);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
  min-width: 110px;
  outline: none;
}

.fv-filter-bar select:focus, .fv-filter-bar input:focus {
  border-color: #6ee7ff;
  box-shadow: 0 0 0 3px rgba(110,231,255,.15);
}

.table-wrap {
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
}

table.consensus-table {
  border-collapse: collapse;
  width: 100%;
  min-width: 1000px;
  background: var(--card);
  color: var(--text);
}

table.consensus-table th,
table.consensus-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}

table.consensus-table th {
  text-align: left;
  position: sticky;
  top: 0;
  background: var(--card);
  z-index: 1;
  font-size: 12px;
  color: var(--muted);
  letter-spacing: .2px;
  font-weight: 600;
}

table.consensus-table td {
  text-align: left;
}

table.consensus-table td.right {
  text-align: right !important;
  font-variant-numeric: tabular-nums;
}

table.consensus-table tr:hover td {
  background: rgba(255,255,255,.02);
}

table.consensus-table .edge-strong td {
  font-weight: 600;
}

table.consensus-table a {
  color: inherit;
  text-decoration: none;
}

</style>

"""

TABS_JS = """
<script>
function showTab(id) {
  document.querySelectorAll('.tab-section').forEach(el => el.style.display = 'none');
  const el = document.getElementById(id);
  if (el) el.style.display = '';
  document.querySelectorAll('.fv-tabs a').forEach(a => {
    if (a.getAttribute('href') === '#' + id) a.classList.add('active');
    else a.classList.remove('active');
  });
}
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.fv-tabs a').forEach(a => {
    a.addEventListener('click', function(evt) {
      evt.preventDefault();
      const id = this.getAttribute('href').slice(1);
      history.replaceState(null, '', '#' + id);
      showTab(id);
    });
  });
  const initial = (location.hash || '#overview').slice(1);

  // after you compute tabs & sections
const sections = document.querySelectorAll('.tab-section');
const tabs = document.querySelectorAll('.fv-tabs a');

function show(hash) {
  sections.forEach(s => s.classList.remove('active'));
  const sec = document.querySelector(hash);
  if (sec) sec.classList.add('active');
  tabs.forEach(a => a.classList.toggle('active', a.getAttribute('href') === hash));
}

// init: if nothing is active (single-tab case), activate the first section
const currentActive = document.querySelector('.tab-section.active');
if (!currentActive && sections.length) {
  sections[0].classList.add('active');
  const firstHash = '#' + sections[0].id;
  tabs.forEach(a => a.classList.toggle('active', a.getAttribute('href') === firstHash));
}

  showTab(initial);
});
</script>
"""

FILTER_BAR_HTML = """
<div class="small" style="margin-bottom:16px;">Select Game → Market → Book → Player filters below. Consensus line = median across books; Market % = de-vigged implied probability. <strong>Fair Odds</strong> = model-implied fair price. <strong>Edge</strong> = Model % − Market % (basis points).</div>

<div class="fv-filter-bar" id="fv-filter-bar">
  <label>Game <select id="filter-game"><option value="">All</option></select></label>
  <label>Market <select id="filter-market"><option value="">All</option></select></label>
  <label>Book <select id="filter-book"><option value="">All</option></select></label>
  <label>Player <input type="text" id="filter-player" placeholder="Search player" /></label>
</div>
"""

FILTER_JS = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  const gameSel = document.getElementById('filter-game');
  const marketSel = document.getElementById('filter-market');
  const bookSel = document.getElementById('filter-book');
  const playerInput = document.getElementById('filter-player');
  if (!gameSel || !marketSel || !bookSel || !playerInput) return;

  const rows = Array.from(document.querySelectorAll('table.consensus-table tbody tr'));
  const bookEntries = new Map();
  const marketEntries = new Map();
  const gameEntries = new Set();

  function addBook(key, label) {
    if (!key) return;
    if (!bookEntries.has(key)) {
      bookEntries.set(key, label || key);
    }
  }

  rows.forEach(row => {
    const game = row.dataset.game || '';
    if (game) gameEntries.add(game);
    const marketKey = row.dataset.market || '';
    const marketLabel = row.dataset.marketLabel || marketKey;
    if (marketKey) marketEntries.set(marketKey, marketLabel);

    if (row.dataset.book) {
      addBook(row.dataset.book, row.dataset.bookLabel || row.dataset.book);
    }
    const bookList = row.dataset.books ? row.dataset.books.split(',') : [];
    const bookLabelList = row.dataset.bookLabels ? row.dataset.bookLabels.split('|') : [];
    bookList.forEach((key, idx) => {
      const label = bookLabelList[idx] || key;
      addBook(key.trim(), label.trim());
    });
  });

  function fillSelect(sel, entries) {
    const opts = Array.from(entries instanceof Map ? entries.entries() : Array.from(entries).map(v => [v,v]));
    opts.sort((a,b) => a[1].localeCompare(b[1]));
    opts.forEach(([key, label]) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = label;
      sel.appendChild(opt);
    });
  }

  fillSelect(gameSel, gameEntries);
  fillSelect(marketSel, marketEntries);
  fillSelect(bookSel, bookEntries);

  function applyFilters() {
    const gameVal = gameSel.value.toLowerCase();
    const marketVal = marketSel.value.toLowerCase();
    const bookVal = bookSel.value.toLowerCase();
    const playerVal = playerInput.value.toLowerCase();

    let visible = 0;
    rows.forEach(row => {
      const game = (row.dataset.game || '').toLowerCase();
      const market = (row.dataset.market || '').toLowerCase();
      const player = (row.dataset.player || '').toLowerCase();
      const bookData = (row.dataset.book || '').toLowerCase();
      const booksData = (row.dataset.books || '').toLowerCase();

      const gameMatch = !gameVal || game === gameVal;
      const marketMatch = !marketVal || market === marketVal;
      const playerMatch = !playerVal || player.includes(playerVal);
      const bookMatch = !bookVal || bookData === bookVal || booksData.includes(bookVal);

      if (gameMatch && marketMatch && playerMatch && bookMatch) {
        row.style.display = '';
        visible++;
      } else {
        row.style.display = 'none';
      }
    });
  }

  gameSel.addEventListener('change', applyFilters);
  marketSel.addEventListener('change', applyFilters);
  bookSel.addEventListener('change', applyFilters);
  playerInput.addEventListener('input', applyFilters);
});
</script>
"""


# ─────────────────────────────────────────────────────────────────────
# Data harmonization & display helpers
# ─────────────────────────────────────────────────────────────────────

def canonical_str(x) -> str:
    """Make NaN/None → empty string, else str."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def display_value(x) -> str:
    """Format cell value: NaN → DISPLAY_DASH, numeric → formatted, else str."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DISPLAY_DASH
    if isinstance(x, (int, float)):
        if math.isnan(x):
            return DISPLAY_DASH
        # Integers or floats with .0
        if isinstance(x, int) or x == int(x):
            return str(int(x))
        return f"{x:.2f}"
    return str(x).strip()


def fmt_line(val):
    """Format a line/point value for display."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return DISPLAY_DASH
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return f"{f:.1f}"
    except:
        return str(val)


def fmt_edge_bps(val):
    """Format edge in basis points."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return DISPLAY_DASH
    try:
        return f"{int(val):+d}"
    except:
        return DISPLAY_DASH


def fmt_odds_dash(val):
    """Format odds with DISPLAY_DASH fallback."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return DISPLAY_DASH
    s = fmt_odds_american(val)
    return s if s else DISPLAY_DASH


def norm_key(s) -> str:
    """Normalize a string for use as a data attribute key."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def choose_player(row: pd.Series) -> str:
    """Pick best available player name from row."""
    for col in ["name_std", "player_name", "player"]:
        if col in row.index:
            val = row[col]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                s = str(val).strip()
                if s and s.lower() not in {"nan", "none", ""}:
                    return s
    return ""


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize column names and add display columns.
    """
    df = df.copy()

    # Alias columns
    if "bookmaker_title" in df.columns and "bookmaker" not in df.columns:
        df["bookmaker"] = df["bookmaker_title"]
    if "player" not in df.columns and "name_std" in df.columns:
        df["player"] = df["name_std"]
    if "market" not in df.columns and "market_std" in df.columns:
        df["market"] = df["market_std"]
    if "side" not in df.columns and "name" in df.columns:
        df["side"] = df["name"]

    # Line columns
    if "mu" in df.columns:
        df["model_line"] = df["mu"]
    if "point" in df.columns:
        df["book_line"] = df["point"]

    # Display columns
    if "model_prob" in df.columns:
        df["model_prob_disp"] = df["model_prob"].apply(fmt_pct)
    if "consensus_prob" in df.columns:
        df["consensus_prob_disp"] = df["consensus_prob"].apply(fmt_pct)
    if "price" in df.columns:
        df["book_price_disp"] = df["price"].apply(fmt_odds_american)

    # Game display
    def make_game(row):
        for c in ["game", "game_display", "matchup"]:
            if c in row.index and pd.notna(row[c]):
                return str(row[c]).strip()
        away = str(row.get("away_team", "")).strip()
        home = str(row.get("home_team", "")).strip()
        if away and home:
            return f"{away} @ {home}"
        return away or home or ""

    df["game"] = df.apply(make_game, axis=1)

    # Kickoff
    if "commence_time" in df.columns and "kick_et" not in df.columns:
        df["kick_et"] = df["commence_time"].apply(kickoff_et)

    return df


# ─────────────────────────────────────────────────────────────────────
# Build dataframes for each tab
# ─────────────────────────────────────────────────────────────────────

def build_overview_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Overview tab: all props, one row per (player, market, side, book).
    """
    return df.copy()


def build_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Value tab: consensus picks with quality filters.

    Criteria:
    1. Consensus agreement: book_count >= 2, consensus_prob not null
    2. Model coverage: model_prob not null
    3. Directional edge: edge_bps >= 25 and positive
    4. Execution: one row per (player, market, side) - keep best edge
    5. Presentation: hide unmodeled markets, missing side, "No Scorer"
    """
    value = df.copy()

    # 1. Consensus agreement
    if "book_count" in value.columns:
        value = value[value["book_count"] >= 2].copy()
    if "consensus_prob" in value.columns:
        value = value[value["consensus_prob"].notna()].copy()

    # 2. Model coverage (exclude unmodeled props)
    if "model_prob" in value.columns:
        value = value[value["model_prob"].notna()].copy()

    # 3. Directional edge
    if "edge_bps" in value.columns:
        value = value[value["edge_bps"] >= 25].copy()
        value = value[value["edge_bps"] > 0].copy()

    # 4. Exclude unmodeled markets
    unmodeled_markets = {"first_td", "last_td", "1st_td", "rush_longest", "reception_longest"}
    if "market_std" in value.columns:
        value = value[~value["market_std"].isin(unmodeled_markets)].copy()

    # 5. Hide missing side and "No Scorer"
    if "side" in value.columns:
        value = value[value["side"].notna()].copy()
        value = value[value["side"] != "nan"].copy()
        value = value[value["side"].str.strip() != ""].copy()
    if "player" in value.columns:
        value = value[value["player"].astype(str).str.lower() != "no scorer"].copy()

    # Keep one row per (player, market, side) - the one at consensus line with best price
    if len(value) > 0:
        group_keys = ["player", "market_std"]
        if "side" in value.columns:
            group_keys.append("side")

        # For each group, find the row at consensus_line with lowest mkt_prob (best price)
        def pick_representative(group):
            # Filter to rows at consensus line (within 0.5 tick)
            if "consensus_line" in group.columns and "point" in group.columns:
                at_consensus = group[
                    (group["point"].notna()) &
                    (group["consensus_line"].notna()) &
                    (abs(group["point"] - group["consensus_line"]) < 0.5)
                ]
                if not at_consensus.empty:
                    group = at_consensus

            # Among those, pick lowest mkt_prob (best price for bettor)
            # Tie-break by highest edge_bps
            if "mkt_prob" in group.columns:
                group = group.sort_values(["mkt_prob", "edge_bps"], ascending=[True, False])
            elif "edge_bps" in group.columns:
                group = group.sort_values("edge_bps", ascending=False)

            return group.iloc[0]

        value = value.groupby(group_keys, as_index=False).apply(pick_representative).reset_index(drop=True)

        # Sort by edge for display
        if "edge_bps" in value.columns:
            value = value.sort_values("edge_bps", ascending=False)

    return value


# ─────────────────────────────────────────────────────────────────────
# HTML table renderers
# ─────────────────────────────────────────────────────────────────────

def html_attrs(d: dict) -> str:
    """Convert dict to HTML attributes string."""
    parts = []
    for k, v in d.items():
        if v is not None:
            parts.append(f'{k}="{escape(str(v))}"')
    return " " + " ".join(parts) if parts else ""


def _cell_value(row_dict: dict, accessor):
    """Extract cell value via accessor (callable or key)."""
    if callable(accessor):
        return accessor(row_dict)
    return row_dict.get(accessor)


def render_overview_table(df: pd.DataFrame) -> str:
    """
    Render the Overview table (all props).
    Columns: Game, Player, Market, Side, Line, Book/Price, Model %, Market %, Edge, Fair Odds, Kick
    """
    def book_cell(row: dict) -> str:
        book = canonical_str(row.get("bookmaker_title")) or canonical_str(row.get("bookmaker"))
        price = canonical_str(row.get("book_price_disp"))
        if price and price != DISPLAY_DASH and not str(price).startswith("("):
            price = f"({price})"
        text = " ".join(part for part in (book, price) if part)
        return text if text else DISPLAY_DASH

    columns = [
        ("Game",   lambda r: canonical_str(r.get("game")) or canonical_str(r.get("game_display")), False, "game"),
        ("Player", lambda r: choose_player(pd.Series(r)), False, "player"),
        ("Market", lambda r: canonical_str(r.get("market")), False, "market"),
        ("Side",   lambda r: canonical_str(r.get("side")).title(), False, "side"),
        ("Line",   lambda r: fmt_line(
            r.get("book_line") if r.get("book_line") is not None else r.get("line")
        ), True, "line"),
        ("Book / Price", book_cell, False, "price"),
        ("Model %",  lambda r: canonical_str(r.get("model_prob_disp")) or DISPLAY_DASH, True, "model-pct"),
        ("Market %", lambda r: canonical_str(r.get("consensus_prob_disp")) or DISPLAY_DASH, True, "mkt-pct"),
        ("Edge (bps)", lambda r: fmt_edge_bps(r.get("edge_bps")), True, "edge"),
        ("Fair Odds",  lambda r: fmt_odds_dash(prob_to_american(r.get("model_prob"))), True, "fair-odds"),
    ]

    header = "".join(f"<th>{escape(label)}</th>" for label, _, _, _ in columns)
    rows_html: list[str] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        tds: list[str] = []
        for label, accessor, is_num, key in columns:
            raw = _cell_value(row_dict, accessor)
            value = display_value(raw)
            cls = "num" if is_num else ""
            is_html_cell = accessor is book_cell or (isinstance(value, str) and value.startswith("<"))
            if is_html_cell:
                tds.append(f'<td class="{cls}">{value}</td>')
            else:
                tds.append(f'<td class="{cls}">{escape(value)}</td>')

        # Optional: style "big edge" rows
        row_attrs = {}
        ebps = row_dict.get("edge_bps")
        if isinstance(ebps, (int, float)) and abs(ebps) >= 1500:
            row_attrs["class"] = "edge-strong"

        rows_html.append(f"<tr{html_attrs(row_attrs)}>{''.join(tds)}</tr>")

    body = "\n".join(rows_html)
    return (
        '<table class="consensus-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_value_table(df: pd.DataFrame) -> str:
    def book_cell(row: dict) -> str:
        book = canonical_str(row.get("bookmaker_title")) or canonical_str(row.get("bookmaker"))
        price = canonical_str(row.get("book_price_disp"))
        if price and price != DISPLAY_DASH and not str(price).startswith("("):
            price = f"({price})"
        text = " ".join(part for part in (book, price) if part)
        return text if text else DISPLAY_DASH

    columns = [
        ("Game",   lambda r: canonical_str(r.get("game")) or canonical_str(r.get("game_display")), False, "game"),
        ("Player", lambda r: choose_player(pd.Series(r)), False, "player"),
        ("Market", lambda r: canonical_str(r.get("market")), False, "market"),
        ("Side",   lambda r: canonical_str(r.get("side")).title(), False, "side"),

        # ✅ Model vs Book vs Consensus Lines
        ("Model Line",        lambda r: fmt_line(r.get("model_line")), True,  "model-line"),
        ("Book Line",         lambda r: fmt_line(r.get("book_line") if r.get("book_line") is not None else r.get("line")), True, "book-line"),
        ("Market Cons. Line", lambda r: fmt_line(r.get("consensus_line")), True, "cons-line"),

        ("Book / Price", book_cell, False, "price"),
        ("Model %",   lambda r: canonical_str(r.get("model_prob_disp")) or DISPLAY_DASH, True, "model-pct"),
        ("Market %",  lambda r: canonical_str(r.get("consensus_prob_disp")) or DISPLAY_DASH, True, "mkt-pct"),
        ("Edge (bps)", lambda r: fmt_edge_bps(r.get("edge_bps")), True, "edge"),
        ("Fair Odds",  lambda r: fmt_odds_dash(prob_to_american(r.get("model_prob"))), True, "fair-odds"),
    ]

    header = "".join(f"<th>{escape(label)}</th>" for label, _, _, _ in columns)
    rows_html: list[str] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        display_player = choose_player(pd.Series(row_dict))

        # ✅ Row attributes for filters
        row_attrs = {
            "data-game":   norm_key(row_dict.get("game") or row_dict.get("game_display")),
            "data-market": norm_key(row_dict.get("market_std") or row_dict.get("market")),
            "data-book":   norm_key(row_dict.get("bookmaker_title")),
            "data-player": norm_key(display_player),
        }

        cells: list[str] = []
        for _, accessor, right_align, css_class in columns:
            if callable(accessor):
                try:
                    value = accessor(row_dict)
                except TypeError:
                    value = accessor(pd.Series(row_dict))
            else:
                value = row_dict.get(accessor)

            is_html_cell = isinstance(value, str) and value.startswith("<")
            class_parts: list[str] = []
            if css_class:
                class_parts.append(css_class)
            if right_align:
                class_parts.append("right")
            class_attr = f' class="{ " ".join(class_parts) }"' if class_parts else ""
            cell_html = value if is_html_cell else escape(canonical_str(value))
            cells.append(f"<td{class_attr}>{cell_html}</td>")

        # Optional highlight for large edges
        ebps = row_dict.get("edge_bps")
        if isinstance(ebps, (int, float)) and abs(ebps) >= 1500:
            row_attrs["class"] = (row_attrs.get("class", "") + " edge-strong").strip()

        rows_html.append(f"<tr{html_attrs(row_attrs)}>{''.join(cells)}</tr>")

    body = "\n".join(rows_html)

    # ✅ Restore the original hook that your filter JS expects
    return (
        '<table id="props-table" class="consensus-table props-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def build_tabs_html(overview_html: str, value_html: str, season: int = None, week: int = None) -> str:
    # Value-only (no tabs). Include CSS, wrap filters and table in cards.
    week_header = ""
    if week:
        if season:
            week_header = f'<h1 style="margin:0 0 12px;font-size:22px;font-weight:700;letter-spacing:.2px;color:#fff;">Week {week}, {season}</h1>'
        else:
            week_header = f'<h1 style="margin:0 0 12px;font-size:22px;font-weight:700;letter-spacing:.2px;color:#fff;">Week {week}</h1>'

    return f"""{TABS_CSS}
<div class="card">
{week_header}{FILTER_BAR_HTML}
</div>

<div class="card table-wrap">
{value_html}
</div>

{FILTER_JS}
"""



def build_tables(df):
    """
    For tonight: only build the Value table. Overview is disabled to avoid
    renderer dependencies (book_cell etc.). Returns empty overview.
    """
    value_df = build_value_df(df)
    value_html = render_value_table(value_df)

    # Empty overview (kept for return signature compatibility)
    overview_df = df.iloc[0:0].copy()
    overview_html = ""

    return overview_df, value_df, overview_html, value_html


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the consensus props page")
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    merged_path = Path(args.merged_csv)
    df = pd.read_csv(merged_path)

    # Add 'side' column from 'name' if missing (BEFORE consensus join)
    if "side" not in df.columns and "name" in df.columns:
        df["side"] = df["name"]

    # Load consensus CSV and join (no normalization - exact match on join keys)
    consensus_path = merged_path.parent / f"consensus_week{args.week}.csv"
    if consensus_path.exists():
        consensus = pd.read_csv(consensus_path)
        # Strip whitespace only (no case changes)
        for col in ["name_std", "market_std", "side"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            if col in consensus.columns:
                consensus[col] = consensus[col].astype(str).str.strip()

        # Join on name_std, market_std, AND side (critical for correct matching)
        df = df.merge(
            consensus[["name_std", "market_std", "side", "consensus_line", "consensus_prob", "book_count"]],
            on=["name_std", "market_std", "side"],
            how="left",
            suffixes=("", "_cons")
        )

        # Acceptance check: print join stats
        match_rate = df["consensus_prob"].notna().mean()
        print(f"[consensus join] non-null Market % rows: {match_rate:.1%}")

        # Spot-check: McCaffrey rows
        mccaffrey = df[df["name_std"].str.contains("mccaffrey", case=False, na=False)]
        if not mccaffrey.empty:
            print("\n[spot-check] McCaffrey rows (first 6):")
            print(mccaffrey[["market_std", "point", "price", "consensus_line", "consensus_prob"]].head(6).to_string())
    else:
        print(f"[warning] Consensus file not found: {consensus_path}")

    df = harmonize(df)

    overview_df, value_df, overview_html, value_html = build_tables(df)
    page_html = build_tabs_html(overview_html, value_html, season=args.season, week=args.week)

    if args.season is not None:
        default_title = f"{BRAND} — Consensus (Week {args.week}, {args.season})"
    else:
        default_title = f"{BRAND} — Consensus (Week {args.week})"
    title = args.title or default_title

    if args.verbose:
        print(f"[consensus] overview rows: {len(overview_df)}")
        print(f"[consensus] value rows: {len(value_df)}")

    out_path = Path(args.out)
    write_with_nav_raw(out_path.as_posix(), title, page_html, active="Consensus")



if __name__ == "__main__":
    main()

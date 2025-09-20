#!/usr/bin/env python3
"""Build the consensus props page with Overview and Value tabs."""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from site_common import (
    BRAND,
    american_to_prob,
    fmt_odds_american,
    fmt_pct,
    kickoff_et,
    pretty_market,
    write_with_nav_raw,
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
.fv-tabs { list-style: none; padding: 0; margin: 0 0 10px 0; display: flex; gap: 14px; }
.fv-tabs li { display: inline; }
.fv-tabs a { text-decoration: none; font-weight: 600; color: inherit; }
.fv-tabs a.active { text-decoration: underline; }
.fv-tabs a:hover { text-decoration: underline; }

.fv-filter-bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0 16px; }
.fv-filter-bar label { font-size: 0.95rem; display: inline-flex; align-items: center; gap: 6px; }
.fv-filter-bar select, .fv-filter-bar input { padding: 4px 6px; }

.consensus-table { border-collapse: collapse; width: 100%; margin: 6px 0 24px; }
.consensus-table th, .consensus-table td { border: 1px solid rgba(255,255,255,0.15); padding: 6px 8px; }
.consensus-table th { text-align: left; }
.consensus-table td { vertical-align: middle; }
.consensus-table td.num { text-align: right; white-space: nowrap; }

/* best book highlight */
.best-book { color: #0a7d32; font-weight: 700; }
.best-price { font-variant-numeric: tabular-nums; }
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
  showTab(initial);
});
</script>
"""

FILTER_BAR_HTML = """
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
      const label = bookLabelList[idx] || bookLabelList[0] || key;
      addBook(key, label);
    });
  });

  function populateSelect(select, entries) {
    const items = Array.from(entries.entries());
    items.sort((a, b) => a[1].localeCompare(b[1]));
    items.forEach(([value, label]) => {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = label || value || '—';
      select.appendChild(opt);
    });
  }

  populateSelect(gameSel, new Map(Array.from(gameEntries).map(v => [v, v])));
  populateSelect(marketSel, marketEntries);
  populateSelect(bookSel, bookEntries);

  function matches(row, state) {
    const game = row.dataset.game || '';
    const market = row.dataset.market || '';
    const player = row.dataset.player || '';
    const book = row.dataset.book || '';
    const books = row.dataset.books || '';

    if (state.game && game !== state.game) return false;
    if (state.market && market !== state.market) return false;
    if (state.player && !player.includes(state.player)) return false;
    if (state.book) {
      if (book) {
        if (book !== state.book) return false;
      } else if (books) {
        if (!("," + books + ",").includes("," + state.book + ",")) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function applyFilters() {
    const state = {
      game: gameSel.value,
      market: marketSel.value,
      book: bookSel.value,
      player: playerInput.value.trim().toLowerCase(),
    };

    rows.forEach(row => {
      row.style.display = matches(row, state) ? '' : 'none';
    });
  }

  gameSel.addEventListener('change', applyFilters);
  marketSel.addEventListener('change', applyFilters);
  bookSel.addEventListener('change', applyFilters);
  playerInput.addEventListener('input', applyFilters);
});
</script>
"""

def prob_to_american(prob: float) -> float:
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return float("nan")
    if not (0 < p < 1):
        return float("nan")
    if p >= 0.5:
        return -round(p * 100.0 / (1.0 - p))
    return round((1.0 - p) * 100.0 / p)


def expected_value_per_100(prob: float, odds: float) -> float:
    try:
        p = float(prob)
        o = float(odds)
    except (TypeError, ValueError):
        return float("nan")
    if not (0 <= p <= 1):
        return float("nan")
    if math.isnan(p) or math.isnan(o):
        return float("nan")
    win_profit = o if o > 0 else 10000.0 / (-o)
    return p * win_profit - (1.0 - p) * 100.0


def fmt_pct_dash(prob: Optional[float]) -> str:
    if prob is None:
        return DISPLAY_DASH
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return DISPLAY_DASH
    if math.isnan(p):
        return DISPLAY_DASH
    text = fmt_pct(p)
    return text if text else DISPLAY_DASH


def fmt_odds_dash(odds: Optional[float]) -> str:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return DISPLAY_DASH
    if math.isnan(o):
        return DISPLAY_DASH
    text = fmt_odds_american(o)
    return text if text else DISPLAY_DASH


def fmt_line(value: Optional[float]) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        if value is None:
            return DISPLAY_DASH
        s = str(value).strip()
        return s if s else DISPLAY_DASH
    if math.isnan(f):
        return DISPLAY_DASH
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:g}"


def fmt_edge_bps(value: Optional[float]) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DISPLAY_DASH
    if math.isnan(v):
        return DISPLAY_DASH
    return f"{int(round(v)):+d}"


def fmt_ev(value: Optional[float]) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DISPLAY_DASH
    if math.isnan(v):
        return DISPLAY_DASH
    return f"${v:,.1f}"


def canonical_str(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def canonical_lower(value: Optional[str]) -> str:
    return canonical_str(value).lower()


def normalize_side(raw: Optional[str], market_std: Optional[str] = None) -> str:
    s = canonical_lower(raw)
    if not s:
        if market_std and str(market_std).lower() in {"anytime_td", "first_td", "last_td"}:
            return "Yes"
        return ""
    if "over" in s:
        return "Over"
    if "under" in s:
        return "Under"
    if "yes" in s:
        return "Yes"
    if "no" in s:
        return "No"
    return s.title()


def choose_player(row: pd.Series) -> str:
    for col in ("player", "name", "name_std"):
        if col in row and canonical_str(row[col]):
            return canonical_str(row[col])
    return ""


def choose_book(row: pd.Series) -> str:
    for col in ("book", "bookmaker_title", "bookmaker", "book_name"):
        if col in row and canonical_str(row[col]):
            return canonical_str(row[col])
    return ""


def choose_market_key(row: pd.Series) -> str:
    for col in ("market_std", "market"):
        if col in row and canonical_str(row[col]):
            return canonical_str(row[col])
    return ""


def choose_model_line(series: pd.Series) -> float:
    for col in MODEL_LINE_CANDIDATES:
        if col in series and pd.notna(series[col]):
            try:
                return float(series[col])
            except (TypeError, ValueError):
                continue
    return float("nan")


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "line" not in df.columns:
        for alt in ("point", "threshold", "points"):
            if alt in df.columns:
                df["line"] = df[alt]
                break
    if "price" not in df.columns and "odds" in df.columns:
        df["price"] = df["odds"]
    if "odds" not in df.columns and "price" in df.columns:
        df["odds"] = df["price"]
    if "book" not in df.columns:
        for alt in ("bookmaker_title", "bookmaker", "book_name"):
            if alt in df.columns:
                df["book"] = df[alt]
                break
    if "market" not in df.columns and "market_std" in df.columns:
        df["market"] = df["market_std"]
    if "player" not in df.columns:
        for alt in ("name", "name_std"):
            if alt in df.columns:
                df["player"] = df[alt]
                break
    if "game" not in df.columns and {"away_team", "home_team"}.issubset(df.columns):
        df["game"] = df[["away_team", "home_team"]].agg(
            lambda s: f"{canonical_str(s['away_team'])} @ {canonical_str(s['home_team'])}", axis=1
        )
    if "kickoff_et" not in df.columns:
        for alt in ("kickoff", "commence_time", "commence", "start_time"):
            if alt in df.columns:
                df["kickoff_et"] = df[alt].map(kickoff_et)
                break
    return df


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df)

    numeric_cols = [
        "line",
        "price",
        "odds",
        "mkt_prob",
        "model_prob",
        "edge_bps",
    ] + [c for c in MODEL_LINE_CANDIDATES if c in df.columns]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mkt_prob" not in df.columns and "odds" in df.columns:
        df["mkt_prob"] = df["odds"].map(american_to_prob)
    if "model_prob" in df.columns:
        df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    if "edge_bps" not in df.columns and {"model_prob", "mkt_prob"}.issubset(df.columns):
        df["edge_bps"] = (df["model_prob"] - df["mkt_prob"]) * 10000.0

    df["player_display"] = df.apply(choose_player, axis=1)
    df["player_key"] = df["player_display"].str.lower()

    df["book_display"] = df.apply(choose_book, axis=1)
    df["book_key"] = df["book_display"].str.lower()

    df["market_key"] = df.apply(choose_market_key, axis=1)
    df["market_disp_base"] = df["market_key"].map(pretty_market)

    df["side"] = df.apply(lambda r: normalize_side(r.get("side"), r.get("market_key")), axis=1)
    df["side_key"] = df["side"].str.lower()

    if "game" in df.columns:
        df["game_display"] = df["game"].fillna("")
    else:
        df["game_display"] = ""

    return df


def best_price_row(group: pd.DataFrame) -> Optional[pd.Series]:
    if group.empty:
        return None
    candidates = group.copy()
    if "edge_bps" in candidates.columns and candidates["edge_bps"].notna().any():
        candidates = candidates.sort_values(["edge_bps"], ascending=False)
    elif {"model_prob", "mkt_prob"}.issubset(candidates.columns):
        diff = (candidates["model_prob"] - candidates["mkt_prob"]).fillna(float("nan"))
        candidates = candidates.assign(_diff=diff).sort_values(["_diff"], ascending=False)
    elif "odds" in candidates.columns:
        candidates = candidates.sort_values(["odds"], ascending=False)
    return candidates.iloc[0]


def collect_best_book_ties(group: pd.DataFrame, best_row: Optional[pd.Series]) -> list[str]:
    if best_row is None:
        return []
    try:
        best_odds = float(best_row.get("odds"))
    except (TypeError, ValueError):
        return []
    if math.isnan(best_odds):
        return []
    best_book = canonical_str(best_row.get("book_display", ""))
    ties: set[str] = set()
    for _, candidate in group.iterrows():
        cand_book = canonical_str(candidate.get("book_display", ""))
        if not cand_book or cand_book == best_book:
            continue
        try:
            cand_odds = float(candidate.get("odds"))
        except (TypeError, ValueError):
            continue
        if math.isnan(cand_odds):
            continue
        if cand_odds == best_odds:
            ties.add(cand_book)
    return sorted(ties)


def aggregate_overview(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_display",
                "market_label",
                "cons_line_disp",
                "cons_prob_disp",
                "best_price_disp",
                "book_count_disp",
                "game_display",
                "market_key",
                "player_key",
                "side_key",
                "books",
                "book_labels",
            ]
        )

    group_cols = [c for c in ("game_display", "player_display", "market_key", "side") if c in df.columns]
    if not group_cols:
        group_cols = ["player_display", "market_key"]

    rows = []
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.copy()
        group["line"] = pd.to_numeric(group.get("line"), errors="coerce")
        group["mkt_prob"] = pd.to_numeric(group.get("mkt_prob"), errors="coerce")
        group["odds"] = pd.to_numeric(group.get("odds"), errors="coerce")

        cons_line = group["line"].median() if "line" in group else float("nan")
        cons_prob = group["mkt_prob"].median() if "mkt_prob" in group else float("nan")
        if "book_key" in group:
            valid_books = group["book_key"].fillna("").astype(str)
            book_count = valid_books[valid_books.str.len() > 0].nunique()
        else:
            book_count = len(group)

        best_row = best_price_row(group)
        best_book = canonical_str(best_row.get("book_display", "")) if best_row is not None else ""
        best_book_key = canonical_lower(best_row.get("book_key", "")) if best_row is not None else ""
        best_book_ties = collect_best_book_ties(group, best_row)

        best_price_value = best_row.get("odds") if best_row is not None else float("nan")
        best_price_disp = fmt_odds_dash(best_price_value)
        if best_price_disp == DISPLAY_DASH and not math.isnan(cons_prob):
            fallback = prob_to_american(cons_prob)
            best_price_disp = fmt_odds_dash(fallback)

        side_label = canonical_str(group["side"].iloc[0]) if "side" in group else ""
        market_label = canonical_str(group["market_disp_base"].iloc[0])
        if side_label:
            market_label = f"{market_label} — {side_label}"

        books_sorted = sorted({(row.get("book_key", ""), row.get("book_display", "")) for _, row in group.iterrows() if row.get("book_key")})
        book_keys = ",".join(key for key, _ in books_sorted)
        book_labels = "|".join(label for _, label in books_sorted)

        rows.append(
            {
                "player_display": group["player_display"].iloc[0],
                "market_label": market_label,
                "cons_line_disp": fmt_line(cons_line),
                "cons_prob_disp": fmt_pct_dash(cons_prob),
                "best_price_disp": best_price_disp,
                "best_book": best_book,
                "best_book_key": best_book_key,
                "best_book_ties": tuple(best_book_ties),
                "book_count_disp": str(int(book_count)) if not math.isnan(book_count) else DISPLAY_DASH,
                "game_display": group["game_display"].iloc[0],
                "market_key": group["market_key"].iloc[0],
                "player_key": group["player_key"].iloc[0],
                "side_key": group["side_key"].iloc[0] if "side_key" in group else "",
                "books": book_keys,
                "book_labels": book_labels,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["player_display", "market_label"]).reset_index(drop=True)


def aggregate_value(df: pd.DataFrame, overview: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_display",
                "market_disp_base",
                "side",
                "cons_line_disp",
                "cons_prob_disp",
                "model_line_disp",
                "model_prob_disp",
                "best_price_disp",
                "best_book",
                "best_book_key",
                "best_book_ties",
                "edge_bps_disp",
                "ev_disp",
                "game_display",
                "market_key",
                "player_key",
                "side_key",
                "book_key",
                "book_display",
            ]
        )

    overview_lookup = overview.set_index(["player_key", "market_key", "side_key"], drop=False)

    group_cols = [c for c in ("game_display", "player_display", "market_key", "side") if c in df.columns]
    if not group_cols:
        group_cols = ["player_display", "market_key"]

    rows = []
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.copy()
        group["line"] = pd.to_numeric(group.get("line"), errors="coerce")
        group["mkt_prob"] = pd.to_numeric(group.get("mkt_prob"), errors="coerce")
        group["model_prob"] = pd.to_numeric(group.get("model_prob"), errors="coerce")
        group["odds"] = pd.to_numeric(group.get("odds"), errors="coerce")
        group["edge_bps"] = pd.to_numeric(group.get("edge_bps"), errors="coerce")

        if {"model_prob", "odds"}.issubset(group.columns):
            group["ev_per_100"] = group.apply(
                lambda r: expected_value_per_100(r.get("model_prob"), r.get("odds")), axis=1
            )
        else:
            group["ev_per_100"] = float("nan")

        best_row = best_price_row(group)
        if best_row is None:
            best_row = group.iloc[0]

        best_odds = best_row.get("odds")
        best_book = canonical_str(best_row.get("book_display", ""))
        best_book_key = canonical_lower(best_row.get("book_key", ""))
        best_book_ties = collect_best_book_ties(group, best_row)
        best_edge = best_row.get("edge_bps")
        if pd.isna(best_edge) and {"model_prob", "mkt_prob"}.issubset(best_row.index):
            mp = best_row.get("model_prob")
            cp = best_row.get("mkt_prob")
            try:
                best_edge = (float(mp) - float(cp)) * 10000.0
            except (TypeError, ValueError):
                best_edge = float("nan")

        best_ev = best_row.get("ev_per_100")

        key = (
            best_row.get("player_key"),
            best_row.get("market_key"),
            best_row.get("side_key", ""),
        )
        cons_line_disp = DISPLAY_DASH
        cons_prob_disp = DISPLAY_DASH
        if key in overview_lookup.index:
            lookup_row = overview_lookup.loc[key]
            if isinstance(lookup_row, pd.DataFrame):
                lookup_row = lookup_row.iloc[0]
            cons_line_disp = lookup_row.get("cons_line_disp", DISPLAY_DASH)
            cons_prob_disp = lookup_row.get("cons_prob_disp", DISPLAY_DASH)

        model_line_val = choose_model_line(best_row)
        model_prob_val = best_row.get("model_prob")

        rows.append(
            {
                "player_display": best_row.get("player_display", ""),
                "market_disp_base": best_row.get("market_disp_base", ""),
                "side": best_row.get("side", ""),
                "cons_line_disp": cons_line_disp,
                "cons_prob_disp": cons_prob_disp,
                "model_line_disp": fmt_line(model_line_val),
                "model_prob_disp": fmt_pct_dash(model_prob_val),
                "best_price_disp": fmt_odds_dash(best_odds),
                "best_book": best_book,
                "best_book_key": best_book_key,
                "best_book_ties": tuple(best_book_ties),
                "edge_bps_disp": fmt_edge_bps(best_edge),
                "ev_disp": fmt_ev(best_ev),
                "game_display": best_row.get("game_display", ""),
                "market_key": best_row.get("market_key", ""),
                "player_key": best_row.get("player_key", ""),
                "side_key": best_row.get("side_key", ""),
                "book_key": best_book_key,
                "book_display": best_book,
            }
        )

    out = pd.DataFrame(rows)
    out["market_label"] = out.apply(
        lambda r: f"{canonical_str(r.get('market_disp_base'))} — {canonical_str(r.get('side'))}".strip(" —"),
        axis=1,
    )
    return out.sort_values(["player_display", "market_label"]).reset_index(drop=True)


def attr_value(value: Optional[object]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def html_attrs(attrs: dict) -> str:
    parts: list[str] = []
    for key, value in attrs.items():
        val = attr_value(value)
        if val == "" and key not in {"data-books", "data-book-labels"}:
            continue
        parts.append(f' {key}="{escape(val, quote=True)}"')
    return "".join(parts)


def display_value(value: Optional[object]) -> str:
    if value is None:
        return DISPLAY_DASH
    if isinstance(value, float) and math.isnan(value):
        return DISPLAY_DASH
    s = str(value)
    return s if s else DISPLAY_DASH


def make_book_price_cell(
    book: Optional[str], price_disp: Optional[str], ties: Optional[Sequence[str]] = None
) -> str:
    book_name = canonical_str(book)
    price_text = canonical_str(price_disp) or DISPLAY_DASH

    also: list[str] = []
    if ties:
        also = sorted({canonical_str(t) for t in ties if canonical_str(t) and canonical_str(t) != book_name})
    title_attr = f' title="Also: {escape(", ".join(also))}"' if also else ""

    if book_name:
        parts = [f'<span class="best-book"{title_attr}>{escape(book_name)}</span>']
        if price_text:
            parts.append(f'<span class="best-price mono">({escape(price_text)})</span>')
        return " ".join(parts).strip()

    if price_text and price_text != DISPLAY_DASH:
        return f'<span class="best-price mono">{escape(price_text)}</span>'

    return DISPLAY_DASH


def render_overview_table(df: pd.DataFrame) -> str:
    book_field = "__book_price__"
    columns = [
        ("Player", "player_display", False, "player"),
        ("Market", "market_label", False, "market"),
        ("Cons. Line", "cons_line_disp", True, "line"),
        ("Impl. %", "cons_prob_disp", True, "prob"),
        ("Book / Price", book_field, False, "price"),
        ("#Books", "book_count_disp", True, "books"),
    ]
    header = "".join(f"<th>{escape(label)}</th>" for label, _, _, _ in columns)
    rows_html: list[str] = []
    for row in df.itertuples(index=False):
        attrs = {
            "data-tab": "overview",
            "data-game": getattr(row, "game_display", ""),
            "data-market": getattr(row, "market_key", ""),
            "data-market-label": getattr(row, "market_label", ""),
            "data-player": getattr(row, "player_key", ""),
            "data-side": getattr(row, "side_key", ""),
            "data-books": getattr(row, "books", ""),
            "data-book-labels": getattr(row, "book_labels", ""),
        }
        cells: list[str] = []
        for label, field, is_num, css_class in columns:
            class_parts: list[str] = []
            if css_class:
                class_parts.append(css_class)
            if is_num:
                class_parts.append("num")
            class_attr = ""
            if class_parts:
                class_attr = f' class="{" ".join(class_parts)}"'
            if field == book_field:
                ties = getattr(row, "best_book_ties", ())
                if isinstance(ties, float) and math.isnan(ties):
                    ties = ()
                value_html = make_book_price_cell(
                    getattr(row, "best_book", ""),
                    getattr(row, "best_price_disp", DISPLAY_DASH),
                    ties,
                )
                cells.append(f"<td{class_attr}>{value_html}</td>")
            else:
                value = display_value(getattr(row, field, DISPLAY_DASH))
                cells.append(f"<td{class_attr}>{escape(value)}</td>")
        rows_html.append(f"<tr{html_attrs(attrs)}>{''.join(cells)}</tr>")
    body = "\n".join(rows_html)
    return (
        "<table id=\"overview-table\" class=\"consensus-table props-table\">"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_value_table(df: pd.DataFrame) -> str:
    book_field = "__book_price__"
    columns = [
        ("Player", "player_display", False, "player"),
        ("Market", "market_label", False, "market"),
        ("Market Cons. Line", "cons_line_disp", True, "line"),
        ("Impl. %", "cons_prob_disp", True, "prob"),
        ("Model Line (μ/P50)", "model_line_disp", True, "model-line"),
        ("Model Prob %", "model_prob_disp", True, "model-prob"),
        ("Book / Price", book_field, False, "price"),
        ("Edge (bps)", "edge_bps_disp", True, "edge"),
        ("EV/$100", "ev_disp", True, "ev"),
    ]
    header = "".join(f"<th>{escape(label)}</th>" for label, _, _, _ in columns)
    rows_html: list[str] = []
    for row in df.itertuples(index=False):
        attrs = {
            "data-tab": "value",
            "data-game": getattr(row, "game_display", ""),
            "data-market": getattr(row, "market_key", ""),
            "data-market-label": getattr(row, "market_label", ""),
            "data-player": getattr(row, "player_key", ""),
            "data-side": getattr(row, "side_key", ""),
            "data-book": getattr(row, "book_key", ""),
            "data-book-label": getattr(row, "book_display", ""),
        }
        cells: list[str] = []
        for label, field, is_num, css_class in columns:
            class_parts: list[str] = []
            if css_class:
                class_parts.append(css_class)
            if is_num:
                class_parts.append("num")
            class_attr = ""
            if class_parts:
                class_attr = f' class="{" ".join(class_parts)}"'
            if field == book_field:
                ties = getattr(row, "best_book_ties", ())
                if isinstance(ties, float) and math.isnan(ties):
                    ties = ()
                value_html = make_book_price_cell(
                    getattr(row, "best_book", ""),
                    getattr(row, "best_price_disp", DISPLAY_DASH),
                    ties,
                )
                cells.append(f"<td{class_attr}>{value_html}</td>")
            else:
                value = display_value(getattr(row, field, DISPLAY_DASH))
                cells.append(f"<td{class_attr}>{escape(value)}</td>")
        rows_html.append(f"<tr{html_attrs(attrs)}>{''.join(cells)}</tr>")
    body = "\n".join(rows_html)
    return (
        "<table id=\"value-table\" class=\"consensus-table props-table\">"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def build_tabs_html(overview_html: str, value_html: str) -> str:
    tabs = f"""
<ul class=\"fv-tabs\">
  <li><a href=\"#overview\">Overview</a></li>
  <li><a href=\"#value\">Value</a></li>
</ul>
<div id=\"overview\" class=\"tab-section\">
  {overview_html}
</div>
<div id=\"value\" class=\"tab-section\" style=\"display:none\">
  {value_html}
</div>
"""
    return TABS_CSS + FILTER_BAR_HTML + tabs + FILTER_JS + TABS_JS


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    overview_df = aggregate_overview(df)
    value_df = aggregate_value(df, overview_df)
    overview_html = render_overview_table(overview_df)
    value_html = render_value_table(value_df)
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
    df = harmonize(df)

    overview_df, value_df, overview_html, value_html = build_tables(df)
    page_html = build_tabs_html(overview_html, value_html)

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

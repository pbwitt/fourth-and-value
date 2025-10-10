#!/usr/bin/env python3
"""
Build NHL props page from props_with_model CSV.

Generates /docs/nhl/props/index.html with table and mobile card views.

Usage:
  python3 scripts/nhl/build_nhl_props_page.py --date 2025-10-08
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


def prob_to_american(p):
    """Convert probability to American odds."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or p <= 0 or p >= 1)):
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def fmt_odds(o):
    """Format American odds as +/- string."""
    if pd.isna(o):
        return ""
    try:
        o = int(round(float(o)))
        return f"{o:+d}"
    except Exception:
        return str(o)


def fmt_pct(p):
    """Format probability as percentage."""
    if pd.isna(p):
        return ""
    try:
        pct = float(p)
        return f"{pct:.1f}%"
    except Exception:
        return str(p)


def to_kick_et(iso_str):
    """Convert ISO8601 UTC time to 'Wed 7 p.m' ET format."""
    if not iso_str or pd.isna(iso_str):
        return ""
    try:
        ts = pd.to_datetime(iso_str, utc=True)
        local = ts.tz_convert("America/New_York")
        dow = local.strftime("%a")
        hour12 = (local.hour % 12) or 12
        minute = local.minute
        ampm = "a.m" if local.hour < 12 else "p.m"
        time_part = f"{hour12}" if minute == 0 else f"{hour12}:{minute:02d}"
        return f"{dow} {time_part} {ampm}"
    except Exception:
        return ""


def pretty_market(market_std):
    """Convert market_std to display name."""
    mapping = {
        "sog": "Shots on Goal",
        "goals": "Goals",
        "assists": "Assists",
        "points": "Points",
    }
    return mapping.get(market_std, market_std.title())


def main():
    parser = argparse.ArgumentParser(description="Build NHL props page")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print(f"[build_nhl_props_page] Building NHL props page for {date_str}...")

    # Load props with model
    props_path = Path(f"data/nhl/props/props_with_model_{date_str}.csv")
    if not props_path.exists():
        print(f"[ERROR] Props file not found: {props_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(props_path)
    print(f"[build_nhl_props_page] Loaded {len(df)} prop rows")

    # Filter to balanced top props per market (for performance)
    # Top 500 goals, 200 assists, 200 points, 200 sog = 1100 total
    df['abs_edge'] = df['edge_bps'].abs()

    top_goals = df[df['market_std'] == 'goals'].nlargest(500, 'abs_edge')
    top_assists = df[df['market_std'] == 'assists'].nlargest(200, 'abs_edge')
    top_points = df[df['market_std'] == 'points'].nlargest(200, 'abs_edge')
    top_sog = df[df['market_std'] == 'sog'].nlargest(200, 'abs_edge')

    df = pd.concat([top_goals, top_assists, top_points, top_sog]).drop(columns=['abs_edge'])
    print(f"[build_nhl_props_page] Filtered to {len(df)} props (500 goals, 200 assists, 200 points, 200 sog)")

    # Add display columns
    df["price_disp"] = df["price"].apply(fmt_odds)
    df["fair_odds_disp"] = df["fair_odds"].apply(fmt_odds)
    df["mkt_pct_disp"] = df["mkt_prob"].apply(lambda p: fmt_pct(p * 100) if not pd.isna(p) else "")
    df["model_pct_disp"] = df["model_prob"].apply(lambda p: fmt_pct(p * 100) if not pd.isna(p) else "")
    df["consensus_pct_disp"] = df["consensus_prob"].apply(lambda p: fmt_pct(p * 100) if not pd.isna(p) else "")
    df["edge_bps_disp"] = df["edge_bps"].apply(lambda e: f"{int(e)}" if not pd.isna(e) else "")
    df["market_disp"] = df["market_std"].apply(pretty_market)
    df["kick_et"] = df["commence_time"].apply(to_kick_et)

    # Line display with decimal if needed
    df["line_disp"] = df["point"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "")
    df["model_line_disp"] = df["model_line"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "")
    df["consensus_line_disp"] = df["consensus_line"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "")
    df["book_count_disp"] = df["book_count"].apply(lambda x: f"{int(x)}" if not pd.isna(x) else "")

    # Sort by abs edge (biggest edges first)
    df["abs_edge"] = df["edge_bps"].abs()
    df = df.sort_values("abs_edge", ascending=False)

    # Build JSON data for page
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "player": row["player"],
            "game": f"{row['away_team']} @ {row['home_team']}",
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "market_disp": row["market_disp"],
            "market_std": row["market_std"],
            "side": row["side"],
            "line_disp": row["line_disp"],
            "consensus_line_disp": row["consensus_line_disp"],
            "model_line_disp": row["model_line_disp"],
            "bookmaker": row["bookmaker_title"],
            "price_disp": row["price_disp"],
            "model_pct_disp": row["model_pct_disp"],
            "consensus_pct_disp": row["consensus_pct_disp"],
            "mkt_pct_disp": row["mkt_pct_disp"],
            "fair_odds_disp": row["fair_odds_disp"],
            "edge_bps_disp": row["edge_bps_disp"],
            "book_count_disp": row["book_count_disp"],
            "kick_et": row["kick_et"],
            "edge_bps": row["edge_bps"],
            "model_prob": row.get("model_prob", 0),
            "consensus_prob": row.get("consensus_prob", 0),
            "mkt_prob": row.get("mkt_prob", 0),
            "book_count": row.get("book_count", 0),
            "is_consensus": False,  # Will be calculated in JS based on edge + model/market agreement
        })

    # Write HTML page
    html = build_html(rows, date_str)

    output_dir = Path("docs/nhl/props")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"[build_nhl_props_page] Wrote {len(rows)} props to {output_path}")


def build_html(rows, date_str):
    """Build complete HTML page."""

    # Convert rows to JSON for JS
    import json
    rows_json = json.dumps(rows, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NHL Props - Fourth & Value</title>
  <style>
    /* Base styles */
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: #0f0f0f;
      color: #fff;
      line-height: 1.6;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem 1rem;
    }}
    h1 {{
      font-size: 2rem;
      margin-bottom: 0.5rem;
      color: #4FC3F7;
    }}
    p {{
      color: #999;
      margin-bottom: 1.5rem;
    }}

    /* Filters */
    .filters {{
      background: #2a2a2a;
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
    }}
    .filter-group {{
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }}
    .filter-group label {{
      font-size: 0.875rem;
      color: #999;
    }}
    .filter-group select, .filter-group input {{
      background: #1a1a1a;
      border: 1px solid #444;
      color: #fff;
      padding: 0.5rem;
      border-radius: 4px;
      font-size: 0.875rem;
    }}

    /* Table */
    .props-table {{
      width: 100%;
      border-collapse: collapse;
      background: #2a2a2a;
      border-radius: 8px;
      overflow: hidden;
    }}
    .props-table thead {{
      background: #1a1a1a;
    }}
    .props-table th {{
      padding: 0.75rem;
      text-align: left;
      font-size: 0.875rem;
      color: #999;
      font-weight: 600;
    }}
    .props-table td {{
      padding: 0.75rem;
      border-top: 1px solid #333;
      font-size: 0.875rem;
    }}
    .props-table tbody tr:hover {{
      background: #333;
    }}
    .consensus-row {{
      background: #1a2a3a;
    }}
    .consensus-row:hover {{
      background: #234a5a;
    }}

    /* Mobile cards */
    .card-grid {{
      display: none;
      gap: 1rem;
    }}
    .prop-card {{
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 8px;
      padding: 1rem;
    }}
    .prop-card.consensus {{
      background: #1a2a3a;
      border-color: #2a4a5a;
    }}
    .prop-card-header {{
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }}
    .prop-card-game {{
      font-size: 0.875rem;
      color: #999;
      margin-bottom: 0.5rem;
    }}
    .prop-card-bet {{
      font-size: 0.875rem;
      margin-bottom: 1rem;
    }}
    .prop-card-stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      font-size: 0.875rem;
    }}
    .prop-card-stats > div:nth-child(odd) {{
      color: #999;
    }}

    /* Multi-select book dropdown */
    .checkbox-dropdown {{ position: relative; min-width: 180px; }}
    .checkbox-dropdown-button {{ width: 100%; background: #14141c; color: #e8eaed; border: 1px solid #23232e; border-radius: 10px; padding: 10px 12px; outline: none; cursor: pointer; text-align: left; display: flex; justify-content: space-between; align-items: center; }}
    .checkbox-dropdown-button:hover {{ border-color: #6ee7ff; }}
    .checkbox-dropdown-button::after {{ content: '▼'; font-size: 10px; opacity: 0.6; }}
    .checkbox-dropdown.open .checkbox-dropdown-button::after {{ content: '▲'; }}
    .checkbox-group {{ display: none; position: absolute; z-index: 1000; flex-direction: column; gap: 6px; background: #14141c; border: 1px solid #23232e; border-radius: 10px; padding: 10px; max-height: 250px; overflow-y: auto; min-width: 100%; margin-top: 4px; box-shadow: 0 4px 12px rgba(0,0,0,.3); }}
    .checkbox-dropdown.open .checkbox-group {{ display: flex; }}
    .checkbox-item {{ display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 4px 6px; border-radius: 6px; transition: background .15s; }}
    .checkbox-item:hover {{ background: rgba(255,255,255,.05); }}
    .checkbox-item input[type="checkbox"] {{ width: 16px; height: 16px; cursor: pointer; accent-color: #34d399; }}
    .checkbox-item label {{ cursor: pointer; flex: 1; font-size: 13px; color: #e8eaed; margin: 0; white-space: nowrap; }}

    @media (max-width: 768px) {{
      .table-wrapper {{ display: none; }}
      .card-grid {{ display: flex; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div id="nav-root"></div>

  <div class="container">
    <h1>NHL Props - {date_str}</h1>
    <p>Model-driven NHL player prop picks with edge analysis.</p>

    <div class="filters">
      <div class="filter-group">
        <label>Game</label>
        <select id="gameFilter">
          <option value="">All Games</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Player</label>
        <input type="text" id="playerFilter" placeholder="Search player...">
      </div>

      <div class="filter-group">
        <label>Market</label>
        <select id="marketFilter">
          <option value="">All Markets</option>
          <option value="sog">Shots on Goal</option>
          <option value="goals">Goals</option>
          <option value="assists">Assists</option>
          <option value="points">Points</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Side</label>
        <select id="sideFilter">
          <option value="">Both</option>
          <option value="Over">Over</option>
          <option value="Under">Under</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Book</label>
        <div id="bookDropdown" class="checkbox-dropdown">
          <button type="button" id="bookButton" class="checkbox-dropdown-button">
            <span id="bookButtonText">All Books</span>
          </button>
          <div id="bookFilter" class="checkbox-group"></div>
        </div>
      </div>

      <div class="filter-group">
        <label style="color: #6ee7ff;">
          <input type="checkbox" id="consensusFilter">
          Consensus Only
        </label>
      </div>
    </div>

    <!-- Desktop table -->
    <div class="table-wrapper">
      <table class="props-table" id="propsTable">
        <thead>
          <tr>
            <th>Player</th>
            <th>Game</th>
            <th>Market</th>
            <th>Side</th>
            <th>Book Line</th>
            <th>Consensus Line</th>
            <th>Model Line</th>
            <th>Book</th>
            <th>Book Odds</th>
            <th>Fair Odds</th>
            <th>Book %</th>
            <th>Market %</th>
            <th>Model %</th>
            <th>Edge</th>
            <th>Books</th>
          </tr>
        </thead>
        <tbody id="propsTableBody">
        </tbody>
      </table>
    </div>

    <!-- Mobile cards -->
    <div class="card-grid" id="cardGrid">
    </div>
  </div>

  <script>
    const propsData = {rows_json};

    // State management
    const state = {{
      selectedBooks: new Set(),
      consensusOnly: false
    }};

    // Consensus logic: book diverges from market, model agrees with market
    function isConsensusPlay(row) {{
      // Need significant edge (book diverges from market)
      const edgeBps = Math.abs(row.edge_bps || 0);
      if (edgeBps < 200) return false; // Minimum 200 bps edge

      const modelProb = row.model_prob || 0;
      const consensusProb = row.consensus_prob || 0;
      const mktProb = row.mkt_prob || 0;

      // Use consensus if available, otherwise market
      const marketRef = consensusProb || mktProb;

      // Model and market must agree on direction vs book
      // Positive edge = model thinks probability is higher than book's implied prob
      // We want model and market both on same side
      const modelEdge = modelProb - marketRef;
      const bookEdge = row.edge_bps || 0;

      return Math.sign(modelEdge) === Math.sign(bookEdge);
    }}

    // Populate book filter with checkboxes
    function populateFilters() {{
      const games = [...new Set(propsData.map(r => r.game))].sort();
      const gameSelect = document.getElementById('gameFilter');
      games.forEach(game => {{
        const option = document.createElement('option');
        option.value = game;
        option.textContent = game;
        gameSelect.appendChild(option);
      }});

      const books = [...new Set(propsData.map(r => r.bookmaker))].sort();
      const bookGroup = document.getElementById('bookFilter');

      books.forEach(book => {{
        const div = document.createElement('div');
        div.className = 'checkbox-item';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `book-${{book}}`;
        checkbox.value = book;
        checkbox.checked = true; // All selected by default
        state.selectedBooks.add(book);

        checkbox.addEventListener('change', e => {{
          if (e.target.checked) {{
            state.selectedBooks.add(book);
          }} else {{
            state.selectedBooks.delete(book);
          }}
          updateBookButtonText();
          renderTable();
        }});

        const label = document.createElement('label');
        label.htmlFor = `book-${{book}}`;
        label.textContent = book;

        div.appendChild(checkbox);
        div.appendChild(label);
        bookGroup.appendChild(div);
      }});

      updateBookButtonText();
    }}

    // Update book button text based on selections
    function updateBookButtonText() {{
      const buttonText = document.getElementById('bookButtonText');
      const totalBooks = [...new Set(propsData.map(r => r.bookmaker))].length;

      if (state.selectedBooks.size === 0) {{
        buttonText.textContent = 'No Books Selected';
      }} else if (state.selectedBooks.size === totalBooks) {{
        buttonText.textContent = 'All Books';
      }} else {{
        buttonText.textContent = `${{state.selectedBooks.size}} Books`;
      }}
    }}

    // Dropdown toggle
    const bookDropdown = document.getElementById('bookDropdown');
    const bookButton = document.getElementById('bookButton');

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

    function renderTable() {{
      const tbody = document.getElementById('propsTableBody');
      const cardGrid = document.getElementById('cardGrid');

      // Get filter values
      const gameFilter = document.getElementById('gameFilter').value;
      const playerFilter = document.getElementById('playerFilter').value.toLowerCase();
      const marketFilter = document.getElementById('marketFilter').value;
      const sideFilter = document.getElementById('sideFilter').value;

      // Filter rows
      const filtered = propsData.filter(r => {{
        if (gameFilter && r.game !== gameFilter) return false;
        if (playerFilter && !r.player.toLowerCase().includes(playerFilter)) return false;
        if (marketFilter && r.market_std !== marketFilter) return false;
        if (sideFilter && r.side !== sideFilter) return false;
        if (state.selectedBooks.size > 0 && !state.selectedBooks.has(r.bookmaker)) return false;
        if (state.consensusOnly && !isConsensusPlay(r)) return false;
        return true;
      }});

      // Render table
      tbody.innerHTML = filtered.map(r => {{
        const modelProbNum = parseFloat((r.model_pct_disp || '').replace('%', ''));
        const isStrongSignal = modelProbNum >= 85;
        const fireEmoji = isStrongSignal ? ' 🔥' : '';
        const consensusClass = isConsensusPlay(r) ? ' consensus-row' : '';

        return `
          <tr class="${{consensusClass}}">
            <td>${{r.player}}</td>
            <td>${{r.game}}</td>
            <td>${{r.market_disp}}</td>
            <td>${{r.side}}</td>
            <td>${{r.line_disp}}</td>
            <td>${{r.consensus_line_disp}}</td>
            <td>${{r.model_line_disp}}</td>
            <td>${{r.bookmaker}}</td>
            <td>${{r.price_disp}}</td>
            <td>${{r.fair_odds_disp}}</td>
            <td>${{r.mkt_pct_disp}}</td>
            <td>${{r.consensus_pct_disp}}</td>
            <td>${{r.model_pct_disp}}${{fireEmoji}}</td>
            <td>${{r.edge_bps_disp}}</td>
            <td>${{r.book_count_disp}}</td>
          </tr>
        `;
      }}).join('');

      // Render mobile cards
      cardGrid.innerHTML = filtered.map(r => {{
        const modelProbNum = parseFloat((r.model_pct_disp || '').replace('%', ''));
        const isStrongSignal = modelProbNum >= 85;
        const fireEmoji = isStrongSignal ? ' 🔥' : '';
        const consensusClass = isConsensusPlay(r) ? ' consensus' : '';

        return `
          <div class="prop-card${{consensusClass}}">
            <div class="prop-card-header">${{r.player}}</div>
            <div class="prop-card-game">${{r.game}}</div>
            <div class="prop-card-bet">${{r.market_disp}} ${{r.side}} ${{r.line_disp}} @ ${{r.bookmaker}}</div>
            <div class="prop-card-stats">
              <div>Model Line</div><div>${{r.model_line_disp}}</div>
              <div>Model %</div><div>${{r.model_pct_disp}}${{fireEmoji}}</div>
              <div>Odds</div><div>${{r.price_disp}}</div>
              <div>Fair Odds</div><div>${{r.fair_odds_disp}}</div>
              <div>Edge</div><div>${{r.edge_bps_disp}} bps</div>
            </div>
          </div>
        `;
      }}).join('');
    }}

    // Attach filter listeners
    document.getElementById('gameFilter').addEventListener('change', renderTable);
    document.getElementById('playerFilter').addEventListener('input', renderTable);
    document.getElementById('marketFilter').addEventListener('change', renderTable);
    document.getElementById('sideFilter').addEventListener('change', renderTable);
    document.getElementById('consensusFilter').addEventListener('change', e => {{
      state.consensusOnly = e.target.checked;
      renderTable();
    }});

    // Initialize
    populateFilters();
    renderTable();
  </script>
  <script src="../../nav.js?v=29"></script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    main()

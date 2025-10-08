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

    # Add display columns
    df["price_disp"] = df["price"].apply(fmt_odds)
    df["fair_odds_disp"] = df["fair_odds"].apply(fmt_odds)
    df["mkt_pct_disp"] = df["mkt_prob"].apply(lambda p: fmt_pct(p * 100) if not pd.isna(p) else "")
    df["model_pct_disp"] = df["model_prob"].apply(lambda p: fmt_pct(p * 100) if not pd.isna(p) else "")
    df["edge_bps_disp"] = df["edge_bps"].apply(lambda e: f"{int(e)}" if not pd.isna(e) else "")
    df["market_disp"] = df["market_std"].apply(pretty_market)
    df["kick_et"] = df["commence_time"].apply(to_kick_et)

    # Line display with decimal if needed
    df["line_disp"] = df["point"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "")
    df["model_line_disp"] = df["model_line"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "")

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
            "bookmaker": row["bookmaker_title"],
            "price_disp": row["price_disp"],
            "model_line_disp": row["model_line_disp"],
            "model_pct_disp": row["model_pct_disp"],
            "fair_odds_disp": row["fair_odds_disp"],
            "edge_bps_disp": row["edge_bps_disp"],
            "mkt_pct_disp": row["mkt_pct_disp"],
            "kick_et": row["kick_et"],
            "edge_bps": row["edge_bps"],
            "is_consensus": row.get("book_count", 0) >= 3,  # 3+ books = consensus
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

    # Shared nav HTML
    nav_html = """
    <nav>
      <ul>
        <li><a href="/index.html">Home</a></li>
        <li><a href="/docs/props/index.html">NFL Props</a></li>
        <li><a href="/docs/nhl/props/index.html" class="active">NHL Props</a></li>
      </ul>
    </nav>
    """

    # Convert rows to JSON for JS
    import json
    rows_json = json.dumps(rows, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NHL Props - Fourth & Value</title>
  <link rel="stylesheet" href="/styles.css">
  <style>
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
      background: #1a3a1a;
    }}
    .consensus-row:hover {{
      background: #234a23;
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
      background: #1a3a1a;
      border-color: #2a5a2a;
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

    @media (max-width: 768px) {{
      .table-wrapper {{ display: none; }}
      .card-grid {{ display: flex; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  {nav_html}

  <div class="container">
    <h1>NHL Props - {date_str}</h1>
    <p>Model-driven NHL player prop picks with edge analysis.</p>

    <div class="filters">
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
        <label>Min Edge (bps)</label>
        <input type="number" id="minEdgeFilter" placeholder="0" value="0">
      </div>

      <div class="filter-group">
        <label>Player</label>
        <input type="text" id="playerFilter" placeholder="Search player...">
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
            <th>Line</th>
            <th>Book</th>
            <th>Odds</th>
            <th>Model Line</th>
            <th>Model %</th>
            <th>Fair Odds</th>
            <th>Edge</th>
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

    function renderTable() {{
      const tbody = document.getElementById('propsTableBody');
      const cardGrid = document.getElementById('cardGrid');

      // Get filter values
      const marketFilter = document.getElementById('marketFilter').value;
      const sideFilter = document.getElementById('sideFilter').value;
      const minEdge = parseFloat(document.getElementById('minEdgeFilter').value) || 0;
      const playerFilter = document.getElementById('playerFilter').value.toLowerCase();

      // Filter rows
      const filtered = propsData.filter(r => {{
        if (marketFilter && r.market_std !== marketFilter) return false;
        if (sideFilter && r.side !== sideFilter) return false;
        if (Math.abs(r.edge_bps) < minEdge) return false;
        if (playerFilter && !r.player.toLowerCase().includes(playerFilter)) return false;
        return true;
      }});

      // Render table
      tbody.innerHTML = filtered.map(r => {{
        const modelProbNum = parseFloat((r.model_pct_disp || '').replace('%', ''));
        const isStrongSignal = modelProbNum >= 85;
        const fireEmoji = isStrongSignal ? ' 🔥' : '';
        const consensusClass = r.is_consensus ? ' consensus-row' : '';

        return `
          <tr class="${{consensusClass}}">
            <td>${{r.player}}</td>
            <td>${{r.game}}</td>
            <td>${{r.market_disp}}</td>
            <td>${{r.side}}</td>
            <td>${{r.line_disp}}</td>
            <td>${{r.bookmaker}}</td>
            <td>${{r.price_disp}}</td>
            <td>${{r.model_line_disp}}</td>
            <td>${{r.model_pct_disp}}${{fireEmoji}}</td>
            <td>${{r.fair_odds_disp}}</td>
            <td>${{r.edge_bps_disp}}</td>
          </tr>
        `;
      }}).join('');

      // Render mobile cards
      cardGrid.innerHTML = filtered.map(r => {{
        const modelProbNum = parseFloat((r.model_pct_disp || '').replace('%', ''));
        const isStrongSignal = modelProbNum >= 85;
        const fireEmoji = isStrongSignal ? ' 🔥' : '';
        const consensusClass = r.is_consensus ? ' consensus' : '';

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
    document.getElementById('marketFilter').addEventListener('change', renderTable);
    document.getElementById('sideFilter').addEventListener('change', renderTable);
    document.getElementById('minEdgeFilter').addEventListener('input', renderTable);
    document.getElementById('playerFilter').addEventListener('input', renderTable);

    // Initial render
    renderTable();
  </script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    main()

"""
Build NHL Totals page with consensus edges
"""
import pandas as pd
import os
import json
from datetime import datetime

def build_totals_page(predictions_path, consensus_path, edges_path, output_path):
    """
    Build HTML page showing:
    - Model predictions
    - Market consensus
    - Consensus edge plays (highlighted)
    - Book lines table (filterable)
    """

    # Load data
    preds = pd.read_csv(predictions_path) if os.path.exists(predictions_path) else pd.DataFrame()
    consensus = pd.read_csv(consensus_path) if os.path.exists(consensus_path) else pd.DataFrame()
    edges = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()

    # Load book lines
    book_lines_path = consensus_path.replace('consensus.csv', 'book_lines.csv')
    book_lines = pd.read_csv(book_lines_path) if os.path.exists(book_lines_path) else pd.DataFrame()

    # Merge predictions with consensus
    if len(preds) > 0 and len(consensus) > 0:
        merged = preds.merge(consensus[['home_team', 'away_team', 'consensus_line', 'median_line', 'num_books']],
                            on=['home_team', 'away_team'], how='left')
    else:
        merged = preds

    # Calculate stats
    avg_model = merged['total_pred'].mean() if len(merged) > 0 else 0
    avg_market = merged.get('consensus_line', pd.Series([0])).mean() if 'consensus_line' in merged.columns and len(merged) > 0 else 0
    num_games = len(merged)
    num_edges = len(edges)

    # Build game data with book lines
    games_data = []
    if len(merged) > 0:
        for _, game in merged.iterrows():
            game_key = f"{game['away_team']} @ {game['home_team']}"

            # Get book lines for this game
            game_books = book_lines[
                (book_lines['home_team'] == game['home_team']) &
                (book_lines['away_team'] == game['away_team'])
            ].to_dict('records') if len(book_lines) > 0 else []

            # Check if this game has edge plays
            game_edges = edges[
                edges['game'].str.contains(game['home_team']) &
                edges['game'].str.contains(game['away_team'])
            ].to_dict('records') if len(edges) > 0 else []

            games_data.append({
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'matchup': game_key,
                'model_total': float(game['total_pred']),
                'home_pred': float(game['home_pred']),
                'away_pred': float(game['away_pred']),
                'consensus_line': float(game.get('consensus_line', 0)) if 'consensus_line' in game and not pd.isna(game.get('consensus_line')) else None,
                'median_line': float(game.get('median_line', 0)) if 'median_line' in game and not pd.isna(game.get('median_line')) else None,
                'edge': float(game.get('edge', 0)) if 'edge' in game and not pd.isna(game.get('edge')) else None,
                'book_lines': game_books,
                'edge_plays': game_edges,
                'has_edge': len(game_edges) > 0
            })

    games_json = json.dumps(games_data, indent=2)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NHL Totals - Fourth & Value</title>
  <style>
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

    /* Header */
    .header {{
      margin-bottom: 2rem;
    }}
    h1 {{
      font-size: 2rem;
      margin-bottom: 0.5rem;
      color: #4FC3F7;
    }}
    .subtitle {{
      color: #999;
      margin-bottom: 1rem;
    }}

    /* Tabs */
    .tabs {{
      display: flex;
      gap: 0.5rem;
      margin-bottom: 2rem;
      border-bottom: 2px solid #2a2a2a;
    }}
    .tab {{
      padding: 0.75rem 1.5rem;
      background: transparent;
      color: #999;
      text-decoration: none;
      border-bottom: 3px solid transparent;
      transition: all 0.2s;
    }}
    .tab:hover {{
      color: #fff;
    }}
    .tab.active {{
      color: #4FC3F7;
      border-bottom-color: #4FC3F7;
    }}

    /* Filters */
    .filters {{
      background: #2a2a2a;
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 1.5rem;
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
    .search-box {{
      padding: 0.5rem;
      background: #1a1a1a;
      border: 1px solid #444;
      border-radius: 4px;
      color: #fff;
      font-size: 0.875rem;
      min-width: 250px;
    }}
    .search-box:focus {{
      outline: none;
      border-color: #4FC3F7;
    }}
    .search-box::placeholder {{
      color: #666;
    }}

    /* Multi-select book dropdown */
    .checkbox-dropdown {{ position: relative; min-width: 180px; }}
    .checkbox-dropdown-button {{ width: 100%; background: #1a1a1a; color: #fff; border: 1px solid #444; border-radius: 4px; padding: 10px 12px; outline: none; cursor: pointer; text-align: left; display: flex; justify-content: space-between; align-items: center; font-size: 0.875rem; }}
    .checkbox-dropdown-button:hover {{ border-color: #4FC3F7; }}
    .checkbox-dropdown-button::after {{ content: '▼'; font-size: 10px; opacity: 0.6; }}
    .checkbox-dropdown.open .checkbox-dropdown-button::after {{ content: '▲'; }}
    .checkbox-group {{ display: none; position: absolute; z-index: 1000; flex-direction: column; gap: 6px; background: #1a1a1a; border: 1px solid #444; border-radius: 4px; padding: 10px; max-height: 250px; overflow-y: auto; min-width: 100%; margin-top: 4px; box-shadow: 0 4px 12px rgba(0,0,0,.3); }}
    .checkbox-dropdown.open .checkbox-group {{ display: flex; }}
    .checkbox-item {{ display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 4px 6px; border-radius: 6px; transition: background .15s; }}
    .checkbox-item:hover {{ background: rgba(255,255,255,.05); }}
    .checkbox-item input[type="checkbox"] {{ width: 16px; height: 16px; cursor: pointer; accent-color: #4FC3F7; }}
    .checkbox-item label {{ cursor: pointer; flex: 1; font-size: 13px; color: #fff; margin: 0; white-space: nowrap; }}

    /* Stats summary */
    .stats-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .stat-card {{
      background: #1a1a1a;
      padding: 1.5rem;
      border-radius: 8px;
      border: 1px solid #2a2a2a;
    }}
    .stat-label {{
      font-size: 0.875rem;
      color: #999;
      margin-bottom: 0.5rem;
    }}
    .stat-value {{
      font-size: 1.75rem;
      font-weight: bold;
      color: #4FC3F7;
    }}

    /* Edge plays section */
    .edge-plays {{
      background: #1a1a1a;
      border: 2px solid #4FC3F7;
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }}
    .edge-plays h2 {{
      color: #4FC3F7;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .edge-plays h2::before {{
      content: "🎯";
    }}

    /* Game cards */
    .games-grid {{
      display: grid;
      gap: 1rem;
    }}
    .game-card {{
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 1.5rem;
      transition: border-color 0.2s;
    }}
    .game-card:hover {{
      border-color: #4FC3F7;
    }}
    .game-card.has-edge {{
      border: 2px solid #4FC3F7;
      background: linear-gradient(135deg, #1a1a1a 0%, #1a2a3a 100%);
    }}

    .game-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid #2a2a2a;
    }}
    .matchup {{
      font-size: 1.25rem;
      font-weight: bold;
    }}

    .totals-row {{
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 0.75rem;
      margin-bottom: 1rem;
    }}
    .total-item {{
      background: #2a2a2a;
      padding: 0.75rem;
      border-radius: 4px;
    }}
    .total-label {{
      font-size: 0.75rem;
      color: #999;
      margin-bottom: 0.25rem;
    }}
    .total-value {{
      font-size: 1.25rem;
      font-weight: bold;
    }}
    .total-value.model {{
      color: #4FC3F7;
    }}
    .total-value.market {{
      color: #FFA726;
    }}
    .total-value.edge {{
      color: #66BB6A;
    }}

    /* Book lines table */
    .book-lines {{
      margin-top: 1rem;
      background: #2a2a2a;
      border-radius: 4px;
      padding: 1rem;
    }}
    .book-lines h4 {{
      font-size: 0.875rem;
      color: #4FC3F7;
      margin-bottom: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .book-lines-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .book-lines-table th {{
      text-align: left;
      font-size: 0.75rem;
      color: #999;
      padding: 0.5rem;
      border-bottom: 1px solid #3a3a3a;
    }}
    .book-lines-table td {{
      padding: 0.5rem;
      font-size: 0.875rem;
      border-bottom: 1px solid #222;
    }}
    .book-lines-table tr:last-child td {{
      border-bottom: none;
    }}
    .book-lines-table tr:hover {{
      background: #333;
    }}
    .consensus-row {{
      background: #1a2a1a !important;
      font-weight: bold;
    }}
    .consensus-row td {{
      color: #66BB6A;
    }}

    /* Edge plays in card */
    .edge-play-list {{
      background: #2a2a2a;
      border-radius: 4px;
      padding: 1rem;
      margin-top: 1rem;
    }}
    .edge-play-item {{
      padding: 0.75rem;
      background: #1a1a1a;
      border-left: 3px solid #4FC3F7;
      margin-bottom: 0.75rem;
      border-radius: 4px;
    }}
    .edge-play-item:last-child {{
      margin-bottom: 0;
    }}
    .play-bet {{
      font-size: 1.125rem;
      font-weight: bold;
      color: #4FC3F7;
      margin-bottom: 0.5rem;
    }}
    .play-details {{
      font-size: 0.875rem;
      color: #999;
    }}

    .no-data {{
      text-align: center;
      padding: 3rem;
      color: #999;
    }}

    .updated {{
      text-align: center;
      color: #666;
      font-size: 0.875rem;
      margin-top: 2rem;
    }}
  </style>
</head>
<body>
  <div id="nav-root"></div>

  <div class="container">
    <div class="header">
      <h1>NHL Totals</h1>
      <p class="subtitle">Model predictions vs market consensus • Find outlier books before lines move</p>
    </div>

    <div class="tabs">
      <a href="../../nhl/props/index.html" class="tab">Props</a>
      <a href="../../nhl/totals/index.html" class="tab active">Totals</a>
    </div>

    <div class="filters">
      <div class="filter-group">
        <label>Game</label>
        <input type="text" id="gameSearch" class="search-box" placeholder="Filter by team (e.g., TOR, BOS)..." />
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
    </div>

    <div class="stats-summary">
      <div class="stat-card">
        <div class="stat-label">Games Today</div>
        <div class="stat-value">{num_games}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Consensus Edges</div>
        <div class="stat-value">{num_edges}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Model Total</div>
        <div class="stat-value">{avg_model:.1f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Market Total</div>
        <div class="stat-value">{avg_market:.1f}</div>
      </div>
    </div>

    <div id="gamesContainer"></div>

    <div class="updated">
      Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}
    </div>
  </div>

  <script>
    const gamesData = {games_json};

    // State
    const state = {{
      selectedBooks: new Set(),
      gameSearch: ''
    }};

    // Get all unique books
    const allBooks = new Set();
    gamesData.forEach(game => {{
      game.book_lines.forEach(line => allBooks.add(line.book_name));
    }});

    // Initialize filters
    function initFilters() {{
      // Populate book filter
      const bookGroup = document.getElementById('bookFilter');
      Array.from(allBooks).sort().forEach(book => {{
        const div = document.createElement('div');
        div.className = 'checkbox-item';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `book-${{book}}`;
        checkbox.value = book;
        checkbox.checked = true;
        state.selectedBooks.add(book);

        checkbox.addEventListener('change', e => {{
          if (e.target.checked) {{
            state.selectedBooks.add(book);
          }} else {{
            state.selectedBooks.delete(book);
          }}
          updateBookButtonText();
          renderGames();
        }});

        const label = document.createElement('label');
        label.htmlFor = `book-${{book}}`;
        label.textContent = book;

        div.appendChild(checkbox);
        div.appendChild(label);
        bookGroup.appendChild(div);
      }});

      updateBookButtonText();

      // Book dropdown toggle
      const bookDropdown = document.getElementById('bookDropdown');
      const bookButton = document.getElementById('bookButton');

      bookButton.addEventListener('click', e => {{
        e.stopPropagation();
        bookDropdown.classList.toggle('open');
      }});

      document.addEventListener('click', e => {{
        if (!bookDropdown.contains(e.target)) {{
          bookDropdown.classList.remove('open');
        }}
      }});

      // Game search
      document.getElementById('gameSearch').addEventListener('input', e => {{
        state.gameSearch = e.target.value.toLowerCase();
        renderGames();
      }});
    }}

    function updateBookButtonText() {{
      const buttonText = document.getElementById('bookButtonText');
      const totalBooks = allBooks.size;

      if (state.selectedBooks.size === 0) {{
        buttonText.textContent = 'No Books Selected';
      }} else if (state.selectedBooks.size === totalBooks) {{
        buttonText.textContent = 'All Books';
      }} else {{
        buttonText.textContent = `${{state.selectedBooks.size}} Books`;
      }}
    }}

    function renderGames() {{
      const container = document.getElementById('gamesContainer');

      // Filter games by search
      const filtered = gamesData.filter(game => {{
        if (state.gameSearch) {{
          return game.matchup.toLowerCase().includes(state.gameSearch) ||
                 game.home_team.toLowerCase().includes(state.gameSearch) ||
                 game.away_team.toLowerCase().includes(state.gameSearch);
        }}
        return true;
      }});

      if (filtered.length === 0) {{
        container.innerHTML = `
          <div class="no-data">
            <h3>No games match your filters</h3>
            <p>Try adjusting your search</p>
          </div>
        `;
        return;
      }}

      container.innerHTML = '<div class="games-grid">' + filtered.map(game => {{
        const cardClass = game.has_edge ? 'game-card has-edge' : 'game-card';

        // Filter book lines by selected books
        const visibleBookLines = game.book_lines.filter(line => state.selectedBooks.has(line.book_name));

        let html = `
          <div class="${{cardClass}}">
            <div class="game-header">
              <div class="matchup">${{game.matchup}}</div>
            </div>

            <div class="totals-row">
              <div class="total-item">
                <div class="total-label">Model Total</div>
                <div class="total-value model">${{game.model_total.toFixed(1)}}</div>
              </div>
              <div class="total-item">
                <div class="total-label">Home (${{game.home_team}})</div>
                <div class="total-value">${{game.home_pred.toFixed(1)}}</div>
              </div>
              <div class="total-item">
                <div class="total-label">Away (${{game.away_team}})</div>
                <div class="total-value">${{game.away_pred.toFixed(1)}}</div>
              </div>`;

        if (game.consensus_line) {{
          html += `
              <div class="total-item">
                <div class="total-label">Market Consensus</div>
                <div class="total-value market">${{game.consensus_line.toFixed(1)}}</div>
              </div>`;
        }}

        if (game.median_line) {{
          html += `
              <div class="total-item">
                <div class="total-label">Median Line</div>
                <div class="total-value">${{game.median_line.toFixed(1)}}</div>
              </div>`;
        }}

        if (game.edge !== null) {{
          const edgeClass = Math.abs(game.edge) > 0.3 ? 'edge' : '';
          html += `
              <div class="total-item">
                <div class="total-label">Model Edge</div>
                <div class="total-value ${{edgeClass}}">${{game.edge >= 0 ? '+' : ''}}${{game.edge.toFixed(1)}}</div>
              </div>`;
        }}

        html += '</div>';

        // Show edge plays
        if (game.edge_plays.length > 0) {{
          html += '<div class="edge-play-list">';
          html += '<div style="font-size: 0.875rem; color: #4FC3F7; margin-bottom: 0.5rem; font-weight: bold;">🎯 Consensus Edge Plays:</div>';
          game.edge_plays.forEach(play => {{
            const bet_emoji = play.bet === 'UNDER' ? '📉' : '📈';
            html += `
              <div style="margin-bottom: 0.5rem; padding: 0.5rem; background: #0f0f0f; border-radius: 4px;">
                <div style="color: #4FC3F7; font-weight: bold;">${{bet_emoji}} ${{play.bet}} ${{play.line}} at ${{play.book}} (${{play.under_price >= 0 ? '+' : ''}}${{play.under_price}})</div>
                <div style="font-size: 0.8rem; color: #999;">Consensus: ${{play.consensus.toFixed(1)}} | Model: ${{play.model.toFixed(1)}} | Edge: ${{play.edge.toFixed(1)}} goals</div>
              </div>
            `;
          }});
          html += '</div>';
        }}

        // Show book lines table
        if (visibleBookLines.length > 0) {{
          html += `
            <div class="book-lines">
              <h4>Book Lines</h4>
              <table class="book-lines-table">
                <thead>
                  <tr>
                    <th>Book</th>
                    <th>Line</th>
                    <th>Over</th>
                    <th>Under</th>
                  </tr>
                </thead>
                <tbody>`;

          // Add consensus row first if available
          if (game.consensus_line) {{
            html += `
                  <tr class="consensus-row">
                    <td>Consensus</td>
                    <td>${{game.consensus_line.toFixed(1)}}</td>
                    <td>—</td>
                    <td>—</td>
                  </tr>`;
          }}

          // Add book lines
          visibleBookLines.forEach(line => {{
            html += `
                  <tr>
                    <td>${{line.book_name}}</td>
                    <td>${{line.line}}</td>
                    <td>${{line.over_price >= 0 ? '+' : ''}}${{line.over_price}}</td>
                    <td>${{line.under_price >= 0 ? '+' : ''}}${{line.under_price}}</td>
                  </tr>`;
          }});

          html += `
                </tbody>
              </table>
            </div>`;
        }}

        html += '</div>';
        return html;
      }}).join('') + '</div>';
    }}

    // Initialize
    initFilters();
    renderGames();
  </script>
  <script src="../../nav.js?v=30"></script>
</body>
</html>
"""

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Built NHL totals page: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build NHL totals page')
    parser.add_argument('--predictions', default='data/nhl/predictions/today.csv', help='Predictions CSV')
    parser.add_argument('--consensus', default='data/nhl/consensus/consensus.csv', help='Consensus CSV')
    parser.add_argument('--edges', default='data/nhl/consensus/edges.csv', help='Edges CSV')
    parser.add_argument('--output', default='docs/nhl/totals/index.html', help='Output HTML')

    args = parser.parse_args()

    build_totals_page(args.predictions, args.consensus, args.edges, args.output)

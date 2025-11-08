"""
Build NFL Totals page with consensus edges
Matches NHL totals page structure
"""
import pandas as pd
import os
from datetime import datetime

def build_totals_page(predictions_path, consensus_path, edges_path, lines_path, output_path, week):
    """
    Build HTML page showing:
    - Model predictions
    - Market consensus (totals + spreads)
    - All book lines
    - Consensus edge plays (highlighted)
    """

    # Load data
    preds = pd.read_csv(predictions_path) if os.path.exists(predictions_path) else pd.DataFrame()
    consensus = pd.read_csv(consensus_path) if os.path.exists(consensus_path) else pd.DataFrame()
    edges = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()
    lines = pd.read_csv(lines_path) if os.path.exists(lines_path) else pd.DataFrame()

    # Merge predictions with consensus (totals only)
    if len(preds) > 0 and len(consensus) > 0:
        totals_consensus = consensus[consensus['market'] == 'total'][['game', 'consensus_line', 'num_books']]
        spread_consensus = consensus[consensus['market'] == 'spread'][['game', 'consensus_line', 'num_books']]

        merged = preds.merge(totals_consensus.rename(columns={'consensus_line': 'consensus_total', 'num_books': 'num_books_total'}),
                            on='game', how='left')
        merged = merged.merge(spread_consensus.rename(columns={'consensus_line': 'consensus_spread', 'num_books': 'num_books_spread'}),
                            on='game', how='left')
        merged['edge'] = merged['total_pred'] - merged['consensus_total']
        # Calculate model spread (home team perspective)
        merged['model_spread'] = merged['home_pred'] - merged['away_pred']
    else:
        merged = preds
        if len(merged) > 0:
            merged['model_spread'] = merged['home_pred'] - merged['away_pred']

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NFL Totals - Fourth & Value</title>
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

    /* Search filter */
    .search-container {{
      margin-bottom: 1.5rem;
    }}
    .search-box {{
      width: 100%;
      max-width: 400px;
      padding: 0.75rem 1rem;
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      color: #fff;
      font-size: 1rem;
      transition: border-color 0.2s;
    }}
    .search-box:focus {{
      outline: none;
      border-color: #4FC3F7;
    }}
    .search-box::placeholder {{
      color: #666;
    }}
  </style>
</head>
<body>
  <div id="nav-root"></div>
  <script src="../../nav.js?v=30"></script>

  <div class="container">
    <div class="header">
      <h1>NFL Totals - Week {week}</h1>
      <p class="subtitle">Model predictions vs market consensus • Find outlier books before lines move</p>
    </div>

    <div class="tabs">
      <a href="../../props/index.html" class="tab">Props</a>
      <a href="../../nfl/totals/index.html" class="tab active">Totals</a>
    </div>

    <div class="search-container">
      <input type="text" id="gameSearch" class="search-box" placeholder="Filter games by team (e.g., KC, BUF, DAL)..." />
    </div>
"""

    # Stats summary
    if len(merged) > 0:
        avg_model = merged['total_pred'].mean()
        avg_market = merged['consensus_total'].mean() if 'consensus_total' in merged.columns and not merged['consensus_total'].isna().all() else 0
        num_games = len(merged)
        num_edges = len(edges)

        html += f"""
    <div class="stats-summary">
      <div class="stat-card">
        <div class="stat-label">Games This Week</div>
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
"""

    # Edge plays section
    if len(edges) > 0:
        html += """
    <div class="edge-plays">
      <h2>Consensus Edge Plays</h2>
      <p style="color: #999; margin-bottom: 1rem;">Books out of sync with market consensus. Bet before lines move!</p>
"""

        for _, edge in edges.iterrows():
            bet_emoji = "📉" if edge['bet'] == 'UNDER' else "📈"
            html += f"""
      <div class="edge-play-item">
        <div class="play-bet">{bet_emoji} {edge['game']} - {edge['bet']} {edge['line']} at {edge['book']}</div>
        <div class="play-details">
          Consensus: {edge['consensus']:.1f} | Model: {edge['model']:.1f} |
          Edge: {edge['edge']:.1f} points
        </div>
      </div>
"""

        html += """
    </div>
"""

    # Games grid
    html += """
    <div class="games-grid">
"""

    if len(merged) > 0:
        for _, game in merged.iterrows():
            # Check if this game has edge plays
            game_edges = edges[edges['game'] == game['game']] if len(edges) > 0 else pd.DataFrame()
            has_edge = len(game_edges) > 0

            card_class = "game-card has-edge" if has_edge else "game-card"

            html += f"""
      <div class="{card_class}">
        <div class="game-header">
          <div class="matchup">{game['game']}</div>
        </div>

        <div class="totals-row">
          <div class="total-item">
            <div class="total-label">Model Prediction</div>
            <div class="total-value model">{game['total_pred']:.1f}</div>
          </div>
          <div class="total-item">
            <div class="total-label">{game['home_team']}</div>
            <div class="total-value">{game['home_pred']:.1f}</div>
          </div>
          <div class="total-item">
            <div class="total-label">{game['away_team']}</div>
            <div class="total-value">{game['away_pred']:.1f}</div>
          </div>
"""

            if 'consensus_total' in game and not pd.isna(game['consensus_total']):
                html += f"""
          <div class="total-item">
            <div class="total-label">Market Total</div>
            <div class="total-value market">{game['consensus_total']:.1f}</div>
          </div>
"""

            if 'model_spread' in game and not pd.isna(game['model_spread']):
                html += f"""
          <div class="total-item">
            <div class="total-label">Model Spread</div>
            <div class="total-value model">{game['home_team']} {game['model_spread']:+.1f}</div>
          </div>
"""

            if 'consensus_spread' in game and not pd.isna(game['consensus_spread']):
                html += f"""
          <div class="total-item">
            <div class="total-label">Market Spread</div>
            <div class="total-value market">{game['home_team']} {game['consensus_spread']:+.1f}</div>
          </div>
"""

            if 'edge' in game and not pd.isna(game['edge']):
                edge_color = "edge" if abs(game['edge']) > 3.0 else ""
                html += f"""
          <div class="total-item">
            <div class="total-label">Model Edge</div>
            <div class="total-value {edge_color}">{game['edge']:+.1f}</div>
          </div>
"""

            html += """
        </div>
"""

            # Show all book lines for this game
            game_lines = lines[lines['game'] == game['game']] if len(lines) > 0 else pd.DataFrame()
            if len(game_lines) > 0:
                html += """
        <div class="book-lines">
          <h4>📊 All Book Lines</h4>
          <table class="book-lines-table">
            <thead>
              <tr>
                <th>Book</th>
                <th>Total</th>
                <th>Over</th>
                <th>Under</th>
                <th>Spread</th>
                <th>Fav</th>
                <th>Dog</th>
              </tr>
            </thead>
            <tbody>
"""
                # Add consensus row first if available
                if 'consensus_total' in game and not pd.isna(game['consensus_total']):
                    spread_str = f"{game['home_team']} {game['consensus_spread']:+.1f}" if 'consensus_spread' in game and not pd.isna(game['consensus_spread']) else '-'
                    html += f"""
              <tr class="consensus-row">
                <td><strong>CONSENSUS</strong></td>
                <td>{game['consensus_total']:.1f}</td>
                <td>-</td>
                <td>-</td>
                <td>{spread_str}</td>
                <td>-</td>
                <td>-</td>
              </tr>
"""

                # Add individual book lines
                for _, line in game_lines.iterrows():
                    total_str = f"{line['total_over_line']:.1f}" if not pd.isna(line.get('total_over_line')) else '-'
                    over_odds = f"{int(line['total_over_price']):+d}" if not pd.isna(line.get('total_over_price')) else '-'
                    under_odds = f"{int(line['total_under_price']):+d}" if not pd.isna(line.get('total_under_price')) else '-'

                    spread_str = f"{game['home_team']} {line['spread_home_line']:+.1f}" if not pd.isna(line.get('spread_home_line')) else '-'
                    spread_home_odds = f"{int(line['spread_home_price']):+d}" if not pd.isna(line.get('spread_home_price')) else '-'
                    spread_away_odds = f"{int(line['spread_away_price']):+d}" if not pd.isna(line.get('spread_away_price')) else '-'

                    html += f"""
              <tr>
                <td>{line['book']}</td>
                <td>{total_str}</td>
                <td>{over_odds}</td>
                <td>{under_odds}</td>
                <td>{spread_str}</td>
                <td>{spread_home_odds}</td>
                <td>{spread_away_odds}</td>
              </tr>
"""

                html += """
            </tbody>
          </table>
        </div>
"""

            # Show edge plays for this game
            if has_edge:
                html += """
        <div class="edge-play-list">
          <div style="font-size: 0.875rem; color: #4FC3F7; margin-bottom: 0.5rem; font-weight: bold;">🎯 Consensus Edge Plays:</div>
"""
                for _, play in game_edges.iterrows():
                    bet_emoji = "📉" if play['bet'] == 'UNDER' else "📈"
                    html += f"""
          <div style="margin-bottom: 0.5rem; padding: 0.5rem; background: #0f0f0f; border-radius: 4px;">
            <div style="color: #4FC3F7; font-weight: bold;">{bet_emoji} {play['bet']} {play['line']} at {play['book']}</div>
            <div style="font-size: 0.8rem; color: #999;">Consensus: {play['consensus']:.1f} | Model: {play['model']:.1f} | Edge: {play['edge']:.1f} points</div>
          </div>
"""
                html += """
        </div>
"""

            html += """
      </div>
"""
    else:
        html += """
      <div class="no-data">
        <h3>No games this week</h3>
        <p>Run: make nfl_totals_daily WEEK=N</p>
      </div>
"""

    html += f"""
    </div>

    <div class="updated">
      Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}
    </div>
  </div>

  <script>
    // Game search filter
    const searchBox = document.getElementById('gameSearch');
    const gameCards = document.querySelectorAll('.game-card');

    searchBox.addEventListener('input', (e) => {{
      const searchTerm = e.target.value.toLowerCase().trim();

      gameCards.forEach(card => {{
        const matchup = card.querySelector('.matchup').textContent.toLowerCase();

        if (searchTerm === '' || matchup.includes(searchTerm)) {{
          card.style.display = 'block';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }});
  </script>
</body>
</html>
"""

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Built NFL totals page: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build NFL totals page')
    parser.add_argument('--predictions', default='data/nfl/predictions/week_predictions.csv', help='Predictions CSV')
    parser.add_argument('--consensus', default='data/nfl/consensus/totals_spreads_consensus.csv', help='Consensus CSV')
    parser.add_argument('--edges', default='data/nfl/consensus/edges.csv', help='Edges CSV')
    parser.add_argument('--lines', default='data/nfl/lines/totals_spreads.csv', help='Book lines CSV')
    parser.add_argument('--output', default='docs/nfl/totals/index.html', help='Output HTML')
    parser.add_argument('--week', type=int, required=True, help='Week number')

    args = parser.parse_args()

    build_totals_page(args.predictions, args.consensus, args.edges, args.lines, args.output, args.week)

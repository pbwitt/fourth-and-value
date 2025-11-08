"""
Build NHL Totals page with consensus edges
"""
import pandas as pd
import os
from datetime import datetime

def build_totals_page(predictions_path, consensus_path, edges_path, output_path):
    """
    Build HTML page showing:
    - Model predictions
    - Market consensus
    - Consensus edge plays (highlighted)
    """

    # Load data
    preds = pd.read_csv(predictions_path) if os.path.exists(predictions_path) else pd.DataFrame()
    consensus = pd.read_csv(consensus_path) if os.path.exists(consensus_path) else pd.DataFrame()
    edges = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()

    # Merge predictions with consensus
    if len(preds) > 0 and len(consensus) > 0:
        merged = preds.merge(consensus[['home_team', 'away_team', 'consensus_line', 'median_line', 'num_books']],
                            on=['home_team', 'away_team'], how='left')
    else:
        merged = preds

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
    .game-time {{
      color: #999;
      font-size: 0.875rem;
    }}

    .totals-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
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
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>NHL Totals</h1>
      <p class="subtitle">Model predictions vs market consensus • Find outlier books before lines move</p>
    </div>

    <div class="tabs">
      <a href="../../nhl/props/index.html" class="tab">Props</a>
      <a href="../../nhl/totals/index.html" class="tab active">Totals</a>
    </div>
"""

    # Stats summary
    if len(merged) > 0:
        avg_model = merged['total_pred'].mean()
        avg_market = merged.get('consensus_line', pd.Series([0])).mean() if 'consensus_line' in merged.columns else 0
        num_games = len(merged)
        num_edges = len(edges)

        html += f"""
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
          Edge: {edge['edge']:.1f} goals | Price: {edge['under_price']:+d}
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
            game_matchup = f"{game['away_team']} @ {game['home_team']}"

            # Check if this game has edge plays
            game_edges = edges[edges['game'].str.contains(game['home_team']) & edges['game'].str.contains(game['away_team'])] if len(edges) > 0 else pd.DataFrame()
            has_edge = len(game_edges) > 0

            card_class = "game-card has-edge" if has_edge else "game-card"

            html += f"""
      <div class="{card_class}">
        <div class="game-header">
          <div class="matchup">{game_matchup}</div>
        </div>

        <div class="totals-row">
          <div class="total-item">
            <div class="total-label">Model Prediction</div>
            <div class="total-value model">{game['total_pred']:.1f}</div>
          </div>
          <div class="total-item">
            <div class="total-label">Home Team</div>
            <div class="total-value">{game['home_pred']:.1f}</div>
          </div>
          <div class="total-item">
            <div class="total-label">Away Team</div>
            <div class="total-value">{game['away_pred']:.1f}</div>
          </div>
"""

            if 'consensus_line' in game and not pd.isna(game['consensus_line']):
                html += f"""
          <div class="total-item">
            <div class="total-label">Market Consensus</div>
            <div class="total-value market">{game['consensus_line']:.1f}</div>
          </div>
"""

            if 'edge' in game and not pd.isna(game['edge']):
                edge_color = "edge" if abs(game['edge']) > 0.3 else ""
                html += f"""
          <div class="total-item">
            <div class="total-label">Model Edge</div>
            <div class="total-value {edge_color}">{game['edge']:+.1f}</div>
          </div>
"""

            html += """
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
            <div style="color: #4FC3F7; font-weight: bold;">{bet_emoji} {play['bet']} {play['line']} at {play['book']} ({play['under_price']:+d})</div>
            <div style="font-size: 0.8rem; color: #999;">Consensus: {play['consensus']:.1f} | Model: {play['model']:.1f} | Edge: {play['edge']:.1f} goals</div>
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
        <h3>No games today</h3>
        <p>Run: make nhl_totals_daily</p>
      </div>
"""

    html += """
    </div>

    <div class="updated">
      Last updated: """ + datetime.now().strftime('%Y-%m-%d %I:%M %p ET') + """
    </div>
  </div>
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

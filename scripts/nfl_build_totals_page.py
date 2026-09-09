"""
Build NFL Totals page with consensus edges
Matches NHL totals page structure
"""
import pandas as pd
import os
from datetime import datetime

def build_totals_page(predictions_path, consensus_path, edges_path, lines_path, output_path, week,
                      team_totals_path=None, priced_path=None):
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
        merged['model_spread'] = merged['away_pred'] - merged['home_pred']
    elif len(consensus) > 0:
        # No predictions - use consensus as base
        totals_consensus = consensus[consensus['market'] == 'total'][['game', 'home_team', 'away_team', 'consensus_line', 'num_books']].copy()
        totals_consensus = totals_consensus.rename(columns={'consensus_line': 'consensus_total', 'num_books': 'num_books_total'})
        spread_consensus = consensus[consensus['market'] == 'spread'][['game', 'consensus_line', 'num_books']]

        merged = totals_consensus.merge(spread_consensus.rename(columns={'consensus_line': 'consensus_spread', 'num_books': 'num_books_spread'}),
                                       on='game', how='left')
        # Add empty prediction columns
        merged['home_pred'] = None
        merged['away_pred'] = None
        merged['total_pred'] = None
        merged['model_spread'] = None
        merged['edge'] = None
    else:
        merged = preds
        if len(merged) > 0:
            merged['model_spread'] = merged['away_pred'] - merged['home_pred']

    # Market-derived team totals: implied team totals, de-vigged prices and
    # best available price. Merged by game so a missing file just omits them.
    team_totals = (pd.read_csv(team_totals_path)
                   if team_totals_path and os.path.exists(team_totals_path)
                   else pd.DataFrame())
    if len(team_totals) > 0 and len(merged) > 0:
        carry = [c for c in ('game', 'commence_time', 'implied_home_total', 'implied_away_total',
                             'fair_over_prob', 'fair_under_prob', 'hold_pct',
                             'best_over_price', 'best_over_book',
                             'best_under_price', 'best_under_book',
                             'books_at_line', 'quoted_at') if c in team_totals.columns]
        merged = merged.merge(team_totals[carry], on='game', how='left')

    # Priced model view: projection shrunk toward the market by the amount the
    # model's measured skill justifies, plus the probability that implies.
    priced = (pd.read_csv(priced_path)
              if priced_path and os.path.exists(priced_path) else pd.DataFrame())
    calibration = {}
    if len(priced) > 0 and len(merged) > 0:
        cols = [c for c in ('game', 'model_projection', 'calibrated_projection',
                            'model_over_prob', 'market_over_prob', 'prob_edge_over_pp',
                            'best_ev_per_100', 'best_side') if c in priced.columns]
        merged = merged.merge(priced[cols], on='game', how='left')
        first = priced.iloc[0]
        calibration = {
            'beta': first.get('calibration_beta'),
            'sd': first.get('calibration_sd'),
            'significant': bool(first.get('calibration_significant')),
        }

    # Model figures are shown only when this slate actually has predictions.
    # A stale predictions file must never be presented against current lines.
    has_model = 'total_pred' in merged.columns and merged['total_pred'].notna().any()
    has_priced = 'calibrated_projection' in merged.columns and merged['calibrated_projection'].notna().any()

    if has_model:
        subtitle = "Model predictions vs market consensus • Find outlier books before lines move"
        notice = ("Research snapshot: quote freshness and model accuracy have not been revalidated. "
                  "Confirm the season and source dates before interpreting these lines.")
    else:
        subtitle = ("Implied team totals, de-vigged prices and best available line "
                    "• Find outlier books before lines move")
        notice = ("Every number on this page is derived from sportsbook prices: team totals come "
                  "from the consensus total and spread, and fair percentages are the two sides "
                  "de-vigged against each other. No model estimate is published for this slate.")

    # Build HTML
    kickoff_col = 'commence_time'
    def kickoff(game_row):
        """Kickoff in Eastern time, from the provider's UTC commence time."""
        raw = game_row.get(kickoff_col) if hasattr(game_row, 'get') else None
        if not raw or pd.isna(raw):
            return ''
        ts = pd.to_datetime(raw, utc=True, errors='coerce')
        if pd.isna(ts):
            return ''
        return ts.tz_convert('America/New_York').strftime('%a %-I:%M %p ET')

    quoted_at = ''
    if 'quoted_at' in merged.columns and merged['quoted_at'].notna().any():
        ts = pd.to_datetime(merged['quoted_at'], utc=True, errors='coerce').max()
        if pd.notna(ts):
            quoted_at = ts.tz_convert('America/New_York').strftime('%b %-d, %-I:%M %p ET')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NFL Team Totals - Week {week} | Fourth &amp; Value</title>
  <meta name="description" content="Implied NFL team totals, de-vigged over/under prices and the best available line across sportsbooks for Week {week}.">
  <link rel="canonical" href="https://fourthandvalue.com/nfl/totals/">
  <link rel="stylesheet" href="../../assets/site.css">
  <style>
    .totals-summary {{ font-variant-numeric: tabular-nums; }}
    .totals-summary th {{ white-space: nowrap; color: var(--muted); font-size: 13px;
      text-transform: uppercase; letter-spacing: .06em; }}
    .totals-summary td {{ white-space: nowrap; }}
    .totals-summary tbody tr:hover {{ background: #172230; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .game-card {{ background: var(--card); border: 1px solid var(--border);
      border-radius: 14px; padding: 22px; margin-bottom: 16px; }}
    .game-card.has-edge {{ border-color: var(--accent); }}
    .game-header {{ display: flex; justify-content: space-between; align-items: baseline;
      gap: 12px; flex-wrap: wrap; padding-bottom: 14px; margin-bottom: 16px;
      border-bottom: 1px solid var(--border); }}
    .matchup {{ font-size: 20px; font-weight: 700; color: #fff; }}
    .teamtotals {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px; margin-bottom: 16px; }}
    .teamtotal {{ background: #101722; border: 1px solid var(--border);
      border-radius: 10px; padding: 14px 16px; }}
    .teamtotal .team {{ color: var(--muted); font-size: 13px; text-transform: uppercase;
      letter-spacing: .06em; }}
    .teamtotal .val {{ font-size: 30px; font-weight: 700; color: var(--accent);
      font-variant-numeric: tabular-nums; letter-spacing: -.02em; }}
    .marketrow {{ display: flex; flex-wrap: wrap; gap: 10px 28px; margin-bottom: 16px; }}
    .marketrow div {{ font-size: 14px; color: var(--muted); }}
    .marketrow b {{ display: block; color: var(--text); font-size: 17px; font-weight: 650;
      font-variant-numeric: tabular-nums; }}
    .book-lines-table {{ font-variant-numeric: tabular-nums; font-size: 14px; }}
    .book-lines-table th {{ color: var(--muted); font-size: 12px; text-transform: uppercase;
      letter-spacing: .06em; }}
    .consensus-row {{ background: #172230; font-weight: 650; }}
    .best {{ color: var(--accent); font-weight: 650; }}
    @media (max-width: 600px) {{
      .teamtotals {{ grid-template-columns: 1fr; }}
      .game-card {{ padding: 18px; }}
      .matchup {{ font-size: 18px; }}
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <div id="nav-root"></div>
  <script src="../../nav.js?v=33"></script>

  <main id="main" class="wrap">
    <p class="eyebrow">NFL &middot; Week {week}</p>
    <h1>NFL team totals</h1>
    <p class="lead">{subtitle}</p>
    <div class="notice"><p>{notice}</p></div>

    <nav class="subnav" aria-label="NFL sections">
      <a href="../../props/index.html">Player props</a>
      <a href="../../props/top.html">Top picks</a>
      <a href="../../nfl/totals/index.html" aria-current="page">Team totals</a>
      <a href="../../props/arbitrage.html">Price checks</a>
      <a href="../../methods.html">Methods</a>
    </nav>

    <div class="filters" style="grid-template-columns:1fr">
      <label for="gameSearch">Filter games by team
        <input type="text" id="gameSearch" class="search-box" placeholder="e.g. KC, BUF, DAL" />
      </label>
    </div>
"""

    # Stats summary
    if len(merged) > 0:
        avg_market = merged['consensus_total'].mean() if 'consensus_total' in merged.columns and not merged['consensus_total'].isna().all() else 0
        num_games = len(merged)
        books = int(lines['book'].nunique()) if len(lines) and 'book' in lines.columns else 0
        avg_hold = merged['hold_pct'].mean() if 'hold_pct' in merged.columns and merged['hold_pct'].notna().any() else float('nan')

        model_card = ""
        if has_model:
            model_card = f"""
      <div class="panel">
        <div class="meta">Avg model total</div>
        <div style="font-size:28px;font-weight:700;color:var(--accent)">{merged['total_pred'].mean():.1f}</div>
      </div>"""

        hold_card = ""
        if avg_hold == avg_hold:
            hold_card = f"""
      <div class="panel">
        <div class="meta">Average book hold</div>
        <div style="font-size:28px;font-weight:700;color:var(--accent)">{avg_hold:.1f}%</div>
      </div>"""

        html += f"""
    <div class="grid">
      <div class="panel">
        <div class="meta">Games this week</div>
        <div style="font-size:28px;font-weight:700;color:var(--accent)">{num_games}</div>
      </div>
      <div class="panel">
        <div class="meta">Sportsbooks compared</div>
        <div style="font-size:28px;font-weight:700;color:var(--accent)">{books}</div>
      </div>{model_card}{hold_card}
      <div class="panel">
        <div class="meta">Average market total</div>
        <div style="font-size:28px;font-weight:700;color:var(--accent)">{avg_market:.1f}</div>
      </div>
    </div>
"""

        if has_priced and calibration:
            beta = calibration.get('beta')
            sig = calibration.get('significant')
            verdict = ("has measurable skill against the closing line"
                       if sig else
                       "has no measurable skill against the closing line")
            html += f"""
    <section class="section">
      <h2>What the model thinks, and what that has been worth</h2>
      <div class="notice">
        <p>Our totals model {verdict}. Measured walk-forward over 863 completed
        games, only <b>{beta:.3f}</b> of each point it claims has historically shown up in
        the result, and that figure is not statistically distinguishable from zero.
        Graded at real prices across the 2025 season it went <b>49.2%</b> on 181 bets
        for <b>&minus;5.8%</b> ROI, and got worse as its claimed edge grew.</p>
        <p>So projections below are shown two ways: what the model says on its own,
        and that number shrunk toward the market by the amount its record justifies.
        The shrunk number is the one any probability here is priced from. We publish
        both rather than only the flattering one.</p>
      </div>
    </section>
"""

        # Scannable summary: every game on one screen, so team totals can be
        # compared across the slate without opening each card.
        if 'implied_home_total' in merged.columns and merged['implied_home_total'].notna().any():
            html += """
    <section class="section">
      <h2>Every game at a glance</h2>
      <p class="meta">Team totals are implied by each game's consensus total and spread. Fair percentages are de-vigged.</p>
      <div class="table-wrap">
        <table class="totals-summary">
          <thead>
            <tr>
              <th scope="col">Game</th>
              <th scope="col">Kickoff</th>
              <th scope="col" class="num">Total</th>
              <th scope="col" class="num">Away TT</th>
              <th scope="col" class="num">Home TT</th>
              <th scope="col" class="num">Fair O/U</th>
              <th scope="col" class="num">Hold</th>
              <th scope="col" class="num">Best over</th>
              <th scope="col" class="num">Best under</th>
            </tr>
          </thead>
          <tbody>
"""
            for _, g in merged.iterrows():
                if pd.isna(g.get('implied_home_total')):
                    continue
                fair = (f"{g['fair_over_prob']*100:.1f}% / {g['fair_under_prob']*100:.1f}%"
                        if pd.notna(g.get('fair_over_prob')) else '&mdash;')
                hold = f"{g['hold_pct']:.1f}%" if pd.notna(g.get('hold_pct')) else '&mdash;'
                b_over = (f"{g['best_over_price']:+.0f} <span class=\"meta\">{g['best_over_book']}</span>"
                          if pd.notna(g.get('best_over_price')) else '&mdash;')
                b_under = (f"{g['best_under_price']:+.0f} <span class=\"meta\">{g['best_under_book']}</span>"
                           if pd.notna(g.get('best_under_price')) else '&mdash;')
                html += f"""            <tr>
              <td><a href="#game-{g['game'].replace(' ', '-').replace('@', 'at')}">{g['game']}</a></td>
              <td class="meta">{kickoff(g)}</td>
              <td class="num">{g['consensus_total']:.1f}</td>
              <td class="num">{g['implied_away_total']:.2f}</td>
              <td class="num">{g['implied_home_total']:.2f}</td>
              <td class="num">{fair}</td>
              <td class="num">{hold}</td>
              <td class="num">{b_over}</td>
              <td class="num">{b_under}</td>
            </tr>
"""
            html += """          </tbody>
        </table>
      </div>
    </section>
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
    <section class="section">
      <h2>Game detail</h2>
      <div class="games-grid">
"""

    if len(merged) > 0:
        for _, game in merged.iterrows():
            game_edges = edges[edges['game'] == game['game']] if len(edges) > 0 else pd.DataFrame()
            has_edge = len(game_edges) > 0
            card_class = "game-card has-edge" if has_edge else "game-card"
            anchor = game['game'].replace(' ', '-').replace('@', 'at')

            html += f"""
      <article class="{card_class}" id="game-{anchor}">
        <div class="game-header">
          <div class="matchup">{game['game']}</div>
          <div class="meta">{kickoff(game)}</div>
        </div>
"""

            # Implied team totals lead the card: they are the headline number.
            if pd.notna(game.get('implied_home_total')):
                html += f"""
        <div class="teamtotals">
          <div class="teamtotal">
            <div class="team">{game['away_team']} team total</div>
            <div class="val">{game['implied_away_total']:.2f}</div>
          </div>
          <div class="teamtotal">
            <div class="team">{game['home_team']} team total</div>
            <div class="val">{game['implied_home_total']:.2f}</div>
          </div>
        </div>
"""

            html += """
        <div class="marketrow">
"""
            if pd.notna(game.get('consensus_total')):
                html += f"""          <div>Market total<b>{game['consensus_total']:.1f}</b></div>
"""
            if pd.notna(game.get('consensus_spread')):
                html += f"""          <div>Market spread<b>{game['home_team']} {game['consensus_spread']:+.1f}</b></div>
"""
            if pd.notna(game.get('fair_over_prob')):
                html += f"""          <div>Fair over / under<b>{game['fair_over_prob']*100:.1f}% / {game['fair_under_prob']*100:.1f}%</b></div>
"""
            if pd.notna(game.get('hold_pct')):
                html += f"""          <div>Book hold<b>{game['hold_pct']:.1f}%</b></div>
"""
            if pd.notna(game.get('best_over_price')):
                html += f"""          <div>Best over &middot; {int(game['books_at_line'])} books at line<b class="best">{game['best_over_price']:+.0f} {game['best_over_book']}</b></div>
"""
            if pd.notna(game.get('best_under_price')):
                html += f"""          <div>Best under<b class="best">{game['best_under_price']:+.0f} {game['best_under_book']}</b></div>
"""
            if pd.notna(game.get('calibrated_projection')):
                html += f"""          <div>Model projection &middot; raw<b>{game['model_projection']:.1f}</b></div>
          <div>Model projection &middot; shrunk to record<b>{game['calibrated_projection']:.1f}</b></div>
          <div>Model over vs market over<b>{game['model_over_prob']*100:.1f}% vs {game['market_over_prob']*100:.1f}%</b></div>
"""
            elif has_model and pd.notna(game.get('total_pred')):
                html += f"""          <div>Model total<b>{game['total_pred']:.1f}</b></div>
"""
            html += """        </div>
"""

            # Show all book lines for this game
            game_lines = lines[lines['game'] == game['game']] if len(lines) > 0 else pd.DataFrame()
            if len(game_lines) > 0:
                n_books = int(game_lines['book'].nunique())
                best_over = game.get('best_over_price')
                best_under = game.get('best_under_price')
                cons_total = game.get('consensus_total')
                html += f"""
        <details class="book-lines">
          <summary>All {n_books} sportsbook lines</summary>
          <div class="table-wrap">
          <table class="book-lines-table">
            <thead>
              <tr>
                <th scope="col">Book</th>
                <th scope="col" class="num">Total</th>
                <th scope="col" class="num">Over</th>
                <th scope="col" class="num">Under</th>
                <th scope="col">Spread</th>
                <th scope="col" class="num">Fav</th>
                <th scope="col" class="num">Dog</th>
              </tr>
            </thead>
            <tbody>
"""
                # Consensus first, as the reference row for everything below it
                if pd.notna(cons_total):
                    spread_str = (f"{game['home_team']} {game['consensus_spread']:+.1f}"
                                  if pd.notna(game.get('consensus_spread')) else '&mdash;')
                    html += f"""
              <tr class="consensus-row">
                <td><strong>Consensus</strong></td>
                <td class="num">{cons_total:.1f}</td>
                <td class="num">&mdash;</td>
                <td class="num">&mdash;</td>
                <td>{spread_str}</td>
                <td class="num">&mdash;</td>
                <td class="num">&mdash;</td>
              </tr>
"""

                for _, line in game_lines.sort_values('book').iterrows():
                    on_line = pd.notna(line.get('total_over_line')) and pd.notna(cons_total) and line['total_over_line'] == cons_total
                    total_str = f"{line['total_over_line']:.1f}" if pd.notna(line.get('total_over_line')) else '&mdash;'
                    over_odds = f"{int(line['total_over_price']):+d}" if pd.notna(line.get('total_over_price')) else '&mdash;'
                    under_odds = f"{int(line['total_under_price']):+d}" if pd.notna(line.get('total_under_price')) else '&mdash;'
                    # Only a book quoting the consensus line can hold the best price.
                    over_cls = ' class="num best"' if on_line and pd.notna(best_over) and line.get('total_over_price') == best_over else ' class="num"'
                    under_cls = ' class="num best"' if on_line and pd.notna(best_under) and line.get('total_under_price') == best_under else ' class="num"'

                    spread_str = f"{game['home_team']} {line['spread_home_line']:+.1f}" if pd.notna(line.get('spread_home_line')) else '&mdash;'
                    spread_home_odds = f"{int(line['spread_home_price']):+d}" if pd.notna(line.get('spread_home_price')) else '&mdash;'
                    spread_away_odds = f"{int(line['spread_away_price']):+d}" if pd.notna(line.get('spread_away_price')) else '&mdash;'

                    html += f"""
              <tr>
                <td>{line['book']}</td>
                <td class="num">{total_str}</td>
                <td{over_cls}>{over_odds}</td>
                <td{under_cls}>{under_odds}</td>
                <td>{spread_str}</td>
                <td class="num">{spread_home_odds}</td>
                <td class="num">{spread_away_odds}</td>
              </tr>
"""

                html += """
            </tbody>
          </table>
          </div>
        </details>
"""

            # Show edge plays for this game
            if has_edge:
                html += """
        <div class="edge-play-list">
          <p class="eyebrow">Consensus edge plays</p>
"""
                for _, play in game_edges.iterrows():
                    html += f"""
          <p class="meta"><span class="tag">{play['bet']} {play['line']}</span> at {play['book']}
          &middot; consensus {play['consensus']:.1f} &middot; model {play['model']:.1f}
          &middot; edge {play['edge']:.1f} pts</p>
"""
                html += """
        </div>
"""

            html += """
      </article>
"""
    else:
        html += """
      <div class="empty">
        <h3>No games on the board</h3>
        <p class="meta">The next scheduled refresh will populate this page.</p>
      </div>
"""

    quote_line = (f"Sportsbook quotes provided at {quoted_at}."
                  if quoted_at else
                  "Sportsbook quote times were not supplied by the provider for this snapshot.")

    html += f"""
      </div>
    </section>

    <footer>
      <p>{quote_line} Page built {datetime.now().strftime('%b %-d, %Y at %-I:%M %p')} local time &mdash;
      build time is not quote time.</p>
      <p>Lines move. Confirm the current price at your sportsbook before betting.</p>
    </footer>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
  <script src="../../tracking/bet-tracking.js"></script>
  <script>
  // Add Track buttons to NFL totals page
  document.addEventListener('DOMContentLoaded', function() {{
    const isAuthenticated = sessionStorage.getItem('betTrackingAuth') === 'true';

    if (!isAuthenticated) return;

    // Find all book lines tables
    const tables = document.querySelectorAll('.book-lines-table');

    tables.forEach(table => {{
      // Add Track header
      const headerRow = table.querySelector('thead tr');
      if (headerRow && !headerRow.querySelector('.track-header')) {{
        const th = document.createElement('th');
        th.className = 'track-header';
        th.textContent = 'Track';
        headerRow.appendChild(th);
      }}

      // Find parent game card to get team info
      const gameCard = table.closest('.game-card');
      if (!gameCard) return;

      const matchup = gameCard.querySelector('.matchup')?.textContent || '';
      const [awayTeam, homeTeam] = matchup.split(' @ ').map(s => s.trim());

      // Add Track buttons to each book line row (skip consensus row)
      const rows = table.querySelectorAll('tbody tr:not(.consensus-row)');
      rows.forEach(row => {{
        if (row.querySelector('.track-cell')) return; // Already added

        const cells = row.querySelectorAll('td');
        if (cells.length < 7) return;

        const book = cells[0].textContent.trim();
        const totalLine = parseFloat(cells[1].textContent) || 0;
        const overPrice = cells[2].textContent.trim();
        const underPrice = cells[3].textContent.trim();

        // Create track cell with O/U buttons
        const trackCell = document.createElement('td');
        trackCell.className = 'track-cell';
        trackCell.style.cssText = 'display:flex; gap:0.25rem;';

        // Over button
        const overBtn = document.createElement('button');
        overBtn.textContent = 'O';
        overBtn.style.cssText = 'min-height:32px;padding:4px 10px;background:var(--accent);color:#10241c;border:none;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;';
        overBtn.onclick = () => window.trackNFLTotal(homeTeam, awayTeam, book, totalLine, overPrice, 'over');

        // Under button
        const underBtn = document.createElement('button');
        underBtn.textContent = 'U';
        underBtn.style.cssText = 'min-height:32px;padding:4px 10px;background:var(--accent);color:#10241c;border:none;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;';
        underBtn.onclick = () => window.trackNFLTotal(homeTeam, awayTeam, book, totalLine, underPrice, 'under');

        trackCell.appendChild(overBtn);
        trackCell.appendChild(underBtn);
        row.appendChild(trackCell);
      }});
    }});
  }});

  // Track NFL Team Total function
  window.trackNFLTotal = function(homeTeam, awayTeam, book, line, odds, side) {{
    const stake = prompt('Enter stake amount ($):', '100');
    if (!stake || isNaN(parseFloat(stake))) {{
      alert('Invalid stake amount');
      return;
    }}

    // Parse odds (remove + or -)
    const oddsNum = parseInt(odds.replace('+', '').replace('-', ''));
    const oddsStr = odds.includes('-') ? `-${{oddsNum}}` : oddsNum.toString();

    const today = new Date().toISOString().split('T')[0];

    const bet = {{
      bet_id: 'bet_' + Date.now(),
      timestamp: new Date().toISOString(),
      league: 'NFL',
      game_date: today,
      team_home: homeTeam,
      team_away: awayTeam,
      player: '',
      market_type: 'team_total',
      side: side,
      line: line,
      book: book,
      odds: oddsStr,
      stake_dollars: parseFloat(stake).toFixed(2),
      status: 'pending',
      actual_result: '',
      payout: '',
      graded_timestamp: '',
      model_prob: '',
      edge_bps: ''
    }};

    // Auto-track bet via GitHub API
    if (window.autoTrackBet) {{
      autoTrackBet(bet);
    }} else {{
      alert('Error: Auto-tracking unavailable. Please refresh the page and try again.');
    }}
  }};
  </script>

  <script>
    // Game search filter
    const searchBox = document.getElementById('gameSearch');
    const gameCards = document.querySelectorAll('.game-card');

    searchBox.addEventListener('input', (e) => {{
      const searchTerm = e.target.value.toLowerCase().trim();

      gameCards.forEach(card => {{
        const matchup = card.querySelector('.matchup').textContent.toLowerCase();
        card.hidden = !(searchTerm === '' || matchup.includes(searchTerm));
      }});

      // Keep the at-a-glance table in step with the cards.
      document.querySelectorAll('.totals-summary tbody tr').forEach(row => {{
        const game = (row.querySelector('td a')?.textContent || '').toLowerCase();
        row.hidden = !(searchTerm === '' || game.includes(searchTerm));
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
    parser.add_argument('--team-totals', default=None, help='Market-derived team totals CSV')
    parser.add_argument('--priced', default=None, help='Priced model projections CSV')

    args = parser.parse_args()

    build_totals_page(args.predictions, args.consensus, args.edges, args.lines, args.output, args.week,
                      args.team_totals, args.priced)

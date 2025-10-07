#!/usr/bin/env python3
"""
qc_family_coherence.py

Quality control and opportunity detection for family-based modeling.

Outputs:
  1. Family arbitrage opportunities (both linked markets show edge)
  2. Book incoherence (unrealistic implied efficiency metrics)
  3. Model validation (all projections within sanity bounds)

Usage:
  python3 scripts/qc_family_coherence.py --props data/props/props_with_model_week5.csv --params data/props/params_week5.csv
  python3 scripts/qc_family_coherence.py --props data/props/props_with_model_week5.csv --params data/props/params_week5.csv --blog-post
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Sanity bounds (from make_player_prop_params.py)
YPC_MIN, YPC_MAX = 2.5, 6.5
YPR_MIN, YPR_MAX = 6.0, 18.0
COMP_PCT_MIN, COMP_PCT_MAX = 0.50, 0.75
YPC_PASS_MIN, YPC_PASS_MAX = 5.0, 12.0
CR_MIN, CR_MAX = 0.50, 0.95

EDGE_THRESHOLD = 1500  # bps threshold for "strong" edge


def load_data(props_path, params_path):
    """Load props and params dataframes."""
    props = pd.read_csv(props_path)
    params = pd.read_csv(params_path)
    return props, params


def calculate_book_implied_metrics(props):
    """
    Calculate book-implied efficiency metrics from line pairs.

    For each player with both volume and yards markets:
    - Rush: implied_ypc = rush_yds_line / rush_attempts_line
    - Receive: implied_ypr = recv_yds_line / receptions_line
    - Pass: implied_ypc_pass = pass_yds_line / pass_completions_line
    """
    results = []

    # Normalize book column name
    book_col = 'bookmaker' if 'bookmaker' in props.columns else 'book'

    # Group by player and book
    grouped = props.groupby(['name_std', book_col])

    for (player, book), grp in grouped:
        record = {'player': player, 'book': book}

        # Rush family
        rush_att = grp[grp['market_std'] == 'rush_attempts']['point'].values
        rush_yds = grp[grp['market_std'] == 'rush_yds']['point'].values
        if len(rush_att) > 0 and len(rush_yds) > 0:
            if rush_att[0] > 0:
                record['book_implied_ypc'] = rush_yds[0] / rush_att[0]
                record['rush_att_line'] = rush_att[0]
                record['rush_yds_line'] = rush_yds[0]

        # Receive family
        receptions = grp[grp['market_std'] == 'receptions']['point'].values
        recv_yds = grp[grp['market_std'] == 'recv_yds']['point'].values
        if len(receptions) > 0 and len(recv_yds) > 0:
            if receptions[0] > 0:
                record['book_implied_ypr'] = recv_yds[0] / receptions[0]
                record['receptions_line'] = receptions[0]
                record['recv_yds_line'] = recv_yds[0]

        # Pass family
        pass_comp = grp[grp['market_std'] == 'pass_completions']['point'].values
        pass_yds = grp[grp['market_std'] == 'pass_yds']['point'].values
        if len(pass_comp) > 0 and len(pass_yds) > 0:
            if pass_comp[0] > 0:
                record['book_implied_ypc_pass'] = pass_yds[0] / pass_comp[0]
                record['pass_comp_line'] = pass_comp[0]
                record['pass_yds_line'] = pass_yds[0]

        if len(record) > 2:  # Has at least one metric
            results.append(record)

    return pd.DataFrame(results)


def find_family_arbitrage(props, threshold=EDGE_THRESHOLD):
    """
    Find opportunities where BOTH linked markets show strong edges in same direction.

    These are high-conviction plays where volume AND efficiency both point same way.
    """
    opportunities = []

    # Normalize book column name
    book_col = 'bookmaker' if 'bookmaker' in props.columns else 'book'

    # Group by player
    grouped = props.groupby('name_std')

    for player, grp in grouped:
        # Rush family
        rush_att = grp[grp['market_std'] == 'rush_attempts']
        rush_yds = grp[grp['market_std'] == 'rush_yds']

        if not rush_att.empty and not rush_yds.empty:
            for _, att_row in rush_att.iterrows():
                for _, yds_row in rush_yds.iterrows():
                    att_edge = att_row.get('edge_bps', 0)
                    yds_edge = yds_row.get('edge_bps', 0)

                    # Both must exceed threshold and point same direction
                    if (abs(att_edge) > threshold and abs(yds_edge) > threshold and
                        np.sign(att_edge) == np.sign(yds_edge)):
                        # Infer side from edge (positive edge = over recommended)
                        side_1 = att_row.get('side') if pd.notna(att_row.get('side')) else ('Over' if att_edge > 0 else 'Under')
                        side_2 = yds_row.get('side') if pd.notna(yds_row.get('side')) else ('Over' if yds_edge > 0 else 'Under')

                        opportunities.append({
                            'family': 'Rush',
                            'player': player,
                            'book': att_row.get(book_col, ''),
                            'market_1': 'rush_attempts',
                            'line_1': att_row.get('point'),
                            'side_1': side_1,
                            'edge_1': att_edge,
                            'market_2': 'rush_yds',
                            'line_2': yds_row.get('point'),
                            'side_2': side_2,
                            'edge_2': yds_edge,
                            'total_edge': att_edge + yds_edge,
                            'model_line_1': att_row.get('model_line'),
                            'model_line_2': yds_row.get('model_line'),
                        })

        # Receive family
        receptions = grp[grp['market_std'] == 'receptions']
        recv_yds = grp[grp['market_std'] == 'recv_yds']

        if not receptions.empty and not recv_yds.empty:
            for _, rec_row in receptions.iterrows():
                for _, yds_row in recv_yds.iterrows():
                    rec_edge = rec_row.get('edge_bps', 0)
                    yds_edge = yds_row.get('edge_bps', 0)

                    if (abs(rec_edge) > threshold and abs(yds_edge) > threshold and
                        np.sign(rec_edge) == np.sign(yds_edge)):
                        opportunities.append({
                            'family': 'Receive',
                            'player': player,
                            'book': rec_row.get(book_col, ''),
                            'market_1': 'receptions',
                            'line_1': rec_row.get('point'),
                            'side_1': rec_row.get('side'),
                            'edge_1': rec_edge,
                            'market_2': 'recv_yds',
                            'line_2': yds_row.get('point'),
                            'side_2': yds_row.get('side'),
                            'edge_2': yds_edge,
                            'total_edge': rec_edge + yds_edge,
                            'model_line_1': rec_row.get('model_line'),
                            'model_line_2': yds_row.get('model_line'),
                        })

        # Pass family
        pass_att = grp[grp['market_std'] == 'pass_attempts']
        pass_yds = grp[grp['market_std'] == 'pass_yds']

        if not pass_att.empty and not pass_yds.empty:
            for _, att_row in pass_att.iterrows():
                for _, yds_row in pass_yds.iterrows():
                    att_edge = att_row.get('edge_bps', 0)
                    yds_edge = yds_row.get('edge_bps', 0)

                    if (abs(att_edge) > threshold and abs(yds_edge) > threshold and
                        np.sign(att_edge) == np.sign(yds_edge)):
                        opportunities.append({
                            'family': 'Pass',
                            'player': player,
                            'book': att_row.get(book_col, ''),
                            'market_1': 'pass_attempts',
                            'line_1': att_row.get('point'),
                            'side_1': att_row.get('side'),
                            'edge_1': att_edge,
                            'market_2': 'pass_yds',
                            'line_2': yds_row.get('point'),
                            'side_2': yds_row.get('side'),
                            'edge_2': yds_edge,
                            'total_edge': att_edge + yds_edge,
                            'model_line_1': att_row.get('model_line'),
                            'model_line_2': yds_row.get('model_line'),
                        })

    return pd.DataFrame(opportunities)


def find_incoherent_books(book_metrics):
    """
    Find books with unrealistic implied efficiency metrics.

    These are mispricings where the book's lines imply impossible efficiency.
    """
    incoherent = []

    for _, row in book_metrics.iterrows():
        # Check YPC
        if 'book_implied_ypc' in row and pd.notna(row['book_implied_ypc']):
            ypc = row['book_implied_ypc']
            if ypc < YPC_MIN or ypc > YPC_MAX:
                incoherent.append({
                    'family': 'Rush',
                    'player': row['player'],
                    'book': row['book'],
                    'metric': 'YPC',
                    'value': ypc,
                    'min_bound': YPC_MIN,
                    'max_bound': YPC_MAX,
                    'severity': 'HIGH' if (ypc < 2.0 or ypc > 7.0) else 'MEDIUM',
                    'details': f"{row['rush_att_line']} att → {row['rush_yds_line']} yds = {ypc:.2f} YPC"
                })

        # Check YPR
        if 'book_implied_ypr' in row and pd.notna(row['book_implied_ypr']):
            ypr = row['book_implied_ypr']
            if ypr < YPR_MIN or ypr > YPR_MAX:
                incoherent.append({
                    'family': 'Receive',
                    'player': row['player'],
                    'book': row['book'],
                    'metric': 'YPR',
                    'value': ypr,
                    'min_bound': YPR_MIN,
                    'max_bound': YPR_MAX,
                    'severity': 'HIGH' if (ypr < 5.0 or ypr > 20.0) else 'MEDIUM',
                    'details': f"{row['receptions_line']} rec → {row['recv_yds_line']} yds = {ypr:.2f} YPR"
                })

        # Check YPC (pass)
        if 'book_implied_ypc_pass' in row and pd.notna(row['book_implied_ypc_pass']):
            ypc_pass = row['book_implied_ypc_pass']
            if ypc_pass < YPC_PASS_MIN or ypc_pass > YPC_PASS_MAX:
                incoherent.append({
                    'family': 'Pass',
                    'player': row['player'],
                    'book': row['book'],
                    'metric': 'Y/C',
                    'value': ypc_pass,
                    'min_bound': YPC_PASS_MIN,
                    'max_bound': YPC_PASS_MAX,
                    'severity': 'HIGH' if (ypc_pass < 4.0 or ypc_pass > 13.0) else 'MEDIUM',
                    'details': f"{row['pass_comp_line']} comp → {row['pass_yds_line']} yds = {ypc_pass:.2f} Y/C"
                })

    return pd.DataFrame(incoherent)


def validate_model_params(params):
    """
    Validate that all model-implied efficiency metrics are within bounds.
    """
    issues = []

    # Check YPC
    rush = params[params['market_std'] == 'rush_yds']
    for _, row in rush.iterrows():
        ypc = row.get('implied_ypc')
        if pd.notna(ypc) and (ypc < YPC_MIN or ypc > YPC_MAX):
            issues.append({
                'player': row['player'],
                'metric': 'implied_ypc',
                'value': ypc,
                'min': YPC_MIN,
                'max': YPC_MAX,
                'status': 'OUT_OF_BOUNDS'
            })

    # Check YPR
    recv = params[params['market_std'] == 'recv_yds']
    for _, row in recv.iterrows():
        ypr = row.get('implied_ypr')
        if pd.notna(ypr) and (ypr < YPR_MIN or ypr > YPR_MAX):
            issues.append({
                'player': row['player'],
                'metric': 'implied_ypr',
                'value': ypr,
                'min': YPR_MIN,
                'max': YPR_MAX,
                'status': 'OUT_OF_BOUNDS'
            })

    # Check comp %
    pass_mkts = params[params['market_std'].isin(['pass_attempts', 'pass_completions', 'pass_yds'])]
    for _, row in pass_mkts.iterrows():
        comp_pct = row.get('implied_comp_pct')
        if pd.notna(comp_pct) and (comp_pct < COMP_PCT_MIN or comp_pct > COMP_PCT_MAX):
            issues.append({
                'player': row['player'],
                'metric': 'implied_comp_pct',
                'value': comp_pct,
                'min': COMP_PCT_MIN,
                'max': COMP_PCT_MAX,
                'status': 'OUT_OF_BOUNDS'
            })

    return pd.DataFrame(issues)


def generate_blog_post(opportunities, incoherent, out_path='docs/blog/family_opportunities.html'):
    """Generate a blog post about family arbitrage opportunities."""

    if opportunities.empty:
        print("[blog] No family arbitrage opportunities found this week")
        return

    # Get top 5 by total edge
    top_opps = opportunities.nlargest(5, 'total_edge')

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Fourth & Value — Family Arbitrage Opportunities</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="When sportsbooks misprice linked markets" />
  <link rel="icon" href="../favicon.ico" />
  <style>
    :root {{ --max: 860px; --bg:#0b0b0b; --fg:#e9e9e9; --muted:#b9b9b9; --line:#222; --input:#121212; --link:#8ab4ff; }}
    html,body {{ margin:0; background:var(--bg); color:var(--fg); }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; line-height:1.65; }}
    a {{ color: var(--link); text-decoration: underline; }}
    .wrap {{ max-width: var(--max); margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 8px 0 4px; }}
    .meta {{ font-size:.9rem; color: var(--muted); margin: 6px 0 18px; }}
    article h2, article h3 {{ margin-top: 1.4em; }}
    code, pre {{ background:#0f0f0f; border:1px solid var(--line); border-radius:8px; color:#e6e6e6; }}
    pre {{ padding:12px; overflow:auto; }}
    .back {{ display:inline-block; margin-top: 28px; color: var(--link); }}
    .opp-card {{ background:#131a20; border-radius:12px; padding:16px; margin:16px 0; border-left:4px solid #3b82f6; }}
    .opp-header {{ font-size:1.15rem; font-weight:600; margin-bottom:8px; }}
    .opp-details {{ font-size:0.95rem; color:var(--muted); line-height:1.8; }}
    .highlight {{ color:#3b82f6; font-weight:600; }}
  </style>
</head>
<body>
  <div id="nav-root"></div>
  <script src="../nav.js?v=24"></script>

  <main class="wrap">
    <article>
      <header>
        <h1>Family Arbitrage Opportunities</h1>
        <div class="meta"><time datetime="{pd.Timestamp.now().strftime('%Y-%m-%d')}">This Week</time></div>
      </header>

      <p>Our new <strong>family-based modeling</strong> links related markets (like rush attempts and rush yards) through shared efficiency metrics. When sportsbooks move one market but not the other, they create arbitrage opportunities where <em>both</em> linked markets show value in the same direction.</p>

      <p>These are high-conviction plays: volume AND efficiency both point the same way.</p>

      <h2>Top Family Plays This Week</h2>
"""

    for i, row in top_opps.iterrows():
        player = str(row.get('player', '')).replace('_', ' ').title()
        family = str(row.get('family', ''))
        book = str(row.get('book', 'Unknown'))
        side1 = str(row.get('side_1', 'Over'))
        side2 = str(row.get('side_2', 'Over'))
        market_1 = str(row.get('market_1', '')).replace('_', ' ').title()
        market_2 = str(row.get('market_2', '')).replace('_', ' ').title()
        line_1 = row.get('line_1', 0)
        line_2 = row.get('line_2', 0)
        model_line_1 = row.get('model_line_1', 0)
        model_line_2 = row.get('model_line_2', 0)
        edge_1 = row.get('edge_1', 0)
        edge_2 = row.get('edge_2', 0)
        total_edge = row.get('total_edge', 0)

        content += f"""
      <div class="opp-card">
        <div class="opp-header">{player} — {family} Family ({book})</div>
        <div class="opp-details">
          <strong>{market_1}:</strong> {side1} {line_1}<br>
          Model line: {model_line_1:.1f} | Edge: <span class="highlight">+{edge_1:.0f} bps</span><br><br>

          <strong>{market_2}:</strong> {side2} {line_2}<br>
          Model line: {model_line_2:.1f} | Edge: <span class="highlight">+{edge_2:.0f} bps</span><br><br>

          <strong>Combined edge:</strong> <span class="highlight">+{total_edge:.0f} bps</span>
        </div>
      </div>
"""

    # Add incoherent books section if any
    if not incoherent.empty:
        high_severity = incoherent[incoherent['severity'] == 'HIGH']
        if not high_severity.empty:
            content += f"""
      <h2>Books with Incoherent Lines</h2>
      <p>These books have lines that imply impossible efficiency metrics. Prime targets for arbitrage:</p>
      <ul>
"""
            for _, row in high_severity.head(5).iterrows():
                player = str(row.get('player', '')).replace('_', ' ').title()
                book = str(row.get('book', 'Unknown'))
                details = str(row.get('details', ''))
                content += f"        <li><strong>{player}</strong> on {book}: {details}</li>\n"

            content += "      </ul>\n"

    content += """
      <h2>How Family Modeling Works</h2>
      <p>Instead of modeling rush attempts and rush yards independently, we model:</p>
      <ul>
        <li><strong>Volume</strong> (carries) with uncertainty</li>
        <li><strong>Efficiency</strong> (yards per carry) with bounds [2.5–6.5]</li>
        <li><strong>Derive</strong> rush yards = carries × YPC</li>
      </ul>

      <p>When a book sets rush attempts at 18.5 and rush yards at 120.5, they're implicitly pricing YPC at 6.5—the upper bound of realistic efficiency. If our model sees 18 attempts at 4.5 YPC (81 yards), both markets show value.</p>

      <a class="back" href="./index.html">← Back to Blog</a>
    </article>
  </main>
</body>
</html>
"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(content)

    print(f"[blog] Wrote family opportunities post to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="QC family-based modeling coherence")
    parser.add_argument('--props', required=True, help="Path to props_with_model CSV")
    parser.add_argument('--params', required=True, help="Path to params CSV")
    parser.add_argument('--blog-post', action='store_true', help="Generate blog post about opportunities")
    parser.add_argument('--out-dir', default='data/qc', help="Output directory for QC reports")
    args = parser.parse_args()

    print("=" * 60)
    print("FAMILY COHERENCE QC REPORT")
    print("=" * 60)
    print()

    # Load data
    props, params = load_data(args.props, args.params)

    # 1. Find family arbitrage opportunities
    print("[1/3] Finding family arbitrage opportunities...")
    opportunities = find_family_arbitrage(props, threshold=EDGE_THRESHOLD)

    if not opportunities.empty:
        print(f"\n✓ Found {len(opportunities)} family arbitrage plays")
        print("\nTop 10 by combined edge:")
        top10 = opportunities.nlargest(10, 'total_edge')
        for _, row in top10.iterrows():
            book = str(row.get('book') or 'Unknown')
            player = str(row.get('player') or 'Unknown')
            family = str(row.get('family') or 'Unknown')
            side_1 = str(row.get('side_1') or 'Over')
            side_2 = str(row.get('side_2') or 'Over')
            market_1 = str(row.get('market_1') or '')
            market_2 = str(row.get('market_2') or '')
            print(f"  • {player:20s} {family:8s} {book:12s} "
                  f"{side_1:5s} {market_1:15s} ({row['edge_1']:+6.0f} bps) + "
                  f"{side_2:5s} {market_2:15s} ({row['edge_2']:+6.0f} bps) = "
                  f"{row['total_edge']:+6.0f} bps")

        # Save to CSV
        out_path = Path(args.out_dir) / 'family_arbitrage.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        opportunities.to_csv(out_path, index=False)
        print(f"\n  Saved to: {out_path}")
    else:
        print("\n✗ No family arbitrage opportunities found (threshold={} bps)".format(EDGE_THRESHOLD))

    print()

    # 2. Find incoherent book lines
    print("[2/3] Checking for incoherent book lines...")
    book_metrics = calculate_book_implied_metrics(props)
    incoherent = find_incoherent_books(book_metrics)

    if not incoherent.empty:
        print(f"\n⚠ Found {len(incoherent)} incoherent book lines")
        print("\nHigh severity violations:")
        high = incoherent[incoherent['severity'] == 'HIGH']
        for _, row in high.head(10).iterrows():
            print(f"  • {row['player']:20s} {row['book']:12s} {row['metric']:8s} = {row['value']:.2f} "
                  f"(bounds: [{row['min_bound']:.1f}, {row['max_bound']:.1f}])")
            print(f"    {row['details']}")

        # Save to CSV
        out_path = Path(args.out_dir) / 'incoherent_books.csv'
        incoherent.to_csv(out_path, index=False)
        print(f"\n  Saved to: {out_path}")
    else:
        print("\n✓ No incoherent book lines found")

    print()

    # 3. Validate model params
    print("[3/3] Validating model parameters...")
    issues = validate_model_params(params)

    if not issues.empty:
        print(f"\n⚠ Found {len(issues)} model parameter violations")
        for _, row in issues.head(10).iterrows():
            print(f"  • {row['player']:20s} {row['metric']:18s} = {row['value']:.2f} "
                  f"(bounds: [{row['min']:.1f}, {row['max']:.1f}])")

        # Save to CSV
        out_path = Path(args.out_dir) / 'model_param_violations.csv'
        issues.to_csv(out_path, index=False)
        print(f"\n  Saved to: {out_path}")
    else:
        print("\n✓ All model parameters within bounds")

    print()
    print("=" * 60)
    print("QC SUMMARY")
    print("=" * 60)
    print(f"Family arbitrage opportunities: {len(opportunities)}")
    print(f"Incoherent book lines:          {len(incoherent)}")
    print(f"Model parameter violations:     {len(issues)}")
    print()

    # Generate blog post if requested
    if args.blog_post and not opportunities.empty:
        print("Generating blog post...")
        generate_blog_post(opportunities, incoherent)

    return 0


if __name__ == '__main__':
    exit(main())

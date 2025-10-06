#!/usr/bin/env python3
"""Build the Incoherent Book Pricing (Arbitrage) page with live data."""

import argparse
import json
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Build incoherent book pricing page')
    parser.add_argument('--incoherent-csv', required=True, help='Path to incoherent_books.csv')
    parser.add_argument('--props-csv', required=True, help='Path to props_with_model CSV for game info')
    parser.add_argument('--out', required=True, help='Output HTML path')
    args = parser.parse_args()

    # Load data
    incoh = pd.read_csv(args.incoherent_csv)
    props = pd.read_csv(args.props_csv)

    # Get game info from props
    game_map = props[['player_key', 'game']].drop_duplicates().set_index('player_key')['game'].to_dict()

    # Build violation records
    records = []
    for idx, row in incoh.iterrows():
        player = row['player']

        # Get player display name from props
        player_rows = props[props['player_key'] == player]
        if len(player_rows) == 0:
            player_display = player  # Fallback to key
        else:
            player_display = player_rows.iloc[0]['player']

        game = game_map.get(player, 'Unknown')

        record = {
            'player': player,
            'player_display': player_display,
            'game': game,
            'book': row['book'],
            'metric': row['metric'],
            'value': float(row['value']),
            'min_bound': float(row['min_bound']),
            'max_bound': float(row['max_bound']),
            'severity': row['severity'],
            'details': row['details']
        }
        records.append(record)

    # Read template
    template_path = Path(args.out)
    if template_path.exists():
        template = template_path.read_text()
    else:
        # Use default template from docs/props/arbitrage.html
        template = Path('docs/props/arbitrage.html').read_text()

    # Replace data placeholder
    data_json = json.dumps(records, indent=2)
    output = template.replace('{DATA_PLACEHOLDER}', data_json)

    # Write output
    Path(args.out).write_text(output)
    print(f"[arbitrage] wrote {args.out} with {len(records)} incoherent book lines")
    print(f"[arbitrage] HIGH severity: {len([r for r in records if r['severity'] == 'HIGH'])}")
    print(f"[arbitrage] MEDIUM severity: {len([r for r in records if r['severity'] == 'MEDIUM'])}")

if __name__ == '__main__':
    main()

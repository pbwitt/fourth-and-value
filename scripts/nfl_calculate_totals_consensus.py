#!/usr/bin/env python3
"""
Calculate consensus totals and spreads from multiple books.

For each game, compute:
- Median total line across books
- Median spread across books
- Number of books offering each market
- Identify outlier books (≥0.5 from consensus)

Usage:
  python3 scripts/nfl_calculate_totals_consensus.py \
    --lines data/nfl/lines/totals_spreads.csv \
    --output data/nfl/consensus/totals_spreads_consensus.csv
"""
import argparse
import pandas as pd
import os


def calculate_consensus(lines_path, output_path):
    """
    Calculate consensus from book lines.
    """
    print("Loading book lines...")
    df = pd.read_csv(lines_path)

    if len(df) == 0:
        print("No lines found")
        return

    print(f"Loaded {len(df)} book lines for {df['game'].nunique()} games")

    # Calculate consensus for totals
    totals_consensus = []

    for game in df['game'].unique():
        game_lines = df[df['game'] == game].copy()

        # Totals consensus
        total_lines = game_lines.dropna(subset=['total_over_line'])
        if len(total_lines) > 0:
            consensus_total = total_lines['total_over_line'].median()
            num_books_total = len(total_lines)

            # Find outliers (≥0.5 from consensus)
            outliers = total_lines[
                (total_lines['total_over_line'] - consensus_total).abs() >= 0.5
            ]

            totals_consensus.append({
                'game': game,
                'home_team': game_lines.iloc[0]['home_team'],
                'away_team': game_lines.iloc[0]['away_team'],
                'market': 'total',
                'consensus_line': consensus_total,
                'num_books': num_books_total,
                'min_line': total_lines['total_over_line'].min(),
                'max_line': total_lines['total_over_line'].max(),
                'outlier_books': len(outliers),
            })

        # Spreads consensus
        spread_lines = game_lines.dropna(subset=['spread_home_line'])
        if len(spread_lines) > 0:
            consensus_spread = spread_lines['spread_home_line'].median()
            num_books_spread = len(spread_lines)

            outliers = spread_lines[
                (spread_lines['spread_home_line'] - consensus_spread).abs() >= 0.5
            ]

            totals_consensus.append({
                'game': game,
                'home_team': game_lines.iloc[0]['home_team'],
                'away_team': game_lines.iloc[0]['away_team'],
                'market': 'spread',
                'consensus_line': consensus_spread,
                'num_books': num_books_spread,
                'min_line': spread_lines['spread_home_line'].min(),
                'max_line': spread_lines['spread_home_line'].max(),
                'outlier_books': len(outliers),
            })

    # Create consensus dataframe
    consensus_df = pd.DataFrame(totals_consensus)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    consensus_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved consensus to {output_path}")
    print(f"  Total markets: {len(consensus_df[consensus_df['market'] == 'total'])}")
    print(f"  Spread markets: {len(consensus_df[consensus_df['market'] == 'spread'])}")

    # Show summary
    print("\nConsensus Summary:")
    for _, row in consensus_df.iterrows():
        outlier_str = f" ({row['outlier_books']} outliers)" if row['outlier_books'] > 0 else ""
        print(f"  {row['game']} - {row['market']}: {row['consensus_line']:.1f} "
              f"({row['num_books']} books){outlier_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate NFL totals/spreads consensus')
    parser.add_argument('--lines', required=True, help='Path to book lines CSV')
    parser.add_argument('--output', required=True, help='Output consensus CSV path')

    args = parser.parse_args()

    calculate_consensus(args.lines, args.output)

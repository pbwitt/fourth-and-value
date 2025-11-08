"""
Find NFL totals consensus edges
Identify books with stale lines before market corrects
"""
import pandas as pd
import os

def find_consensus_edges(predictions_path='data/nfl/predictions/week_predictions.csv',
                        odds_path='data/odds/latest.csv',
                        output_path='data/nfl/consensus/edges.csv',
                        threshold=0.5,
                        model_consensus_threshold=0.3):
    """
    Find consensus edges for NFL totals

    Strategy:
    1. Calculate market consensus (mean across all books)
    2. Find outlier books (≥threshold from consensus)
    3. Check if model agrees with consensus (within model_consensus_threshold)
    4. Flag plays where model+consensus agree but one book is off
    """
    print("====================================================================")
    print("NFL Totals Consensus Edge Finder")
    print("====================================================================\n")

    # Load predictions
    try:
        preds = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"⚠ Predictions file not found: {predictions_path}")
        return pd.DataFrame()

    # Load odds
    try:
        odds = pd.read_csv(odds_path)
        totals_odds = odds[odds['market'] == 'totals'].copy()
    except FileNotFoundError:
        print("⚠ No odds file found")
        return pd.DataFrame()

    if len(totals_odds) == 0:
        print("⚠ No totals odds found")
        return pd.DataFrame()

    print(f"Found totals odds for {totals_odds['game'].nunique()} games across {totals_odds['bookmaker'].nunique()} books\n")

    # Calculate consensus for each game
    consensus = totals_odds.groupby('game').agg({
        'price': 'mean'  # This is the over/under line
    }).rename(columns={'price': 'consensus_line'}).reset_index()

    consensus['num_books'] = totals_odds.groupby('game').size().values
    consensus['median_line'] = totals_odds.groupby('game')['price'].median().values

    print("Market consensus:")
    print(consensus.to_string(index=False))
    print()

    # Find outlier books
    totals_with_consensus = totals_odds.merge(consensus[['game', 'consensus_line']], on='game')
    totals_with_consensus['diff_from_consensus'] = abs(totals_with_consensus['price'] - totals_with_consensus['consensus_line'])

    outliers = totals_with_consensus[totals_with_consensus['diff_from_consensus'] >= threshold].copy()

    print(f"Found {len(outliers)} outlier lines (≥{threshold} from consensus):\n")

    if len(outliers) > 0:
        print(outliers[['game', 'bookmaker', 'price', 'consensus_line', 'diff_from_consensus']].to_string(index=False))
        print()

    # Merge with model predictions
    outliers_with_model = outliers.merge(
        preds[['game', 'total_pred']].rename(columns={'total_pred': 'model'}),
        on='game',
        how='left'
    )

    # Identify plays where model + consensus agree
    outliers_with_model['model_consensus_diff'] = abs(outliers_with_model['model'] - outliers_with_model['consensus_line'])
    outliers_with_model['model_agrees'] = outliers_with_model['model_consensus_diff'] <= model_consensus_threshold

    plays = outliers_with_model[outliers_with_model['model_agrees']].copy()

    # Determine bet direction
    plays['bet'] = plays.apply(lambda row: 'UNDER' if row['price'] > row['consensus_line'] else 'OVER', axis=1)
    plays['line'] = plays['price']
    plays['edge'] = plays['diff_from_consensus']
    plays['consensus'] = plays['consensus_line']

    # Format output
    plays = plays[['game', 'bookmaker', 'bet', 'line', 'consensus', 'model', 'edge']].copy()
    plays = plays.rename(columns={'bookmaker': 'book'})

    # Save
    if len(plays) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plays.to_csv(output_path, index=False)

        print("====================================================================")
        print(f"✓ Found {len(plays)} potential plays:")
        print("====================================================================\n")

        for _, play in plays.iterrows():
            print(f"{play['game']}")
            print(f"  Book: {play['book']}")
            print(f"  {play['bet']}: {play['bet']} {play['line']}")
            print(f"  Consensus: {play['consensus']:.1f} | Model: {play['model']:.1f}")
            print(f"  Edge: {play['edge']:.1f} points from consensus")
            print(f"  Book {play['line']} vs Consensus {play['consensus']:.1f} (Model: {play['model']:.1f})")
            print()

        print("====================================================================")
        print(f"✓ Saved plays to {output_path}")

        # Also save consensus for reference
        consensus_out = output_path.replace('edges.csv', 'consensus.csv')
        consensus.to_csv(consensus_out, index=False)
        print(f"✓ Saved consensus to {consensus_out}")

    else:
        print("====================================================================")
        print("⚠ No consensus edge plays found")
        print("====================================================================")
        print("This could mean:")
        print("  - All books are in sync with consensus")
        print("  - Model doesn't agree with consensus on any outliers")
        print(f"  - Try lowering threshold (currently {threshold})")

    return plays


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find NFL totals consensus edges')
    parser.add_argument('--predictions', default='data/nfl/predictions/week_predictions.csv', help='Predictions CSV')
    parser.add_argument('--odds', default='data/odds/latest.csv', help='Odds CSV')
    parser.add_argument('--output', default='data/nfl/consensus/edges.csv', help='Output edges CSV')
    parser.add_argument('--threshold', type=float, default=0.5, help='Min difference from consensus (default: 0.5 points)')
    parser.add_argument('--model-threshold', type=float, default=0.3, help='Max model-consensus diff for agreement (default: 0.3 points)')

    args = parser.parse_args()

    find_consensus_edges(args.predictions, args.odds, args.output, args.threshold, args.model_threshold)

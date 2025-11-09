"""
Find NHL totals where individual books differ from consensus
Perfect for exploiting stale lines before they move
"""
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime


def fetch_all_totals_odds(api_key):
    """
    Fetch NHL totals from all books
    """
    SPORT = 'icehockey_nhl'
    REGIONS = 'us'
    MARKETS = 'totals'
    ODDS_FORMAT = 'american'

    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'

    params = {
        'api_key': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching odds: {e}")

    return []


def parse_totals_by_book(odds_json):
    """
    Parse totals for each book separately
    Returns: DataFrame with one row per book per game
    """
    all_totals = []

    for game in odds_json:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']

        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker['title']
            book_key = bookmaker['key']

            for market in bookmaker.get('markets', []):
                if market['key'] == 'totals':
                    # Get the line (point)
                    outcomes = market['outcomes']

                    # Find over and under
                    over_outcome = [o for o in outcomes if o['name'] == 'Over']
                    under_outcome = [o for o in outcomes if o['name'] == 'Under']

                    if over_outcome and under_outcome:
                        line = over_outcome[0].get('point')
                        over_price = over_outcome[0].get('price')
                        under_price = under_outcome[0].get('price')

                        all_totals.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'game': f"{away_team} @ {home_team}",
                            'commence_time': commence_time,
                            'book_name': book_name,
                            'book_key': book_key,
                            'line': line,
                            'over_price': over_price,
                            'under_price': under_price
                        })

    return pd.DataFrame(all_totals)


def calculate_consensus(totals_df):
    """
    Calculate market consensus line for each game
    """
    consensus = totals_df.groupby(['home_team', 'away_team', 'game']).agg({
        'line': ['mean', 'median', 'std', 'count'],
        'commence_time': 'first'
    }).reset_index()

    consensus.columns = ['home_team', 'away_team', 'game', 'consensus_line', 'median_line',
                         'line_std', 'num_books', 'commence_time']

    return consensus


def find_outlier_books(totals_df, consensus_df, threshold=0.5):
    """
    Find books with lines that differ from consensus by threshold

    Args:
        threshold: Minimum difference from consensus to flag (e.g., 0.5 goals)
    """
    # Merge consensus back into totals
    merged = totals_df.merge(
        consensus_df[['home_team', 'away_team', 'consensus_line', 'median_line', 'num_books']],
        on=['home_team', 'away_team']
    )

    # Calculate difference from consensus
    merged['diff_from_consensus'] = merged['line'] - merged['consensus_line']
    merged['diff_from_median'] = merged['line'] - merged['median_line']

    # Flag outliers
    merged['is_outlier'] = abs(merged['diff_from_consensus']) >= threshold

    # Sort by absolute difference
    merged['abs_diff'] = abs(merged['diff_from_consensus'])
    outliers = merged[merged['is_outlier']].sort_values('abs_diff', ascending=False)

    return outliers


def add_model_predictions(outliers_df, predictions_path):
    """
    Add model predictions to outliers
    """
    if not os.path.exists(predictions_path):
        print(f"Warning: Predictions file not found: {predictions_path}")
        return outliers_df

    preds = pd.read_csv(predictions_path)

    # Merge predictions
    result = outliers_df.merge(
        preds[['home_team', 'away_team', 'total_pred']],
        on=['home_team', 'away_team'],
        how='left'
    )

    # Calculate model vs book line
    result['model_vs_book'] = result['total_pred'] - result['line']

    # Calculate model vs consensus
    result['model_vs_consensus'] = result['total_pred'] - result['consensus_line']

    return result


def identify_plays(outliers_with_model, model_consensus_threshold=0.3):
    """
    Identify specific betting opportunities

    Strategy:
    - Model agrees with consensus (within threshold)
    - Book is an outlier from consensus
    - Bet towards consensus before the line moves
    """
    plays = []

    for _, row in outliers_with_model.iterrows():
        game = row['game']
        book = row['book_name']
        book_line = row['line']
        consensus = row['consensus_line']
        model_pred = row.get('total_pred')

        if pd.isna(model_pred):
            continue

        # Check if model agrees with consensus
        model_consensus_diff = abs(model_pred - consensus)
        model_agrees = model_consensus_diff <= model_consensus_threshold

        if not model_agrees:
            continue

        # Book is higher than consensus → Bet UNDER at the book
        if book_line > consensus:
            plays.append({
                'game': game,
                'book': book,
                'line': book_line,
                'consensus': round(consensus, 1),
                'model': round(model_pred, 1),
                'bet': 'UNDER',
                'book_line_display': f"Under {book_line}",
                'edge': round(book_line - consensus, 1),
                'model_edge': round(book_line - model_pred, 1),
                'under_price': row['under_price'],
                'num_books': row['num_books'],
                'reasoning': f"Book {book_line} vs Consensus {round(consensus, 1)} (Model: {round(model_pred, 1)})"
            })

        # Book is lower than consensus → Bet OVER at the book
        elif book_line < consensus:
            plays.append({
                'game': game,
                'book': book,
                'line': book_line,
                'consensus': round(consensus, 1),
                'model': round(model_pred, 1),
                'bet': 'OVER',
                'book_line_display': f"Over {book_line}",
                'edge': round(consensus - book_line, 1),
                'model_edge': round(model_pred - book_line, 1),
                'under_price': row['over_price'],
                'num_books': row['num_books'],
                'reasoning': f"Book {book_line} vs Consensus {round(consensus, 1)} (Model: {round(model_pred, 1)})"
            })

    return pd.DataFrame(plays)


def main():
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description='Find NHL consensus edges')
    parser.add_argument('--predictions', default='data/nhl/predictions/today.csv', help='Model predictions CSV')
    parser.add_argument('--output', default='data/nhl/consensus/edges.csv', help='Output file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Outlier threshold (goals)')
    parser.add_argument('--model-threshold', type=float, default=0.3, help='Model-consensus agreement threshold')

    args = parser.parse_args()

    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print("Error: ODDS_API_KEY not found in .env")
        return

    print("=" * 70)
    print("NHL Consensus Edge Finder")
    print("=" * 70)

    # Fetch odds
    print("\n[1/5] Fetching odds from all books...")
    odds_json = fetch_all_totals_odds(api_key)
    print(f"✓ Fetched odds for {len(odds_json)} games")

    # Parse by book
    print("\n[2/5] Parsing totals by book...")
    totals_df = parse_totals_by_book(odds_json)
    print(f"✓ Found {len(totals_df)} book-game combinations")
    print(f"  Books: {totals_df['book_name'].nunique()}")

    # Calculate consensus
    print("\n[3/5] Calculating market consensus...")
    consensus_df = calculate_consensus(totals_df)
    print(f"✓ Calculated consensus for {len(consensus_df)} games")

    # Find outliers
    print(f"\n[4/5] Finding outlier books (threshold: ±{args.threshold} goals)...")
    outliers = find_outlier_books(totals_df, consensus_df, args.threshold)
    print(f"✓ Found {len(outliers)} outlier book lines")

    # Add model predictions
    print("\n[5/5] Adding model predictions...")
    outliers_with_model = add_model_predictions(outliers, args.predictions)

    # Identify plays
    print(f"\n[PLAYS] Identifying betting opportunities...")
    plays = identify_plays(outliers_with_model, args.model_threshold)

    if len(plays) > 0:
        print(f"\n✓ Found {len(plays)} potential plays:")
        print("\n" + "=" * 70)
        for _, play in plays.iterrows():
            print(f"\n{play['game']}")
            print(f"  Book: {play['book']}")
            print(f"  {play['bet']}: {play['book_line_display']} ({play['under_price']:+d})")
            print(f"  Consensus: {play['consensus']} | Model: {play['model']}")
            print(f"  Edge: {play['edge']} goals from consensus")
            print(f"  {play['reasoning']}")
        print("\n" + "=" * 70)

        # Save plays
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        plays.to_csv(args.output, index=False)
        print(f"\n✓ Saved plays to {args.output}")

        # Also save full outliers data
        outliers_path = args.output.replace('edges.csv', 'outliers_full.csv')
        outliers_with_model.to_csv(outliers_path, index=False)
        print(f"✓ Saved full outlier data to {outliers_path}")

    else:
        print("\n  No plays found matching criteria")
        print(f"  - Model must agree with consensus (within {args.model_threshold} goals)")
        print(f"  - Book must differ from consensus (by at least {args.threshold} goals)")

    # Save consensus for reference
    consensus_path = args.output.replace('edges.csv', 'consensus.csv')
    consensus_df.to_csv(consensus_path, index=False)
    print(f"✓ Saved consensus lines to {consensus_path}")

    # Save all book lines for display on website
    book_lines_path = args.output.replace('edges.csv', 'book_lines.csv')
    totals_df.to_csv(book_lines_path, index=False)
    print(f"✓ Saved all book lines to {book_lines_path}")


if __name__ == '__main__':
    main()

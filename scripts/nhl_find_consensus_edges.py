"""
Find NHL totals where individual books differ from consensus
Perfect for exploiting stale lines before they move
"""
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime


# NHL team name to abbreviation mapping
NHL_TEAM_ABBREV = {
    'Anaheim Ducks': 'ANA',
    'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF',
    'Calgary Flames': 'CGY',
    'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL',
    'Detroit Red Wings': 'DET',
    'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA',
    'Los Angeles Kings': 'LAK',
    'Minnesota Wild': 'MIN',
    'Montreal Canadiens': 'MTL',
    'Nashville Predators': 'NSH',
    'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI',
    'New York Rangers': 'NYR',
    'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT',
    'San Jose Sharks': 'SJS',
    'Seattle Kraken': 'SEA',
    'St Louis Blues': 'STL',
    'Tampa Bay Lightning': 'TBL',
    'Toronto Maple Leafs': 'TOR',
    'Vancouver Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH',
    'Winnipeg Jets': 'WPG',
    'Utah Mammoth': 'UTA',  # New team
}


def fetch_all_odds(api_key):
    """
    Fetch NHL totals and spreads from all books
    """
    SPORT = 'icehockey_nhl'
    REGIONS = 'us'
    MARKETS = 'totals,spreads'  # Fetch totals and puck line spreads (not h2h moneyline)
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


def parse_odds_by_book(odds_json):
    """
    Parse totals and spreads for each book separately
    Returns: DataFrame with one row per book per game (with both totals and spreads)
    """
    all_odds = []

    for game in odds_json:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']

        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker['title']
            book_key = bookmaker['key']

            book_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_abbrev': NHL_TEAM_ABBREV.get(home_team, home_team),
                'away_abbrev': NHL_TEAM_ABBREV.get(away_team, away_team),
                'game': f"{away_team} @ {home_team}",
                'commence_time': commence_time,
                'book_name': book_name,
                'book_key': book_key,
            }

            # Parse totals market
            for market in bookmaker.get('markets', []):
                if market['key'] == 'totals':
                    outcomes = market['outcomes']
                    over_outcome = [o for o in outcomes if o['name'] == 'Over']
                    under_outcome = [o for o in outcomes if o['name'] == 'Under']

                    if over_outcome and under_outcome:
                        book_data['total_line'] = over_outcome[0].get('point')
                        book_data['over_price'] = over_outcome[0].get('price')
                        book_data['under_price'] = under_outcome[0].get('price')

                # Parse spreads market (puck line - typically ±1.5)
                elif market['key'] == 'spreads':
                    outcomes = market['outcomes']
                    home_outcome = [o for o in outcomes if o['name'] == home_team]
                    away_outcome = [o for o in outcomes if o['name'] == away_team]

                    if home_outcome and away_outcome:
                        home_spread = home_outcome[0].get('point', 0)
                        away_spread = away_outcome[0].get('point', 0)
                        home_price = home_outcome[0].get('price')
                        away_price = away_outcome[0].get('price')

                        # Format spread from home team perspective
                        if home_spread > 0:
                            # Home team is the underdog (e.g., +1.5)
                            spread_line = home_spread
                            book_data['spread_fav_team'] = away_team
                            book_data['spread_dog_team'] = home_team
                            book_data['spread_fav_price'] = away_price
                            book_data['spread_dog_price'] = home_price
                        else:
                            # Home team is the favorite (e.g., -1.5)
                            spread_line = home_spread
                            book_data['spread_fav_team'] = home_team
                            book_data['spread_dog_team'] = away_team
                            book_data['spread_fav_price'] = home_price
                            book_data['spread_dog_price'] = away_price

                        # Store spread line (home team perspective)
                        book_data['spread_line'] = spread_line
                        book_data['spread_display'] = f"{NHL_TEAM_ABBREV.get(home_team, home_team)} {spread_line:+.1f}"

            # Only add if we have at least totals data
            if 'total_line' in book_data:
                all_odds.append(book_data)

    return pd.DataFrame(all_odds)


def calculate_consensus(odds_df):
    """
    Calculate market consensus for totals and spreads
    """
    agg_dict = {
        'total_line': ['mean', 'median', 'std', 'count'],
        'commence_time': 'first'
    }

    # Add spread consensus if spread data exists
    if 'spread_line' in odds_df.columns:
        agg_dict['spread_line'] = ['mean']

    consensus = odds_df.groupby(['home_team', 'away_team', 'game']).agg(agg_dict).reset_index()

    if 'spread_line' in odds_df.columns:
        consensus.columns = ['home_team', 'away_team', 'game', 'consensus_total', 'median_total',
                             'total_std', 'num_books', 'commence_time', 'consensus_spread']
    else:
        consensus.columns = ['home_team', 'away_team', 'game', 'consensus_total', 'median_total',
                             'total_std', 'num_books', 'commence_time']

    # Add abbreviation columns for easier merging
    consensus['home_abbrev'] = consensus['home_team'].map(NHL_TEAM_ABBREV)
    consensus['away_abbrev'] = consensus['away_team'].map(NHL_TEAM_ABBREV)

    return consensus


def find_outlier_books(odds_df, consensus_df, threshold=0.5):
    """
    Find books with totals lines that differ from consensus by threshold

    Args:
        threshold: Minimum difference from consensus to flag (e.g., 0.5 goals)
    """
    # Merge consensus back into odds
    merged = odds_df.merge(
        consensus_df[['home_team', 'away_team', 'consensus_total', 'median_total', 'num_books']],
        on=['home_team', 'away_team']
    )

    # Calculate difference from consensus
    merged['diff_from_consensus'] = merged['total_line'] - merged['consensus_total']
    merged['diff_from_median'] = merged['total_line'] - merged['median_total']

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
    result['model_vs_book'] = result['total_pred'] - result['total_line']

    # Calculate model vs consensus
    result['model_vs_consensus'] = result['total_pred'] - result['consensus_total']

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
        book_line = row['total_line']
        consensus = row['consensus_total']
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
    print("\n[1/5] Fetching totals and spreads from all books...")
    odds_json = fetch_all_odds(api_key)
    print(f"✓ Fetched odds for {len(odds_json)} games")

    # Parse by book
    print("\n[2/5] Parsing totals and spreads by book...")
    odds_df = parse_odds_by_book(odds_json)
    print(f"✓ Found {len(odds_df)} book-game combinations")
    print(f"  Books: {odds_df['book_name'].nunique()}")

    # Calculate consensus
    print("\n[3/5] Calculating market consensus...")
    consensus_df = calculate_consensus(odds_df)
    print(f"✓ Calculated consensus for {len(consensus_df)} games")

    # Find outliers
    print(f"\n[4/5] Finding outlier books (threshold: ±{args.threshold} goals)...")
    outliers = find_outlier_books(odds_df, consensus_df, args.threshold)
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

    # Save all book lines for display on website (includes both totals and spreads)
    book_lines_path = args.output.replace('edges.csv', 'book_lines.csv')
    odds_df.to_csv(book_lines_path, index=False)
    print(f"✓ Saved all book lines to {book_lines_path}")


if __name__ == '__main__':
    main()

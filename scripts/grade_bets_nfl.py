#!/usr/bin/env python3
"""
Grade pending NFL bets based on player stats.

This script:
1. Fetches every pending NFL bet, across every user, from Supabase
2. For each one where the game has been played:
   - Fetches actual player stats from game logs
   - Determines if bet won/lost/pushed
   - Calculates payout
   - Updates the bet's status via the Supabase API
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from supabase_client import fetch_pending_bets, update_bet


# Market type mapping to stat column names
MARKET_MAP = {
    'rec_yds': 'receiving_yards',
    'receiving_yds': 'receiving_yards',
    'rush_yds': 'rushing_yards',
    'rushing_yds': 'rushing_yards',
    'pass_yds': 'passing_yards',
    'passing_yds': 'passing_yards',
    'receptions': 'receptions',
    'rec': 'receptions',
    'rush_att': 'rushing_attempts',
    'pass_td': 'passing_touchdowns',
    'rush_td': 'rushing_touchdowns',
    'rec_td': 'receiving_touchdowns',
}


def american_odds_to_decimal(odds):
    """Convert American odds to decimal odds."""
    odds = int(odds)
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))


def calculate_payout(stake, odds, won):
    """
    Calculate payout for a bet.

    Args:
        stake: Bet amount in dollars
        odds: American odds (e.g., -110, +150)
        won: Boolean, True if bet won

    Returns:
        Payout amount (stake + winnings if won, 0 if lost, stake if push)
    """
    stake = float(stake)

    if won is None:  # Push
        return stake
    elif not won:  # Lost
        return 0.0
    else:  # Won
        decimal_odds = american_odds_to_decimal(odds)
        return stake * decimal_odds


def grade_bet(bet_row, stats_df=None, scores_df=None):
    """
    Grade a single bet against actual player stats or game totals.

    Args:
        bet_row: Dictionary containing bet data
        stats_df: DataFrame of player stats for the game date (optional, for player props)
        scores_df: DataFrame of game scores (optional, for game totals)

    Returns:
        Updated bet_row dictionary, or None if cannot grade yet
    """
    market = bet_row['market_type']
    side = bet_row['side'].lower()
    line = float(bet_row['line'])

    # Handle game totals
    if market == 'team_total':
        if scores_df is None:
            print(f"  ⚠️  No scores data provided for game total bet")
            return None

        team_home = bet_row['team_home']
        team_away = bet_row['team_away']

        # Find game in scores
        game_scores = scores_df[
            ((scores_df['home_team'] == team_home) & (scores_df['away_team'] == team_away)) |
            ((scores_df['home_team'] == team_away) & (scores_df['away_team'] == team_home))
        ]

        if len(game_scores) == 0:
            print(f"  ⚠️  Game {team_home} vs {team_away} not found in scores")
            return None

        if not game_scores.iloc[0]['completed']:
            print(f"  ⚠️  Game {team_home} vs {team_away} not completed yet")
            return None

        # Get combined score (game total)
        game_row = game_scores.iloc[0]
        home_score = float(game_row['home_score'])
        away_score = float(game_row['away_score'])
        actual_value = home_score + away_score

        # Determine won/lost/push
        won = None  # None = push
        if side == 'over':
            if actual_value > line:
                won = True
            elif actual_value < line:
                won = False
        elif side == 'under':
            if actual_value < line:
                won = True
            elif actual_value > line:
                won = False
        else:
            print(f"  ⚠️  Unknown side: {side}")
            return None

        # Calculate payout
        payout = calculate_payout(bet_row['stake_dollars'], bet_row['odds'], won)

        # Determine status
        if won is None:
            status = 'push'
        elif won:
            status = 'won'
        else:
            status = 'lost'

        # Update bet row
        bet_row['status'] = status
        bet_row['actual_result'] = str(actual_value)
        bet_row['payout'] = f"{payout:.2f}"
        bet_row['graded_timestamp'] = datetime.now().isoformat()

        result_str = f"{'WON' if won else 'LOST' if won is False else 'PUSH'}"
        print(f"  ✓ {team_away} @ {team_home} total {side} {line}: {actual_value} → {result_str} (${payout:.2f})")

        return bet_row

    # Handle player props
    player_name = bet_row['player']

    # Check if market is supported
    if market not in MARKET_MAP:
        print(f"  ⚠️  Unsupported market type: {market}")
        return None

    if stats_df is None:
        print(f"  ⚠️  No stats data provided for player prop bet")
        return None

    stat_col = MARKET_MAP[market]

    # Find player in stats
    player_stats = stats_df[stats_df['player'] == player_name]

    if len(player_stats) == 0:
        print(f"  ⚠️  Player {player_name} not found in stats for {bet_row['game_date']}")
        return None

    # Get actual stat value
    actual_value = float(player_stats.iloc[0][stat_col])

    # Determine won/lost/push
    won = None  # None = push
    if side == 'over':
        if actual_value > line:
            won = True
        elif actual_value < line:
            won = False
        # else: push (actual == line)
    elif side == 'under':
        if actual_value < line:
            won = True
        elif actual_value > line:
            won = False
        # else: push (actual == line)
    else:
        print(f"  ⚠️  Unknown side: {side}")
        return None

    # Calculate payout
    payout = calculate_payout(bet_row['stake_dollars'], bet_row['odds'], won)

    # Determine status
    if won is None:
        status = 'push'
    elif won:
        status = 'won'
    else:
        status = 'lost'

    # Update bet row
    bet_row['status'] = status
    bet_row['actual_result'] = str(actual_value)
    bet_row['payout'] = f"{payout:.2f}"
    bet_row['graded_timestamp'] = datetime.now().isoformat()

    result_str = f"{'WON' if won else 'LOST' if won is False else 'PUSH'}"
    print(f"  ✓ {player_name} {market} {side} {line}: {actual_value} → {result_str} (${payout:.2f})")

    return bet_row


def main():
    parser = argparse.ArgumentParser(description='Grade pending NFL bets')
    parser.add_argument('--date', help='Grade bets for specific date (YYYY-MM-DD)', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Show what would be graded without writing')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    nfl_data_dir = repo_root / 'data' / 'weekly'

    print("=" * 70)
    print("NFL Bet Auto-Grading")
    print("=" * 70)

    pending_nfl_bets = fetch_pending_bets('NFL')
    print(f"⏳ Found {len(pending_nfl_bets)} pending NFL bets (across all users)")

    if len(pending_nfl_bets) == 0:
        print("✓ No pending NFL bets to grade")
        return

    # Determine dates to check
    if args.date:
        check_dates = [args.date]
    else:
        # Check last 7 days for NFL games
        today = datetime.now().date()
        check_dates = [(today - timedelta(days=i)).isoformat() for i in range(7)]

    print(f"📅 Checking game dates: {', '.join(check_dates[:3])}...")

    # Load scores once for all team totals
    scores_file = repo_root / 'data' / 'nfl' / 'scores' / 'scores_latest.csv'
    all_scores_df = None
    if scores_file.exists():
        try:
            all_scores_df = pd.read_csv(scores_file)
            print(f"✓ Loaded {len(all_scores_df)} game scores")
        except Exception as e:
            print(f"⚠️  Error reading scores: {e}")
    else:
        print(f"⚠️  Scores file not found: {scores_file}")

    # Grade bets
    graded_count = 0
    for bet in pending_nfl_bets:
        game_date = bet['game_date']

        print(f"\n🎲 Grading bet {bet['id']}:")
        player_display = bet.get('player', '')
        print(f"   {player_display} {bet['market_type']} {bet['side']} {bet['line']} @ {bet['book']}")

        # Load data based on bet type
        stats_df = None
        scores_df = None

        if bet['market_type'] == 'team_total':
            # Team totals need scores, not player stats
            # Check if this game has a score available (by team matchup, not date)
            if all_scores_df is None:
                print(f"  ⚠️  Scores not available")
                continue

            team_home = bet['team_home']
            team_away = bet['team_away']

            print(f"  Looking for: {team_away} @ {team_home}")

            # Find game in scores by teams
            game_match = all_scores_df[
                ((all_scores_df['home_team'] == team_home) & (all_scores_df['away_team'] == team_away)) |
                ((all_scores_df['home_team'] == team_away) & (all_scores_df['away_team'] == team_home))
            ]

            print(f"  Found {len(game_match)} matching games")

            if len(game_match) == 0:
                print(f"  ⚠️  Game {team_away} @ {team_home} not found in scores")
                print(f"  Available teams in scores:")
                for _, row in all_scores_df.head(5).iterrows():
                    print(f"    {row['away_team']} @ {row['home_team']}")
                continue

            if not game_match.iloc[0]['completed']:
                print(f"  ⚠️  Game {team_away} @ {team_home} not completed yet")
                continue

            scores_df = all_scores_df
        else:
            # Player props need player stats
            possible_stats_files = [
                nfl_data_dir / f"logs_{game_date}.parquet",
                nfl_data_dir / "logs_latest.parquet",
            ]

            for stats_file in possible_stats_files:
                if stats_file.exists():
                    try:
                        stats_df = pd.read_parquet(stats_file)
                        break
                    except Exception as e:
                        print(f"  ⚠️  Error reading {stats_file}: {e}")
                        continue

            if stats_df is None:
                print(f"  ⚠️  Stats not available yet for {game_date}")
                continue

        try:
            # Grade the bet
            updated_bet = grade_bet(bet, stats_df=stats_df, scores_df=scores_df)

            if updated_bet:
                if not args.dry_run:
                    update_bet(updated_bet['id'], {
                        'status': updated_bet['status'],
                        'actual_result': updated_bet['actual_result'],
                        'payout': updated_bet['payout'],
                        'graded_timestamp': updated_bet['graded_timestamp'],
                    })
                graded_count += 1

        except Exception as e:
            print(f"  ❌ Error grading bet: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"📈 Grading Summary: {graded_count} bets graded")
    print("=" * 70)

    if args.dry_run:
        print("🔍 DRY RUN - No changes written")
    elif graded_count > 0:
        print("✓ Updated bets written to Supabase")
    else:
        print("ℹ️  No bets graded")


if __name__ == '__main__':
    main()

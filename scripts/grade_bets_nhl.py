#!/usr/bin/env python3
"""
Grade pending NHL bets based on player stats.

This script:
1. Reads bets from data/bets/bets.csv
2. For each pending NHL bet where the game has been played:
   - Fetches actual player stats from skater logs
   - Determines if bet won/lost/pushed
   - Calculates payout
   - Updates bet status
3. Writes updated bets back to CSV
"""

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


# Market type mapping to stat column names
MARKET_MAP = {
    'goals': 'goals',
    'assists': 'assists',
    'points': 'points',
    'sog': 'shots',
    'shots': 'shots',
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


def grade_bet(bet_row, stats_df):
    """
    Grade a single bet against actual player stats.

    Args:
        bet_row: Dictionary containing bet data
        stats_df: DataFrame of player stats for the game date

    Returns:
        Updated bet_row dictionary, or None if cannot grade yet
    """
    player_name = bet_row['player']
    market = bet_row['market_type']
    side = bet_row['side'].lower()
    line = float(bet_row['line'])

    # Check if market is supported
    if market not in MARKET_MAP:
        print(f"  ⚠️  Unsupported market type: {market}")
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
    parser = argparse.ArgumentParser(description='Grade pending NHL bets')
    parser.add_argument('--date', help='Grade bets for specific date (YYYY-MM-DD)', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Show what would be graded without writing')
    args = parser.parse_args()

    # Paths
    repo_root = Path(__file__).parent.parent
    bets_csv = repo_root / 'data' / 'bets' / 'bets.csv'
    nhl_proc_dir = repo_root / 'data' / 'nhl' / 'processed'

    print("=" * 70)
    print("NHL Bet Auto-Grading")
    print("=" * 70)

    # Read bets CSV
    if not bets_csv.exists():
        print(f"❌ Bets file not found: {bets_csv}")
        return

    with open(bets_csv, 'r') as f:
        reader = csv.DictReader(f)
        bets = list(reader)

    print(f"📊 Loaded {len(bets)} total bets")

    # Filter to pending NHL bets
    pending_nhl_bets = [b for b in bets if b['league'] == 'NHL' and b['status'] == 'pending']
    print(f"⏳ Found {len(pending_nhl_bets)} pending NHL bets")

    if len(pending_nhl_bets) == 0:
        print("✓ No pending NHL bets to grade")
        return

    # Determine dates to check
    if args.date:
        check_dates = [args.date]
    else:
        # Check yesterday and today (in case games finished late)
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        check_dates = [yesterday.isoformat(), today.isoformat()]

    print(f"📅 Checking game dates: {', '.join(check_dates)}")

    # Grade bets
    graded_count = 0
    for bet in pending_nhl_bets:
        game_date = bet['game_date']

        # Skip if game hasn't happened yet
        if game_date not in check_dates:
            continue

        print(f"\n🎲 Grading bet {bet['bet_id']}:")
        print(f"   {bet['player']} {bet['market_type']} {bet['side']} {bet['line']} @ {bet['book']}")

        # Load stats for game date
        stats_file = nhl_proc_dir / f"skater_logs_{game_date}.parquet"

        if not stats_file.exists():
            print(f"  ⚠️  Stats not available yet for {game_date}")
            continue

        try:
            stats_df = pd.read_parquet(stats_file)

            # Grade the bet
            updated_bet = grade_bet(bet, stats_df)

            if updated_bet:
                # Update in main list
                bet_idx = bets.index(bet)
                bets[bet_idx] = updated_bet
                graded_count += 1

        except Exception as e:
            print(f"  ❌ Error grading bet: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"📈 Grading Summary: {graded_count} bets graded")
    print("=" * 70)

    # Write updated bets back to CSV
    if graded_count > 0 and not args.dry_run:
        fieldnames = list(bets[0].keys())

        with open(bets_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bets)

        print(f"✓ Updated bets written to {bets_csv}")
    elif args.dry_run:
        print("🔍 DRY RUN - No changes written")
    else:
        print("ℹ️  No bets graded, CSV not modified")


if __name__ == '__main__':
    main()

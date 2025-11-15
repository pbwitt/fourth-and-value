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
    'Shots on Goal': 'shots',
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


def grade_bet(bet_row, stats_df, games_df=None):
    """
    Grade a single bet against actual player stats or game totals.

    Args:
        bet_row: Dictionary containing bet data
        stats_df: DataFrame of player stats for the game date
        games_df: DataFrame of game scores (optional, for team_total bets)

    Returns:
        Updated bet_row dictionary, or None if cannot grade yet
    """
    market = bet_row['market_type']
    side = bet_row['side'].lower()
    line = float(bet_row['line'])

    # Handle team_total bets separately
    if market == 'team_total':
        if games_df is None:
            print(f"  ⚠️  No games data provided for team_total bet")
            return None

        team_home = bet_row['team_home']
        team_away = bet_row['team_away']
        game_date = bet_row['game_date']

        # Find the game
        game = games_df[
            (games_df['home_team'] == team_home) &
            (games_df['away_team'] == team_away) &
            (games_df['game_date'] == game_date)
        ]

        if len(game) == 0:
            print(f"  ⚠️  Game {team_away} @ {team_home} not found for {game_date}")
            return None

        # Get total score
        home_score = float(game.iloc[0]['home_score']) if 'home_score' in game.columns else 0
        away_score = float(game.iloc[0]['away_score']) if 'away_score' in game.columns else 0
        actual_value = home_score + away_score

    else:
        # Player prop bet
        player_name = bet_row['player']

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

    # Print appropriate message based on bet type
    if market == 'team_total':
        print(f"  ✓ {bet_row['team_away']} @ {bet_row['team_home']} {market} {side} {line}: {actual_value} → {result_str} (${payout:.2f})")
    else:
        print(f"  ✓ {player_name} {market} {side} {line}: {actual_value} → {result_str} (${payout:.2f})")

    return bet_row


def main():
    parser = argparse.ArgumentParser(description='Grade pending NHL bets')
    parser.add_argument('--date', help='Grade bets for specific date (YYYY-MM-DD)', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Show what would be graded without writing')
    args = parser.parse_args()

    # Paths
    repo_root = Path(__file__).parent.parent
    # Use production bets file (docs/data/bets/bets.csv) as source of truth
    bets_csv = repo_root / 'docs' / 'data' / 'bets' / 'bets.csv'
    nhl_proc_dir = repo_root / 'data' / 'nhl' / 'processed'
    nhl_raw_dir = repo_root / 'data' / 'nhl' / 'raw'

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
        # Check all dates from pending bets (up to 7 days old to avoid grading very old games)
        today = datetime.now().date()
        unique_dates = sorted(set(bet['game_date'] for bet in pending_nhl_bets))
        cutoff_date = (today - timedelta(days=7)).isoformat()
        check_dates = [d for d in unique_dates if d >= cutoff_date]

    print(f"📅 Checking game dates: {', '.join(check_dates)}")

    # Load games data for team_total bets
    games_csv = nhl_raw_dir / 'games.csv'
    games_df = None
    if games_csv.exists():
        games_df = pd.read_csv(games_csv)
        # Load player stats to get actual scores
        player_stats_csv = nhl_raw_dir / 'player_stats.csv'
        if player_stats_csv.exists():
            player_stats = pd.read_csv(player_stats_csv)
            # Get scores by aggregating goalie stats
            game_scores = player_stats[player_stats['position_type'] == 'G'].groupby(
                ['game_id', 'game_date']
            ).agg({
                'team': 'first',
                'opponent': 'first',
                'goals_against': 'first'
            }).reset_index()

            # Pivot to get home/away scores
            games_with_scores = []
            for game_id in game_scores['game_id'].unique():
                game_data = game_scores[game_scores['game_id'] == game_id]
                if len(game_data) == 2:
                    game_date = game_data.iloc[0]['game_date']
                    team1 = game_data.iloc[0]['team']
                    team2 = game_data.iloc[1]['team']
                    score1 = game_data.iloc[1]['goals_against']  # Team 1's score is team 2's goals against
                    score2 = game_data.iloc[0]['goals_against']  # Team 2's score is team 1's goals against

                    # Match to games.csv to get home/away
                    game_info = games_df[games_df['game_id'] == game_id]
                    if len(game_info) > 0:
                        home_team = game_info.iloc[0]['home_team']
                        away_team = game_info.iloc[0]['away_team']

                        if team1 == home_team:
                            home_score, away_score = score1, score2
                        else:
                            home_score, away_score = score2, score1

                        games_with_scores.append({
                            'game_id': game_id,
                            'game_date': game_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score
                        })

            if games_with_scores:
                games_df = pd.DataFrame(games_with_scores)

    # Grade bets
    graded_count = 0
    for bet in pending_nhl_bets:
        game_date = bet['game_date']

        # Skip if game hasn't happened yet
        if game_date not in check_dates:
            continue

        print(f"\n🎲 Grading bet {bet['bet_id']}:")
        if bet['market_type'] == 'team_total':
            print(f"   {bet['team_away']} @ {bet['team_home']} {bet['market_type']} {bet['side']} {bet['line']} @ {bet['book']}")
        else:
            print(f"   {bet['player']} {bet['market_type']} {bet['side']} {bet['line']} @ {bet['book']}")

        # Load stats for game date
        stats_file = nhl_proc_dir / f"skater_logs_{game_date}.parquet"

        if not stats_file.exists() and bet['market_type'] != 'team_total':
            print(f"  ⚠️  Stats not available yet for {game_date}")
            continue

        try:
            stats_df = None
            if stats_file.exists():
                stats_df = pd.read_parquet(stats_file)

            # Grade the bet
            updated_bet = grade_bet(bet, stats_df, games_df)

            if updated_bet:
                # Update in main list
                bet_idx = bets.index(bet)
                bets[bet_idx] = updated_bet
                graded_count += 1

        except Exception as e:
            print(f"  ❌ Error grading bet: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print(f"📈 Grading Summary: {graded_count} bets graded")
    print("=" * 70)

    # Write updated bets back to CSV (both locations)
    if graded_count > 0 and not args.dry_run:
        fieldnames = list(bets[0].keys())

        # Write to both docs/data/bets/bets.csv (production) and data/bets/bets.csv (backup)
        bets_csv_backup = repo_root / 'data' / 'bets' / 'bets.csv'

        for csv_path in [bets_csv, bets_csv_backup]:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(bets)

        print(f"✓ Updated bets written to {bets_csv}")
        print(f"✓ Updated bets written to {bets_csv_backup}")
    elif args.dry_run:
        print("🔍 DRY RUN - No changes written")
    else:
        print("ℹ️  No bets graded, CSV not modified")


if __name__ == '__main__':
    main()

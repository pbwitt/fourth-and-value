"""
Quick viewer for NHL consensus edges
Shows today's plays in a readable format
"""
import pandas as pd
import os
import sys

def view_plays():
    """Display today's consensus edge plays"""

    edges_file = 'data/nhl/consensus/edges.csv'
    consensus_file = 'data/nhl/consensus/consensus.csv'

    print("=" * 80)
    print(" " * 20 + "NHL TOTALS - CONSENSUS EDGES")
    print("=" * 80)

    # Check if we have plays
    if not os.path.exists(edges_file):
        print("\n❌ No plays file found")
        print(f"Run: make nhl_totals_daily")
        return

    plays = pd.read_csv(edges_file)

    if len(plays) == 0:
        print("\n📊 No consensus edges found for today")
        print("\nThis means:")
        print("  • All books are in sync with market consensus")
        print("  • Or model doesn't agree with consensus")
        print("\nTry:")
        print("  • Running again later (lines change throughout the day)")
        print("  • Lowering threshold: python3 scripts/nhl_find_consensus_edges.py --threshold 0.3")

        # Show consensus for reference
        if os.path.exists(consensus_file):
            print("\n" + "=" * 80)
            print("TODAY'S CONSENSUS LINES")
            print("=" * 80)
            consensus = pd.read_csv(consensus_file)
            for _, row in consensus.iterrows():
                print(f"\n{row['game']}")
                print(f"  Consensus: {row['consensus_line']:.1f} ({row['num_books']} books)")

        return

    print(f"\n🎯 Found {len(plays)} consensus edge plays for today:\n")
    print("=" * 80)

    for idx, play in plays.iterrows():
        print(f"\n{idx+1}. {play['game']}")
        print(f"   {'─' * 76}")

        # Book and bet
        bet_emoji = "📉" if play['bet'] == 'UNDER' else "📈"
        print(f"   {bet_emoji} BET: {play['bet']} {play['line']} at {play['book']} ({play['under_price']:+d})")

        # Key numbers
        print(f"   📊 Market Consensus: {play['consensus']}")
        print(f"   🤖 Model Prediction: {play['model']}")
        print(f"   📏 Edge from Consensus: {play['edge']} goals")
        print(f"   📚 Books in consensus: {play['num_books']}")

        # Reasoning
        print(f"   💡 {play['reasoning']}")

    print("\n" + "=" * 80)
    print(f"✅ Plays saved to: {edges_file}")
    print("=" * 80)

    # Summary stats
    print(f"\nSummary:")
    print(f"  • {len(plays[plays['bet'] == 'UNDER'])} Under plays")
    print(f"  • {len(plays[plays['bet'] == 'OVER'])} Over plays")
    print(f"  • Average edge: {plays['edge'].mean():.2f} goals")
    print(f"  • Books involved: {', '.join(plays['book'].unique())}")


if __name__ == '__main__':
    view_plays()

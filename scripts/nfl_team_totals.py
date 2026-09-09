#!/usr/bin/env python3
"""Market-derived team totals for the NFL totals board.

Everything here is arithmetic on sportsbook prices: implied team totals from
the consensus total and spread, de-vigged over/under probabilities, and the
best available price. No model estimate is involved.

Prices are only ever combined across books that are quoting the SAME line.
A price at 44.0 and a price at 44.5 are quotes on different outcomes.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_math import implied_probability


def devig_pair(over_price, under_price):
    """Fair (vig-free) over probability and the book's hold, for one book's
    own line. Returns (fair_over, fair_under, hold)."""
    p_over = implied_probability(over_price)
    p_under = implied_probability(under_price)
    if not (np.isfinite(p_over) and np.isfinite(p_under)):
        return np.nan, np.nan, np.nan
    booked = p_over + p_under
    if booked <= 0:
        return np.nan, np.nan, np.nan
    return p_over / booked, p_under / booked, booked - 1.0


def build_team_totals(lines_path, output_path):
    df = pd.read_csv(lines_path)
    if len(df) == 0:
        raise SystemExit(f'No book lines in {lines_path}')

    rows = []
    for game, g in df.groupby('game'):
        totals = g.dropna(subset=['total_over_line'])
        spreads = g.dropna(subset=['spread_home_line'])
        if len(totals) == 0 or len(spreads) == 0:
            continue

        # One vote per book on the points line.
        consensus_total = totals.groupby('book')['total_over_line'].median().median()
        consensus_spread = spreads.groupby('book')['spread_home_line'].median().median()

        # A negative home handicap means the home team is favoured, so it
        # carries the larger share of the total.
        implied_home = consensus_total / 2 - consensus_spread / 2
        implied_away = consensus_total / 2 + consensus_spread / 2

        # De-vig each book against its own pair of prices, then summarise only
        # the books actually quoting the consensus line.
        at_line = totals[totals['total_over_line'] == consensus_total].copy()
        fair = at_line.apply(
            lambda r: devig_pair(r['total_over_price'], r['total_under_price']),
            axis=1, result_type='expand')
        if len(fair):
            at_line[['fair_over', 'fair_under', 'hold']] = fair

        fair_over = at_line['fair_over'].median() if len(at_line) else np.nan
        hold = at_line['hold'].median() if len(at_line) else np.nan

        # Best price is only comparable among books on the same line.
        best_over = at_line.loc[at_line['total_over_price'].idxmax()] if len(at_line) else None
        best_under = at_line.loc[at_line['total_under_price'].idxmax()] if len(at_line) else None

        rows.append({
            'game': game,
            'home_team': g.iloc[0]['home_team'],
            'away_team': g.iloc[0]['away_team'],
            'commence_time': g.iloc[0]['commence_time'],
            'consensus_total': consensus_total,
            'consensus_spread_home': consensus_spread,
            'implied_home_total': implied_home,
            'implied_away_total': implied_away,
            'num_books': totals['book'].nunique(),
            'books_at_line': int(len(at_line)),
            'fair_over_prob': fair_over,
            'fair_under_prob': 1 - fair_over if np.isfinite(fair_over) else np.nan,
            'hold_pct': hold * 100 if np.isfinite(hold) else np.nan,
            'best_over_price': best_over['total_over_price'] if best_over is not None else np.nan,
            'best_over_book': best_over['book'] if best_over is not None else '',
            'best_under_price': best_under['total_under_price'] if best_under is not None else np.nan,
            'best_under_book': best_under['book'] if best_under is not None else '',
            'quoted_at': totals['totals_last_update'].max() if 'totals_last_update' in totals else '',
        })

    out = pd.DataFrame(rows).sort_values('commence_time').reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f'✓ Wrote {len(out)} games to {output_path}')
    show = ['game', 'consensus_total', 'consensus_spread_home',
            'implied_away_total', 'implied_home_total', 'fair_over_prob',
            'hold_pct', 'books_at_line']
    print(out[show].to_string(index=False))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Market-derived NFL team totals')
    ap.add_argument('--lines', required=True, help='Book lines CSV')
    ap.add_argument('--output', required=True, help='Output CSV')
    args = ap.parse_args()
    build_team_totals(args.lines, args.output)

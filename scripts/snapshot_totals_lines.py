#!/usr/bin/env python3
"""Append the current NFL totals/spreads snapshot to a durable history.

One row per distinct quote: a (game, book) pair is stored again only when its
line or one of its prices actually changes. Running this on a schedule builds
two things the site cannot get from a single snapshot:

  - line movement, which is what a reader actually wants to see
  - closing line value, which is the only grading metric that converges
    inside one season

Re-running is safe and cheap. The provider restamps every quote on every poll,
so timestamps cannot be used to detect a change; a poll where nothing repriced
adds no rows.
"""
import argparse
import os
from datetime import datetime, timezone

import pandas as pd

# The provider restamps last_update on every poll even when nothing repriced,
# so a row is distinct by its CONTENT - the line and the prices - not by its
# timestamp. Storing on timestamp would grow the history by a full copy per
# poll while recording no actual price change.
KEYS = ['game', 'book', 'commence_time', 'total_over_line', 'total_over_price',
        'total_under_price', 'spread_home_line', 'spread_home_price',
        'spread_away_price']
STORE = 'data/nfl/lines/history/totals_spreads_history.parquet'


def snapshot(lines_path, store_path=STORE):
    fresh = pd.read_csv(lines_path)
    if len(fresh) == 0:
        raise SystemExit(f'No lines in {lines_path}')

    fresh['captured_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    # Keep the first observation of each distinct quote, so the stored
    # timestamp is when that price was first seen rather than last polled.
    keys = [k for k in KEYS if k in fresh.columns]
    if 'totals_last_update' not in fresh.columns:
        raise SystemExit(
            'Lines carry no provider timestamp. Re-fetch with the current '
            'scripts/nfl_fetch_totals_spreads.py before snapshotting.')

    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    if os.path.exists(store_path):
        history = pd.read_parquet(store_path)
        combined = pd.concat([history, fresh], ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=keys, keep='first')
        added = len(combined) - len(history)
        print(f'{before - len(combined)} unchanged quotes skipped')
    else:
        history = pd.DataFrame()
        combined = fresh.drop_duplicates(subset=keys, keep='first')
        added = len(combined)

    combined.to_parquet(store_path, index=False)
    print(f'✓ {added} new rows -> {store_path}')
    print(f'  history now {len(combined)} rows, '
          f'{combined["game"].nunique()} games, '
          f'{combined["book"].nunique()} books')
    if 'captured_at' in combined.columns:
        print(f'  captures span {combined.captured_at.min()} to {combined.captured_at.max()}')
    return combined


def movement(store_path=STORE, output_path='data/nfl/lines/line_movement.csv'):
    """Per game and book: first quote seen, latest quote, and the move."""
    if not os.path.exists(store_path):
        raise SystemExit(f'No history at {store_path}; run a snapshot first')
    h = pd.read_parquet(store_path).sort_values('totals_last_update')
    h = h.dropna(subset=['total_over_line'])

    rows = []
    for (game, book), g in h.groupby(['game', 'book']):
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({
            'game': game,
            'book': book,
            'opened_total': first['total_over_line'],
            'current_total': last['total_over_line'],
            'total_move': last['total_over_line'] - first['total_over_line'],
            'opened_spread': first.get('spread_home_line'),
            'current_spread': last.get('spread_home_line'),
            'first_seen': first['totals_last_update'],
            'last_seen': last['totals_last_update'],
            'quotes': len(g),
        })
    out = pd.DataFrame(rows)

    # Consensus move across books, one vote each.
    game_move = out.groupby('game').agg(
        opened=('opened_total', 'median'),
        current=('current_total', 'median'),
        books=('book', 'nunique'),
        quotes=('quotes', 'sum')).reset_index()
    game_move['move'] = game_move['current'] - game_move['opened']

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)
    game_move.to_csv(output_path.replace('.csv', '_by_game.csv'), index=False)
    print(f'✓ Wrote {len(out)} book series and {len(game_move)} game summaries')
    moved = game_move[game_move['move'].abs() > 0]
    if len(moved):
        print(moved.sort_values('move', key=abs, ascending=False).to_string(index=False))
    else:
        print('  No movement yet - a single capture cannot show a move.')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Snapshot NFL totals lines to history')
    ap.add_argument('--lines', help='Current lines CSV to append')
    ap.add_argument('--store', default=STORE, help='Parquet history store')
    ap.add_argument('--movement', action='store_true', help='Recompute line movement from history')
    ap.add_argument('--movement-out', default='data/nfl/lines/line_movement.csv')
    args = ap.parse_args()

    if args.lines:
        snapshot(args.lines, args.store)
    if args.movement:
        movement(args.store, args.movement_out)

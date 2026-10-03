"""Track how prices moved after a published pick, using each sport's own pregame snapshots.

Each scheduled sport refresh records, for every pick on a recent published card, the
latest observation before first pitch/puck drop:
  * the same book's line and price for that outcome, and
  * the other-book fair probability at the pick's exact line (conditional on non-push).
The last observation before the start is kept as the "last pregame" quote. Refreshes run
a few times a day, so this is the latest snapshot we hold, not necessarily the true close.
A changed line is a different contract: it is reported as a line move, never as a
same-line probability change.

  python scripts/line_movement.py --sport mlb
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT/'docs/briefing/cards'
WINDOW_DAYS = 4


def stamp(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def iso(value):
    return value.isoformat().replace('+00:00', 'Z')


def decimal(price):
    return 1+price/100 if price > 0 else 1+100/abs(price)


def outcome_key(row):
    return (row.get('event_id'), row.get('market'), row.get('player') or '', row.get('side'))


def picks(sport, now, cards=CARDS):
    """Published card rows for one sport from recent editions, one entry per edition and offer."""
    found = {}
    for path in sorted(cards.glob('*.json')):
        try:
            card = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        published = stamp(card.get('published_at'))
        if not published or now-published > timedelta(days=WINDOW_DAYS):
            continue
        for i, row in enumerate(card.get('rows', [])):
            if str(row.get('sport', '')).upper() != sport.upper() or not isinstance(row.get('price'), (int, float)):
                continue
            pick_id = f"{card.get('edition_id', path.stem)}:{row.get('offer_id') or i}"
            found[pick_id] = dict(id=pick_id, edition_id=card.get('edition_id'), published_at=card.get('published_at'),
                sport=sport.upper(), event_id=row.get('event_id'), game=row.get('game'), commence_time=row.get('commence_time'),
                market=row.get('market'), market_label=row.get('market_label'), player=row.get('player') or '',
                side=row.get('side'), line=row.get('line'), book=row.get('book'), book_label=row.get('book_label'),
                price=row['price'], book_probability=1/decimal(row['price']), quoted_at=row.get('quoted_at'))
    return found


def observe(pick, rows):
    """The pick's book at any line, and the other-book fair probability at the pick's line."""
    same = [r for r in rows if outcome_key(r) == outcome_key(pick)]
    own = next((r for r in same if r.get('book') == pick['book']), None)
    others = [r['fair_probability'] for r in same if r.get('line') == pick['line'] and r.get('book') != pick['book']
              and isinstance(r.get('fair_probability'), (int, float))]
    observation = dict(book_line=own.get('line') if own else None, book_price=own.get('price') if own else None,
        book_quoted_at=own.get('quoted_at') if own else None,
        other_fair=sum(others)/len(others) if others else None, other_books=len(others))
    if pick['line'] is not None and observation['book_line'] is not None and observation['book_line'] != pick['line']:
        move = observation['book_line']-pick['line']
        # Holding a lower Over (or a higher Under) than the market now offers is favorable.
        favorable = move > 0 if pick['side'] == 'Over' else move < 0 if pick['side'] == 'Under' else None
        observation.update(line_move=move, line_move_favorable=favorable)
    if observation['other_fair'] is not None:
        observation.update(probability_move=observation['other_fair']-pick['book_probability'],
                           price_value=decimal(pick['price'])*observation['other_fair']-1)
    return observation


def update(ledger, sport, feed, now, cards=CARDS):
    entries = ledger.setdefault('entries', {})
    checked = stamp(feed.get('checked_at'))
    rows = feed.get('rows', []) if feed.get('status') == 'ready' else []
    for pick_id, pick in picks(sport, now, cards).items():
        start = stamp(pick['commence_time'])
        if pick_id not in entries and (not start or start <= now):
            continue  # Picks first seen after the start can never get a pregame observation.
        entry = entries.setdefault(pick_id, dict(pick, observations=0))
        # Only snapshots ingested after the card was published (the card's own snapshot is
        # older) and before the start count.
        published = stamp(pick['published_at'])
        if (not checked or not start or not published or not published < checked < start
            or entry.get('last_pregame_at') and stamp(entry['last_pregame_at']) >= checked):
            continue
        observation = observe(pick, rows)
        if observation['book_line'] is None and observation['other_fair'] is None:
            continue
        entry.update(last_pregame=observation, last_pregame_at=iso(checked), observations=entry['observations']+1)
    cutoff = now-timedelta(days=30)
    ledger['entries'] = {k: v for k, v in entries.items() if (stamp(v.get('commence_time')) or now) >= cutoff}
    ledger.update(sport=sport.upper(), updated_at=iso(now), summary=summary(ledger['entries'].values()),
        basis='Latest scheduled pregame snapshot after publication; not necessarily the closing line. '
              'Probability move = other-book fair probability at the same line minus the published price’s break-even rate.')
    return ledger


def summary(entries):
    entries = list(entries)
    moved = [e['last_pregame'] for e in entries if e.get('last_pregame')]
    same = [o['probability_move'] for o in moved if o.get('probability_move') is not None]
    lines = [o for o in moved if o.get('line_move_favorable') is not None]
    return dict(picks=len(entries), observed=len(moved),
        same_line=len(same), same_line_beat=sum(m > 0 for m in same),
        average_probability_move=sum(same)/len(same) if same else None,
        line_moves=len(lines), line_moves_favorable=sum(o['line_move_favorable'] for o in lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sport', required=True, choices=['mlb', 'nhl'])
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    feed_path = ROOT/f'docs/{args.sport}/data/latest.json'
    out = ROOT/f'docs/{args.sport}/data/line-movement.json'
    feed = json.loads(feed_path.read_text()) if feed_path.exists() else {}
    ledger = json.loads(out.read_text()) if out.exists() else {}
    ledger = update(ledger, args.sport, feed, now)
    out.write_text(json.dumps(ledger, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    s = ledger['summary']
    print(f"{args.sport.upper()} line movement: {s['observed']} of {s['picks']} published picks observed pregame; "
          f"{s['same_line_beat']}/{s['same_line']} same-line moves in our favor")


if __name__ == '__main__':
    main()

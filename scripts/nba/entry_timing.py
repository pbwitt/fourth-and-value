#!/usr/bin/env python3
"""When to bet NBA props: do earlier prices beat the final pregame price?

The owner asked whether NBA bets should go in the day before. This measures it from the committed
refresh snapshots (artifacts/nba/runs) without new requests. For each game that tipped, every
earlier pregame snapshot is an entry point, bucketed by when it was taken:

* day_before: an Eastern date before the game's (books rarely post NBA props this early);
* game_morning: the game's date, before noon Eastern (the 6:45 a.m. refresh the card uses).

At each entry point, the offers that pass the card's rule (10% model weight, +1% expected value
after blending, two books posting the line; props only) are compared with the last snapshot saved
before tipoff, for the same player, market, side and line:

* closing value: the close's no-vig chance for that side times the entry's decimal price, minus 1.
  Positive means the early price beat where the market settled (closing-line value, measured
  against the last saved snapshot, which is hours before tipoff, not the true close);
* movement: change in the no-vig chance for the bet's side, in probability points;
* the same bet at the best price available at the close, so the early price's advantage is direct;
* when the line itself moved, whether it moved toward the bet (a better number held) or away;
* results at the entry price and at the close price, where ESPN finals are saved.

Intervals are 95% bootstrap intervals over games. Nothing here selects or ranks picks.

Output: artifacts/nba/evaluation/entry-timing.json (summaries) and ENTRY_TIMING.md.
    python scripts/nba/entry_timing.py --offline
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_market_results as mr  # noqa: E402
from display_time import eastern_label  # noqa: E402
from nba import grade_shadow as g  # noqa: E402

BUCKETS = ('day_before', 'game_morning')
LABELS = {'day_before': 'Day before', 'game_morning': 'Game morning (before noon ET)'}
NOON = 12


def snapshots(paths, version):
    """event id -> game details and every pregame snapshot (time, units) for the current model version."""
    events, seen = {}, set()
    for path in sorted(paths):
        with gzip.open(path, 'rt') as f:
            run = json.load(f)
        if not run.get('checked_at') or run['checked_at'] in seen:
            continue
        seen.add(run['checked_at'])
        at = g.parse(run['checked_at'])
        by_event = defaultdict(list)
        for row in run.get('rows', []):
            if (row.get('price') is None or not row.get('event_id') or row.get('model_probability') is None
                    or row.get('model_version') != version):
                continue
            start = g.parse(row['commence_time'])
            if start <= at or (row.get('quoted_at') and g.parse(row['quoted_at']) >= start):
                continue
            by_event[row['event_id']].append(row)
        for event_id, rows in by_event.items():
            entry = events.setdefault(event_id, dict(event_id=event_id, snapshots=[]))
            entry.update(commence_time=rows[0]['commence_time'], home_team=rows[0]['home_team'],
                         away_team=rows[0]['away_team'], game=rows[0].get('game'))
            entry['snapshots'].append((at, g.snapshot_units(rows)))
    for entry in events.values():
        entry['snapshots'].sort(key=lambda s: s[0])
    return events


def bucket(at, start):
    a, s = at.astimezone(g.ET), start.astimezone(g.ET)
    if a.date() < s.date():
        return 'day_before'
    if a.date() == s.date() and a.hour < NOON:
        return 'game_morning'
    return None


def entry_points(event):
    """The latest day-before snapshot and the first game-morning snapshot, each before the close."""
    start = g.parse(event['commence_time'])
    points = {}
    for at, units in event['snapshots'][:-1]:
        b = bucket(at, start)
        if b == 'day_before':
            points[b] = (at, units)
        elif b == 'game_morning' and b not in points:
            points[b] = (at, units)
    return points


def card_offer(unit):
    """The best offer passing the card's blend rule: (expected value, offer) or None."""
    best = None
    for offer in unit['offers']:
        if offer['market'] is None or offer['books'] < g.MIN_LINE_BOOKS:
            continue
        final = g.BLEND_WEIGHT * offer['model'] + (1 - g.BLEND_WEIGHT) * offer['market']
        ev = final * g.decimal(offer['price']) - 1
        if ev >= g.EV_FLOOR and (best is None or ev > best[0]):
            best = (ev, offer)
    return best


def side_chance(unit, ref):
    """The unit's no-vig consensus for the reference side (ref=True) or the other side."""
    return None if unit is None or unit['market_p'] is None else unit['market_p'] if ref else 1 - unit['market_p']


def best_close_price(unit, ref):
    prices = [o['price'] for o in (unit or {}).get('offers', []) if o['ref'] == ref]
    return max(prices, key=g.decimal) if prices else None


def line_move(key, ref, close):
    """'toward' if the close's main line for this player and market makes the entry line the better number."""
    main = [k for k in g.main_lines(close) if k[:2] == key[:2]]
    if not main or main[0][2] is None or main[0][2] == key[2]:
        return None
    later = main[0][2]
    # An Over at 24.5 is the better number when the close line is 25.5; an Under, when it is 23.5.
    return 'toward' if (later > key[2]) == ref else 'away'


def evaluate(events, games, players, resolved):
    bets = []
    for eid, event in sorted(events.items(), key=lambda kv: (kv[1]['commence_time'], kv[0])):
        if len(event['snapshots']) < 2:
            continue
        close_at, close = event['snapshots'][-1]
        game = games.get(resolved.get(eid, {}).get('nba_game_id'))
        final = bool(game and game.get('final'))
        for name, (at, units) in entry_points(event).items():
            for key, unit in units.items():
                if key[0] in g.GAME_MARKETS:
                    continue
                chosen = card_offer(unit)
                if chosen is None:
                    continue
                ev, offer = chosen
                ref, price = offer['ref'], offer['price']
                entry_chance = side_chance(unit, ref)
                later = close.get(key)
                close_chance = side_chance(later, ref)
                close_price = best_close_price(later, ref)
                bet = dict(event_id=eid, bucket=name, market=key[0], player=unit['player'], side='ref' if ref else 'other',
                           line=key[2], entry_at=at.isoformat(), close_at=close_at.isoformat(),
                           hours_before_close=round((close_at - at).total_seconds() / 3600, 2),
                           entry_price=price, blended_ev=round(ev, 5), entry_chance=entry_chance,
                           close_chance=close_chance, close_price=close_price, line_move=None, result=None)
                if close_chance is None:
                    bet['line_move'] = line_move(key, ref, close)
                else:
                    bet['closing_value'] = close_chance * g.decimal(price) - 1
                    bet['movement'] = close_chance - entry_chance if entry_chance is not None else None
                    if close_price is not None:
                        bet['close_price_value'] = close_chance * g.decimal(close_price) - 1
                if final:
                    outcome, status = g.settle(unit, event, game, players)
                    if status in ('graded', 'push'):
                        won = None if outcome is None else outcome == (1 if ref else 0)
                        bet['result'] = 'push' if won is None else 'win' if won else 'loss'
                        bet['profit'] = 0.0 if won is None else g.decimal(price) - 1 if won else -1.0
                        if close_price is not None:
                            bet['close_profit'] = 0.0 if won is None else g.decimal(close_price) - 1 if won else -1.0
                for k in ('entry_chance', 'close_chance', 'closing_value', 'movement', 'close_price_value', 'profit', 'close_profit'):
                    if bet.get(k) is not None:
                        bet[k] = round(bet[k], 5)
                bets.append(bet)
    return bets


def summarize(bets, rng):
    if not bets:
        return dict(bets=0, games=0)
    games = lambda rows: [b['event_id'] for b in rows]
    same_line = [b for b in bets if b.get('closing_value') is not None]
    paired = [b for b in same_line if b.get('close_price_value') is not None]
    moved = [b for b in bets if b['line_move']]
    graded = [b for b in bets if b.get('profit') is not None]
    both = [b for b in graded if b.get('close_profit') is not None]
    out = dict(
        bets=len(bets), games=len(set(games(bets))),
        median_hours_before_close=round(float(np.median([b['hours_before_close'] for b in bets])), 1),
        same_line_at_close=len(same_line),
        closing_value=g.interval([b['closing_value'] for b in same_line], games(same_line), rng),
        beat_close_share=round(sum(b['closing_value'] > 0 for b in same_line) / len(same_line), 4) if same_line else None,
        movement_points=g.interval([100 * b['movement'] for b in same_line if b.get('movement') is not None],
                                   [b['event_id'] for b in same_line if b.get('movement') is not None], rng),
        early_price_advantage=g.interval([b['closing_value'] - b['close_price_value'] for b in paired], games(paired), rng),
        line_moved=dict(toward=sum(b['line_move'] == 'toward' for b in moved), away=sum(b['line_move'] == 'away' for b in moved),
                        not_comparable=len(bets) - len(same_line) - len(moved)),
        results=dict(graded=len(graded), wins=sum(b['result'] == 'win' for b in graded),
                     losses=sum(b['result'] == 'loss' for b in graded), pushes=sum(b['result'] == 'push' for b in graded),
                     units_at_entry=round(sum(b['profit'] for b in graded), 3),
                     roi_at_entry=g.interval([b['profit'] for b in graded], games(graded), rng),
                     same_bets_units_at_close=round(sum(b['close_profit'] for b in both), 3),
                     same_bets_units_at_entry=round(sum(b['profit'] for b in both), 3), same_bets=len(both)))
    return out


def build(season=2026, runs_root=g.RUNS, root=mr.ROOT, fetch=None, offline=True, now=None, version=None):
    now = now or datetime.now(timezone.utc)
    version = version or json.loads(g.MODEL_PATH.read_text())['version']
    first, after = (datetime(year, 10, 1, tzinfo=g.ET) for year in (season, season + 1))
    events = {eid: e for eid, e in snapshots(Path(runs_root).glob('*.json.gz'), version).items()
              if first <= g.parse(e['commence_time']) < after and g.parse(e['commence_time']) <= now}
    quotes = {eid: (e['snapshots'][-1][0], [e]) for eid, e in events.items()}
    games, players, resolved = mr.nba_results(quotes, root, fetch, offline) if quotes else ({}, {}, {})
    bets = evaluate(events, games, players, resolved)
    rng = np.random.default_rng(g.SEED)
    days = defaultdict(lambda: dict(games=set(), day_before=set()))
    for eid, e in events.items():
        day = g.parse(e['commence_time']).astimezone(g.ET).date().isoformat()
        days[day]['games'].add(eid)
        if 'day_before' in entry_points(e):
            days[day]['day_before'].add(eid)
    return dict(
        schema=1, generated_at=now.isoformat(), model_version=version, status='research_only',
        coverage=dict(games_tipped=len(events), games_with_day_before_props=sum(len(d['day_before']) for d in days.values()),
                      games_with_game_morning_props=sum('game_morning' in entry_points(e) for e in events.values())),
        buckets={b: dict(label=LABELS[b], **summarize([x for x in bets if x['bucket'] == b], rng)) for b in BUCKETS},
        by_market={b: {m: summarize([x for x in bets if x['bucket'] == b and x['market'] == m], rng)
                       for m in sorted({x['market'] for x in bets if x['bucket'] == b})} for b in BUCKETS},
        criteria=dict(
            entry='Latest day-before snapshot and first game-morning snapshot (before noon ET) for each game.',
            selection=f'Offers passing the card rule: {g.BLEND_WEIGHT:.0%} model weight, {g.EV_FLOOR:.0%} expected value '
                      f'after blending, {g.MIN_LINE_BOOKS} books posting the line; player props only.',
            close='Last snapshot saved before tipoff (6:45 a.m. or 5 p.m. ET), not the true closing line.',
            closing_value="Close's no-vig chance for the bet's side x entry decimal price - 1.",
            early_price_advantage='Closing value at the entry price minus closing value at the best close price for the same bet.',
            intervals=f'95% bootstrap intervals over games, from {g.MIN_CI_GAMES} games.'),
        bets=bets)


def fmt_ci(block, scale=1, digits=1, suffix=''):
    if not block or block.get('mean') is None:
        return '—'
    m = block['mean'] * scale
    text = f'{m:+.{digits}f}{suffix}'
    if block.get('ci95'):
        lo, hi = (x * scale for x in block['ci95'])
        text += f' ({lo:+.{digits}f} to {hi:+.{digits}f})'
    return text


def markdown(payload):
    c = payload['coverage']
    lines = ['# NBA entry timing: earlier prices against the final pregame price', '',
             f"Model `{payload['model_version']}`, generated {eastern_label(payload['generated_at'])}. "
             f"{c['games_tipped']} games tipped; {c['games_with_day_before_props']} had props in a day-before snapshot "
             f"and {c['games_with_game_morning_props']} in a game-morning snapshot.", '',
             'Bets are the offers that pass the card\'s rule at that time. Closing value compares the entry price with '
             'the no-vig chance in the last snapshot before tipoff; positive means the early price beat where the market '
             'settled. Early-price advantage is the same comparison against the best price still available at the close.', '',
             '| Entry | Bets | Games | Same line at close | Closing value | Beat close | Moved toward bet (pts) | '
             'Early-price advantage | Line moved toward / away | Results at entry |', '|---|---|---|---|---|---|---|---|---|---|']
    for b in BUCKETS:
        s = payload['buckets'][b]
        if not s.get('bets'):
            lines.append(f"| {s['label']} | 0 | 0 | — | — | — | — | — | — | — |")
            continue
        r = s['results']
        res = f"{r['wins']}-{r['losses']}-{r['pushes']}, {r['units_at_entry']:+.2f}u" if r['graded'] else '—'
        beat = f"{s['beat_close_share']:.0%}" if s['beat_close_share'] is not None else '—'
        lines.append(f"| {s['label']} | {s['bets']} | {s['games']} | {s['same_line_at_close']} | "
                     f"{fmt_ci(s['closing_value'], 100, 1, '%')} | {beat} | {fmt_ci(s['movement_points'], 1, 2)} | "
                     f"{fmt_ci(s['early_price_advantage'], 100, 1, '%')} | {s['line_moved']['toward']} / {s['line_moved']['away']} | {res} |")
    lines += ['', 'Caveats: the "close" is the last saved snapshot (6:45 a.m. or 5 p.m. ET), hours before tipoff, so this '
              'understates late moves either way. Day-before entries exist only when books post next-day props by the '
              '5 p.m. refresh, which since opening week has been rare. Bets in one game are correlated; intervals resample '
              'whole games and appear from 10 games. Research only: nothing here selects, ranks or times a pick.', '']
    return '\n'.join(lines)


def write(payload, out=g.OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # The per-bet list is left out of the committed file: it grows all season, and the committed runs and
    # results reproduce it exactly (build() returns it).
    saved = {k: v for k, v in payload.items() if k != 'bets'}
    saved['bets_omitted'] = 'Recompute with scripts/nba/entry_timing.py from artifacts/nba/runs and artifacts/nba/results.'
    (out / 'entry-timing.json').write_text(json.dumps(saved, indent=1, allow_nan=False) + '\n')
    (out / 'ENTRY_TIMING.md').write_text(markdown(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--offline', action='store_true', help='use saved ESPN results only (artifacts/nba/results)')
    parser.add_argument('--out', default=str(g.OUT))
    args = parser.parse_args()
    payload = build(args.season, offline=args.offline)
    write(payload, args.out)
    for b in BUCKETS:
        s = payload['buckets'][b]
        print(f"NBA entry timing, {b}: {s.get('bets', 0)} bets in {s.get('games', 0)} games; closing value "
              f"{fmt_ci(s.get('closing_value'), 100, 1, '%')}")


if __name__ == '__main__':
    main()

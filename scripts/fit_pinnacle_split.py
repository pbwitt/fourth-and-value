"""Score Pinnacle against the US-book median on graded lines (Step 2c follow-up).

Joins Pinnacle's no-vig chances (the optional snapshots in artifacts/pinnacle/snapshots/
and, from 2026-10-21, the core-refresh reference in artifacts/pinnacle/reference/) to the
graded lines in the weekly recap data, then reports log loss for the US median, for
Pinnacle, and for the best mix of the two, by sport and market. Lower is better.

No lookahead: each graded line uses the latest Pinnacle capture taken at or before the
moment our own market probability was recorded (the board row's decision time). NFL prop
tickets carry no decision time, so they use the latest capture before kickoff and are
marked approximate. The 90% range of the Pinnacle-minus-US difference resamples whole
games. Research only: it never changes the market term or a pick
(docs/model-improvement-plan.md, Step 2c).
"""
import argparse
import collections
import glob
import json
import math
import random
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NFL_MARKETS = {'receptions': 'player_receptions', 'recv_yds': 'player_reception_yds',
               'rush_yds': 'player_rush_yds', 'rush_attempts': 'player_rush_attempts',
               'pass_yds': 'player_pass_yds', 'pass_tds': 'player_pass_tds',
               'pass_attempts': 'player_pass_attempts', 'pass_completions': 'player_pass_completions'}


def stamp(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def clip(p):
    return min(1 - 1e-4, max(1e-4, p))


def logit(p):
    p = clip(p)
    return math.log(p / (1 - p))


def implied(price):
    price = float(price)
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def norm(text):
    return str(text or '').strip().casefold()


def line_key(market, line):
    return abs(line) if market == 'spreads' and line is not None else line


def pinnacle_quotes(root, kickoffs=None):
    """Every Pinnacle quote as (sport, event, market, player, side, line, price, captured);
    fills `kickoffs` with each event's start time."""
    out, kickoffs = [], kickoffs if kickoffs is not None else {}
    for path in sorted(glob.glob(str(root / 'artifacts/pinnacle/snapshots/*.json'))):
        data = json.loads(Path(path).read_text())
        fields = data.get('fields', [])
        for row in data.get('quotes', []):
            q = dict(zip(fields, row))
            kickoffs[q['event_id']] = stamp(q['commence_time'])
            out.append((q['sport'].lower(), q['event_id'], q['market'], norm(q['player']), norm(q['side']),
                        q['line'], q['price'], data['generated_at']))
    for path in sorted(glob.glob(str(root / 'artifacts/pinnacle/reference/*/*.json'))):
        data = json.loads(Path(path).read_text())
        fields = data.get('fields', [])
        for row in data.get('quotes', []):
            q = dict(zip(fields, row))
            kickoffs[q['event_id']] = stamp(q['commence_time'])
            out.append((data['sport'], q['event_id'], q['market'], norm(q['player']), norm(q['side']),
                        q['line'], q['price'], q['captured_at']))
    return out


def no_vig(quotes):
    """{(sport, event, market, player, side, line): [(captured, chance)]}, two-way pairs only."""
    pairs = collections.defaultdict(list)
    for sport, event, market, player, side, line, price, captured in quotes:
        pairs[(sport, event, market, player, line_key(market, line), captured)].append((side, line, price))
    chances = collections.defaultdict(list)
    for (sport, event, market, player, _, captured), sides in pairs.items():
        if len(sides) != 2 or sides[0][0] == sides[1][0]:
            continue
        raw = [implied(p) for _, _, p in sides]
        for (side, line, _), r in zip(sides, raw):
            chances[(sport, event, market, player, side, line)].append((stamp(captured), r / sum(raw)))
    for v in chances.values():
        v.sort()
    return chances


def latest(captures, cutoff):
    before = [p for t, p in captures if t <= cutoff]
    return before[-1] if before else None


def board_lines(root, chances):
    """NHL, MLB and NBA recap boards: one side per line, US median at our decision time."""
    lines = []
    for path in sorted(glob.glob(str(root / 'docs/blog/*-recap-*.json'))):
        sport = Path(path).name.split('-')[0]
        groups = collections.defaultdict(list)
        for r in json.loads(Path(path).read_text()).get('board', []):
            when = r.get('decision_at') or r.get('comparison_evidence_at') or r.get('evidence_at')
            if r.get('outcome') not in ('win', 'loss') or r.get('market_probability') is None or not when:
                continue
            groups[(r['event_id'], r['market'], norm(r.get('player')), norm(r['side']), r.get('line'))].append((stamp(when), r))
        seen = set()
        for key in sorted(groups, key=str):
            event, market, player, side, line = key
            base = (event, market, player, line_key(market, line))
            if base in seen:
                continue
            first = min(t for t, _ in groups[key])
            rows = [r for t, r in groups[key] if t == first]
            pin = latest(chances.get((sport, event, market, player, side, line), []), first)
            if pin is None:
                continue
            seen.add(base)
            lines.append(dict(sport=sport.upper(), market=market, game=event, approx=False,
                              us=statistics.median(r['market_probability'] for r in rows), pin=pin,
                              y=rows[0]['outcome'] == 'win'))
    return lines


def week_windows(files):
    """{week: (first day, last day)} in UTC dates, Thursday to Tuesday, anchored on any
    review file that records its week's dates; None when no file does."""
    anchor = None
    for week, data in files:
        clv = (data.get('summary') or {}).get('clv') or {}
        if clv.get('start'):
            anchor = (week, datetime.fromisoformat(clv['start']).date())
    if not anchor:
        return None
    week0, start0 = anchor
    return {w: (start0 + timedelta(days=7 * (w - week0) - 1), start0 + timedelta(days=7 * (w - week0) + 6))
            for w, _ in files}


def nfl_lines(root, chances, kickoffs):
    """NFL prop tickets: no decision time, so the latest Pinnacle capture before kickoff,
    matched to the same week's game by player, market, side and line."""
    by_player = collections.defaultdict(list)
    for (sport, event, market, player, side, line), caps in chances.items():
        if sport == 'nfl' and event in kickoffs:
            by_player[(market, player, side, line)].append((event, caps))
    files = []
    for path in sorted(glob.glob(str(root / 'docs/blog/week-*-review-data.json'))):
        found = re.search(r'week-(\d+)-', Path(path).name)
        if found:
            files.append((int(found.group(1)), json.loads(Path(path).read_text())))
    windows = week_windows(files)
    lines, seen = [], set()
    for week, data in files if windows else []:
        first, last = windows[week]
        for t in data.get('tickets', []):
            if t.get('consensus_prob') is None or t.get('outcome') not in ('win', 'loss'):
                continue
            market = NFL_MARKETS.get(t['market_std'], t['market_std'])
            base = (t['stat_game'], norm(t['player']), market, t['point'])
            matches = [(e, caps) for e, caps in by_player.get((market, norm(t['player']), norm(t['name']), t['point']), [])
                       if first <= kickoffs[e].date() <= last]
            if base in seen or len(matches) != 1:
                continue
            event, caps = matches[0]
            pin = latest(caps, kickoffs[event])
            if pin is None:
                continue
            seen.add(base)
            lines.append(dict(sport='NFL', market=market, game=t['stat_game'], approx=True,
                              us=t['consensus_prob'], pin=pin, y=t['outcome'] == 'win'))
    return lines


def loss(lines, w):
    """Mean log loss of sigmoid(w*logit(pinnacle) + (1-w)*logit(US median))."""
    total = 0.0
    for r in lines:
        p = clip(1 / (1 + math.exp(-(w * logit(r['pin']) + (1 - w) * logit(r['us'])))))
        total -= math.log(p if r['y'] else 1 - p)
    return total / len(lines)


def best(lines):
    return min((loss(lines, w / 10), w / 10) for w in range(11))[1]


def diff_range(lines, draws=400, seed=7):
    games = collections.defaultdict(list)
    for r in lines:
        games[r['game']].append(r)
    keys, rng, diffs = sorted(games), random.Random(seed), []
    for _ in range(draws):
        sample = [r for g in rng.choices(keys, k=len(keys)) for r in games[g]]
        diffs.append(loss(sample, 1) - loss(sample, 0))
    diffs.sort()
    return diffs[int(.05 * draws)], diffs[int(.95 * draws)]


def row(name, lines, minimum):
    games = len({r['game'] for r in lines})
    if len(lines) < minimum:
        return f'| {name} | {len(lines)} | {games} | | | | | too few lines |'
    lo, hi = diff_range(lines)
    verdict = 'Pinnacle better' if hi < 0 else 'US median better' if lo > 0 else 'no clear difference'
    return (f'| {name} | {len(lines)} | {games} | {loss(lines, 0):.4f} | {loss(lines, 1):.4f} | '
            f'{loss(lines, .5):.4f} | {best(lines):.1f} | {lo:+.4f} to {hi:+.4f}: {verdict} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--minimum', type=int, default=30, help='Fewest lines to score a group')
    args = parser.parse_args()
    kickoffs = {}
    chances = no_vig(pinnacle_quotes(args.root, kickoffs))
    lines = board_lines(args.root, chances) + nfl_lines(args.root, chances, kickoffs)
    print(f'Pinnacle two-way chances: {len(chances)}; graded lines joined: {len(lines)}\n')
    print('| Lines | Count | Games | Log loss, US median | Log loss, Pinnacle | Log loss, 50/50 | Best Pinnacle weight | Pinnacle minus US, 90% range |')
    print('|---|---|---|---|---|---|---|---|')
    print(row('All', lines, args.minimum))
    for sport in sorted({r['sport'] for r in lines}):
        group = [r for r in lines if r['sport'] == sport]
        print(row(sport + (' (timing approximate)' if group[0]['approx'] else ''), group, args.minimum))
        for kind, keep in (('game lines', lambda m: m in ('h2h', 'spreads', 'totals')),
                           ('props', lambda m: m not in ('h2h', 'spreads', 'totals'))):
            print(row(f'  {sport} {kind}', [r for r in group if keep(r['market'])], args.minimum))
        for market in sorted({r['market'] for r in group}):
            print(row('    ' + market, [r for r in group if r['market'] == market], args.minimum))


if __name__ == '__main__':
    main()

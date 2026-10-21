#!/usr/bin/env python3
"""Grade the NBA v1 shadow forecasts against prices and results as the season is played.

Phase 2's open item in NBA_LAUNCH_PLAN.md: the evidence Top Picks needs before NBA joins the card.
Every input is committed: refresh snapshots in artifacts/nba/runs (each row carries the model's
forecast and the books' paired no-vig consensus at that moment) and ESPN finals in
artifacts/nba/results, read through the same resolver Market Analytics uses. Nothing is fitted here.

For each final game the last snapshot saved before tipoff is scored. One forecast per market,
player and line is compared with the books' consensus at that line, both conditional on no push:

* log loss and Brier score of the model, the market and model-market blends;
* whether the market moved toward the model between the last two pregame snapshots. The
  snapshots are hours before tipoff, so this is a closing-line-value proxy, not CLV;
* the offers that would have passed the NHL card's price rule (10% weight on the model, 1%
  expected value after blending, two books posting the line), settled at the offered price.

Main lines (the line most books post for that player and market) are the headline sample;
alternate lines are reported separately. Intervals are 95% percentile intervals from a bootstrap
over games, because one game's props are correlated. Only the production model version is graded;
forecasts from other versions are counted and left out, so a retrain starts a fresh record.

Output: artifacts/nba/evaluation/prospective.json and PROSPECTIVE.md.
    python scripts/nba/grade_shadow.py            # asks ESPN only for games not saved yet
    python scripts/nba/grade_shadow.py --offline  # saved results only (the daily workflow)
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_market_results as mr  # noqa: E402
from display_time import eastern_label  # noqa: E402

ET = ZoneInfo('America/New_York')
MODEL_PATH = ROOT / 'artifacts/nba/model/nba-v1.json'
RUNS = ROOT / 'artifacts/nba/runs'
OUT = ROOT / 'artifacts/nba/evaluation'

GAME_MARKETS = ('h2h', 'spreads', 'totals')
MARKET_LABELS = {m['key']: m['label'] for m in mr.NBA_MARKETS}
BLEND_WEIGHT = 0.10        # The NHL card's market-blend-3 weight on the model.
BLEND_CURVE = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
EV_FLOOR = 0.01            # The NHL card's floor: 1% expected value after blending.
MIN_LINE_BOOKS = 2         # The NHL card's rule: two books, counting the offered one, post the line.
MOVE_GAP = 0.03            # Movement is measured where the model and market disagreed by 3+ points.
MOVE_WINDOW = timedelta(hours=36)
MIN_GAMES = 150            # Proposed sample before a market's evidence is read as decisive.
BOOTSTRAP = 1000
MIN_CI_GAMES = 10          # A bootstrap over a handful of games understates the uncertainty; no interval below this.
SEED = 20261020
EPS = 1e-4


def parse(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def decimal(price):
    return 1 + price / 100 if price > 0 else 1 + 100 / -price


def conditional(win, push):
    """The model's win probability given no push, the basis of the books' paired no-vig consensus."""
    push = push or 0.0
    if win is None or not 0 <= push < 1:
        return None
    return min(max(win / (1 - push), 0.0), 1.0)


def reference_side(row):
    """Each unit is scored from one side: Over for totals and props, the home team for spreads and moneylines."""
    return row['home_team'] if row['market'] in ('h2h', 'spreads') else 'Over'


def unit_key(row):
    """(market, player, line) with spreads on the home team's handicap; game lines have no player."""
    market, line = row['market'], row.get('line')
    who = '' if market in GAME_MARKETS else str(row.get('espn_player_id') or 'name:' + mr.fold(row.get('player') or ''))
    if market == 'h2h':
        line = None
    elif market == 'spreads' and row['side'] != row['home_team']:
        line = -line
    return market, who, None if line is None else float(line)


def snapshot_units(rows):
    """One unit per market, player and line from one snapshot of one game, with every book's offer."""
    units = {}
    for row in rows:
        p = conditional(row.get('model_probability'), row.get('model_push_probability'))
        if p is None or (row['market'] != 'h2h' and row.get('line') is None):
            continue
        key = unit_key(row)
        ref = row['side'] == reference_side(row)
        unit = units.setdefault(key, dict(market=key[0], player=row.get('player') or '', espn_player_id=row.get('espn_player_id'),
                                          line=key[2], model=None, market_p=None, books=0, offers=[]))
        if unit['model'] is None or ref:
            unit['model'] = p if ref else 1 - p
        c = row.get('consensus_probability')
        if c is not None and 0 < c < 1 and (unit['market_p'] is None or ref):
            unit['market_p'] = c if ref else 1 - c
        books = int(row.get('paired_books') or 0)
        unit['books'] = max(unit['books'], books)
        unit['offers'].append(dict(book=row.get('book'), ref=ref, price=float(row['price']), model=p,
                                   market=c if c is not None and 0 < c < 1 else None, books=books))
    return units


def event_snapshots(paths, version):
    """event id -> game details and its last two pregame snapshots, read one archive at a time.

    Rows follow Market Analytics' pregame rule (priced, before tipoff, quoted before tipoff) and must
    carry a forecast from ``version``; other versions are counted, not graded."""
    events, other_versions, seen = {}, defaultdict(int), set()
    for path in sorted(paths):
        with gzip.open(path, 'rt') as f:
            run = json.load(f)
        if not run.get('checked_at') or run['checked_at'] in seen:
            continue
        seen.add(run['checked_at'])
        at = parse(run['checked_at'])
        by_event = defaultdict(list)
        for row in run.get('rows', []):
            if row.get('price') is None or not row.get('event_id') or row.get('model_probability') is None:
                continue
            start = parse(row['commence_time'])
            if start <= at or (row.get('quoted_at') and parse(row['quoted_at']) >= start):
                continue
            if row.get('model_version') != version:
                other_versions[row.get('model_version') or 'unknown'] += 1
                continue
            by_event[row['event_id']].append(row)
        for event_id, rows in by_event.items():
            entry = events.setdefault(event_id, dict(snapshots=[]))
            entry.update(event_id=event_id, commence_time=rows[0]['commence_time'], home_team=rows[0]['home_team'],
                         away_team=rows[0]['away_team'], game=rows[0].get('game'))
            entry['snapshots'] = sorted(entry['snapshots'] + [(at, snapshot_units(rows))], key=lambda s: s[0])[-2:]
    return events, dict(other_versions)


def settle(unit, event, game, players):
    """1 if the reference side won, 0 if it lost, None for a push or a void; plus the reason."""
    from nba_weekly_results import get_actual
    row = dict(market=unit['market'], player=unit['player'], side=reference_side(dict(event, market=unit['market'])),
               home_team=event['home_team'], away_team=event['away_team'], nba_game_id=game['id'],
               resolution_status='resolved')
    if unit.get('espn_player_id'):
        row['espn_player_id'] = str(unit['espn_player_id'])
    actual, status = get_actual(row, {game['id']: game}, players)
    if actual is None:  # Did not play, an ambiguous name or a missing stat: void, as books void them.
        return None, status
    edge = actual if unit['market'] == 'h2h' else actual + unit['line'] if unit['market'] == 'spreads' else actual - unit['line']
    return (1 if edge > 0 else 0 if edge < 0 else None), ('graded' if edge else 'push')


def main_lines(units):
    """Per player and market, the line the most books pair (ties: nearest the median, then lowest)."""
    groups = defaultdict(list)
    for key, unit in units.items():
        if unit['market_p'] is not None:
            groups[key[:2]].append(key)
    main = set()
    for keys in groups.values():
        lines = [k[2] for k in keys if k[2] is not None]
        median = float(np.median(lines)) if lines else 0.0
        main.add(min(keys, key=lambda k: (-units[k]['books'], abs((k[2] or 0.0) - median), k[2] or 0.0)))
    return main


def week_label(day):
    monday = day - timedelta(days=day.weekday())
    return monday.isoformat(), f'Week of {monday:%b} {monday.day}'


def log_loss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def interval(values, games, rng):
    """Mean and 95% interval of per-record values, resampling whole games."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci95=None)
    ids, index = np.unique(np.asarray(games), return_inverse=True)
    sums, counts = np.bincount(index, weights=values), np.bincount(index).astype(float)
    mean = float(values.mean())
    if len(ids) < MIN_CI_GAMES:
        return dict(mean=round(mean, 5), ci95=None)
    draw = rng.integers(0, len(ids), size=(BOOTSTRAP, len(ids)))
    reps = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return dict(mean=round(mean, 5), ci95=[round(float(x), 5) for x in np.percentile(reps, [2.5, 97.5])])


def score(records, rng):
    """Model versus market on graded units: proper scores, blends and agreement."""
    if not records:
        return dict(units=0, games=0)
    y = np.array([r['outcome'] for r in records], dtype=float)
    pm = np.array([r['model'] for r in records])
    pk = np.array([r['market_p'] for r in records])
    games = [r['event_id'] for r in records]
    blend = lambda w: w * pm + (1 - w) * pk
    market_ll = log_loss(pk, y)
    curve = [dict(weight=w, log_loss=round(float(log_loss(blend(w), y).mean()), 5)) for w in BLEND_CURVE]
    return dict(
        units=len(records), games=len(set(games)), outcome_rate=round(float(y.mean()), 4),
        model_mean=round(float(pm.mean()), 4), market_mean=round(float(pk.mean()), 4),
        mean_abs_disagreement=round(float(np.abs(pm - pk).mean()), 4),
        log_loss=dict(model=round(float(log_loss(pm, y).mean()), 5), market=round(float(market_ll.mean()), 5),
                      blend=round(float(log_loss(blend(BLEND_WEIGHT), y).mean()), 5)),
        brier=dict(model=round(float(((pm - y) ** 2).mean()), 5), market=round(float(((pk - y) ** 2).mean()), 5)),
        model_minus_market=interval(log_loss(pm, y) - market_ll, games, rng),
        blend_minus_market=dict(weight=BLEND_WEIGHT, **interval(log_loss(blend(BLEND_WEIGHT), y) - market_ll, games, rng)),
        blend_curve=curve, best_weight_in_sample=min(curve, key=lambda c: c['log_loss'])['weight'])


def calibration(records):
    bins = defaultdict(list)
    for r in records:
        bins[min(int(r['model'] * 10), 9)].append(r)
    return [dict(bin=f'{b / 10:.1f}-{(b + 1) / 10:.1f}', n=len(rs), model=round(float(np.mean([r['model'] for r in rs])), 4),
                 market=round(float(np.mean([r['market_p'] for r in rs])), 4),
                 observed=round(float(np.mean([r['outcome'] for r in rs])), 4)) for b, rs in sorted(bins.items())]


def movement(moves, rng):
    """How far the consensus moved toward the model's earlier lean, in probability points."""
    if not moves:
        return dict(units=0, games=0)
    values = [m['move'] for m in moves]
    return dict(units=len(moves), games=len({m['event_id'] for m in moves}),
                toward=round(sum(v > 0 for v in values) / len(values), 4),
                away=round(sum(v < 0 for v in values) / len(values), 4),
                points=interval([100 * v for v in values], [m['event_id'] for m in moves], rng))


def pool_summary(bets, rng, split=True):
    if not bets:
        return dict(bets=0)
    profit = [b['profit'] for b in bets]
    summary = dict(bets=len(bets), games=len({b['event_id'] for b in bets}), wins=sum(b['result'] == 'win' for b in bets),
                   losses=sum(b['result'] == 'loss' for b in bets), pushes=sum(b['result'] == 'push' for b in bets),
                   units=round(sum(profit), 3), claimed_ev=round(float(np.mean([b['ev'] for b in bets])), 4),
                   roi=interval(profit, [b['event_id'] for b in bets], rng))
    if split:
        summary['by_source'] = {s: pool_summary([b for b in bets if b['source'] == s], rng, split=False)
                                for s in ('price', 'model')}
    return summary


def readiness(main, moves):
    """A proposed bar, not a switch: the owner decides when a market joins Top Picks."""
    blend = (main or {}).get('blend_minus_market', {})
    move = (moves or {}).get('points', {})
    if (main or {}).get('games', 0) < MIN_GAMES:
        status = 'insufficient_sample'
    elif blend.get('ci95') and blend['ci95'][1] < 0 and move.get('ci95') and move['ci95'][0] > 0:
        status = 'supported'
    else:
        status = 'not_supported'
    return dict(status=status, games=(main or {}).get('games', 0), games_needed=MIN_GAMES)


def card_bet(unit, outcome, event_id):
    """The best offer on either side that passes the NHL card's blend rule, settled at its price."""
    best = None
    for offer in unit['offers']:
        if offer['market'] is None or offer['books'] < MIN_LINE_BOOKS:
            continue
        final = BLEND_WEIGHT * offer['model'] + (1 - BLEND_WEIGHT) * offer['market']
        ev = final * decimal(offer['price']) - 1
        if ev >= EV_FLOOR and (best is None or ev > best[0]):
            best = (ev, offer)
    if best is None:
        return None
    ev, offer = best
    won = None if outcome is None else outcome == (1 if offer['ref'] else 0)
    result = 'push' if won is None else 'win' if won else 'loss'
    # Where the value comes from: one book off the rest of the market clears the floor at the
    # consensus alone (no model needed); otherwise it rests on the model disagreeing with the market.
    source = 'price' if offer['market'] * decimal(offer['price']) - 1 >= EV_FLOOR else 'model'
    return dict(event_id=event_id, market=unit['market'], ev=ev, price=offer['price'], book=offer['book'],
                source=source, result=result, profit=0.0 if won is None else decimal(offer['price']) - 1 if won else -1.0)


def grade(season=2026, runs_root=RUNS, root=mr.ROOT, fetch=None, offline=False, now=None, version=None):
    now = now or datetime.now(timezone.utc)
    version = version or json.loads(MODEL_PATH.read_text())['version']
    first, after = (datetime(year, 10, 1, tzinfo=ET) for year in (season, season + 1))
    events, other_versions = event_snapshots(Path(runs_root).glob('*.json.gz'), version)
    started = {eid: e for eid, e in events.items() if first <= parse(e['commence_time']) < after and parse(e['commence_time']) <= now}
    quotes = {eid: (e['snapshots'][-1][0], [e]) for eid, e in started.items()}
    games, players, resolved = mr.nba_results(quotes, root, fetch, offline)
    rng = np.random.default_rng(SEED)
    records, moves, bets, statuses = [], [], [], defaultdict(int)
    for eid in sorted(started, key=lambda e: (started[e]['commence_time'], e)):
        event = started[eid]
        (late_at, late), early = event['snapshots'][-1], event['snapshots'][:-1]
        main = main_lines(late)
        # Movement needs no result: every game that tipped with two recent pregame snapshots counts.
        if early and late_at - early[0][0] <= MOVE_WINDOW:
            for key in main:
                before, unit = early[0][1].get(key), late[key]
                if before and before['market_p'] is not None and abs(before['model'] - before['market_p']) >= MOVE_GAP:
                    lean = 1 if before['model'] > before['market_p'] else -1
                    moves.append(dict(event_id=eid, market=key[0], move=lean * (unit['market_p'] - before['market_p'])))
        game = games.get(resolved.get(eid, {}).get('nba_game_id'))
        if not game or not game.get('final'):
            statuses['result_pending_or_unmatched'] += 1
            continue
        day = parse(event['commence_time']).astimezone(ET).date()
        for key, unit in late.items():
            outcome, status = settle(unit, event, game, players)
            statuses[status] += 1
            if status not in ('graded', 'push'):
                continue
            bet = card_bet(unit, outcome, eid)
            if bet:
                bets.append(bet)
            if outcome is None or unit['market_p'] is None:
                continue
            records.append(dict(event_id=eid, day=day, market=key[0], main=key in main, model=unit['model'],
                                market_p=unit['market_p'], outcome=outcome))
    return package(records, moves, bets, statuses, started, games, other_versions, version, now, rng)


def package(records, moves, bets, statuses, started, games, other_versions, version, now, rng):
    group = lambda m: 'game_lines' if m in GAME_MARKETS else 'props'
    main = [r for r in records if r['main']]
    markets = sorted({r['market'] for r in records}, key=lambda m: (group(m) == 'props', m))
    by_market = {}
    for m in markets:
        sample = [r for r in main if r['market'] == m]
        market_moves = movement([x for x in moves if x['market'] == m], rng)
        by_market[m] = dict(label=MARKET_LABELS.get(m, m), group=group(m), main_lines=score(sample, rng),
                            all_lines=score([r for r in records if r['market'] == m], rng), movement=market_moves,
                            card_pool=pool_summary([b for b in bets if b['market'] == m], rng))
        by_market[m]['readiness'] = readiness(by_market[m]['main_lines'], market_moves)
    groups = {}
    for g in ('props', 'game_lines'):
        sample = [r for r in main if group(r['market']) == g]
        g_moves = movement([x for x in moves if group(x['market']) == g], rng)
        groups[g] = dict(main_lines=score(sample, rng), alternate_lines=score(
            [r for r in records if group(r['market']) == g and not r['main']], rng), movement=g_moves,
            card_pool=pool_summary([b for b in bets if group(b['market']) == g], rng),
            calibration=calibration(sample))
        groups[g]['readiness'] = readiness(groups[g]['main_lines'], g_moves)
    weeks = []
    for key in sorted({week_label(r['day'])[0] for r in main}):
        label = week_label(datetime.fromisoformat(key).date())[1]
        weeks.append(dict(week=key, label=label, **{g: score([r for r in main if week_label(r['day'])[0] == key
                                                              and group(r['market']) == g], rng) for g in ('props', 'game_lines')}))
    graded_games = sorted({r['event_id'] for r in records})
    final_days = [parse(started[e]['commence_time']).astimezone(ET).date() for e in graded_games]
    return dict(
        schema=1, generated_at=now.isoformat(), model_version=version,
        status='prospective_shadow_only',
        through=max(final_days).isoformat() if final_days else None,
        coverage=dict(games_tipped=len(started), games_graded=len(graded_games), units_graded=len(records),
                      main_line_units=len(main), settlement=dict(sorted(statuses.items())),
                      other_model_versions_excluded=other_versions),
        criteria=dict(
            unit='One forecast per game, market, player and line from the last snapshot saved before tipoff; '
                 'model and market probabilities are conditional on no push; pushes and voids are left out.',
            market='Average of each paired book\'s no-vig probability at that line (one vote per book).',
            main_line='The line the most books pair for that player and market (ties: nearest the median).',
            blend_weight=BLEND_WEIGHT, card_rule=dict(weight=BLEND_WEIGHT, ev_floor=EV_FLOOR, min_line_books=MIN_LINE_BOOKS),
            movement=f'Consensus change between the last two pregame snapshots (within {MOVE_WINDOW.total_seconds() / 3600:.0f} '
                     f'hours) in the direction of the earlier model lean, where the lean was at least {MOVE_GAP:.0%}.',
            readiness=f'Proposed, not automatic: at least {MIN_GAMES} graded games, the {BLEND_WEIGHT:.0%} blend beating '
                      'the market on log loss with the whole 95% interval below zero, and the market moving toward '
                      'the model with the whole interval above zero.',
            intervals=f'95% percentile intervals from {BOOTSTRAP} bootstrap replicates over games, shown once at '
                      f'least {MIN_CI_GAMES} games are in the sample.'),
        groups=groups, by_market=by_market, by_week=weeks)


def fmt(value, digits=3, signed=False):
    return '—' if value is None else f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def ci(block, digits=3):
    if not block or block.get('mean') is None:
        return '—'
    text = fmt(block['mean'], digits, signed=True)
    return text + (f" ({block['ci95'][0]:+.{digits}f} to {block['ci95'][1]:+.{digits}f})" if block.get('ci95') else '')


def markdown(payload):
    """A short reader's version of the JSON, in Eastern Time."""
    c = payload['coverage']
    lines = ['# NBA v1 shadow model: prospective grading', '',
             f"Model `{payload['model_version']}`, generated {eastern_label(payload['generated_at'])}. "
             f"{c['games_graded']} of {c['games_tipped']} tipped games graded"
             + (f" through {datetime.fromisoformat(payload['through']):%b} {datetime.fromisoformat(payload['through']).day}" if payload['through'] else '')
             + f"; {c['main_line_units']} main-line and {c['units_graded'] - c['main_line_units']} alternate-line forecasts.", '',
             'Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability '
             'points toward the model\'s earlier lean (positive means the market came to the model). The card pool is '
             'every offer that would have passed the NHL card\'s price rule, before research review and card caps.', '',
             *([f'Intervals appear once {MIN_CI_GAMES} games are graded; until then the numbers below are '
                'early anecdotes, not evidence.', ''] if c['games_graded'] < MIN_CI_GAMES else []),
             '| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |',
             '|---|---|---|---|---|---|---|']
    def row(label, block):
        m, mv, pool = block.get('main_lines', {}), block.get('movement', {}), block.get('card_pool', {})
        pool_text = '—' if not pool.get('bets') else f"{pool['wins']}-{pool['losses']}-{pool['pushes']}, {pool['units']:+.2f}u"
        return (f"| {label} | {m.get('games', 0)} | {ci(m.get('model_minus_market'))} | {ci(m.get('blend_minus_market'))} | "
                f"{ci(mv.get('points'), 2)} | {pool_text} | {block['readiness']['status'].replace('_', ' ')} |")
    lines.append(row('**All props**', payload['groups']['props']))
    lines.append(row('**All game lines**', payload['groups']['game_lines']))
    for block in payload['by_market'].values():
        lines.append(row(block['label'], block))
    lines += ['', '**Card pool by source.** *Price*: one book is far enough off the other books that the consensus '
              'alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.', '']
    for g, label in (('props', 'Props'), ('game_lines', 'Game lines')):
        for source, block in (payload['groups'][g]['card_pool'].get('by_source') or {}).items():
            if block.get('bets'):
                lines.append(f"- {label}, {source}: {block['bets']} bets, {block['wins']}-{block['losses']}-{block['pushes']}, "
                             f"{block['units']:+.2f}u (ROI {ci(block['roi'], 2)})")
    lines += ['', payload['criteria']['readiness'], '',
              'Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for '
              'closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it '
              'scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of '
              'them say little about edge.', '']
    return '\n'.join(lines)


def write(payload, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prospective.json').write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n')
    (out / 'PROSPECTIVE.md').write_text(markdown(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--offline', action='store_true', help='grade from saved ESPN responses only (artifacts/nba/results)')
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    payload = grade(args.season, offline=args.offline)
    write(payload, args.out)
    c = payload['coverage']
    print(f"NBA shadow grading: {c['games_graded']} games, {c['units_graded']} forecasts "
          f"({c['main_line_units']} main lines); props {payload['groups']['props']['readiness']['status']}, "
          f"game lines {payload['groups']['game_lines']['readiness']['status']}")


if __name__ == '__main__':
    main()

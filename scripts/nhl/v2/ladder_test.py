"""Grade the one-week pregame milestone-prop test (ladders.py). Research only: no
bet was placed, and nothing here changes Top Picks, Market Watch or public pages.

Each contract (game, player, offered market, line, side) counts once, at its last
pregame decision. The model's bet is the verified-settlement price with the
highest estimated EV at that decision, one flat unit, graded on the official box
score; a player who did not play is void. Writes report.json and REPORT.md next
to the collected quotes.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); __package__ = 'nhl.v2'
from .data import ROOT, iso, load, stamp, write_json
from .grading import betting_metrics, settle
from .ladders import OUT, WINDOW

THRESHOLDS = (.02, .10)
NAMES = {'player_goal_scorer_anytime': 'Anytime goal scorer', 'player_shots_on_goal_alternate': 'Alternate shots',
         'player_points_alternate': 'Alternate points'}


def snapshots(root=OUT):
    out = []
    for path in sorted(Path(root).glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            out.append(json.load(f))
    return out


def contracts(snaps):
    """The last pregame decision per contract, with every book quoted at that decision."""
    latest = {}
    for snap in snaps:
        decided = stamp(snap['decided_at'])
        groups = defaultdict(list)
        for r in snap['rows']:
            if r.get('player_id') is None or stamp(r['commence_time']) <= decided:
                continue
            groups[(r['nhl_game_id'], r['player_id'], r['offered_market'], r['line'], r['side'])].append(r)
        for key, quotes in groups.items():
            if key not in latest or decided > latest[key][0]:
                latest[key] = (decided, quotes)
    return latest


def grade(snaps, games, players):
    by_game = {g['game_id']: g for g in games}
    by_player = {(r['game_id'], r['player_id']): r for r in players}
    out = []
    for key, (decided, quotes) in sorted(contracts(snaps).items(), key=lambda kv: (kv[1][0], str(kv[0]))):
        result = settle(quotes[0], by_game.get(key[0]), by_player.get((key[0], key[1])))
        model = next((q['conditional_probability'] for q in quotes if q.get('conditional_probability') is not None), None)
        best = min(quotes, key=lambda q: q['book_probability'])
        priced = [q for q in quotes if q.get('estimated_ev') is not None]
        pick = max(priced, key=lambda q: q['estimated_ev'], default=None)
        out.append(dict(offered_market=key[2], decided_at=iso(decided), result=result, model=model,
                        best=best, pick=pick, books=len(quotes)))
    return out


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else None


def calibration(rows):
    settled = [r for r in rows if r['result'] in ('won', 'lost') and r['model'] is not None]
    hits = [r['result'] == 'won' for r in settled]
    return dict(contracts=len(settled), model_mean=sum(r['model'] for r in settled) / len(settled) if settled else None,
                hit_rate=sum(hits) / len(hits) if hits else None,
                model_brier=brier([(r['model'], y) for r, y in zip(settled, hits)]),
                # The best price still carries the book's margin: a reference, not a fair probability.
                best_price_brier=brier([(r['best']['book_probability'], y) for r, y in zip(settled, hits)]))


def summarize(graded, snaps, now):
    markets = sorted({r['offered_market'] for r in graded})
    value = {}
    for t in THRESHOLDS:
        picks = [r for r in graded if r['pick'] and r['pick']['estimated_ev'] >= t]
        tickets = lambda rows: [dict(r['pick'], result=r['result']) for r in rows]
        value[f'{t:.2f}'] = dict(all=betting_metrics(tickets(picks)),
            by_market={m: betting_metrics(tickets([r for r in picks if r['offered_market'] == m])) for m in markets})
    counts = defaultdict(int)
    for r in graded:
        counts[r['result']] += 1
    return dict(status='research_only', note='No bet was placed. Contracts count once, at the last pregame decision.',
                generated_at=iso(now), window=[d.isoformat() for d in WINDOW], snapshots=len(snaps),
                model_versions=sorted({s.get('model_version') or 'unknown' for s in snaps}),
                contracts=len(graded), results=dict(counts),
                calibration=dict(all=calibration(graded),
                                 by_market={m: calibration([r for r in graded if r['offered_market'] == m]) for m in markets}),
                value_bets=value)


def verdict(metrics):
    if metrics['count'] < 100:
        return f"{metrics['count']} graded bets so far: too few to judge."
    low, high = metrics['roi_ci']
    if low > 0:
        return f"Profitable so far: ROI {metrics['roi']:+.1%} (95% range {low:+.1%} to {high:+.1%})."
    if high < 0:
        return f"Losing: ROI {metrics['roi']:+.1%} (95% range {low:+.1%} to {high:+.1%})."
    return f"Inconclusive: ROI {metrics['roi']:+.1%}, and the 95% range ({low:+.1%} to {high:+.1%}) includes zero."


def markdown(report):
    pct = lambda v: '—' if v is None else f'{v:.1%}'
    num = lambda v: '—' if v is None else f'{v:.4f}'
    units = lambda v: '—' if v is None else f'{v:+.2f}'
    span = lambda ci: '—' if not ci else f'{ci[0]:+.1%} to {ci[1]:+.1%}'
    lines = ['# NHL milestone-prop test', '',
             f"Research only; no bets were placed. Window {report['window'][0]} to {report['window'][1]} (end exclusive), "
             f"{report['snapshots']} refreshes, model {', '.join(report['model_versions'])}. Generated {report['generated_at']}.", '',
             f"{report['contracts']} contracts priced; results: " + ', '.join(f'{k} {v}' for k, v in sorted(report['results'].items())) + '.', '',
             '## Model bets (best verified price, estimated EV at or above the threshold)', '',
             '| Threshold | Market | Graded | Net units | ROI | 95% range |', '|---|---|---|---|---|---|']
    for t, block in report['value_bets'].items():
        for name, m in [('All', block['all'])] + [(NAMES.get(k, k), v) for k, v in block['by_market'].items()]:
            lines.append(f"| {float(t):.0%} | {name} | {m['count']} | {units(m['net_units'])} | {pct(m['roi'])} | {span(m.get('roi_ci'))} |")
    lines += ['', f"**2% threshold:** {verdict(report['value_bets']['0.02']['all'])}", '',
              '## Calibration (won or lost contracts with a model probability)', '',
              '| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |', '|---|---|---|---|---|---|']
    for name, c in [('All', report['calibration']['all'])] + [(NAMES.get(k, k), v) for k, v in report['calibration']['by_market'].items()]:
        lines.append(f"| {name} | {c['contracts']} | {pct(c['model_mean'])} | {pct(c['hit_rate'])} | {num(c['model_brier'])} | {num(c['best_price_brier'])} |")
    lines += ['', 'Lower Brier is better. The best price still includes the book\'s margin, so it is a reference, not a fair probability.']
    return '\n'.join(lines) + '\n'


def history(path=None):
    """The frozen training history plus this season's cached game records, as in decisions.py."""
    path = path or ROOT/'models/nhl/v2/history.json.gz'
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        data = json.load(f)
    if path == ROOT/'models/nhl/v2/history.json.gz':
        season = json.loads((ROOT/'docs/nhl/data/latest.json').read_text())['season']
        games, players, _ = load(ROOT/'data/nhl/v2/history', [season])
        for key, values in [('games', games), ('players', players)]:
            data[key] = [r for r in data[key] if r['season'] != season] + values
    return data['games'], data['players']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, help='Normalized games and players JSON; default is the cached history')
    parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args(argv)
    snaps = snapshots(args.root)
    if not snaps:
        print('NHL milestone test: no quotes collected yet.')
        return 0
    games, players = history(args.history)
    report = summarize(grade(snaps, games, players), snaps, datetime.now(timezone.utc))
    write_json(args.root/'report.json', report)
    (args.root/'REPORT.md').write_text(markdown(report))
    print(f"NHL milestone test: {report['contracts']} contracts; {verdict(report['value_bets']['0.02']['all'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

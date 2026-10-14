"""Fit the market-blend model weight per sport from graded recap lines.

Scores f = sigmoid(w*logit(model) + (1-w)*logit(market)) by log loss for w in
0..1 on every graded line in the weekly recap data (docs/blog/*-recap-*.json for
NHL/MLB boards, week-*-review-data.json tickets for NFL props). One side per line;
the market is the median no-vig chance of the other books. The 90% range comes from
resampling whole games. Research only: it never changes a pick. See
docs/model-improvement-plan.md, Step 2c.
"""
import argparse
import collections
import glob
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / 'docs/blog'


def clip(p):
    return min(1 - 1e-4, max(1e-4, p))


def logit(p):
    p = clip(p)
    return math.log(p / (1 - p))


def board_lines(sport, after=None):
    """NHL/MLB recap boards: one side per graded line, across every recap file."""
    quotes = collections.defaultdict(list)
    for path in sorted(BLOG.glob(f'{sport.lower()}-recap-*.json')):
        for r in json.loads(path.read_text()).get('board', []):
            if (r.get('model_probability') is None or r.get('market_probability') is None or
                    r.get('outcome') not in ('win', 'loss') or (after and r['commence_time'][:10] <= after)):
                continue
            quotes[(r['event_id'], r['market'], r.get('player') or '', r.get('line'), r['side'])].append(r)
    seen, lines = set(), []
    for key in sorted(quotes, key=str):
        event, market, player, line, _ = key
        base = (event, market, player, abs(line) if market == 'spreads' and line is not None else line)
        if base in seen:
            continue
        seen.add(base)
        rows = quotes[key]
        lines.append(dict(game=event, market=market, model=rows[0]['model_probability'],
                          market_p=statistics.median(r['market_probability'] for r in rows),
                          y=rows[0]['outcome'] == 'win'))
    return lines


def nfl_lines(after_week=None):
    """NFL prop tickets from the weekly review data (weeks with a consensus price)."""
    seen, lines = set(), []
    for path in sorted(glob.glob(str(BLOG / 'week-*-review-data.json'))):
        for r in json.loads(Path(path).read_text()).get('tickets', []):
            if r.get('consensus_prob') is None or r.get('outcome') not in ('win', 'loss'):
                continue
            if after_week and int(r['stat_game'].split('_')[1]) <= after_week:   # '2026_05_SF_SEA'
                continue
            base = (r['stat_game'], r['player'], r['market_std'], r['point'])
            if base in seen:
                continue
            seen.add(base)
            lines.append(dict(game=r['stat_game'], market=r['market_std'], model=r['model_prob'],
                              market_p=r['consensus_prob'], y=r['outcome'] == 'win'))
    return lines


def loss(lines, w):
    total = 0.0
    for r in lines:
        p = clip(1 / (1 + math.exp(-(w * logit(r['model']) + (1 - w) * logit(r['market_p'])))))
        total -= math.log(p if r['y'] else 1 - p)
    return total / len(lines)


def best(lines):
    return min((loss(lines, w / 100), w / 100) for w in range(101))[1]


def game_range(lines, draws=300, seed=7):
    games = collections.defaultdict(list)
    for r in lines:
        games[r['game']].append(r)
    keys, rng, fits = sorted(games), random.Random(seed), []
    for _ in range(draws):
        fits.append(best([r for g in rng.choices(keys, k=len(keys)) for r in games[g]]))
    fits.sort()
    return fits[int(.05 * draws)], fits[int(.95 * draws)]


def report(name, lines, minimum=30):
    if len(lines) < minimum:
        return f'| {name} | {len(lines)} | — | — | — | — | too few lines |'
    lo, hi = game_range(lines)
    return (f"| {name} | {len(lines)} | {len({r['game'] for r in lines})} | {loss(lines, 0):.4f} | "
            f"{loss(lines, 1):.4f} | {best(lines):.2f} | {lo:.2f}–{hi:.2f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', help='NHL/MLB: only games after this date (YYYY-MM-DD)')
    parser.add_argument('--nfl-after-week', type=int, help='NFL: only weeks after this one')
    args = parser.parse_args()
    sets = {'NHL': board_lines('NHL', args.after), 'MLB': board_lines('MLB', args.after),
            'NFL props': nfl_lines(args.nfl_after_week)}
    print('| Lines | Count | Games | Log loss, market | Log loss, model | Best w | 90% range |')
    print('|---|---|---|---|---|---|---|')
    for name, lines in sets.items():
        print(report(name, lines))
        print(report('  model and market 10+ points apart', [r for r in lines if abs(r['model'] - r['market_p']) >= .10]))
        if name != 'NFL props':
            for market in sorted({r['market'] for r in lines}):
                print(report('  ' + market, [r for r in lines if r['market'] == market]))


if __name__ == '__main__':
    main()

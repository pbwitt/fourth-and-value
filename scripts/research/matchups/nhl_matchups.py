"""NHL player props: do home/away splits or a player's history against one opponent help?

Research only. The baseline is the production player model (nhl-v2.3, `opportunity_nb_opp`) on
the production features, trained on earlier seasons exactly as `scripts/nhl/v2/evaluate.py` does.
Shrinkage strengths are chosen on 2024-25 and scored once on 2025-26, the production test season.

    python scripts/research/matchups/nhl_matchups.py --features features.pkl --out reports/matchups/nhl.json

`--features` is a pickle of `scripts/nhl/v2/features.build(games, players)` on
models/nhl/v2/history.json.gz (built if missing; it takes a few minutes).
"""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INF, Ledger, shrunk, cluster_ci, brier, fmt_k, weight_after  # noqa: E402
from nhl.v2.models import PlayerModel  # noqa: E402

STATS = ['shots', 'goals', 'assists', 'points']
FAMILY = {'shots': 0, 'goals': 1, 'assists': 1, 'points': 1}   # one factor for shots, one for scoring
LINES = {'shots': [1.5, 2.5, 3.5, 4.5], 'goals': [.5], 'assists': [.5], 'points': [.5, 1.5]}
GRID = [3, 10, 30, 100, 300, 1000, INF]
SUPPORT = np.arange(48)
VALID, TEST = 20242025, 20252026


def load(features, history):
    if features.exists():
        tr, pr = pickle.load(open(features, 'rb'))
    else:
        from nhl.v2.features import build
        d = json.load(gzip.open(history))
        tr, pr = build(d['games'], d['players'])
        pickle.dump((tr, pr), open(features, 'wb'))
    d = json.load(gzip.open(history))
    games = {g['game_id']: g for g in d['games']}
    side = {(p['game_id'], p['player_id']): p['home'] for p in d['players']}
    for r in pr:
        g, home = games[r['game_id']], side[(r['game_id'], r['player_id'])]
        r['home'] = bool(home)
        r['opponent'] = g['away_id'] if home else g['home_id']
    pr.sort(key=lambda r: (r['game_date'], r['game_id'], r['player_id']))
    return pr


def baseline(pr, season):
    """Production model fitted on seasons before `season`; its means for every row."""
    model = PlayerModel('opportunity_nb_opp').fit([r for r in pr if r['season'] < season])
    means = np.array([model.means(r) for r in pr])
    return model, means


def league_home(pr, means, season):
    """Actual / expected at home and away, per family, from the training seasons."""
    y = np.array([r['targets'] for r in pr], float)
    train = np.array([r['season'] < season for r in pr])
    home = np.array([r['home'] for r in pr])
    out = {}
    for fam, cols in [(0, [0]), (1, [3])]:
        level = y[train][:, cols].sum() / means[train][:, cols].sum()   # the gap alone, not a level shift
        for h in (True, False):
            m = train & (home == h)
            out[(fam, h)] = float(y[m][:, cols].sum() / means[m][:, cols].sum() / level)
    return out


def walk(pr, means, h, k):
    """Per-row factors from strictly earlier dates: overall, venue split and opponent split.

    Residuals are measured against the baseline times the league home factor, so a split is
    what the player does beyond the league-wide venue effect.
    """
    y = np.array([r['targets'] for r in pr], float)
    led = {fam: (Ledger(), Ledger(), Ledger()) for fam in (0, 1)}
    out = np.ones((len(pr), 2, 5))   # family x [overall, venue split, opponent split, naive h2h, prior n vs opp]
    i = 0
    while i < len(pr):
        j = i
        while j < len(pr) and pr[j]['game_date'] == pr[i]['game_date']:
            j += 1
        for t in range(i, j):
            r = pr[t]
            for fam in (0, 1):
                every, venue, opp = led[fam]
                ka, kv, ko = k[fam]
                a, e, _ = every.get(r['player_id'])
                overall = shrunk(a, e, ka)
                av, ev, _ = venue.get((r['player_id'], r['home']))
                ao, eo, no = opp.get((r['player_id'], r['opponent']))
                out[t, fam] = [overall, shrunk(av, ev, kv, overall) / overall, shrunk(ao, eo, ko, overall) / overall,
                               shrunk(ao, eo, ko), no]
        for t in range(i, j):
            r = pr[t]
            for fam, col in [(0, 0), (1, 3)]:
                exp = means[t, col] * h[(fam, r['home'])]
                every, venue, opp = led[fam]
                every.add(r['player_id'], y[t, col], exp)
                venue.add((r['player_id'], r['home']), y[t, col], exp)
                opp.add((r['player_id'], r['opponent']), y[t, col], exp)
        i = j
    return out


def pmf(mean, alpha):
    mean = np.clip(mean, 1e-6, 18)[:, None]
    p = nbinom.pmf(SUPPORT[None, :], 1 / alpha, 1 / (1 + alpha * mean)) if alpha > 1e-6 else poisson.pmf(SUPPORT[None, :], mean)
    return p / p.sum(axis=1, keepdims=True)


def score(model, means, rows, idx):
    """Per-row log loss and Brier at each reference line, for each stat."""
    y = np.array([rows[t]['targets'] for t in idx], int)
    out = {}
    for j, stat in enumerate(STATS):
        alpha = model.alpha_shots if j == 0 else model.alpha_scoring
        p = pmf(means[idx, j], alpha)
        nll = -np.log(np.clip(p[np.arange(len(idx)), y[:, j]], 1e-12, 1))
        out[stat] = dict(nll=nll, **{f'brier_{line}': brier(p[:, SUPPORT > line].sum(axis=1), y[:, j] > line) for line in LINES[stat]})
    return out


VARIANTS = {
    'production': lambda f, h: np.ones(f.shape[:2]),
    '+ league home/away': lambda f, h: h,
    '+ player overall (control)': lambda f, h: h * f[:, :, 0],
    '+ player home/away split': lambda f, h: h * f[:, :, 0] * f[:, :, 1],
    '+ player vs opponent': lambda f, h: h * f[:, :, 0] * f[:, :, 2],
    '+ both splits': lambda f, h: h * f[:, :, 0] * f[:, :, 1] * f[:, :, 2],
    'naive: league home + vs opponent only': lambda f, h: h * f[:, :, 3],
}


def adjusted(means, factors, hrow, name):
    mult = VARIANTS[name](factors, hrow)         # rows x family
    return means * mult[:, [0, 1, 1, 1]]


def fit_k(pr, model, means, h, idx):
    """Choose k per family on the validation season: overall first, then each split given it."""
    best = {0: [INF, INF, INF], 1: [INF, INF, INF]}
    hrow = np.array([[h[(0, r['home'])], h[(1, r['home'])]] for r in pr])
    path = []
    for slot, variant in [(0, '+ player overall (control)'), (1, '+ player home/away split'), (2, '+ player vs opponent')]:
        for fam, stats in [(0, ['shots']), (1, ['goals', 'assists', 'points'])]:
            trials = {}
            for k in GRID:
                trial = {f: list(v) for f, v in best.items()}
                trial[fam][slot] = k
                f = walk(pr, means, h, trial)
                s = score(model, adjusted(means, f, hrow, variant), pr, idx)
                trials[k] = float(np.mean([s[st]['nll'].mean() for st in stats]))
            pick = min(trials, key=trials.get)
            best[fam][slot] = pick
            path.append(dict(family='shots' if fam == 0 else 'scoring', component=['overall', 'home/away split', 'vs opponent'][slot],
                             chosen_k=fmt_k(pick), validation_log_loss={fmt_k(k): round(v, 6) for k, v in trials.items()}))
            print(path[-1], flush=True)
    return best, path


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--features', type=Path, required=True)
    a.add_argument('--history', type=Path, default=ROOT / 'models/nhl/v2/history.json.gz')
    a.add_argument('--out', type=Path, default=ROOT / 'reports/matchups/nhl.json')
    args = a.parse_args()
    pr = load(args.features, args.history)
    report = dict(baseline='nhl-v2.3 opportunity_nb_opp, refit per fold on earlier seasons', rows=len(pr))

    # 1. Choose shrinkage on 2024-25 with a model trained through 2023-24.
    model, means = baseline(pr, VALID)
    h = league_home(pr, means, VALID)
    vidx = np.array([t for t, r in enumerate(pr) if r['season'] == VALID])
    k, report["selection"] = fit_k(pr, model, means, h, vidx)

    # 2. Score once on 2025-26 with a model trained through 2024-25 and the chosen k.
    model, means = baseline(pr, TEST)
    h = league_home(pr, means, TEST)
    report['league_home_factor'] = {('shots' if f == 0 else 'scoring') + (' home' if hh else ' away'): round(v, 4) for (f, hh), v in h.items()}
    hrow = np.array([[h[(0, r['home'])], h[(1, r['home'])]] for r in pr])
    f = walk(pr, means, h, k)
    tidx = np.array([t for t, r in enumerate(pr) if r['season'] == TEST])
    groups = np.array([pr[t]['game_id'] for t in tidx])
    scores = {name: score(model, adjusted(means, f, hrow, name), pr, tidx) for name in VARIANTS}
    base = scores['production']
    control = scores['+ player overall (control)']
    results = {}
    for name, s in scores.items():
        results[name] = {}
        for stat in STATS:
            row = {}
            for metric in s[stat]:
                row[metric] = round(float(s[stat][metric].mean()), 6)
                row[metric + '_vs_production'] = {x: round(v, 6) if isinstance(v, float) else v for x, v in cluster_ci(s[stat][metric] - base[stat][metric], groups).items()}
                if name in ('+ player home/away split', '+ player vs opponent', '+ both splits'):
                    row[metric + '_vs_control'] = {x: round(v, 6) if isinstance(v, float) else v for x, v in cluster_ci(s[stat][metric] - control[stat][metric], groups).items()}
            results[name][stat] = row
    report['test'] = dict(season='2025-26', rows=int(len(tidx)), games=int(len(set(groups))), results=results)

    # 3. Matchup "streaks": players who beat their expectation by 30%+ against this opponent in
    # 4+ earlier games. How did they do against that opponent this time, versus the forecast?
    y = np.array([r['targets'] for r in pr], float)
    ctrl = adjusted(means, f, hrow, '+ player overall (control)')
    streak = {}
    raw_walk = walk(pr, means, h, {0: [INF, INF, 0], 1: [INF, INF, 0]})   # unshrunk ratio vs each opponent
    for fam, col, name in [(0, 0, 'shots'), (1, 3, 'points')]:
        raw = []
        fr = raw_walk[tidx, fam]
        ratio, n = fr[:, 3], fr[:, 4]
        for label, mask in [('everyone with 4+ prior games vs this opponent (reference)', n >= 4),
                            ('beat expectation by 30%+ vs this opponent (4+ prior games)', (ratio >= 1.3) & (n >= 4)),
                            ('fell 23%+ short vs this opponent (4+ prior games)', (ratio <= 1 / 1.3) & (n >= 4))]:
            ids = tidx[mask]
            raw.append(dict(group=label, rows=int(mask.sum()),
                            prior_ratio=round(float(ratio[mask].mean()), 3) if mask.any() else None,
                            next_game_actual_over_forecast=round(float(y[ids, col].sum() / ctrl[ids, col].sum()), 3) if mask.any() else None))
        streak[name] = raw
    report['matchup_streaks'] = streak
    report['k'] = {('shots' if fam == 0 else 'scoring'): dict(zip(['overall', 'home/away split', 'vs opponent'], map(fmt_k, v))) for fam, v in k.items()}
    report['weight_examples'] = {('shots' if fam == 0 else 'scoring'): {
        'vs opponent after 8 games': round(weight_after((8 * (2.4 if fam == 0 else .55)), v[2]), 3),
        'home/away split after 40 games': round(weight_after((40 * (2.4 if fam == 0 else .55)), v[1]), 3)} for fam, v in k.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({n: {s: (v[s]['nll'], v[s]['nll_vs_production']) for s in ('shots', 'points')} for n, v in results.items()}, indent=1))


if __name__ == '__main__':
    main()

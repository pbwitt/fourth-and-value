"""NFL home/away multipliers: fitted gaps vs. the fixed production values.

Research for make_player_prop_params.HOME_AWAY_MULTIPLIERS. For each market the gap is the ratio of
actual to baseline expected output at home and away, each divided by the overall ratio, so an
era-level bias in the baseline is not counted as a venue effect. Neutral sites are excluded.

1. Holdout check: gaps fitted on 2012-2021, scored on 2024 through 2026 week 4 against the fixed values
   (squared error, with game-clustered 95% intervals).
2. Production values: gaps refitted on every completed regular season (2012-2025), with 95% intervals
   from resampling games.

    python scripts/research/matchups/nfl_venue.py --data DIR --out reports/matchups/nfl_venue.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cluster_ci  # noqa: E402
from nfl_matchups import FIXED_HOME, MARKETS, ROOT, TRAIN, defense, load, rows_for  # noqa: E402


def gaps(y, base, venue, mask):
    level = y[mask & (venue != 'neutral')].sum() / base[mask & (venue != 'neutral')].sum()
    return {v: float(y[mask & (venue == v)].sum() / base[mask & (venue == v)].sum() / level) for v in ('home', 'away')}


def multipliers(gap, venue):
    return np.array([gap.get(v, 1.) for v in venue])


def boot(y, base, venue, games, mask, draws=400, seed=11):
    """95% intervals for the fitted home and away factors, resampling whole games."""
    keys, inv = np.unique(games[mask], return_inverse=True)
    yy, bb, vv = y[mask], base[mask], venue[mask]
    sums = {}
    for name, sel in [('all', vv != 'neutral'), ('home', vv == 'home'), ('away', vv == 'away')]:
        sums[name] = (np.bincount(inv, yy * sel, len(keys)), np.bincount(inv, bb * sel, len(keys)))
    rng = np.random.default_rng(seed)
    out = {'home': [], 'away': []}
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        level = sums['all'][0][pick].sum() / sums['all'][1][pick].sum()
        for v in ('home', 'away'):
            out[v].append(sums[v][0][pick].sum() / sums[v][1][pick].sum() / level)
    return {v: [round(float(np.quantile(x, .025)), 4), round(float(np.quantile(x, .975)), 4)] for v, x in out.items()}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--out', type=Path, default=ROOT / 'reports/matchups/nfl_venue.json')
    args = a.parse_args()
    d = load(args.data)
    dm = defense(d)
    report = dict(method=__doc__.split('\n\n')[1].replace('\n', ' '), markets={})
    for market, (stat, _, positions, side, _) in MARKETS.items():
        pool = d[d.season.isin(TRAIN) & d.position.isin(positions)].groupby('position')[stat].mean().to_dict()
        rows = rows_for(d, market, dm, pool)
        y, venue, games = rows.y.values, rows.venue.values, rows.game.values
        train, test, done = rows.season.isin(TRAIN).values, (rows.season >= 2024).values, (rows.season <= 2025).values
        blend = min(np.round(np.arange(0, 1.01, .1), 1),
                    key=lambda w: np.mean(((w * rows.recent + (1 - w) * rows.career)[train] * rows.defense[train] - rows.y[train]) ** 2))
        base = (blend * rows.recent + (1 - blend) * rows.career).values * rows.defense.values
        fixed_home = FIXED_HOME.get(market, 1.)
        fixed = {'home': fixed_home, 'away': 2 - fixed_home}
        fit = gaps(y, base, venue, train)
        err = lambda g: (base[test] * multipliers(g, venue[test]) - y[test]) ** 2
        final = gaps(y, base, venue, done)
        report['markets'][market] = dict(
            rows=dict(train=int(train.sum()), test=int(test.sum()), all_completed=int(done.sum())),
            production_fixed=fixed, fitted_2012_2021={k: round(v, 4) for k, v in fit.items()},
            test_sq_error_fixed=round(float(err(fixed).mean()), 4), test_sq_error_fitted=round(float(err(fit).mean()), 4),
            test_fitted_minus_fixed=cluster_ci(err(fit) - err(fixed), games[test]),
            fitted_2012_2025={k: round(v, 4) for k, v in final.items()}, interval_2012_2025=boot(y, base, venue, games, done))
        r = report['markets'][market]
        print(market, r['production_fixed'], r['fitted_2012_2025'], r['interval_2012_2025'],
              'test fitted-fixed', round(r['test_fitted_minus_fixed']['mean'], 4), (round(r['test_fitted_minus_fixed']['lo'], 4), round(r['test_fitted_minus_fixed']['hi'], 4)), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()

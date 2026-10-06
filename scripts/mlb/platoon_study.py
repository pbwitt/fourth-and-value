"""Does platoon (batter side vs pitcher hand) improve the production MLB model?

Every target is fitted twice on identical rows, outcomes and dates: once with the platoon
features and once without. Splits are train.py's own: fold 0 is its regular-season test, and
three earlier 30-day windows repeat the same 60/30/30-day recipe for more held-out games. The
prior-postseason audit is repeated as well. Changes are paired forecast by forecast and their
intervals resample whole games. The decision rule in RULE was fixed before any result was seen.

    python scripts/mlb/platoon_study.py [--cached-history] [--out reports/mlb-platoon/results.json]
"""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from nba.pipeline import iso, save_json
from mlb.model_data import load, update, update_players
from mlb.models import TARGETS, PLATOON, dataset, without_platoon, baseline
from mlb.train import fit_model, forecasts, evaluate, partition, chronology, postseason_dates, score

MARKETS = TARGETS[1:]+['totals', 'h2h', 'spreads']
FOLDS = 4
BOOTSTRAP = 2000
ECE_TOLERANCE = .005
RULE = dict(
    comparison='Platoon minus no platoon, paired on the same held-out forecasts; negative is better.',
    interval='95% percentile interval from 2,000 bootstrap resamples of whole games.',
    worse='A market is worse when its pooled Brier or log-loss interval lies entirely above zero.',
    better='A market is better when its pooled Brier interval lies entirely below zero.',
    calibration=f'No market\'s pooled calibration gap rises by more than {ECE_TOLERANCE}, and no market that '
                'passes train.py\'s regular-season or postseason checks without platoon fails them with it.',
    ship='Ship when no market is worse, the calibration guard holds, and either at least one market '
         'is better or the average pooled Brier change across the nine markets is at or below zero.')


def fold_dates(games, k):
    latest, train_end, cal_end = chronology(games)
    if k == 0:
        return train_end, cal_end, latest.isoformat()
    end = latest-timedelta(days=30*k)
    return (end-timedelta(days=60)).isoformat(), (end-timedelta(days=30)).isoformat(), end.isoformat()


def fit_all(training, calibration):
    return {target: fit_model(target, training[target], calibration[target]) for target in TARGETS}


def interval(deltas, ids, rng):
    """Mean paired change and a game-resampled 95% interval."""
    deltas = np.asarray(deltas, float)
    _, inverse = np.unique(np.asarray(ids), return_inverse=True)
    sums = np.bincount(inverse, weights=deltas); counts = np.bincount(inverse).astype(float)
    draws = rng.integers(0, len(sums), size=(BOOTSTRAP, len(sums)))
    means = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(deltas.mean()), low=float(low), high=float(high))


def losses(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, float)
    return (p-y)**2, -(y*np.log(p)+(1-y)*np.log(1-p))


def paired(with_values, without_values, rng, post=False):
    result = {}
    for market in MARKETS:
        a, b = with_values[market], without_values[market]
        if not a['y']:
            result[market] = dict(forecasts=0)
            continue
        if a['y'] != b['y'] or a['ids'] != b['ids']:
            raise ValueError(f'{market}: the two variants were not scored on identical forecasts')
        brier_a, log_a = losses(a['p'], a['y']); brier_b, log_b = losses(b['p'], b['y'])
        with_score = score(a['p'], a['y'], a['b'], a['ids'], post); without_score = score(b['p'], b['y'], b['b'], b['ids'], post)
        result[market] = dict(forecasts=len(a['y']), games=len(set(a['ids'])),
            brier=interval(brier_a-brier_b, a['ids'], rng), log_loss=interval(log_a-log_b, a['ids'], rng),
            with_platoon={k: with_score[k] for k in ['brier', 'log_loss', 'ece', 'brier_skill']},
            without_platoon={k: without_score[k] for k in ['brier', 'log_loss', 'ece', 'brier_skill']})
    return result


def pooled(folds):
    values = {}
    for fold in folds:
        for market in MARKETS:
            target = values.setdefault(market, {variant: dict(p=[], y=[], b=[], ids=[]) for variant in ['with', 'without']})
            for variant in ['with', 'without']:
                for key in ['p', 'y', 'b', 'ids']:
                    target[variant][key] += fold[variant][market][key]
    return {variant: {m: values[m][variant] for m in MARKETS} for variant in ['with', 'without']}


def coverage(games, hands, samples):
    starters = [t['starter'] for g in games for t in g['teams'].values()]
    batter_pa = [(b['id'], b['plateAppearances']) for g in games for t in g['teams'].values() for b in t['batters']]
    total = sum(pa for _, pa in batter_pa) or 1
    known = sum(pa for i, pa in batter_pa if (hands.get(i) or {}).get('bats'))
    sides = {code: sum(pa for i, pa in batter_pa if (hands.get(i) or {}).get('bats') == code)/total for code in 'LRS'}
    # Box-score sanity check of direction: actual outcome over the rolling baseline by matchup.
    direction = {}
    for target in ['batter_hits', 'batter_total_bases', 'batter_home_runs']:
        groups = {}
        for row in samples[target]:
            label = {0.: 'opposite hand or switch', 1.: 'same hand'}.get(row['x'].get('batter_same_hand'))
            if label:
                g = groups.setdefault(label, [0., 0., 0])
                g[0] += row['y']; g[1] += baseline(row['x'], target); g[2] += 1
        direction[target] = {k: dict(rows=v[2], actual_over_baseline=round(v[0]/v[1], 4)) for k, v in groups.items() if v[1]}
    return dict(players=len(hands), starters_with_hand=round(sum(bool((hands.get(i) or {}).get('throws')) for i in starters)/len(starters), 4),
        batter_pa_with_side=round(known/total, 4), batter_pa_share_by_side={k: round(v, 4) for k, v in sides.items()},
        direction=direction)


def decide(pool, primary, postseason):
    worse = [m for m, r in pool.items() if r.get('forecasts') and (r['brier']['low'] > 0 or r['log_loss']['low'] > 0)]
    better = [m for m, r in pool.items() if r.get('forecasts') and r['brier']['high'] < 0]
    average = float(np.mean([r['brier']['mean'] for r in pool.values() if r.get('forecasts')]))
    ece = [m for m, r in pool.items() if r.get('forecasts')
           and r['with_platoon']['ece']-r['without_platoon']['ece'] > ECE_TOLERANCE]
    flips = [f'{name}:{m}' for name, report in [('regular', primary), ('postseason', postseason)]
             for m in MARKETS if report['without'].get(m, {}).get('passed') and not report['with'].get(m, {}).get('passed')]
    calibration_ok = not ece and not flips
    ship = not worse and calibration_ok and (bool(better) or average <= 0)
    return dict(ship=ship, worse=worse, better=better, average_brier_change=average,
                calibration_worse=ece, validation_flips=flips, calibration_ok=calibration_ok)


def summary(results):
    def fmt(r):
        return f"{1e4*r['mean']:+.2f} ({1e4*r['low']:+.2f} to {1e4*r['high']:+.2f})"
    lines = ['# MLB platoon study', '', f"Games {results['games']:,}, through {results['input_through']}. "
             f"Pooled regular-season test: {results['pooled_windows']}.", '',
             '| Market | Forecasts | Games | Brier change ×10⁻⁴ | Log-loss change ×10⁻⁴ | ECE without → with |',
             '|---|---|---|---|---|---|']
    for market, r in results['pooled'].items():
        if r.get('forecasts'):
            lines.append(f"| {market} | {r['forecasts']:,} | {r['games']} | {fmt(r['brier'])} | {fmt(r['log_loss'])} | "
                         f"{r['without_platoon']['ece']:.4f} → {r['with_platoon']['ece']:.4f} |")
    d = results['decision']
    lines += ['', f"Decision: {'SHIP' if d['ship'] else 'RESEARCH ONLY'}. Worse: {d['worse'] or 'none'}. "
              f"Better: {d['better'] or 'none'}. Average Brier change ×10⁻⁴: {1e4*d['average_brier_change']:+.2f}. "
              f"Calibration guard: {'holds' if d['calibration_ok'] else 'fails'} "
              f"({d['calibration_worse'] or 'no ECE rise'}; {d['validation_flips'] or 'no pass/fail flips'})."]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cached-history', action='store_true', help='skip the box-score refresh')
    parser.add_argument('--out', default=str(ROOT/'reports/mlb-platoon/results.json'))
    args = parser.parse_args(argv)
    if not args.cached_history:
        update()
    games, manifest = load()
    missing = manifest['expected_games']-len(games)
    if missing:
        raise ValueError(f'{missing} expected game observations are missing; study stopped')
    hands = update_players(games)
    print(f'MLB platoon study: {len(games)} games, {len(hands)} players with handedness', flush=True)
    samples, _ = dataset(games, hands)
    plain = without_platoon(samples)
    rng = np.random.default_rng(20261006)
    folds, windows, kinds = [], [], {}
    for k in range(FOLDS):
        train_end, cal_end, test_end = fold_dates(games, k)
        variants = {}
        for name, data in [('with', samples), ('without', plain)]:
            training, calibration, test = partition(data, train_end, cal_end, test_end)
            models = fit_all(training, calibration)
            variants[name] = forecasts(models, test)
            if k == 0:
                variants[name+'_report'] = evaluate(models, test)
                kinds[name] = {t: m['kind'] for t, m in models.items()}
        start = (date.fromisoformat(cal_end)+timedelta(days=1)).isoformat()
        windows.append(dict(fold=k, training_through=train_end, calibration_through=cal_end, test_start=start, test_end=test_end))
        print(f'MLB platoon study: fold {k} tested {start} to {test_end}', flush=True)
        folds.append(variants)
    latest = chronology(games)[0]
    prior_year, post_train_end, reg_end = postseason_dates(games, latest)
    post = {}
    for name, data in [('with', samples), ('without', plain)]:
        pt, pc, pe = partition(data, post_train_end, reg_end, f'{prior_year}-12-31', post_only=True)
        models = fit_all(pt, pc)
        post[name] = forecasts(models, pe); post[name+'_report'] = evaluate(models, pe, post=True)
    pool = pooled(folds)
    primary = {'with': folds[0]['with_report'], 'without': folds[0]['without_report']}
    postseason = {'with': post['with_report'], 'without': post['without_report']}
    results = dict(created_at=iso(datetime.now(timezone.utc)), games=len(games), input_through=latest.isoformat(),
        features=list(PLATOON), rule=RULE, coverage=coverage(games, hands, samples), windows=windows,
        pooled_windows=f"{windows[-1]['test_start']} to {windows[0]['test_end']} ({FOLDS} windows, regular season)",
        model_kinds=kinds,
        pooled=paired(pool['with'], pool['without'], rng),
        primary=paired(folds[0]['with'], folds[0]['without'], rng),
        primary_validation={v: {m: dict(passed=r.get('passed'), brier=r.get('brier'), ece=r.get('ece')) for m, r in rep.items()}
                            for v, rep in primary.items()},
        postseason_year=prior_year, postseason=paired(post['with'], post['without'], rng, post=True),
        postseason_validation={v: {m: dict(passed=r.get('passed'), brier=r.get('brier'), ece=r.get('ece')) for m, r in rep.items()}
                               for v, rep in postseason.items()})
    results['decision'] = decide(results['pooled'], primary, postseason)
    save_json(Path(args.out), results)
    text = summary(results)
    print(text, flush=True)
    print('PLATOON-RESULTS-JSON-BEGIN', flush=True)
    print(json.dumps(results, separators=(',', ':')), flush=True)
    print('PLATOON-RESULTS-JSON-END', flush=True)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(text+'\n')
    return results


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train and evaluate NBA v1 from the committed ESPN history (no network, no odds).

Chronological protocol:
  * select hyperparameters: fit on 2023-24 regular season from December 1 (earlier games
    have no prior-season state), validate on 2024-25 regular season;
  * evaluate: refit those choices on both seasons, test once on 2025-26 regular season;
  * production: refit the same choices on all three seasons -> artifacts/nba/model/.

Writes reports/nba-v1/evaluation.json and EVALUATION.md. Pseudo-lines (a player's last-10
average rounded to a half point) are diagnostics, not sportsbook offers; no price claim is
made until archived 2026-27 odds are graded.
"""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats as sps
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from nba import model as M  # noqa: E402
from nba.espn_history import HISTORY, load_all  # noqa: E402

MODEL_PATH = ROOT / 'artifacts/nba/model/nba-v1.json'
REPORT = ROOT / 'reports/nba-v1'
TEAM_GRID = [dict(half_life=h, carry=c, prior_games=k) for h, c, k in itertools.product((10, 20, 40), (0.25, 0.5), (3, 8))]
PLAYER_GRID = [dict(rate_half_life=h, prior_minutes=k, opponent_weight=b)
               for h, k, b in itertools.product((20, 50), (100, 300), (0.0, 0.5, 1.0))]
NODES = 11


def season_of(row):
    day = M.day_of(row['game']['GAME_DATE'] if 'game' in row else row['row']['GAME_DATE'])
    return M.season_key(day)


def in_split(row, split):
    g = row['game'] if 'game' in row else row['row']
    kind = g['SEASON_TYPE']
    if kind != 'regular':
        return False
    year, day = season_of(row), g['GAME_DATE']
    return {'fit': year == 2023 and day >= '2023-12-01', 'valid': year == 2024, 'test': year == 2025,
            'fit+valid': (year == 2023 and day >= '2023-12-01') or year == 2024,
            'all': (year == 2023 and day >= '2023-12-01') or year in (2024, 2025)}[split]


def gaussian_nll(params, rows):
    cov = np.array(params['cov'])
    X = M.design(rows, M.GAME_FEATURES)
    e = np.column_stack([np.array([r['home_reg'] for r in rows]) - M.predict_ridge(params['home'], X),
                         np.array([r['away_reg'] for r in rows]) - M.predict_ridge(params['away'], X)])
    inv, (_, logdet) = np.linalg.inv(cov), np.linalg.slogdet(cov)
    return float(np.mean(0.5 * np.einsum('ij,jk,ik->i', e, inv, e) + 0.5 * logdet + math.log(2 * math.pi)))


def with_regulation(rows):
    return [r for r in rows if r['home_reg'] is not None and r['away_reg'] is not None]


def select_team(history):
    results = []
    for cfg in TEAM_GRID:
        rows, overtime, _ = M.game_table(history, cfg)
        fit = with_regulation([r for r in rows if in_split(r, 'fit')])
        valid = with_regulation([r for r in rows if in_split(r, 'valid')])
        ot = [o for o in overtime if o['day'] < '2024-07-01']
        params = M.fit_game(fit, ot)
        results.append((gaussian_nll(params, valid), cfg))
    results.sort(key=lambda x: x[0])
    return results[0][1], [dict(cfg, valid_nll=round(n, 4)) for n, cfg in results]


def attach_means(rows, params):
    X = M.design(rows, M.GAME_FEATURES)
    for r, mh, ma in zip(rows, M.predict_ridge(params['home'], X), M.predict_ridge(params['away'], X)):
        r['mean_home'], r['mean_away'] = float(mh), float(ma)


def fit_players(snapshots, cfg):
    priors = M.position_priors(snapshots)
    X = np.array([[s['minutes'][k] for k in M.MINUTE_FEATURES] for s in snapshots])
    y = np.array([float(s['row']['MIN']) for s in snapshots])
    minutes = M.fit_ridge(X, y, alpha=1.0)
    sd = M.minutes_sd(snapshots, minutes)
    # Scale: observed totals / rate-model totals at actual minutes (corrects shrinkage bias).
    raw = defaultdict(float)
    obs = defaultdict(float)
    for s in snapshots:
        r = M.rate_means(s, cfg, priors)
        for st in M.STATS:
            raw[st] += r[st] * float(s['row']['MIN'])
            obs[st] += s['row'][st]
    scale = {st: obs[st] / raw[st] for st in M.STATS}
    params = dict(config=cfg, priors=priors, minutes=minutes, minutes_sd=sd, scale=scale)
    # Blend: actual count ~ structural mean at projected minutes + last-10 and last-30 averages.
    projected = M.predict_ridge(minutes, X).clip(1, 46)
    stack = {}
    for st in M.STATS:
        Z = np.array([M.stack_inputs(s, params, m)[st] for s, m in zip(snapshots, projected)])
        stack[st] = np.linalg.lstsq(Z, np.array([s['row'][st] for s in snapshots], float), rcond=None)[0].tolist()
    params['stack'] = stack
    _, _, nodes, rates = vector_inputs(params, snapshots)
    per_snapshot = [{st: float(rates[st][i]) for st in M.STATS} for i in range(len(snapshots))]
    params['shape'], params['correlation'] = M.fit_dispersion(snapshots, per_snapshot, nodes)
    params['train_appearances'] = len(snapshots)
    return params


def vector_inputs(params, snapshots):
    """Minutes nodes (apps x NODES) and per-minute rates per stat (apps)."""
    X = np.array([[s['minutes'][k] for k in M.MINUTE_FEATURES] for s in snapshots])
    mu = np.clip(M.predict_ridge(params['minutes'], X), 1, 46)
    sd = np.interp(mu, params['minutes_sd']['centers'], params['minutes_sd']['sds'])
    q = (np.arange(NODES) + 0.5) / NODES
    a, b = (0.5 - mu) / sd, (48 - mu) / sd
    nodes = sps.truncnorm.ppf(q[None, :], a[:, None], b[:, None], loc=mu[:, None], scale=sd[:, None])
    final = [M.final_rates(s, params, m) for s, m in zip(snapshots, mu)]
    rates = {st: np.array([r[st] for r in final]) for st in M.STATS}
    return mu, sd, nodes, rates


def count_logloss(params, snapshots, inputs=None):
    mu, sd, nodes, rates = inputs or vector_inputs(params, snapshots)
    out = {}
    for st in M.STATS:
        k = params['shape'][st]
        means = np.maximum(rates[st][:, None] * nodes, 1e-9)
        y = np.array([s['row'][st] for s in snapshots])[:, None]
        lp = sps.nbinom.logpmf(y, k, k / (k + means))
        out[st] = float(-np.mean(logsumexp(lp, axis=1) - math.log(NODES)))
    return out


def over_probability(params, inputs, st, lines):
    _, _, nodes, rates = inputs
    k = params['shape'][st]
    means = np.maximum(rates[st][:, None] * nodes, 1e-9)
    cdf = sps.nbinom.cdf(np.floor(lines)[:, None], k, k / (k + means)).mean(axis=1)
    return 1 - cdf


def select_players(history, team_cfg):
    rows, overtime, _ = M.game_table(history, team_cfg)
    game = M.fit_game(with_regulation([r for r in rows if in_split(r, 'fit')]), [o for o in overtime if o['day'] < '2024-07-01'])
    attach_means(rows, game)
    results = []
    cache = {}
    for cfg in PLAYER_GRID:
        if cfg['rate_half_life'] not in cache:
            cache[cfg['rate_half_life']] = M.player_table(history, rows, (cfg['rate_half_life'],))[0]
        snaps = cache[cfg['rate_half_life']]
        fit = [s for s in snaps if in_split(s, 'fit')]
        valid = [s for s in snaps if in_split(s, 'valid')]
        params = fit_players(fit, cfg)
        ll = count_logloss(params, valid)
        results.append((sum(ll.values()), cfg, ll))
    results.sort(key=lambda x: x[0])
    return results[0][1], [dict(cfg, valid_logloss_sum=round(t, 4)) for t, cfg, _ in results]


def baselines(history):
    """Production baseline inputs per (game, player): last 30 regular games within 370 days."""
    out, past = {}, defaultdict(deque)
    by_day = defaultdict(list)
    for r in history['players']:
        if not r.get('DNP') and r.get('MIN') and r['SEASON_TYPE'] == 'regular':
            by_day[r['GAME_DATE']].append(r)
    for day in sorted(by_day):
        cutoff = (M.day_of(day) - timedelta(days=370)).isoformat()
        for r in by_day[day]:
            games = [g for g in past[r['PLAYER_ID']] if g['GAME_DATE'] > cutoff][-30:]
            out[(r['GAME_ID'], r['PLAYER_ID'])] = games
        for r in by_day[day]:
            past[r['PLAYER_ID']].append(r)
            if len(past[r['PLAYER_ID']]) > 40:
                past[r['PLAYER_ID']].popleft()
    return out


def binary_scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), brier=float(np.mean((p - y) ** 2)))


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum() >= 30:
            out.append(dict(bin=f'{lo:.1f}-{hi:.1f}', n=int(mask.sum()), mean_forecast=round(float(p[mask].mean()), 4),
                            observed=round(float(y[mask].mean()), 4)))
    return out


def evaluate_players(params, test, base, slopes=None):
    inputs = vector_inputs(params, test)
    results = dict(count_logloss_model=count_logloss(params, test, inputs))
    first = min(s['row']['GAME_DATE'] for s in test)
    early = [s for s in test if M.day_of(s['row']['GAME_DATE']) < M.day_of(first) + timedelta(days=21)]
    results['first_three_weeks'] = dict(appearances=len(early), count_logloss_model=count_logloss(params, early))
    # Recency baseline: NB around the production 30-game mean with the same shapes.
    eligible = np.array([len(base.get((s['row']['GAME_ID'], s['row']['PLAYER_ID']), [])) >= 20 for s in test])
    results['appearances'] = len(test)
    results['appearances_with_baseline'] = int(eligible.sum())
    sub = [s for s, e in zip(test, eligible) if e]
    sub_inputs = tuple(x[eligible] if isinstance(x, np.ndarray) else {k: v[eligible] for k, v in x.items()} for x in inputs)
    results['count_logloss_model_on_baseline_subset'] = count_logloss(params, sub, sub_inputs)
    base_ll = {}
    for st in M.STATS:
        k = params['shape'][st]
        mean = np.array([max(np.mean([g[st] for g in base[(s['row']['GAME_ID'], s['row']['PLAYER_ID'])]]), 0.05) for s in sub])
        y = np.array([s['row'][st] for s in sub])
        base_ll[st] = float(-np.mean(sps.nbinom.logpmf(y, k, k / (k + mean))))
    results['count_logloss_recency_baseline'] = base_ll
    markets = {}
    for market, (model_p, base_p, y) in pseudo_line_predictions(params, sub, sub_inputs, base).items():
        entry = dict(n=int(len(y)), pseudo_line='floor(last-10 mean) + 0.5', over_rate=round(float(y.mean()), 4),
                     model_raw=binary_scores(model_p, y), baseline=binary_scores(base_p, y),
                     model_raw_calibration=calibration(model_p, y), baseline_calibration=calibration(base_p, y))
        if slopes:
            cal = M.calibrate(model_p, slopes[market])
            entry.update(calibration_slope=round(slopes[market], 4), model=binary_scores(cal, y),
                         model_calibration=calibration(cal, y))
        markets[market] = entry
    results['pseudo_line_markets'] = markets
    return results


def pseudo_line_predictions(params, sub, sub_inputs, base, sims_sample=3000, sims=4000):
    """Per market: (model P(over), production-baseline P(over), outcome) at last-10 pseudo-lines."""
    out = {}
    rng = np.random.default_rng(7)
    sample = rng.choice(len(sub), size=min(sims_sample, len(sub)), replace=False)
    for market, keys in M.MARKET_STATS.items():
        hist = [np.array([sum(g[k] for k in keys) for g in base[(s['row']['GAME_ID'], s['row']['PLAYER_ID'])]]) for s in sub]
        lines = np.array([math.floor(h[-10:].mean()) + 0.5 for h in hist])
        actual = np.array([sum(s['row'][k] for k in keys) for s in sub])
        y = (actual > lines).astype(float)
        base_p = np.array([((h > l).sum() + 1) / (len(h) + 2) for h, l in zip(hist, lines)])
        if len(keys) == 1:
            model_p = over_probability(params, sub_inputs, keys[0], lines)
            idx = np.arange(len(sub))
        else:
            idx = sample
            mu, sd, _, rates = sub_inputs
            model_p = np.empty(len(idx))
            for j, i in enumerate(idx):
                draws, _ = M.simulate(params, {st: rates[st][i] for st in M.STATS}, mu[i], sd[i], n=sims, seed=int(i))
                model_p[j] = float((sum(draws[k] for k in keys) > lines[i]).mean())
        out[market] = (model_p, base_p[idx], y[idx])
    return out


def with_baseline(snaps, base, params):
    keep = [s for s in snaps if len(base.get((s['row']['GAME_ID'], s['row']['PLAYER_ID']), [])) >= 20]
    return keep, vector_inputs(params, keep)


def fit_calibration(params, valid, base):
    """One logit slope per market, fit on validation predictions (shrinks overconfident tails)."""
    from scipy.optimize import minimize_scalar
    sub, inputs = with_baseline(valid, base, params)
    slopes = {}
    for market, (p, _, y) in pseudo_line_predictions(params, sub, inputs, base).items():
        loss = lambda a: binary_scores(M.calibrate(p, a), y)['logloss']
        slopes[market] = float(minimize_scalar(loss, bounds=(0.3, 1.5), method='bounded').x)
    return slopes


def evaluate_games(game, rows, train_rows):
    home_rate = float(np.mean([r['home_pts'] > r['away_pts'] for r in train_rows]))
    p_home, pit_total, pit_margin, e_total, e_margin, naive_total = [], [], [], [], [], []
    cover80 = []
    for r in rows:
        matrix, mh, ma = M.score_matrix(game, r['features'])
        grid = np.arange(M.SCORE_MIN, M.SCORE_MAX + 1)
        H, A = np.meshgrid(grid, grid, indexing='ij')
        tot, mar = H + A, H - A
        at, am = r['home_pts'] + r['away_pts'], r['home_pts'] - r['away_pts']
        p_home.append(float(matrix[mar > 0].sum()))
        pit_total.append(float(matrix[tot < at].sum() + 0.5 * matrix[tot == at].sum()))
        pit_margin.append(float(matrix[mar < am].sum() + 0.5 * matrix[mar == am].sum()))
        mean_total, mean_margin = float((matrix * tot).sum()), float((matrix * mar).sum())
        e_total.append(at - mean_total)
        e_margin.append(am - mean_margin)
        naive_total.append(at - (r['features']['naive_home'] + r['features']['naive_away']))
        cover80.append(0.1 <= pit_total[-1] <= 0.9)
    y = np.array([r['home_pts'] > r['away_pts'] for r in rows], float)
    p = np.array(p_home)
    # Deployment starts at the opener: the first three weeks carry last season's states.
    first = min(r['game']['GAME_DATE'] for r in rows)
    early = np.array([M.day_of(r['game']['GAME_DATE']) < M.day_of(first) + timedelta(days=21) for r in rows])
    early_metrics = dict(games=int(early.sum()), moneyline_model=binary_scores(p[early], y[early]),
                         total_bias=float(np.mean(np.array(e_total)[early])),
                         total_rmse=float(np.sqrt(np.mean(np.square(np.array(e_total)[early])))),
                         margin_rmse=float(np.sqrt(np.mean(np.square(np.array(e_margin)[early])))))
    hist = lambda v: np.histogram(v, bins=10, range=(0, 1))[0].tolist()
    return dict(
        games=len(rows), home_win_rate_train=round(home_rate, 4),
        moneyline_model=binary_scores(p, y), moneyline_home_rate_baseline=binary_scores(np.full(len(y), home_rate), y),
        moneyline_calibration=calibration(p, y),
        total_rmse=float(np.sqrt(np.mean(np.square(e_total)))), total_bias=float(np.mean(e_total)),
        total_rmse_unadjusted_pace_efficiency=float(np.sqrt(np.mean(np.square(naive_total)))),
        margin_rmse=float(np.sqrt(np.mean(np.square(e_margin)))), margin_bias=float(np.mean(e_margin)),
        total_80pct_interval_coverage=float(np.mean(cover80)),
        total_pit_deciles=hist(pit_total), margin_pit_deciles=hist(pit_margin), first_three_weeks=early_metrics)


def write_markdown(e):
    """EVALUATION.md from evaluation.json, so reported numbers cannot drift from the data."""
    g, p = e['games'], e['players']
    f3 = g['first_three_weeks']
    lines = [f"# NBA v1 evaluation ({e['version']})", '',
             f"Generated {e['created_at'][:10]} by `scripts/nba/train.py` from `evaluation.json`. Test season: 2025–26 regular",
             'season, never used for fitting or selection. Hyperparameters were selected on 2024–25 with a 2023–24 fit;',
             'the evaluated model was then refit on 2023–24 (from December 1) and 2024–25. No sportsbook prices were used.', '',
             '## Games', '',
             f"{g['games']} test games. Home teams won {100 * g['home_win_rate_train']:.1f}% of training games.", '',
             '| Measure | NBA v1 | Baseline |', '|---|---|---|',
             f"| Moneyline log loss | {g['moneyline_model']['logloss']:.4f} | {g['moneyline_home_rate_baseline']['logloss']:.4f} (training home-win rate) |",
             f"| Moneyline Brier | {g['moneyline_model']['brier']:.4f} | {g['moneyline_home_rate_baseline']['brier']:.4f} |",
             f"| Total RMSE (points) | {g['total_rmse']:.2f} | {g['total_rmse_unadjusted_pace_efficiency']:.2f} (unadjusted pace × efficiency) |",
             f"| Total bias (actual − forecast) | {g['total_bias']:+.2f} | |",
             f"| Margin RMSE (points) | {g['margin_rmse']:.2f} | |",
             f"| 80% total interval coverage | {100 * g['total_80pct_interval_coverage']:.1f}% | 80% target |", '',
             f"First three weeks ({f3['games']} games, states carried from the prior season): moneyline log loss "
             f"{f3['moneyline_model']['logloss']:.4f}, total bias {f3['total_bias']:+.2f}, total RMSE {f3['total_rmse']:.2f}.", '',
             'Moneyline calibration (forecast → observed home-win rate, n):', '',
             ', '.join(f"{c['mean_forecast']:.2f} → {c['observed']:.2f} ({c['n']})" for c in g['moneyline_calibration']), '',
             f"Total PIT deciles (uniform is {g['games'] // 10} each): {g['total_pit_deciles']}. Margin: {g['margin_pit_deciles']}.", '',
             '## Players', '',
             f"{p['appearances']} test appearances by players with at least {M.MIN_APPEARANCES} earlier appearances; "
             f"{p['appearances_with_baseline']} also meet the production baseline's 20-game rule and form the comparison set.",
             'Forecasts are conditional on the player appearing (books void props for players who do not).', '',
             '### Count log loss (lower is better)', '',
             '| Stat | NBA v1 | Recency baseline (30-game mean, same dispersion) |', '|---|---|---|']
    for st in M.STATS:
        lines.append(f"| {st} | {p['count_logloss_model_on_baseline_subset'][st]:.4f} | {p['count_logloss_recency_baseline'][st]:.4f} |")
    lines += ['', '### Over/under at pseudo-lines', '',
              "Line = floor(last-10 average) + 0.5, so no pushes. These are diagnostics, not offered lines; real lines are",
              "sharper, so these numbers do not establish profitability. Baseline = the production smoothed hit rate.",
              'Combos are scored on a seeded random sample of 3,000 appearances (4,000 joint draws each).', '',
              '| Market | n | Over rate | Slope | v1 log loss | Raw v1 | Baseline | v1 Brier | Baseline Brier |',
              '|---|---|---|---|---|---|---|---|---|']
    for m, v in p['pseudo_line_markets'].items():
        lines.append(f"| {m.replace('player_', '')} | {v['n']} | {v['over_rate']:.3f} | {v.get('calibration_slope', 1):.3f} | "
                     f"{v['model']['logloss']:.4f} | {v['model_raw']['logloss']:.4f} | {v['baseline']['logloss']:.4f} | "
                     f"{v['model']['brier']:.4f} | {v['baseline']['brier']:.4f} |")
    lines += ['', 'Calibration of the calibrated v1 probabilities (forecast → observed, n):', '']
    for m, v in p['pseudo_line_markets'].items():
        lines.append(f"- {m.replace('player_', '')}: " + ', '.join(f"{c['mean_forecast']:.2f} → {c['observed']:.2f} ({c['n']})" for c in v['model_calibration']))
    lines += ['', f"First three weeks: {p['first_three_weeks']['appearances']} appearances; count log loss "
              + ', '.join(f"{k} {v:.3f}" for k, v in p['first_three_weeks']['count_logloss_model'].items()) + '.', '',
              '## Selection', '', 'Team grid (validation Gaussian NLL of regulation scores, best first): ' +
              '; '.join(f"half-life {c['half_life']}, carry {c['carry']}, prior {c['prior_games']} → {c['valid_nll']}" for c in e['team_selection'][:4]) + '.', '',
              'Player grid (sum of validation count log loss, best first): ' +
              '; '.join(f"rate half-life {c['rate_half_life']}, prior {c['prior_minutes']} min, opponent weight {c['opponent_weight']} → {c['valid_logloss_sum']}" for c in e['player_selection'][:4]) + '.', '',
              '## Data', ''] + [f"- `artifacts/nba/history/{k}` sha256 `{v}`" for k, v in e['data'].items()]
    (REPORT / 'EVALUATION.md').write_text('\n'.join(lines) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-selection', action='store_true', help='Reuse the saved hyperparameters')
    parser.add_argument('--report-only', action='store_true', help='Rewrite EVALUATION.md from evaluation.json')
    args = parser.parse_args(argv)
    if args.report_only:
        write_markdown(json.loads((REPORT / 'evaluation.json').read_text()))
        return 0
    history = load_all(season_types=M.STATE_TYPES)
    saved = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else None
    if args.skip_selection and saved:
        team_cfg, player_cfg = saved['team_config'], saved['players']['config']
        previous = json.loads((REPORT / 'evaluation.json').read_text())
        team_grid, player_grid = previous['team_selection'], previous['player_selection']
    else:
        team_cfg, team_grid = select_team(history)
        print('team config', team_cfg)
        player_cfg, player_grid = select_players(history, team_cfg)
        print('player config', player_cfg)

    base = baselines(history)
    # Calibration slopes: a 2023-24 fit predicting 2024-25 (never the test season).
    rows, overtime, _ = M.game_table(history, team_cfg)
    attach_means(rows, M.fit_game(with_regulation([r for r in rows if in_split(r, 'fit')]),
                                  [o for o in overtime if o['day'] < '2024-07-01']))
    snaps, _ = M.player_table(history, rows, (player_cfg['rate_half_life'],))
    fit_only = fit_players([s for s in snaps if in_split(s, 'fit')], player_cfg)
    slopes = fit_calibration(fit_only, [s for s in snaps if in_split(s, 'valid')], base)
    print('calibration slopes', {k: round(v, 3) for k, v in slopes.items()})

    # Evaluation: fit on 2023-24 (Dec+) and 2024-25, test on 2025-26.
    rows, overtime, _ = M.game_table(history, team_cfg)
    train_rows = with_regulation([r for r in rows if in_split(r, 'fit+valid')])
    game = M.fit_game(train_rows, [o for o in overtime if o['day'] < '2025-07-01'])
    attach_means(rows, game)
    snaps, _ = M.player_table(history, rows, (player_cfg['rate_half_life'],))
    players = fit_players([s for s in snaps if in_split(s, 'fit+valid')], player_cfg)
    test_games = [r for r in rows if in_split(r, 'test')]
    evaluation = dict(
        version=M.VERSION, created_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        protocol=__doc__.strip(), team_config=team_cfg, player_config=player_cfg,
        team_selection=team_grid, player_selection=player_grid,
        games=evaluate_games(game, test_games, train_rows),
        players=evaluate_players(players, [s for s in snaps if in_split(s, 'test')], base, slopes),
        data={p.name: sha256(p) for p in sorted(HISTORY.glob('*.json.gz'))})
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'evaluation.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    write_markdown(evaluation)

    # Production: same choices, refit on every regular season available.
    all_rows = with_regulation([r for r in rows if in_split(r, 'all')])
    game_all = M.fit_game(all_rows, overtime)
    attach_means(rows, game_all)
    snaps_all, _ = M.player_table(history, rows, (player_cfg['rate_half_life'],))
    players_all = fit_players([s for s in snaps_all if in_split(s, 'all')], player_cfg)
    players_all['calibration'] = slopes
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(dict(
        version=M.VERSION, trained_at=evaluation['created_at'], team_config=team_cfg, game=game_all, players=players_all,
        training=dict(regular_season_games=len(all_rows), through=max(r['game']['GAME_DATE'] for r in all_rows)),
        data=evaluation['data']), indent=1) + '\n')
    print(json.dumps(dict(games={k: v for k, v in evaluation['games'].items() if 'pit' not in k and 'calibration' not in k},
                          players={m: dict(model=v['model'], raw=v['model_raw'], baseline=v['baseline'], n=v['n'])
                                   for m, v in evaluation['players']['pseudo_line_markets'].items()}), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

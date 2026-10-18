#!/usr/bin/env python3
"""Evidence for the NBA v1 technical paper (FV-2026-05). Offline; no paid data.

Inputs: the committed ESPN history (artifacts/nba/history/), the v1 code and selected
hyperparameters (reports/nba-v1/evaluation.json), and DraftKings lines that ESPN embeds in
cached game summaries (data/nba/espn/summary/, gitignored; 2025-26 only). Output:
reports/nba-paper/evidence.json. Every comparison uses the same chronological protocol as
scripts/nba/train.py: select on 2024-25 with a 2023-24 fit, test once on 2025-26 with a fit
on 2023-24 (from December 1) and 2024-25. Intervals resample whole games (500 replicates).
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats as sps
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from nba import model as M  # noqa: E402
from nba import train as T  # noqa: E402
from nba.espn_history import HISTORY, load_all  # noqa: E402

OUT = ROOT / 'reports/nba-paper'
SUMMARIES = ROOT / 'data/nba/espn/summary'
REPS = 500
RNG = np.random.default_rng(20261018)
FOCUS = ('PTS', 'REB', 'AST', 'FG3M')


def r(x, d=5):
    return None if x is None else round(float(x), d)


def boot(diff, clusters, reps=REPS):
    """Mean of per-row differences with a whole-game percentile interval."""
    diff, clusters = np.asarray(diff, float), np.asarray(clusters)
    keys, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse)
    means = []
    for _ in range(reps):
        pick = RNG.integers(0, len(keys), len(keys))
        means.append(sums[pick].sum() / counts[pick].sum())
    return dict(mean=r(diff.mean(), 6), ci95=[r(np.percentile(means, 2.5), 6), r(np.percentile(means, 97.5), 6)])


def logloss_rows(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def no_vig(a, b):
    pa, pb = (100 / (a + 100) if a > 0 else -a / (-a + 100)), (100 / (b + 100) if b > 0 else -b / (-b + 100))
    return pa / (pa + pb)


def decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / -price)


# ------------------------------------------------------------------ games

def matrix_eval(params, rows):
    """Per-game: joint log score of the final, P(home), P(regulation tie), means, matrices."""
    out = []
    for row in rows:
        matrix, mh, ma = M.score_matrix(params, row['features'])
        h, a = row['home_pts'] - M.SCORE_MIN, row['away_pts'] - M.SCORE_MIN
        inside = 0 <= h < matrix.shape[0] and 0 <= a < matrix.shape[1]
        cov = np.array(params['cov'])
        sd = math.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        grid = np.arange(M.SCORE_MIN, M.SCORE_MAX + 1)
        H, A = np.meshgrid(grid, grid, indexing='ij')
        diff = (mh - ma)
        tie = float(sps.norm.cdf((0.5 - diff) / sd) - sps.norm.cdf((-0.5 - diff) / sd))
        out.append(dict(joint=-math.log(max(matrix[h, a], 1e-12)) if inside else -math.log(1e-12),
                        p_home=float(matrix[H > A].sum()), tie=tie, mean_total=float((matrix * (H + A)).sum()),
                        mean_margin=float((matrix * (H - A)).sum()), matrix=matrix))
    return out


def games(history, team_cfg, history_with_playoffs):
    rows, overtime, _ = M.game_table(history, team_cfg)
    train = T.with_regulation([x for x in rows if T.in_split(x, 'fit+valid')])
    test = [x for x in rows if T.in_split(x, 'test')]
    ot = [o for o in overtime if o['day'] < '2025-07-01']
    variants = {
        'pace_efficiency_only': M.fit_game(train, ot, ('naive_home', 'naive_away', 'neutral')),
        'v1': M.fit_game(train, ot),
    }
    y = np.array([x['home_pts'] > x['away_pts'] for x in test], float)
    home_rate = float(np.mean([x['home_pts'] > x['away_pts'] for x in train]))
    gid = [x['game']['GAME_ID'] for x in test]
    total = np.array([x['home_pts'] + x['away_pts'] for x in test])
    margin = np.array([x['home_pts'] - x['away_pts'] for x in test])
    first = min(x['game']['GAME_DATE'] for x in test)
    early = np.array([M.day_of(x['game']['GAME_DATE']) < M.day_of(first) + timedelta(days=21) for x in test])
    results, per = {}, {}
    for name, params in variants.items():
        ev = matrix_eval(params, test)
        per[name] = ev
        p = np.array([e['p_home'] for e in ev])
        results[name] = dict(
            joint_nll=r(np.mean([e['joint'] for e in ev])), moneyline_log_loss=r(logloss_rows(p, y).mean()),
            moneyline_brier=r(np.mean((p - y) ** 2)),
            total_rmse=r(np.sqrt(np.mean((total - [e['mean_total'] for e in ev]) ** 2)), 3),
            margin_rmse=r(np.sqrt(np.mean((margin - [e['mean_margin'] for e in ev]) ** 2)), 3),
            total_bias=r(np.mean(total - [e['mean_total'] for e in ev]), 3),
            early_total_bias=r(np.mean((total - [e['mean_total'] for e in ev])[early]), 3),
            predicted_regulation_tie=r(np.mean([e['tie'] for e in ev]), 4))
    results['home_rate'] = dict(moneyline_log_loss=r(logloss_rows(np.full(len(y), home_rate), y).mean()),
                                moneyline_brier=r(np.mean((home_rate - y) ** 2)))
    observed_ot = float(np.mean([(x['game'].get('PERIODS') or 4) > 4 for x in test]))
    paired = dict(
        v1_minus_pace_efficiency_joint=boot([a['joint'] - b['joint'] for a, b in zip(per['v1'], per['pace_efficiency_only'])], gid),
        v1_minus_home_rate_moneyline=boot(logloss_rows([e['p_home'] for e in per['v1']], y) - logloss_rows(np.full(len(y), home_rate), y), gid))
    # Playoff contamination: the same model with playoff and play-in games in the states.
    rows_p, overtime_p, _ = M.game_table(history_with_playoffs, team_cfg)
    train_p = T.with_regulation([x for x in rows_p if T.in_split(x, 'fit+valid')])
    test_p = [x for x in rows_p if T.in_split(x, 'test')]
    params_p = M.fit_game(train_p, [o for o in overtime_p if o['day'] < '2025-07-01'])
    ev_p = matrix_eval(params_p, test_p)
    tot_p = np.array([x['home_pts'] + x['away_pts'] for x in test_p])
    early_p = np.array([M.day_of(x['game']['GAME_DATE']) < M.day_of(first) + timedelta(days=21) for x in test_p])
    errs = tot_p - np.array([e['mean_total'] for e in ev_p])
    results['v1_with_playoff_states'] = dict(total_bias=r(errs.mean(), 3), early_total_bias=r(errs[early_p].mean(), 3),
                                             total_rmse=r(np.sqrt(np.mean(errs ** 2)), 3))
    coefficients = {side: dict(zip(['intercept'] + list(M.GAME_FEATURES),
                                   [r(b, 4) for b in [variants['v1'][side]['beta'][0]] +
                                    [b / s for b, s in zip(variants['v1'][side]['beta'][1:], variants['v1'][side]['scale'])]]))
                    for side in ('home', 'away')}
    return dict(test_games=len(test), early_games=int(early.sum()), home_win_rate_train=r(home_rate, 4),
                observed_overtime_rate=r(observed_ot, 4), candidates=results, paired=paired,
                coefficients_per_unit=coefficients, residual_cov=[[r(v, 2) for v in row] for row in variants['v1']['cov']],
                ), test, per['v1'], variants['v1']


def market(test, ev):
    """DraftKings lines ESPN embeds in final summaries (one snapshot, time not stated)."""
    rows = []
    for x, e in zip(test, ev):
        path = SUMMARIES / f"{x['game']['GAME_ID']}.json"
        if not path.exists():
            continue
        pc = (json.loads(path.read_text()).get('pickcenter') or [None])[0]
        if not pc:
            continue
        home, away = pc.get('homeTeamOdds') or {}, pc.get('awayTeamOdds') or {}
        if str(home.get('teamId')) != x['game']['HOME_TEAM_ID']:
            continue
        rows.append(dict(x=x, e=e, pc=pc, home=home, away=away))
    grid = np.arange(M.SCORE_MIN, M.SCORE_MAX + 1)
    H, A = np.meshgrid(grid, grid, indexing='ij')
    out, policy = {}, defaultdict(list)
    for kind in ('moneyline', 'spread', 'total'):
        model_p, market_p, y, gids, prices, dates = [], [], [], [], [], []
        for row in rows:
            x, e, pc, home, away = row['x'], row['e'], row['pc'], row['home'], row['away']
            hp, ap = x['home_pts'], x['away_pts']
            m = e['matrix']
            if kind == 'moneyline':
                if home.get('moneyLine') is None or away.get('moneyLine') is None:
                    continue
                q, mk, yy = e['p_home'], no_vig(home['moneyLine'], away['moneyLine']), float(hp > ap)
                pr = (home['moneyLine'], away['moneyLine'])
            elif kind == 'spread':
                line = pc.get('spread')  # home handicap
                if line is None or home.get('spreadOdds') is None or away.get('spreadOdds') is None:
                    continue
                cover, push = float(m[(H - A + line) > 0].sum()), float(m[(H - A + line) == 0].sum())
                if hp - ap + line == 0:
                    continue  # pushes are refunds; conditional non-push scoring
                q, mk, yy = cover / (1 - push), no_vig(home['spreadOdds'], away['spreadOdds']), float(hp - ap + line > 0)
                pr = (home['spreadOdds'], away['spreadOdds'])
            else:
                line = pc.get('overUnder')
                if line is None or pc.get('overOdds') is None or pc.get('underOdds') is None:
                    continue
                over, push = float(m[(H + A) > line].sum()), float(m[(H + A) == line].sum())
                if hp + ap == line:
                    continue
                q, mk, yy = over / (1 - push), no_vig(pc['overOdds'], pc['underOdds']), float(hp + ap > line)
                pr = (pc['overOdds'], pc['underOdds'])
            model_p.append(q); market_p.append(mk); y.append(yy); gids.append(x['game']['GAME_ID'])
            prices.append(pr); dates.append(x['game']['GAME_DATE'])
        model_p, market_p, y = np.array(model_p), np.array(market_p), np.array(y)
        lm, lk = logloss_rows(model_p, y), logloss_rows(market_p, y)
        # Out-of-sample blend: weight on the model fit on the earlier half, scored on the later half.
        order = np.argsort(dates)
        half = len(order) // 2
        a, b = order[:half], order[half:]
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        weights = np.linspace(0, 1, 21)
        blend = lambda w, idx: 1 / (1 + np.exp(-(w * logit(model_p[idx]) + (1 - w) * logit(market_p[idx]))))
        w_best = float(min(weights, key=lambda w: logloss_rows(blend(w, a), y[a]).mean()))
        out[kind] = dict(
            n=int(len(y)), model_log_loss=r(lm.mean()), market_log_loss=r(lk.mean()),
            model_minus_market=boot(lm - lk, gids), mean_abs_disagreement=r(np.mean(np.abs(model_p - market_p)), 4),
            disagreement_sd=r(np.std(model_p - market_p), 4),
            blend_weight_on_model_first_half=w_best,
            second_half=dict(market=r(lk[b].mean()), model=r(lm[b].mean()), blend=r(logloss_rows(blend(w_best, b), y[b]).mean())),
            model_calibration=T.calibration(model_p, y), market_calibration=T.calibration(market_p, y))
        # Hypothetical flat-stake policy at the listed prices (exploratory; not pre-registered).
        for threshold in (0.0, 0.025, 0.05):
            units, picks = [], []
            for q, pr, yy, g in zip(model_p, prices, y, gids):
                for side_p, price, won in ((q, pr[0], yy), (1 - q, pr[1], 1 - yy)):
                    if side_p - 1 / decimal(price) > threshold:
                        units.append(decimal(price) - 1 if won else -1.0)
                        picks.append(g)
            if units:
                ci = boot(units, picks)
                policy[kind].append(dict(threshold=threshold, bets=len(units), roi=r(np.mean(units), 4), roi_ci95=ci['ci95']))
            else:
                policy[kind].append(dict(threshold=threshold, bets=0))
    return dict(games_with_lines=len(rows), provider='DraftKings (via ESPN game summaries)', markets=out, policy=dict(policy))


# ------------------------------------------------------------------ players

def nb_ll(y, mean, k):
    mean = np.maximum(mean, 1e-9)
    return -sps.nbinom.logpmf(y, k, k / (k + mean))


def mixture_ll(y, rates, nodes, k):
    means = np.maximum(rates[:, None] * nodes, 1e-9)
    if k is None:
        lp = sps.poisson.logpmf(y[:, None], means)
    else:
        lp = sps.nbinom.logpmf(y[:, None], k, k / (k + means))
    return -(logsumexp(lp, axis=1) - math.log(nodes.shape[1]))


def players(history, team_cfg, player_cfg, slopes):
    rows, overtime, _ = M.game_table(history, team_cfg)
    game = M.fit_game(T.with_regulation([x for x in rows if T.in_split(x, 'fit+valid')]),
                      [o for o in overtime if o['day'] < '2025-07-01'])
    T.attach_means(rows, game)
    snaps, _ = M.player_table(history, rows, (player_cfg['rate_half_life'],))
    train = [s for s in snaps if T.in_split(s, 'fit+valid')]
    test = [s for s in snaps if T.in_split(s, 'test')]
    params = T.fit_players(train, player_cfg)
    base = T.baselines(history)
    sub = [s for s in test if len(base.get((s['row']['GAME_ID'], s['row']['PLAYER_ID']), [])) >= 20]
    gid = np.array([s['row']['GAME_ID'] for s in sub])
    mu, sd, nodes, rates = T.vector_inputs(params, sub)
    # Structural rates without the recent-average blend.
    struct = {st: np.array([M.rate_means(s, player_cfg, params['priors'], params['scale'])[st] for s in sub]) for st in M.STATS}
    # Direct per-game model: no minutes separation, EW per-game average (half-life 20) with a fitted shape.
    def per_game_mean(s, st):
        ew = s['rates'][player_cfg['rate_half_life']]
        games_w = ew['MIN'] / max(s['minutes']['m_long'], 1)  # recover the EW appearance weight
        return ew[st] / max(games_w, 1e-9)
    # m_long and the rate sums share the 20-appearance half-life, so MIN / m_long is the EW weight.
    assert player_cfg['rate_half_life'] == 20
    direct = {st: np.array([per_game_mean(s, st) for s in sub]) for st in M.STATS}
    direct_train = {st: np.array([per_game_mean(s, st) for s in train]) for st in M.STATS}
    count = {}
    direct_shapes = {}
    paired = {}
    for st in M.STATS:
        y = np.array([s['row'][st] for s in sub], float)
        yt = np.array([s['row'][st] for s in train], float)
        from scipy.optimize import minimize_scalar
        kd = math.exp(minimize_scalar(lambda lk: nb_ll(yt, np.maximum(direct_train[st], 0.02), math.exp(lk)).mean(),
                                      bounds=(math.log(0.5), math.log(2000)), method='bounded').x)
        hist_mean = np.array([max(np.mean([g[st] for g in base[(s['row']['GAME_ID'], s['row']['PLAYER_ID'])]]), 0.05) for s in sub])
        ll = dict(
            recency_30=nb_ll(y, hist_mean, params['shape'][st]),
            direct_per_game=nb_ll(y, np.maximum(direct[st], 0.02), kd),
            minutes_x_rate_poisson=mixture_ll(y, rates[st], nodes, None),
            minutes_x_rate_structural=mixture_ll(y, struct[st], nodes, params['shape'][st]),
            v1=mixture_ll(y, rates[st], nodes, params['shape'][st]))
        count[st] = {k: r(v.mean()) for k, v in ll.items()}
        direct_shapes[st] = r(kd, 3)
        if st in FOCUS:
            paired[st] = dict(v1_minus_recency=boot(ll['v1'] - ll['recency_30'], gid),
                              v1_minus_direct=boot(ll['v1'] - ll['direct_per_game'], gid),
                              v1_minus_poisson=boot(ll['v1'] - ll['minutes_x_rate_poisson'], gid),
                              blend_effect=boot(ll['v1'] - ll['minutes_x_rate_structural'], gid))
    # Distribution diagnostics: zero rates and 80% central interval coverage.
    dist = {}
    q = (np.arange(nodes.shape[1]) + 0.5) / nodes.shape[1]
    for st in M.STATS:
        y = np.array([s['row'][st] for s in sub])
        k = params['shape'][st]
        means = np.maximum(rates[st][:, None] * nodes, 1e-9)
        p0 = sps.nbinom.pmf(0, k, k / (k + means)).mean(axis=1)
        cdf = lambda v: sps.nbinom.cdf(v, k, k / (k + means)).mean(axis=1)
        lo = np.zeros(len(y)); hi = np.zeros(len(y))
        for v in range(0, 80):
            c = cdf(v)
            lo = np.where((c < 0.1), v + 1, lo)
            hi = np.where((c < 0.9), v + 1, hi)
        dist[st] = dict(observed_zero=r(np.mean(y == 0), 4), predicted_zero=r(p0.mean(), 4),
                        interval80_coverage=r(np.mean((y >= lo) & (y <= hi)), 4))
    # Minutes model.
    actual_min = np.array([float(s['row']['MIN']) for s in sub])
    naive_min = np.array([s['minutes']['m_short'] for s in sub])
    minutes = dict(v1_mae=r(np.mean(np.abs(actual_min - mu)), 3), v1_rmse=r(np.sqrt(np.mean((actual_min - mu) ** 2)), 3),
                   recent5_mae=r(np.mean(np.abs(actual_min - naive_min)), 3),
                   recent5_rmse=r(np.sqrt(np.mean((actual_min - naive_min) ** 2)), 3),
                   coefficients_per_unit=dict(zip(['intercept'] + list(M.MINUTE_FEATURES),
                       [r(params['minutes']['beta'][0], 3)] + [r(b / s, 4) for b, s in zip(params['minutes']['beta'][1:], params['minutes']['scale'])])),
                   residual_sd_by_projection=dict(zip([str(c) for c in params['minutes_sd']['centers']], [r(v, 2) for v in params['minutes_sd']['sds']])))
    # Joint vs independent combos at pseudo-lines, on a seeded sample.
    sample = RNG.choice(len(sub), size=min(3000, len(sub)), replace=False)
    combos = {}
    independent = dict(params, correlation=np.eye(len(M.STATS)).tolist())
    for market, keys in M.MARKET_STATS.items():
        if len(keys) == 1:
            continue
        hist = [np.array([sum(g[k] for k in keys) for g in base[(sub[i]['row']['GAME_ID'], sub[i]['row']['PLAYER_ID'])]]) for i in sample]
        lines = np.array([math.floor(h[-10:].mean()) + 0.5 for h in hist])
        actual = np.array([sum(sub[i]['row'][k] for k in keys) for i in sample])
        y = (actual > lines).astype(float)
        pj, pi, var_j, var_i = [], [], [], []
        for j, i in enumerate(sample):
            rr = {st: rates[st][i] for st in M.STATS}
            dj, _ = M.simulate(params, rr, mu[i], sd[i], n=4000, seed=int(i))
            # Independent marginals: separate minutes draws per stat and no copula.
            di = {}
            for t, st in enumerate(keys):
                d1, _ = M.simulate(independent, rr, mu[i], sd[i], n=4000, seed=int(i) * 31 + t)
                di[st] = d1[st]
            sj, si = sum(dj[k] for k in keys), sum(di[k] for k in keys)
            pj.append(float((sj > lines[j]).mean())); pi.append(float((si > lines[j]).mean()))
            var_j.append(sj.var()); var_i.append(si.var())
        pj, pi = M.calibrate(np.array(pj), slopes[market]), M.calibrate(np.array(pi), slopes[market])
        g = gid[sample]
        mean_pred = np.array([sum(rates[k][i] * mu[i] for k in keys) for i in sample])
        combos[market] = dict(n=len(sample), joint_log_loss=r(logloss_rows(pj, y).mean()), independent_log_loss=r(logloss_rows(pi, y).mean()),
                              joint_minus_independent=boot(logloss_rows(pj, y) - logloss_rows(pi, y), g),
                              observed_residual_variance=r(np.var(actual - mean_pred), 3),
                              joint_predicted_variance=r(np.mean(var_j), 3), independent_predicted_variance=r(np.mean(var_i), 3))
    return dict(test_appearances=len(test), comparison_appearances=len(sub), count_log_loss=count, direct_shapes=direct_shapes, paired=paired,
                distribution=dist, minutes=minutes, combos=combos, shapes={k: r(v, 3) for k, v in params['shape'].items()},
                stack={k: [r(v, 4) for v in vals] for k, vals in params['stack'].items()}), (snaps, params)


def matchups(history, team_cfg, player_cfg):
    """Shrunk player histories over a fixed v1 forecast (the matchup paper's method)."""
    rows, overtime, _ = M.game_table(history, team_cfg)
    game = M.fit_game(T.with_regulation([x for x in rows if T.in_split(x, 'fit')]), [o for o in overtime if o['day'] < '2024-07-01'])
    T.attach_means(rows, game)
    snaps, _ = M.player_table(history, rows, (player_cfg['rate_half_life'],))
    fitted = T.fit_players([s for s in snaps if T.in_split(s, 'fit')], player_cfg)
    scored = [s for s in snaps if s['row']['SEASON_TYPE'] == 'regular' and s['row']['GAME_DATE'] >= '2023-12-01']
    scored.sort(key=lambda s: (s['row']['GAME_DATE'], s['row']['GAME_ID']))
    mu, sd, nodes, rates = T.vector_inputs(fitted, scored)
    means = {st: (rates[st][:, None] * nodes).mean(axis=1) for st in FOCUS}
    gid = np.array([s['row']['GAME_ID'] for s in scored])
    season = np.array([M.season_key(M.day_of(s['row']['GAME_DATE'])) for s in scored])
    rest_b2b = np.array([s['minutes']['b2b'] > 0 for s in scored])
    groups = {
        'venue': np.array(['home' if s['row']['HOME'] else 'away' for s in scored]),
        'opponent': np.array([s['row']['OPP_TEAM_ID'] for s in scored]),
        'back_to_back': np.where(rest_b2b, 'b2b', 'rested'),
    }
    pid = np.array([s['row']['PLAYER_ID'] for s in scored])
    days = np.array([s['row']['GAME_DATE'] for s in scored])
    grid = [1, 3, 10, 30, 100, 300, math.inf]
    out = {}
    for st in FOCUS:
        y = np.array([s['row'][st] for s in scored], float)
        E = means[st]
        k_shape = fitted['shape'][st]
        # Earlier-game sums per player (and per split), excluding the same date.
        def prior_sums(keys):
            A, B = np.zeros(len(y)), np.zeros(len(y))
            acc_a, acc_e = defaultdict(float), defaultdict(float)
            i = 0
            while i < len(y):
                j = i
                while j < len(y) and days[j] == days[i]:
                    j += 1
                for t in range(i, j):
                    A[t], B[t] = acc_a[keys[t]], acc_e[keys[t]]
                for t in range(i, j):
                    acc_a[keys[t]] += y[t]; acc_e[keys[t]] += E[t]
                i = j
            return A, B
        a_all, e_all = prior_sums(list(pid))
        valid, test = season == 2024, season == 2025
        def score(mean, mask):
            return nb_ll(y[mask], np.maximum(mean[mask], 0.01), k_shape).mean()
        def choose(build):
            return min(grid, key=lambda k: score(build(k), valid))
        k_all = choose(lambda k: E * (np.ones(len(y)) if not math.isfinite(k) else (a_all + k) / (e_all + k)))
        r_all = np.ones(len(y)) if not math.isfinite(k_all) else (a_all + k_all) / (e_all + k_all)
        result = dict(k_overall=k_all if math.isfinite(k_all) else 'not used',
                      baseline_test_ll=r(score(E, test)), overall_test_ll=r(score(E * r_all, test)),
                      overall_minus_baseline=boot(nb_ll(y[test], np.maximum(E * r_all, 0.01)[test], k_shape) - nb_ll(y[test], np.maximum(E, 0.01)[test], k_shape), gid[test]))
        for name, labels in groups.items():
            a_s, e_s = prior_sums([f'{p}|{l}' for p, l in zip(pid, labels)])
            def build(k, a_s=a_s, e_s=e_s):
                split = r_all if not math.isfinite(k) else (a_s + k * r_all) / (e_s + k)
                return E * split
            k_s = choose(build)
            mean_s = build(k_s)
            result[name] = dict(k=k_s if math.isfinite(k_s) else 'not used', test_ll=r(score(mean_s, test)),
                                minus_overall=boot(nb_ll(y[test], np.maximum(mean_s, 0.01)[test], k_shape) -
                                                   nb_ll(y[test], np.maximum(E * r_all, 0.01)[test], k_shape), gid[test]))
            # The k grid's best finite value, scored on test, shows the effect even if not selected.
            k_f = min([k for k in grid if math.isfinite(k)], key=lambda k: score(build(k), valid))
            result[name]['best_finite_k'] = k_f
            result[name]['best_finite_minus_overall'] = boot(nb_ll(y[test], np.maximum(build(k_f), 0.01)[test], k_shape) -
                                                            nb_ll(y[test], np.maximum(E * r_all, 0.01)[test], k_shape), gid[test])
        # League-level back-to-back effect on this stat relative to the forecast.
        result['b2b_actual_over_forecast'] = r(y[test & rest_b2b].sum() / E[test & rest_b2b].sum(), 4)
        result['rested_actual_over_forecast'] = r(y[test & ~rest_b2b].sum() / E[test & ~rest_b2b].sum(), 4)
        out[st] = result
    return dict(baseline='nba-v1 player model fit on 2023-24 (December onward); k selected on 2024-25; scored on 2025-26',
                k_grid=['inf' if not math.isfinite(k) else k for k in grid], stats=out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((ROOT / 'reports/nba-v1/evaluation.json').read_text())
    model = json.loads((ROOT / 'artifacts/nba/model/nba-v1.json').read_text())
    team_cfg, player_cfg = evaluation['team_config'], evaluation['player_config']
    history = load_all(season_types=M.STATE_TYPES)
    with_playoffs = load_all(season_types=None)
    print('games...')
    game_results, test, ev, _ = games(history, team_cfg, with_playoffs)
    print('market...')
    market_results = market(test, ev)
    print('players...')
    player_results, _ = players(history, team_cfg, player_cfg, model['players']['calibration'])
    print('matchups...')
    matchup_results = matchups(history, team_cfg, player_cfg)
    manifest = json.loads((HISTORY / 'manifest.json').read_text())
    data_counts = dict(seasons={s: dict(games=v['games'], by_type=v['games_by_type'], player_rows=v['player_rows'],
                                        missing=v['missing_games']) for s, v in manifest['seasons'].items()})
    evidence = dict(paper='FV-2026-05', model_version=model['version'],
                    created_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    protocol=__doc__.strip(), team_config=team_cfg, player_config=player_cfg,
                    selection=dict(team=evaluation['team_selection'], player=evaluation['player_selection']),
                    calibration_slopes={k: r(v, 4) for k, v in model['players']['calibration'].items()},
                    v1_evaluation=dict(games=evaluation['games'], players=evaluation['players']),
                    data=data_counts, games=game_results, market=market_results, players=player_results,
                    matchups=matchup_results,
                    hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [*sorted(HISTORY.glob('*.json.gz')), ROOT / 'artifacts/nba/model/nba-v1.json',
                             ROOT / 'reports/nba-v1/evaluation.json']})
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=1, default=lambda o: o if not isinstance(o, float) else r(o)) + '\n')
    print('wrote', OUT / 'evidence.json')


if __name__ == '__main__':
    main()

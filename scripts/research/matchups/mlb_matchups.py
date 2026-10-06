"""MLB: does batter-vs-pitcher history, a home/away split or a pitcher's history against one
team add anything to a forecast that already knows the batter, the pitcher, the park and the venue?

Research only. Plate-appearance data from Retrosheet event files (Chadwick Bureau mirror):
    The information used here was obtained free of charge from and is copyrighted by
    Retrosheet. Interested parties may contact Retrosheet at "www.retrosheet.org".

Baseline, per plate appearance and outcome (hit, strikeout, walk/HBP, home run): the batter's and
pitcher's decayed, shrunk rates combined by odds ratio against the league rate, times a shrunk park
factor and the league home factor. This mirrors what the production MLB model knows (batter and
starter rates, park, home) but works per plate appearance so a matchup is exact. Platoon (batter
side vs pitcher hand) is tested as one more addition, because the production model has none.

Constants are fitted on 2012-2019, shrinkage strengths chosen on 2021-2022, and everything is
scored once on 2023-2025. History accumulates from 2012.

    python scripts/research/matchups/mlb_matchups.py --retro DIR --out reports/matchups/mlb.json
"""
import argparse
from collections import defaultdict
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INF, cluster_ci, fmt_k, weight_after  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUTCOMES = ['hit', 'k', 'bb', 'hr']
PRIOR_PA = {'hit': 300, 'k': 100, 'bb': 150, 'hr': 400}       # shrinkage of player rates to league
HALF_LIFE = {'bat': 1000, 'pit': 1500}                          # decay, in the player's own PA
GRID = [10, 30, 100, 300, 1000, INF]                             # prior strength, in PA
TRAIN, VALID, TEST = range(2012, 2020), range(2021, 2023), range(2023, 2026)
SKIP = ('NP', 'SB', 'CS', 'PO', 'DI', 'OA', 'WP', 'PB', 'BK', 'FLE')


def classify(event):
    """Plate-appearance outcome from a Retrosheet event, or None for non-PA plays."""
    basic = event.split('.')[0].split('/')[0]
    if basic.startswith(SKIP):
        return None
    if basic.startswith('HP'):
        return dict(hit=0, k=0, bb=1, hr=0)
    if basic.startswith('HR') or re.fullmatch(r'H\d*', basic):
        return dict(hit=1, k=0, bb=0, hr=1)
    if re.fullmatch(r'S\d*', basic) or re.fullmatch(r'D\d*', basic) or basic == 'DGR' or re.fullmatch(r'T\d*', basic):
        return dict(hit=1, k=0, bb=0, hr=0)
    if basic.startswith('K'):
        return dict(hit=0, k=1, bb=0, hr=0)
    if basic.startswith(('W', 'IW')) or basic == 'I':
        return dict(hit=0, k=0, bb=1, hr=0)
    if basic.startswith('C') and not basic.startswith('CS'):
        return None   # catcher's interference: not an at-bat outcome we model
    return dict(hit=0, k=0, bb=0, hr=0)


def parse(retro):
    """Every regular-season plate appearance, 2012 onward, in date order."""
    hands = {}
    for path in glob.glob(os.path.join(retro, '*', '*.ROS')):
        for line in open(path, encoding='latin-1'):
            p = line.strip().split(',')
            if len(p) >= 5:
                hands[p[0]] = (p[3], p[4])
    pas = []
    for path in sorted(glob.glob(os.path.join(retro, '*', '*.EV?'))):
        game = None
        for line in open(path, encoding='latin-1'):
            p = line.rstrip('\n').split(',')
            tag = p[0]
            if tag == 'id':
                game = dict(id=p[1], pitcher={}, badj={}, padj=None)
            elif tag == 'info' and p[1] in ('visteam', 'hometeam', 'site', 'date'):
                game[p[1]] = p[2]
            elif tag in ('start', 'sub'):
                if p[-1].strip() == '1':
                    game['pitcher'][int(p[3])] = p[1]
            elif tag == 'badj':
                game['badj'][p[1]] = p[2]
            elif tag == 'padj':
                game['padj'] = (p[1], p[2])
            elif tag == 'play':
                out = classify(p[6])
                if out is None:
                    continue
                bat_team = int(p[2])
                batter, pitcher = p[3], game['pitcher'].get(1 - bat_team)
                if pitcher is None:
                    continue
                throws = hands.get(pitcher, ('?', '?'))[1]
                if game['padj'] and game['padj'][0] == pitcher:
                    throws = game['padj'][1]
                bats = game['badj'].get(batter) or hands.get(batter, ('?', '?'))[0]
                if bats == 'B':
                    bats = 'L' if throws == 'R' else 'R'
                date = game['date'].replace('/', '-')
                pas.append((date, int(date[:4]), game['id'], batter, pitcher, bats, throws, bat_team == 1, game['site'],
                            game['hometeam'] if bat_team == 0 else game['visteam'], game['visteam'] if bat_team == 0 else game['hometeam'],
                            out['hit'], out['k'], out['bb'], out['hr']))
    pas.sort(key=lambda r: (r[0], r[2]))
    cols = ['date', 'season', 'game', 'batter', 'pitcher', 'bats', 'throws', 'home', 'park', 'pitch_team', 'bat_team'] + OUTCOMES
    return {c: np.array([r[i] for r in pas]) for i, c in enumerate(cols)}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def baseline(d):
    """Walk-forward odds-ratio expectations; per-PA rates are read before the day's PAs are added."""
    n = len(d['date'])
    rate = {o: np.zeros((n, 3)) for o in OUTCOMES}   # batter, pitcher, league rate at the time
    bat = defaultdict(lambda: np.zeros(len(OUTCOMES) + 1))
    pit = defaultdict(lambda: np.zeros(len(OUTCOMES) + 1))
    league = np.array([.235, .21, .09, .03, 1.]) * 50000
    db, dp, dl = .5 ** (1 / HALF_LIFE['bat']), .5 ** (1 / HALF_LIFE['pit']), .5 ** (1 / 60000)
    ys = np.stack([d[o] for o in OUTCOMES], axis=1).astype(float)
    dates = d['date']
    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        lg = league[:-1] / league[-1]
        for t in range(i, j):
            b, p = bat[d['batter'][t]], pit[d['pitcher'][t]]
            for q, o in enumerate(OUTCOMES):
                s = PRIOR_PA[o]
                rate[o][t] = [(b[q] + s * lg[q]) / (b[-1] + s), (p[q] + s * lg[q]) / (p[-1] + s), lg[q]]
        for t in range(i, j):
            for state, decay in ((bat[d['batter'][t]], db), (pit[d['pitcher'][t]], dp)):
                state *= decay
                state[:-1] += ys[t]
                state[-1] += 1
            league *= dl
            league[:-1] += ys[t]
            league[-1] += 1
        i = j
    return {o: 1 / (1 + np.exp(-(logit(rate[o][:, 0]) + logit(rate[o][:, 1]) - logit(rate[o][:, 2])))) for o in OUTCOMES}


def prior_sums(d, expected, keyfn, names):
    """For each PA: actual and expected totals of earlier days for its key, per outcome."""
    n = len(d['date'])
    a = {o: np.zeros(n) for o in OUTCOMES}
    e = {o: np.zeros(n) for o in OUTCOMES}
    cnt = np.zeros(n)
    acc = defaultdict(lambda: np.zeros(2 * len(OUTCOMES) + 1))
    keys = keyfn(d)
    ys = np.stack([d[o] for o in OUTCOMES], axis=1).astype(float)
    es = np.stack([expected[o] for o in OUTCOMES], axis=1)
    dates = d['date']
    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        for t in range(i, j):
            s = acc[keys[t]]
            for q, o in enumerate(OUTCOMES):
                a[o][t], e[o][t] = s[q], s[len(OUTCOMES) + q]
            cnt[t] = s[-1]
        for t in range(i, j):
            s = acc[keys[t]]
            s[:len(OUTCOMES)] += ys[t]
            s[len(OUTCOMES):-1] += es[t]
            s[-1] += 1
        i = j
    return a, e, cnt


def shrink(a, e, k_pa, p, prior=1.):
    """(actual + k*prior) / (expected + k), with k in PA converted to expected events."""
    if k_pa == INF:
        return np.full_like(p, prior) if np.ndim(prior) == 0 else prior
    k = k_pa * p
    return (a + k * prior) / (e + k)


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--retro', type=Path, required=True)
    a.add_argument('--out', type=Path, default=ROOT / 'reports/matchups/mlb.json')
    args = a.parse_args()
    d = parse(args.retro)
    n = len(d['date'])
    print('plate appearances', n, flush=True)
    season = d['season']
    train, valid, test = np.isin(season, TRAIN), np.isin(season, VALID), np.isin(season, TEST)
    raw = baseline(d)
    print('baseline done', flush=True)

    # Park factor: shrunk ratio of actual to expected at the park, earlier days only (k = 3000 PA).
    pa, pe, _ = prior_sums(d, raw, lambda d: d['park'], None)
    with_park = {o: np.clip(raw[o] * shrink(pa[o], pe[o], 3000, raw[o]), 1e-4, .9) for o in OUTCOMES}
    home = d['home']
    # Home/away and platoon factors are gaps around 1 (each group's actual/expected over the overall
    # ratio), so neither absorbs a level bias in the baseline.
    lvl = {o: d[o][train].sum() / with_park[o][train].sum() for o in OUTCOMES}
    hf = {o: {h: float(d[o][train & (home == h)].sum() / with_park[o][train & (home == h)].sum() / lvl[o]) for h in (True, False)} for o in OUTCOMES}
    base = {o: np.clip(with_park[o] * np.where(home, hf[o][True], hf[o][False]), 1e-4, .9) for o in OUTCOMES}
    same = d['bats'] == d['throws']
    lvl = {o: d[o][train].sum() / base[o][train].sum() for o in OUTCOMES}
    pf = {o: {s: float(d[o][train & (same == s)].sum() / base[o][train & (same == s)].sum() / lvl[o]) for s in (True, False)} for o in OUTCOMES}
    platoon = {o: np.clip(base[o] * np.where(same, pf[o][True], pf[o][False]), 1e-4, .9) for o in OUTCOMES}
    print('league factors done', flush=True)

    # Residual histories against the platoon-aware baseline (the strongest league-level model).
    sums = {
        'overall': prior_sums(d, platoon, lambda d: d['batter'], None),
        'venue': prior_sums(d, platoon, lambda d: list(zip(d['batter'], d['home'])), None),
        'bvp': prior_sums(d, platoon, lambda d: list(zip(d['batter'], d['pitcher'])), None),
        'platoon_own': prior_sums(d, platoon, lambda d: list(zip(d['batter'], d['throws'])), None),
        'pvt': prior_sums(d, platoon, lambda d: list(zip(d['pitcher'], d['bat_team'])), None),
        'pitcher': prior_sums(d, platoon, lambda d: d['pitcher'], None),
    }
    print('histories done', flush=True)

    def factors(o, k):
        # k[component][outcome]: each outcome has its own strength (hits are far noisier than strikeouts).
        p = platoon[o]
        overall = shrink(sums['overall'][0][o], sums['overall'][1][o], k['overall'][o], p)
        pitcher = shrink(sums['pitcher'][0][o], sums['pitcher'][1][o], k['pitcher'][o], p)
        out = dict(overall=overall * pitcher)
        for name, prior_from in [('venue', overall), ('bvp', overall), ('platoon_own', overall), ('pvt', pitcher)]:
            sa, se, _ = sums[name]
            out[name] = shrink(sa[o], se[o], k[name][o], p, prior_from) / prior_from
        out['bvp_naive'] = shrink(sums['bvp'][0][o], sums['bvp'][1][o], k['bvp'][o], p)
        return out

    def variants(o, f):
        b = platoon[o]
        return {
            'production-like (rates, park, home)': base[o],
            '+ platoon (bat side vs pitcher hand)': b,
            '+ player overall (control)': b * f['overall'],
            '+ batter home/away split': b * f['overall'] * f['venue'],
            '+ batter vs this pitcher': b * f['overall'] * f['bvp'],
            '+ pitcher vs this team': b * f['overall'] * f['pvt'],
            "+ batter's own platoon split": b * f['overall'] * f['platoon_own'],
            'naive: platoon + batter vs pitcher only': b * f['bvp_naive'],
        }

    # Choose k on 2021-2022, per outcome, in order: player overall, pitcher overall, then each split.
    k = {c: {o: INF for o in OUTCOMES} for c in ('overall', 'pitcher', 'venue', 'bvp', 'platoon_own', 'pvt')}
    selection = []
    for comp, variant in [('overall', '+ player overall (control)'), ('pitcher', '+ player overall (control)'), ('venue', '+ batter home/away split'),
                          ('bvp', '+ batter vs this pitcher'), ('pvt', '+ pitcher vs this team'), ('platoon_own', "+ batter's own platoon split")]:
        for o in OUTCOMES:
            trials = {}
            for c in GRID:
                kk = {x: dict(v) for x, v in k.items()}
                kk[comp][o] = c
                trials[c] = float(log_loss(np.clip(variants(o, factors(o, kk))[variant][valid], 1e-4, .9), d[o][valid]).mean())
            k[comp][o] = min(trials, key=trials.get)
            selection.append(dict(component=comp, outcome=o, chosen_pa_of_prior=fmt_k(k[comp][o]), validation_log_loss={fmt_k(c): round(v, 6) for c, v in trials.items()}))
            print(selection[-1], flush=True)

    groups = d['game'][test]
    results = {}
    for o in OUTCOMES:
        f = factors(o, k)
        v = {name: np.clip(p, 1e-4, .9) for name, p in variants(o, f).items()}
        ll = {name: log_loss(p[test], d[o][test]) for name, p in v.items()}
        base_ll, ctrl = ll['production-like (rates, park, home)'], ll['+ player overall (control)']
        results[o] = {}
        for name, x in ll.items():
            row = dict(log_loss=round(float(x.mean()), 6), vs_production=cluster_ci(x - base_ll, groups))
            if name.startswith('+ batter') or name.startswith('+ pitcher vs'):
                row['vs_control'] = cluster_ci(x - ctrl, groups)
            results[o][name] = row
        # Where matchup history is deepest: 25+ earlier PA between this batter and pitcher.
        deep = test & (sums['bvp'][2] >= 25)
        results[o]['_deep_bvp_25pa'] = dict(rows=int(deep.sum()), vs_control=cluster_ci(
            log_loss(v['+ batter vs this pitcher'][deep], d[o][deep]) - log_loss(v['+ player overall (control)'][deep], d[o][deep]), d['game'][deep]))
    # Matchup streaks: batter-pitcher pairs with 20+ earlier PA and 40%+ more (or fewer) hits than expected.
    ctrl_hit = np.clip(variants('hit', factors('hit', k))['+ player overall (control)'], 1e-4, .9)
    ratio = np.where(sums['bvp'][1]['hit'] > 0, sums['bvp'][0]['hit'] / np.maximum(sums['bvp'][1]['hit'], 1e-9), 1)
    streaks = []
    for label, mask in [('every pair with 20+ PA (reference)', sums['bvp'][2] >= 20),
                        ('20+ PA vs this pitcher, hits 40%+ above expectation', (sums['bvp'][2] >= 20) & (ratio >= 1.4)),
                        ('20+ PA vs this pitcher, hits 30%+ below expectation', (sums['bvp'][2] >= 20) & (ratio <= .7))]:
        m = mask & test
        streaks.append(dict(group=label, plate_appearances=int(m.sum()), prior_ratio=round(float(ratio[m].mean()), 3),
                            next_pa_hits_actual_over_forecast=round(float(d['hit'][m].sum() / ctrl_hit[m].sum()), 3)))
    # Aaron Judge's longest matchups: what the history said and what happened next (2023-2025).
    judge = test & (d['batter'] == 'judga001') & (sums['bvp'][2] >= 15)
    story = dict(plate_appearances=int(judge.sum()), prior_bvp_hit_ratio=round(float(ratio[judge].mean()), 3) if judge.any() else None,
                 hits_actual=int(d['hit'][judge].sum()), hits_forecast_without_bvp=round(float(ctrl_hit[judge].sum()), 1))
    report = dict(source='Retrosheet event files 2012-2025 via Chadwick Bureau (regular season)', plate_appearances=int(n),
                  seasons=dict(train='2012-2019', validation='2021-2022', test='2023-2025'),
                  league_home_factor={o: {('home' if h else 'away'): round(x, 4) for h, x in v.items()} for o, v in hf.items()},
                  platoon_factor={o: {('same hand' if s else 'opposite hand'): round(x, 4) for s, x in v.items()} for o, v in pf.items()},
                  k_pa={c: {o: fmt_k(x) for o, x in v.items()} for c, v in k.items()}, selection=selection,
                  weight_bvp_after_30_pa={o: round(weight_after(30, k['bvp'][o]), 3) for o in OUTCOMES},
                  weight_venue_after_600_pa={o: round(weight_after(600, k['venue'][o]), 3) for o in OUTCOMES},
                  test=dict(plate_appearances=int(test.sum()), results=results), matchup_streaks=streaks, aaron_judge_15pa_matchups=story)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(json.dumps({o: {nm: (r['log_loss'], round(r['vs_production']['mean'], 6)) for nm, r in v.items() if not nm.startswith('_')} for o, v in results.items()}, indent=1))
    print(json.dumps(streaks), json.dumps(story))


if __name__ == '__main__':
    main()

"""NFL player props: do home/away splits or a player's history against one opponent help?

Research only. The baseline follows the production recipe in make_player_prop_params.py:
recent form (EWMA of the last 4 games, alpha 0.4) blended with a decayed career mean shrunk to the
position average, times the opponent-defense rating (yards allowed, the production z-score
mapping) and the fixed home/away multipliers. Blend weights and spreads are fitted on 2012-2021,
shrinkage strengths are chosen on 2022-2023, and everything is scored once on 2024 through
2026 week 4. Regular season only.

    python scripts/research/matchups/nfl_matchups.py --data DIR --out reports/matchups/nfl.json

DIR holds nflverse `stats_player_week_<year>.parquet` (as week_<year>.parquet) and `games.csv`.
"""
import argparse
from collections import defaultdict, deque
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INF, Ledger, shrunk, cluster_ci, brier, fmt_k, weight_after  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# market: stat column, role-volume column, eligible positions, defense side, volume floor (EWMA)
MARKETS = {
    'pass_yds': ('passing_yards', 'attempts', {'QB'}, 'pass', 20),
    'pass_attempts': ('attempts', 'attempts', {'QB'}, 'pass', 20),
    'completions': ('completions', 'attempts', {'QB'}, 'pass', 20),
    'rush_yds': ('rushing_yards', 'carries', {'RB'}, 'rush', 8),
    'rush_attempts': ('carries', 'carries', {'RB'}, 'rush', 8),
    'recv_yds': ('receiving_yards', 'targets', {'WR', 'TE', 'RB'}, 'pass', 3.5),
    'receptions': ('receptions', 'targets', {'WR', 'TE', 'RB'}, 'pass', 3.5),
    # Touchdowns and interceptions get no opponent-defense rating in production.
    'pass_tds': ('passing_tds', 'attempts', {'QB'}, None, 20),
    'pass_interceptions': ('passing_interceptions', 'attempts', {'QB'}, None, 20),
    'anytime_td': ('total_tds', 'touches', {'RB', 'WR', 'TE'}, None, 5),
}
# Production's fixed multipliers (make_player_prop_params.HOME_AWAY_MULTIPLIERS); none for attempts/completions.
FIXED_HOME = {'pass_yds': 1.06, 'rush_yds': 1.04, 'rush_attempts': 1.02, 'recv_yds': 1.06, 'receptions': 1.03,
              'pass_tds': 1.06, 'pass_interceptions': .95, 'anytime_td': 1.05}
RENAME = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA'}
GRID = [1, 3, 10, 30, 100, 300, INF]     # prior strength in games of league-average output
TRAIN, VALID = range(2012, 2022), range(2022, 2024)   # test: 2024 onward


def load(data):
    frames = []
    for path in sorted(Path(data).glob('week_*.parquet')):
        d = pd.read_parquet(path)
        frames.append(d[d.season_type == 'REG'])
    d = pd.concat(frames, ignore_index=True)
    for c in ['team', 'opponent_team']:
        d[c] = d[c].replace(RENAME)
    g = pd.read_csv(Path(data) / 'games.csv', usecols=['game_id', 'home_team', 'away_team', 'location'])
    g['home_team'] = g.home_team.replace(RENAME)
    d = d.merge(g[['game_id', 'home_team', 'location']], on='game_id', how='left')
    d['venue'] = np.where(d.location == 'Neutral', 'neutral', np.where(d.team == d.home_team, 'home', 'away'))
    d = d[d.position.isin({'QB', 'RB', 'WR', 'TE'})].copy()
    for c in ['passing_yards', 'attempts', 'completions', 'rushing_yards', 'carries', 'receiving_yards', 'receptions', 'targets',
              'passing_tds', 'passing_interceptions', 'rushing_tds', 'receiving_tds']:
        d[c] = d[c].fillna(0).astype(float)
    d['total_tds'] = d.rushing_tds + d.receiving_tds
    d['touches'] = d.carries + d.targets
    return d.sort_values(['season', 'week', 'game_id', 'player_id']).reset_index(drop=True)


def defense(d):
    """Production defensive multiplier per (season, week, team, side) from earlier weeks, else last season."""
    allowed = d.groupby(['season', 'week', 'opponent_team'])[['receiving_yards', 'rushing_yards']].sum().reset_index()
    out = {}
    for season in sorted(d.season.unique()):
        s = allowed[allowed.season == season]
        last = allowed[allowed.season == season - 1]
        for week in sorted(d[d.season == season].week.unique()):
            prior = s[s.week < week]
            src = prior if len(prior) else last
            if not len(src):
                continue
            per = src.groupby('opponent_team').agg(pass_=('receiving_yards', 'sum'), rush=('rushing_yards', 'sum'), games=('week', 'nunique'))
            for side, col in [('pass', 'pass_'), ('rush', 'rush')]:
                v = per[col] / per.games
                z = (v - v.mean()) / v.std() if v.std() > 0 else v * 0
                # rating = 1 - 0.25 z (clipped 0.5-2), multiplier = 1 - 0.3 (rating - 1)
                rating = (1 - .25 * z).clip(.5, 2)
                for team, r in rating.items():
                    out[(season, week, team, side)] = 1 - .3 * (r - 1)
    return out


def ewma4(values):
    v = list(values)[-4:]
    if not v:
        return None
    w = np.array([(1 - .4) ** (len(v) - 1 - i) for i in range(len(v))])
    return float(np.dot(w, v) / w.sum())


def rows_for(d, market, defense_mult, pool):
    """Walk every player-game; emit eligible rows with baseline parts from earlier games only."""
    stat, vol, positions, side, floor = MARKETS[market]
    hist = defaultdict(lambda: deque(maxlen=400))
    rows = []
    for (season, week), wk in d.groupby(['season', 'week'], sort=True):
        emit = []
        for r in wk.itertuples(index=False):
            h = hist[r.player_id]
            if r.position in positions and len(h) >= 3:
                vols = [x[1] for x in h]
                if ewma4(vols) >= floor:
                    stats = np.array([x[0] for x in h])
                    ages = np.arange(len(h))[::-1]
                    w = .5 ** (ages / 16)
                    career = (w @ stats + 3 * pool[r.position]) / (w.sum() + 3)
                    emit.append(dict(season=season, week=week, game=r.game_id, player=r.player_id, name=r.player_display_name,
                                     position=r.position, opp=r.opponent_team, venue=r.venue, y=getattr(r, stat),
                                     recent=ewma4(stats), career=float(career),
                                     defense=defense_mult.get((season, week, r.opponent_team, side), 1.)))
        rows.extend(emit)
        for r in wk.itertuples(index=False):
            hist[r.player_id].append((getattr(r, stat), getattr(r, vol)))
    return pd.DataFrame(rows)


def walk(rows, expected, k):
    """Overall, venue and opponent factors per row from strictly earlier weeks."""
    ka, kv, ko = k
    every, venue, opp = Ledger(), Ledger(), Ledger()
    out = np.ones((len(rows), 5))
    order = rows.sort_values(['season', 'week']).index.values
    keys = list(zip(rows.season.values, rows.week.values))
    y, players, venues, opps = rows.y.values, rows.player.values, rows.venue.values, rows.opp.values
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and keys[order[j]] == keys[order[i]]:
            j += 1
        for t in order[i:j]:
            a, e, _ = every.get(players[t])
            # k is in games of the player's own expected output
            unit = expected[t]
            overall = shrunk(a, e, ka * unit)
            av, ev, _ = venue.get((players[t], venues[t]))
            ao, eo, no = opp.get((players[t], opps[t]))
            out[t] = [overall, shrunk(av, ev, kv * unit, overall) / overall, shrunk(ao, eo, ko * unit, overall) / overall,
                      shrunk(ao, eo, ko * unit), no]
        for t in order[i:j]:
            every.add(players[t], y[t], expected[t])
            venue.add((players[t], venues[t]), y[t], expected[t])
            opp.add((players[t], opps[t]), y[t], expected[t])
        i = j
    return out


VARIANTS = ['production', '+ fitted league home/away', '+ player overall (control)', '+ player home/away split',
            '+ player vs opponent', '+ both splits', 'naive: fitted home + vs opponent only']


def variant_means(name, prod, fitted, f):
    return {'production': prod, '+ fitted league home/away': fitted,
            '+ player overall (control)': fitted * f[:, 0], '+ player home/away split': fitted * f[:, 0] * f[:, 1],
            '+ player vs opponent': fitted * f[:, 0] * f[:, 2], '+ both splits': fitted * f[:, 0] * f[:, 1] * f[:, 2],
            'naive: fitted home + vs opponent only': fitted * f[:, 3]}[name]


def run_market(d, market, defense_mult):
    stat, _, positions, _, _ = MARKETS[market]
    train_rows = d[d.season.isin(TRAIN) & d.position.isin(positions)]
    pool = train_rows.groupby('position')[stat].mean().to_dict()
    rows = rows_for(d, market, defense_mult, pool)
    train, valid, test = rows.season.isin(TRAIN).values, rows.season.isin(VALID).values, (rows.season >= 2024).values
    # Recent-vs-career blend weight, fitted on training seasons.
    blend = min(np.round(np.arange(0, 1.01, .1), 1), key=lambda w: np.mean(((w * rows.recent + (1 - w) * rows.career)[train] * rows.defense[train] - rows.y[train]) ** 2))
    base = (blend * rows.recent + (1 - blend) * rows.career).values * rows.defense.values
    fixed = np.where(rows.venue == 'home', FIXED_HOME.get(market, 1.), np.where(rows.venue == 'away', 2 - FIXED_HOME.get(market, 1.), 1.))
    prod = base * fixed
    venue = rows.venue.values
    # The home/away gap alone: each venue's actual/expected ratio over the overall ratio, so an
    # era-level bias in the baseline is not mistaken for a venue effect.
    level = rows.y[train & (venue != 'neutral')].sum() / base[train & (venue != 'neutral')].sum()
    home_fit = {v: float(rows.y[train & (venue == v)].sum() / base[train & (venue == v)].sum() / level) for v in ('home', 'away')}
    home_fit['neutral'] = 1.
    fitted = base * np.array([home_fit[v] for v in venue])
    # Normal spread grows with the mean: |residual| * sqrt(pi/2) regressed on the mean (training).
    res = np.abs(rows.y.values - fitted)[train] * math.sqrt(math.pi / 2)
    slope, intercept = np.polyfit(fitted[train], res, 1)
    sigma = lambda m: np.maximum(1., intercept + slope * m)
    line = np.floor(prod) + .5           # a proxy market line: our production number, on the half point

    def scores(mean, idx):
        y = rows.y.values[idx]
        p_over = 1 - norm.cdf(line[idx], mean[idx], sigma(mean[idx]))
        return dict(sq_error=(mean[idx] - y) ** 2, brier_at_line=brier(p_over, y > line[idx]))

    # Choose k on the validation seasons, in order: overall, then each split given it.
    k = [INF, INF, INF]
    selection = []
    for slot, name in [(0, '+ player overall (control)'), (1, '+ player home/away split'), (2, '+ player vs opponent')]:
        trials = {}
        for c in GRID:
            trial = list(k)
            trial[slot] = c
            f = walk(rows, fitted, trial)
            trials[c] = float(scores(variant_means(name, prod, fitted, f), valid)['sq_error'].mean())
        k[slot] = min(trials, key=trials.get)
        selection.append(dict(component=['overall', 'home/away split', 'vs opponent'][slot], chosen_games_of_prior=fmt_k(k[slot]),
                              validation_mse={fmt_k(c): round(v, 3) for c, v in trials.items()}))
    f = walk(rows, fitted, k)
    groups = rows.game.values[test]
    all_scores = {n: scores(variant_means(n, prod, fitted, f), test) for n in VARIANTS}
    out = {}
    for n, s in all_scores.items():
        out[n] = {}
        for metric, v in s.items():
            out[n][metric] = round(float(v.mean()), 4)
            out[n][metric + '_vs_production'] = {x: round(z, 4) if isinstance(z, float) else z for x, z in cluster_ci(v - all_scores['production'][metric], groups).items()}
            if n in ('+ player home/away split', '+ player vs opponent', '+ both splits'):
                out[n][metric + '_vs_control'] = {x: round(z, 4) if isinstance(z, float) else z for x, z in cluster_ci(v - all_scores['+ player overall (control)'][metric], groups).items()}
    # Matchup streaks: 4+ earlier games against this opponent at 15%+ above (or below) expectation.
    raw = walk(rows, fitted, [INF, INF, 0])
    ctrl = variant_means('+ player overall (control)', prod, fitted, f)
    streaks = []
    for label, mask in [('everyone with 4+ prior games vs this opponent (reference)', raw[:, 4] >= 4),
                        ('beat expectation by 15%+ vs this opponent (4+ prior games)', (raw[:, 3] >= 1.15) & (raw[:, 4] >= 4)),
                        ('fell 13%+ short vs this opponent (4+ prior games)', (raw[:, 3] <= 1 / 1.15) & (raw[:, 4] >= 4))]:
        m = mask & test
        streaks.append(dict(group=label, rows=int(m.sum()), prior_ratio=round(float(raw[m, 3].mean()), 3) if m.any() else None,
                            next_game_actual_over_forecast=round(float(rows.y.values[m].sum() / ctrl[m].sum()), 3) if m.any() else None))
    return rows, f, raw, dict(rows=dict(train=int(train.sum()), validation=int(valid.sum()), test=int(test.sum())),
                              recent_weight=float(blend), fitted_home_factor={v: round(x, 4) for v, x in home_fit.items()},
                              production_home_multiplier=FIXED_HOME.get(market, 1.), k_games=dict(zip(['overall', 'home/away split', 'vs opponent'], map(fmt_k, k))),
                              weight_vs_opponent_after_8_games=round(weight_after(8, k[2]), 3),
                              weight_venue_split_after_40_games=round(weight_after(40, k[1]), 3),
                              selection=selection, test=out, matchup_streaks=streaks), fitted, prod


def named(rows, raw, fitted, market, who, opp):
    """One storyline: every game of `who` against `opp`, and what the history said beforehand."""
    m = (rows.name.values == who) & (rows.opp.values == opp)
    out = []
    for t in np.where(m)[0]:
        out.append(dict(season=int(rows.season[t]), week=int(rows.week[t]), venue=rows.venue[t], actual=float(rows.y[t]),
                        forecast=round(float(fitted[t]), 1), prior_games_vs_opp=int(raw[t, 4]), prior_ratio_vs_opp=round(float(raw[t, 3]), 3)))
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--out', type=Path, default=ROOT / 'reports/matchups/nfl.json')
    args = a.parse_args()
    d = load(args.data)
    dm = defense(d)
    report = dict(baseline='production recipe: EWMA(4) + career blend, opponent-defense rating, fixed home multipliers',
                  seasons=dict(train='2012-2021', validation='2022-2023', test='2024-2026 wk4'), markets={})
    for market in [m for m in MARKETS if MARKETS[m][3] is not None]:
        rows, f, raw, result, fitted, prod = run_market(d, market, dm)
        if market == 'pass_yds':
            result['storyline'] = {'Patrick Mahomes vs DEN': named(rows, raw, fitted, market, 'Patrick Mahomes', 'DEN')}
        report['markets'][market] = result
        t = result['test']
        print(market, {n: (t[n]['sq_error'], t[n]['sq_error_vs_production']['mean'], t[n]['brier_at_line']) for n in t}, result['k_games'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()

"""NHL player props (2026-27 season to date): would the game environment improve the
v2.4 player means? Market environment = regulation goals implied by the consensus
moneyline + total (the site's own coherence solver); our environment = the team
model's projected regulation goals stored on each snapshot.

One decision per (game, player, market): the last morning snapshot before puck drop.
Leave-one-week-out; mean' = mean * (1 + a + b.x), level-only (a) as the fair baseline.
"""
import collections
import glob
import gzip
import json
import sys
from statistics import median

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, f'{REPO}/scripts')
from nhl.v2 import data as nhl_data          # noqa: E402
from nhl.v2.coherence import team_means      # noqa: E402

MARKETS = {'player_shots_on_goal': (0, 'shots'), 'player_goals': (1, 'goals'),
           'player_assists': (2, 'assists'), 'player_points': (3, 'points')}


def load_runs():
    runs = []
    for f in glob.glob(f'{REPO}/artifacts/nhl/runs/*.json.gz'):
        r = json.load(gzip.open(f))
        runs.append(r)
    runs.sort(key=lambda r: r['snapshot']['checked_at'])
    return runs


def results(latest):
    def obj(src):
        o = next(o for o in latest['input_objects'] if o['source'].endswith(src))
        return json.load(gzip.open(f"{REPO}/artifacts/nhl/{o['path']}"))
    team_rows, player_rows = [], []
    for o in latest['input_objects']:
        if '/history/raw/' not in o['source']:
            continue
        page = json.load(gzip.open(f"{REPO}/artifacts/nhl/{o['path']}"))
        rows = page['payload']['data']
        (team_rows if '/raw/team-' in o['source'] else player_rows).extend(rows)
    games, players = nhl_data.normalize(team_rows, player_rows, 20262027)[:2]
    return {g['game_id']: g for g in games}, {(p['game_id'], p['player_id']): p for p in players}


def main():
    runs = load_runs()
    games, players = results(runs[-1])
    alphas = runs[-1]['snapshot']['model_distribution']['alphas']
    ot_home = runs[-1]['snapshot']['model_distribution']['ot_home']

    decisions = {}   # (game_id, player_id, market) -> record from the latest pre-game snapshot
    for r in runs:
        s = r['snapshot']
        if s.get('model_version') != 'nhl-v2.4':
            continue
        checked = pd.Timestamp(s['checked_at'])
        rows = s['rows']
        mkt_env = team_means(rows, ot_home)
        ours = {}
        for x in rows:
            if x.get('projected_home_reg_goals') is not None:
                ours.setdefault(x['event_id'], (x['projected_home_reg_goals'], x['projected_away_reg_goals']))
        ev = {e['nhl_game_id']: e for e in s.get('events', [])}
        offers = collections.defaultdict(list)
        for x in rows:
            if x['market'] in MARKETS and x.get('projected_mean') is not None and pd.Timestamp(x['commence_time']) > checked:
                offers[(x['nhl_game_id'], x.get('player_id'), x['market'])].append(x)
        for k, xs in offers.items():
            gid, pid, market = k
            e = ev.get(gid)
            if pid is None or e is None:
                continue
            x0 = xs[0]
            team = x0.get('player_team_id')
            if team not in (e['home_id'], e['away_id']):
                continue
            home = team == e['home_id']
            overs = [x for x in xs if x['side'] == 'Over']
            if not overs:
                continue
            line = collections.Counter(x['line'] for x in overs).most_common(1)[0][0]
            fair = [x['fair_probability'] for x in overs if x['line'] == line and x.get('fair_probability') is not None]
            m = mkt_env.get(x0['event_id'])
            o = ours.get(x0['event_id'])
            decisions[k] = dict(game_id=gid, player_id=pid, market=market, player=x0['player'], home=home,
                                commence=x0['commence_time'], checked_at=s['checked_at'], mean=x0['projected_mean'],
                                line=line, mkt_p_over=median(fair) if fair else np.nan,
                                mkt_team=(m['home'] if home else m['away']) if m else np.nan,
                                mkt_opp=(m['away'] if home else m['home']) if m else np.nan,
                                our_team=(o[0] if home else o[1]) if o else np.nan,
                                our_opp=(o[1] if home else o[0]) if o else np.nan)
    df = pd.DataFrame(decisions.values())
    stat = df.market.map(lambda m: MARKETS[m][1])
    df['actual'] = [players[(g, p)][s] if (g, p) in players else np.nan for g, p, s in zip(df.game_id, df.player_id, stat)]
    df['final'] = df.game_id.map(lambda g: g in games)
    print(f'decisions {len(df)} | game final {int(df.final.sum())} | player appeared {int(df.actual.notna().sum())}')
    df = df[df.final & df.actual.notna() & df.mkt_team.notna() & df.our_team.notna()].copy()
    df['date'] = pd.to_datetime(df.commence).dt.tz_convert('America/New_York').dt.date
    df['week'] = pd.to_datetime(df.date).dt.isocalendar().week
    df['mkt_total'] = df.mkt_team + df.mkt_opp
    df['mkt_minus_ours'] = df.mkt_team - df.our_team
    df['alpha'] = df.market.map(lambda m: alphas[MARKETS[m][0]])
    df.to_csv(str(Path(__file__).resolve().parent / 'results/nhl_dataset.csv'), index=False)
    print(f'usable {len(df)} player-markets, {df.game_id.nunique()} games, {df.date.min()} to {df.date.max()}, weeks {sorted(df.week.unique())}')
    print(df.groupby('market').size().to_string())
    print('corr(mkt team goals, our team goals) =', round(np.corrcoef(df.mkt_team, df.our_team)[0, 1], 3),
          '| mean mkt', round(df.mkt_team.mean(), 3), 'ours', round(df.our_team.mean(), 3))
    evaluate(df)


def p_over(mean, alpha, line):
    k = np.floor(line)
    if alpha > 1e-6:
        return 1 - nbinom.cdf(k, 1 / alpha, 1 / (1 + alpha * mean))
    return 1 - poisson.cdf(k, mean)


VARIANTS = {'level only': [], 'mkt team goals': ['mkt_team'], 'mkt game total': ['mkt_total'],
            'our team goals': ['our_team'], 'mkt team goals vs ours': ['mkt_minus_ours'],
            'mkt team + opp goals': ['mkt_team', 'mkt_opp']}


def fit(tr, feats):
    means = {f: tr[f].mean() for f in feats}
    X = np.column_stack([tr['mean']] + [tr['mean'] * (tr[f] - means[f]) for f in feats])
    coef, *_ = np.linalg.lstsq(X, (tr.actual - tr['mean']).values, rcond=None)
    return coef, means


def cluster_t(sub, feats):
    means = {f: sub[f].mean() for f in feats}
    X = np.column_stack([sub['mean']] + [sub['mean'] * (sub[f] - means[f]) for f in feats])
    y = (sub.actual - sub['mean']).values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[i].T @ e[i], X[i].T @ e[i]) for i in sub.groupby('game_id').indices.values())
    G = sub.game_id.nunique()
    return coef, np.sqrt(np.diag(bread @ meat @ bread) * G / (G - 1))


def evaluate(df):
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('\n=== Diagnostic: % change in stat per +1 goal of environment (game-clustered t) ===')
    out = []
    for mkt, sub in df.groupby('market'):
        for f in ['mkt_team', 'mkt_total', 'our_team', 'mkt_minus_ours']:
            c, se = cluster_t(sub, [f])
            out.append(dict(market=mkt, feature=f, pct_per_goal=100 * c[1], t=c[1] / se[1], level_pct=100 * c[0], n=len(sub)))
    print(pd.DataFrame(out).pivot_table(index='market', columns='feature', values=['pct_per_goal', 't']).round(2).to_string())

    two = df.dropna(subset=['mkt_p_over'])
    print(f'\nTwo-sided offers with a market no-vig price: {len(two)} of {len(df)} (rest are one-sided, mostly anytime goal)')
    for label, scope in [('ALL (two-sided)', two)] + [(m, two[two.market == m]) for m in MARKETS]:
        rows = []
        for name, feats in [('(current model)', None)] + list(VARIANTS.items()):
            if feats is None:
                pred = scope['mean'].values
            else:
                pred = pd.Series(np.nan, index=scope.index)
                for m, mdf in scope.groupby('market'):
                    for w in sorted(mdf.week.unique()):
                        tr, te = mdf[mdf.week != w], mdf[mdf.week == w]
                        coef, means = fit(tr, feats)
                        mult = 1 + coef[0] + sum(coef[i + 1] * (te[f] - means[f]) for i, f in enumerate(feats))
                        pred.loc[te.index] = te['mean'] * np.clip(mult, .3, 2)
                pred = pred.values
            p = np.clip([p_over(mu, a, l) for mu, a, l in zip(pred, scope.alpha, scope.line)], .01, .99)
            y = (scope.actual > scope.line).astype(float).values
            rows.append(dict(variant=name, n=len(scope), mae=np.mean(np.abs(scope.actual - pred)),
                             logloss=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), brier=np.mean((p - y) ** 2)))
        mk = scope.dropna(subset=['mkt_p_over'])
        ym = (mk.actual > mk.line).astype(float)
        pm = mk.mkt_p_over.clip(.01, .99)
        rows.append(dict(variant='(market no-vig)', n=len(mk), logloss=-np.mean(ym * np.log(pm) + (1 - ym) * np.log(1 - pm)),
                         brier=np.mean((pm - ym) ** 2)))
        print(f'\n=== Leave-one-week-out: {label} ===')
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()

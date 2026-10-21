"""Score replay_means.py variants against production on fixed row sets.

Rows are fixed from the production variant so every variant is scored on the same player-weeks:
passing markets on QB starts; rushing markets where production expected 6+ carries (RB) or 4+
(QB starts); receiving markets where production expected 2+ receptions (WR, TE, RB).
Intervals resample whole games. usage: score_replay.py REPLAY_DIR GRADED_CSV
"""
import glob
import math
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

REPLAY, GRADED = sys.argv[1:3]


def key(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


def boot(values, groups, n=2000, seed=5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'v': values, 'g': groups}).groupby('g').v.agg(['sum', 'size'])
    s, c = df['sum'].to_numpy(), df['size'].to_numpy()
    idx = rng.integers(0, len(df), (n, len(df)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


runs = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in sorted(glob.glob(f'{REPLAY}/*.csv'))}
base = runs['production']
ident = ['season', 'week', 'game_id', 'player', 'market']
carries = base[base.market == 'rush_attempts'].set_index(['season', 'week', 'player']).mu
catches = base[base.market == 'receptions'].set_index(['season', 'week', 'player']).mu
b = base.set_index(['season', 'week', 'player'])
eligible = np.where(base.market.str.startswith('pass'), base.qb_start,
            np.where(base.market.str.startswith('rush'),
                     (base.position.eq('RB') & base.set_index(['season', 'week', 'player']).index.map(carries).to_numpy() >= 6) |
                     (base.position.eq('QB') & base.qb_start & (base.set_index(['season', 'week', 'player']).index.map(carries).to_numpy() >= 4)),
                     base.position.isin(['WR', 'TE', 'RB']) & (base.set_index(['season', 'week', 'player']).index.map(catches).to_numpy() >= 2)))
rows = base[eligible.astype(bool)][ident + ['actual']].rename(columns={'actual': 'y'})
for name, r in runs.items():
    rows = rows.merge(r[ident + ['mu']].rename(columns={'mu': name}), on=ident, how='left')
rows['period'] = np.where(rows.season == 2026, '2026 wk1-6', '2024-25')

graded = pd.read_csv(GRADED)
graded = graded.assign(k=graded.player.map(key))[['nfl_game', 'k', 'market']].drop_duplicates()
rows['k'] = rows.player.map(key)
board = rows[rows.season == 2026].merge(graded.rename(columns={'nfl_game': 'game_id'}), on=['game_id', 'k', 'market'])
board = board.assign(period='2026 board rows')

variants = [v for v in runs if v != 'production']
pd.set_option('display.width', 250)
for period, sub in pd.concat([rows, board]).groupby('period', sort=False):
    print(f'\n== {period} ==')
    out = []
    for m, g in sub.groupby('market'):
        g = g.dropna(subset=list(runs))
        line = dict(market=m, n=len(g), games=g.game_id.nunique(),
                    bias_prod=(g.y - g.production).mean(), rmse_prod=math.sqrt(((g.y - g.production) ** 2).mean()))
        for v in variants:
            d = (g.y - g[v]) ** 2 - (g.y - g.production) ** 2
            lo, hi = boot(d.to_numpy(), g.game_id.to_numpy())
            line[v] = f'{math.sqrt(((g.y - g[v]) ** 2).mean()):.2f} ({(g.y - g[v]).mean():+.2f}) [{lo:+.1f},{hi:+.1f}]'
        out.append(line)
    t = pd.DataFrame(out).set_index('market')
    print(t[['n', 'games', 'bias_prod', 'rmse_prod']].round(2).to_string())
    for v in variants:
        print(f'  {v:22s} ' + ' | '.join(f'{m} {t.at[m, v]}' for m in t.index))
print('\ncells: RMSE (bias) [95% interval of MSE difference vs production; negative = better]')

"""Check that the replay reproduces production: compare its 2026 'production' means with the
pre-adjustment means saved on the archived boards (production_mean_test.py's graded rows).

usage: check_fidelity.py REPLAY_DIR GRADED_CSV
"""
import re
import sys
import unicodedata

import pandas as pd

REPLAY, GRADED = sys.argv[1:3]


def key(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', s)
    return re.sub(r'[^a-z]', '', s)


r = pd.read_csv(f'{REPLAY}/production.csv')
r = r[r.season == 2026].assign(k=r.player.map(key))
g = pd.read_csv(GRADED).dropna(subset=['stage_before_adjustments'])
m = g.merge(r[['game_id', 'k', 'market', 'mu']].rename(columns={'game_id': 'nfl_game', 'mu': 'replay'}),
            on=['nfl_game', 'k', 'market'])
m['exact'] = (m.replay - m.stage_before_adjustments).abs() < 0.01
print(f'{len(g)} board rows with recorded stages; {len(m)} matched to the replay')
print(m.groupby('market').agg(rows=('exact', 'size'), exact_share=('exact', 'mean')).to_string())

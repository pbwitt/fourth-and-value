"""Curve shape on the real 2026 lines: is a skewed curve better than the Normal where books post?

One row per game, player, market and half-point line (Over side) with 2+ books, at the last board
before kickoff. Scores the multi-book no-vig chance, the published (calibrated) probability and raw
Normal / skewed curves built from the shipped means with (a) the model's own sigma and (b) a
predictive spread fitted on 2024-25 forecast errors. Main line = the line most books post.
usage: curves.py BOARDS_CSV GRADED_CSV REPLAY_CSV
"""
import sys

import numpy as np
import pandas as pd

from common import COUNT, decision_boards, game_interval, key, log_loss, p_over, predictive_variance

BOARDS, GRADED, REPLAY = sys.argv[1:4]
replay = pd.read_csv(REPLAY)
coef = predictive_variance(replay)
print('Predictive spread fitted on 2024-25 errors vs the model sigma (median sd / mean):')
for m, c in coef.items():
    r = replay[(replay.market == m) & replay.season.isin([2024, 2025]) & (replay.mu > 0)]
    print(f'  {m:16s} {np.median(np.sqrt(np.maximum(c[0] * r.mu + c[1] * r.mu ** 2, 1e-6)) / r.mu):.2f} vs {np.median(r.sigma / r.mu):.2f}')

b = decision_boards(BOARDS)
b = b[b.name.eq('over') & b.consensus_prob.notna() & b.model_prob.notna() & (b.book_count >= 2)]
lines = b.groupby(['game_id', 'player', 'market_std', 'point']).agg(
    market=('consensus_prob', 'median'), published=('model_prob', 'median')).reset_index()
g = pd.read_csv(GRADED)[['event', 'player', 'market', 'nfl_game', 'actual', 'line', 'k']]
lines = lines.merge(g.rename(columns={'event': 'game_id', 'market': 'market_std', 'line': 'main_line'}),
                    on=['game_id', 'player', 'market_std'])
new = replay[replay.season == 2026].assign(k=replay.player.map(key))
lines = lines.merge(new[['game_id', 'k', 'market', 'mu', 'sigma']].rename(columns={'game_id': 'nfl_game', 'market': 'market_std'}),
                    on=['nfl_game', 'k', 'market_std']).dropna(subset=['mu'])
lines['y'] = (lines.actual > lines.point).astype(float)
lines['where'] = np.where(lines.point == lines.main_line, 'main line', 'other books\' lines')

print('\nLog loss by line (lower is better); interval = difference from the market, resampling games')
for (m, where), d in lines.groupby(['market_std', 'where']):
    if len(d) < 40:
        continue
    c, cnt = coef[m], m in COUNT
    pv = np.maximum(c[0] * d.mu + c[1] * d.mu ** 2, 1e-6)
    cols = {'market': d.market, 'published': d.published,
            'Normal, model sigma': p_over('normal', d.mu, d.sigma ** 2, d.point, cnt),
            'Normal, predictive': p_over('normal', d.mu, pv, d.point, cnt),
            'skewed, model sigma': p_over('skew', d.mu, d.sigma ** 2, d.point, cnt),
            'skewed, predictive': p_over('skew', d.mu, pv, d.point, cnt)}
    base = log_loss(cols['market'], d.y)
    print(f'\n{m}, {where}: {len(d)} lines, {d.nfl_game.nunique()} games; Over hit {d.y.mean():.3f}, market said {d.market.mean():.3f}')
    for name, p in cols.items():
        lo, hi = game_interval(log_loss(p, d.y) - base, d.nfl_game.to_numpy())
        print(f'  {name:20s} mean P(over) {np.mean(p):.3f}  log loss {log_loss(p, d.y).mean():.4f}  [{lo:+.4f}, {hi:+.4f}]')

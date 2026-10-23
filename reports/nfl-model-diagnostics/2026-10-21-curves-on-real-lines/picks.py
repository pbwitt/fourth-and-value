"""Where does the NFL candidate pool's return come from: the model's probability or the price?

Offers: best price per game, player, market, side and half-point line at the last board before
kickoff, 2+ books. Rules: the Top Picks blend gate (final = sigmoid(0.10 logit(model) +
0.90 logit(market)), EV >= 1% at the offered price, model >= market); a price-only gate (market
no-vig chance x price - 1 >= 1%); and model agreement without a price edge. This is the
candidate pool, not the published card (no per-game caps or review). Then: does the model's gap
from the market predict results once the market price is held fixed?
usage: picks.py BOARDS_CSV GRADED_CSV
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import game_interval, logit, offers

b = offers(*sys.argv[1:3])
b = b[b.book_count >= 2].copy()
b['final'] = [1 / (1 + np.exp(-(.10 * logit(x) + .90 * logit(y)))) for x, y in zip(b.model, b.q)]
rules = {'blend gate (production)': (b.final * b.dec - 1 >= .01) & (b.model >= b.q),
         'price only (model ignored)': b.q * b.dec - 1 >= .01,
         'model >= market, no price edge': (b.q * b.dec - 1 < .01) & (b.model >= b.q)}
print(f'{len(b)} offers, {b.nfl_game.nunique()} games')
for name, mask in rules.items():
    d = b[mask]
    lo, hi = game_interval(d.profit.to_numpy(), d.nfl_game.to_numpy())
    print(f'\n{name}: {len(d)} bets, won {d.y.mean():.3f}, ROI {d.profit.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')
    print('  ' + ' | '.join(f'{k} {len(x)}: {x.profit.mean():+.2f}' for k, x in d.groupby('market_std')))

print('\nResult minus market chance, by the model\'s gap above the market (log-odds)')
b['bucket'] = pd.cut(b.gap, [-9, -0.5, -0.2, 0, 0.2, 0.5, 1, 9])
for name, x in [('receptions', b[b.market_std == 'receptions']), ('other markets', b[b.market_std != 'receptions'])]:
    t = x.groupby('bucket', observed=True).agg(n=('y', 'size'), won=('y', 'mean'), market_said=('q', 'mean'), roi=('profit', 'mean'))
    print(f'\n{name}'); print(t.assign(beat_market=t.won - t.market_said).round(3).to_string())
    x = x.assign(lq=x.q.map(logit))
    fit = smf.glm('y ~ lq + gap', data=x, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(x.nfl_game)[0]})
    print('  logistic y ~ logit(market) + gap, game-clustered: ' +
          ' | '.join(f'{k} {fit.params[k]:+.3f} (se {fit.bse[k]:.3f})' for k in ['lq', 'gap']))

r = b[b.market_std == 'receptions'].copy()
r['price'] = pd.cut(r.q, [0, .4, .45, .55, .6, 1], labels=['<40%', '40-45%', '45-55%', '55-60%', '>60%'])
print('\nReceptions by side and market chance')
print(r.groupby(['name', 'price'], observed=True).agg(n=('y', 'size'), won=('y', 'mean'), market_said=('q', 'mean'),
      roi=('profit', 'mean'), mean_line=('point', 'mean')).round(3).to_string())
low = r[(r.name == 'over') & (r.q < .4)]
lo, hi = game_interval(low.profit.to_numpy(), low.nfl_game.to_numpy())
print(f'\nReceptions Overs priced under 40%: {len(low)} offers, {low.nfl_game.nunique()} games, {low.player.nunique()} players; '
      f'won {low.y.mean():.3f} vs market {low.q.mean():.3f}; ROI {low.profit.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')

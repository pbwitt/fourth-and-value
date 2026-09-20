#!/usr/bin/env python3
"""Reproduce the September 20 Jones article from saved quotes; no paid API calls.
Run: .venv/bin/python scripts/analyze_snf_jones_2026_week2.py --schedule PATH
The schedule must be nflverse games.csv, with actual ET gametime and location.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import career_baseline as cb
import make_player_prop_params as params
from make_props_edges import load_calibration
from market_math import add_market_comparisons, expected_profit, implied_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', required=True)
    args = parser.parse_args()
    src = Path('reports/colts-chiefs-2026-week2')
    out = Path('docs/blog/colts-chiefs-jones-2026')
    out.mkdir(exist_ok=True)
    quotes = pd.read_csv(src / 'fresh-props.csv')
    quotes = quotes[(quotes.player == 'Daniel Jones') & (quotes.market == 'player_pass_attempts')].copy()
    quotes['market_std'] = 'pass_attempts'
    quotes = add_market_comparisons(quotes)
    assert quotes.game_id.nunique() == 1
    assert quotes.bookmaker.nunique() == 4
    quotes.to_csv(out / 'book-prices.csv', index=False)
    row = pd.read_csv('data/props/params_week2.csv').query("player == 'Daniel Jones' and market_std == 'pass_attempts'").iloc[0]
    board = pd.read_csv('data/props/props_with_model_week2.csv', low_memory=False)
    saved = board[(board.player == 'Daniel Jones') & (board.market_std == 'pass_attempts') & (board.point == 31.5) & (board['name'] == 'under')].iloc[0]
    career = cb.load_career_logs(list(range(2020, 2027)))
    career = career[(career.season < 2026) | ((career.season == 2026) & (career.week < 2))]
    prior = career[(career.player == 'Daniel Jones') & (career.season < 2026)]
    current = career[(career.player == 'Daniel Jones') & (career.season == 2026)]
    pool_mu, pool_sigma = cb.compute_position_pool(career, 'QB', 'attempts')
    career_mu, career_sigma, n = cb.player_career_baseline(prior, 2026, 'attempts', None, pool_mu, pool_sigma)
    age_weights = .75 ** (2026 - prior.season)
    own_mu = float(np.average(prior.attempts, weights=age_weights))
    recent_mu = params.exponential_weighted_mean(current.attempts.tail(4), .4)
    base_mu, base_sigma = cb.blend_recent_with_career(recent_mu, np.nan, len(current), career_mu, career_sigma)
    defense = float(params.calculate_defensive_ratings(2026, 2).loc['KC', 'pass_def_rating'])
    multiplier = 1 - (defense - 1) * .3
    assert np.isclose(base_mu * multiplier, row.mu)
    assert np.isclose(base_sigma, row.sigma)
    assert row.used_logs == 1 and not row.no_real_data
    curve = load_calibration()['markets']['pass_attempts']
    def probabilities(mean, line=31.5):
        raw = float(norm.cdf(line, mean, row.sigma))
        return raw, float(np.interp(raw, curve['x'], curve['y']))
    raw, calibrated = probabilities(row.mu)
    assert np.isclose(raw, saved.model_prob_raw) and np.isclose(calibrated, saved.model_prob)
    offers = []
    for book, g in quotes.groupby('bookmaker', sort=False):
        u = g[g.side == 'under'].iloc[0]; o = g[g.side == 'over'].iloc[0]
        r, c = probabilities(row.mu, u.point)
        offers.append(dict(book=u.bookmaker_title, line=float(u.point), under=int(u.price), over=int(o.price),
                           under_devig=float(u.prob_devig), raw=r, calibrated=c,
                           model_ev_per_100=float(expected_profit(c, u.price)), timestamp=u.last_update))
    scenarios = []
    for name, mean in [('Published model', row.mu), ('Half the defense adjustment', base_mu * (1 + multiplier) / 2),
                       ('No defense adjustment', base_mu), ('No defense adjustment, +2 attempts', base_mu + 2)]:
        r, c = probabilities(mean)
        scenarios.append(dict(name=name, mean=float(mean), raw=r, calibrated=c, ev_per_100=float(expected_profit(c, -110))))
    sched = pd.read_csv(args.schedule)
    p25 = prior[prior.season == 2025][['game_id','season','week','team','opponent_team','attempts']].merge(
        sched[['game_id','gameday','gametime','home_team','away_team','location']], on='game_id', validate='one_to_one')
    assert len(p25) == 13 and p25.gametime.notna().all()
    p25['venue'] = np.where(p25.location == 'Neutral', 'Neutral', np.where(p25.home_team == p25.team, 'Home', 'Away'))
    p25['night_ET'] = p25.gametime.str.slice(0,2).astype(int) >= 19
    p25['under_31_5'] = p25.attempts <= 31
    p25['early_injury_exit'] = p25.game_id == '2025_14_IND_JAX'
    p25.to_csv(out / 'jones-2025-game-log.csv', index=False)
    splits = []
    groups = [('All 2025 appearances', p25), ('Home (excluding neutral site)', p25[p25.venue == 'Home']),
              ('Away', p25[p25.venue == 'Away']), ('Away, excluding Jacksonville injury exit', p25[(p25.venue == 'Away') & ~p25.early_injury_exit]),
              ('Neutral (Berlin)', p25[p25.venue == 'Neutral']), ('Night (kickoff at/after 7 p.m. ET)', p25[p25.night_ET])]
    for label, g in groups:
        splits.append(dict(label=label, n=len(g), mean=float(g.attempts.mean()) if len(g) else None,
                           unders=int(g.under_31_5.sum()), hit_rate=float(g.under_31_5.mean()) if len(g) else None))
    for file in ['params_week2.csv', 'props_with_model_week2.csv']:
        table=pd.read_csv('data/props/'+file, low_memory=False)
        table[(table.player=='Daniel Jones') & (table.market_std=='pass_attempts')].to_csv(out / ('model-'+file),index=False)
    market=quotes[(quotes.point==31.5)&(quotes.side=='under')].iloc[0]
    evidence = dict(season=2026,week=2,game='Indianapolis Colts at Kansas City Chiefs',kickoff_utc='2026-09-21T00:20:00Z',
        model_generated_at=saved.generated_at,quote_min=quotes.last_update.min(),quote_max=quotes.last_update.max(),
        model=dict(mu=float(row.mu),sigma=float(row.sigma),raw=raw,calibrated=calibrated,
            career_games=n,own_weight=n/(n+40),own_weighted_mean=own_mu,pool_mean=pool_mu,pool_sigma=pool_sigma,
            career_mean=career_mu,career_sigma=career_sigma,recent_games=len(current),recent_mean=recent_mu,
            base_mean=base_mu,defense_rating=defense,defense_multiplier=multiplier,
            poisson_comparison=float(poisson.cdf(31,row.mu)),normal_negative_mass=float(norm.cdf(-.5,row.mu,row.sigma))),
        price=dict(preferred_book='DraftKings',line=31.5,odds=-110,break_even=implied_probability(-110),
            fair_american=-100*calibrated/(1-calibrated),consensus_line=float(market.consensus_line),
            exact_line_consensus=float(market.consensus_prob),exact_line_books=int(market.book_count),
            ev_per_100=float(expected_profit(calibrated,-110))),
        calibration=load_calibration()['_meta'],offers=offers,scenarios=scenarios,splits=splits,
        provenance=dict(schedule='https://github.com/nflverse/nfldata/blob/master/data/games.csv',
            stats='https://github.com/nflverse/nflverse-data/releases/tag/player_stats',
            odds='The Odds API: US region, event-specific player_pass_attempts, retrieved 2026-09-20; best among returned books only',
            night_definition='Kickoff at or after 19:00 America/New_York; not all nationally televised games'))
    (out/'analysis.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__ == '__main__': main()

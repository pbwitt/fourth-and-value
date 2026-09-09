"""NFL price comparisons. Never combine prices for different events or lines."""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

NORMAL_MARKETS = {'rush_yds', 'recv_yds', 'pass_yds', 'receptions',
                  'rush_attempts', 'pass_attempts', 'pass_completions'}
POISSON_MARKETS = {'pass_tds', 'pass_interceptions', 'interceptions'}


def implied_probability(price):
    try:
        price = float(price)
        if not math.isfinite(price) or abs(price) < 100:
            return np.nan
        return 100 / (price + 100) if price > 0 else -price / (100 - price)
    except (TypeError, ValueError):
        return np.nan


def outcome_probabilities(market, side, point, mu=np.nan, sigma=np.nan, lam=np.nan):
    """Return (win probability conditional on no push, push probability).

    Integer lines settle equality as a push. Normal yard/count approximations
    allocate the integer's half-unit interval to a push; half-lines retain the
    existing approximation. This agrees with calibration's push-excluded labels.
    """
    side = str(side).lower().strip()
    try:
        if market == 'anytime_td':
            if side not in {'yes', 'no', 'over', 'under'} or not math.isfinite(lam) or lam < 0:
                return np.nan, np.nan
            yes = -math.expm1(-lam)
            return (yes if side in {'yes', 'over'} else 1 - yes), 0.0
        point = float(point)
        if not math.isfinite(point) or side not in {'over', 'under'}:
            return np.nan, np.nan
        integer = point.is_integer()
        if market in POISSON_MARKETS and math.isfinite(lam) and lam >= 0:
            over = poisson.sf(math.floor(point), lam)
            under = poisson.cdf(math.ceil(point) - 1, lam)
            push = poisson.pmf(point, lam) if integer else 0.0
        elif market in NORMAL_MARKETS and math.isfinite(mu) and math.isfinite(sigma) and sigma > 0:
            over = norm.sf(point + (0.5 if integer else 0), mu, sigma)
            under = norm.cdf(point - (0.5 if integer else 0), mu, sigma)
            push = max(0.0, 1 - over - under) if integer else 0.0
        else:
            return np.nan, np.nan
        settled = over + under
        return ((over if side == 'over' else under) / settled if settled > 0 else np.nan), float(push)
    except (ValueError, TypeError):
        return np.nan, np.nan


def expected_profit(probability, price, push_probability=0.0, stake=100):
    """Expected profit with a refunded stake on pushes, conditional model p."""
    q = implied_probability(price)
    if not (math.isfinite(q) and 0 <= probability <= 1 and 0 <= push_probability <= 1):
        return np.nan
    return stake * (1 - push_probability) * (probability / q - 1)


def add_market_comparisons(frame):
    """Paired de-vig at the SAME line; one observation per sportsbook.

    consensus_prob and book_count describe this exact line and side. The median
    line is a separate descriptive measure (one median line per book). Missing
    opposite sides remain unknown, including one-sided anytime-TD offers.
    """
    d = frame.drop(columns=['prob_devig', 'consensus_prob', 'consensus_line', 'book_count'], errors='ignore').copy()
    d['mkt_prob'] = d['price'].map(implied_probability)
    d['side'] = d['name'].astype(str).str.lower().str.strip()
    event = next((c for c in ['game_id', 'event_id', 'game', 'commence_time'] if c in d), None)
    player = next((c for c in ['player_key', 'name_std', 'player'] if c in d), None)
    book = next((c for c in ['bookmaker', 'book', 'bookmaker_title'] if c in d), None)
    if not all([event, player, book]):
        raise ValueError('Market comparison requires event, player and sportsbook identity')
    keys = [event, player, 'market_std']
    d['_line'] = pd.to_numeric(d['point'], errors='coerce').round(6).fillna('__binary__')
    # Exact duplicates do not get additional votes. Conflicting duplicate prices
    # are not a defensible quote, so suppress their de-vig estimate.
    pair_keys = keys + ['_line', book]
    quotes = d.groupby(pair_keys + ['side'], dropna=False)['mkt_prob'].agg(['first', 'nunique']).reset_index()
    quotes.loc[quotes['nunique'] != 1, 'first'] = np.nan
    opposite = {'over': 'under', 'under': 'over', 'yes': 'no', 'no': 'yes'}
    other = quotes[pair_keys + ['side', 'first']].rename(columns={'first': 'opposite_prob'})
    other['side'] = other['side'].map(opposite)
    quotes = quotes.merge(other, on=pair_keys + ['side'], how='left', validate='one_to_one')
    quotes['prob_devig'] = quotes['first'] / (quotes['first'] + quotes['opposite_prob'])
    quotes = quotes.rename(columns={'first': 'quote_prob'})
    d = d.merge(quotes[pair_keys + ['side', 'prob_devig']], on=pair_keys + ['side'], how='left', validate='many_to_one')
    exact = quotes.groupby(keys + ['_line', 'side'], dropna=False).agg(
        consensus_prob=('prob_devig', 'median'), book_count=('prob_devig', 'count')).reset_index()
    per_book = d.groupby(keys + [book], dropna=False)['point'].median().reset_index()
    lines = per_book.groupby(keys, dropna=False)['point'].median().rename('consensus_line').reset_index()
    return (d.merge(exact, on=keys + ['_line', 'side'], how='left', validate='many_to_one')
             .merge(lines, on=keys, how='left', validate='many_to_one').drop(columns='_line'))

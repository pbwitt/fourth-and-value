#!/usr/bin/env python3
"""Price NFL totals: turn a projection into a probability and an expected value.

A projection alone cannot be bet. What matters is P(total > line) against the
price on offer, so this applies the measured calibration from
nfl_calibrate_totals.py:

    calibrated = line + beta * (projection - line)
    P(over)    = 1 - Phi((line - calibrated) / residual_sd)

beta is how much of the model's claimed edge historically survived. When beta
is not distinguishable from zero the calibrated projection sits on the market
line and P(over) sits on the market's own de-vigged number, which is the honest
output for a model with no demonstrated skill.

Expected value is computed against the best available price at the consensus
line, with a push probability for integer totals.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_math import implied_probability


def price(predictions_path, team_totals_path, calibration_path, output_path):
    with open(calibration_path) as f:
        cal = json.load(f)
    beta = cal['shrinkage_beta']
    sd = cal['residual_sd']

    preds = pd.read_csv(predictions_path)
    market = pd.read_csv(team_totals_path)

    df = market.merge(preds[['game', 'total_pred']], on='game', how='left')
    df = df.dropna(subset=['total_pred', 'consensus_total'])
    if len(df) == 0:
        raise SystemExit('No games with both a projection and a consensus line')

    line = df['consensus_total']
    df['model_projection'] = df['total_pred']
    df['claimed_edge'] = df['model_projection'] - line
    # Only the surviving fraction of the claimed edge moves the number.
    df['calibrated_projection'] = line + beta * df['claimed_edge']
    df['calibrated_edge'] = df['calibrated_projection'] - line

    # An integer total can push; the win probabilities are conditional on the
    # bet settling, which is what the prices are quoting.
    integer = line == line.round()
    z_over = (line + np.where(integer, 0.5, 0.0) - df['calibrated_projection']) / sd
    z_under = (line - np.where(integer, 0.5, 0.0) - df['calibrated_projection']) / sd
    over = 1 - norm.cdf(z_over)
    under = norm.cdf(z_under)
    push = np.where(integer, np.clip(1 - over - under, 0, None), 0.0)
    settled = over + under

    df['push_prob'] = push
    df['model_over_prob'] = over / settled
    df['model_under_prob'] = under / settled
    df['market_over_prob'] = df['fair_over_prob']
    df['prob_edge_over_pp'] = (df['model_over_prob'] - df['fair_over_prob']) * 100

    def ev(prob, price_col):
        q = df[price_col].apply(implied_probability)
        return 100 * (1 - df['push_prob']) * (prob / q - 1)

    df['ev_over_per_100'] = ev(df['model_over_prob'], 'best_over_price')
    df['ev_under_per_100'] = ev(df['model_under_prob'], 'best_under_price')
    df['best_side'] = np.where(df['ev_over_per_100'] >= df['ev_under_per_100'], 'over', 'under')
    df['best_ev_per_100'] = df[['ev_over_per_100', 'ev_under_per_100']].max(axis=1)

    df['calibration_beta'] = beta
    df['calibration_sd'] = sd
    df['calibration_significant'] = cal['skill_is_significant']

    keep = ['game', 'commence_time', 'consensus_total', 'model_projection',
            'claimed_edge', 'calibrated_projection', 'calibrated_edge',
            'model_over_prob', 'model_under_prob', 'market_over_prob',
            'prob_edge_over_pp', 'push_prob', 'best_over_price', 'best_over_book',
            'best_under_price', 'best_under_book', 'ev_over_per_100',
            'ev_under_per_100', 'best_side', 'best_ev_per_100',
            'calibration_beta', 'calibration_sd', 'calibration_significant']
    out = df[[c for c in keep if c in df.columns]].sort_values('commence_time')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f'✓ Wrote {len(out)} priced games to {output_path}')
    print(f'  calibration: beta={beta:+.4f} (p={cal["shrinkage_pvalue"]:.3f}), '
          f'residual sd={sd:.2f}, n={cal["n_games"]}')
    show = out[['game', 'consensus_total', 'model_projection', 'claimed_edge',
                'calibrated_projection', 'model_over_prob', 'market_over_prob',
                'prob_edge_over_pp', 'best_ev_per_100']].copy()
    print(show.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    if not cal['skill_is_significant']:
        print('\n  Calibration is not significant, so every calibrated projection sits')
        print('  on the market line and every probability edge is ~0. Publishing these')
        print('  as picks would be publishing the market back with extra decimals.')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Price NFL totals into probabilities and EV')
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--team-totals', required=True)
    ap.add_argument('--calibration', default='data/nfl/models/totals_calibration.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    price(args.predictions, args.team_totals, args.calibration, args.output)

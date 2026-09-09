#!/usr/bin/env python3
"""Measure how much of the totals model's claimed edge actually survives.

Walk-forward over completed seasons: for every game, train only on games that
finished earlier, record what the model claimed (projection minus closing
line) and what happened (actual total minus closing line), then regress the
second on the first.

The slope is the shrinkage factor. If the model has no skill the slope is zero
and the calibrated projection collapses onto the market line, which is the
correct answer rather than a failure. The residual SD is what turns a
projection into a probability; without it a points edge cannot be priced.

Writes data/nfl/models/totals_calibration.json so the published probabilities
are traceable to a measurement instead of an assumption.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def build_matrix(features_path, seasons):
    feats = pd.read_csv(features_path)
    fcols = [c for c in feats.columns if c.endswith('_L3') or c.endswith('_L5')]
    feats['is_home'] = (feats['home_away'] == 'home').astype(int)

    games = pd.read_csv(GAMES_URL)
    games = games[(games.season.isin(seasons)) & (games.game_type == 'REG')]
    games = games.dropna(subset=['home_score', 'away_score', 'total_line'])
    games['actual'] = games.home_score + games.away_score

    home = feats.rename(columns={c: 'h_' + c for c in fcols})
    away = feats.rename(columns={c: 'a_' + c for c in fcols})
    m = games.merge(home[['season', 'week', 'team'] + ['h_' + c for c in fcols]],
                    left_on=['season', 'week', 'home_team'],
                    right_on=['season', 'week', 'team'])
    m = m.merge(away[['season', 'week', 'team'] + ['a_' + c for c in fcols]],
                left_on=['season', 'week', 'away_team'],
                right_on=['season', 'week', 'team'])
    xcols = ['h_' + c for c in fcols] + ['a_' + c for c in fcols]
    m = m.dropna(subset=xcols + ['total_line', 'actual'])
    return m.sort_values(['season', 'week']).reset_index(drop=True), xcols


def calibrate(features_path, seasons, output_path, min_train=200):
    m, xcols = build_matrix(features_path, seasons)

    frames = []
    for season, week in m[['season', 'week']].drop_duplicates().values.tolist():
        train = m[(m.season < season) | ((m.season == season) & (m.week < week))]
        test = m[(m.season == season) & (m.week == week)]
        if len(train) < min_train or len(test) == 0:
            continue
        model = Ridge(alpha=1.0).fit(train[xcols].values, train['actual'].values)
        frames.append(pd.DataFrame({
            'projection': model.predict(test[xcols].values),
            'line': test['total_line'].values,
            'actual': test['actual'].values,
        }))

    d = pd.concat(frames, ignore_index=True)
    claimed = d.projection - d.line
    happened = d.actual - d.line

    fit = stats.linregress(claimed, happened)
    shrunk_residual = happened - fit.slope * claimed

    params = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'seasons': sorted(int(s) for s in seasons),
        'n_games': int(len(d)),
        'shrinkage_beta': float(fit.slope),
        'shrinkage_stderr': float(fit.stderr),
        'shrinkage_pvalue': float(fit.pvalue),
        'correlation': float(fit.rvalue),
        'residual_sd': float(shrunk_residual.std(ddof=1)),
        'market_mae': float(happened.abs().mean()),
        'model_mae': float((d.projection - d.actual).abs().mean()),
        'skill_is_significant': bool(fit.pvalue < 0.05),
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f'✓ Wrote {output_path}')
    for k, v in params.items():
        print(f'  {k}: {v}')
    if not params['skill_is_significant']:
        print('\n  Shrinkage is not distinguishable from zero: calibrated projections')
        print('  will sit on the market line, and published probabilities will sit')
        print('  on the market price. That is the measurement, not a bug.')
    return params


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Calibrate NFL totals model against closing lines')
    ap.add_argument('--features', default='data/nfl/processed/team_features.csv')
    ap.add_argument('--seasons', default='2022,2023,2024,2025')
    ap.add_argument('--output', default='data/nfl/models/totals_calibration.json')
    args = ap.parse_args()
    calibrate(args.features, [int(s) for s in args.seasons.split(',')], args.output)

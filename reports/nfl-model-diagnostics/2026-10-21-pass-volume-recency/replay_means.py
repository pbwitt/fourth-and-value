"""Replay the production NFL base means week by week and grade them against box scores.

Calls make_player_prop_params' own estimators with only the data production would have had
before each week (current-season logs and a six-season career window, both truncated at the
week). Variants are set through career_baseline's settings. Output: one CSV row per player,
week and market with the mean and the result.

usage: replay_means.py DATA_ROOT REPO OUT_CSV VARIANT   (DATA_ROOT holds data/weekly_player_stats_*.parquet)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

DATA_ROOT, REPO, OUT, VARIANT = sys.argv[1:5]
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(DATA_ROOT)
import career_baseline as cb  # noqa: E402
import make_player_prop_params as mpp  # noqa: E402

VARIANTS = {  # name: (RECENT_GAMES, RECENT_ALPHA, QB_STARTS_ONLY, K_RECENT_MU)
    'production': (4, 0.4, False, 4),
    'starts': (4, 0.4, True, 4),
    'recent8_hl4': (8, 0.5 ** (1 / 4), False, 4),
    'recent17_hl4': (17, 0.5 ** (1 / 4), False, 4),
    'recent17_hl6': (17, 0.5 ** (1 / 6), False, 4),
    'recent17_hl4_k8': (17, 0.5 ** (1 / 4), False, 8),
    'starts_recent17_hl4': (17, 0.5 ** (1 / 4), True, 4),
    'starts_recent17_hl6': (17, 0.5 ** (1 / 6), True, 4),
}
cb.RECENT_GAMES, cb.RECENT_ALPHA, cb.QB_STARTS_ONLY, cb.K_RECENT_MU = VARIANTS[VARIANT]
SEASONS = {2024: range(1, 19), 2025: range(1, 19), 2026: range(1, 7)}
FAMILIES = {'pass': (['QB'], ['pass_attempts', 'pass_completions', 'pass_yds']),
            'rush': (['RB', 'QB'], ['rush_attempts', 'rush_yds']),
            'receive': (['WR', 'TE', 'RB'], ['receptions', 'recv_yds'])}
ACTUAL = {'pass_attempts': 'attempts', 'pass_completions': 'completions', 'pass_yds': 'passing_yards',
          'rush_attempts': 'carries', 'rush_yds': 'rushing_yards', 'receptions': 'receptions',
          'recv_yds': 'receiving_yards'}

allc = cb.load_career_logs(list(range(2018, 2027)))
weekly = {s: mpp._load_season_weekly(s) for s in SEASONS}
team = 'team' if 'team' in allc else 'recent_team'
reg = allc[allc.season_type.eq('REG')].copy()
lead = reg.attempts.where(reg.position.eq('QB')).groupby([reg.season, reg.week, reg[team]]).transform('max')
reg['qb_start'] = reg.position.eq('QB') & reg.attempts.gt(0) & reg.attempts.eq(lead)

rows, t0 = [], time.time()
for season, weeks in SEASONS.items():
    window = allc[allc.season.between(season - 6, season)]
    for week in weeks:
        targets = reg[(reg.season == season) & (reg.week == week)]
        if targets.empty:
            continue
        cur = weekly[season][weekly[season].week < week]
        logs = (cur.sort_values(['player', 'season', 'week']).groupby('player').tail(mpp.WINDOW_GAMES)
                .reset_index(drop=True))
        career = window[(window.season < season) | ((window.season == season) & (window.week < week))]
        for family, (positions, markets) in FAMILIES.items():
            who = targets[targets.position.isin(positions)].drop_duplicates('player')
            idx = pd.Index(who.player)
            est = getattr(mpp, f'estimate_{family}_latents')
            latents = {family: est(logs, idx, alpha=cb.RECENT_ALPHA, career_df=career, season=season)}
            for market in markets:
                mu, _ = mpp.derive_market_from_latents(market, latents, idx)
                part = who.assign(mu=who.player.map(mu.to_dict()), market=market, actual=who[ACTUAL[market]].fillna(0))
                rows.append(part[['season', 'week', 'game_id', 'player', 'position', 'market', 'mu', 'actual', 'qb_start']])
        print(f'{VARIANT} {season} wk{week} {time.time() - t0:.0f}s', flush=True)
pd.concat(rows).assign(variant=VARIANT).to_csv(OUT, index=False)

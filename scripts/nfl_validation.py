#!/usr/bin/env python3
"""Point-in-time NFL prop reconstruction, calibration refit and chronological evaluation.

Every forecast is rebuilt by the production code path (make_player_prop_params.build_params
and the same loaders) using only games completed before the target week. Calibration
fitting and evaluation are separated by season; all rows from one game stay in one
window and uncertainty resamples whole games.

Windows (fixed before any result was computed; see RESEARCH_SYSTEM.md decision log):
  * hyperparameters: fixed during 2025 development (2025 is therefore NOT untouched);
  * evaluation calibration artifact: fit on 2024 REG weeks 2-18, evaluated on 2025 REG
    weeks 2-18 (a development-period holdout for the calibration map only);
  * deployment artifact: fit on 2024 + 2025 REG weeks 2-18 with the same specification,
    scored out of time on 2026 weeks 2-4 at archived, provider-timestamped book lines,
    and prospectively from 2026 week 5 onward. No untouched historical holdout exists.

Calibration population: fixed line grids per market (as in the MLB audit), not
historical bookmaker lines. The only 2025 lines in the repository are published pages
truncated to 3,000 rows sorted by the previous model's edge, without provider quote
times; they are a selected subset and are not used. Market comparisons use the exact
same outcome and line where timestamped offers exist (2026 weeks 2-4) and are labelled
unavailable otherwise.

  python scripts/nfl_validation.py fetch
  python scripts/nfl_validation.py run --run-id 2026-10-05
  python scripts/nfl_validation.py install --run-id 2026-10-05   # deployment artifact
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
DATA = ROOT/'data/nfl_validation'
REPORTS = ROOT/'reports/nfl-validation'
RELEASE = 'https://github.com/nflverse/nflverse-data/releases/download'
SEASONS = range(2017, 2027)
FIT_EVAL, EVAL, FIT_DEPLOY = [2024], [2025], [2024, 2025]
WEEKS = range(2, 19)
BOOK_WEEKS = {2: 'reports/week2-2026/offers.csv', 3: 'reports/nfl-weekly/2026/week-3/pregame/props.csv',
              4: 'reports/nfl-weekly/2026/week-4/pregame/props.csv'}
STAT = {'receptions': 'receptions', 'recv_yds': 'receiving_yards', 'rush_attempts': 'carries',
        'rush_yds': 'rushing_yards', 'pass_attempts': 'attempts', 'pass_completions': 'completions',
        'pass_yds': 'passing_yards', 'pass_tds': 'passing_tds'}
# Fixed thresholds spanning typical posted lines; chosen from market structure, not outcomes.
GRIDS = {'receptions': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
         'recv_yds': [14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5, 89.5],
         'rush_attempts': [7.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5],
         'rush_yds': [24.5, 34.5, 44.5, 54.5, 64.5, 74.5, 89.5],
         'pass_attempts': [26.5, 29.5, 31.5, 33.5, 35.5, 37.5, 39.5],
         'pass_completions': [17.5, 19.5, 21.5, 23.5, 25.5],
         'pass_yds': [189.5, 209.5, 224.5, 239.5, 254.5, 269.5, 289.5],
         'pass_tds': [.5, 1.5, 2.5]}
# Pre-game eligibility (from earlier games only), approximating which players get props.
ROLES = {'pass': ('attempts', 15, ['pass_attempts', 'pass_completions', 'pass_yds', 'pass_tds']),
         'rush': ('carries', 4, ['rush_attempts', 'rush_yds']),
         'receive': ('targets', 2, ['receptions', 'recv_yds'])}
MIN_MARKET_N = 500
BAND = (.03, .97)
SEED = 20261005


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- data
def fetch(seasons=SEASONS, data=DATA):
    """Download nflverse weekly stats, snap counts and schedules; record hashes."""
    import requests
    data.mkdir(parents=True, exist_ok=True)
    manifest = dict(fetched_at=datetime.now(timezone.utc).isoformat(), files={})
    for season in seasons:
        for name, url in [(f'weekly_player_stats_{season}.parquet', f'{RELEASE}/stats_player/stats_player_week_{season}.csv.gz'),
                          (f'snap_counts_{season}.parquet', f'{RELEASE}/snap_counts/snap_counts_{season}.csv.gz')]:
            response = requests.get(url, timeout=120)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            frame = pd.read_csv(io.BytesIO(gzip.decompress(response.content)), low_memory=False)
            frame.to_parquet(data/name, index=False)
            manifest['files'][name] = dict(url=url, source_sha256=hashlib.sha256(response.content).hexdigest(), rows=len(frame))
    response = requests.get(f'{RELEASE}/schedules/games.csv.gz', timeout=120)
    response.raise_for_status()
    pd.read_csv(io.BytesIO(gzip.decompress(response.content)), low_memory=False).to_parquet(data/'schedule.parquet', index=False)
    manifest['files']['schedule.parquet'] = dict(url=f'{RELEASE}/schedules/games.csv.gz',
                                                 source_sha256=hashlib.sha256(response.content).hexdigest())
    (data/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


@contextmanager
def production_environment(data=DATA, injuries=None):
    """Run the production builder against frozen inputs with point-in-time-safe patches.

    The builder reads data/weekly_player_stats_{season}.parquet relative to the working
    directory; it runs inside DATA's parent layout. Weekly injury files and manual
    overrides are not archived for past seasons, so they are replaced by the archived
    snapshot for that week (when one exists) or by nothing; schedules come from the
    frozen nflverse file instead of a live request.
    """
    import nfl_data_py
    import make_player_prop_params as mpp
    work = data/'work'
    (work/'data').mkdir(parents=True, exist_ok=True)
    for f in data.glob('weekly_player_stats_*.parquet'):
        target = work/'data'/f.name
        if not target.exists():
            target.symlink_to(f)
    schedule = pd.read_parquet(data/'schedule.parquet')
    saved = (mpp.load_week_injuries, mpp.load_player_adjustments, nfl_data_py.import_schedules, os.getcwd())
    from injury_adjustments import normalize_injury_report
    mpp.load_week_injuries = lambda path, season, week: (normalize_injury_report(pd.read_csv(injuries, low_memory=False), season, week)
                                                         if injuries else pd.DataFrame())
    mpp.load_player_adjustments = lambda: {}
    nfl_data_py.import_schedules = lambda seasons: schedule[schedule.season.isin(seasons)].copy()
    os.chdir(work)
    try:
        yield mpp
    finally:
        mpp.load_week_injuries, mpp.load_player_adjustments, nfl_data_py.import_schedules, cwd = saved
        os.chdir(cwd)


def name_key(name):
    from common_markets import std_player_name, strip_generational_suffix
    return std_player_name(strip_generational_suffix(name))


def team_names():
    from make_player_prop_params import TEAM_TO_ABBREV
    names = {v: k for k, v in TEAM_TO_ABBREV.items()}
    names.setdefault('LAR', names.get('LA', 'Los Angeles Rams'))
    return names


def universe(season, week, data=DATA):
    """Pre-game candidate (player, market) pairs: role in 2 of the team's last 3 games."""
    stats = pd.read_parquet(data/f'weekly_player_stats_{season}.parquet')
    stats = stats[(stats.season_type == 'REG') & (stats.week < week)]
    schedule = pd.read_parquet(data/'schedule.parquet')
    games = schedule[(schedule.season == season) & (schedule.week == week) & (schedule.game_type == 'REG')]
    names = team_names()
    rows = []
    for _, g in games.iterrows():
        for team in (g.home_team, g.away_team):
            history = stats[stats.team == team]
            recent = sorted(history.week.unique())[-3:]
            if len(recent) < 2:
                continue
            window = history[history.week.isin(recent)]
            for (player, pid), p in window.groupby(['player_display_name', 'player_id']):
                for role, (column, minimum, markets) in ROLES.items():
                    played = p.groupby('week')[column].sum()
                    if (played >= minimum).sum() >= 2:
                        for market in markets:
                            rows.append(dict(player=player, player_id=pid, team=team, market_std=market, season=season,
                                             week=week, game_id=g.game_id, home_team=names.get(g.home_team, g.home_team),
                                             away_team=names.get(g.away_team, g.away_team),
                                             commence_time=f'{g.gameday}T{g.gametime}'))
    return pd.DataFrame(rows).drop_duplicates(['player', 'market_std', 'game_id'])


def reconstruct(cands, season, week, *, data=DATA, injuries=None):
    """Forecast parameters from the production builder with a strict season/week cutoff."""
    import career_baseline as cb
    from common_markets import strip_generational_suffix, ensure_param_schema, apply_priors_if_missing
    cands = cands.copy()
    cands['player'] = cands['player'].map(strip_generational_suffix)
    with production_environment(data, injuries) as mpp:
        logs = mpp.fetch_recent_game_logs(season, week)
        career = cb.load_career_logs(list(range(season-6, season+1)))
        future = [frame for frame in (logs, career) if frame is not None and not frame.empty and
                  ((frame['season'] > season) | ((frame['season'] == season) & (frame['week'] >= week))).any()]
        if logs is not None and not logs.empty and ((logs['season'] == season) & (logs['week'] >= week)).any():
            raise ValueError('Cutoff violated in current-season logs')
        defense = mpp.calculate_defensive_ratings(season, week)
        props = cands.rename(columns={})
        opponents = mpp.create_opponent_map(props, logs, career_df=career)
        home = mpp.create_home_away_map(props, logs, career_df=career)
        params = mpp.build_params(cands[['player', 'market_std', 'season', 'week']].drop_duplicates(), logs, season, week,
                                  defensive_ratings=defense, opponent_map=opponents, home_away_map=home, career_df=career)
        params = apply_priors_if_missing(ensure_param_schema(params))
    # Career frames legitimately contain later seasons on disk; build_params drops them.
    return params, dict(career_frames_with_future_rows=len(future))


def verify_cutoff(season=2025, week=8, data=DATA):
    """The builder must ignore every row on or after the target week, wherever it is passed."""
    import career_baseline as cb
    cands = universe(season, week, data).head(80)
    with production_environment(data) as mpp:
        logs = mpp.fetch_recent_game_logs(season, week)
        career = cb.load_career_logs(list(range(season-6, season+1)))
        stats = pd.read_parquet(data/f'weekly_player_stats_{season}.parquet')
        leaked = career[(career['season'] == season) & (career['week'] >= week)]
        base = mpp.build_params(cands[['player', 'market_std', 'season', 'week']].drop_duplicates(), logs, season, week, career_df=career)
        # Append future rows to the current-season logs too: results must not change.
        future = cb._clean_weekly(stats[stats.week >= week]).assign(recent_team=lambda d: d.get('team'))
        poisoned = mpp.build_params(cands[['player', 'market_std', 'season', 'week']].drop_duplicates(),
                                    pd.concat([logs, future], ignore_index=True), season, week, career_df=career)
    cols = ['player', 'market_std', 'mu', 'sigma', 'lam']
    a = base[cols].sort_values(cols[:2]).reset_index(drop=True)
    b = poisoned[cols].sort_values(cols[:2]).reset_index(drop=True)
    same = a.equals(b) or np.allclose(a[['mu', 'sigma', 'lam']].astype(float).fillna(-1), b[['mu', 'sigma', 'lam']].astype(float).fillna(-1))
    return dict(season=season, week=week, career_rows_after_cutoff_on_disk=int(len(leaked)),
                identical_with_future_rows_supplied=bool(same), players=int(cands.player.nunique()))


def outcomes(season, week, data=DATA):
    """Stat per (player, team) for the target week; participation from offensive snaps.

    An active player with snaps but no box-score row recorded zero; a player without
    offensive snaps is a non-participant (props void), never a loss.
    """
    stats = pd.read_parquet(data/f'weekly_player_stats_{season}.parquet')
    stats = stats[(stats.week == week) & (stats.season_type == 'REG')]
    snaps = pd.read_parquet(data/f'snap_counts_{season}.parquet')
    snaps = snaps[(snaps.week == week) & (snaps.game_type == 'REG')]
    played = {(name_key(r.player), r.team) for r in snaps.itertuples() if (r.offense_snaps or 0) > 0}
    values = {(name_key(r.player_display_name), r.team): r for r in stats.itertuples()}
    return played, values


def outcome_value(played, values, player, team, market):
    key = (name_key(player), team)
    if key not in played:
        return None
    row = values.get(key)
    return float(getattr(row, STAT[market]) or 0) if row is not None else 0.0


# ---------------------------------------------------------------- probabilities
def grid_rows(params, cands, season, week, data=DATA):
    from market_math import outcome_probabilities
    played, values = outcomes(season, week, data)
    teams = cands.drop_duplicates(['player', 'market_std']).set_index(['player', 'market_std'])
    rows = []
    for r in params.itertuples():
        if r.market_std not in GRIDS or str(getattr(r, 'no_real_data', False)).lower() in ('true', '1'):
            continue
        try:
            c = teams.loc[(r.player, r.market_std)]
        except KeyError:
            continue
        y = outcome_value(played, values, r.player, c.team, r.market_std)
        if y is None:
            continue
        for line in GRIDS[r.market_std]:
            over, push = outcome_probabilities(r.market_std, 'over', line, r.mu, r.sigma, r.lam)
            if not (isinstance(over, float) and math.isfinite(over)) or not BAND[0] <= over <= BAND[1]:
                continue
            for side, p in (('over', over), ('under', 1-over)):
                rows.append(dict(season=season, week=week, game_id=c.game_id, player=r.player, team=c.team,
                                 market=r.market_std, line=line, side=side, raw=p, mu=r.mu,
                                 outcome=int(y > line) if side == 'over' else int(y < line)))
    return rows


def apply_curve(calibration, market, p):
    if calibration is None or market not in calibration.get('_eligible_markets', []):
        return p
    curve = calibration.get('markets', {}).get(market, calibration.get('_pooled_fallback'))
    return float(np.interp(p, curve['x'], curve['y'])) if curve else p


def fit(frame, provenance):
    """Same specification as scripts/fit_calibration.py (isotonic, pooled fallback below 500)."""
    from fit_calibration import fit_curve
    result = {'_meta': dict(fitted_on_seasons=sorted(int(s) for s in frame.season.unique()),
                            fitted_on_weeks=sorted(int(w) for w in frame.week.unique()),
                            n_rows=int(len(frame)), population='fixed_line_grid_reconstructed_forecasts'),
              '_pooled_fallback': fit_curve(frame.raw.values, frame.outcome.values),
              '_eligible_markets': [m for m in GRIDS if m != 'pass_tds'] + ['pass_tds'], 'markets': {},
              '_market_counts': {}, '_provenance': provenance}
    for market, sub in frame.groupby('market'):
        result['_market_counts'][market] = dict(rows=int(len(sub)), games=int(sub.game_id.nunique()),
                                                forecasts=int(sub[['game_id', 'player']].drop_duplicates().shape[0]))
        if len(sub) >= MIN_MARKET_N:
            result['markets'][market] = fit_curve(sub.raw.values, sub.outcome.values)
    return result


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [round(centre-half, 4), round(centre+half, 4)]


def bins(p, y):
    out = []
    for low in np.arange(0, 1, .1):
        mask = (p >= low) & ((p < low+.1) if low < .9 else (p <= 1))
        n = int(mask.sum())
        if n:
            k = int(y[mask].sum())
            interval = wilson(k, n)
            out.append(dict(predicted=round(float(p[mask].mean()), 4), observed=round(k/n, 4), n=n, observed_ci95=interval,
                            outside_interval=bool(n >= 30 and not interval[0] <= float(p[mask].mean()) <= interval[1])))
    return out


def clustered(frame, columns, reps=1000, seed=SEED):
    """Game-clustered bootstrap of mean squared error for each column and pairwise differences."""
    groups = frame.groupby('game_id')
    ids = list(groups.groups)
    sq = {c: groups.apply(lambda g: float(((g[c]-g.outcome)**2).sum())).reindex(ids).values for c in columns}
    n = groups.size().reindex(ids).values.astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), (reps, len(ids)))
    total = n[draws].sum(axis=1)
    est = {c: sq[c][draws].sum(axis=1)/total for c in columns}
    out = {c: dict(brier=round(float(sq[c].sum()/n.sum()), 6), ci95=[round(float(v), 6) for v in np.quantile(est[c], [.025, .975])])
           for c in columns}
    for i, a in enumerate(columns):
        for b in columns[i+1:]:
            diff = est[a]-est[b]
            out[f'{a}_minus_{b}'] = dict(difference=round(float((sq[a].sum()-sq[b].sum())/n.sum()), 6),
                                         ci95=[round(float(v), 6) for v in np.quantile(diff, [.025, .975])])
    return out


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1-1e-6)
    return round(float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))), 6)


def evaluate(frame, columns, label):
    """Per market and overall, with explicit counts of games, forecasts and rows."""
    report = {}
    for market, sub in [('all', frame)] + list(frame.groupby('market')):
        if sub.empty:
            continue
        entry = dict(games=int(sub.game_id.nunique()), forecasts=int(sub[['game_id', 'player', 'market']].drop_duplicates().shape[0]),
                     rows=int(len(sub)), lines=sorted(float(x) for x in sub.line.unique()) if market != 'all' else None,
                     scores=clustered(sub, columns), log_loss={c: log_loss(sub[c].values, sub.outcome.values) for c in columns})
        over = sub[sub.side == 'over']
        for c in columns:
            entry.setdefault('calibration_bins', {})[c] = bins(sub[c].values, sub.outcome.values)
        if len(over):
            # Directional check on the Over side: mean forecast minus observed frequency.
            entry['over_side_bias'] = {c: round(float(over[c].mean()-over.outcome.mean()), 4) for c in columns}
        report[market] = entry
    return dict(label=label, markets=report)


# ---------------------------------------------------------------- 2026 book lines
def book_offers(week, path):
    df = pd.read_csv(ROOT/path, low_memory=False)
    df = df[df.market_std.isin(STAT) & df.name.str.lower().isin(['over', 'under']) & df.point.notna() & df.price.notna()]
    df = df.assign(side=df.name.str.lower(), quoted_at=pd.to_datetime(df.last_update, utc=True),
                   start=pd.to_datetime(df.commence_time, utc=True))
    # Point-in-time: only quotes timestamped before the game started.
    return df[df.quoted_at < df.start]


def devig_consensus(df):
    """Same-book over/under pairs at the exact line, de-vigged, averaged across books."""
    def implied(price):
        return 100/(price+100) if price > 0 else -price/(-price+100)
    pairs = df.pivot_table(index=['game_id', 'player', 'market_std', 'point', 'bookmaker'], columns='side', values='price', aggfunc='last').dropna()
    pairs['over_fair'] = pairs.apply(lambda r: implied(r['over'])/(implied(r['over'])+implied(r['under'])), axis=1)
    consensus = pairs.groupby(level=[0, 1, 2, 3]).agg(market_over=('over_fair', 'mean'), books=('over_fair', 'size'))
    return consensus.reset_index()


def book_rows(week, path, calibrations, data=DATA, injuries=None):
    from market_math import outcome_probabilities
    from make_player_prop_params import TEAM_TO_ABBREV
    offers = book_offers(week, path)
    cands = offers[['player', 'market_std', 'home_team', 'away_team', 'game_id']].drop_duplicates()
    cands = cands.assign(season=2026, week=week)
    params, _ = reconstruct(cands, 2026, week, data=data, injuries=injuries)
    params = params.set_index(['player', 'market_std'])
    consensus = devig_consensus(offers)
    played, values = outcomes(2026, week, data)
    schedule = pd.read_parquet(data/'schedule.parquet')
    sched = schedule[(schedule.season == 2026) & (schedule.week == week)]
    # Schedules use abbreviations; offers use full names. Join on abbreviations ('LA' is the Rams).
    abbrev = lambda team: {'LAR': 'LA'}.get(TEAM_TO_ABBREV.get(team, team), TEAM_TO_ABBREV.get(team, team))
    team_of = {(g.home_team, g.away_team): (g.home_team, g.away_team, g.game_id) for g in sched.itertuples()}
    stats = pd.read_parquet(data/'weekly_player_stats_2026.parquet')
    roster = {name_key(r.player_display_name): r.team for r in stats[stats.week < week].sort_values('week').itertuples()}
    rows = []
    for c in consensus.itertuples():
        offer = offers[(offers.game_id == c.game_id) & (offers.player == c.player)].iloc[0]
        teams = team_of.get((abbrev(offer.home_team), abbrev(offer.away_team)))
        team = roster.get(name_key(c.player))
        if not teams or team not in teams[:2]:
            continue
        y = outcome_value(played, values, c.player, team, c.market_std)
        try:
            p = params.loc[(c.player, c.market_std)]
        except KeyError:
            continue
        if y is None or str(p.get('no_real_data', False)).lower() in ('true', '1'):
            continue
        over, push = outcome_probabilities(c.market_std, 'over', c.point, p.mu, p.sigma, p.lam)
        if not isinstance(over, float) or not math.isfinite(over) or push:
            continue  # Integer lines (pushes) are excluded from binary scoring.
        n_offers = int(((offers.game_id == c.game_id) & (offers.player == c.player) & (offers.market_std == c.market_std) & (offers.point == c.point)).sum())
        for side, raw, market in (('over', over, c.market_over), ('under', 1-over, 1-c.market_over)):
            row = dict(season=2026, week=week, game_id=teams[2], player=c.player, market=c.market_std, line=float(c.point),
                       side=side, raw=raw, market_prob=market, books=int(c.books), offers=n_offers,
                       outcome=int(y > c.point) if side == 'over' else int(y < c.point))
            for name, cal in calibrations.items():
                row[name] = apply_curve(cal, c.market_std, raw)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- pipeline
def book_evaluation(out, deploy, legacy, data=DATA):
    """Exact-line comparison at archived, provider-timestamped 2026 offers."""
    books = []
    for week, path in BOOK_WEEKS.items():
        injuries = ROOT/f'reports/nfl-weekly/2026/week-{week}/pregame/injuries.csv'
        books += book_rows(week, path, dict(refit=deploy, legacy=legacy), data,
                           injuries=injuries if injuries.exists() else None)
    book = pd.DataFrame(books)
    book.to_csv(out/'book-line-forecasts.csv.gz', index=False, compression='gzip')
    if book.empty:
        return dict(status='unavailable', reason='no settled offers matched')
    report = evaluate(book, ['raw', 'refit', 'legacy', 'market_prob'], '2026 weeks 2-4 exact-line comparison (seen weeks)')
    report['offers'] = int(book.drop_duplicates(['game_id', 'player', 'market', 'line']).offers.sum())
    report['weeks'] = sorted(int(w) for w in book.week.unique())
    report['market_basis'] = 'same-book over/under pairs at the exact line, de-vigged, averaged across books; quotes timestamped before kickoff'
    return report


def rerun_books(run_id, data=DATA, reports=REPORTS):
    out = reports/run_id
    report = json.loads((out/'validation.json').read_text())
    deploy = json.loads((out/'calibration-deployment.json').read_text())
    legacy = json.loads((ROOT/'models/nfl_prop_calibration.json').read_text()) if '_provenance' not in json.loads(
        (ROOT/'models/nfl_prop_calibration.json').read_text()) else json.loads(
        __import__('subprocess').run(['git', 'show', 'HEAD:models/nfl_prop_calibration.json'], capture_output=True, text=True, cwd=ROOT).stdout)
    logging.disable(logging.WARNING)
    report['book_line_evaluation'] = book_evaluation(out, deploy, legacy, data)
    (out/'validation.json').write_text(json.dumps(report, indent=2))
    (out/'README.md').write_text(readme(report))
    return report


def model_identity():
    import make_player_prop_params as mpp
    files = ['scripts/make_player_prop_params.py', 'scripts/career_baseline.py', 'scripts/market_math.py',
             'scripts/common_markets.py', 'scripts/injury_adjustments.py']
    return dict(model_version=mpp.MODEL_VERSION,
                code_sha256=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in files)).hexdigest(), code_files=files)


def run(run_id, data=DATA, reports=REPORTS):
    logging.disable(logging.WARNING)
    out = reports/run_id
    out.mkdir(parents=True, exist_ok=True)
    identity = model_identity()
    cutoff = verify_cutoff(data=data)
    if not cutoff['identical_with_future_rows_supplied']:
        raise SystemExit('Forecast cutoff verification failed; no calibration is fitted.')
    frames = {}
    for season in sorted(set(FIT_DEPLOY+EVAL)):
        rows = []
        for week in WEEKS:
            cands = universe(season, week, data)
            if cands.empty:
                continue
            params, _ = reconstruct(cands, season, week, data=data)
            rows += grid_rows(params, cands, season, week, data)
            print(f'{season} week {week}: {len(rows):,} grid rows', flush=True)
        frames[season] = pd.DataFrame(rows)
    grid = pd.concat(frames.values(), ignore_index=True)
    grid.to_csv(out/'grid-forecasts.csv.gz', index=False, compression='gzip')
    legacy = json.loads((ROOT/'models/nfl_prop_calibration.json').read_text())
    base = dict(identity, cutoff_policy='strict season/week: games before the target week only',
                cutoff_verification=cutoff, created_at=datetime.now(timezone.utc).isoformat(), run_id=run_id,
                reproduce=f'python scripts/nfl_validation.py fetch && python scripts/nfl_validation.py run --run-id {run_id}',
                input_manifest=json.loads((data/'manifest.json').read_text()) if (data/'manifest.json').exists() else None)
    fit_eval = grid[grid.season.isin(FIT_EVAL)]
    holdout = grid[grid.season.isin(EVAL)].copy()
    eval_cal = fit(fit_eval, dict(base, role='evaluation', fit_window=dict(seasons=FIT_EVAL, weeks=[min(WEEKS), max(WEEKS)]),
                                  evaluation_window=dict(seasons=EVAL, weeks=[min(WEEKS), max(WEEKS)])))
    reference = fit_eval.groupby(['market', 'line', 'side']).outcome.mean().rename('reference')
    holdout = holdout.join(reference, on=['market', 'line', 'side'])
    holdout['reference'] = holdout['reference'].fillna(fit_eval.outcome.mean())
    holdout['refit'] = [apply_curve(eval_cal, m, p) for m, p in zip(holdout.market, holdout.raw)]
    holdout['legacy'] = [apply_curve(legacy, m, p) for m, p in zip(holdout.market, holdout.raw)]
    grid_report = evaluate(holdout, ['raw', 'refit', 'legacy', 'reference'], 'fixed-grid 2025 development holdout (calibration only)')
    deploy = fit(grid[grid.season.isin(FIT_DEPLOY)], dict(base, role='deployment',
                 fit_window=dict(seasons=FIT_DEPLOY, weeks=[min(WEEKS), max(WEEKS)]),
                 evaluation_window=dict(out_of_time='2026 weeks 2-4 book lines', prospective='2026 week 5 onward')))
    book_report = book_evaluation(out, deploy, legacy, data)
    deploy['_provenance']['evaluation'] = dict(report=f'reports/nfl-validation/{run_id}/validation.json',
        status='chronological_development_holdout_and_seen_out_of_time_book_lines; no untouched holdout; prospective from 2026 week 5')
    (out/'calibration-evaluation.json').write_text(json.dumps(eval_cal, indent=2))
    (out/'calibration-deployment.json').write_text(json.dumps(deploy, indent=2))
    report = dict(schema='nfl-validation-1', run_id=run_id, created_at=base['created_at'], model=identity,
        calibration=dict(evaluation_artifact_sha256=sha256(out/'calibration-evaluation.json'),
                         deployment_artifact_sha256=sha256(out/'calibration-deployment.json'),
                         legacy_artifact_sha256=hashlib.sha256((ROOT/'models/nfl_prop_calibration.json').read_bytes()).hexdigest()),
        data_cutoff=dict(stats_through=f"2026 week {int(pd.read_parquet(data/'weekly_player_stats_2026.parquet').week.max())}",
                         input_manifest=base['input_manifest']),
        windows=dict(hyperparameters='fixed during 2025 development (2025 not untouched)',
                     calibration_fit_for_evaluation=FIT_EVAL, calibration_evaluation=EVAL,
                     deployment_fit=FIT_DEPLOY, weeks=[min(WEEKS), max(WEEKS)],
                     book_line_check='2026 weeks 2-4 (provider-timestamped archives; seen during model development)',
                     prospective='2026 week 5 onward'),
        cutoff_verification=cutoff, grid=dict(lines=GRIDS, probability_band=BAND, roles=ROLES), grid_evaluation=grid_report,
        book_line_evaluation=book_report, reproduce=base['reproduce'],
        limitations=['No untouched historical holdout: 2025 informed hyperparameters and 2026 weeks 1-4 informed the October 4 model change.',
                     'Fixed line grids are not bookmaker lines; grid accuracy is not demonstrated performance against prices.',
                     'Historical injury reports and manual overrides are not archived for 2024-2025 and are omitted; 2026 weeks with archived injury snapshots use them.',
                     'Candidate universe approximates which players receive props using earlier usage only.',
                     'Rows from the same game are correlated; intervals resample whole games.',
                     'Integer book lines (pushes) are excluded from binary scoring.',
                     'No returns at offered prices are claimed from this report.'])
    (out/'validation.json').write_text(json.dumps(report, indent=2))
    (out/'README.md').write_text(readme(report))
    return report


def difference(scores, a, b):
    """a minus b from a clustered() result, whichever order the pair was stored in."""
    if f'{a}_minus_{b}' in scores:
        return scores[f'{a}_minus_{b}']
    d = scores[f'{b}_minus_{a}']
    return dict(difference=-d['difference'], ci95=[-d['ci95'][1], -d['ci95'][0]])


def readme(report):
    lines = [f"# NFL prop validation · {report['run_id']}", '',
             f"Model `{report['model']['model_version']}` (code `{report['model']['code_sha256'][:12]}`). Reproduce: `{report['reproduce']}`.", '',
             'Brier score: 0 is perfect; a constant 50% forecast scores 0.25 on binary outcomes. Lower is better.', '',
             '## Fixed-grid 2025 holdout (calibration fitted on 2024)', '',
             '| Market | Games | Forecasts | Rows | Raw | Refit | Legacy | Empirical reference | Refit − raw (95% CI) |',
             '|---|---|---|---|---|---|---|---|---|']
    for market, m in report['grid_evaluation']['markets'].items():
        s = m['scores']
        d = difference(s, 'refit', 'raw')
        lines.append(f"| {market} | {m['games']} | {m['forecasts']} | {m['rows']} | {s['raw']['brier']:.4f} | {s['refit']['brier']:.4f} | "
                     f"{s['legacy']['brier']:.4f} | {s['reference']['brier']:.4f} | {d['difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    b = report.get('book_line_evaluation')
    if b and b.get('markets'):
        lines += ['', '## 2026 weeks 2-4 at exact archived book lines (deployment artifact; seen weeks)', '',
                  '| Market | Games | Forecasts | Rows | Raw | Refit | Legacy | Market (de-vigged) | Refit − market (95% CI) |',
                  '|---|---|---|---|---|---|---|---|---|']
        for market, m in b['markets'].items():
            s = m['scores']; d = difference(s, 'refit', 'market_prob')
            lines.append(f"| {market} | {m['games']} | {m['forecasts']} | {m['rows']} | {s['raw']['brier']:.4f} | {s['refit']['brier']:.4f} | "
                         f"{s['legacy']['brier']:.4f} | {s['market_prob']['brier']:.4f} | {d['difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    diag = report.get('book_line_calibration_diagnostic')
    if diag:
        d = diag['book_fit_minus_market']
        lines += ['', f"Book-line calibration diagnostic (fit on weeks {diag['fit_weeks']}, scored on week {diag['evaluation_week']}, "
                  f"{diag['games']} games): Brier {diag['brier']['book_fit']:.4f} vs market {diag['brier']['market_prob']:.4f} and constant 50% "
                  f"{diag['brier']['constant']:.4f}; model − market {d['difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]. Not installed."]
    decision = report.get('decision')
    if decision:
        lines += ['', '## Decision', '', f"- Installed: {decision['installed']}. {decision['production_artifact']}.",
                  f"- Reason: {decision['reason']}", f"- Effect: {decision['top_picks_effect']}",
                  f"- Requalification: {decision['requalification']}"]
    lines += ['', '## Limitations', ''] + ['- '+x for x in report['limitations']]
    lines += ['', f"Cutoff verification: {json.dumps(report['cutoff_verification'])}", '']
    return '\n'.join(lines)


def install(run_id, reports=REPORTS):
    """Copy the evaluated deployment artifact into production after compatibility checks."""
    import make_player_prop_params as mpp
    artifact = json.loads((reports/run_id/'calibration-deployment.json').read_text())
    if artifact['_provenance']['model_version'] != mpp.MODEL_VERSION:
        raise SystemExit('Deployment artifact was fitted for another model version')
    (ROOT/'models/nfl_prop_calibration.json').write_text(json.dumps(artifact, indent=2)+'\n')
    return artifact['_provenance']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('fetch')
    for name in ('run', 'install', 'books'):
        p = sub.add_parser(name)
        p.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.command == 'fetch':
        print(json.dumps(fetch(), indent=2)[:2000])
    elif args.command == 'run':
        report = run(args.run_id)
        print((REPORTS/args.run_id/'README.md').read_text())
    elif args.command == 'books':
        rerun_books(args.run_id)
        print((REPORTS/args.run_id/'README.md').read_text())
    else:
        print(json.dumps(install(args.run_id), indent=2))


if __name__ == '__main__':
    main()

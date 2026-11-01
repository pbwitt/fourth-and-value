#!/usr/bin/env python3
"""NFL spread shadow for player props: logged and graded, never used for picks.

The prop model ignores game script. Over weeks 3-7 of 2026, players on underdogs beat
their projections on passing and receiving props by about 0.5-1.1% per point of spread.
This records what the projection would have been with that adjustment, from the frozen
pregame archive, so it can be graded on weeks the coefficients never saw.

  build   reports/nfl-weekly/<season>/week-<w>/pregame/{props,totals_spreads}.csv
          -> reports/nfl-weekly/<season>/week-<w>/shadow/spread_shadow.csv (+ manifest.json)
  report  every week with both a shadow file and review/probability_comparison.csv
          -> reports/nfl-weekly/<season>/spread-shadow/{summary.json,README.md}

Nothing here writes to data/props, docs/, Top Picks or the published probabilities.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_math import outcome_probabilities  # noqa: E402
from display_time import eastern_label         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/nfl_spread_shadow.json'


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def week_dir(root, season, week):
    return Path(root) / 'reports/nfl-weekly' / str(season) / f'week-{week}'


def player_is_home(diagnostics):
    """The projection trace records whether the player's team is at home."""
    try:
        value = json.loads(diagnostics).get('home')
    except (TypeError, ValueError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def consensus_home_spread(lines):
    """Median home spread per event across books in the frozen snapshot."""
    lines = lines.dropna(subset=['event_id', 'spread_home_line'])
    return lines.groupby('event_id').spread_home_line.median()


def multiplier(market, team_spread, config):
    pct = config['pct_per_point'].get(market)
    if pct is None or not math.isfinite(team_spread):
        return np.nan
    lo, hi = config['multiplier_limits']
    return float(np.clip(1 + pct / 100 * team_spread, lo, hi))


def p_over(market, line, mu, sigma, lam):
    return outcome_probabilities(market, 'over', line, mu, sigma, lam)[0]


def build_shadow(props, lines, config):
    """One row per (player, market) the shadow covers; inputs are not modified."""
    props = props[props.market_std.isin(config['pct_per_point']) & props.mu.notna()].copy()
    if 'projection_diagnostics' not in props.columns:
        props['projection_diagnostics'] = None
    spreads = consensus_home_spread(lines)
    out = []
    for (player, market), g in props.groupby(['player', 'market_std'], sort=True):
        first = g.iloc[0]
        home = player_is_home(first.projection_diagnostics)
        home_line = spreads.get(first.game_id, np.nan)
        team_spread = np.nan if home is None or not math.isfinite(home_line) else (home_line if home else -home_line)
        mult = multiplier(market, team_spread, config)
        line = g.consensus_line.median() if 'consensus_line' in g and g.consensus_line.notna().any() else g.point.median()
        overs = g[(g.point == line) & (g.name.astype(str).str.lower() == 'over')]
        mkt = overs.prob_devig.median() if 'prob_devig' in overs and len(overs) else np.nan
        mu, sigma, lam = float(first.mu), float(first.sigma) if pd.notna(first.sigma) else np.nan, \
            float(first.lam) if pd.notna(first.lam) else np.nan
        shadow_mu = mu * mult if math.isfinite(mult) else np.nan
        shadow_lam = lam * mult if math.isfinite(mult) and math.isfinite(lam) else np.nan
        out.append(dict(
            season=int(first.season), week=int(first.week), game_id=first.game_id, commence_time=first.commence_time,
            game=first.game, player=player, market_std=market, player_home=home, team_spread=team_spread,
            mu=mu, sigma=sigma, lam=lam, shadow_multiplier=mult, shadow_mu=shadow_mu, shadow_lam=shadow_lam,
            consensus_line=line, model_p_over=p_over(market, line, mu, sigma, lam),
            shadow_p_over=p_over(market, line, shadow_mu, sigma, shadow_lam) if math.isfinite(mult) else np.nan,
            market_p_over=mkt, shadow_version=config['version'],
            skipped=None if math.isfinite(mult) else ('no home/away trace' if home is None else 'no consensus spread')))
    return pd.DataFrame(out)


def build(season, week, root=ROOT, config_path=CONFIG, now=None):
    config = load_config(config_path)
    pregame = week_dir(root, season, week) / 'pregame'
    sources = {'props': pregame / 'props.csv', 'lines': pregame / 'totals_spreads.csv',
               'manifest': pregame / 'manifest.json'}
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise SystemExit('Spread shadow needs the frozen pregame archive; missing: ' + ', '.join(missing))
    existing = week_dir(root, season, week) / 'shadow/manifest.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        same_inputs = {i['path']: i['sha256'] for i in saved.get('inputs', [])} == \
            {str(p.relative_to(root)): sha(p) for p in (sources['props'], sources['lines'])}
        if same_inputs and saved.get('config_sha256') == sha(config_path):
            print(json.dumps(dict(status='existing', season=season, week=week)))
            return saved
    shadow = build_shadow(pd.read_csv(sources['props'], low_memory=False), pd.read_csv(sources['lines']), config)
    dest = week_dir(root, season, week) / 'shadow'
    dest.mkdir(parents=True, exist_ok=True)
    shadow.to_csv(dest / 'spread_shadow.csv', index=False)
    built = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert('UTC')
    snapshot = json.loads(sources['manifest'].read_text()).get('snapshot_at')
    manifest = dict(schema=1, version=config['version'], season=season, week=week, built_at=built.isoformat(),
                    pregame_snapshot_at=snapshot, in_sample=week < config['first_prospective_week'],
                    config_sha256=sha(config_path), rows=int(len(shadow)), adjusted=int(shadow.skipped.isna().sum()) if len(shadow) else 0,
                    inputs=[dict(path=str(p.relative_to(root)), sha256=sha(p)) for p in (sources['props'], sources['lines'])],
                    output=dict(path=str((dest / 'spread_shadow.csv').relative_to(root)), sha256=sha(dest / 'spread_shadow.csv')))
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='built', season=season, week=week, rows=manifest['rows'], adjusted=manifest['adjusted'],
                          in_sample=manifest['in_sample'])))
    return manifest


def log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(frame):
    y = (frame.actual > frame.consensus_line).astype(float).values
    row = dict(n=int(len(frame)),
               model_log_loss=log_loss(frame.model_p_over, y), shadow_log_loss=log_loss(frame.shadow_p_over, y),
               model_brier=float(np.mean((frame.model_p_over - y) ** 2)),
               shadow_brier=float(np.mean((frame.shadow_p_over - y) ** 2)),
               model_side=float(np.mean((frame.model_p_over > 0.5) == (y == 1))),
               shadow_side=float(np.mean((frame.shadow_p_over > 0.5) == (y == 1))))
    mk = frame.dropna(subset=['market_p_over'])
    row['market_n'] = int(len(mk))
    row['market_log_loss'] = log_loss(mk.market_p_over, (mk.actual > mk.consensus_line).astype(float).values) if len(mk) else None
    row['shadow_minus_model_log_loss'] = row['shadow_log_loss'] - row['model_log_loss']
    return row


def graded(root, season, config):
    frames = []
    for wd in sorted((Path(root) / 'reports/nfl-weekly' / str(season)).glob('week-*')):
        shadow_path, review_path = wd / 'shadow/spread_shadow.csv', wd / 'review/probability_comparison.csv'
        if not (shadow_path.exists() and review_path.exists()):
            continue
        shadow = pd.read_csv(shadow_path)
        actuals = pd.read_csv(review_path, low_memory=False)[['player', 'market_std', 'actual']].dropna()
        actuals = actuals.drop_duplicates(['player', 'market_std'])
        g = shadow[shadow.skipped.isna()].merge(actuals, on=['player', 'market_std'], how='inner')
        frames.append(g[g.actual != g.consensus_line])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['prospective'] = out.week >= config['first_prospective_week']
    return out


def report(season, root=ROOT, config_path=CONFIG, now=None):
    config = load_config(config_path)
    g = graded(root, season, config)
    dest = Path(root) / 'reports/nfl-weekly' / str(season) / 'spread-shadow'
    dest.mkdir(parents=True, exist_ok=True)
    generated = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert('UTC')
    weeks = [dict(week=int(w), in_sample=bool(w < config['first_prospective_week']), **score(f))
             for w, f in g.groupby('week')] if len(g) else []
    pro = g[g.prospective] if len(g) else g
    summary = dict(schema=1, version=config['version'], season=season, generated_at=generated.isoformat(),
                   first_prospective_week=config['first_prospective_week'], weeks=weeks,
                   prospective=score(pro) if len(pro) else None,
                   prospective_by_market={m: score(f) for m, f in pro.groupby('market_std')} if len(pro) else {},
                   note='Raw (uncalibrated) probabilities at the consensus line, pushes excluded. Lower log loss is better. '
                        'In-sample weeks trained the coefficients and are not evidence.')
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (dest / 'README.md').write_text(render(summary, generated))
    print(json.dumps(dict(status='reported', season=season, weeks=[w['week'] for w in weeks],
                          prospective_n=summary['prospective']['n'] if summary['prospective'] else 0)))
    return summary


def render(summary, generated):
    lines = [f"# NFL spread shadow, {summary['season']}", '',
             f"Version `{summary['version']}`. Generated {eastern_label(generated.to_pydatetime())}.", '',
             'A logged-only adjustment to passing and receiving prop projections for the team spread. '
             'It never changes published projections, probabilities or picks. See `config/nfl_spread_shadow.json`.', '',
             'Log loss uses raw (uncalibrated) probabilities at the consensus line with pushes excluded; lower is better. '
             f"Weeks before {summary['first_prospective_week']} trained the coefficients and are shown for reference only.", '',
             '| Week | Sample | Props | Current | Shadow | Shadow − current | Market | Side hit (current → shadow) |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for w in summary['weeks']:
        mkt = f"{w['market_log_loss']:.4f}" if w['market_log_loss'] is not None else '—'
        lines.append(f"| {w['week']} | {'in-sample' if w['in_sample'] else 'prospective'} | {w['n']} | {w['model_log_loss']:.4f} | "
                     f"{w['shadow_log_loss']:.4f} | {w['shadow_minus_model_log_loss']:+.4f} | {mkt} | "
                     f"{w['model_side']:.1%} → {w['shadow_side']:.1%} |")
    p = summary['prospective']
    lines.append('')
    if p:
        lines.append(f"**Prospective total:** {p['n']} props, log loss {p['model_log_loss']:.4f} current vs "
                     f"{p['shadow_log_loss']:.4f} shadow ({p['shadow_minus_model_log_loss']:+.4f}).")
    else:
        lines.append('**Prospective total:** no graded prospective weeks yet.')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--season', type=int, required=True)
    b.add_argument('--week', type=int, required=True)
    r = sub.add_parser('report')
    r.add_argument('--season', type=int, required=True)
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.season, a.week)
    else:
        report(a.season)


if __name__ == '__main__':
    main()

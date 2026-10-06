"""Render reports/matchups/README.md from the three backtest JSON files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/matchups'


def ci(x, scale=1., digits=4):
    if not x:
        return '—'
    return f"{x['mean'] * scale:+.{digits}f} ({x['lo'] * scale:+.{digits}f} to {x['hi'] * scale:+.{digits}f})"


def verdict(x, lower_is_better=True):
    """Better / worse / no detectable change from a 95% interval on (variant - reference)."""
    if not x:
        return ''
    if x['hi'] < 0:
        return 'better' if lower_is_better else 'worse'
    if x['lo'] > 0:
        return 'worse' if lower_is_better else 'better'
    return 'no detectable change'


def nhl(r):
    lines = ['## NHL player props', '',
             f"Baseline: {r['baseline']}. Test: {r['test']['season']}, {r['test']['rows']:,} player-games in {r['test']['games']:,} games. "
             'Count log loss (lower is better); change vs. production with a 95% interval from resampling games.', '',
             '| Variant | Shots log loss | vs production | Points log loss | vs production |', '|---|---|---|---|---|']
    for name, v in r['test']['results'].items():
        s, p = v['shots'], v['points']
        lines.append(f"| {name} | {s['nll']:.4f} | {ci(s['nll_vs_production'])} | {p['nll']:.4f} | {ci(p['nll_vs_production'])} |")
    lines += ['', 'Splits measured against the control (the same forecast without the split):', '',
              '| Split | Shots | Goals | Assists | Points |', '|---|---|---|---|---|']
    for name in ('+ player home/away split', '+ player vs opponent', '+ both splits'):
        v = r['test']['results'][name]
        lines.append(f"| {name} | " + ' | '.join(f"{verdict(v[s]['nll_vs_control'])} {ci(v[s]['nll_vs_control'])}" for s in ('shots', 'goals', 'assists', 'points')) + ' |')
    lines += ['', f"League home/away factor (training seasons): {r['league_home_factor']}.",
              f"Chosen shrinkage (expected-count units of prior; 'none' = the history is ignored): {r['k']}.",
              f"Weight the history earns: {r['weight_examples']}.", '', '**Matchup streaks** (test season, next game vs the same opponent):', '']
    for fam, rows in r['matchup_streaks'].items():
        for x in rows:
            lines.append(f"- {fam}: {x['group']}: {x['rows']:,} games; earlier ratio {x['prior_ratio']}; next game actual ÷ forecast {x['next_game_actual_over_forecast']}")
    return lines


def nfl(r):
    lines = ['## NFL player props', '', f"Baseline: {r['baseline']}. Seasons: {r['seasons']}. "
             'Squared error of the projection (lower is better) and Brier score at a half-point line on our production number.', '',
             '| Market | Test rows | Fitted home ÷ away (production) | Home/away fitted vs fixed | Player home/away split | Player vs opponent | Naive vs opponent |',
             '|---|---|---|---|---|---|---|']
    for m, x in r['markets'].items():
        t = x['test']
        fh = x['fitted_home_factor']
        lines.append(f"| {m} | {x['rows']['test']:,} | {fh['home']:.3f} / {fh['away']:.3f} ({x['production_home_multiplier']:.2f} / {2 - x['production_home_multiplier']:.2f}) | "
                     f"{verdict(t['+ fitted league home/away']['sq_error_vs_production'])} {ci(t['+ fitted league home/away']['sq_error_vs_production'], digits=2)} | "
                     f"{verdict(t['+ player home/away split']['sq_error_vs_control'])} {ci(t['+ player home/away split']['sq_error_vs_control'], digits=2)} | "
                     f"{verdict(t['+ player vs opponent']['sq_error_vs_control'])} {ci(t['+ player vs opponent']['sq_error_vs_control'], digits=2)} | "
                     f"{verdict(t['naive: fitted home + vs opponent only']['sq_error_vs_production'])} {ci(t['naive: fitted home + vs opponent only']['sq_error_vs_production'], digits=2)} |")
    lines += ['', 'Chosen shrinkage, in games of prior (none = ignored), and Brier score at the line:', '',
              '| Market | k overall / venue / vs opponent | Brier production | Brier + vs opponent | Brier + home/away split |', '|---|---|---|---|---|']
    for m, x in r['markets'].items():
        t, k = x['test'], x['k_games']
        lines.append(f"| {m} | {k['overall']} / {k['home/away split']} / {k['vs opponent']} | {t['production']['brier_at_line']:.4f} | "
                     f"{t['+ player vs opponent']['brier_at_line']:.4f} | {t['+ player home/away split']['brier_at_line']:.4f} |")
    lines += ['', '**Matchup streaks** (test seasons, next game vs the same opponent):', '']
    for m, x in r['markets'].items():
        for s in x['matchup_streaks']:
            lines.append(f"- {m}: {s['group']}: {s['rows']:,} games; earlier ratio {s['prior_ratio']}; next game actual ÷ forecast {s['next_game_actual_over_forecast']}")
    story = r['markets'].get('pass_yds', {}).get('storyline', {})
    for title, games in story.items():
        lines += ['', f'**{title}** (passing yards, every regular-season meeting in the data):', '',
                  '| Season | Week | Venue | Yards | Forecast | Earlier meetings | Earlier actual ÷ forecast |', '|---|---|---|---|---|---|---|']
        for g in games:
            lines.append(f"| {g['season']} | {g['week']} | {g['venue']} | {g['actual']:.0f} | {g['forecast']:.0f} | {g['prior_games_vs_opp']} | {g['prior_ratio_vs_opp']} |")
    return lines


def mlb(r):
    lines = ['## MLB plate appearances', '', f"Source: {r['source']}. {r['plate_appearances']:,} plate appearances. Seasons: {r['seasons']}. "
             'Log loss per plate appearance (lower is better), change vs. the production-like baseline.', '',
             '| Variant | Hit | Strikeout | Walk/HBP | Home run |', '|---|---|---|---|---|']
    res = r['test']['results']
    names = [n for n in res['hit'] if not n.startswith('_')]
    for name in names:
        lines.append(f"| {name} | " + ' | '.join(f"{verdict(res[o][name]['vs_production'])} {ci(res[o][name]['vs_production'], 1e4, 1)}" for o in ('hit', 'k', 'bb', 'hr')) + ' |')
    lines += ['', 'Interval values are ×10⁻⁴ log loss per plate appearance.', '',
              'Matchup additions measured against the control:', '', '| Addition | Hit | Strikeout | Walk/HBP | Home run |', '|---|---|---|---|---|']
    for name in names:
        if 'vs_control' in res['hit'][name]:
            lines.append(f"| {name} | " + ' | '.join(f"{verdict(res[o][name]['vs_control'])} {ci(res[o][name]['vs_control'], 1e4, 1)}" for o in ('hit', 'k', 'bb', 'hr')) + ' |')
    lines.append('| batter vs pitcher, only matchups with 25+ earlier PA | ' + ' | '.join(
        f"{verdict(res[o]['_deep_bvp_25pa']['vs_control'])} {ci(res[o]['_deep_bvp_25pa']['vs_control'], 1e4, 1)}" for o in ('hit', 'k', 'bb', 'hr')) + ' |')
    lines += ['', f"Chosen shrinkage per outcome, in plate appearances of prior (none = ignored): {r['k_pa']}.",
              f"Weight batter-vs-pitcher history earns after 30 PA: {r['weight_bvp_after_30_pa']}.",
              f"League home factor: {r['league_home_factor']}.", f"Platoon factor: {r['platoon_factor']}.", '',
              '**Matchup streaks** (2023-2025, the next plate appearance against the same pitcher):', '']
    for s in r['matchup_streaks']:
        lines.append(f"- {s['group']}: {s['plate_appearances']:,} PA; earlier ratio {s['prior_ratio']}; hits actual ÷ forecast {s['next_pa_hits_actual_over_forecast']}")
    j = r['aaron_judge_15pa_matchups']
    lines.append(f"- Aaron Judge against pitchers he had faced 15+ times: {j['plate_appearances']} PA, earlier hit ratio {j['prior_bvp_hit_ratio']}, "
                 f"{j['hits_actual']} hits vs {j['hits_forecast_without_bvp']} forecast without matchup history")
    return lines


def main():
    summary = OUT / 'summary.md'
    parts = [summary.read_text().rstrip(), '', '# Details', ''] if summary.exists() else ['# Matchup backtest: player vs opponent and home/away', '']
    parts += ['Research only. No production model, pick rule or published page reads these files. '
              'Every adjustment uses games before the one forecast; shrinkage is chosen on validation seasons and scored once on later test seasons. '
              'Scripts: `scripts/research/matchups/`; regenerate this page with `python scripts/research/matchups/report.py`.', '']
    for name, fn in [('nhl', nhl), ('nfl', nfl), ('mlb', mlb)]:
        path = OUT / f'{name}.json'
        if path.exists():
            parts += fn(json.loads(path.read_text())) + ['']
    parts += ['MLB data: The information used here was obtained free of charge from and is copyrighted by Retrosheet. '
              'Interested parties may contact Retrosheet at "www.retrosheet.org". NFL data: nflverse. NHL data: the frozen nhl-v2.3 history.', '']
    (OUT / 'README.md').write_text('\n'.join(parts))
    print('\n'.join(parts))


if __name__ == '__main__':
    main()

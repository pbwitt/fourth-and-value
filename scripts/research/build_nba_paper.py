"""Render research report FV-2026-05 (NBA forecasting and market pricing) from committed evidence.

Run from the repository root: python3 scripts/research/build_nba_paper.py
It formats existing results only; it does not retrain, refit or contact a provider. Every table comes
from reports/nba-paper/evidence.json (written by scripts/research/nba/study.py), and the key figures
quoted in the text are asserted against it below.
"""
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from site_notices import apply

EVIDENCE = ROOT / 'reports/nba-paper/evidence.json'
DEST = ROOT / 'docs/research'
SLUG = 'nba-forecasting-and-market-pricing'
MODEL_COMMIT = 'ded4221d3729d740c64a639e0020644d14231fa6'  # merge of PR #149 (nba-v1.0)
STAT = {'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'FG3M': 'Threes', 'BLK': 'Blocks', 'STL': 'Steals',
        'TOV': 'Turnovers'}
MARKET = {'player_points': 'Points', 'player_rebounds': 'Rebounds', 'player_assists': 'Assists', 'player_threes': 'Threes',
          'player_blocks': 'Blocks', 'player_steals': 'Steals', 'player_turnovers': 'Turnovers',
          'player_points_rebounds_assists': 'Pts + Reb + Ast', 'player_points_rebounds': 'Pts + Reb',
          'player_points_assists': 'Pts + Ast', 'player_rebounds_assists': 'Reb + Ast'}
FEATURE = {'intercept': 'Intercept (points)', 'naive_home': 'Naive home score', 'naive_away': 'Naive away score',
           'home_b2b': 'Home on back-to-back', 'away_b2b': 'Away on back-to-back', 'home_rest': 'Home days of rest',
           'away_rest': 'Away days of rest', 'neutral': 'Neutral site'}


def table(caption, headers, rows):
    return '<div class="table-wrap"><table><caption>' + html.escape(caption) + '</caption><thead><tr>' + ''.join(
        '<th scope="col">' + html.escape(x) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        '<th scope="row">' + html.escape(str(x)) + '</th>' if i == 0 else '<td>' + html.escape(str(x)) + '</td>'
        for i, x in enumerate(row)) + '</tr>' for row in rows) + '</tbody></table></div>'


def num(x, digits=4, sign=False):
    text = f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'
    return text.replace('-', '−')


def diff(b, digits=4):
    return f'{num(b["mean"], digits, True)} [{num(b["ci95"][0], digits)}, {num(b["ci95"][1], digits)}]'


def pct(x, digits=1):
    return num(100 * x, digits) + '%'


def season(s):
    return s[:4] + '–' + s[-2:]


def check(e):
    """Figures quoted in the manuscript text must match the evidence."""
    g, m, p = e['games'], e['market']['markets'], e['players']
    assert e['model_version'] == 'nba-v1.0' and e['paper'] == 'FV-2026-05'
    assert round(e['v1_evaluation']['games']['moneyline_model']['logloss'], 4) == 0.6025
    assert round(g['candidates']['home_rate']['moneyline_log_loss'], 4) == 0.6876
    assert round(g['paired']['v1_minus_pace_efficiency_joint']['mean'], 4) == 0.0009
    assert [round(x, 4) for x in g['paired']['v1_minus_pace_efficiency_joint']['ci95']] == [-0.0059, 0.0089]
    assert [round(x, 4) for x in [g['paired']['v1_minus_home_rate_moneyline']['mean'], *g['paired']['v1_minus_home_rate_moneyline']['ci95']]] == [-0.0851, -0.1066, -0.0650]
    assert (round(g['candidates']['v1']['early_total_bias'], 2), round(g['candidates']['v1_with_playoff_states']['early_total_bias'], 2)) == (2.81, 4.43)
    assert (round(g['candidates']['v1']['predicted_regulation_tie'], 4), g['observed_overtime_rate']) == (0.0248, 0.0439)
    assert e['market']['games_with_lines'] == 979
    assert round(m['moneyline']['model_minus_market']['mean'], 4) == 0.0367
    assert [round(x, 4) for x in m['moneyline']['model_minus_market']['ci95']] == [0.0232, 0.0511]
    assert [m[k]['blend_weight_on_model_first_half'] for k in ('moneyline', 'spread', 'total')] == [0.05, 0.0, 0.0]
    assert all(row['roi'] < 0 for rows in e['market']['policy'].values() for row in rows)
    assert round(p['paired']['PTS']['v1_minus_direct']['mean'], 3) == -0.079
    assert round(p['paired']['PTS']['v1_minus_poisson']['mean'], 3) == -0.155
    assert (p['distribution']['PTS']['observed_zero'], p['distribution']['PTS']['predicted_zero']) == (0.0916, 0.0265)
    assert p['comparison_appearances'] == 23979
    assert round(p['combos']['player_points_rebounds_assists']['joint_minus_independent']['mean'], 4) == -0.0033
    stats = e['matchups']['stats']
    assert all(v['k_overall'] == 'not used' for v in stats.values())
    assert all(v[s]['best_finite_minus_overall']['mean'] >= 0 for v in stats.values() for s in ('venue', 'opponent', 'back_to_back'))
    assert (stats['PTS']['b2b_actual_over_forecast'], stats['PTS']['rested_actual_over_forecast']) == (1.0009, 0.9914)


def main():
    e = json.loads(EVIDENCE.read_text())
    check(e)
    g, mk, p, v1 = e['games'], e['market'], e['players'], e['v1_evaluation']
    tags = {'MODEL_COMMIT': MODEL_COMMIT, 'MODEL_COMMIT_SHORT': MODEL_COMMIT[:8]}

    rows = []
    for s, v in e['data']['seasons'].items():
        t = v['by_type']
        rows.append([season(s), f'{t["regular"]:,}', t['cup_final'], t['play_in'], t['playoffs'], f'{v["games"]:,}',
                     f'{v["player_rows"]:,}', len(v['missing'])])
    tags['DATA'] = table('Table 1. ESPN games and player rows by season. Missing: playoff summaries ESPN could not supply.',
                         ['Season', 'Regular', 'Cup final', 'Play-in', 'Playoffs', 'Games', 'Player rows', 'Missing'], rows)

    tags['TEAM_SELECTION'] = table('Table 2. Team-state selection: 2024–25 joint-score NLL per game (nats), fit on 2023–24. Lower is better; first row selected.',
        ['Half-life (games)', 'Season carry', 'Prior games', 'Validation NLL'],
        [[c['half_life'], c['carry'], c['prior_games'], num(c['valid_nll'])] for c in e['selection']['team']])
    tags['PLAYER_SELECTION'] = table('Table 3. Player-rate selection: 2024–25 count NLL summed over seven statistics. First row selected.',
        ['Rate half-life (apps)', 'Prior minutes', 'Opponent weight ω', 'Validation NLL sum'],
        [[c['rate_half_life'], c['prior_minutes'], c['opponent_weight'], num(c['valid_logloss_sum'])] for c in e['selection']['player']])

    c = g['candidates']
    names = {'home_rate': 'Home-win rate', 'pace_efficiency_only': 'Pace × efficiency only', 'v1': 'v1 (adds rest, back-to-back)'}
    rows = []
    for key in ('home_rate', 'pace_efficiency_only', 'v1'):
        v = c[key]
        rows.append([names[key], num(v['joint_nll']) if 'joint_nll' in v else '—', num(v['moneyline_log_loss']),
                     num(v['moneyline_brier']), num(v['total_rmse'], 2) if 'total_rmse' in v else '—',
                     num(v['margin_rmse'], 2) if 'margin_rmse' in v else '—',
                     num(v['total_bias'], 2, True) if 'total_bias' in v else '—',
                     num(v['early_total_bias'], 2, True) if 'early_total_bias' in v else '—'])
    v = c['v1_with_playoff_states']
    rows.append(['v1 with playoff games in states', '—', '—', '—', num(v['total_rmse'], 2), '—', num(v['total_bias'], 2, True),
                 num(v['early_total_bias'], 2, True)])
    tags['GAME_RESULTS'] = table(f'Table 4. 2025–26 game forecasts, {g["test_games"]:,} games. Bias is actual minus expected total points; '
                                 f'early is the first three weeks ({g["early_games"]} games).',
        ['Model', 'Joint NLL', 'ML log loss', 'ML Brier', 'Total RMSE', 'Margin RMSE', 'Total bias', 'Early bias'], rows)

    co = g['coefficients_per_unit']
    tags['GAME_COEF'] = table('Table 5. v1 regulation-point regressions (fit 2023–24 and 2024–25), per unit of each feature.',
        ['Feature', 'Home points', 'Away points'],
        [[FEATURE[k], num(co['home'][k], 3), num(co['away'][k], 3)] for k in co['home']])

    tags['ML_CALIBRATION'] = table('Table 6. 2025–26 moneyline reliability: home-win forecasts by bin.',
        ['Forecast bin', 'Games', 'Mean forecast', 'Home won'],
        [[b['bin'], b['n'], num(b['mean_forecast'], 3), num(b['observed'], 3)] for b in v1['games']['moneyline_calibration']])

    cl = p['count_log_loss']
    cols = ['recency_30', 'direct_per_game', 'minutes_x_rate_poisson', 'minutes_x_rate_structural', 'v1']
    tags['PLAYER_COUNTS'] = table(f'Table 7. 2025–26 count NLL per appearance (nats), {p["comparison_appearances"]:,} appearances. Lower is better.',
        ['Statistic', 'Recency 30', 'Direct per-game', 'Minutes × rate, Poisson', 'Structural (no blend)', 'v1'],
        [[STAT[s]] + [num(cl[s][k]) for k in cols] for s in cl])

    pp = p['paired']
    tags['PLAYER_PAIRED'] = table('Table 8. Paired differences in count NLL, v1 minus each alternative, with 95% game-bootstrap intervals. Negative favors v1.',
        ['Statistic', 'vs recency 30', 'vs direct per-game', 'vs Poisson', 'Blend effect'],
        [[STAT[s], diff(v['v1_minus_recency']), diff(v['v1_minus_direct']), diff(v['v1_minus_poisson']), diff(v['blend_effect'])]
         for s, v in pp.items()])

    d = p['distribution']
    tags['DISTRIBUTION'] = table('Table 9. Distribution diagnostics, 2025–26. Coverage of the central 80% discrete interval; κ is the gamma shape.',
        ['Statistic', 'Observed zero', 'Predicted zero', '80% coverage', 'Shape κ'],
        [[STAT[s], pct(v['observed_zero'], 2), pct(v['predicted_zero'], 2), pct(v['interval80_coverage']),
          f'{p["shapes"][s]:,.1f}'] for s, v in d.items()])

    mn = p['minutes']
    rows = [['Mean absolute error (min)', num(mn['v1_mae'], 2), num(mn['recent5_mae'], 2)],
            ['RMSE (min)', num(mn['v1_rmse'], 2), num(mn['recent5_rmse'], 2)]]
    rows += [[f'Residual SD at {float(k):.0f} projected min', num(sd, 2), '—'] for k, sd in mn['residual_sd_by_projection'].items()
             if float(k) in (2, 10, 18, 26, 34, 38)]
    tags['MINUTES'] = table('Table 10. Minutes projection, 2025–26 appearances, against the player’s last-five-game average.',
                            ['Measure', 'v1 regression', 'Last 5 average'], rows)

    pl = v1['players']['pseudo_line_markets']
    tags['PSEUDO'] = table('Table 11. Over/under log loss at pseudo-lines (floor of last-10 average + 0.5), 2025–26. Slopes fitted on 2024–25.',
        ['Market', 'Appearances', 'Over rate', 'Baseline', 'v1 raw', 'v1 calibrated', 'Slope'],
        [[MARKET[k], f'{v["n"]:,}', pct(v['over_rate']), num(v['baseline']['logloss']), num(v['model_raw']['logloss']),
          num(v['model']['logloss']), num(v['calibration_slope'], 3)] for k, v in pl.items()])

    cb = p['combos']
    tags['COMBOS'] = table('Table 12. Combined markets: joint simulation (shared minutes, copula) against independent marginals, 3,000 sampled appearances.',
        ['Market', 'Joint log loss', 'Independent', 'Joint − independent [95%]', 'Observed var.', 'Joint var.', 'Independent var.'],
        [[MARKET[k], num(v['joint_log_loss']), num(v['independent_log_loss']), diff(v['joint_minus_independent']),
          num(v['observed_residual_variance'], 1), num(v['joint_predicted_variance'], 1), num(v['independent_predicted_variance'], 1)]
         for k, v in cb.items()])

    rows = []
    for k, v in mk['markets'].items():
        h = v['second_half']
        rows.append([k.capitalize(), v['n'], num(v['model_log_loss']), num(v['market_log_loss']), diff(v['model_minus_market']),
                     pct(v['mean_abs_disagreement']), num(v['blend_weight_on_model_first_half'], 2),
                     f'{num(h["market"])} / {num(h["blend"])}'])
    tags['MARKET'] = table('Table 13. v1 against DraftKings no-vig probabilities, 2025–26, pushes excluded. Blend weight fitted on the first half; second-half market / blend log loss.',
        ['Market', 'Games', 'Model', 'Market', 'Model − market [95%]', 'Mean |gap|', 'Blend weight', '2nd half'], rows)

    rows = []
    for k, items in mk['policy'].items():
        for v in items:
            rows.append([k.capitalize(), f'{v["threshold"] * 100:.1f} pp', v['bets'], pct(v['roi']),
                         f'[{pct(v["roi_ci95"][0])}, {pct(v["roi_ci95"][1])}]'])
    tags['POLICY'] = table('Table 14. Exploratory flat-stake rule at DraftKings prices, 2025–26: bet sides whose model probability beats break-even by the threshold. Pushes refunded.',
        ['Market', 'Threshold', 'Bets', 'ROI', 'ROI 95% interval'], rows)

    rows = []
    label = {'venue': 'Home/away split', 'opponent': 'Vs this opponent', 'back_to_back': 'Back-to-back split'}
    for s, v in e['matchups']['stats'].items():
        change = lambda k, b: 'no change' if k == 'not used' else diff(b, 5)
        rows.append([f'{STAT[s]}: player bias', v['k_overall'], change(v['k_overall'], v['overall_minus_baseline']), '—', '—'])
        for key in ('venue', 'opponent', 'back_to_back'):
            x = v[key]
            rows.append([f'{STAT[s]}: {label[key].lower()}', x['k'], change(x['k'], x['minus_overall']), x['best_finite_k'],
                         diff(x['best_finite_minus_overall'], 5)])
    tags['MATCHUPS'] = table('Table 15. Shrunk player histories over v1, scored on 2025–26. Change in count NLL against no split; positive is worse. “not used” is k = ∞.',
        ['Statistic: adjustment', 'Selected k', 'Test change [95%]', 'Best finite k', 'Change at that k [95%]'], rows)

    raw = EVIDENCE.read_bytes()
    evidence = dict(paper='FV-2026-05', paper_version='1.0', model_version=e['model_version'], model_commit=MODEL_COMMIT,
                    evidence_sha256=hashlib.sha256(raw).hexdigest(), evidence=e)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / (SLUG + '.json')).write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + '\n')
    manuscript = Path(__file__).with_name('nba_paper.html').read_text()
    for key, value in tags.items():
        assert '@@' + key + '@@' in manuscript, key
        manuscript = manuscript.replace('@@' + key + '@@', value)
    assert '@@' not in manuscript, 'Unfilled manuscript tag'
    (DEST / (SLUG + '.html')).write_text(apply(manuscript, 'research/' + SLUG + '.html'))
    print(f'Rendered docs/research/{SLUG}.html and .json from reports/nba-paper/evidence.json.')


if __name__ == '__main__':
    main()

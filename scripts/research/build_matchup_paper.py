"""Render research report FV-2026-03 (player matchups and home field) from committed evidence.

Run from the repository root: python3 scripts/research/build_matchup_paper.py
It formats existing results only; it does not retrain, refit or contact a provider. Every table and
the key figures in the text come from the evidence files listed in SOURCES, whose hashes are
written to the evidence JSON.
"""
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from site_notices import apply

DEST = ROOT / 'docs/research'
SLUG = 'player-matchups-and-home-field'
SOURCES = ['reports/matchups/nhl.json', 'reports/matchups/nfl.json', 'reports/matchups/mlb.json',
           'reports/matchups/nfl_venue.json', 'reports/nhl-v2.4/evaluation.json', 'reports/nhl-v2.4/paired-differences.json',
           'reports/nhl-v2.4/selection-lock.json']
NFL_LABEL = {'pass_yds': 'Passing yards', 'pass_attempts': 'Pass attempts', 'completions': 'Completions', 'rush_yds': 'Rushing yards',
             'rush_attempts': 'Rush attempts', 'recv_yds': 'Receiving yards', 'receptions': 'Receptions', 'pass_tds': 'Passing TDs',
             'pass_interceptions': 'Interceptions', 'anytime_td': 'Anytime TD'}
MLB_LABEL = {'hit': 'Hit', 'k': 'Strikeout', 'bb': 'Walk or HBP', 'hr': 'Home run'}


def read(path):
    return json.loads((ROOT / path).read_text())


def table(caption, headers, rows):
    return '<div class="table-wrap"><table><caption>' + html.escape(caption) + '</caption><thead><tr>' + ''.join(
        '<th scope="col">' + html.escape(x) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        '<th scope="row">' + html.escape(str(x)) + '</th>' if i == 0 else '<td>' + html.escape(str(x)) + '</td>'
        for i, x in enumerate(row)) + '</tr>' for row in rows) + '</tbody></table></div>'


def signed(x, digits):
    return f'{x:+.{digits}f}'.replace('-', '−')


def ci(d, digits, scale=1.):
    return f"{signed(d['mean'] * scale, digits)} [{signed(d['lo'] * scale, digits)}, {signed(d['hi'] * scale, digits)}]"


def verdict(d):
    return 'better' if d['hi'] < 0 else 'worse' if d['lo'] > 0 else 'no detectable change'


def k_text(k, unit):
    return 'not used (zero weight)' if k == 'none' else f'{k} {unit} of prior'


def pct(x):
    return signed((x - 1) * 100, 1) + '%'


def main():
    nhl, nfl, mlb, venue = (read(p) for p in SOURCES[:4])
    ev, paired, lock = (read(p) for p in SOURCES[4:])
    assert ev['version'] == 'nhl-v2.4' and lock['shots'] == lock['scoring'] == 'opportunity_nb_opp_player'
    assert not lock['recommendations_enabled'] and lock['market_weight'] == 0
    tags = {}

    # Table 1: data and windows.
    nfl_rows = {m: x['rows'] for m, x in nfl['markets'].items()}
    tags['DATA'] = table('Table 1. Data, history and evaluation windows. Tuning chooses how much weight each history gets; the test season is scored once.',
        ['Sport', 'Source', 'Unit', 'History from', 'Tuning', 'Test', 'Test rows'],
        [['NHL', 'nhl-v2.3 archive (official NHL game logs)', 'player-game', '2022–23', '2024–25', '2025–26', f"{nhl['test']['rows']:,}"],
         ['NFL', 'nflverse weekly player stats', 'player-game', '2012', '2022–2023', '2024 to 2026 wk 4', f"{sum(r['test'] for r in nfl_rows.values()):,} (7 markets)"],
         ['MLB', 'Retrosheet event files', 'plate appearance', '2012', '2021–2022', '2023–2025', f"{mlb['test']['plate_appearances']:,}"]])

    # Table 2: weight the histories earned on tuning data.
    rows = []
    for fam in ('shots', 'scoring'):
        k = nhl['k'][fam]
        rows.append([f'NHL {fam}', k_text(k['vs opponent'], 'expected events'), k_text(k['home/away split'], 'expected events')])
    for m, x in nfl['markets'].items():
        rows.append([f'NFL {NFL_LABEL[m].lower()}', k_text(x['k_games']['vs opponent'], 'games'), k_text(x['k_games']['home/away split'], 'games')])
    for o, label in MLB_LABEL.items():
        rows.append([f'MLB {label.lower()}', k_text(mlb['k_pa']['bvp'][o], 'PA') + ' (batter vs pitcher)', k_text(mlb['k_pa']['venue'][o], 'PA')])
    tags['WEIGHTS'] = table('Table 2. Prior strength chosen on tuning seasons. A history of n units gets weight n ÷ (n + prior); "not used" means the tuning seasons gave it zero weight.',
        ['Market', 'Player vs this opponent', "Player's own home/away split"], rows)

    # Table 3: test effect of opponent history and home/away splits, measured against the control.
    rows = []
    for stat in ('shots', 'goals', 'assists', 'points'):
        r = nhl['test']['results']
        rows.append([f'NHL {stat} (log loss)', ci(r['+ player vs opponent'][stat]['nll_vs_control'], 5), ci(r['+ player home/away split'][stat]['nll_vs_control'], 5)])
    for m, x in nfl['markets'].items():
        t = x['test']
        rows.append([f'NFL {NFL_LABEL[m].lower()} (squared error)', ci(t['+ player vs opponent']['sq_error_vs_control'], 2), ci(t['+ player home/away split']['sq_error_vs_control'], 2)])
    for o, label in MLB_LABEL.items():
        r = mlb['test']['results'][o]
        rows.append([f'MLB {label.lower()} (log loss ×10⁻⁴)', ci(r['+ batter vs this pitcher']['vs_control'], 1, 1e4), ci(r['+ batter home/away split']['vs_control'], 1, 1e4)])
    tags['SPLITS'] = table('Table 3. Test-season change from adding each history to the same forecast without it. Mean per row [95% interval, resampling whole games]; negative is better; 0 [0, 0] means tuning gave it zero weight.',
        ['Market (metric)', 'Player vs this opponent', "Player's own home/away split"], rows)

    # Table 4: matchup streaks.
    rows = []
    def streak(label, group, n, prior, nxt):
        rows.append([label, group, f'{n:,}', f'{prior:.3f}', f'{nxt:.3f}'])
    for fam, label in (('shots', 'NHL shots'), ('points', 'NHL points')):
        for x in nhl['matchup_streaks'][fam]:
            streak(label, x['group'], x['rows'], x['prior_ratio'], x['next_game_actual_over_forecast'])
    for m in ('recv_yds', 'rush_yds', 'pass_yds'):
        for x in nfl['markets'][m]['matchup_streaks']:
            streak(f'NFL {NFL_LABEL[m].lower()}', x['group'], x['rows'], x['prior_ratio'], x['next_game_actual_over_forecast'])
    for x in mlb['matchup_streaks']:
        streak('MLB hits', x['group'], x['plate_appearances'], x['prior_ratio'], x['next_pa_hits_actual_over_forecast'])
    tags['STREAKS'] = table('Table 4. Players who had beaten or missed the forecast against one opponent, and how they did the next time (actual ÷ forecast, test seasons). The reference row is everyone with that much history.',
        ['Market', 'Group', 'Next meetings', 'Earlier ratio', 'Next-meeting ratio'], rows)
    hot, cold, ref = (next(x for x in nhl['matchup_streaks']['shots'] if x['group'].startswith(w)) for w in ('beat', 'fell', 'every'))
    mh, mc, mr = (next(x for x in mlb['matchup_streaks'] if w in x['group']) for w in ('above', 'below', 'reference'))
    tags.update(NHL_HOT=f"{(hot['prior_ratio'] - 1) * 100:.0f}%", NHL_HOT_NEXT=f"{hot['next_game_actual_over_forecast']:.3f}",
                NHL_COLD=f"{(1 - cold['prior_ratio']) * 100:.0f}%", NHL_COLD_NEXT=f"{cold['next_game_actual_over_forecast']:.3f}",
                NHL_REF_NEXT=f"{ref['next_game_actual_over_forecast']:.3f}",
                MLB_HOT=f"{(mh['prior_ratio'] - 1) * 100:.0f}%", MLB_HOT_NEXT=f"{mh['next_pa_hits_actual_over_forecast']:.3f}",
                MLB_COLD=f"{(1 - mc['prior_ratio']) * 100:.0f}%", MLB_COLD_NEXT=f"{mc['next_pa_hits_actual_over_forecast']:.3f}",
                MLB_REF_NEXT=f"{mr['next_pa_hits_actual_over_forecast']:.3f}",
                MLB_GAP_BEFORE=f"{(mh['prior_ratio'] - mc['prior_ratio']) * 100:.0f}",
                MLB_GAP_AFTER=f"{(mh['next_pa_hits_actual_over_forecast'] - mc['next_pa_hits_actual_over_forecast']) * 100:.1f}")
    j = mlb['aaron_judge_15pa_matchups']
    tags.update(JUDGE_PA=str(j['plate_appearances']), JUDGE_HITS=str(j['hits_actual']), JUDGE_FORECAST=f"{j['hits_forecast_without_bvp']:.1f}")

    # Table 5: NFL venue factors.
    rows = []
    for m, x in venue['markets'].items():
        fixed, final, interval = x['production_fixed'], x['fitted_2012_2025'], x['interval_2012_2025']
        rows.append([NFL_LABEL[m], f"{pct(fixed['home'])} / {pct(fixed['away'])}", f"{pct(final['home'])} / {pct(final['away'])}",
                     f"{pct(interval['home'][0])} to {pct(interval['home'][1])}", ci(x['test_fitted_minus_fixed'], 2), verdict(x['test_fitted_minus_fixed'])])
    tags['VENUE'] = table('Table 5. NFL home / away factors. Earlier fixed values, values refitted on 2012–2025, the 95% interval for the home factor, and the 2024–26 holdout change in squared error when gaps fitted on 2012–2021 replace the fixed values.',
        ['Market', 'Previous fixed', 'Fitted 2012–2025', 'Home 95% interval', 'Holdout change (fitted − fixed)', 'Holdout'], rows)
    pv = venue['markets']['pass_yds']
    tags.update(PASS_HOME=pct(pv['fitted_2012_2025']['home']), PASS_HOLDOUT=ci(pv['test_fitted_minus_fixed'], 1),
                RECV_HOLDOUT=ci(venue['markets']['recv_yds']['test_fitted_minus_fixed'], 1))

    # Table 6: platoon and league home factors in MLB.
    rows = []
    for o, label in MLB_LABEL.items():
        pf, hf, r = mlb['platoon_factor'][o], mlb['league_home_factor'][o], mlb['test']['results'][o]
        rows.append([label, f"{pf['same hand'] / pf['opposite hand']:.3f}", f"{hf['home'] / hf['away']:.3f}",
                     ci(r['+ platoon (bat side vs pitcher hand)']['vs_production'], 1, 1e4),
                     ci(r["+ batter's own platoon split"]['vs_control'], 1, 1e4)])
    tags['PLATOON'] = table('Table 6. MLB plate appearances. Rate ratios from 2012–2019, and test-season (2023–25) change in log loss ×10⁻⁴ per plate appearance. Negative is better.',
        ['Outcome', 'Same-hand ÷ opposite-hand', 'Home ÷ away', 'Adding league platoon', "Adding batter's own platoon split"], rows)

    # Table 7: NHL v2.4 locked evaluation.
    final = ev['final']['player']
    rows = []
    for stat in ('shots', 'goals', 'assists', 'points'):
        d = paired[f'{stat} vs opportunity_nb_opp']
        a, b = final['opportunity_nb_opp'][stat], final['opportunity_nb_opp_player'][stat]
        rows.append([stat, f"{a['count_log_loss']:.5f}", f"{b['count_log_loss']:.5f}",
                     f"{signed(d['selected_minus_baseline'], 5)} [{signed(d['game_cluster_bootstrap95'][0], 5)}, {signed(d['game_cluster_bootstrap95'][1], 5)}]",
                     f"{a['ece']:.4f} → {b['ece']:.4f}"])
    tags['NHL24'] = table('Table 7. NHL v2.4 under the locked protocol: 2025–26 final test, paired against v2.3 on the same 47,230 player-games. Count log loss; 95% interval from resampling whole games; calibration error (ECE) at the reference line.',
        ['Market', 'v2.3', 'v2.4', 'Difference [95% interval]', 'ECE'], rows)
    shots = paired['shots vs opportunity_nb_opp']
    tags['NHL24_SHOTS'] = f"{signed(shots['selected_minus_baseline'], 5)} [{signed(shots['game_cluster_bootstrap95'][0], 5)}, {signed(shots['game_cluster_bootstrap95'][1], 5)}]"

    source = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    evidence = dict(paper='FV-2026-03', paper_version='1.0', hashes=source,
                    matchups=dict(nhl=nhl, nfl=nfl, mlb=mlb), nfl_venue=venue,
                    nhl_v24=dict(selection=lock, final_player=final, paired=paired))
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / (SLUG + '.json')).write_text(json.dumps(evidence, indent=2) + '\n')
    manuscript = Path(__file__).with_name('matchup_paper.html').read_text()
    for key, value in tags.items():
        assert '@@' + key + '@@' in manuscript, key
        manuscript = manuscript.replace('@@' + key + '@@', value)
    assert '@@' not in manuscript, 'Unfilled manuscript tag'
    (DEST / (SLUG + '.html')).write_text(apply(manuscript, 'research/' + SLUG + '.html'))
    print(f'Rendered FV-2026-03 and its evidence JSON from {len(SOURCES)} evidence files.')


if __name__ == '__main__':
    main()

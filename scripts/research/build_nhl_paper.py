"""Render the fixed NHL technical paper from versioned evaluation evidence (stdlib only).

Run from the repository root: python3 scripts/research/build_nhl_paper.py
This formats existing results; it does not retrain, select models, or contact a provider.
"""
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from site_notices import apply

REPORT = ROOT / 'reports/nhl-rebuild'
DEST = ROOT / 'docs/research'
SLUG = 'nhl-forecasting-and-market-pricing'
SOURCE_COMMIT = 'e1f91609524f74a6b99cf86cdebb2328cd5e08d1'


def read(name):
    return json.loads((REPORT / name).read_text())


def table(caption, headers, rows):
    return '<div class="table-wrap"><table><caption>' + html.escape(caption) + '</caption><thead><tr>' + ''.join(
        '<th scope="col">' + html.escape(x) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        '<th scope="row">' + html.escape(str(x)) + '</th>' if i == 0 else '<td>' + html.escape(str(x)) + '</td>'
        for i, x in enumerate(row)) + '</tr>' for row in rows) + '</tbody></table></div>'


def interval(values, digits=5, percent=False):
    return '[' + ', '.join(f'{x * (100 if percent else 1):.{digits}f}' + ('%' if percent else '') for x in values) + ']'


def main():
    e, d, m = read('evaluation.json'), read('paired-differences.json'), read('historical-market-evaluation.json')
    assert e['selection']['team'] == 'poisson_core'
    assert e['selection']['shots'] == e['selection']['scoring'] == 'opportunity_nb'
    assert e['selection']['market_weight'] == 0 and not e['selection']['recommendations_enabled']
    tags = {'SOURCE_COMMIT': SOURCE_COMMIT}
    rows = []
    for fold in e['validation']:
        rows.append([str(fold['season'])[:4] + '–' + str(fold['season'])[-2:],
                     fold['training']['start'] + ' to ' + fold['training']['end'],
                     fold['validation']['start'] + ' to ' + fold['validation']['end'], '1,312'])
    rows.append(['2025–26 final', e['final']['training']['start'] + ' to ' + e['final']['training']['end'],
                 e['final']['test']['start'] + ' to ' + e['final']['test']['end'], '1,312'])
    tags['WINDOWS'] = table('Table 1. Expanding training windows and season-level evaluation windows.',
                           ['Evaluation', 'Parameter fitting', 'Evaluation dates', 'Games'], rows)
    tags['TEAM_SELECTION'] = table('Table 2. Development joint-score negative log likelihood, in nats per game. Lower is better.',
        ['Candidate', '2023–24', '2024–25', 'Mean'], [[name] + [f'{f["team"][name]["joint_log_loss"]:.5f}' for f in e['validation']] +
        [f'{sum(f["team"][name]["joint_log_loss"] for f in e["validation"])/2:.5f}'] for name in e['validation'][0]['team']])
    rows = []
    for name in e['validation'][0]['player']:
        vals = [f['player'][name] for f in e['validation']]
        rows.append([name] + [f'{v["shots"]["count_log_loss"]:.5f}' for v in vals] +
                    [f'{sum(v[s]["count_log_loss"] for s in ["goals","assists","points"])/3:.5f}' for v in vals])
    tags['PLAYER_SELECTION'] = table('Table 3. Development count NLL. Scoring averages goals, assists and points with equal weights.',
        ['Candidate', 'Shots 23–24', 'Shots 24–25', 'Scoring 23–24', 'Scoring 24–25'], rows)
    rows = []
    for name, v in e['final']['team'].items():
        rows.append([name, f'{v["joint_log_loss"]:.5f}', f'{v["total_mae"]:.4f}', f'{v["total_rmse"]:.4f}',
                     f'{v["markets"]["moneyline"]["log_loss"]:.5f}', f'{v["markets"]["moneyline"]["brier"]:.5f}'])
    tags['TEAM_FINAL'] = table('Table 4. Final-season team results, 1,312 games.',
        ['Model', 'Joint NLL', 'Total MAE', 'Total RMSE', 'ML log loss', 'ML Brier'], rows)
    tags['TEAM_CALIBRATION'] = table('Table 5. Selected-model binary diagnostics. These fixed lines are not historical quoted offers.',
        ['Outcome', 'Log loss', 'Brier', 'ECE'], [[name] + [f'{v[k]:.5f}' for k in ['log_loss','brier','ece']]
        for name,v in e['final']['team']['poisson_core']['markets'].items()])
    rows = []
    for stat, v in e['final']['player']['opportunity_nb'].items():
        rows.append([stat, f'{e["final"]["player"]["rate_poisson"][stat]["count_log_loss"]:.5f}',
                     f'{v["count_log_loss"]:.5f}', f'{v["mae"]:.4f}', f'{v["rmse"]:.4f}', f'{v["brier"]:.5f}', f'{v["ece"]:.5f}'])
    tags['PLAYER_FINAL'] = table('Table 6. Final-season player results, 47,230 appearances per market. Binary thresholds: shots >2.5; all scoring markets >0.5.',
        ['Market', 'Rate NLL', 'Selected NLL', 'MAE', 'RMSE', 'Brier', 'ECE'], rows)
    tags['PAIRED'] = table('Table 7. Paired selected-minus-baseline NLL. Negative favors the selected model; 500 whole-game bootstrap replicates.',
        ['Target', 'Difference', '95% percentile interval'], [[name, f'{d[name]["selected_minus_baseline"]:.6f}',
        interval(d[name]['game_cluster_bootstrap95'],6)] for name in ['joint_score','shots','goals','assists','points']])
    tags['DISTRIBUTION'] = table('Table 8. Selected player-distribution diagnostics. Zero rates and central discrete interval coverage are percentages.',
        ['Market', 'Observed zero', 'Predicted zero', '90% coverage', 'RPS'], [[s, f'{v["observed_zero"]*100:.2f}',
        f'{v["predicted_zero"]*100:.2f}', f'{v["interval90_coverage"]*100:.2f}', f'{v["rps"]:.5f}']
        for s,v in e['final']['player']['opportunity_nb'].items()])
    rows = []
    for season, markets in m['metrics'].items():
        for name, v in markets.items():
            rows.append([season[:4]+'–'+season[-2:]+' / '+name, f'{v["independent"]["n"]} / {v["unique_games"]}',
                         f'{v["independent"]["log_loss"]:.5f}', f'{v["market"]["log_loss"]:.5f}', interval(v['difference_ci'])])
    tags['MARKET'] = table('Table 9. Timestamped price diagnostic: conditional non-push log loss. Intervals are hockey minus market; every interval contains zero.',
        ['Season / market', 'Obs. / games', 'Hockey', 'Market', 'Difference 95% interval'],rows)
    rows = []
    for season, v in m['shadow'].items():
        n=v['nominal']
        rows.append([season[:4]+'–'+season[-2:],str(n['count']),f'{n["net_units"]:.4f}',f'{n["roi"]*100:.2f}%',
                     f'{n["max_drawdown"]:.4f}',interval(n['roi_ci'],2,True),f'{v["worse_execution"]["roi"]*100:.2f}%'])
    tags['SHADOW'] = table('Table 10. Fixed quoted-price shadow policy. One flat unit per selection; turnover equals count. These are not realized bets.',
        ['Season', 'Count', 'Net units', 'ROI', 'Drawdown', 'ROI 95% interval', 'Worse price ROI'],rows)
    tags['ODDS'] = table('Table 11. Decimal odds distribution among shadow selections; five quantiles [minimum, 25%, median, 75%, maximum].',
        ['Season', 'Min', '25%', 'Median', '75%', 'Max'], [[s[:4]+'–'+s[-2:]]+[f'{x:.4f}' for x in v['nominal']['decimal_odds_quantiles']] for s,v in m['shadow'].items()])
    source_files = ['evaluation.json','paired-differences.json','historical-market-evaluation.json','selection-lock.json']
    evidence = dict(paper='FV-2026-02',model_version=e['version'],source_commit=SOURCE_COMMIT,
        hashes={name:hashlib.sha256((REPORT/name).read_bytes()).hexdigest() for name in source_files},
        evaluation=e,paired_differences=d,historical_market_evaluation=m)
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/(SLUG+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')
    manuscript=Path(__file__).with_name('nhl_paper.html').read_text()
    for key,value in tags.items():
        assert '@@'+key+'@@' in manuscript, key
        manuscript=manuscript.replace('@@'+key+'@@',value)
    assert '@@' not in manuscript, 'Unfilled manuscript tag'
    (DEST/(SLUG+'.html')).write_text(apply(manuscript,'research/'+SLUG+'.html'))
    print('Rendered paper and evidence JSON from four archived evaluation artifacts.')


if __name__ == '__main__':
    main()

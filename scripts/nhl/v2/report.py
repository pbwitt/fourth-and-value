"""Render the checked-in evaluation and paired game-cluster uncertainty report."""
import gzip
import json
from pathlib import Path
import sys
from collections import defaultdict

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2])); __package__='nhl.v2'
from .data import ROOT,write_json
from .evaluate import interval_mean


def render(root=ROOT/'reports/nhl-rebuild'):
    report=json.loads((root/'evaluation.json').read_text()); selection=report['selection']
    differences={}
    for kind,baseline in [('team','rate'),('player','rate_poisson')]:
        grouped=defaultdict(dict)
        with gzip.open(root/f'{kind}-predictions.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line)
                k=(r['game_id'],r.get('player_id'),r.get('market','joint_score'))
                grouped[k][r['model']]=r['score_nll' if kind=='team' else 'count_log_loss']
        for market in sorted({k[2] for k in grouped}):
            chosen=selection['team' if kind=='team' else 'shots' if market=='shots' else 'scoring']
            pairs=[(k,v) for k,v in grouped.items() if k[2]==market and chosen in v and baseline in v]
            vals=[v[chosen]-v[baseline] for k,v in pairs]; ids=[k[0] for k,v in pairs]
            differences[market]=dict(selected_minus_baseline=sum(vals)/len(vals),
                game_cluster_bootstrap95=interval_mean(vals,ids),records=len(vals),games=len(set(ids)))
    write_json(root/'paired-differences.json',differences)
    lines=['# NHL evaluation: actual results and limitations','',
        'All seven markets are **experimental forecasts**. No market is approved as a validated betting recommendation. '
        'Exact-line price comparison is usable with matched rules and fresh paired quotes; it is not evidence of prediction skill.','',
        'The implementation selected regularized Poisson team rates and negative-binomial player opportunity models '
        'using only the two development folds. One joint game distribution supplies moneyline, puck-line and totals probabilities. '
        'One shared scoring distribution preserves goals + assists = points. Market blending is disabled.','',
        '## Data and time windows','',
        '5,248 completed regular-season games and 188,883 skater appearances across four seasons. '
        'Raw source responses, manifests, normalized history, model parameters and graded final-test forecasts are retained. '
        'Postseason games are excluded. Forecasts are conditional on player participation.','',
        '| Fold | Training dates | Evaluation dates | Games |','|---|---|---|---:|']
    for f in report['validation']:
        lines.append(f"| Validation {f['season']} | {f['training']['start']} – {f['training']['end']} | {f['validation']['start']} – {f['validation']['end']} | 1,312 |")
    f=report['final'];lines.append(f"| Final test | {f['training']['start']} – {f['training']['end']} | {f['test']['start']} – {f['test']['end']} | 1,312 |")
    lines += ['', 'Calibration: no separately fitted probability remapping (identity). Distribution dispersion and OT tendency '
        'are estimated exclusively from each training window. Candidate families and fixed regularization settings were '
        'chosen before final testing; validation mean log score chooses the model. Daily histories update only after the '
        'availability cutoff, while fitted parameters stay frozen within each held-out season. Live artifacts are then '
        'refit on all completed seasons using the already selected specification.','',
        'Morning cutoff: 10:30 America/New_York. Reconstructed box-score availability: next day 12:00 UTC. '
        'Original publication/revision timestamps are absent, so this is **reconstructed predictive evaluation**, '
        'not a certified vintage-data replay. The 16:30 update uses the same result cutoff plus fresher available quotes; '
        'morning-versus-later execution cannot be compared without historical quote snapshots.','',
        '## Team candidate comparison (validation joint-score log loss; lower is better)','',
        '| Candidate | 2023–24 | 2024–25 | Mean |','|---|---:|---:|---:|']
    for n in report['validation'][0]['team']:
        vals=[f['team'][n]['joint_log_loss'] for f in report['validation']]
        lines.append(f'| {n} | {vals[0]:.5f} | {vals[1]:.5f} | {sum(vals)/2:.5f} |')
    lines += ['', 'Opponent strength and home advantage add useful information in validation. Regularization is slightly '
        'better than the multiplicative opponent baseline. Adding the bundled shot-volume, team save-rate proxy, '
        'special-teams and rest inputs improves one fold and worsens the next; those extra inputs were rejected. '
        'Boosting also failed to improve consistently. These are grouped ablations, not evidence that every individual '
        'context feature is useless. Starting-goalie identity and xG were not evaluated.','',
        '## Final team results','',
        '| Model | Joint log loss | Total MAE | Total RMSE | ML log loss | ML Brier |','|---|---:|---:|---:|---:|---:|']
    for n,r in report['final']['team'].items():
        lines.append(f"| {n} | {r['joint_log_loss']:.5f} | {r['total_mae']:.4f} | {r['total_rmse']:.4f} | {r['markets']['moneyline']['log_loss']:.5f} | {r['markets']['moneyline']['brier']:.5f} |")
    lines += ['', '| Selected-model market | Fixed diagnostic line | Log loss | Brier | ECE |','|---|---|---:|---:|---:|']
    for n,r in report['final']['team'][selection['team']]['markets'].items():
        lines.append(f"| {n} | {n.split('_')[-1] if '_' in n else 'Winner'} | {r['log_loss']:.5f} | {r['brier']:.5f} | {r['ece']:.5f} |")
    lines += ['', 'The final moneyline log loss is slightly worse than the rate baseline. We retain the preselected '
        'coherent score model rather than select a moneyline-specific winner after examining the test. Joint-score '
        'improvement is small; the paired uncertainty interval below includes zero.','',
        '## Player results','',
        '| Market | Baseline count log loss | Selected count log loss | MAE | RMSE | Binary Brier | ECE |','|---|---:|---:|---:|---:|---:|---:|']
    for stat in ['shots','goals','assists','points']:
        r=report['final']['player'][selection['shots' if stat=='shots' else 'scoring']][stat]
        b=report['final']['player']['rate_poisson'][stat]
        lines.append(f"| {stat} | {b['count_log_loss']:.5f} | {r['count_log_loss']:.5f} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['brier']:.5f} | {r['ece']:.5f} |")
    lines += ['', 'Binary thresholds are shots >2.5 and goals/assists/points >0.5. These are diagnostic thresholds, '
        'not archived sportsbook offers. All 47,230 final-season skater appearances are evaluated, not just stars '
        'or players with posted props. This universe mismatch limits transfer to the betting board.','',
        'Opportunity/rate separation improved all player count scores on both validation seasons. Extra dispersion '
        'helped shots; the scoring improvement over opportunity Poisson was very small. The hurdle challenger '
        'did not improve scoring forecasts. Count models are not assumed equally calibrated simply because one '
        'joint family is used. Full metrics include observed/predicted zeros, ranked probability scores, discrete '
        '90% interval coverage and reliability-bin counts. Nominal 90% discrete intervals cover about 97–99%; '
        'they are conservative and should not be presented as sharp uncertainty intervals.','',
        '## Paired uncertainty (selected minus baseline count/joint log loss)','',
        '500 bootstrap replicates resample entire games, retaining related player records. Negative favors the selected model.','',
        '| Market | Difference | Game-cluster 95% interval |','|---|---:|---|']
    for n,r in differences.items():
        lo,hi=r['game_cluster_bootstrap95'];lines.append(f"| {n} | {r['selected_minus_baseline']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '## Betting, markets and qualitative evaluation','',
        'The existing odds plan was subsequently verified to include historical access. A fixed monthly game-price '
        'sample now supports a separate [timestamped market comparison and shadow price simulation](HISTORICAL_MARKETS.md). '
        'It covers 130 games across 21 morning dates, costs 630 existing credits, and leaves model and policy choices '
        'unchanged. The independent-versus-market paired intervals all include zero. Three final-season shadow '
        'selections lose 1.5238 units; this is too little evidence for profitability conclusions. It is not realized '
        'execution or a full daily backtest. No validated production betting strategy is enabled.','',
        'Historical player prices, full daily game-price coverage, closing/later snapshots and qualitative records '
        'remain absent. Sparse development price coverage does not qualify a learned blend. Early-versus-later '
        'and CLV diagnostics remain unavailable. The old ledger contains only five NHL bets with inconsistent '
        'labels; it is not reused. Analyst adjustments begin as sourced, append-only prospective shadow records, '
        'preserving original forecasts. Unknown participation remains unresolved, not a zero or automatic loss.','',
        '## Reproduction and audit notes','',
        'See [RUNBOOK.md](RUNBOOK.md), [MODEL_CARD.md](MODEL_CARD.md), [SOURCES.md](SOURCES.md), '
        '[CONTRACT.md](CONTRACT.md) and [PLAN.md](PLAN.md). `evaluation.json` contains exact fold metrics and '
        'source hashes. Compressed prediction files include actual outcomes and scores; `paired-differences.json` '
        'contains paired uncertainty. No legacy performance claim enters this report.','',
        'During implementation review, the first evaluation was found to use target-game position for cold-start '
        'player priors. This was corrected to last-observed position or a fixed unknown-position prior, and the '
        'same locked protocol was rerun. Algorithm choices and thresholds were unchanged. Thus the final period '
        'was opened before this correctness repair; it was never used for tuning, and the repair is disclosed '
        'rather than describing the repeated computation as a new untouched test.']
    (root/'EVALUATION.md').write_text('\n'.join(lines)+'\n')
    return differences


if __name__=='__main__':print(json.dumps(render(),indent=2))

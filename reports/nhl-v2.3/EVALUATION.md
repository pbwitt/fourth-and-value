# NHL evaluation: actual results and limitations

All seven markets are **experimental forecasts**. No market is approved as a validated betting recommendation. Exact-line price comparison is usable with matched rules and fresh paired quotes; it is not evidence of prediction skill.

The implementation selected regularized Poisson team rates and negative-binomial player opportunity models using only the two development folds. One joint game distribution supplies moneyline, puck-line and totals probabilities. One shared scoring distribution preserves goals + assists = points. Market blending is disabled.

## Data and time windows

5,248 completed regular-season games and 188,883 skater appearances across four seasons. Raw source responses, manifests, normalized history, model parameters and graded final-test forecasts are retained. Postseason games are excluded. Forecasts are conditional on player participation.

| Fold | Training dates | Evaluation dates | Games |
|---|---|---|---:|
| Validation 20232024 | 2022-10-07 – 2023-04-14 | 2023-10-10 – 2024-04-18 | 1,312 |
| Validation 20242025 | 2022-10-07 – 2024-04-18 | 2024-10-04 – 2025-04-17 | 1,312 |
| Final test | 2022-10-07 – 2025-04-17 | 2025-10-07 – 2026-04-16 | 1,312 |

Calibration: no separately fitted probability remapping (identity). Distribution dispersion and OT tendency are estimated exclusively from each training window. Candidate families and fixed regularization settings were chosen before final testing; validation mean log score chooses the model. Daily histories update only after the availability cutoff, while fitted parameters stay frozen within each held-out season. Live artifacts are then refit on all completed seasons using the already selected specification.

Morning cutoff: 10:30 America/New_York. Reconstructed box-score availability: next day 12:00 UTC. Original publication/revision timestamps are absent, so this is **reconstructed predictive evaluation**, not a certified vintage-data replay. The 16:30 update uses the same result cutoff plus fresher available quotes; morning-versus-later execution cannot be compared without historical quote snapshots.

## Team candidate comparison (validation joint-score log loss; lower is better)

| Candidate | 2023–24 | 2024–25 | Mean |
|---|---:|---:|---:|
| rate | 3.74192 | 3.74018 | 3.74105 |
| opponent | 3.72330 | 3.72427 | 3.72378 |
| poisson_core | 3.72314 | 3.71863 | 3.72089 |
| poisson_context | 3.71437 | 3.73204 | 3.72320 |
| boosting | 3.72239 | 3.74500 | 3.73370 |

Opponent strength and home advantage add useful information in validation. Regularization is slightly better than the multiplicative opponent baseline. Adding the bundled shot-volume, team save-rate proxy, special-teams and rest inputs improves one fold and worsens the next; those extra inputs were rejected. Boosting also failed to improve consistently. These are grouped ablations, not evidence that every individual context feature is useless. Starting-goalie identity and xG were not evaluated.

## Final team results

| Model | Joint log loss | Total MAE | Total RMSE | ML log loss | ML Brier |
|---|---:|---:|---:|---:|---:|
| rate | 3.68481 | 1.8547 | 2.3151 | 0.68599 | 0.24646 |
| poisson_core | 3.68073 | 1.8433 | 2.3060 | 0.68715 | 0.24704 |

| Selected-model market | Fixed diagnostic line | Log loss | Brier | ECE |
|---|---|---:|---:|---:|
| moneyline | Winner | 0.68715 | 0.24704 | 0.01844 |
| total_5.5 | 5.5 | 0.68482 | 0.24587 | 0.03684 |
| total_6.5 | 6.5 | 0.69356 | 0.25015 | 0.03976 |
| puck_-1.5 | -1.5 | 0.60004 | 0.20553 | 0.01485 |

The final moneyline log loss is slightly worse than the rate baseline. We retain the preselected coherent score model rather than select a moneyline-specific winner after examining the test. Joint-score improvement is small; the paired uncertainty interval below includes zero.

## Player results

| Market | Baseline count log loss | Selected count log loss | MAE | RMSE | Binary Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| shots | 1.55291 | 1.54065 | 1.0215 | 1.3227 | 0.14653 | 0.01673 |
| goals | 0.45349 | 0.45255 | 0.2732 | 0.4150 | 0.12035 | 0.00565 |
| assists | 0.64574 | 0.64402 | 0.4070 | 0.5383 | 0.17057 | 0.01262 |
| points | 0.84129 | 0.83855 | 0.5347 | 0.6786 | 0.20412 | 0.02006 |

Binary thresholds are shots >2.5 and goals/assists/points >0.5. These are diagnostic thresholds, not archived sportsbook offers. All 47,230 final-season skater appearances are evaluated, not just stars or players with posted props. This universe mismatch limits transfer to the betting board.

Opportunity/rate separation improved all player count scores on both validation seasons. Extra dispersion helped shots; the scoring improvement over opportunity Poisson was very small. The hurdle challenger did not improve scoring forecasts. Count models are not assumed equally calibrated simply because one joint family is used. Full metrics include observed/predicted zeros, ranked probability scores, discrete 90% interval coverage and reliability-bin counts. Nominal 90% discrete intervals cover about 97–99%; they are conservative and should not be presented as sharp uncertainty intervals.

## Paired uncertainty (selected minus baseline count/joint log loss)

500 bootstrap replicates resample entire games, retaining related player records. Negative favors the selected model.

| Market | Difference | Game-cluster 95% interval |
|---|---:|---|
| joint_score | -0.004079 | [-0.013763, 0.005311] |
| assists | -0.001719 | [-0.002212, -0.001218] |
| goals | -0.000935 | [-0.001272, -0.000616] |
| points | -0.002741 | [-0.003478, -0.002019] |
| shots | -0.012258 | [-0.013507, -0.010989] |
| assists vs opportunity_nb | -0.000328 | [-0.000747, 0.000072] |
| goals vs opportunity_nb | -0.000143 | [-0.000389, 0.000090] |
| points vs opportunity_nb | -0.000481 | [-0.001152, 0.000156] |
| shots vs opportunity_nb | -0.003881 | [-0.004694, -0.003047] |

## Betting, markets and qualitative evaluation

The existing odds plan was subsequently verified to include historical access. A fixed monthly game-price sample now supports a separate [timestamped market comparison and shadow price simulation](HISTORICAL_MARKETS.md). It covers 130 games across 21 morning dates, costs 630 existing credits, and leaves model and policy choices unchanged. The independent-versus-market paired intervals all include zero. Three final-season shadow selections lose 1.5238 units; this is too little evidence for profitability conclusions. It is not realized execution or a full daily backtest. No validated production betting strategy is enabled.

Historical player prices, full daily game-price coverage, closing/later snapshots and qualitative records remain absent. Sparse development price coverage does not qualify a learned blend. Early-versus-later and CLV diagnostics remain unavailable. The old ledger contains only five NHL bets with inconsistent labels; it is not reused. Analyst adjustments begin as sourced, append-only prospective shadow records, preserving original forecasts. Unknown participation remains unresolved, not a zero or automatic loss.

## Reproduction and audit notes

See [RUNBOOK.md](RUNBOOK.md), [MODEL_CARD.md](MODEL_CARD.md), [SOURCES.md](SOURCES.md), [CONTRACT.md](CONTRACT.md) and [PLAN.md](PLAN.md). `evaluation.json` contains exact fold metrics and source hashes. Compressed prediction files include actual outcomes and scores; `paired-differences.json` contains paired uncertainty. No legacy performance claim enters this report.

During implementation review, the first evaluation was found to use target-game position for cold-start player priors. This was corrected to last-observed position or a fixed unknown-position prior, and the same locked protocol was rerun. Algorithm choices and thresholds were unchanged. Thus the final period was opened before this correctness repair; it was never used for tuning, and the repair is disclosed rather than describing the repeated computation as a new untouched test.

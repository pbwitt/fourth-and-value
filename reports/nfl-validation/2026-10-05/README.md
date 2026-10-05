# NFL prop validation · 2026-10-05

Model `nfl-props-2026-10-04` (code `8afa7e8378af`). Reproduce: `python scripts/nfl_validation.py fetch && python scripts/nfl_validation.py run --run-id 2026-10-05`.

Brier score: 0 is perfect; a constant 50% forecast scores 0.25 on binary outcomes. Lower is better.

## Fixed-grid 2025 holdout (calibration fitted on 2024)

| Market | Games | Forecasts | Rows | Raw | Refit | Legacy | Empirical reference | Refit − raw (95% CI) |
|---|---|---|---|---|---|---|---|---|
| all | 240 | 8015 | 89944 | 0.1864 | 0.1861 | 0.2283 | 0.2099 | -0.0003 [-0.0007, +0.0001] |
| pass_attempts | 233 | 406 | 5674 | 0.2012 | 0.2014 | 0.2085 | 0.2091 | +0.0002 [-0.0009, +0.0011] |
| pass_completions | 233 | 406 | 3986 | 0.2082 | 0.2080 | 0.2397 | 0.2153 | -0.0002 [-0.0034, +0.0028] |
| pass_tds | 233 | 405 | 2372 | 0.2061 | 0.2057 | 0.2393 | 0.1966 | -0.0004 [-0.0034, +0.0023] |
| pass_yds | 233 | 406 | 5614 | 0.2126 | 0.2124 | 0.2423 | 0.2199 | -0.0002 [-0.0025, +0.0020] |
| receptions | 240 | 2318 | 23342 | 0.1816 | 0.1807 | 0.2195 | 0.2039 | -0.0009 [-0.0015, -0.0001] |
| recv_yds | 240 | 2318 | 29034 | 0.1847 | 0.1844 | 0.2447 | 0.2076 | -0.0002 [-0.0006, +0.0002] |
| rush_attempts | 239 | 878 | 9284 | 0.1698 | 0.1697 | 0.2155 | 0.2181 | -0.0001 [-0.0005, +0.0002] |
| rush_yds | 239 | 878 | 10638 | 0.1822 | 0.1823 | 0.2106 | 0.2179 | +0.0001 [-0.0005, +0.0008] |

## 2026 weeks 2-4 at exact archived book lines (deployment artifact; seen weeks)

| Market | Games | Forecasts | Rows | Raw | Refit | Legacy | Market (de-vigged) | Refit − market (95% CI) |
|---|---|---|---|---|---|---|---|---|
| all | 46 | 1547 | 7192 | 0.2775 | 0.2724 | 0.2513 | 0.2474 | +0.0250 [+0.0142, +0.0362] |
| pass_attempts | 42 | 82 | 238 | 0.2655 | 0.2627 | 0.2543 | 0.2510 | +0.0117 [-0.0212, +0.0475] |
| pass_completions | 42 | 82 | 216 | 0.2999 | 0.2881 | 0.2525 | 0.2469 | +0.0412 [+0.0062, +0.0778] |
| pass_tds | 46 | 90 | 192 | 0.2853 | 0.2776 | 0.2450 | 0.2401 | +0.0375 [-0.0123, +0.0901] |
| pass_yds | 46 | 90 | 1006 | 0.2645 | 0.2604 | 0.2519 | 0.2444 | +0.0160 [-0.0100, +0.0412] |
| receptions | 45 | 414 | 958 | 0.2923 | 0.2814 | 0.2530 | 0.2487 | +0.0328 [+0.0132, +0.0539] |
| recv_yds | 45 | 421 | 2764 | 0.2745 | 0.2688 | 0.2497 | 0.2473 | +0.0215 [+0.0084, +0.0353] |
| rush_attempts | 44 | 143 | 342 | 0.3044 | 0.3048 | 0.2563 | 0.2476 | +0.0572 [+0.0292, +0.0910] |
| rush_yds | 46 | 225 | 1476 | 0.2738 | 0.2725 | 0.2517 | 0.2490 | +0.0234 [+0.0081, +0.0403] |

Book-line calibration diagnostic (fit on weeks [2, 3], scored on week 4, 15 games): Brier 0.2527 vs market 0.2512 and constant 50% 0.2500; model − market +0.0015 [-0.0016, +0.0047]. Not installed.

## Decision

- Installed: False. legacy models/nfl_prop_calibration.json retained for display; labelled incompatible.
- Reason: At exact 2026 book lines the legacy artifact is no better than a constant 50% and worse than the market; the grid refit is worse still. No artifact supports NFL Top Picks.
- Effect: NFL props become research-only (model_status no longer starts with "Calibration fitted") from the next NFL refresh.
- Requalification: Book-line calibration fitted on earlier timestamped weeks must, on at least 60 later games, score below a constant 50% and have a 95% interval for (model - market) Brier whose upper bound is below +0.002.

## Limitations

- No untouched historical holdout: 2025 informed hyperparameters and 2026 weeks 1-4 informed the October 4 model change.
- Fixed line grids are not bookmaker lines; grid accuracy is not demonstrated performance against prices.
- Historical injury reports and manual overrides are not archived for 2024-2025 and are omitted; 2026 weeks with archived injury snapshots use them.
- Candidate universe approximates which players receive props using earlier usage only.
- Rows from the same game are correlated; intervals resample whole games.
- Integer book lines (pushes) are excluded from binary scoring.
- No returns at offered prices are claimed from this report.

Cutoff verification: {"season": 2025, "week": 8, "career_rows_after_cutoff_on_disk": 12022, "identical_with_future_rows_supplied": true, "players": 30}

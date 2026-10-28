# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 10 refreshes, model nhl-v2.1, nhl-v2.3, nhl-v2.4. Generated 2026-10-28T11:09:53.090434Z.

9420 contracts priced; results: lost 7009, unresolved_participation 142, won 2269.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 1558 | -195.14 | -12.5% | -25.7% to -0.7% |
| 2% | Anytime goal scorer | 569 | -120.50 | -21.2% | -39.8% to -3.0% |
| 2% | Alternate points | 468 | -27.18 | -5.8% | -29.6% to +20.4% |
| 2% | Alternate shots | 521 | -47.47 | -9.1% | -31.8% to +16.0% |
| 10% | All | 858 | -121.78 | -14.2% | -31.8% to +5.8% |
| 10% | Anytime goal scorer | 400 | -73.45 | -18.4% | -40.8% to +6.5% |
| 10% | Alternate points | 218 | -23.49 | -10.8% | -43.1% to +32.6% |
| 10% | Alternate shots | 240 | -24.85 | -10.4% | -50.5% to +36.3% |

**2% threshold:** Losing: ROI -12.5% (95% range -25.7% to -0.7%).

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 9278 | 24.8% | 24.5% | 0.1236 | 0.1238 |
| Anytime goal scorer | 1470 | 16.4% | 14.5% | 0.1166 | 0.1156 |
| Alternate points | 3470 | 17.1% | 16.2% | 0.1009 | 0.1012 |
| Alternate shots | 4338 | 33.8% | 34.4% | 0.1441 | 0.1447 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

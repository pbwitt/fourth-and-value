# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 8 refreshes, model nhl-v2.1, nhl-v2.3, nhl-v2.4. Generated 2026-10-09T11:08:19.927149Z.

5888 contracts priced; results: lost 4347, unresolved_participation 87, won 1454.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 994 | -140.76 | -14.2% | -31.0% to +2.5% |
| 2% | Anytime goal scorer | 353 | -80.10 | -22.7% | -47.9% to +5.3% |
| 2% | Alternate points | 303 | -44.17 | -14.6% | -34.5% to +7.0% |
| 2% | Alternate shots | 338 | -16.48 | -4.9% | -34.2% to +24.0% |
| 10% | All | 538 | -124.09 | -23.1% | -44.3% to -0.2% |
| 10% | Anytime goal scorer | 248 | -66.50 | -26.8% | -56.5% to +4.5% |
| 10% | Alternate points | 135 | -45.08 | -33.4% | -54.5% to -13.7% |
| 10% | Alternate shots | 155 | -12.51 | -8.1% | -52.6% to +45.5% |

**2% threshold:** Inconclusive: ROI -14.2%, and the 95% range (-31.0% to +2.5%) includes zero.

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 5801 | 24.5% | 25.1% | 0.1259 | 0.1253 |
| Anytime goal scorer | 894 | 16.4% | 15.5% | 0.1223 | 0.1200 |
| Alternate points | 2211 | 16.9% | 17.5% | 0.1063 | 0.1051 |
| Alternate shots | 2696 | 33.5% | 34.4% | 0.1432 | 0.1437 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

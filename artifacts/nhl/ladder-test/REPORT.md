# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 7 refreshes, model nhl-v2.1, nhl-v2.3, nhl-v2.4. Generated 2026-10-08T11:07:44.293113Z.

3509 contracts priced; results: lost 2552, unresolved_participation 53, won 904.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 568 | -87.65 | -15.4% | -41.2% to +11.0% |
| 2% | Anytime goal scorer | 218 | -59.50 | -27.3% | -66.4% to +13.8% |
| 2% | Alternate points | 154 | -4.26 | -2.8% | -38.5% to +34.7% |
| 2% | Alternate shots | 196 | -23.89 | -12.2% | -54.3% to +31.5% |
| 10% | All | 318 | -78.64 | -24.7% | -50.0% to +6.8% |
| 10% | Anytime goal scorer | 153 | -46.80 | -30.6% | -71.6% to +14.9% |
| 10% | Alternate points | 70 | -16.14 | -23.1% | -50.5% to +5.3% |
| 10% | Alternate shots | 95 | -15.70 | -16.5% | -70.1% to +61.1% |

**2% threshold:** Inconclusive: ROI -15.4%, and the 95% range (-41.2% to +11.0%) includes zero.

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 3456 | 25.0% | 26.2% | 0.1254 | 0.1237 |
| Anytime goal scorer | 549 | 16.5% | 16.2% | 0.1235 | 0.1201 |
| Alternate points | 1281 | 17.5% | 19.3% | 0.1134 | 0.1108 |
| Alternate shots | 1626 | 33.7% | 34.9% | 0.1355 | 0.1351 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

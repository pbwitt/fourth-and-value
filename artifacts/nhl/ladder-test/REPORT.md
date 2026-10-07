# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 7 refreshes, model nhl-v2.1, nhl-v2.3, nhl-v2.4. Generated 2026-10-07T20:32:19.208591Z.

3509 contracts priced; results: lost 2083, pending 640, unresolved_participation 49, won 737.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 429 | -37.50 | -8.7% | -41.7% to +19.5% |
| 2% | Anytime goal scorer | 169 | -39.50 | -23.4% | -68.0% to +24.3% |
| 2% | Alternate points | 116 | -8.31 | -7.2% | -47.3% to +19.8% |
| 2% | Alternate shots | 144 | +10.31 | 7.2% | -50.3% to +60.8% |
| 10% | All | 245 | -43.87 | -17.9% | -52.8% to +22.9% |
| 10% | Anytime goal scorer | 120 | -26.55 | -22.1% | -73.5% to +30.4% |
| 10% | Alternate points | 56 | -20.67 | -36.9% | -66.7% to -19.3% |
| 10% | Alternate shots | 69 | +3.35 | 4.8% | -66.9% to +103.2% |

**2% threshold:** Inconclusive: ROI -8.7%, and the 95% range (-41.7% to +19.5%) includes zero.

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 2820 | 24.5% | 26.1% | 0.1248 | 0.1229 |
| Anytime goal scorer | 445 | 16.3% | 16.2% | 0.1234 | 0.1204 |
| Alternate points | 1056 | 17.0% | 19.5% | 0.1137 | 0.1103 |
| Alternate shots | 1319 | 33.3% | 34.8% | 0.1341 | 0.1339 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

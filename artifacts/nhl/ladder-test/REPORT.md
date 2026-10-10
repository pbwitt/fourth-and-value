# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 10 refreshes, model nhl-v2.1, nhl-v2.3, nhl-v2.4. Generated 2026-10-10T20:32:35.936495Z.

9420 contracts priced; results: lost 5094, pending 2581, unresolved_participation 99, won 1646.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 1134 | -161.87 | -14.3% | -29.2% to +1.8% |
| 2% | Anytime goal scorer | 409 | -63.90 | -15.6% | -38.1% to +9.1% |
| 2% | Alternate points | 352 | -56.56 | -16.1% | -35.4% to +8.0% |
| 2% | Alternate shots | 373 | -41.41 | -11.1% | -39.9% to +16.9% |
| 10% | All | 617 | -129.75 | -21.0% | -39.9% to +0.4% |
| 10% | Anytime goal scorer | 287 | -40.70 | -14.2% | -40.9% to +16.9% |
| 10% | Alternate points | 157 | -60.78 | -38.7% | -57.0% to -18.9% |
| 10% | Alternate shots | 173 | -28.27 | -16.3% | -57.2% to +32.7% |

**2% threshold:** Inconclusive: ROI -14.3%, and the 95% range (-29.2% to +1.8%) includes zero.

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 6740 | 24.5% | 24.4% | 0.1242 | 0.1238 |
| Anytime goal scorer | 1031 | 16.4% | 15.3% | 0.1220 | 0.1203 |
| Alternate points | 2572 | 16.8% | 16.6% | 0.1028 | 0.1021 |
| Alternate shots | 3137 | 33.5% | 33.8% | 0.1424 | 0.1429 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

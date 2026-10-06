# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 5 refreshes, model nhl-v2.1, nhl-v2.3, nhl-v2.4. Generated 2026-10-06T20:33:28.825823Z.

2869 contracts priced; results: lost 847, pending 1704, unresolved_participation 28, won 290.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 128 | -2.47 | -1.9% | -53.0% to +66.2% |
| 2% | Anytime goal scorer | 82 | -18.30 | -22.3% | -85.2% to +58.1% |
| 2% | Alternate points | 21 | -18.90 | -90.0% | -100.0% to -76.7% |
| 2% | Alternate shots | 25 | +34.73 | 138.9% | -100.0% to +518.5% |
| 10% | All | 83 | +24.69 | 29.7% | -37.9% to +130.0% |
| 10% | Anytime goal scorer | 59 | +0.85 | 1.4% | -80.1% to +92.9% |
| 10% | Alternate points | 11 | -8.90 | -80.9% | -100.0% to -58.0% |
| 10% | Alternate shots | 13 | +32.74 | 251.8% | +126.9% to +668.3% |

**2% threshold:** Inconclusive: ROI -1.9%, and the 95% range (-53.0% to +66.2%) includes zero.

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 1137 | 24.7% | 25.5% | 0.1226 | 0.1202 |
| Anytime goal scorer | 200 | 16.6% | 15.0% | 0.1173 | 0.1155 |
| Alternate points | 417 | 18.1% | 17.7% | 0.1048 | 0.1002 |
| Alternate shots | 520 | 33.1% | 35.8% | 0.1390 | 0.1381 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

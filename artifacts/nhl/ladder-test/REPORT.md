# NHL milestone-prop test

Research only; no bets were placed. Window 2026-10-04 to 2026-10-11 (end exclusive), 4 refreshes, model nhl-v2.1, nhl-v2.3. Generated 2026-10-06T11:07:52.641022Z.

2418 contracts priced; results: lost 763, pending 1374, unresolved_participation 24, won 257.

## Model bets (best verified price, estimated EV at or above the threshold)

| Threshold | Market | Graded | Net units | ROI | 95% range |
|---|---|---|---|---|---|
| 2% | All | 110 | +4.43 | 4.0% | -57.2% to +88.9% |
| 2% | Anytime goal scorer | 65 | -12.40 | -19.1% | -93.5% to +78.6% |
| 2% | Alternate points | 21 | -18.90 | -90.0% | -100.0% to -76.7% |
| 2% | Alternate shots | 24 | +35.73 | 148.9% | +75.0% to +518.5% |
| 10% | All | 72 | +24.59 | 34.2% | -49.3% to +152.2% |
| 10% | Anytime goal scorer | 48 | +0.75 | 1.6% | -91.3% to +104.7% |
| 10% | Alternate points | 11 | -8.90 | -80.9% | -100.0% to -58.0% |
| 10% | Alternate shots | 13 | +32.74 | 251.8% | +126.9% to +668.3% |

**2% threshold:** Inconclusive: ROI +4.0%, and the 95% range (-57.2% to +88.9%) includes zero.

## Calibration (won or lost contracts with a model probability)

| Market | Contracts | Model mean | Hit rate | Model Brier | Best-price Brier |
|---|---|---|---|---|---|
| All | 1020 | 23.8% | 25.2% | 0.1176 | 0.1148 |
| Anytime goal scorer | 167 | 16.5% | 15.0% | 0.1184 | 0.1164 |
| Alternate points | 393 | 16.9% | 17.3% | 0.1024 | 0.0972 |
| Alternate shots | 460 | 32.4% | 35.7% | 0.1302 | 0.1293 |

Lower Brier is better. The best price still includes the book's margin, so it is a reference, not a fair probability.

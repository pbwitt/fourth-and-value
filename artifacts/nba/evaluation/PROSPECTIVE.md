# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Wed, Nov 18, 2026, 5:06 PM EST. 195 of 195 tipped games graded through Nov 17; 18442 main-line and 38171 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 195 | +0.013 (+0.007 to +0.018) | -0.000 (-0.001 to +0.000) | +0.86 (+0.75 to +0.97) | 941-1097-0, -59.16u | not supported |
| **All game lines** | 195 | +0.032 (+0.012 to +0.052) | +0.002 (-0.000 to +0.004) | +0.33 (+0.04 to +0.62) | 24-59-0, -28.04u | not supported |
| Moneyline | 195 | +0.044 (+0.018 to +0.071) | +0.003 (-0.000 to +0.005) | +0.67 (+0.20 to +1.18) | 11-51-0, -32.51u | not supported |
| Spread | 193 | +0.028 (-0.002 to +0.056) | +0.001 (-0.002 to +0.004) | -0.06 (-0.29 to +0.17) | 12-6-0, +5.51u | not supported |
| Game total | 193 | +0.025 (+0.003 to +0.048) | +0.001 (-0.001 to +0.004) | +0.03 (-0.20 to +0.29) | 1-2-0, -1.05u | not supported |
| Assists | 195 | +0.018 (+0.009 to +0.027) | -0.000 (-0.001 to +0.001) | +1.03 (+0.82 to +1.23) | 135-171-0, -16.77u | not supported |
| Blocks | 129 | +0.008 (-0.011 to +0.031) | -0.000 (-0.002 to +0.002) | +0.69 (+0.33 to +1.09) | 1-4-0, -3.07u | insufficient sample |
| Points | 195 | +0.007 (+0.000 to +0.013) | -0.001 (-0.001 to +0.000) | +0.86 (+0.68 to +1.05) | 153-172-0, -9.98u | not supported |
| Points + assists | 195 | +0.012 (+0.004 to +0.021) | -0.000 (-0.001 to +0.000) | +0.84 (+0.67 to +1.01) | 35-52-0, -16.59u | not supported |
| Points + rebounds | 195 | +0.006 (-0.001 to +0.013) | -0.001 (-0.002 to -0.000) | +0.90 (+0.74 to +1.06) | 104-113-0, -3.87u | supported |
| Points + rebounds + assists | 195 | +0.016 (+0.006 to +0.025) | -0.000 (-0.001 to +0.001) | +0.83 (+0.63 to +1.01) | 114-142-0, -24.14u | not supported |
| Rebounds | 195 | +0.015 (+0.008 to +0.024) | -0.000 (-0.001 to +0.001) | +1.01 (+0.85 to +1.17) | 165-166-0, +10.11u | not supported |
| Rebounds + assists | 195 | +0.021 (+0.010 to +0.033) | -0.000 (-0.001 to +0.001) | +0.87 (+0.68 to +1.07) | 113-108-0, +7.93u | not supported |
| Steals | 160 | -0.000 (-0.012 to +0.011) | -0.001 (-0.002 to +0.000) | +0.38 (+0.14 to +0.67) | 4-2-0, +2.02u | not supported |
| Threes made | 195 | +0.013 (+0.006 to +0.020) | +0.000 (-0.001 to +0.001) | +0.56 (+0.43 to +0.70) | 117-167-0, -4.79u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 939 bets, 401-538-0, -64.65u (ROI -0.07 (-0.14 to +0.01))
- Props, model: 1099 bets, 540-559-0, +5.49u (ROI +0.00 (-0.06 to +0.07))
- Game lines, price: 19 bets, 0-19-0, -19.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 64 bets, 24-40-0, -9.04u (ROI -0.14 (-0.43 to +0.18))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

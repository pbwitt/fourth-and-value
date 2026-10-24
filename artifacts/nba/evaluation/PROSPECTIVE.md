# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Oct 24, 2026, 5:03 PM EDT. 28 of 28 tipped games graded through Oct 23; 2806 main-line and 6090 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 28 | +0.011 (-0.009 to +0.034) | -0.002 (-0.004 to -0.000) | +0.65 (+0.43 to +0.86) | 216-210-0, +19.51u | insufficient sample |
| **All game lines** | 28 | +0.000 (-0.064 to +0.070) | -0.002 (-0.009 to +0.004) | +0.62 (+0.14 to +1.13) | 7-9-0, -2.07u | insufficient sample |
| Moneyline | 28 | +0.018 (-0.077 to +0.097) | -0.001 (-0.010 to +0.008) | +0.89 (+0.07 to +1.61) | 3-6-0, -2.82u | insufficient sample |
| Spread | 27 | -0.032 (-0.125 to +0.062) | -0.006 (-0.015 to +0.004) | +0.48 (+0.07 to +0.91) | 4-2-0, +1.76u | insufficient sample |
| Game total | 28 | +0.014 (-0.056 to +0.094) | -0.000 (-0.008 to +0.006) | +0.11 | 0-1-0, -1.00u | insufficient sample |
| Assists | 28 | +0.018 (-0.012 to +0.050) | -0.002 (-0.005 to +0.001) | +0.92 (+0.50 to +1.38) | 38-42-0, +2.67u | insufficient sample |
| Blocks | 22 | +0.037 (-0.029 to +0.115) | +0.002 (-0.005 to +0.009) | +0.29 (-0.41 to +1.25) | 1-0-0, +0.93u | insufficient sample |
| Points | 28 | +0.001 (-0.024 to +0.024) | -0.002 (-0.005 to -0.000) | +0.52 (+0.15 to +0.87) | 37-39-0, -1.26u | insufficient sample |
| Points + assists | 28 | +0.009 (-0.020 to +0.043) | -0.002 (-0.005 to +0.001) | +0.75 (+0.35 to +1.14) | 6-14-0, -8.00u | insufficient sample |
| Points + rebounds | 28 | -0.009 (-0.037 to +0.016) | -0.003 (-0.006 to -0.001) | +0.63 (+0.37 to +0.94) | 18-10-0, +7.67u | insufficient sample |
| Points + rebounds + assists | 28 | +0.002 (-0.032 to +0.040) | -0.003 (-0.007 to -0.001) | +0.60 (+0.22 to +0.96) | 26-23-0, +2.71u | insufficient sample |
| Rebounds | 28 | +0.026 (-0.001 to +0.054) | -0.001 (-0.003 to +0.002) | +0.75 (+0.47 to +1.07) | 44-33-0, +13.35u | insufficient sample |
| Rebounds + assists | 28 | +0.039 (-0.005 to +0.089) | -0.001 (-0.005 to +0.003) | +0.84 (+0.43 to +1.25) | 21-19-0, +3.00u | insufficient sample |
| Steals | 21 | +0.019 (-0.014 to +0.054) | -0.000 (-0.003 to +0.003) | +0.18 (-0.34 to +0.78) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 28 | +0.002 (-0.021 to +0.028) | -0.002 (-0.004 to +0.000) | +0.38 (+0.11 to +0.67) | 21-29-0, -4.58u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 135 bets, 60-75-0, -5.62u (ROI -0.04 (-0.23 to +0.13))
- Props, model: 291 bets, 156-135-0, +25.12u (ROI +0.09 (-0.05 to +0.22))
- Game lines, model: 16 bets, 7-9-0, -2.07u (ROI -0.13 (-0.65 to +0.42))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

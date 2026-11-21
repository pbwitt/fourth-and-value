# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Nov 21, 2026, 5:05 PM EST. 221 of 221 tipped games graded through Nov 21; 20983 main-line and 43744 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 221 | +0.012 (+0.007 to +0.017) | -0.000 (-0.001 to +0.000) | +0.86 (+0.76 to +0.95) | 1065-1237-0, -57.76u | not supported |
| **All game lines** | 221 | +0.034 (+0.016 to +0.052) | +0.002 (+0.000 to +0.004) | +0.40 (+0.12 to +0.68) | 26-63-0, -25.13u | not supported |
| Moneyline | 221 | +0.044 (+0.021 to +0.067) | +0.003 (+0.000 to +0.005) | +0.82 (+0.35 to +1.32) | 13-55-0, -29.60u | not supported |
| Spread | 219 | +0.032 (+0.004 to +0.060) | +0.001 (-0.001 to +0.004) | -0.06 (-0.29 to +0.16) | 12-6-0, +5.51u | not supported |
| Game total | 219 | +0.027 (+0.007 to +0.049) | +0.002 (-0.000 to +0.004) | +0.04 (-0.22 to +0.27) | 1-2-0, -1.05u | not supported |
| Assists | 221 | +0.015 (+0.007 to +0.025) | -0.000 (-0.001 to +0.000) | +1.00 (+0.81 to +1.20) | 150-189-0, -16.79u | not supported |
| Blocks | 146 | +0.007 (-0.010 to +0.026) | -0.000 (-0.002 to +0.001) | +0.76 (+0.35 to +1.14) | 1-4-0, -3.07u | insufficient sample |
| Points | 221 | +0.006 (-0.000 to +0.012) | -0.001 (-0.001 to -0.000) | +0.85 (+0.68 to +1.02) | 173-198-0, -12.88u | supported |
| Points + assists | 221 | +0.011 (+0.003 to +0.019) | -0.000 (-0.001 to +0.000) | +0.84 (+0.68 to +0.99) | 53-61-0, -6.60u | not supported |
| Points + rebounds | 221 | +0.006 (-0.001 to +0.013) | -0.001 (-0.001 to -0.000) | +0.92 (+0.79 to +1.07) | 117-123-0, +0.10u | supported |
| Points + rebounds + assists | 221 | +0.015 (+0.007 to +0.024) | -0.000 (-0.001 to +0.001) | +0.83 (+0.66 to +0.99) | 126-160-0, -29.41u | not supported |
| Rebounds | 221 | +0.015 (+0.008 to +0.022) | -0.000 (-0.001 to +0.001) | +1.04 (+0.88 to +1.20) | 177-183-0, +6.50u | not supported |
| Rebounds + assists | 221 | +0.019 (+0.009 to +0.030) | -0.000 (-0.001 to +0.001) | +0.89 (+0.70 to +1.07) | 131-129-0, +5.82u | not supported |
| Steals | 183 | -0.003 (-0.014 to +0.008) | -0.001 (-0.002 to -0.000) | +0.41 (+0.14 to +0.71) | 5-2-0, +3.43u | supported |
| Threes made | 221 | +0.013 (+0.006 to +0.019) | +0.000 (-0.001 to +0.001) | +0.53 (+0.41 to +0.66) | 132-188-0, -4.85u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1075 bets, 458-617-0, -75.64u (ROI -0.07 (-0.14 to +0.01))
- Props, model: 1227 bets, 607-620-0, +17.87u (ROI +0.01 (-0.05 to +0.09))
- Game lines, price: 23 bets, 2-21-0, -14.09u (ROI -0.61 (-1.00 to -0.08))
- Game lines, model: 66 bets, 24-42-0, -11.04u (ROI -0.17 (-0.47 to +0.14))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

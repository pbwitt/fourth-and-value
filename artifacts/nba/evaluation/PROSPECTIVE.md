# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sun, Oct 25, 2026, 6:47 AM EDT. 36 of 36 tipped games graded through Oct 24; 3505 main-line and 7563 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 36 | +0.011 (-0.006 to +0.027) | -0.002 (-0.003 to -0.000) | +0.67 (+0.46 to +0.89) | 253-253-0, +18.34u | insufficient sample |
| **All game lines** | 36 | +0.012 (-0.050 to +0.065) | -0.001 (-0.007 to +0.004) | +0.54 (+0.07 to +1.06) | 9-14-0, -4.65u | insufficient sample |
| Moneyline | 36 | +0.019 (-0.061 to +0.098) | -0.001 (-0.009 to +0.007) | +0.66 (-0.13 to +1.44) | 4-10-0, -5.36u | insufficient sample |
| Spread | 35 | -0.017 (-0.099 to +0.065) | -0.004 (-0.013 to +0.004) | +0.41 (-0.01 to +0.82) | 5-3-0, +1.71u | insufficient sample |
| Game total | 36 | +0.031 (-0.026 to +0.097) | +0.002 (-0.004 to +0.007) | +0.44 (-0.33 to +1.23) | 0-1-0, -1.00u | insufficient sample |
| Assists | 36 | +0.019 (-0.007 to +0.048) | -0.002 (-0.004 to +0.001) | +0.87 (+0.45 to +1.30) | 45-52-0, +1.57u | insufficient sample |
| Blocks | 28 | +0.013 (-0.043 to +0.078) | -0.000 (-0.006 to +0.005) | +0.28 (-0.29 to +1.11) | 1-1-0, -0.07u | insufficient sample |
| Points | 36 | +0.002 (-0.019 to +0.022) | -0.002 (-0.004 to +0.000) | +0.58 (+0.25 to +0.97) | 43-44-0, +0.09u | insufficient sample |
| Points + assists | 36 | +0.010 (-0.017 to +0.038) | -0.002 (-0.004 to +0.001) | +0.70 (+0.38 to +1.03) | 8-18-0, -10.06u | insufficient sample |
| Points + rebounds | 36 | -0.006 (-0.028 to +0.017) | -0.003 (-0.005 to -0.001) | +0.66 (+0.39 to +0.92) | 19-16-0, +2.67u | insufficient sample |
| Points + rebounds + assists | 36 | +0.005 (-0.024 to +0.032) | -0.003 (-0.005 to -0.000) | +0.67 (+0.30 to +1.05) | 27-26-0, +0.81u | insufficient sample |
| Rebounds | 36 | +0.023 (+0.000 to +0.045) | -0.001 (-0.002 to +0.001) | +0.76 (+0.51 to +1.05) | 50-37-0, +15.31u | insufficient sample |
| Rebounds + assists | 36 | +0.027 (-0.009 to +0.069) | -0.002 (-0.005 to +0.001) | +0.87 (+0.48 to +1.27) | 29-25-0, +5.07u | insufficient sample |
| Steals | 27 | +0.024 (-0.005 to +0.055) | +0.001 (-0.002 to +0.003) | +0.16 (-0.32 to +0.77) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 36 | +0.004 (-0.015 to +0.025) | -0.002 (-0.004 to +0.000) | +0.45 (+0.22 to +0.75) | 27-33-0, -0.06u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 179 bets, 80-99-0, -5.86u (ROI -0.03 (-0.18 to +0.10))
- Props, model: 327 bets, 173-154-0, +24.20u (ROI +0.07 (-0.07 to +0.21))
- Game lines, price: 2 bets, 0-2-0, -2.00u (ROI -1.00)
- Game lines, model: 21 bets, 9-12-0, -2.65u (ROI -0.13 (-0.65 to +0.36))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

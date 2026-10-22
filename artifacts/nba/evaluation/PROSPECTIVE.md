# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Thu, Oct 22, 2026, 5:03 PM EDT. 14 of 14 tipped games graded through Oct 21; 1451 main-line and 3126 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 14 | +0.026 (-0.009 to +0.059) | -0.002 (-0.005 to +0.001) | +0.49 (+0.19 to +0.77) | 140-128-0, +17.72u | insufficient sample |
| **All game lines** | 14 | -0.030 (-0.127 to +0.079) | -0.006 (-0.017 to +0.005) | +0.22 (-0.39 to +0.78) | 4-4-0, +0.20u | insufficient sample |
| Moneyline | 14 | -0.013 (-0.149 to +0.127) | -0.005 (-0.022 to +0.010) | +0.26 (-0.85 to +1.26) | 2-2-0, +0.33u | insufficient sample |
| Spread | 13 | -0.064 (-0.202 to +0.080) | -0.010 (-0.027 to +0.004) | +0.14 (-0.32 to +0.60) | 2-1-0, +0.87u | insufficient sample |
| Game total | 14 | -0.014 (-0.122 to +0.111) | -0.004 (-0.015 to +0.008) | +0.26 | 0-1-0, -1.00u | insufficient sample |
| Assists | 14 | +0.028 (-0.015 to +0.071) | -0.002 (-0.006 to +0.002) | +0.71 (-0.06 to +1.48) | 25-26-0, +2.82u | insufficient sample |
| Blocks | 14 | +0.052 (-0.065 to +0.180) | +0.003 (-0.008 to +0.014) | +0.16 | — | insufficient sample |
| Points | 14 | +0.011 (-0.029 to +0.047) | -0.002 (-0.006 to +0.001) | +0.22 (-0.21 to +0.62) | 24-20-0, +3.76u | insufficient sample |
| Points + assists | 14 | +0.029 (-0.020 to +0.080) | -0.001 (-0.006 to +0.003) | +0.42 (-0.03 to +0.84) | 4-10-0, -6.07u | insufficient sample |
| Points + rebounds | 14 | +0.003 (-0.043 to +0.045) | -0.003 (-0.008 to +0.001) | +0.49 (+0.19 to +0.80) | 8-3-0, +4.83u | insufficient sample |
| Points + rebounds + assists | 14 | +0.023 (-0.034 to +0.080) | -0.003 (-0.008 to +0.002) | +0.32 (-0.08 to +0.71) | 16-16-0, -0.13u | insufficient sample |
| Rebounds | 14 | +0.041 (-0.003 to +0.096) | -0.001 (-0.005 to +0.004) | +0.88 (+0.43 to +1.38) | 33-22-0, +11.91u | insufficient sample |
| Rebounds + assists | 14 | +0.079 (+0.009 to +0.156) | +0.001 (-0.005 to +0.006) | +0.73 (+0.12 to +1.31) | 11-15-0, -3.49u | insufficient sample |
| Steals | 13 | +0.016 (-0.033 to +0.067) | -0.001 (-0.006 to +0.003) | +0.45 | 4-1-0, +3.02u | insufficient sample |
| Threes made | 14 | +0.003 (-0.039 to +0.046) | -0.003 (-0.006 to +0.001) | +0.10 (-0.14 to +0.38) | 15-15-0, +1.07u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 62 bets, 30-32-0, +3.26u (ROI +0.05 (-0.19 to +0.27))
- Props, model: 206 bets, 110-96-0, +14.45u (ROI +0.07 (-0.09 to +0.24))
- Game lines, model: 8 bets, 4-4-0, +0.20u (ROI +0.02)

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

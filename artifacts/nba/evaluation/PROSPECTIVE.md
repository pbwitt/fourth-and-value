# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Fri, Oct 30, 2026, 5:03 PM EDT. 71 of 71 tipped games graded through Oct 29; 6698 main-line and 14273 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 71 | +0.014 (+0.003 to +0.024) | -0.001 (-0.002 to +0.000) | +0.84 (+0.65 to +1.03) | 440-461-0, +17.02u | insufficient sample |
| **All game lines** | 71 | +0.033 (-0.007 to +0.072) | +0.001 (-0.003 to +0.005) | +0.45 (+0.02 to +0.97) | 15-28-0, -11.69u | insufficient sample |
| Moneyline | 71 | +0.054 (+0.004 to +0.111) | +0.003 (-0.002 to +0.008) | +0.75 (+0.04 to +1.55) | 6-22-0, -14.30u | insufficient sample |
| Spread | 70 | +0.012 (-0.047 to +0.072) | -0.002 (-0.007 to +0.004) | +0.05 (-0.45 to +0.51) | 9-5-0, +3.61u | insufficient sample |
| Game total | 70 | +0.032 (-0.007 to +0.069) | +0.002 (-0.002 to +0.006) | +0.26 (-0.23 to +0.84) | 0-1-0, -1.00u | insufficient sample |
| Assists | 71 | +0.021 (+0.003 to +0.039) | -0.001 (-0.002 to +0.001) | +1.00 (+0.63 to +1.34) | 69-83-0, -3.34u | insufficient sample |
| Blocks | 50 | +0.008 (-0.023 to +0.041) | -0.001 (-0.004 to +0.003) | +0.65 (-0.05 to +1.36) | 1-2-0, -1.07u | insufficient sample |
| Points | 71 | +0.004 (-0.010 to +0.017) | -0.001 (-0.003 to +0.000) | +0.76 (+0.52 to +1.04) | 69-75-0, -2.43u | insufficient sample |
| Points + assists | 71 | +0.010 (-0.006 to +0.027) | -0.001 (-0.003 to +0.001) | +0.90 (+0.61 to +1.20) | 19-25-0, -5.74u | insufficient sample |
| Points + rebounds | 71 | +0.002 (-0.012 to +0.016) | -0.002 (-0.003 to -0.000) | +0.78 (+0.53 to +1.03) | 40-33-0, +7.56u | insufficient sample |
| Points + rebounds + assists | 71 | +0.015 (-0.003 to +0.032) | -0.001 (-0.003 to +0.000) | +0.92 (+0.62 to +1.24) | 50-60-0, -9.92u | insufficient sample |
| Rebounds | 71 | +0.026 (+0.011 to +0.042) | +0.000 (-0.001 to +0.001) | +0.96 (+0.71 to +1.24) | 76-74-0, +5.90u | insufficient sample |
| Rebounds + assists | 71 | +0.027 (+0.004 to +0.052) | -0.001 (-0.003 to +0.001) | +0.93 (+0.63 to +1.23) | 56-46-0, +11.50u | insufficient sample |
| Steals | 52 | +0.014 (-0.008 to +0.036) | -0.000 (-0.002 to +0.002) | +0.31 (-0.10 to +0.77) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 71 | +0.008 (-0.005 to +0.021) | -0.001 (-0.002 to +0.000) | +0.64 (+0.42 to +0.85) | 56-62-0, +11.55u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 345 bets, 160-185-0, +1.72u (ROI +0.00 (-0.11 to +0.11))
- Props, model: 556 bets, 280-276-0, +15.30u (ROI +0.03 (-0.07 to +0.13))
- Game lines, price: 7 bets, 0-7-0, -7.00u (ROI -1.00)
- Game lines, model: 36 bets, 15-21-0, -4.69u (ROI -0.13 (-0.52 to +0.26))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

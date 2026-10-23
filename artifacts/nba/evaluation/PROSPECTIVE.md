# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Fri, Oct 23, 2026, 6:48 AM EDT. 16 of 16 tipped games graded through Oct 22; 1673 main-line and 3615 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 16 | +0.021 (-0.012 to +0.053) | -0.002 (-0.005 to +0.001) | +0.60 (+0.31 to +0.87) | 152-142-0, +16.42u | insufficient sample |
| **All game lines** | 16 | -0.006 (-0.103 to +0.084) | -0.004 (-0.014 to +0.006) | +0.32 (-0.22 to +0.86) | 4-6-0, -1.80u | insufficient sample |
| Moneyline | 16 | +0.018 (-0.122 to +0.138) | -0.002 (-0.018 to +0.012) | +0.36 (-0.53 to +1.30) | 2-4-0, -1.67u | insufficient sample |
| Spread | 15 | -0.016 (-0.159 to +0.132) | -0.005 (-0.020 to +0.009) | +0.30 (-0.18 to +0.81) | 2-1-0, +0.87u | insufficient sample |
| Game total | 16 | -0.019 (-0.112 to +0.098) | -0.004 (-0.014 to +0.007) | +0.26 | 0-1-0, -1.00u | insufficient sample |
| Assists | 16 | +0.020 (-0.020 to +0.065) | -0.003 (-0.007 to +0.001) | +0.74 (+0.07 to +1.45) | 29-30-0, +3.90u | insufficient sample |
| Blocks | 16 | +0.059 (-0.037 to +0.154) | +0.004 (-0.006 to +0.012) | +0.38 (-0.38 to +1.50) | 1-0-0, +0.93u | insufficient sample |
| Points | 16 | +0.016 (-0.020 to +0.050) | -0.001 (-0.005 to +0.002) | +0.44 (+0.00 to +0.92) | 24-26-0, -2.24u | insufficient sample |
| Points + assists | 16 | +0.018 (-0.027 to +0.064) | -0.002 (-0.006 to +0.002) | +0.60 (+0.14 to +1.06) | 5-10-0, -5.10u | insufficient sample |
| Points + rebounds | 16 | -0.004 (-0.046 to +0.039) | -0.004 (-0.008 to -0.000) | +0.71 (+0.34 to +1.11) | 10-4-0, +5.67u | insufficient sample |
| Points + rebounds + assists | 16 | +0.014 (-0.038 to +0.062) | -0.004 (-0.008 to +0.001) | +0.57 (+0.10 to +1.04) | 18-16-0, +1.73u | insufficient sample |
| Rebounds | 16 | +0.033 (-0.009 to +0.078) | -0.001 (-0.005 to +0.003) | +0.85 (+0.43 to +1.34) | 34-22-0, +12.78u | insufficient sample |
| Rebounds + assists | 16 | +0.054 (-0.009 to +0.127) | -0.001 (-0.006 to +0.004) | +0.79 (+0.24 to +1.34) | 12-16-0, -3.33u | insufficient sample |
| Steals | 15 | +0.025 (-0.019 to +0.070) | -0.000 (-0.005 to +0.004) | +0.20 (-0.44 to +0.92) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 16 | +0.009 (-0.025 to +0.046) | -0.002 (-0.005 to +0.001) | +0.23 (-0.13 to +0.70) | 15-17-0, -0.93u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 75 bets, 35-40-0, +1.27u (ROI +0.02 (-0.21 to +0.21))
- Props, model: 219 bets, 117-102-0, +15.15u (ROI +0.07 (-0.08 to +0.25))
- Game lines, model: 10 bets, 4-6-0, -1.80u (ROI -0.18)

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

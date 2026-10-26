# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Mon, Oct 26, 2026, 5:04 PM EDT. 43 of 43 tipped games graded through Oct 25; 4106 main-line and 8853 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 43 | +0.012 (-0.003 to +0.026) | -0.001 (-0.003 to -0.000) | +0.68 (+0.47 to +0.89) | 282-286-0, +19.15u | insufficient sample |
| **All game lines** | 43 | +0.004 (-0.046 to +0.049) | -0.002 (-0.006 to +0.003) | +0.67 (+0.27 to +1.14) | 11-16-0, -4.71u | insufficient sample |
| Moneyline | 43 | +0.020 (-0.052 to +0.085) | -0.000 (-0.007 to +0.007) | +0.90 (+0.21 to +1.62) | 4-12-0, -7.36u | insufficient sample |
| Spread | 42 | -0.027 (-0.096 to +0.045) | -0.005 (-0.013 to +0.002) | +0.41 (+0.03 to +0.78) | 7-3-0, +3.65u | insufficient sample |
| Game total | 43 | +0.018 (-0.033 to +0.075) | +0.001 (-0.004 to +0.006) | +0.44 (-0.34 to +1.16) | 0-1-0, -1.00u | insufficient sample |
| Assists | 43 | +0.019 (-0.004 to +0.042) | -0.002 (-0.004 to +0.001) | +0.93 (+0.56 to +1.30) | 47-56-0, -0.14u | insufficient sample |
| Blocks | 33 | +0.005 (-0.044 to +0.056) | -0.001 (-0.005 to +0.004) | +0.28 (-0.27 to +1.04) | 1-1-0, -0.07u | insufficient sample |
| Points | 43 | +0.004 (-0.016 to +0.022) | -0.002 (-0.003 to +0.000) | +0.51 (+0.21 to +0.84) | 50-51-0, +0.85u | insufficient sample |
| Points + assists | 43 | +0.012 (-0.012 to +0.038) | -0.001 (-0.004 to +0.001) | +0.67 (+0.38 to +0.95) | 10-19-0, -8.83u | insufficient sample |
| Points + rebounds | 43 | -0.004 (-0.026 to +0.018) | -0.003 (-0.005 to -0.001) | +0.61 (+0.36 to +0.86) | 22-18-0, +3.86u | insufficient sample |
| Points + rebounds + assists | 43 | +0.008 (-0.018 to +0.033) | -0.002 (-0.005 to -0.000) | +0.71 (+0.38 to +1.05) | 30-27-0, +2.94u | insufficient sample |
| Rebounds | 43 | +0.029 (+0.008 to +0.053) | +0.000 (-0.002 to +0.002) | +0.79 (+0.53 to +1.08) | 53-45-0, +10.10u | insufficient sample |
| Rebounds + assists | 43 | +0.028 (-0.005 to +0.064) | -0.001 (-0.004 to +0.002) | +0.89 (+0.54 to +1.26) | 32-31-0, +2.11u | insufficient sample |
| Steals | 33 | +0.019 (-0.006 to +0.046) | +0.000 (-0.002 to +0.003) | +0.21 (-0.25 to +0.70) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 43 | +0.005 (-0.012 to +0.024) | -0.001 (-0.003 to +0.000) | +0.46 (+0.19 to +0.74) | 33-37-0, +5.31u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 206 bets, 92-114-0, -6.31u (ROI -0.03 (-0.17 to +0.10))
- Props, model: 362 bets, 190-172-0, +25.46u (ROI +0.07 (-0.05 to +0.20))
- Game lines, price: 3 bets, 0-3-0, -3.00u (ROI -1.00)
- Game lines, model: 24 bets, 11-13-0, -1.71u (ROI -0.07 (-0.54 to +0.35))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

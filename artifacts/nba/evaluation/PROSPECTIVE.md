# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Tue, Nov 10, 2026, 5:04 PM EST. 139 of 139 tipped games graded through Nov 9; 13146 main-line and 28027 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 139 | +0.015 (+0.007 to +0.022) | -0.000 (-0.001 to +0.000) | +0.85 (+0.73 to +0.99) | 731-852-0, -53.65u | insufficient sample |
| **All game lines** | 139 | +0.034 (+0.010 to +0.059) | +0.002 (-0.001 to +0.004) | +0.38 (+0.04 to +0.77) | 19-44-0, -21.13u | insufficient sample |
| Moneyline | 139 | +0.051 (+0.019 to +0.084) | +0.003 (-0.000 to +0.007) | +0.68 (+0.09 to +1.33) | 9-37-0, -23.72u | insufficient sample |
| Spread | 138 | +0.020 (-0.014 to +0.059) | -0.000 (-0.004 to +0.003) | +0.00 (-0.32 to +0.30) | 10-6-0, +3.59u | insufficient sample |
| Game total | 138 | +0.033 (+0.007 to +0.061) | +0.002 (-0.001 to +0.005) | +0.13 (-0.22 to +0.50) | 0-1-0, -1.00u | insufficient sample |
| Assists | 139 | +0.022 (+0.011 to +0.035) | -0.000 (-0.001 to +0.001) | +0.98 (+0.75 to +1.23) | 105-141-0, -22.59u | insufficient sample |
| Blocks | 96 | +0.011 (-0.012 to +0.037) | -0.000 (-0.002 to +0.002) | +0.56 (+0.12 to +1.00) | 1-3-0, -2.07u | insufficient sample |
| Points | 139 | +0.008 (+0.000 to +0.017) | -0.001 (-0.001 to +0.000) | +0.87 (+0.67 to +1.07) | 124-129-0, +1.40u | insufficient sample |
| Points + assists | 139 | +0.014 (+0.003 to +0.024) | -0.000 (-0.001 to +0.001) | +0.86 (+0.66 to +1.06) | 30-40-0, -9.64u | insufficient sample |
| Points + rebounds | 139 | +0.007 (-0.003 to +0.016) | -0.001 (-0.002 to -0.000) | +0.93 (+0.75 to +1.09) | 74-81-0, -4.11u | insufficient sample |
| Points + rebounds + assists | 139 | +0.017 (+0.005 to +0.029) | -0.000 (-0.001 to +0.001) | +0.89 (+0.69 to +1.09) | 89-116-0, -24.84u | insufficient sample |
| Rebounds | 139 | +0.019 (+0.009 to +0.029) | +0.000 (-0.001 to +0.001) | +0.97 (+0.78 to +1.18) | 125-135-0, -2.07u | insufficient sample |
| Rebounds + assists | 139 | +0.025 (+0.011 to +0.042) | -0.000 (-0.001 to +0.001) | +0.89 (+0.65 to +1.11) | 88-88-0, +2.02u | insufficient sample |
| Steals | 112 | +0.008 (-0.006 to +0.022) | -0.000 (-0.002 to +0.001) | +0.34 (+0.05 to +0.66) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 139 | +0.011 (+0.002 to +0.019) | -0.000 (-0.001 to +0.000) | +0.55 (+0.41 to +0.69) | 91-117-0, +6.24u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 678 bets, 292-386-0, -46.13u (ROI -0.07 (-0.15 to +0.02))
- Props, model: 905 bets, 439-466-0, -7.52u (ROI -0.01 (-0.09 to +0.08))
- Game lines, price: 14 bets, 0-14-0, -14.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 49 bets, 19-30-0, -7.13u (ROI -0.15 (-0.49 to +0.19))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

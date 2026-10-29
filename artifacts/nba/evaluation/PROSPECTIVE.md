# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Thu, Oct 29, 2026, 6:48 AM EDT. 68 of 68 tipped games graded through Oct 28; 6381 main-line and 13652 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 68 | +0.013 (+0.003 to +0.025) | -0.001 (-0.002 to +0.000) | +0.83 (+0.63 to +1.04) | 421-440-0, +17.72u | insufficient sample |
| **All game lines** | 68 | +0.032 (-0.007 to +0.072) | +0.001 (-0.003 to +0.005) | +0.44 (-0.05 to +0.95) | 15-28-0, -11.69u | insufficient sample |
| Moneyline | 68 | +0.051 (+0.001 to +0.107) | +0.003 (-0.003 to +0.008) | +0.76 (+0.00 to +1.53) | 6-22-0, -14.30u | insufficient sample |
| Spread | 67 | +0.009 (-0.049 to +0.070) | -0.002 (-0.008 to +0.004) | -0.00 (-0.50 to +0.43) | 9-5-0, +3.61u | insufficient sample |
| Game total | 67 | +0.034 (-0.007 to +0.075) | +0.002 (-0.002 to +0.006) | +0.28 (-0.27 to +0.91) | 0-1-0, -1.00u | insufficient sample |
| Assists | 68 | +0.019 (+0.000 to +0.039) | -0.001 (-0.003 to +0.001) | +1.02 (+0.66 to +1.41) | 68-79-0, -0.05u | insufficient sample |
| Blocks | 48 | +0.007 (-0.027 to +0.050) | -0.001 (-0.004 to +0.003) | +0.45 (-0.22 to +1.20) | 1-2-0, -1.07u | insufficient sample |
| Points | 68 | +0.006 (-0.008 to +0.020) | -0.001 (-0.003 to +0.000) | +0.73 (+0.47 to +0.99) | 68-73-0, -1.73u | insufficient sample |
| Points + assists | 68 | +0.012 (-0.005 to +0.029) | -0.001 (-0.002 to +0.001) | +0.88 (+0.62 to +1.19) | 16-25-0, -8.79u | insufficient sample |
| Points + rebounds | 68 | +0.002 (-0.013 to +0.018) | -0.002 (-0.003 to -0.000) | +0.78 (+0.53 to +1.04) | 38-32-0, +6.16u | insufficient sample |
| Points + rebounds + assists | 68 | +0.014 (-0.004 to +0.031) | -0.001 (-0.003 to +0.000) | +0.89 (+0.60 to +1.20) | 49-55-0, -6.12u | insufficient sample |
| Rebounds | 68 | +0.026 (+0.012 to +0.042) | +0.000 (-0.001 to +0.002) | +0.98 (+0.73 to +1.24) | 73-69-0, +8.11u | insufficient sample |
| Rebounds + assists | 68 | +0.026 (-0.000 to +0.051) | -0.001 (-0.003 to +0.001) | +0.93 (+0.60 to +1.25) | 51-44-0, +8.50u | insufficient sample |
| Steals | 50 | +0.015 (-0.006 to +0.037) | +0.000 (-0.002 to +0.002) | +0.26 (-0.16 to +0.72) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 68 | +0.008 (-0.004 to +0.021) | -0.001 (-0.002 to +0.000) | +0.63 (+0.42 to +0.87) | 53-60-0, +9.71u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 323 bets, 149-174-0, +0.51u (ROI +0.00 (-0.11 to +0.12))
- Props, model: 538 bets, 272-266-0, +17.21u (ROI +0.03 (-0.08 to +0.14))
- Game lines, price: 7 bets, 0-7-0, -7.00u (ROI -1.00)
- Game lines, model: 36 bets, 15-21-0, -4.69u (ROI -0.13 (-0.50 to +0.25))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

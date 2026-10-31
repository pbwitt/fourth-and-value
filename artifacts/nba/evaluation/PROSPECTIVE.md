# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Oct 31, 2026, 6:49 AM EDT. 81 of 81 tipped games graded through Oct 30; 7664 main-line and 16330 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 81 | +0.014 (+0.005 to +0.023) | -0.001 (-0.002 to +0.000) | +0.88 (+0.71 to +1.05) | 488-518-0, +15.65u | insufficient sample |
| **All game lines** | 81 | +0.039 (+0.004 to +0.075) | +0.002 (-0.001 to +0.005) | +0.40 (-0.01 to +0.82) | 16-30-0, -12.31u | insufficient sample |
| Moneyline | 81 | +0.064 (+0.014 to +0.112) | +0.004 (-0.001 to +0.009) | +0.69 (+0.08 to +1.41) | 7-24-0, -14.92u | insufficient sample |
| Spread | 80 | +0.028 (-0.026 to +0.082) | +0.000 (-0.006 to +0.005) | +0.03 (-0.42 to +0.44) | 9-5-0, +3.61u | insufficient sample |
| Game total | 80 | +0.026 (-0.009 to +0.061) | +0.001 (-0.002 to +0.005) | +0.19 (-0.21 to +0.71) | 0-1-0, -1.00u | insufficient sample |
| Assists | 81 | +0.023 (+0.006 to +0.041) | -0.001 (-0.002 to +0.001) | +1.03 (+0.72 to +1.35) | 72-90-0, -6.47u | insufficient sample |
| Blocks | 57 | +0.006 (-0.024 to +0.039) | -0.001 (-0.003 to +0.003) | +0.68 (+0.05 to +1.35) | 1-2-0, -1.07u | insufficient sample |
| Points | 81 | +0.004 (-0.008 to +0.017) | -0.001 (-0.003 to -0.000) | +0.83 (+0.59 to +1.12) | 80-84-0, -0.23u | insufficient sample |
| Points + assists | 81 | +0.011 (-0.004 to +0.027) | -0.001 (-0.002 to +0.000) | +0.84 (+0.56 to +1.13) | 20-28-0, -7.72u | insufficient sample |
| Points + rebounds | 81 | +0.006 (-0.009 to +0.021) | -0.001 (-0.003 to -0.000) | +0.87 (+0.65 to +1.09) | 47-40-0, +8.33u | insufficient sample |
| Points + rebounds + assists | 81 | +0.016 (+0.000 to +0.034) | -0.001 (-0.002 to +0.001) | +0.91 (+0.66 to +1.21) | 55-64-0, -8.42u | insufficient sample |
| Rebounds | 81 | +0.026 (+0.013 to +0.039) | +0.000 (-0.001 to +0.002) | +0.99 (+0.75 to +1.26) | 83-87-0, +0.92u | insufficient sample |
| Rebounds + assists | 81 | +0.027 (+0.005 to +0.049) | -0.001 (-0.002 to +0.001) | +0.89 (+0.62 to +1.18) | 58-51-0, +8.65u | insufficient sample |
| Steals | 62 | +0.008 (-0.012 to +0.028) | -0.001 (-0.002 to +0.001) | +0.32 (-0.10 to +0.77) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 81 | +0.008 (-0.005 to +0.019) | -0.001 (-0.002 to +0.000) | +0.71 (+0.53 to +0.92) | 68-70-0, +19.65u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 397 bets, 179-218-0, -6.01u (ROI -0.02 (-0.12 to +0.10))
- Props, model: 609 bets, 309-300-0, +21.65u (ROI +0.04 (-0.06 to +0.14))
- Game lines, price: 8 bets, 0-8-0, -8.00u (ROI -1.00)
- Game lines, model: 38 bets, 16-22-0, -4.31u (ROI -0.11 (-0.49 to +0.27))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

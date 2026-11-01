# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sun, Nov 1, 2026, 6:48 AM EST. 88 of 88 tipped games graded through Oct 31; 8323 main-line and 17731 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 88 | +0.012 (+0.003 to +0.023) | -0.001 (-0.002 to -0.000) | +0.88 (+0.71 to +1.04) | 529-567-0, +12.07u | insufficient sample |
| **All game lines** | 88 | +0.037 (+0.003 to +0.070) | +0.002 (-0.002 to +0.005) | +0.42 (+0.03 to +0.80) | 16-31-0, -13.31u | insufficient sample |
| Moneyline | 88 | +0.062 (+0.015 to +0.108) | +0.004 (-0.001 to +0.008) | +0.75 (+0.14 to +1.36) | 7-25-0, -15.92u | insufficient sample |
| Spread | 87 | +0.023 (-0.025 to +0.079) | -0.000 (-0.005 to +0.005) | +0.03 (-0.42 to +0.41) | 9-5-0, +3.61u | insufficient sample |
| Game total | 87 | +0.025 (-0.010 to +0.059) | +0.001 (-0.002 to +0.005) | +0.15 (-0.27 to +0.66) | 0-1-0, -1.00u | insufficient sample |
| Assists | 88 | +0.020 (+0.003 to +0.036) | -0.001 (-0.002 to +0.001) | +1.04 (+0.75 to +1.35) | 75-99-0, -12.40u | insufficient sample |
| Blocks | 60 | +0.004 (-0.023 to +0.034) | -0.001 (-0.004 to +0.002) | +0.65 (+0.04 to +1.32) | 1-2-0, -1.07u | insufficient sample |
| Points | 88 | +0.003 (-0.009 to +0.014) | -0.001 (-0.003 to -0.000) | +0.89 (+0.65 to +1.14) | 86-90-0, +0.09u | insufficient sample |
| Points + assists | 88 | +0.007 (-0.007 to +0.023) | -0.001 (-0.003 to +0.000) | +0.83 (+0.58 to +1.10) | 22-32-0, -9.57u | insufficient sample |
| Points + rebounds | 88 | +0.003 (-0.009 to +0.016) | -0.002 (-0.003 to -0.000) | +0.89 (+0.68 to +1.10) | 53-45-0, +9.56u | insufficient sample |
| Points + rebounds + assists | 88 | +0.012 (-0.004 to +0.027) | -0.001 (-0.003 to +0.000) | +0.91 (+0.66 to +1.17) | 62-71-0, -8.25u | insufficient sample |
| Rebounds | 88 | +0.023 (+0.011 to +0.036) | +0.000 (-0.001 to +0.001) | +0.98 (+0.75 to +1.21) | 90-93-0, +1.71u | insufficient sample |
| Rebounds + assists | 88 | +0.027 (+0.006 to +0.048) | -0.001 (-0.002 to +0.001) | +0.87 (+0.60 to +1.12) | 62-55-0, +8.82u | insufficient sample |
| Steals | 65 | +0.009 (-0.011 to +0.029) | -0.000 (-0.002 to +0.001) | +0.30 (-0.10 to +0.70) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 88 | +0.010 (-0.002 to +0.021) | -0.001 (-0.002 to +0.000) | +0.70 (+0.52 to +0.88) | 74-78-0, +21.16u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 444 bets, 198-246-0, -10.36u (ROI -0.02 (-0.12 to +0.07))
- Props, model: 652 bets, 331-321-0, +22.43u (ROI +0.03 (-0.06 to +0.13))
- Game lines, price: 8 bets, 0-8-0, -8.00u (ROI -1.00)
- Game lines, model: 39 bets, 16-23-0, -5.31u (ROI -0.14 (-0.50 to +0.23))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

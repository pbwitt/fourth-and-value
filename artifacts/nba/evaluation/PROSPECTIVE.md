# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Wed, Oct 28, 2026, 5:07 PM EDT. 56 of 56 tipped games graded through Oct 27; 5334 main-line and 11395 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 56 | +0.013 (+0.000 to +0.026) | -0.001 (-0.002 to -0.000) | +0.80 (+0.59 to +1.02) | 343-350-0, +23.92u | insufficient sample |
| **All game lines** | 56 | +0.014 (-0.023 to +0.058) | -0.001 (-0.005 to +0.003) | +0.55 (+0.08 to +1.02) | 14-22-0, -6.68u | insufficient sample |
| Moneyline | 56 | +0.022 (-0.033 to +0.078) | -0.000 (-0.006 to +0.006) | +0.90 (+0.31 to +1.51) | 6-17-0, -9.30u | insufficient sample |
| Spread | 55 | -0.011 (-0.079 to +0.050) | -0.004 (-0.010 to +0.002) | +0.09 (-0.43 to +0.58) | 8-4-0, +3.62u | insufficient sample |
| Game total | 55 | +0.030 (-0.011 to +0.072) | +0.002 (-0.002 to +0.006) | +0.30 (-0.39 to +1.04) | 0-1-0, -1.00u | insufficient sample |
| Assists | 56 | +0.020 (+0.000 to +0.042) | -0.001 (-0.003 to +0.001) | +1.02 (+0.62 to +1.42) | 57-68-0, -1.13u | insufficient sample |
| Blocks | 43 | +0.011 (-0.028 to +0.058) | -0.000 (-0.004 to +0.004) | +0.36 (-0.35 to +1.18) | 1-2-0, -1.07u | insufficient sample |
| Points | 56 | +0.003 (-0.013 to +0.019) | -0.001 (-0.003 to +0.000) | +0.65 (+0.36 to +0.95) | 57-58-0, +1.32u | insufficient sample |
| Points + assists | 56 | +0.010 (-0.010 to +0.029) | -0.001 (-0.003 to +0.000) | +0.86 (+0.54 to +1.21) | 14-21-0, -6.84u | insufficient sample |
| Points + rebounds | 56 | +0.002 (-0.016 to +0.019) | -0.002 (-0.004 to -0.000) | +0.76 (+0.50 to +1.04) | 28-22-0, +5.92u | insufficient sample |
| Points + rebounds + assists | 56 | +0.012 (-0.009 to +0.033) | -0.002 (-0.004 to +0.000) | +0.83 (+0.50 to +1.19) | 33-41-0, -8.06u | insufficient sample |
| Rebounds | 56 | +0.025 (+0.009 to +0.043) | +0.000 (-0.001 to +0.002) | +0.93 (+0.69 to +1.21) | 63-53-0, +12.88u | insufficient sample |
| Rebounds + assists | 56 | +0.028 (+0.002 to +0.057) | -0.001 (-0.003 to +0.001) | +0.94 (+0.62 to +1.31) | 40-41-0, +0.19u | insufficient sample |
| Steals | 45 | +0.016 (-0.006 to +0.040) | +0.000 (-0.002 to +0.002) | +0.26 (-0.18 to +0.76) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 56 | +0.006 (-0.008 to +0.021) | -0.001 (-0.003 to +0.000) | +0.60 (+0.37 to +0.88) | 46-43-0, +17.70u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 252 bets, 116-136-0, +0.57u (ROI +0.00 (-0.12 to +0.12))
- Props, model: 441 bets, 227-214-0, +23.35u (ROI +0.05 (-0.07 to +0.17))
- Game lines, price: 6 bets, 0-6-0, -6.00u (ROI -1.00)
- Game lines, model: 30 bets, 14-16-0, -0.68u (ROI -0.02 (-0.48 to +0.42))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

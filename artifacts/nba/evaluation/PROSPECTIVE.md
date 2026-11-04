# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Wed, Nov 4, 2026, 5:04 PM EST. 92 of 92 tipped games graded through Nov 1; 8699 main-line and 18548 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 92 | +0.013 (+0.004 to +0.022) | -0.001 (-0.002 to -0.000) | +0.88 (+0.74 to +1.04) | 552-595-0, +9.76u | insufficient sample |
| **All game lines** | 92 | +0.038 (+0.006 to +0.070) | +0.002 (-0.001 to +0.005) | +0.43 (+0.02 to +0.83) | 16-31-0, -13.31u | insufficient sample |
| Moneyline | 92 | +0.065 (+0.019 to +0.110) | +0.004 (-0.000 to +0.009) | +0.77 (+0.09 to +1.39) | 7-25-0, -15.92u | insufficient sample |
| Spread | 91 | +0.027 (-0.023 to +0.076) | +0.000 (-0.005 to +0.005) | +0.01 (-0.41 to +0.39) | 9-5-0, +3.61u | insufficient sample |
| Game total | 91 | +0.024 (-0.006 to +0.061) | +0.001 (-0.002 to +0.004) | +0.15 (-0.25 to +0.63) | 0-1-0, -1.00u | insufficient sample |
| Assists | 92 | +0.022 (+0.005 to +0.039) | -0.001 (-0.002 to +0.001) | +1.04 (+0.76 to +1.33) | 80-103-0, -11.09u | insufficient sample |
| Blocks | 62 | +0.001 (-0.026 to +0.031) | -0.001 (-0.004 to +0.002) | +0.65 (+0.05 to +1.28) | 1-2-0, -1.07u | insufficient sample |
| Points | 92 | +0.004 (-0.008 to +0.015) | -0.001 (-0.002 to -0.000) | +0.90 (+0.68 to +1.17) | 89-95-0, -1.88u | insufficient sample |
| Points + assists | 92 | +0.008 (-0.006 to +0.022) | -0.001 (-0.003 to +0.000) | +0.81 (+0.55 to +1.09) | 22-32-0, -9.57u | insufficient sample |
| Points + rebounds | 92 | +0.004 (-0.008 to +0.018) | -0.001 (-0.003 to -0.000) | +0.90 (+0.70 to +1.11) | 55-48-0, +8.79u | insufficient sample |
| Points + rebounds + assists | 92 | +0.012 (-0.002 to +0.028) | -0.001 (-0.003 to +0.000) | +0.90 (+0.66 to +1.14) | 64-76-0, -11.00u | insufficient sample |
| Rebounds | 92 | +0.022 (+0.010 to +0.035) | -0.000 (-0.001 to +0.001) | +0.98 (+0.76 to +1.22) | 94-97-0, +2.29u | insufficient sample |
| Rebounds + assists | 92 | +0.024 (+0.004 to +0.045) | -0.001 (-0.003 to +0.001) | +0.87 (+0.61 to +1.14) | 66-60-0, +7.89u | insufficient sample |
| Steals | 69 | +0.009 (-0.010 to +0.029) | -0.000 (-0.002 to +0.001) | +0.30 (-0.08 to +0.69) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 92 | +0.011 (-0.000 to +0.021) | -0.001 (-0.002 to +0.000) | +0.72 (+0.55 to +0.93) | 77-80-0, +23.39u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 464 bets, 206-258-0, -12.30u (ROI -0.03 (-0.13 to +0.07))
- Props, model: 683 bets, 346-337-0, +22.06u (ROI +0.03 (-0.06 to +0.13))
- Game lines, price: 8 bets, 0-8-0, -8.00u (ROI -1.00)
- Game lines, model: 39 bets, 16-23-0, -5.31u (ROI -0.14 (-0.50 to +0.23))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

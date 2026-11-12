# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Thu, Nov 12, 2026, 6:49 AM EST. 155 of 155 tipped games graded through Nov 11; 14643 main-line and 30834 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 155 | +0.013 (+0.006 to +0.019) | -0.000 (-0.001 to +0.000) | +0.83 (+0.71 to +0.96) | 823-930-0, -22.29u | not supported |
| **All game lines** | 155 | +0.036 (+0.013 to +0.060) | +0.002 (-0.000 to +0.004) | +0.39 (+0.05 to +0.71) | 20-50-0, -26.17u | not supported |
| Moneyline | 155 | +0.053 (+0.023 to +0.085) | +0.004 (+0.000 to +0.007) | +0.71 (+0.13 to +1.27) | 9-43-0, -29.72u | not supported |
| Spread | 154 | +0.024 (-0.006 to +0.058) | +0.000 (-0.003 to +0.004) | -0.00 (-0.28 to +0.27) | 10-6-0, +3.59u | not supported |
| Game total | 153 | +0.032 (+0.007 to +0.058) | +0.002 (-0.000 to +0.004) | +0.09 (-0.24 to +0.43) | 1-1-0, -0.05u | not supported |
| Assists | 155 | +0.019 (+0.008 to +0.032) | -0.000 (-0.001 to +0.001) | +0.98 (+0.76 to +1.21) | 117-146-0, -11.85u | not supported |
| Blocks | 106 | +0.013 (-0.008 to +0.036) | +0.000 (-0.002 to +0.002) | +0.60 (+0.20 to +1.03) | 1-4-0, -3.07u | insufficient sample |
| Points | 155 | +0.006 (-0.001 to +0.015) | -0.001 (-0.002 to +0.000) | +0.82 (+0.60 to +1.04) | 139-145-0, +2.12u | not supported |
| Points + assists | 155 | +0.013 (+0.003 to +0.022) | -0.000 (-0.001 to +0.000) | +0.87 (+0.68 to +1.08) | 31-45-0, -13.64u | not supported |
| Points + rebounds | 155 | +0.005 (-0.005 to +0.014) | -0.001 (-0.002 to -0.000) | +0.90 (+0.74 to +1.07) | 90-91-0, +3.26u | supported |
| Points + rebounds + assists | 155 | +0.016 (+0.004 to +0.027) | -0.001 (-0.001 to +0.001) | +0.85 (+0.66 to +1.04) | 103-127-0, -20.49u | not supported |
| Rebounds | 155 | +0.018 (+0.008 to +0.027) | +0.000 (-0.001 to +0.001) | +0.96 (+0.79 to +1.15) | 139-144-0, +4.37u | not supported |
| Rebounds + assists | 155 | +0.023 (+0.009 to +0.036) | -0.000 (-0.001 to +0.001) | +0.82 (+0.61 to +1.05) | 98-96-0, +5.17u | not supported |
| Steals | 127 | +0.002 (-0.011 to +0.015) | -0.001 (-0.002 to +0.000) | +0.31 (+0.03 to +0.62) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 155 | +0.010 (+0.002 to +0.018) | -0.000 (-0.001 to +0.000) | +0.55 (+0.41 to +0.70) | 101-130-0, +9.83u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 769 bets, 339-430-0, -29.49u (ROI -0.04 (-0.11 to +0.05))
- Props, model: 984 bets, 484-500-0, +7.20u (ROI +0.01 (-0.07 to +0.08))
- Game lines, price: 16 bets, 0-16-0, -16.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 54 bets, 20-34-0, -10.17u (ROI -0.19 (-0.46 to +0.13))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

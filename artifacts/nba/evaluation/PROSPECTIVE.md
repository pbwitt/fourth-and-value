# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Wed, Nov 25, 2026, 5:05 PM EST. 246 of 246 tipped games graded through Nov 24; 23213 main-line and 48510 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 246 | +0.011 (+0.006 to +0.015) | -0.000 (-0.001 to -0.000) | +0.85 (+0.75 to +0.95) | 1155-1346-0, -63.34u | supported |
| **All game lines** | 246 | +0.032 (+0.015 to +0.048) | +0.002 (+0.000 to +0.003) | +0.36 (+0.06 to +0.64) | 31-69-0, -20.77u | not supported |
| Moneyline | 246 | +0.040 (+0.017 to +0.064) | +0.002 (+0.000 to +0.005) | +0.74 (+0.24 to +1.22) | 18-59-0, -23.23u | not supported |
| Spread | 243 | +0.027 (+0.002 to +0.053) | +0.001 (-0.002 to +0.004) | -0.08 (-0.29 to +0.13) | 12-8-0, +3.52u | not supported |
| Game total | 244 | +0.028 (+0.009 to +0.048) | +0.002 (-0.000 to +0.004) | +0.04 (-0.18 to +0.27) | 1-2-0, -1.05u | not supported |
| Assists | 246 | +0.013 (+0.006 to +0.021) | -0.001 (-0.001 to +0.000) | +1.00 (+0.83 to +1.18) | 165-200-0, -9.04u | not supported |
| Blocks | 161 | +0.006 (-0.010 to +0.024) | -0.001 (-0.002 to +0.001) | +0.73 (+0.32 to +1.16) | 1-4-0, -3.07u | not supported |
| Points | 246 | +0.006 (+0.000 to +0.011) | -0.001 (-0.001 to -0.000) | +0.85 (+0.70 to +1.03) | 189-213-0, -9.56u | supported |
| Points + assists | 246 | +0.010 (+0.002 to +0.017) | -0.000 (-0.001 to +0.000) | +0.87 (+0.73 to +1.02) | 59-71-0, -10.06u | not supported |
| Points + rebounds | 246 | +0.005 (-0.002 to +0.011) | -0.001 (-0.001 to -0.000) | +0.90 (+0.77 to +1.04) | 126-132-0, +1.00u | supported |
| Points + rebounds + assists | 246 | +0.012 (+0.004 to +0.020) | -0.000 (-0.001 to +0.000) | +0.81 (+0.66 to +0.98) | 140-170-0, -25.29u | not supported |
| Rebounds | 246 | +0.013 (+0.007 to +0.020) | -0.000 (-0.001 to +0.001) | +1.02 (+0.87 to +1.16) | 185-201-0, -1.45u | not supported |
| Rebounds + assists | 246 | +0.020 (+0.011 to +0.030) | +0.000 (-0.001 to +0.001) | +0.87 (+0.69 to +1.03) | 139-151-0, -8.24u | not supported |
| Steals | 201 | +0.000 (-0.010 to +0.011) | -0.001 (-0.002 to +0.000) | +0.44 (+0.18 to +0.73) | 5-2-0, +3.43u | not supported |
| Threes made | 246 | +0.013 (+0.007 to +0.018) | +0.000 (-0.001 to +0.001) | +0.52 (+0.41 to +0.64) | 146-202-0, -1.05u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1166 bets, 497-669-0, -81.63u (ROI -0.07 (-0.14 to -0.01))
- Props, model: 1335 bets, 658-677-0, +18.30u (ROI +0.01 (-0.05 to +0.08))
- Game lines, price: 28 bets, 5-23-0, -10.43u (ROI -0.37 (-0.86 to +0.23))
- Game lines, model: 72 bets, 26-46-0, -10.34u (ROI -0.14 (-0.45 to +0.16))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

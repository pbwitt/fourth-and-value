# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Nov 14, 2026, 6:49 AM EST. 168 of 168 tipped games graded through Nov 13; 15967 main-line and 33237 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 168 | +0.012 (+0.006 to +0.018) | -0.000 (-0.001 to +0.000) | +0.84 (+0.72 to +0.96) | 863-984-0, -33.50u | not supported |
| **All game lines** | 168 | +0.038 (+0.016 to +0.059) | +0.002 (-0.000 to +0.004) | +0.38 (+0.06 to +0.71) | 20-53-0, -29.17u | not supported |
| Moneyline | 168 | +0.051 (+0.024 to +0.081) | +0.003 (+0.001 to +0.006) | +0.74 (+0.21 to +1.28) | 9-46-0, -32.72u | not supported |
| Spread | 167 | +0.029 (-0.001 to +0.064) | +0.001 (-0.003 to +0.004) | -0.01 (-0.27 to +0.24) | 10-6-0, +3.59u | not supported |
| Game total | 166 | +0.032 (+0.008 to +0.055) | +0.002 (-0.000 to +0.004) | +0.05 (-0.25 to +0.34) | 1-1-0, -0.05u | not supported |
| Assists | 168 | +0.017 (+0.007 to +0.029) | -0.000 (-0.001 to +0.000) | +0.99 (+0.78 to +1.21) | 125-152-0, -8.56u | not supported |
| Blocks | 115 | +0.012 (-0.007 to +0.033) | -0.000 (-0.002 to +0.002) | +0.62 (+0.26 to +1.02) | 1-4-0, -3.07u | insufficient sample |
| Points | 168 | +0.007 (-0.001 to +0.015) | -0.001 (-0.001 to +0.000) | +0.84 (+0.65 to +1.02) | 146-155-0, -0.22u | not supported |
| Points + assists | 168 | +0.012 (+0.003 to +0.021) | -0.000 (-0.001 to +0.000) | +0.87 (+0.69 to +1.06) | 31-47-0, -15.64u | not supported |
| Points + rebounds | 168 | +0.005 (-0.003 to +0.014) | -0.001 (-0.002 to -0.000) | +0.90 (+0.74 to +1.06) | 93-98-0, -0.69u | supported |
| Points + rebounds + assists | 168 | +0.015 (+0.004 to +0.025) | -0.001 (-0.001 to +0.000) | +0.83 (+0.63 to +1.02) | 108-132-0, -20.36u | not supported |
| Rebounds | 168 | +0.015 (+0.007 to +0.025) | -0.000 (-0.001 to +0.001) | +0.99 (+0.81 to +1.18) | 147-152-0, +4.79u | not supported |
| Rebounds + assists | 168 | +0.021 (+0.008 to +0.035) | -0.000 (-0.001 to +0.001) | +0.83 (+0.61 to +1.05) | 103-100-0, +5.40u | not supported |
| Steals | 140 | -0.000 (-0.014 to +0.014) | -0.001 (-0.002 to +0.000) | +0.33 (+0.06 to +0.60) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 168 | +0.011 (+0.004 to +0.018) | -0.000 (-0.001 to +0.000) | +0.58 (+0.44 to +0.72) | 105-142-0, +2.85u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 824 bets, 361-463-0, -37.33u (ROI -0.05 (-0.12 to +0.04))
- Props, model: 1023 bets, 502-521-0, +3.83u (ROI +0.00 (-0.07 to +0.07))
- Game lines, price: 16 bets, 0-16-0, -16.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 57 bets, 20-37-0, -13.17u (ROI -0.23 (-0.51 to +0.07))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Nov 21, 2026, 6:50 AM EST. 220 of 220 tipped games graded through Nov 20; 20886 main-line and 43521 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 220 | +0.012 (+0.007 to +0.017) | -0.000 (-0.001 to +0.000) | +0.86 (+0.76 to +0.96) | 1064-1233-0, -54.80u | not supported |
| **All game lines** | 220 | +0.034 (+0.016 to +0.053) | +0.002 (+0.000 to +0.004) | +0.40 (+0.13 to +0.68) | 26-63-0, -25.13u | not supported |
| Moneyline | 220 | +0.044 (+0.020 to +0.068) | +0.003 (+0.000 to +0.005) | +0.82 (+0.35 to +1.32) | 13-55-0, -29.60u | not supported |
| Spread | 218 | +0.031 (+0.005 to +0.060) | +0.001 (-0.001 to +0.004) | -0.06 (-0.29 to +0.16) | 12-6-0, +5.51u | not supported |
| Game total | 218 | +0.027 (+0.007 to +0.048) | +0.002 (-0.000 to +0.004) | +0.04 (-0.19 to +0.28) | 1-2-0, -1.05u | not supported |
| Assists | 220 | +0.015 (+0.007 to +0.024) | -0.000 (-0.001 to +0.000) | +1.00 (+0.82 to +1.20) | 150-188-0, -15.79u | not supported |
| Blocks | 146 | +0.007 (-0.011 to +0.027) | -0.000 (-0.002 to +0.001) | +0.76 (+0.34 to +1.20) | 1-4-0, -3.07u | insufficient sample |
| Points | 220 | +0.006 (+0.000 to +0.012) | -0.001 (-0.001 to -0.000) | +0.85 (+0.68 to +1.02) | 173-197-0, -11.88u | supported |
| Points + assists | 220 | +0.011 (+0.003 to +0.019) | -0.000 (-0.001 to +0.000) | +0.84 (+0.68 to +0.99) | 53-60-0, -5.60u | not supported |
| Points + rebounds | 220 | +0.006 (-0.000 to +0.013) | -0.001 (-0.001 to -0.000) | +0.92 (+0.78 to +1.07) | 117-123-0, +0.10u | supported |
| Points + rebounds + assists | 220 | +0.015 (+0.007 to +0.024) | -0.000 (-0.001 to +0.001) | +0.83 (+0.66 to +0.99) | 126-159-0, -28.41u | not supported |
| Rebounds | 220 | +0.015 (+0.008 to +0.022) | -0.000 (-0.001 to +0.001) | +1.04 (+0.88 to +1.21) | 177-183-0, +6.50u | not supported |
| Rebounds + assists | 220 | +0.019 (+0.010 to +0.030) | -0.000 (-0.001 to +0.001) | +0.89 (+0.70 to +1.07) | 131-129-0, +5.82u | not supported |
| Steals | 183 | -0.003 (-0.014 to +0.008) | -0.001 (-0.002 to -0.000) | +0.41 (+0.13 to +0.69) | 5-2-0, +3.43u | supported |
| Threes made | 220 | +0.013 (+0.006 to +0.019) | +0.000 (-0.001 to +0.001) | +0.53 (+0.42 to +0.66) | 131-188-0, -5.89u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1072 bets, 458-614-0, -72.64u (ROI -0.07 (-0.14 to +0.00))
- Props, model: 1225 bets, 606-619-0, +17.83u (ROI +0.01 (-0.06 to +0.09))
- Game lines, price: 23 bets, 2-21-0, -14.09u (ROI -0.61 (-1.00 to -0.03))
- Game lines, model: 66 bets, 24-42-0, -11.04u (ROI -0.17 (-0.44 to +0.14))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Fri, Nov 13, 2026, 5:02 PM EST. 159 of 159 tipped games graded through Nov 12; 15059 main-line and 31333 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 159 | +0.013 (+0.006 to +0.019) | -0.000 (-0.001 to +0.000) | +0.82 (+0.71 to +0.95) | 836-940-0, -17.01u | not supported |
| **All game lines** | 159 | +0.037 (+0.015 to +0.060) | +0.002 (-0.000 to +0.004) | +0.41 (+0.08 to +0.77) | 20-51-0, -27.17u | not supported |
| Moneyline | 159 | +0.053 (+0.023 to +0.082) | +0.004 (+0.000 to +0.007) | +0.77 (+0.21 to +1.35) | 9-44-0, -30.72u | not supported |
| Spread | 158 | +0.026 (-0.008 to +0.060) | +0.001 (-0.003 to +0.004) | -0.00 (-0.25 to +0.25) | 10-6-0, +3.59u | not supported |
| Game total | 157 | +0.032 (+0.007 to +0.056) | +0.002 (-0.000 to +0.004) | +0.05 (-0.25 to +0.41) | 1-1-0, -0.05u | not supported |
| Assists | 159 | +0.018 (+0.007 to +0.030) | -0.000 (-0.001 to +0.000) | +0.97 (+0.75 to +1.21) | 121-145-0, -5.83u | not supported |
| Blocks | 109 | +0.016 (-0.005 to +0.039) | +0.000 (-0.002 to +0.003) | +0.60 (+0.17 to +1.03) | 1-4-0, -3.07u | insufficient sample |
| Points | 159 | +0.007 (-0.001 to +0.015) | -0.001 (-0.001 to +0.000) | +0.82 (+0.64 to +1.02) | 140-149-0, -0.68u | not supported |
| Points + assists | 159 | +0.012 (+0.002 to +0.022) | -0.000 (-0.001 to +0.001) | +0.87 (+0.69 to +1.07) | 31-45-0, -13.64u | not supported |
| Points + rebounds | 159 | +0.005 (-0.004 to +0.014) | -0.001 (-0.002 to -0.000) | +0.87 (+0.72 to +1.03) | 91-92-0, +3.21u | supported |
| Points + rebounds + assists | 159 | +0.015 (+0.005 to +0.027) | -0.001 (-0.002 to +0.001) | +0.81 (+0.63 to +1.01) | 106-129-0, -19.47u | not supported |
| Rebounds | 159 | +0.017 (+0.008 to +0.027) | -0.000 (-0.001 to +0.001) | +0.96 (+0.78 to +1.15) | 141-145-0, +5.56u | not supported |
| Rebounds + assists | 159 | +0.022 (+0.009 to +0.037) | -0.000 (-0.001 to +0.001) | +0.81 (+0.60 to +1.05) | 99-96-0, +6.01u | not supported |
| Steals | 131 | +0.002 (-0.011 to +0.017) | -0.001 (-0.002 to +0.000) | +0.33 (+0.03 to +0.63) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 159 | +0.010 (+0.002 to +0.018) | -0.000 (-0.001 to +0.000) | +0.55 (+0.42 to +0.70) | 102-133-0, +8.88u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 781 bets, 347-434-0, -23.23u (ROI -0.03 (-0.11 to +0.05))
- Props, model: 995 bets, 489-506-0, +6.22u (ROI +0.01 (-0.07 to +0.08))
- Game lines, price: 16 bets, 0-16-0, -16.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 55 bets, 20-35-0, -11.17u (ROI -0.20 (-0.53 to +0.11))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

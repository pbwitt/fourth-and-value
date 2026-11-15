# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sun, Nov 15, 2026, 6:49 AM EST. 173 of 173 tipped games graded through Nov 14; 16369 main-line and 34131 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 173 | +0.012 (+0.006 to +0.019) | -0.000 (-0.001 to +0.000) | +0.85 (+0.73 to +0.96) | 873-1009-0, -48.76u | not supported |
| **All game lines** | 173 | +0.037 (+0.016 to +0.059) | +0.002 (-0.000 to +0.004) | +0.37 (+0.06 to +0.71) | 22-54-0, -27.51u | not supported |
| Moneyline | 173 | +0.052 (+0.022 to +0.079) | +0.003 (+0.001 to +0.006) | +0.72 (+0.16 to +1.27) | 10-47-0, -32.01u | not supported |
| Spread | 172 | +0.028 (-0.002 to +0.061) | +0.001 (-0.002 to +0.004) | -0.01 (-0.29 to +0.24) | 11-6-0, +4.54u | not supported |
| Game total | 171 | +0.030 (+0.006 to +0.056) | +0.002 (-0.000 to +0.004) | +0.02 (-0.25 to +0.32) | 1-1-0, -0.05u | not supported |
| Assists | 173 | +0.018 (+0.008 to +0.029) | -0.000 (-0.001 to +0.001) | +1.00 (+0.77 to +1.23) | 126-156-0, -11.82u | not supported |
| Blocks | 116 | +0.012 (-0.008 to +0.033) | -0.000 (-0.002 to +0.002) | +0.61 (+0.26 to +1.01) | 1-4-0, -3.07u | insufficient sample |
| Points | 173 | +0.007 (-0.001 to +0.014) | -0.001 (-0.001 to +0.000) | +0.83 (+0.64 to +1.02) | 148-160-0, -3.23u | not supported |
| Points + assists | 173 | +0.012 (+0.002 to +0.021) | -0.000 (-0.001 to +0.000) | +0.86 (+0.66 to +1.05) | 32-47-0, -14.69u | not supported |
| Points + rebounds | 173 | +0.005 (-0.003 to +0.013) | -0.001 (-0.002 to -0.000) | +0.88 (+0.73 to +1.03) | 95-102-0, -2.54u | supported |
| Points + rebounds + assists | 173 | +0.015 (+0.005 to +0.026) | -0.000 (-0.001 to +0.000) | +0.83 (+0.64 to +1.03) | 109-136-0, -23.39u | not supported |
| Rebounds | 173 | +0.016 (+0.008 to +0.025) | -0.000 (-0.001 to +0.001) | +1.00 (+0.81 to +1.18) | 148-155-0, +2.76u | not supported |
| Rebounds + assists | 173 | +0.021 (+0.009 to +0.034) | -0.000 (-0.001 to +0.001) | +0.84 (+0.62 to +1.06) | 105-102-0, +5.36u | not supported |
| Steals | 144 | +0.001 (-0.012 to +0.014) | -0.001 (-0.002 to +0.000) | +0.39 (+0.11 to +0.69) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 173 | +0.012 (+0.005 to +0.019) | -0.000 (-0.001 to +0.001) | +0.59 (+0.45 to +0.73) | 105-145-0, -0.15u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 841 bets, 364-477-0, -48.18u (ROI -0.06 (-0.14 to +0.02))
- Props, model: 1041 bets, 509-532-0, -0.58u (ROI -0.00 (-0.08 to +0.08))
- Game lines, price: 16 bets, 0-16-0, -16.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 60 bets, 22-38-0, -11.51u (ROI -0.19 (-0.50 to +0.13))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

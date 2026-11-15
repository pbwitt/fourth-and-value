# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sun, Nov 15, 2026, 5:04 PM EST. 174 of 176 tipped games graded through Nov 15; 16455 main-line and 34326 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 174 | +0.012 (+0.006 to +0.018) | -0.000 (-0.001 to +0.000) | +0.85 (+0.73 to +0.96) | 876-1016-0, -52.36u | not supported |
| **All game lines** | 174 | +0.037 (+0.015 to +0.057) | +0.002 (-0.000 to +0.004) | +0.37 (+0.08 to +0.71) | 22-55-0, -28.51u | not supported |
| Moneyline | 174 | +0.052 (+0.022 to +0.081) | +0.003 (+0.001 to +0.006) | +0.72 (+0.16 to +1.27) | 10-48-0, -33.01u | not supported |
| Spread | 173 | +0.029 (-0.004 to +0.062) | +0.001 (-0.002 to +0.004) | -0.01 (-0.29 to +0.24) | 11-6-0, +4.54u | not supported |
| Game total | 172 | +0.030 (+0.006 to +0.055) | +0.002 (-0.000 to +0.004) | +0.02 (-0.25 to +0.33) | 1-1-0, -0.05u | not supported |
| Assists | 174 | +0.018 (+0.007 to +0.029) | -0.000 (-0.001 to +0.001) | +1.00 (+0.79 to +1.20) | 128-159-0, -12.36u | not supported |
| Blocks | 116 | +0.012 (-0.008 to +0.034) | -0.000 (-0.002 to +0.002) | +0.61 (+0.24 to +1.03) | 1-4-0, -3.07u | insufficient sample |
| Points | 174 | +0.007 (-0.001 to +0.014) | -0.001 (-0.001 to +0.000) | +0.83 (+0.64 to +1.02) | 149-160-0, -2.28u | not supported |
| Points + assists | 174 | +0.012 (+0.003 to +0.021) | -0.000 (-0.001 to +0.000) | +0.86 (+0.66 to +1.03) | 32-47-0, -14.69u | not supported |
| Points + rebounds | 174 | +0.005 (-0.003 to +0.013) | -0.001 (-0.002 to -0.000) | +0.88 (+0.73 to +1.05) | 95-103-0, -3.54u | supported |
| Points + rebounds + assists | 174 | +0.015 (+0.005 to +0.026) | -0.000 (-0.001 to +0.000) | +0.83 (+0.64 to +1.01) | 109-136-0, -23.39u | not supported |
| Rebounds | 174 | +0.016 (+0.007 to +0.025) | -0.000 (-0.001 to +0.001) | +1.00 (+0.81 to +1.19) | 148-155-0, +2.76u | not supported |
| Rebounds + assists | 174 | +0.021 (+0.009 to +0.033) | -0.000 (-0.001 to +0.001) | +0.84 (+0.63 to +1.06) | 105-102-0, +5.36u | not supported |
| Steals | 144 | +0.001 (-0.013 to +0.015) | -0.001 (-0.002 to +0.000) | +0.39 (+0.11 to +0.70) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 174 | +0.012 (+0.005 to +0.019) | -0.000 (-0.001 to +0.001) | +0.59 (+0.44 to +0.73) | 105-148-0, -3.15u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 849 bets, 365-484-0, -53.80u (ROI -0.06 (-0.14 to +0.00))
- Props, model: 1043 bets, 511-532-0, +1.44u (ROI +0.00 (-0.07 to +0.07))
- Game lines, price: 16 bets, 0-16-0, -16.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 61 bets, 22-39-0, -12.51u (ROI -0.21 (-0.50 to +0.12))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

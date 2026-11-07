# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Nov 7, 2026, 5:04 PM EST. 117 of 117 tipped games graded through Nov 6; 11099 main-line and 23670 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 117 | +0.014 (+0.006 to +0.022) | -0.001 (-0.001 to +0.000) | +0.85 (+0.71 to +1.00) | 659-745-0, -23.56u | insufficient sample |
| **All game lines** | 117 | +0.034 (+0.007 to +0.062) | +0.002 (-0.001 to +0.004) | +0.33 (-0.01 to +0.69) | 18-38-0, -17.98u | insufficient sample |
| Moneyline | 117 | +0.053 (+0.013 to +0.092) | +0.003 (-0.001 to +0.007) | +0.57 (+0.03 to +1.14) | 8-32-0, -21.57u | insufficient sample |
| Spread | 116 | +0.019 (-0.023 to +0.058) | -0.000 (-0.005 to +0.003) | -0.00 (-0.37 to +0.34) | 10-5-0, +4.59u | insufficient sample |
| Game total | 116 | +0.030 (-0.001 to +0.060) | +0.002 (-0.001 to +0.005) | +0.17 (-0.18 to +0.60) | 0-1-0, -1.00u | insufficient sample |
| Assists | 117 | +0.023 (+0.010 to +0.036) | -0.000 (-0.001 to +0.001) | +0.99 (+0.73 to +1.26) | 93-125-0, -17.71u | insufficient sample |
| Blocks | 81 | +0.010 (-0.014 to +0.035) | -0.000 (-0.003 to +0.002) | +0.51 (+0.04 to +0.98) | 1-3-0, -2.07u | insufficient sample |
| Points | 117 | +0.007 (-0.003 to +0.016) | -0.001 (-0.002 to +0.000) | +0.82 (+0.61 to +1.04) | 110-118-0, -2.04u | insufficient sample |
| Points + assists | 117 | +0.011 (-0.001 to +0.022) | -0.001 (-0.002 to +0.000) | +0.79 (+0.58 to +1.00) | 29-38-0, -8.62u | insufficient sample |
| Points + rebounds | 117 | +0.005 (-0.006 to +0.016) | -0.001 (-0.002 to -0.000) | +0.88 (+0.71 to +1.05) | 71-69-0, +4.54u | insufficient sample |
| Points + rebounds + assists | 117 | +0.016 (+0.003 to +0.029) | -0.001 (-0.002 to +0.001) | +0.86 (+0.65 to +1.09) | 77-104-0, -25.27u | insufficient sample |
| Rebounds | 117 | +0.021 (+0.011 to +0.031) | +0.000 (-0.001 to +0.001) | +1.00 (+0.80 to +1.21) | 109-116-0, -0.93u | insufficient sample |
| Rebounds + assists | 117 | +0.026 (+0.009 to +0.043) | -0.000 (-0.002 to +0.001) | +0.92 (+0.67 to +1.16) | 79-73-0, +8.00u | insufficient sample |
| Steals | 92 | +0.010 (-0.006 to +0.028) | -0.000 (-0.002 to +0.001) | +0.30 (-0.02 to +0.66) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 117 | +0.010 (+0.000 to +0.019) | -0.001 (-0.001 to +0.000) | +0.62 (+0.46 to +0.78) | 86-97-0, +18.53u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 599 bets, 261-338-0, -32.06u (ROI -0.05 (-0.15 to +0.04))
- Props, model: 805 bets, 398-407-0, +8.50u (ROI +0.01 (-0.07 to +0.09))
- Game lines, price: 11 bets, 0-11-0, -11.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 45 bets, 18-27-0, -6.98u (ROI -0.16 (-0.48 to +0.18))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Wed, Nov 11, 2026, 5:03 PM EST. 145 of 145 tipped games graded through Nov 10; 13727 main-line and 28960 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 145 | +0.014 (+0.007 to +0.021) | -0.000 (-0.001 to +0.000) | +0.83 (+0.71 to +0.96) | 772-883-0, -38.38u | insufficient sample |
| **All game lines** | 145 | +0.036 (+0.013 to +0.059) | +0.002 (-0.000 to +0.004) | +0.37 (+0.05 to +0.72) | 19-47-0, -24.13u | insufficient sample |
| Moneyline | 145 | +0.053 (+0.019 to +0.085) | +0.003 (+0.000 to +0.006) | +0.66 (+0.08 to +1.26) | 9-40-0, -26.72u | insufficient sample |
| Spread | 144 | +0.020 (-0.016 to +0.058) | -0.000 (-0.004 to +0.004) | +0.01 (-0.27 to +0.29) | 10-6-0, +3.59u | insufficient sample |
| Game total | 143 | +0.035 (+0.010 to +0.062) | +0.002 (-0.000 to +0.005) | +0.12 (-0.22 to +0.52) | 0-1-0, -1.00u | insufficient sample |
| Assists | 145 | +0.020 (+0.008 to +0.032) | -0.000 (-0.001 to +0.001) | +0.97 (+0.73 to +1.20) | 111-143-0, -17.60u | insufficient sample |
| Blocks | 101 | +0.013 (-0.009 to +0.037) | -0.000 (-0.002 to +0.002) | +0.58 (+0.20 to +0.99) | 1-4-0, -3.07u | insufficient sample |
| Points | 145 | +0.007 (-0.001 to +0.015) | -0.001 (-0.001 to +0.000) | +0.81 (+0.58 to +1.02) | 132-133-0, +6.25u | insufficient sample |
| Points + assists | 145 | +0.013 (+0.003 to +0.024) | -0.000 (-0.001 to +0.001) | +0.88 (+0.69 to +1.08) | 31-44-0, -12.64u | insufficient sample |
| Points + rebounds | 145 | +0.005 (-0.004 to +0.014) | -0.001 (-0.002 to -0.000) | +0.91 (+0.73 to +1.07) | 82-85-0, +0.71u | insufficient sample |
| Points + rebounds + assists | 145 | +0.016 (+0.005 to +0.028) | -0.001 (-0.002 to +0.001) | +0.83 (+0.63 to +1.01) | 94-120-0, -23.45u | insufficient sample |
| Rebounds | 145 | +0.019 (+0.010 to +0.029) | +0.000 (-0.001 to +0.001) | +0.96 (+0.78 to +1.15) | 129-139-0, -2.41u | insufficient sample |
| Rebounds + assists | 145 | +0.025 (+0.010 to +0.040) | -0.000 (-0.001 to +0.001) | +0.85 (+0.62 to +1.08) | 94-91-0, +5.99u | insufficient sample |
| Steals | 118 | +0.004 (-0.009 to +0.017) | -0.001 (-0.002 to +0.001) | +0.33 (+0.04 to +0.65) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 145 | +0.011 (+0.003 to +0.019) | -0.000 (-0.001 to +0.000) | +0.57 (+0.42 to +0.72) | 94-122-0, +5.83u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 713 bets, 310-403-0, -41.28u (ROI -0.06 (-0.14 to +0.02))
- Props, model: 942 bets, 462-480-0, +2.90u (ROI +0.00 (-0.08 to +0.08))
- Game lines, price: 15 bets, 0-15-0, -15.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 51 bets, 19-32-0, -9.13u (ROI -0.18 (-0.51 to +0.16))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

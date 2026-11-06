# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Fri, Nov 6, 2026, 6:49 AM EST. 108 of 108 tipped games graded through Nov 5; 10246 main-line and 21831 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 108 | +0.013 (+0.005 to +0.020) | -0.001 (-0.001 to -0.000) | +0.88 (+0.74 to +1.03) | 610-677-0, -7.00u | insufficient sample |
| **All game lines** | 108 | +0.036 (+0.007 to +0.067) | +0.002 (-0.001 to +0.005) | +0.39 (+0.02 to +0.76) | 17-36-0, -17.33u | insufficient sample |
| Moneyline | 108 | +0.058 (+0.017 to +0.100) | +0.004 (+0.000 to +0.007) | +0.71 (+0.14 to +1.30) | 7-30-0, -20.92u | insufficient sample |
| Spread | 107 | +0.022 (-0.022 to +0.066) | -0.000 (-0.004 to +0.004) | -0.00 (-0.36 to +0.34) | 10-5-0, +4.59u | insufficient sample |
| Game total | 107 | +0.030 (-0.002 to +0.060) | +0.002 (-0.001 to +0.005) | +0.17 (-0.19 to +0.58) | 0-1-0, -1.00u | insufficient sample |
| Assists | 108 | +0.024 (+0.010 to +0.037) | -0.000 (-0.002 to +0.001) | +1.04 (+0.77 to +1.29) | 85-114-0, -15.59u | insufficient sample |
| Blocks | 75 | +0.009 (-0.016 to +0.036) | -0.000 (-0.003 to +0.002) | +0.57 (+0.10 to +1.05) | 1-2-0, -1.07u | insufficient sample |
| Points | 108 | +0.005 (-0.005 to +0.015) | -0.001 (-0.002 to -0.000) | +0.87 (+0.67 to +1.10) | 104-110-0, -0.32u | insufficient sample |
| Points + assists | 108 | +0.008 (-0.006 to +0.021) | -0.001 (-0.002 to +0.000) | +0.80 (+0.57 to +1.00) | 27-33-0, -5.52u | insufficient sample |
| Points + rebounds | 108 | +0.004 (-0.007 to +0.016) | -0.001 (-0.002 to -0.000) | +0.92 (+0.74 to +1.11) | 63-60-0, +4.90u | insufficient sample |
| Points + rebounds + assists | 108 | +0.013 (-0.000 to +0.026) | -0.001 (-0.002 to +0.000) | +0.91 (+0.71 to +1.17) | 71-90-0, -17.79u | insufficient sample |
| Rebounds | 108 | +0.021 (+0.010 to +0.032) | +0.000 (-0.001 to +0.001) | +1.01 (+0.81 to +1.23) | 101-107-0, +0.09u | insufficient sample |
| Rebounds + assists | 108 | +0.025 (+0.007 to +0.043) | -0.000 (-0.002 to +0.001) | +0.93 (+0.68 to +1.18) | 70-66-0, +5.77u | insufficient sample |
| Steals | 85 | +0.011 (-0.005 to +0.029) | -0.000 (-0.002 to +0.002) | +0.36 (-0.00 to +0.72) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 108 | +0.010 (-0.001 to +0.019) | -0.001 (-0.001 to +0.000) | +0.64 (+0.47 to +0.82) | 84-93-0, +20.51u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 533 bets, 239-294-0, -10.62u (ROI -0.02 (-0.11 to +0.07))
- Props, model: 754 bets, 371-383-0, +3.62u (ROI +0.00 (-0.08 to +0.09))
- Game lines, price: 10 bets, 0-10-0, -10.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 43 bets, 17-26-0, -7.33u (ROI -0.17 (-0.51 to +0.18))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

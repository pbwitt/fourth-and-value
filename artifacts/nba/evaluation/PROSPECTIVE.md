# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Thu, Nov 5, 2026, 6:48 AM EST. 106 of 106 tipped games graded through Nov 4; 10031 main-line and 21372 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 106 | +0.012 (+0.004 to +0.020) | -0.001 (-0.001 to -0.000) | +0.86 (+0.73 to +1.00) | 603-673-0, -10.46u | insufficient sample |
| **All game lines** | 106 | +0.036 (+0.010 to +0.066) | +0.002 (-0.001 to +0.005) | +0.40 (+0.01 to +0.80) | 17-36-0, -17.33u | insufficient sample |
| Moneyline | 106 | +0.060 (+0.018 to +0.100) | +0.004 (-0.000 to +0.008) | +0.72 (+0.12 to +1.37) | 7-30-0, -20.92u | insufficient sample |
| Spread | 105 | +0.023 (-0.019 to +0.070) | -0.000 (-0.005 to +0.004) | -0.00 (-0.41 to +0.35) | 10-5-0, +4.59u | insufficient sample |
| Game total | 105 | +0.027 (-0.003 to +0.057) | +0.001 (-0.002 to +0.004) | +0.14 (-0.24 to +0.55) | 0-1-0, -1.00u | insufficient sample |
| Assists | 106 | +0.022 (+0.007 to +0.036) | -0.000 (-0.002 to +0.001) | +1.00 (+0.72 to +1.26) | 84-113-0, -15.85u | insufficient sample |
| Blocks | 74 | +0.005 (-0.020 to +0.033) | -0.001 (-0.003 to +0.002) | +0.57 (+0.07 to +1.09) | 1-2-0, -1.07u | insufficient sample |
| Points | 106 | +0.005 (-0.005 to +0.014) | -0.001 (-0.002 to -0.000) | +0.86 (+0.65 to +1.08) | 102-110-0, -2.37u | insufficient sample |
| Points + assists | 106 | +0.007 (-0.005 to +0.021) | -0.001 (-0.002 to +0.000) | +0.80 (+0.58 to +1.04) | 27-33-0, -5.52u | insufficient sample |
| Points + rebounds | 106 | +0.005 (-0.007 to +0.016) | -0.001 (-0.002 to -0.000) | +0.91 (+0.74 to +1.11) | 62-60-0, +3.85u | insufficient sample |
| Points + rebounds + assists | 106 | +0.012 (-0.001 to +0.026) | -0.001 (-0.002 to +0.000) | +0.91 (+0.69 to +1.15) | 70-89-0, -17.84u | insufficient sample |
| Rebounds | 106 | +0.021 (+0.010 to +0.032) | +0.000 (-0.001 to +0.001) | +0.98 (+0.77 to +1.18) | 100-107-0, -0.96u | insufficient sample |
| Rebounds + assists | 106 | +0.024 (+0.007 to +0.043) | -0.001 (-0.002 to +0.001) | +0.88 (+0.67 to +1.11) | 69-65-0, +5.77u | insufficient sample |
| Steals | 83 | +0.009 (-0.008 to +0.026) | -0.000 (-0.002 to +0.001) | +0.36 (-0.00 to +0.76) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 106 | +0.009 (-0.000 to +0.019) | -0.001 (-0.002 to +0.000) | +0.66 (+0.49 to +0.83) | 84-92-0, +21.51u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 525 bets, 233-292-0, -14.82u (ROI -0.03 (-0.12 to +0.07))
- Props, model: 751 bets, 370-381-0, +4.36u (ROI +0.01 (-0.08 to +0.09))
- Game lines, price: 10 bets, 0-10-0, -10.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 43 bets, 17-26-0, -7.33u (ROI -0.17 (-0.51 to +0.16))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

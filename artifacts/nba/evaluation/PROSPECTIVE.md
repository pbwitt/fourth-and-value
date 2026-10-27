# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Tue, Oct 27, 2026, 6:47 AM EDT. 52 of 52 tipped games graded through Oct 26; 4962 main-line and 10596 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 52 | +0.013 (-0.001 to +0.027) | -0.001 (-0.002 to -0.000) | +0.69 (+0.53 to +0.88) | 330-331-0, +29.07u | insufficient sample |
| **All game lines** | 52 | +0.006 (-0.039 to +0.048) | -0.001 (-0.006 to +0.003) | +0.57 (+0.08 to +1.08) | 14-18-0, -2.68u | insufficient sample |
| Moneyline | 52 | +0.017 (-0.047 to +0.077) | -0.001 (-0.007 to +0.006) | +0.90 (+0.28 to +1.58) | 6-14-0, -6.30u | insufficient sample |
| Spread | 51 | -0.028 (-0.091 to +0.038) | -0.005 (-0.012 to +0.001) | +0.17 (-0.42 to +0.62) | 8-3-0, +4.62u | insufficient sample |
| Game total | 51 | +0.029 (-0.017 to +0.073) | +0.002 (-0.003 to +0.006) | +0.30 (-0.35 to +0.96) | 0-1-0, -1.00u | insufficient sample |
| Assists | 52 | +0.018 (-0.004 to +0.038) | -0.001 (-0.003 to +0.001) | +0.90 (+0.56 to +1.24) | 55-65-0, +0.01u | insufficient sample |
| Blocks | 40 | +0.008 (-0.032 to +0.054) | -0.001 (-0.004 to +0.004) | +0.22 (-0.42 to +0.98) | 1-2-0, -1.07u | insufficient sample |
| Points | 52 | +0.002 (-0.014 to +0.017) | -0.002 (-0.003 to -0.000) | +0.54 (+0.26 to +0.80) | 55-55-0, +2.12u | insufficient sample |
| Points + assists | 52 | +0.011 (-0.009 to +0.032) | -0.001 (-0.003 to +0.001) | +0.69 (+0.42 to +0.96) | 14-21-0, -6.84u | insufficient sample |
| Points + rebounds | 52 | +0.002 (-0.017 to +0.021) | -0.002 (-0.004 to -0.000) | +0.65 (+0.41 to +0.89) | 26-21-0, +4.74u | insufficient sample |
| Points + rebounds + assists | 52 | +0.012 (-0.010 to +0.034) | -0.002 (-0.004 to +0.000) | +0.69 (+0.42 to +0.98) | 32-39-0, -7.11u | insufficient sample |
| Rebounds | 52 | +0.026 (+0.008 to +0.045) | +0.000 (-0.001 to +0.002) | +0.84 (+0.59 to +1.09) | 63-50-0, +15.88u | insufficient sample |
| Rebounds + assists | 52 | +0.029 (-0.002 to +0.060) | -0.001 (-0.003 to +0.002) | +0.85 (+0.51 to +1.20) | 37-35-0, +3.26u | insufficient sample |
| Steals | 41 | +0.019 (-0.005 to +0.044) | +0.000 (-0.002 to +0.003) | +0.22 (-0.22 to +0.70) | 4-1-0, +3.02u | insufficient sample |
| Threes made | 52 | +0.006 (-0.010 to +0.022) | -0.001 (-0.003 to +0.000) | +0.55 (+0.32 to +0.78) | 43-42-0, +15.07u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 242 bets, 113-129-0, +4.59u (ROI +0.02 (-0.10 to +0.13))
- Props, model: 419 bets, 217-202-0, +24.48u (ROI +0.06 (-0.07 to +0.19))
- Game lines, price: 4 bets, 0-4-0, -4.00u (ROI -1.00)
- Game lines, model: 28 bets, 14-14-0, +1.32u (ROI +0.05 (-0.39 to +0.51))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

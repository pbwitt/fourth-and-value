# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sat, Nov 28, 2026, 6:50 AM EST. 266 of 266 tipped games graded through Nov 27; 25220 main-line and 52510 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 266 | +0.011 (+0.006 to +0.015) | -0.000 (-0.001 to -0.000) | +0.87 (+0.78 to +0.96) | 1237-1457-0, -82.15u | not supported |
| **All game lines** | 266 | +0.030 (+0.014 to +0.045) | +0.002 (+0.000 to +0.003) | +0.36 (+0.12 to +0.63) | 35-72-0, -16.82u | not supported |
| Moneyline | 266 | +0.037 (+0.015 to +0.060) | +0.002 (+0.000 to +0.004) | +0.73 (+0.29 to +1.17) | 21-62-0, -20.28u | not supported |
| Spread | 263 | +0.025 (+0.001 to +0.051) | +0.001 (-0.001 to +0.003) | -0.11 (-0.33 to +0.11) | 13-8-0, +4.50u | not supported |
| Game total | 264 | +0.028 (+0.011 to +0.048) | +0.002 (+0.000 to +0.004) | +0.07 (-0.13 to +0.28) | 1-2-0, -1.05u | not supported |
| Assists | 266 | +0.014 (+0.006 to +0.022) | -0.000 (-0.001 to +0.000) | +1.03 (+0.86 to +1.22) | 178-213-0, -6.17u | not supported |
| Blocks | 170 | +0.005 (-0.011 to +0.023) | -0.001 (-0.002 to +0.001) | +0.72 (+0.36 to +1.10) | 1-4-0, -3.07u | not supported |
| Points | 266 | +0.006 (+0.001 to +0.011) | -0.001 (-0.001 to -0.000) | +0.87 (+0.71 to +1.03) | 205-226-0, -5.06u | supported |
| Points + assists | 266 | +0.010 (+0.003 to +0.016) | -0.000 (-0.001 to +0.000) | +0.88 (+0.75 to +1.02) | 66-77-0, -8.98u | not supported |
| Points + rebounds | 266 | +0.005 (-0.001 to +0.011) | -0.001 (-0.001 to -0.000) | +0.93 (+0.79 to +1.06) | 134-149-0, -7.69u | supported |
| Points + rebounds + assists | 266 | +0.013 (+0.006 to +0.020) | -0.000 (-0.001 to +0.000) | +0.85 (+0.68 to +1.00) | 145-178-0, -27.84u | not supported |
| Rebounds | 266 | +0.013 (+0.007 to +0.019) | -0.000 (-0.001 to +0.000) | +1.01 (+0.87 to +1.16) | 197-226-0, -12.97u | not supported |
| Rebounds + assists | 266 | +0.020 (+0.010 to +0.029) | +0.000 (-0.001 to +0.001) | +0.94 (+0.78 to +1.10) | 149-166-0, -12.83u | not supported |
| Steals | 214 | +0.001 (-0.010 to +0.012) | -0.001 (-0.002 to +0.000) | +0.43 (+0.17 to +0.69) | 5-2-0, +3.43u | not supported |
| Threes made | 266 | +0.012 (+0.005 to +0.017) | +0.000 (-0.001 to +0.001) | +0.51 (+0.39 to +0.62) | 157-216-0, -0.96u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1270 bets, 541-729-0, -91.56u (ROI -0.07 (-0.14 to -0.01))
- Props, model: 1424 bets, 696-728-0, +9.42u (ROI +0.01 (-0.06 to +0.07))
- Game lines, price: 30 bets, 6-24-0, -8.12u (ROI -0.27 (-0.80 to +0.34))
- Game lines, model: 77 bets, 29-48-0, -8.70u (ROI -0.11 (-0.40 to +0.19))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

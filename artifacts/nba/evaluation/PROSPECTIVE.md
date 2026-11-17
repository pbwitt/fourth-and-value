# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Tue, Nov 17, 2026, 5:04 PM EST. 190 of 190 tipped games graded through Nov 16; 17942 main-line and 37107 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 190 | +0.012 (+0.007 to +0.018) | -0.000 (-0.001 to +0.000) | +0.86 (+0.76 to +0.97) | 929-1075-0, -50.58u | not supported |
| **All game lines** | 190 | +0.033 (+0.013 to +0.053) | +0.002 (-0.000 to +0.004) | +0.35 (+0.07 to +0.67) | 23-59-0, -31.54u | not supported |
| Moneyline | 190 | +0.046 (+0.020 to +0.074) | +0.003 (+0.000 to +0.006) | +0.71 (+0.22 to +1.21) | 10-51-0, -36.01u | not supported |
| Spread | 188 | +0.026 (-0.004 to +0.056) | +0.001 (-0.002 to +0.004) | -0.05 (-0.30 to +0.18) | 12-6-0, +5.51u | not supported |
| Game total | 188 | +0.026 (+0.004 to +0.048) | +0.001 (-0.001 to +0.004) | +0.03 (-0.23 to +0.30) | 1-2-0, -1.05u | not supported |
| Assists | 190 | +0.018 (+0.008 to +0.027) | -0.000 (-0.001 to +0.001) | +1.01 (+0.82 to +1.24) | 133-168-0, -15.96u | not supported |
| Blocks | 124 | +0.009 (-0.010 to +0.028) | -0.000 (-0.002 to +0.002) | +0.67 (+0.28 to +1.15) | 1-4-0, -3.07u | insufficient sample |
| Points | 190 | +0.007 (+0.000 to +0.014) | -0.001 (-0.001 to +0.000) | +0.88 (+0.69 to +1.07) | 152-169-0, -7.98u | not supported |
| Points + assists | 190 | +0.012 (+0.004 to +0.021) | -0.000 (-0.001 to +0.000) | +0.86 (+0.70 to +1.03) | 35-51-0, -15.59u | not supported |
| Points + rebounds | 190 | +0.005 (-0.003 to +0.013) | -0.001 (-0.002 to -0.000) | +0.91 (+0.76 to +1.07) | 103-108-0, -0.01u | supported |
| Points + rebounds + assists | 190 | +0.015 (+0.005 to +0.025) | -0.000 (-0.001 to +0.001) | +0.84 (+0.65 to +1.04) | 110-141-0, -27.34u | not supported |
| Rebounds | 190 | +0.015 (+0.007 to +0.023) | -0.000 (-0.001 to +0.001) | +1.00 (+0.84 to +1.17) | 163-162-0, +12.36u | not supported |
| Rebounds + assists | 190 | +0.020 (+0.009 to +0.032) | -0.000 (-0.001 to +0.001) | +0.87 (+0.66 to +1.05) | 113-108-0, +7.93u | not supported |
| Steals | 155 | +0.000 (-0.012 to +0.013) | -0.001 (-0.002 to +0.000) | +0.38 (+0.12 to +0.68) | 4-2-0, +2.02u | not supported |
| Threes made | 190 | +0.013 (+0.005 to +0.020) | -0.000 (-0.001 to +0.001) | +0.56 (+0.42 to +0.70) | 115-162-0, -2.92u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 925 bets, 397-528-0, -59.97u (ROI -0.06 (-0.13 to +0.01))
- Props, model: 1079 bets, 532-547-0, +9.39u (ROI +0.01 (-0.07 to +0.08))
- Game lines, price: 19 bets, 0-19-0, -19.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 63 bets, 23-40-0, -12.54u (ROI -0.20 (-0.48 to +0.13))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

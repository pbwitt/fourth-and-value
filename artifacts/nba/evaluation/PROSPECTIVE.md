# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Mon, Nov 16, 2026, 5:05 PM EST. 182 of 182 tipped games graded through Nov 15; 17130 main-line and 35709 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 182 | +0.012 (+0.007 to +0.018) | -0.000 (-0.001 to +0.000) | +0.86 (+0.75 to +0.99) | 903-1046-0, -51.01u | not supported |
| **All game lines** | 182 | +0.034 (+0.013 to +0.056) | +0.002 (-0.000 to +0.004) | +0.37 (+0.07 to +0.68) | 22-58-0, -31.51u | not supported |
| Moneyline | 182 | +0.048 (+0.020 to +0.077) | +0.003 (+0.000 to +0.006) | +0.73 (+0.22 to +1.25) | 10-51-0, -36.01u | not supported |
| Spread | 180 | +0.027 (-0.004 to +0.057) | +0.001 (-0.002 to +0.004) | -0.03 (-0.30 to +0.22) | 11-6-0, +4.54u | not supported |
| Game total | 180 | +0.026 (+0.005 to +0.049) | +0.001 (-0.001 to +0.004) | +0.02 (-0.25 to +0.31) | 1-1-0, -0.05u | not supported |
| Assists | 182 | +0.017 (+0.007 to +0.027) | -0.000 (-0.001 to +0.000) | +1.03 (+0.83 to +1.24) | 130-161-0, -12.51u | not supported |
| Blocks | 119 | +0.013 (-0.007 to +0.034) | +0.000 (-0.002 to +0.002) | +0.63 (+0.24 to +1.01) | 1-4-0, -3.07u | insufficient sample |
| Points | 182 | +0.007 (-0.000 to +0.014) | -0.001 (-0.001 to +0.000) | +0.88 (+0.69 to +1.06) | 149-166-0, -8.28u | not supported |
| Points + assists | 182 | +0.012 (+0.004 to +0.021) | -0.000 (-0.001 to +0.001) | +0.87 (+0.69 to +1.07) | 33-50-0, -16.62u | not supported |
| Points + rebounds | 182 | +0.004 (-0.004 to +0.012) | -0.001 (-0.002 to -0.000) | +0.91 (+0.75 to +1.06) | 100-107-0, -2.32u | supported |
| Points + rebounds + assists | 182 | +0.015 (+0.005 to +0.025) | -0.000 (-0.001 to +0.000) | +0.85 (+0.67 to +1.04) | 110-138-0, -24.34u | not supported |
| Rebounds | 182 | +0.015 (+0.006 to +0.023) | -0.000 (-0.001 to +0.001) | +1.00 (+0.83 to +1.18) | 157-157-0, +11.20u | not supported |
| Rebounds + assists | 182 | +0.020 (+0.007 to +0.033) | -0.000 (-0.001 to +0.001) | +0.85 (+0.65 to +1.04) | 110-105-0, +7.71u | not supported |
| Steals | 147 | +0.002 (-0.012 to +0.015) | -0.001 (-0.002 to +0.000) | +0.40 (+0.13 to +0.73) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 182 | +0.013 (+0.006 to +0.021) | +0.000 (-0.001 to +0.001) | +0.57 (+0.43 to +0.70) | 109-156-0, -4.78u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 892 bets, 382-510-0, -59.70u (ROI -0.07 (-0.15 to +0.01))
- Props, model: 1057 bets, 521-536-0, +8.69u (ROI +0.01 (-0.07 to +0.08))
- Game lines, price: 19 bets, 0-19-0, -19.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 61 bets, 22-39-0, -12.51u (ROI -0.21 (-0.48 to +0.11))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

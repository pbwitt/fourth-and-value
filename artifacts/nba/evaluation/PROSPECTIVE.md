# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Fri, Nov 20, 2026, 5:05 PM EST. 211 of 211 tipped games graded through Nov 19; 19988 main-line and 41556 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 211 | +0.011 (+0.007 to +0.017) | -0.000 (-0.001 to +0.000) | +0.85 (+0.74 to +0.95) | 1034-1194-0, -50.88u | not supported |
| **All game lines** | 211 | +0.034 (+0.015 to +0.053) | +0.002 (+0.000 to +0.004) | +0.41 (+0.12 to +0.70) | 25-62-0, -26.48u | not supported |
| Moneyline | 211 | +0.044 (+0.017 to +0.070) | +0.003 (+0.000 to +0.005) | +0.82 (+0.34 to +1.31) | 12-54-0, -30.95u | not supported |
| Spread | 209 | +0.030 (+0.005 to +0.059) | +0.001 (-0.001 to +0.004) | -0.06 (-0.30 to +0.16) | 12-6-0, +5.51u | not supported |
| Game total | 209 | +0.027 (+0.007 to +0.049) | +0.002 (-0.001 to +0.004) | +0.02 (-0.22 to +0.27) | 1-2-0, -1.05u | not supported |
| Assists | 211 | +0.016 (+0.007 to +0.026) | -0.000 (-0.001 to +0.000) | +1.01 (+0.81 to +1.21) | 144-185-0, -20.72u | not supported |
| Blocks | 139 | +0.006 (-0.013 to +0.026) | -0.001 (-0.002 to +0.001) | +0.76 (+0.36 to +1.18) | 1-4-0, -3.07u | insufficient sample |
| Points | 211 | +0.006 (-0.000 to +0.013) | -0.001 (-0.001 to -0.000) | +0.84 (+0.66 to +1.02) | 167-187-0, -8.88u | not supported |
| Points + assists | 211 | +0.011 (+0.003 to +0.019) | -0.000 (-0.001 to +0.000) | +0.83 (+0.67 to +1.00) | 50-56-0, -5.25u | not supported |
| Points + rebounds | 211 | +0.006 (-0.001 to +0.013) | -0.001 (-0.002 to -0.000) | +0.89 (+0.74 to +1.04) | 115-121-0, -0.05u | supported |
| Points + rebounds + assists | 211 | +0.014 (+0.005 to +0.023) | -0.000 (-0.001 to +0.000) | +0.82 (+0.65 to +0.97) | 124-155-0, -26.66u | not supported |
| Rebounds | 211 | +0.015 (+0.008 to +0.023) | -0.000 (-0.001 to +0.001) | +1.01 (+0.84 to +1.18) | 175-177-0, +10.17u | not supported |
| Rebounds + assists | 211 | +0.018 (+0.008 to +0.028) | -0.000 (-0.001 to +0.001) | +0.86 (+0.66 to +1.04) | 126-123-0, +6.85u | not supported |
| Steals | 174 | -0.003 (-0.014 to +0.009) | -0.001 (-0.002 to -0.000) | +0.45 (+0.19 to +0.74) | 5-2-0, +3.43u | supported |
| Threes made | 211 | +0.013 (+0.006 to +0.020) | +0.000 (-0.001 to +0.001) | +0.54 (+0.41 to +0.67) | 127-184-0, -6.68u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1028 bets, 441-587-0, -65.51u (ROI -0.06 (-0.13 to +0.01))
- Props, model: 1200 bets, 593-607-0, +14.64u (ROI +0.01 (-0.06 to +0.08))
- Game lines, price: 21 bets, 1-20-0, -15.44u (ROI -0.74 (-1.00 to -0.21))
- Game lines, model: 66 bets, 24-42-0, -11.04u (ROI -0.17 (-0.44 to +0.15))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

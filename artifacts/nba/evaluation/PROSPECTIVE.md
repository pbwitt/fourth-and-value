# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Mon, Nov 23, 2026, 6:49 AM EST. 233 of 233 tipped games graded through Nov 22; 22032 main-line and 45902 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 233 | +0.011 (+0.006 to +0.016) | -0.000 (-0.001 to -0.000) | +0.84 (+0.74 to +0.93) | 1126-1291-0, -43.44u | supported |
| **All game lines** | 233 | +0.036 (+0.017 to +0.053) | +0.002 (+0.000 to +0.004) | +0.37 (+0.08 to +0.66) | 28-66-0, -24.91u | not supported |
| Moneyline | 233 | +0.046 (+0.023 to +0.070) | +0.003 (+0.000 to +0.005) | +0.74 (+0.25 to +1.27) | 15-56-0, -27.38u | not supported |
| Spread | 230 | +0.033 (+0.008 to +0.060) | +0.002 (-0.001 to +0.004) | -0.07 (-0.30 to +0.16) | 12-8-0, +3.52u | not supported |
| Game total | 231 | +0.027 (+0.008 to +0.049) | +0.002 (-0.000 to +0.004) | +0.02 (-0.19 to +0.26) | 1-2-0, -1.05u | not supported |
| Assists | 233 | +0.015 (+0.006 to +0.023) | -0.000 (-0.001 to +0.000) | +0.99 (+0.82 to +1.18) | 160-194-0, -9.23u | not supported |
| Blocks | 154 | +0.006 (-0.011 to +0.025) | -0.001 (-0.002 to +0.001) | +0.76 (+0.38 to +1.20) | 1-4-0, -3.07u | not supported |
| Points | 233 | +0.006 (-0.001 to +0.011) | -0.001 (-0.001 to -0.000) | +0.82 (+0.66 to +0.98) | 184-205-0, -7.46u | supported |
| Points + assists | 233 | +0.009 (+0.002 to +0.017) | -0.001 (-0.001 to +0.000) | +0.85 (+0.71 to +0.99) | 59-66-0, -5.06u | not supported |
| Points + rebounds | 233 | +0.005 (-0.002 to +0.011) | -0.001 (-0.002 to -0.000) | +0.91 (+0.77 to +1.04) | 125-128-0, +3.95u | supported |
| Points + rebounds + assists | 233 | +0.013 (+0.005 to +0.022) | -0.001 (-0.001 to +0.000) | +0.79 (+0.63 to +0.95) | 137-163-0, -21.38u | not supported |
| Rebounds | 233 | +0.014 (+0.007 to +0.021) | -0.000 (-0.001 to +0.001) | +1.03 (+0.89 to +1.17) | 181-192-0, +2.40u | not supported |
| Rebounds + assists | 233 | +0.020 (+0.011 to +0.030) | -0.000 (-0.001 to +0.001) | +0.84 (+0.67 to +1.02) | 137-142-0, -1.22u | not supported |
| Steals | 194 | -0.001 (-0.013 to +0.010) | -0.001 (-0.002 to -0.000) | +0.41 (+0.15 to +0.70) | 5-2-0, +3.43u | supported |
| Threes made | 233 | +0.012 (+0.006 to +0.018) | +0.000 (-0.001 to +0.001) | +0.51 (+0.38 to +0.62) | 137-195-0, -5.79u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1127 bets, 486-641-0, -67.75u (ROI -0.06 (-0.13 to +0.01))
- Props, model: 1290 bets, 640-650-0, +24.30u (ROI +0.02 (-0.05 to +0.09))
- Game lines, price: 25 bets, 3-22-0, -14.57u (ROI -0.58 (-1.00 to -0.00))
- Game lines, model: 69 bets, 25-44-0, -10.34u (ROI -0.15 (-0.45 to +0.15))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

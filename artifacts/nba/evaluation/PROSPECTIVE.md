# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Tue, Nov 24, 2026, 6:50 AM EST. 243 of 243 tipped games graded through Nov 23; 22934 main-line and 47891 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 243 | +0.011 (+0.006 to +0.015) | -0.000 (-0.001 to +0.000) | +0.85 (+0.76 to +0.94) | 1148-1334-0, -59.77u | not supported |
| **All game lines** | 243 | +0.032 (+0.015 to +0.050) | +0.002 (+0.000 to +0.003) | +0.37 (+0.10 to +0.65) | 31-68-0, -19.77u | not supported |
| Moneyline | 243 | +0.041 (+0.017 to +0.064) | +0.003 (+0.000 to +0.005) | +0.74 (+0.25 to +1.23) | 18-58-0, -22.23u | not supported |
| Spread | 240 | +0.028 (+0.004 to +0.053) | +0.001 (-0.001 to +0.003) | -0.07 (-0.30 to +0.14) | 12-8-0, +3.52u | not supported |
| Game total | 241 | +0.027 (+0.007 to +0.046) | +0.002 (-0.000 to +0.004) | +0.02 (-0.20 to +0.27) | 1-2-0, -1.05u | not supported |
| Assists | 243 | +0.014 (+0.006 to +0.022) | -0.001 (-0.001 to +0.000) | +1.01 (+0.82 to +1.19) | 162-199-0, -11.27u | not supported |
| Blocks | 160 | +0.006 (-0.010 to +0.024) | -0.001 (-0.002 to +0.001) | +0.73 (+0.33 to +1.14) | 1-4-0, -3.07u | not supported |
| Points | 243 | +0.006 (+0.000 to +0.012) | -0.001 (-0.001 to -0.000) | +0.85 (+0.70 to +1.02) | 188-209-0, -6.91u | supported |
| Points + assists | 243 | +0.010 (+0.003 to +0.017) | -0.000 (-0.001 to +0.000) | +0.86 (+0.72 to +1.00) | 59-71-0, -10.06u | not supported |
| Points + rebounds | 243 | +0.005 (-0.001 to +0.011) | -0.001 (-0.001 to -0.000) | +0.91 (+0.77 to +1.04) | 126-132-0, +1.00u | supported |
| Points + rebounds + assists | 243 | +0.013 (+0.005 to +0.021) | -0.000 (-0.001 to +0.000) | +0.82 (+0.66 to +0.98) | 140-169-0, -24.29u | not supported |
| Rebounds | 243 | +0.013 (+0.007 to +0.020) | -0.000 (-0.001 to +0.001) | +1.03 (+0.88 to +1.17) | 184-197-0, +1.35u | not supported |
| Rebounds + assists | 243 | +0.020 (+0.010 to +0.030) | -0.000 (-0.001 to +0.001) | +0.86 (+0.69 to +1.04) | 139-151-0, -8.24u | not supported |
| Steals | 199 | -0.000 (-0.011 to +0.009) | -0.001 (-0.002 to +0.000) | +0.45 (+0.21 to +0.76) | 5-2-0, +3.43u | not supported |
| Threes made | 243 | +0.012 (+0.006 to +0.018) | +0.000 (-0.001 to +0.001) | +0.52 (+0.40 to +0.64) | 144-200-0, -1.70u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1158 bets, 495-663-0, -78.01u (ROI -0.07 (-0.13 to -0.00))
- Props, model: 1324 bets, 653-671-0, +18.25u (ROI +0.01 (-0.06 to +0.08))
- Game lines, price: 27 bets, 5-22-0, -9.43u (ROI -0.35 (-0.87 to +0.30))
- Game lines, model: 72 bets, 26-46-0, -10.34u (ROI -0.14 (-0.45 to +0.16))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

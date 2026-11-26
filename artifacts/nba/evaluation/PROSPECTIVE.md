# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Thu, Nov 26, 2026, 5:05 PM EST. 255 of 255 tipped games graded through Nov 25; 24058 main-line and 49970 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 255 | +0.011 (+0.006 to +0.015) | -0.000 (-0.001 to +0.000) | +0.86 (+0.76 to +0.95) | 1190-1383-0, -61.93u | not supported |
| **All game lines** | 255 | +0.031 (+0.014 to +0.046) | +0.002 (-0.000 to +0.003) | +0.36 (+0.10 to +0.64) | 33-70-0, -19.11u | not supported |
| Moneyline | 255 | +0.037 (+0.015 to +0.061) | +0.002 (-0.000 to +0.004) | +0.75 (+0.32 to +1.26) | 20-60-0, -21.58u | not supported |
| Spread | 252 | +0.025 (-0.001 to +0.048) | +0.001 (-0.002 to +0.003) | -0.10 (-0.31 to +0.10) | 12-8-0, +3.52u | not supported |
| Game total | 253 | +0.030 (+0.011 to +0.049) | +0.002 (-0.000 to +0.004) | +0.05 (-0.15 to +0.27) | 1-2-0, -1.05u | not supported |
| Assists | 255 | +0.013 (+0.005 to +0.021) | -0.001 (-0.001 to +0.000) | +1.00 (+0.84 to +1.17) | 168-204-0, -9.79u | not supported |
| Blocks | 163 | +0.005 (-0.011 to +0.024) | -0.001 (-0.002 to +0.001) | +0.73 (+0.36 to +1.16) | 1-4-0, -3.07u | not supported |
| Points | 255 | +0.006 (+0.000 to +0.012) | -0.001 (-0.001 to +0.000) | +0.86 (+0.71 to +1.02) | 194-217-0, -8.11u | not supported |
| Points + assists | 255 | +0.009 (+0.002 to +0.016) | -0.000 (-0.001 to +0.000) | +0.88 (+0.73 to +1.02) | 62-73-0, -9.05u | not supported |
| Points + rebounds | 255 | +0.005 (-0.001 to +0.011) | -0.001 (-0.001 to -0.000) | +0.92 (+0.79 to +1.04) | 129-137-0, -0.70u | supported |
| Points + rebounds + assists | 255 | +0.013 (+0.005 to +0.020) | -0.000 (-0.001 to +0.000) | +0.82 (+0.68 to +0.98) | 143-173-0, -25.04u | not supported |
| Rebounds | 255 | +0.013 (+0.007 to +0.020) | -0.000 (-0.001 to +0.001) | +1.02 (+0.87 to +1.16) | 190-209-0, -4.10u | not supported |
| Rebounds + assists | 255 | +0.020 (+0.010 to +0.029) | +0.000 (-0.001 to +0.001) | +0.90 (+0.72 to +1.08) | 147-156-0, -4.94u | not supported |
| Steals | 204 | +0.000 (-0.011 to +0.012) | -0.001 (-0.002 to +0.000) | +0.44 (+0.18 to +0.72) | 5-2-0, +3.43u | not supported |
| Threes made | 255 | +0.013 (+0.007 to +0.019) | +0.000 (-0.000 to +0.001) | +0.52 (+0.40 to +0.64) | 151-208-0, -0.55u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1206 bets, 516-690-0, -81.85u (ROI -0.07 (-0.13 to -0.00))
- Props, model: 1367 bets, 674-693-0, +19.92u (ROI +0.01 (-0.05 to +0.08))
- Game lines, price: 28 bets, 5-23-0, -10.43u (ROI -0.37 (-0.86 to +0.20))
- Game lines, model: 75 bets, 28-47-0, -8.69u (ROI -0.12 (-0.41 to +0.18))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sun, Nov 22, 2026, 6:50 AM EST. 226 of 226 tipped games graded through Nov 21; 21422 main-line and 44745 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 226 | +0.011 (+0.007 to +0.016) | -0.000 (-0.001 to +0.000) | +0.85 (+0.75 to +0.95) | 1099-1265-0, -48.23u | not supported |
| **All game lines** | 226 | +0.036 (+0.018 to +0.055) | +0.002 (+0.000 to +0.004) | +0.41 (+0.16 to +0.69) | 26-63-0, -25.13u | not supported |
| Moneyline | 226 | +0.046 (+0.022 to +0.069) | +0.003 (+0.001 to +0.005) | +0.84 (+0.37 to +1.32) | 13-55-0, -29.60u | not supported |
| Spread | 224 | +0.034 (+0.008 to +0.061) | +0.002 (-0.001 to +0.004) | -0.06 (-0.30 to +0.16) | 12-6-0, +5.51u | not supported |
| Game total | 224 | +0.027 (+0.007 to +0.048) | +0.002 (-0.000 to +0.003) | +0.04 (-0.20 to +0.28) | 1-2-0, -1.05u | not supported |
| Assists | 226 | +0.015 (+0.006 to +0.023) | -0.000 (-0.001 to +0.000) | +1.00 (+0.82 to +1.18) | 153-191-0, -14.97u | not supported |
| Blocks | 150 | +0.006 (-0.012 to +0.023) | -0.001 (-0.002 to +0.001) | +0.76 (+0.38 to +1.21) | 1-4-0, -3.07u | not supported |
| Points | 226 | +0.006 (-0.000 to +0.012) | -0.001 (-0.001 to -0.000) | +0.84 (+0.67 to +0.99) | 179-203-0, -11.03u | supported |
| Points + assists | 226 | +0.010 (+0.003 to +0.017) | -0.000 (-0.001 to +0.000) | +0.84 (+0.70 to +1.00) | 57-66-0, -7.41u | not supported |
| Points + rebounds | 226 | +0.006 (-0.001 to +0.012) | -0.001 (-0.002 to -0.000) | +0.92 (+0.79 to +1.05) | 123-126-0, +3.70u | supported |
| Points + rebounds + assists | 226 | +0.014 (+0.005 to +0.022) | -0.000 (-0.001 to +0.000) | +0.82 (+0.65 to +0.98) | 133-163-0, -25.61u | not supported |
| Rebounds | 226 | +0.015 (+0.008 to +0.022) | -0.000 (-0.001 to +0.001) | +1.04 (+0.88 to +1.19) | 180-183-0, +10.05u | not supported |
| Rebounds + assists | 226 | +0.020 (+0.010 to +0.030) | -0.000 (-0.001 to +0.001) | +0.86 (+0.67 to +1.03) | 134-137-0, +1.01u | not supported |
| Steals | 187 | -0.002 (-0.012 to +0.011) | -0.001 (-0.002 to -0.000) | +0.41 (+0.14 to +0.69) | 5-2-0, +3.43u | supported |
| Threes made | 226 | +0.012 (+0.006 to +0.019) | +0.000 (-0.001 to +0.001) | +0.52 (+0.40 to +0.65) | 134-190-0, -4.32u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 1100 bets, 471-629-0, -72.52u (ROI -0.07 (-0.13 to +0.00))
- Props, model: 1264 bets, 628-636-0, +24.29u (ROI +0.02 (-0.05 to +0.09))
- Game lines, price: 23 bets, 2-21-0, -14.09u (ROI -0.61 (-1.00 to +0.02))
- Game lines, model: 66 bets, 24-42-0, -11.04u (ROI -0.17 (-0.46 to +0.15))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

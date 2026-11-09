# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Mon, Nov 9, 2026, 5:05 PM EST. 131 of 131 tipped games graded through Nov 8; 12431 main-line and 26622 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 131 | +0.014 (+0.007 to +0.022) | -0.000 (-0.001 to +0.000) | +0.83 (+0.70 to +0.96) | 710-826-0, -49.57u | insufficient sample |
| **All game lines** | 131 | +0.033 (+0.006 to +0.057) | +0.002 (-0.001 to +0.004) | +0.31 (-0.04 to +0.66) | 18-42-0, -21.98u | insufficient sample |
| Moneyline | 131 | +0.051 (+0.018 to +0.085) | +0.003 (-0.000 to +0.006) | +0.55 (+0.01 to +1.10) | 8-36-0, -25.57u | insufficient sample |
| Spread | 130 | +0.017 (-0.024 to +0.056) | -0.001 (-0.004 to +0.003) | +0.00 (-0.32 to +0.30) | 10-5-0, +4.59u | insufficient sample |
| Game total | 130 | +0.031 (+0.004 to +0.060) | +0.002 (-0.001 to +0.004) | +0.13 (-0.25 to +0.52) | 0-1-0, -1.00u | insufficient sample |
| Assists | 131 | +0.023 (+0.011 to +0.035) | -0.000 (-0.001 to +0.001) | +0.95 (+0.70 to +1.22) | 99-137-0, -24.15u | insufficient sample |
| Blocks | 89 | +0.008 (-0.014 to +0.034) | -0.000 (-0.003 to +0.002) | +0.58 (+0.13 to +1.08) | 1-3-0, -2.07u | insufficient sample |
| Points | 131 | +0.008 (-0.002 to +0.016) | -0.001 (-0.001 to +0.000) | +0.81 (+0.60 to +1.03) | 119-126-0, -0.70u | insufficient sample |
| Points + assists | 131 | +0.012 (+0.002 to +0.024) | -0.001 (-0.002 to +0.000) | +0.79 (+0.59 to +1.02) | 30-39-0, -8.64u | insufficient sample |
| Points + rebounds | 131 | +0.006 (-0.004 to +0.016) | -0.001 (-0.002 to -0.000) | +0.88 (+0.72 to +1.05) | 74-77-0, -0.11u | insufficient sample |
| Points + rebounds + assists | 131 | +0.016 (+0.004 to +0.029) | -0.001 (-0.002 to +0.001) | +0.86 (+0.68 to +1.06) | 87-115-0, -25.89u | insufficient sample |
| Rebounds | 131 | +0.021 (+0.011 to +0.031) | +0.000 (-0.001 to +0.001) | +0.96 (+0.76 to +1.17) | 120-129-0, -1.91u | insufficient sample |
| Rebounds + assists | 131 | +0.026 (+0.010 to +0.043) | -0.000 (-0.002 to +0.001) | +0.87 (+0.64 to +1.11) | 85-84-0, +2.64u | insufficient sample |
| Steals | 104 | +0.008 (-0.006 to +0.023) | -0.000 (-0.002 to +0.001) | +0.36 (+0.03 to +0.72) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 131 | +0.011 (+0.001 to +0.019) | -0.000 (-0.001 to +0.000) | +0.56 (+0.41 to +0.73) | 91-114-0, +9.24u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 661 bets, 284-377-0, -45.41u (ROI -0.07 (-0.15 to +0.02))
- Props, model: 875 bets, 426-449-0, -4.16u (ROI -0.00 (-0.08 to +0.08))
- Game lines, price: 13 bets, 0-13-0, -13.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 47 bets, 18-29-0, -8.98u (ROI -0.19 (-0.52 to +0.11))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

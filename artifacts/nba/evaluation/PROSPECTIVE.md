# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Sun, Nov 8, 2026, 6:49 AM EST. 124 of 124 tipped games graded through Nov 7; 11774 main-line and 25163 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 124 | +0.015 (+0.008 to +0.022) | -0.000 (-0.001 to +0.000) | +0.83 (+0.69 to +0.97) | 686-798-0, -47.04u | insufficient sample |
| **All game lines** | 124 | +0.037 (+0.010 to +0.061) | +0.002 (-0.001 to +0.004) | +0.30 (-0.04 to +0.64) | 18-40-0, -19.98u | insufficient sample |
| Moneyline | 124 | +0.054 (+0.020 to +0.089) | +0.003 (+0.000 to +0.007) | +0.53 (-0.02 to +1.10) | 8-34-0, -23.57u | insufficient sample |
| Spread | 123 | +0.025 (-0.014 to +0.064) | +0.000 (-0.004 to +0.004) | +0.00 (-0.32 to +0.31) | 10-5-0, +4.59u | insufficient sample |
| Game total | 123 | +0.031 (+0.003 to +0.060) | +0.002 (-0.001 to +0.005) | +0.13 (-0.22 to +0.55) | 0-1-0, -1.00u | insufficient sample |
| Assists | 124 | +0.025 (+0.012 to +0.037) | -0.000 (-0.001 to +0.001) | +0.96 (+0.70 to +1.21) | 95-134-0, -24.72u | insufficient sample |
| Blocks | 85 | +0.010 (-0.014 to +0.036) | -0.000 (-0.003 to +0.002) | +0.53 (+0.11 to +1.00) | 1-3-0, -2.07u | insufficient sample |
| Points | 124 | +0.007 (-0.002 to +0.016) | -0.001 (-0.002 to +0.000) | +0.83 (+0.60 to +1.05) | 115-122-0, -0.69u | insufficient sample |
| Points + assists | 124 | +0.012 (+0.000 to +0.024) | -0.001 (-0.002 to +0.001) | +0.81 (+0.59 to +1.03) | 29-38-0, -8.62u | insufficient sample |
| Points + rebounds | 124 | +0.006 (-0.005 to +0.015) | -0.001 (-0.002 to -0.000) | +0.88 (+0.71 to +1.06) | 72-74-0, +0.84u | insufficient sample |
| Points + rebounds + assists | 124 | +0.017 (+0.004 to +0.030) | -0.001 (-0.002 to +0.001) | +0.86 (+0.66 to +1.07) | 84-111-0, -25.04u | insufficient sample |
| Rebounds | 124 | +0.022 (+0.012 to +0.033) | +0.000 (-0.001 to +0.001) | +0.97 (+0.77 to +1.21) | 115-122-0, -0.42u | insufficient sample |
| Rebounds + assists | 124 | +0.028 (+0.013 to +0.045) | +0.000 (-0.001 to +0.001) | +0.87 (+0.62 to +1.13) | 83-82-0, +2.83u | insufficient sample |
| Steals | 99 | +0.010 (-0.005 to +0.026) | -0.000 (-0.002 to +0.001) | +0.31 (-0.01 to +0.62) | 4-2-0, +2.02u | insufficient sample |
| Threes made | 124 | +0.010 (+0.001 to +0.019) | -0.000 (-0.001 to +0.000) | +0.57 (+0.41 to +0.74) | 88-110-0, +8.83u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 637 bets, 275-362-0, -40.35u (ROI -0.06 (-0.15 to +0.02))
- Props, model: 847 bets, 411-436-0, -6.69u (ROI -0.01 (-0.09 to +0.08))
- Game lines, price: 12 bets, 0-12-0, -12.00u (ROI -1.00 (-1.00 to -1.00))
- Game lines, model: 46 bets, 18-28-0, -7.98u (ROI -0.17 (-0.52 to +0.17))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

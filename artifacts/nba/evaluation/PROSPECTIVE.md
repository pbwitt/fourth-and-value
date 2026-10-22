# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Wed, Oct 21, 2026, 8:02 PM EDT. 3 of 8 tipped games graded through Oct 20; 326 main-line and 700 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

Intervals appear once 10 games are graded; until then the numbers below are early anecdotes, not evidence.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 3 | +0.015 | -0.002 | +0.33 | 35-30-0, +7.72u | insufficient sample |
| **All game lines** | 3 | +0.030 | +0.001 | +0.28 | — | insufficient sample |
| Moneyline | 3 | +0.104 | +0.008 | +0.35 | — | insufficient sample |
| Spread | 3 | +0.093 | +0.007 | +0.12 | — | insufficient sample |
| Game total | 3 | -0.108 | -0.012 | +0.43 | — | insufficient sample |
| Assists | 3 | -0.026 | -0.006 | +0.27 | 6-4-0, +3.04u | insufficient sample |
| Blocks | 3 | +0.184 | +0.014 | +0.14 | — | insufficient sample |
| Points | 3 | +0.016 | -0.001 | +0.24 | 8-7-0, +1.29u | insufficient sample |
| Points + assists | 3 | -0.021 | -0.005 | +0.81 | 0-4-0, -4.00u | insufficient sample |
| Points + rebounds | 3 | -0.005 | -0.004 | +0.39 | 1-0-0, +0.96u | insufficient sample |
| Points + rebounds + assists | 3 | +0.031 | -0.002 | +0.31 | 3-4-0, -1.05u | insufficient sample |
| Rebounds | 3 | +0.044 | +0.000 | +0.34 | 9-4-0, +5.45u | insufficient sample |
| Rebounds + assists | 3 | +0.057 | +0.001 | +0.46 | 3-3-0, -0.01u | insufficient sample |
| Steals | 3 | -0.080 | -0.011 | -0.05 | 2-0-0, +2.66u | insufficient sample |
| Threes made | 3 | +0.017 | -0.001 | -0.03 | 3-4-0, -0.62u | insufficient sample |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 21 bets, 12-9-0, +5.77u (ROI +0.27)
- Props, model: 44 bets, 23-21-0, +1.95u (ROI +0.04)

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

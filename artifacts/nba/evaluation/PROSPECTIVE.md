# NBA v1 shadow model: prospective grading

Model `nba-v1.0`, generated Thu, Nov 19, 2026, 5:05 PM EST. 207 of 207 tipped games graded through Nov 18; 19595 main-line and 40700 alternate-line forecasts.

Log loss is lower-is-better; negative differences favor the model or blend. Movement is in probability points toward the model's earlier lean (positive means the market came to the model). The card pool is every offer that would have passed the NHL card's price rule, before research review and card caps.

| Market | Games | Model − market | 10% blend − market | Market moved toward model | Card pool | Readiness |
|---|---|---|---|---|---|---|
| **All props** | 207 | +0.012 (+0.007 to +0.017) | -0.000 (-0.001 to +0.000) | +0.85 (+0.75 to +0.96) | 996-1171-0, -70.44u | not supported |
| **All game lines** | 207 | +0.033 (+0.012 to +0.053) | +0.002 (-0.000 to +0.004) | +0.38 (+0.10 to +0.68) | 25-61-0, -25.48u | not supported |
| Moneyline | 207 | +0.044 (+0.018 to +0.071) | +0.003 (+0.000 to +0.005) | +0.78 (+0.26 to +1.27) | 12-53-0, -29.95u | not supported |
| Spread | 205 | +0.030 (+0.002 to +0.056) | +0.001 (-0.002 to +0.004) | -0.07 (-0.32 to +0.17) | 12-6-0, +5.51u | not supported |
| Game total | 205 | +0.025 (+0.005 to +0.047) | +0.001 (-0.001 to +0.004) | +0.02 (-0.21 to +0.27) | 1-2-0, -1.05u | not supported |
| Assists | 207 | +0.016 (+0.008 to +0.026) | -0.000 (-0.001 to +0.000) | +1.02 (+0.82 to +1.20) | 142-180-0, -18.57u | not supported |
| Blocks | 136 | +0.007 (-0.010 to +0.026) | -0.001 (-0.002 to +0.002) | +0.74 (+0.34 to +1.16) | 1-4-0, -3.07u | insufficient sample |
| Points | 207 | +0.006 (-0.001 to +0.013) | -0.001 (-0.001 to -0.000) | +0.85 (+0.68 to +1.03) | 161-182-0, -10.83u | supported |
| Points + assists | 207 | +0.012 (+0.003 to +0.019) | -0.000 (-0.001 to +0.000) | +0.84 (+0.66 to +1.00) | 44-56-0, -11.48u | not supported |
| Points + rebounds | 207 | +0.006 (-0.000 to +0.014) | -0.001 (-0.001 to -0.000) | +0.89 (+0.75 to +1.05) | 110-120-0, -4.26u | supported |
| Points + rebounds + assists | 207 | +0.015 (+0.006 to +0.025) | -0.000 (-0.001 to +0.000) | +0.83 (+0.65 to +1.00) | 118-154-0, -32.21u | not supported |
| Rebounds | 207 | +0.015 (+0.008 to +0.023) | -0.000 (-0.001 to +0.001) | +0.99 (+0.84 to +1.15) | 170-175-0, +6.90u | not supported |
| Rebounds + assists | 207 | +0.019 (+0.008 to +0.030) | -0.000 (-0.001 to +0.001) | +0.86 (+0.68 to +1.05) | 122-121-0, +4.66u | not supported |
| Steals | 171 | -0.002 (-0.014 to +0.010) | -0.001 (-0.002 to -0.000) | +0.40 (+0.13 to +0.67) | 5-2-0, +3.43u | supported |
| Threes made | 207 | +0.013 (+0.006 to +0.019) | +0.000 (-0.001 to +0.001) | +0.55 (+0.40 to +0.68) | 123-177-0, -5.00u | not supported |

**Card pool by source.** *Price*: one book is far enough off the other books that the consensus alone clears the floor (no model needed). *Model*: the value rests on the model disagreeing with the market.

- Props, price: 999 bets, 424-575-0, -73.17u (ROI -0.07 (-0.14 to -0.00))
- Props, model: 1168 bets, 572-596-0, +2.73u (ROI +0.00 (-0.07 to +0.07))
- Game lines, price: 20 bets, 1-19-0, -14.44u (ROI -0.72 (-1.00 to -0.17))
- Game lines, model: 66 bets, 24-42-0, -11.04u (ROI -0.17 (-0.44 to +0.15))

Proposed, not automatic: at least 150 graded games, the 10% blend beating the market on log loss with the whole 95% interval below zero, and the market moving toward the model with the whole interval above zero.

Caveats: snapshots are taken at 6:45 AM and 5 PM ET, hours before tipoff, so movement is a proxy for closing-line value, not CLV. The in-sample best blend weight in the JSON is chosen on the same games it scores and is optimistic. Card-pool results are flat one-unit bets at the offered price; a few weeks of them say little about edge.

# NBA v1 evaluation (nba-v1.0)

Generated 2026-10-17 by `scripts/nba/train.py` from `evaluation.json`. Test season: 2025–26 regular
season, never used for fitting or selection. Hyperparameters were selected on 2024–25 with a 2023–24 fit;
the evaluated model was then refit on 2023–24 (from December 1) and 2024–25. No sportsbook prices were used.

## Games

1230 test games. Home teams won 54.0% of training games.

| Measure | NBA v1 | Baseline |
|---|---|---|
| Moneyline log loss | 0.6025 | 0.6876 (training home-win rate) |
| Moneyline Brier | 0.2076 | 0.2472 |
| Total RMSE (points) | 18.97 | 19.01 (unadjusted pace × efficiency) |
| Total bias (actual − forecast) | +0.96 | |
| Margin RMSE (points) | 14.59 | |
| 80% total interval coverage | 77.6% | 80% target |

First three weeks (156 games, states carried from the prior season): moneyline log loss 0.6131, total bias +2.81, total RMSE 18.55.

Moneyline calibration (forecast → observed home-win rate, n):

0.16 → 0.21 (38), 0.26 → 0.18 (95), 0.36 → 0.37 (142), 0.45 → 0.46 (204), 0.55 → 0.54 (248), 0.65 → 0.68 (211), 0.75 → 0.79 (161), 0.85 → 0.83 (112)

Total PIT deciles (uniform is 123 each): [122, 119, 127, 116, 124, 126, 115, 102, 126, 153]. Margin: [135, 121, 94, 145, 116, 127, 142, 109, 120, 121].

## Players

26224 test appearances by players with at least 3 earlier appearances; 23979 also meet the production baseline's 20-game rule and form the comparison set.
Forecasts are conditional on the player appearing (books void props for players who do not).

### Count log loss (lower is better)

| Stat | NBA v1 | Recency baseline (30-game mean, same dispersion) |
|---|---|---|
| PTS | 3.1179 | 3.2394 |
| REB | 2.1782 | 2.2542 |
| AST | 1.7872 | 1.8342 |
| FG3M | 1.3375 | 1.3585 |
| BLK | 0.8201 | 0.8348 |
| STL | 1.1567 | 1.1761 |
| TOV | 1.3900 | 1.4184 |

### Over/under at pseudo-lines

Line = floor(last-10 average) + 0.5, so no pushes. These are diagnostics, not offered lines; real lines are
sharper, so these numbers do not establish profitability. Baseline = the production smoothed hit rate.
Combos are scored on a seeded random sample of 3,000 appearances (4,000 joint draws each).

| Market | n | Over rate | Slope | v1 log loss | Raw v1 | Baseline | v1 Brier | Baseline Brier |
|---|---|---|---|---|---|---|---|---|
| points | 23979 | 0.457 | 0.777 | 0.6741 | 0.6749 | 0.6957 | 0.2407 | 0.2505 |
| rebounds | 23979 | 0.443 | 1.020 | 0.6607 | 0.6608 | 0.6819 | 0.2342 | 0.2440 |
| assists | 23979 | 0.428 | 1.069 | 0.6575 | 0.6573 | 0.6789 | 0.2328 | 0.2423 |
| threes | 23979 | 0.365 | 1.036 | 0.5957 | 0.5961 | 0.6067 | 0.2074 | 0.2127 |
| blocks | 23979 | 0.279 | 1.078 | 0.5479 | 0.5478 | 0.5592 | 0.1836 | 0.1878 |
| steals | 23979 | 0.387 | 1.018 | 0.6296 | 0.6297 | 0.6447 | 0.2198 | 0.2264 |
| turnovers | 23979 | 0.419 | 1.075 | 0.6426 | 0.6423 | 0.6629 | 0.2257 | 0.2349 |
| points_rebounds_assists | 3000 | 0.474 | 0.830 | 0.6751 | 0.6759 | 0.7038 | 0.2412 | 0.2541 |
| points_rebounds | 3000 | 0.466 | 0.753 | 0.6777 | 0.6792 | 0.7082 | 0.2424 | 0.2563 |
| points_assists | 3000 | 0.461 | 0.759 | 0.6764 | 0.6776 | 0.7026 | 0.2418 | 0.2538 |
| rebounds_assists | 3000 | 0.464 | 1.008 | 0.6730 | 0.6729 | 0.6970 | 0.2401 | 0.2509 |

Calibration of the calibrated v1 probabilities (forecast → observed, n):

- points: 0.28 → 0.23 (385), 0.37 → 0.35 (5458), 0.45 → 0.45 (11586), 0.54 → 0.55 (5332), 0.64 → 0.65 (1050), 0.73 → 0.62 (159)
- rebounds: 0.18 → 0.18 (72), 0.27 → 0.24 (1516), 0.36 → 0.33 (5855), 0.45 → 0.44 (8061), 0.54 → 0.53 (5833), 0.64 → 0.62 (2168), 0.73 → 0.71 (431), 0.82 → 0.72 (43)
- assists: 0.17 → 0.17 (452), 0.26 → 0.26 (3045), 0.35 → 0.36 (6365), 0.45 → 0.44 (6890), 0.55 → 0.53 (4723), 0.64 → 0.59 (1962), 0.74 → 0.69 (497), 0.82 → 0.76 (38)
- threes: 0.05 → 0.02 (1676), 0.15 → 0.12 (1383), 0.26 → 0.26 (3819), 0.35 → 0.35 (6196), 0.45 → 0.43 (5628), 0.55 → 0.53 (3406), 0.64 → 0.63 (1581), 0.73 → 0.68 (277)
- blocks: 0.08 → 0.10 (1860), 0.15 → 0.16 (6604), 0.25 → 0.25 (6356), 0.35 → 0.35 (4356), 0.45 → 0.44 (2976), 0.54 → 0.54 (1412), 0.64 → 0.58 (342), 0.73 → 0.67 (73)
- steals: 0.17 → 0.17 (1513), 0.25 → 0.24 (4980), 0.35 → 0.33 (5735), 0.45 → 0.45 (5867), 0.54 → 0.53 (4423), 0.64 → 0.63 (1304), 0.72 → 0.79 (131)
- turnovers: 0.09 → 0.19 (42), 0.17 → 0.18 (1398), 0.26 → 0.28 (4151), 0.35 → 0.35 (5946), 0.45 → 0.45 (5587), 0.55 → 0.54 (4096), 0.64 → 0.63 (2090), 0.73 → 0.71 (591), 0.83 → 0.83 (78)
- points_rebounds_assists: 0.27 → 0.20 (56), 0.36 → 0.33 (606), 0.45 → 0.48 (1294), 0.54 → 0.55 (798), 0.64 → 0.58 (211), 0.73 → 0.84 (31)
- points_rebounds: 0.28 → 0.23 (30), 0.37 → 0.34 (569), 0.45 → 0.46 (1426), 0.54 → 0.54 (789), 0.63 → 0.58 (166)
- points_assists: 0.28 → 0.17 (40), 0.37 → 0.34 (612), 0.45 → 0.46 (1450), 0.54 → 0.53 (725), 0.63 → 0.62 (153)
- rebounds_assists: 0.27 → 0.27 (162), 0.36 → 0.38 (651), 0.45 → 0.44 (1029), 0.55 → 0.54 (799), 0.64 → 0.60 (288), 0.74 → 0.60 (63)

First three weeks: 3294 appearances; count log loss PTS 3.104, REB 2.167, AST 1.753, FG3M 1.308, BLK 0.790, STL 1.138, TOV 1.388.

## Selection

Team grid (validation Gaussian NLL of regulation scores, best first): half-life 10, carry 0.5, prior 3 → 7.6561; half-life 10, carry 0.25, prior 3 → 7.6567; half-life 20, carry 0.25, prior 3 → 7.658; half-life 10, carry 0.5, prior 8 → 7.6606.

Player grid (sum of validation count log loss, best first): rate half-life 20, prior 100 min, opponent weight 1.0 → 11.6329; rate half-life 20, prior 100 min, opponent weight 0.5 → 11.6338; rate half-life 50, prior 100 min, opponent weight 1.0 → 11.6341; rate half-life 50, prior 100 min, opponent weight 0.5 → 11.6368.

## Data

- `artifacts/nba/history/2023-24.json.gz` sha256 `5ab2dc68df9fce9171f91f21fd00b998d6b327f8fddb08de567a589ccb4e7b8c`
- `artifacts/nba/history/2024-25.json.gz` sha256 `b48576b2b4021be8689c1a0148e38e5aa5492d5b5a1793baa0a63a3e1fcd7bcf`
- `artifacts/nba/history/2025-26.json.gz` sha256 `7a1270067871534b3420c51de791214a9639c2ebe2f30bab2269f34d345f85f0`

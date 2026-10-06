# NHL v2.4: each player's record against our own forecasts

v2.4 keeps v2.3 unchanged (player history aged by games played, opponent defense) and adds one
factor per player. Each skater carries a ledger of his earlier games: what he actually produced, and
what the model's pre-game opportunity means expected for those same games. His forecast is scaled by
a shrunk ratio of the two:

    shots  = v2.3 shots mean  × clip((actual shots  + k_shots)   / (expected shots  + k_shots),   0.5, 2)
    goals, assists, points = v2.3 means × clip((actual points + k_scoring) / (expected points + k_scoring), 0.5, 2)

With little history the factor stays near 1. With a long record of beating (or missing) his own
forecasts it moves toward that record. Goals, assists and points share one factor, so goals + assists
still equal points. The ledger is point in time: a game enters it only once its results are available
(next day 12:00 UTC in the reconstruction), and its expected value is the opportunity mean his
features gave before that game (`History.player_features`, feature schema nhl-pit-4). `k` is chosen
by Poisson likelihood on training rows only (grid 10, 30, 100, 300, 1,000, 3,000 and "off"), the same
way v2.3 fits its opponent strength; dispersion is then refitted on the adjusted means.

Live artifact (refit on all four seasons): k_shots = k_scoring = 100 expected events; β_shots = 0.90,
β_scoring = 0.85; dispersion α = 0.061 (shots), 0.0066 (scoring). For players with 300 or more expected
shots in the ledger the shots factor runs from 0.88 (5th percentile) to 1.08 (95th), median 0.98.

The bundle carries each player's ledger through the archive (`ledger`, through 2025–26). Daily
inference adds later seasons' games by rebuilding their pre-game features exactly as evaluation does
(`features.forecast_ledger`); kinds without a player factor ignore it.

## Why this change

A research backtest of player-vs-opponent history and home/away splits
(`reports/matchups/`, research only) found that neither predicts the next game. The same backtest
found that correcting for players the model consistently over- or under-projected did help. This
version tests that correction through the locked protocol.

## Protocol

Unchanged locked protocol (`scripts/nhl/v2/evaluate.py`). The new kind `opportunity_nb_opp_player`
joined the player candidates; the existing rule chose it for shots and scoring on the 2023–24 and
2024–25 validation folds and wrote `selection-lock.json` before the 2025–26 final test was scored. The
final test also scores the previous production kind (`opportunity_nb_opp`, v2.3) for a paired
comparison on the same 47,230 player-games. The v2.3 kind reproduces its published validation and
final-test numbers exactly, so the ledger fields leave every earlier feature unchanged.

## Validation (count log loss, v2.3 kind → v2.4 kind)

| Season | Shots | Goals | Assists | Points |
|---|---|---|---|---|
| 20232024 | 1.58690 → 1.58378 | 0.45328 → 0.45320 | 0.64304 → 0.64259 | 0.83965 → 0.83912 |
| 20242025 | 1.54916 → 1.54758 | 0.44392 → 0.44391 | 0.63438 → 0.63413 | 0.82645 → 0.82619 |

All four markets improve in both folds.

## Final test, 2025–26 (paired against v2.3 on the same games)

| Market | v2.3 | v2.4 | Difference [95% game-cluster interval] | ECE (Over at the reference line) |
|---|---:|---:|---|---|
| shots | 1.54065 | 1.53846 | -0.00220 [-0.00298, -0.00141] | 0.0167 → 0.0060 |
| goals | 0.45255 | 0.45237 | -0.00019 [-0.00038, +0.00001] | 0.0056 → 0.0044 |
| assists | 0.64402 | 0.64364 | -0.00037 [-0.00065, -0.00010] | 0.0126 → 0.0062 |
| points | 0.83855 | 0.83799 | -0.00056 [-0.00093, -0.00016] | 0.0201 → 0.0114 |

Shots, assists and points improve clearly and calibration error at the reference lines falls by about
half; goals improve on average with an interval that touches zero. Team (game-line) models are
unchanged.

## Limits

- 2025–26 was the final test for v2.1 through v2.3, and the research backtest that suggested this
  change also scored 2025–26. The validation folds alone selected it and improved every market in both
  folds, but the final test is not untouched for this idea.
- Player evaluation is conditional on participation and covers every skater who played, not only
  players with posted props. Reference lines are diagnostic thresholds, not archived prices.
- The ledger is measured against pre-opponent opportunity means; the opponent factor averages out
  over a season of opponents but is not removed game by game.
- The starting goalie is not modeled; no betting edge is claimed; recommendations stay disabled.

## Reproduce

    python3 scripts/nhl/v2/restore.py --root /tmp/nhl-history
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/nhl/v2/evaluate.py --history /tmp/nhl-history --models models/nhl/v2
    python3 -c "import sys;sys.path.insert(0,'scripts');from nhl.v2.report import render;render()"

Prediction files are deterministic and not committed (see `.gitignore`):

- `player-predictions.jsonl.gz` sha256 `71cb5de5a04fed07883ae82d62724a5fd4de8a48821e56e3c3c290dce4c1b013`
- `team-predictions.jsonl.gz` sha256 `a7dd9e4ff1533ebe9ea803b2631023b36e53a3211af5b6ccb80138bba5d58158`

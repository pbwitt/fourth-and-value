# NHL v2.3: opponent defense in player forecasts

v2.3 keeps v2.2's player history (aged by games played) and adds the opponent. Expected shots scale with
the opponent's shots allowed per game, and goals, assists and points share one factor from its regulation
goals allowed, each relative to the league average:

    shots  = ice time × shots per 60 × (opponent shots allowed / league)^β_shots
    goals, assists, points = … × (opponent goals allowed / league)^β_scoring

The opponent figures are the recency-weighted values the team model already uses for game lines
(`History.team_features`: shots against and regulation goals against, half-life 90 days, prior strength 12).
The player's team is the one he last appeared for; when that team is not in the game the factor is 1.
β values are fitted by Poisson likelihood on training rows only (grid 0–2, step 0.05); dispersion is then
refitted on the adjusted means. Goals + assists still equal points.

Live artifact (refit on all four seasons): β_shots = 0.90, β_scoring = 0.85;
league means 29.77 shots and 3.002 goals allowed per game;
dispersion α = 0.0652 (shots), 0.0072 (scoring).

## Protocol

Unchanged locked protocol (`scripts/nhl/v2/evaluate.py`). The new kind `opportunity_nb_opp` joined the
player candidates; the existing rule chose it for shots and scoring on the 2023–24 and 2024–25 validation
folds and wrote `selection-lock.json` before the 2025–26 final test was scored. The final test also scores
the previous production kind (`opportunity_nb`, v2.2) for a paired comparison on the same 47,230
player-games. An exploratory check before this run used only the validation folds.

## Validation (count log loss, v2.2 kind → v2.3 kind)

| Season | Shots | Goals | Assists | Points |
|---|---|---|---|---|
| 20232024 | 1.5899 → 1.5869 | 0.4538 → 0.4533 | 0.6440 → 0.6430 | 0.8411 → 0.8396 |
| 20242025 | 1.5540 → 1.5492 | 0.4445 → 0.4439 | 0.6353 → 0.6344 | 0.8280 → 0.8265 |

## Final test, 2025–26 (paired against v2.2 on the same games)

| Market | v2.2 | v2.3 | Difference [95% game-cluster interval] | ECE (Over at the reference line) |
|---|---:|---:|---|---|
| shots | 1.54453 | 1.54065 | -0.00388 [-0.00469, -0.00305] | 0.0220 → 0.0167 |
| goals | 0.45270 | 0.45255 | -0.00014 [-0.00039, +0.00009] | 0.0074 → 0.0056 |
| assists | 0.64434 | 0.64402 | -0.00033 [-0.00075, +0.00007] | 0.0127 → 0.0126 |
| points | 0.83903 | 0.83855 | -0.00048 [-0.00115, +0.00016] | 0.0211 → 0.0201 |

Shots improve clearly. Goals, assists and points improve on average with lower calibration error, but
each interval includes zero. Team (game-line) models are unchanged.

## Limits

- 2025–26 was already used as the final test for v2.1 and v2.2, so it is not untouched for this model
  family, though it played no part in choosing this change.
- Player evaluation is conditional on participation and covers every skater who played, not only players
  with posted props. Reference lines are diagnostic thresholds, not archived prices.
- The starting goalie is not modeled; team defense blends the goalies a team used.
- No betting edge is claimed; recommendations stay disabled.

## Reproduce

    python3 scripts/nhl/v2/restore.py --root /tmp/nhl-history
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/nhl/v2/evaluate.py --history /tmp/nhl-history --models models/nhl/v2
    python3 -c "import sys;sys.path.insert(0,'scripts');from nhl.v2.report import render;render()"

Prediction files are deterministic and not committed (see `.gitignore`):

- `player-predictions.jsonl.gz` sha256 `4d854998581f9be276d51874d940b76ebab85e5c50d66ce9bd3b5933ea22f5ca`
- `team-predictions.jsonl.gz` sha256 `f73a696c36e60a1027cd26f9d2b6b47beae10b74199c0b4215b8c9a9977830de`

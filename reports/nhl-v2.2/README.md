# NHL v2.2: player history aged by games played

Model `nhl-v2.2`, feature schema `nhl-pit-2`. One change from v2.1, in
`History.player_features` (`scripts/nhl/v2/features.py`): a player's past games are
aged by the player's own later appearances (newest = 0) instead of by calendar days. Half-lives
are in games: projected ice time 14, per-minute rates 110, base means 82.5. Position
priors, prior strengths (ice time 5 games, rates 12), the 164-record window, the
availability rule, team features and the model family are unchanged. Recommendations
remain disabled.

## Why

v2.1 discounted history by calendar days (ice time 30-day half-life, rates 120). After the
five-month offseason last season carried almost no weight, so early-season projections
collapsed toward the league-wide position prior. A single prior also pulled every player
toward the middle: depth players were projected too high and the high-volume players that
books post props on too low. On v2.1's own 2025-26 final test, players with 2.5+ shots per
game the previous season were projected at 0.913 of their actual shots (0.843 in October),
while all players together were projected at 1.065. Live 2026-27 props showed the same
pattern: v2.1 projected about 15-18% below the market for the median prop player, and 97% of
the current NHL screen was Unders. A single global calibration factor makes this worse,
because it lowers the high-volume players further.

## Evidence

Half-lives were chosen on the 2023-24 and 2024-25 validation folds only (calendar baseline,
compressed-offseason variants, games-based age, then half-life and offseason-penalty grids),
locked, then scored once on 2025-26. An independent reviewer rebuilt the features from scratch,
reproduced every number, and found no leakage. This retrain through `evaluate.py` reproduces the
locked experiment's final test exactly.

Validation count log loss, v2.1 -> v2.2:

| fold | shots | goals | assists | points |
|---|---|---|---|---|
| 2023-24 | 1.59703 -> 1.58991 | 0.45530 -> 0.45376 | 0.64694 -> 0.64404 | 0.84577 -> 0.84114 |
| 2024-25 | 1.56157 -> 1.55403 | 0.44724 -> 0.44446 | 0.63849 -> 0.63533 | 0.83313 -> 0.82796 |

Final test 2025-26 (47,230 player-games, 1,312 games). Paired difference on identical rows with a
2,000-draw game-cluster bootstrap. Tiers use the previous season's shots per game.

| | shots | goals | assists | points |
|---|---|---|---|---|
| count log loss | 1.55355 -> 1.54453 | 0.45492 -> 0.45270 | 0.64798 -> 0.64434 | 0.84410 -> 0.83903 |
| difference [95% CI] | -0.0090 [-0.0101, -0.0079] | -0.0022 [-0.0029, -0.0016] | -0.0036 [-0.0044, -0.0029] | -0.0051 [-0.0061, -0.0041] |
| level, all players | 1.065 -> 1.056 | 1.073 -> 1.039 | 1.046 -> 1.019 | 1.056 -> 1.026 |
| level, 2.5+ shots/game | 0.913 -> 0.985 | 0.907 -> 0.981 | 0.873 -> 0.926 | 0.887 -> 0.948 |
| same, October | 0.843 -> 0.988 | 0.830 -> 0.981 | 0.816 -> 0.955 | 0.822 -> 0.966 |
| level, under 1.5 shots/game | 1.203 -> 1.114 | 1.277 -> 1.107 | 1.206 -> 1.085 | 1.230 -> 1.093 |
| binary log loss, reference line | 0.46265 -> 0.45787 | 0.39640 -> 0.39416 | 0.52331 -> 0.52025 | 0.60068 -> 0.59631 |
| calibration error (ECE) | 0.037 -> 0.022 | 0.013 -> 0.007 | 0.023 -> 0.013 | 0.035 -> 0.021 |

Level is predicted total divided by actual total. Reference lines: shots 2.5, others 0.5.

Live 2026-27 check through production inference: the 19 archived snapshots with player props were
re-forecast at their own decision times with only results available then, and settled props were
scored on the Over side at the first snapshot per outcome (games of September 29 to October 2).

| | shots | goals | assists | points |
|---|---|---|---|---|
| median model mean / market-implied mean | 0.846 -> 0.958 | 0.818 -> 0.958 | 0.821 -> 0.942 | 0.859 -> 0.986 |
| log loss, model (market) | 0.730 -> 0.698 (0.689) | 0.569 -> 0.574 (0.564) | 0.620 -> 0.617 (0.612) | 0.702 -> 0.692 (0.685) |
| current NHL screen, Overs/Unders | 0/55 -> 2/12 | 1/11 -> 6/1 | 2/35 -> 8/9 | 1/31 -> 7/6 |

The current screen is the latest snapshot (October 3 morning) under `candidates.exclusion`: 136
candidates at 97% Unders become 51 at 55% Unders.

## Limits

- This corrects the level; it does not establish an edge. On live props v2.2 is still slightly
  worse than the market consensus in every market, and goals are slightly worse than v2.1 on
  these 145 outcomes.
- Assists and points for high-volume players remain about 5-7% low, and players with little or
  no prior-season history are still projected 9-31% high. The position prior keeps about a fifth
  of the weight once history saturates: a steady 20-minute player projects near 18.6 minutes. A
  player-specific (hierarchical) prior scored better in a separate experiment but has not been
  independently verified; it is the candidate for v2.3.
- The live sample is 20 games. Settled outcomes at books with unverified settlement rules are
  scored on the official statistic.

## Reproduce

```bash
python scripts/nhl/v2/restore.py --root /tmp/nhl-history
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/nhl/v2/evaluate.py \
  --history /tmp/nhl-history --output reports/nhl-v2.2 --models models/nhl/v2
```

`evaluation.json`, `selection-lock.json` and `grading-summary.json` are committed. The
deterministic prediction files are not, to limit repository growth. SHA-256 of their
decompressed contents (the gzip container records a timestamp, so compressed bytes vary):
`player-predictions.jsonl` `4da749eb491e5d346d61b26c6932ac07aa83b927801d1f5f15d3f6b76366f99b`,
`team-predictions.jsonl` `a30601fa922f58e71609440cba49bd64f9ef7ccc591e36185e5856debde08f19`. The v2.1 evaluation in
`reports/nhl-rebuild/` is unchanged and still describes v2.1.

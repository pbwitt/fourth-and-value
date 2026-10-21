# NFL passing volume and recency window — October 21, 2026

Follow-up to [the curve-shape and published-mean audit](../2026-10-21-shape-and-level/), which
found the published NFL mean ran low on passing volume in 2026 Weeks 1–6 (+14.4 passing yards,
+2.3 attempts, +1.5 completions per game) and that a smoother recency average beat the
production one on receiving yards and receptions. This report tests two changes and ships them:

1. **Quarterback starts only.** Passing history (the season form, each quarterback's career and
   the position pool) counts only games in which he led his team's quarterbacks in pass
   attempts. Relief and garbage-time appearances are 17% of quarterback appearances; the old
   all-appearance pool averaged 26–28 attempts while starts average 31–33. Partial starts
   (injury exits, blowouts) stay, because a prop on a starter is graded on them too.
2. **Full-season recency window.** Season form uses every current-season game with a game's
   weight halving every six games (decay 0.891), replacing the last four games at decay 0.4
   (61.6% on the newest game). The season/career blend is unchanged: n / (n + 4) after n games.

No probability formula, calibration curve, Top Picks rule or market blend changes. The NFL
props model weight in the market blend stays 0.10.

## Method (`replay_means.py`, `score_replay.py`, `check_fidelity.py`)

`replay_means.py` calls the production estimators in `make_player_prop_params.py` week by week
for 2024, 2025 and 2026 Weeks 1–6, with only what production had before each week: current-season
logs and the six-season career window, both cut at the week. Variants are set through
`career_baseline`'s settings. **Fidelity:** with the old settings the replay reproduces every
published pre-adjustment mean on the archived 2026 boards exactly (1,399 of 1,399 matched rows;
`results/fidelity.txt`).

`score_replay.py` scores each variant on fixed rows chosen from the old settings: passing
markets on quarterback starts; rushing markets where the old mean was 6+ carries (RB) or 4+
(starting QB); receiving markets where it was 2+ receptions. It reports RMSE, bias (result minus
mean) and the interval of the MSE difference from resampling whole games. The 2026 board rows are
the player-markets actually published, graded in the shape-and-level audit.

## Results (shipped variant: starts only + full-season window, half-life 6)

Bias is the result minus the mean; positive means the mean was too low. Intervals are for the
change in mean squared error; negative is better.

**Passing**

| Rows | Market | Old RMSE (bias) | New RMSE (bias) | MSE change [95%] |
|---|---|---|---|---|
| 2026 board rows (93 games) | Attempts | 9.14 (+2.96) | 8.41 (+0.56) | [−21.8, −4.4] |
| | Completions | 6.32 (+1.94) | 5.82 (+0.25) | [−11.0, −1.3] |
| | Passing yards | 74.44 (+19.3) | 71.61 (−0.8) | [−1,096, +194] |
| 2026 Weeks 1–6, all starts | Attempts | 9.06 (+2.84) | 8.44 (+0.32) | [−19.5, −2.6] |
| | Passing yards | 73.83 (+18.1) | 71.94 (−2.8) | [−942, +315] |
| 2024–25 starts (544 games) | Attempts | 8.46 (+1.52) | 8.16 (−0.99) | [−9.6, −0.5] |
| | Completions | 6.06 (+1.01) | 5.85 (−0.67) | [−4.7, −0.6] |
| | Passing yards | 75.31 (+9.7) | 74.10 (−10.4) | [−469, +117] |

**Receiving and rushing** (recency change only; the starts filter does not touch them)

| Rows | Market | Old RMSE | New RMSE | MSE change [95%] |
|---|---|---|---|---|
| 2024–25 | Receiving yards | 30.09 | 29.14 | [−67.9, −43.5] |
| | Receptions | 2.17 | 2.10 | [−0.3, −0.2] |
| | Rush attempts | 2.86 | 2.75 | [−0.8, −0.4] |
| | Rushing yards | 19.35 | 18.97 | [−27.3, −1.9] |
| 2026 Weeks 1–6 | Receiving yards | 31.22 | 31.00 | [−32.5, +6.6] |
| | Rushing yards | 20.08 | 19.58 | [−49.0, +1.8] |
| 2026 board rows | Rush attempts | 2.68 | 2.62 | [−0.6, −0.1] |

The other variants are in `results/scores.txt`. A last-eight-games window and a half-life of four
gave nearly the same results; doubling the career weight (`K_RECENT_MU` 8) helped 2024–25 passing
but was worse on 2026 receiving (not significantly), so it was not adopted.

## What remains

- **League drift.** In 2024–25 the starts filter overshoots passing volume by about one attempt
  a game (bias −0.99) because league passing fell in those seasons against the earlier seasons in
  the career window (team-leading quarterbacks: 33.4 attempts in 2021, 31.2 in 2025, 32.9 in 2026
  so far). The career baseline does not yet adjust for that drift.
- **Calibration.** The probability calibration curve was fitted on the old model's raw
  probabilities (2025, nine weeks) and is not refitted here; the historical prop snapshots it
  needs are not in the repository. Means are more accurate, but the published probabilities
  still pass through the old curve.
- **Rows chosen post hoc.** Passing rows use the week's actual starter, which books know in
  advance for almost every game. The variant was chosen on 2024–25; 2026 and its board rows
  confirm the direction on a smaller sample.

## Reproduce

Data: nflverse `stats_player_week_<season>.csv.gz` for 2018–2026 (retrieved October 20–21, 2026),
saved as `data/weekly_player_stats_<season>.parquet` under one folder (production's own file
format). The graded 2026 rows come from the shape-and-level audit
(`production_mean_test.py ... --max-week 6 --out graded.csv`).

```
for v in production starts recent8_hl4 recent17_hl4 recent17_hl6 recent17_hl4_k8 \
         starts_recent17_hl4 starts_recent17_hl6; do
  python replay_means.py <data-folder> <repo> replay/$v.csv $v
done
python check_fidelity.py replay graded.csv
python score_replay.py replay graded.csv
```

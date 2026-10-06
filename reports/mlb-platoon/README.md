# MLB platoon on the production model: research only

2026-10-06. Owner: pbwitt. Follows research report FV-2026-03 (`reports/matchups/`), which found
on Retrosheet plate appearances (2012–2025) that same-handed matchups have about 12% fewer home
runs, 13% fewer walks and 8% more strikeouts than opposite-handed ones.

**Question.** Does adding batter side vs pitcher hand improve the production MLB model, which
trains daily on MLB Stats API box scores (`scripts/mlb/model_data.py`, `train.py`)?

**Answer.** No reliable gain. Pooled over four held-out 30-day windows (1,448 regular-season
games), no market got significantly better or worse. In `train.py`'s own split, moneylines and run
lines got significantly worse, and the 2025 postseason home-run check flipped from pass to fail.
Under the decision rule fixed before any production result, platoon stays **research only**.
`PLATOON_LIVE` remains `False` in `scripts/mlb/models.py`, so production features are unchanged.

## What was tested

- **Data.** The production game history: 3,285 completed games, 2025-08-01 through 2026-10-05
  (the daily Actions cache plus that day's refresh). Batter side and pitcher hand come from
  `sports/1/players` for 1,841 players, covering 100% of starters and 100% of batter plate
  appearances (40% left, 49% right, 11% switch).
- **Features** (`scripts/mlb/models.py`, built only from earlier dates):
  - `starter_left`, `opp_starter_left`: the two starters' throwing hands
  - `team_adv_share`, `opp_adv_share`: the share of a team's PA (last 40 games) from batters with
    the platoon edge against today's starter hand. Games against starters of that hand count
    first, shrunk toward the team's overall mix
  - `batter_same_hand`, `batter_switch`: the batter's side against today's starter
  - `batter_same_share`: the batter's expected share of PA against same-hand pitchers, from the
    starter's share of batters faced plus the opposing bullpen's recent left/right mix
- **Comparison** (`scripts/mlb/platoon_study.py`). Every target is fitted twice on identical
  rows, outcomes and dates, with and without these columns. Splits:
  - fold 0 is `train.py`'s own split (weights through 60 days before the latest game,
    calibration for the next 30, test on the last 30)
  - folds 1–3 repeat that recipe ending 30, 60 and 90 days earlier
  - the prior-postseason audit is repeated as in `train.py`

  Changes are paired forecast by forecast at `train.py`'s fixed line grid. Intervals are 95%
  bootstrap intervals over 2,000 resamples of whole games.

## Decision rule

Fixed before any production result (see `rule` in `results.json`):

- **Worse:** a market's pooled Brier or log-loss interval lies entirely above zero.
- **Better:** its pooled Brier interval lies entirely below zero.
- **Neutral:** the precision-weighted average of pooled Brier changes is at or below zero.
- **Calibration guard:** no market's calibration gap rises by more than 0.005 with an interval
  above zero, and no market that passes `train.py`'s regular-season or postseason checks fails
  them with platoon.
- **Ship:** no market is worse, the guard holds, and the result is better or neutral.

The rule was amended once, before any production result was read. A synthetic league with no
platoon effect tripped the original calibration guard (any rise above 0.005) on moneylines by
chance, and the unweighted average was decided by the noisy game markets. The original rule gives
the same verdict here. After the production run, one bug was fixed: pitcher strikeouts gives
identical forecasts either way, and its zero-width interval swamped the weighted average. It now
gets no weight. The corrected weighted change is −0.003 ×10⁻⁴ (neutral), and the verdict is
unchanged.

On a synthetic league with a planted platoon effect the same study ships (total bases
significantly better). With no effect it does not.

## Results

Changes are platoon minus no platoon, ×10⁻⁴; negative is better.

**Pooled, four regular-season windows (2026-06-08 to 2026-10-05)**

| Market | Forecasts | Games | Brier change | Log-loss change | Calibration gap without → with |
|---|---|---|---|---|---|
| Pitcher strikeouts | 18,760 | 1,437 | 0 (rolling model selected both ways) | 0 | 0.0112 → 0.0112 |
| Pitcher outs | 10,720 | 1,437 | +1.39 (−2.74 to +5.49) | +1.37 (−8.72 to +11.55) | 0.0319 → 0.0317 |
| Batter hits | 74,676 | 1,448 | +0.34 (−0.80 to +1.39) | +1.19 (−2.74 to +4.97) | 0.0022 → 0.0017 |
| Total bases | 99,568 | 1,448 | −0.80 (−1.77 to +0.20) | −2.61 (−5.55 to +0.41) | 0.0049 → 0.0051 |
| Home runs | 24,892 | 1,448 | +0.05 (−0.89 to +1.07) | +0.26 (−3.90 to +4.87) | 0.0036 → 0.0027 |
| RBIs | 49,784 | 1,448 | +0.82 (−0.63 to +2.24) | +3.66 (−0.93 to +8.61) | 0.0058 → 0.0050 |
| Game totals | 7,240 | 1,448 | −17.74 (−39.58 to +5.29) | −37.86 (−85.74 to +12.48) | 0.0222 → 0.0203 |
| Moneylines | 1,448 | 1,448 | +19.96 (−9.93 to +50.74) | +41.24 (−19.82 to +103.70) | 0.0331 → 0.0348 |
| Run lines | 2,896 | 1,448 | +12.79 (−11.65 to +36.24) | +26.71 (−25.23 to +76.49) | 0.0354 → 0.0289 |

**`train.py`'s own split (fold 0, test 2026-09-06 to 2026-10-05, 288–290 games)**

| Market | Brier change | Log-loss change | Validation check |
|---|---|---|---|
| Pitcher strikeouts | 0 (identical) | 0 | passes either way |
| Pitcher outs | −8.37 (−20.70 to +4.65) | −23.61 (−54.56 to +8.22) | passes either way |
| Batter hits | +1.44 (−0.89 to +4.06) | +7.03 (−1.47 to +16.70) | passes either way |
| Total bases | +1.54 (−0.33 to +3.47) | +3.64 (−1.38 to +8.78) | passes either way |
| Home runs | −0.03 (−2.54 to +2.56) | −0.60 (−10.73 to +8.82) | passes either way |
| RBIs | +2.11 (−1.33 to +5.88) | +10.91 (−2.69 to +26.97) | passes either way |
| Game totals | +4.70 (−17.51 to +27.33) | +10.20 (−37.75 to +59.51) | passes either way |
| Moneylines | **+36.11 (+4.52 to +65.69)** | **+73.98 (+9.44 to +134.16)** | fails either way |
| Run lines | **+32.05 (+8.09 to +55.39)** | **+68.26 (+17.25 to +117.18)** | passes either way |

**2025 postseason audit (46–47 games; earlier model trained through 2025-09-07)**

| Market | Brier change | Validation check |
|---|---|---|
| Batter hits | +4.20 (−3.22 to +11.23) | passes either way |
| Total bases | −0.10 (−9.44 to +8.92) | passes either way |
| Home runs | +37.61 (−18.21 to +101.08) | **passes without, fails with** (Brier skill +1.8% → −1.7%) |
| RBIs | −4.17 (−12.55 to +4.43) | passes either way |
| Pitchers, game lines | 0 (identical forecasts) | unchanged |

The postseason check gates playoff picks, which are live now.

## Why it does not help here

1. **Box scores blur the matchup.** The research effect is per plate appearance. A box score gives
   only game totals, and a batter faces the starter in roughly 60% of his plate appearances. On
   these games, outcomes relative to each batter's own rolling rate barely differ by today's
   starter:

   | | Same hand ÷ opposite hand (box scores) | Retrosheet, per PA |
   |---|---|---|
   | Hits | 1.02 (wrong direction) | 0.96 |
   | Total bases | 1.00 | — |
   | Home runs | 0.99 | 0.88 |

2. **Managers already platoon, and the rates already know it.** Weak-side platoon hitters sit
   against same-handed starters, so same-handed starts go mostly to regulars who hit both sides.
   Each batter's rolling rates are built from the mix of pitchers he usually faces.
3. **Even a correct effect is below what this data can resolve.** Applying the research HR
   factor to 60% of a game's PA moves a typical P(HR ≥ 1) of about 0.12 by under 0.01. That is a
   Brier change on the order of 0.1–0.3 ×10⁻⁴, inside intervals about ±1 ×10⁻⁴ wide. Meanwhile
   the boosted models' extra splits add variance: that is what showed up in fold 0's game lines
   and the postseason home-run check.

## What would change this

- **Plate-appearance data from the Stats API play-by-play feed** (`game/{pk}/playByPlay`). With
  each PA's batter and pitcher known, platoon can enter as a per-PA multiplier with factors fitted
  on earlier seasons, as in the Retrosheet study, rather than as learned box-score splits.
- **A fixed multiplier on the rolling baseline** (opportunity × rate × platoon factor), with
  factors estimated on seasons before every test window, run through this same study.
- **More held-out games.** Each extra month of regular season narrows the intervals.

## What stays in the code

- `model_data.update_players()` fetches and caches handedness under
  `data/mlb/model_data/players/`. Only the study calls it; production training does not, because
  `PLATOON_LIVE` is `False`.
- The platoon features exist only when `State` is given handedness. Production features are
  byte-identical to before.
- The **MLB Platoon Study** workflow (`.github/workflows/mlb-platoon-study.yml`) reruns this
  comparison on the production cache whenever the model, training or data code changes, and on
  manual dispatch.

## Reproduce

```
python scripts/mlb/platoon_study.py               # refreshes box scores and handedness (needs statsapi.mlb.com)
python scripts/mlb/platoon_study.py --cached-history
```

`results.json` is the output of the Actions run on this PR's final study commit, copied from its
log. It contains the full per-market detail, windows, coverage and the decision.

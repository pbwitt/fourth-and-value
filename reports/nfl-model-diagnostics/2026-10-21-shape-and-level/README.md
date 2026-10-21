# NFL prop curve shape and published-mean audit — October 21, 2026

Research diagnostics, not a model change or betting claim. Nothing here alters a published
probability, Top Picks eligibility or ranking. The scripts and their saved output are in this
folder; the data sources and exact revisions are listed under **Reproduce**.

Prompted by a comparison with Unabated's published approach: sportsbooks set player-prop lines
near the median outcome, while projections are averages, and yardage is skewed upward, so the
average sits above the median. Our NFL props price yardage, receptions, attempts and completions
with a Normal (bell) curve centred on the average, which makes the median equal the average
(touchdowns and interceptions use Poisson counts).

## 1. Curve shape (`shape_backtest.py`)

Forecasts use only games played before each one: a recency-weighted average (half-life six
games, two games of position-prior weight) over the last two seasons. The variance curve
(`var = a·m + b·m²`) is fitted on 2022–23 forecasts and frozen. Scoring is on 2024–25 regular
seasons: 4,884 receiving, 2,352 rushing, 1,045 passing and 4,920 reception forecasts (538–544
games). Lines are placed at 0.6, 0.8, 1.0, 1.2 and 1.4 × the forecast average, rounded to a half
point, and each curve's Over probability is scored on the result. The Normal and skewed curves get
the **same average and spread**, so only the shape differs. These are simple forecasts, not the
production mean.

Log loss of the skewed curve minus the Normal (negative favours the skewed curve), 95% interval
from resampling whole games:

| Market | Skewed curve | Smooth recency (half-life 6) | Production-style recency (half-life 0.76) |
|---|---|---|---|
| Receiving yards | gamma | −0.0132 [−0.0178, −0.0083] | −0.0106 [−0.0155, −0.0055] |
| Receptions | negative binomial | −0.0045 [−0.0068, −0.0021] | −0.0035 [−0.0060, −0.0011] |
| Rushing yards | gamma | −0.0118 [−0.0187, −0.0058] | +0.0022 [−0.0046, +0.0092] |
| Passing yards | gamma | +0.0013 [−0.0020, +0.0045] | +0.0046 [+0.0015, +0.0077] |

At a line equal to the forecast average (half-life 6, pooled spread):

| Market | Normal says Over | Over actually hit | Gamma says Over |
|---|---|---|---|
| Receiving yards | 50.0% | 41.3% | 39.2% |
| Rushing yards | 50.0% | 42.8% | 39.3% |
| Passing yards | 50.0% | 49.8% | 44.8% |

Reading: the Normal curve overstates the Over near the average by about 9 points for receiving
yards and 7 for rushing yards. A gamma curve fixes receiving yards and receptions (negative
binomial). For rushing yards the Normal is too high and the gamma too low by a similar margin, so
neither is right. Passing yards are not skewed and the Normal is fine. The 2026 graded tickets
(337 lines with a consensus price, all lines our model picked) were too few and too selected to
decide anything; every interval there spans zero except four passing lines.

## 2. Published mean (`extract_boards.py`, `production_mean_test.py`)

Every archived props board that records the mean (`docs/props/index.html`, September 9 to
October 20, 2026) was extracted from git history. For each event, player and market, the decision
snapshot is the last board committed before kickoff (median 2 hours 19 minutes before). Results are
nflverse box scores for 2026 Weeks 1–6: 3,595 graded player-markets across 93 games. 86 players who
did not play were excluded (books void those props); 76 who had offensive snaps but no stat line
count as zero.

Bias is the result minus the published mean (positive: the mean was too low):

| Market | Player-games | Bias [95% interval] |
|---|---|---|
| Passing yards | 185 | **+14.4** [+3.7, +24.5]; positive in every week |
| Pass attempts | 185 | **+2.3** [+1.0, +3.6] |
| Completions | 185 | **+1.5** [+0.7, +2.3] |
| Rush attempts | 337 | **+1.2** [+0.8, +1.7] |
| Receiving yards | 1,086 | +0.6 [−1.1, +2.2] |
| Receptions | 1,081 | +0.1 [−0.0, +0.2] |
| Rushing yards | 536 | +0.4 [−1.4, +2.2] |

- Receiving and rushing means sit about 4 and 3.5 yards above the main market line, which is the
  ordinary gap between an average and a median. The Normal curve reads that gap as an Over edge;
  the fitted calibration flattens it.
- The market's implied mean (backed out of the main line and its no-vig price) was more accurate
  than the published mean in 6 of 7 markets; rushing yards was a tie.
- The simple average from section 1 beat the published mean on receiving yards (RMSE 29.3 vs 30.8;
  MSE difference −90 [−133, −47]) and receptions (2.18 vs 2.30), and was ahead but not
  significantly on passing and rushing yards. With production-style recency (half-life 0.76
  games) it was no better than the published mean, and worse on rushing yards; half-lives of 4 to
  10 games all did better on receiving yards and receptions. The production four-game window, with
  61.6% of the recent-form weight on the latest game, is a likely source of the extra noise.
- On the boards that record the adjustment stages (35–43 games), the defensive and home/away
  adjustments made no measurable difference to accuracy; every interval spans zero.

Untested explanation for the passing gap: history averages count partial games (injury exits,
blowouts) as full ones, which the projection trace itself lists as a limitation.

## Limitations

- Section 1 uses simple forecasts; it isolates curve shape, not production accuracy.
- Section 2 covers six weeks of one season; players in one game are correlated, so intervals
  resample whole games. Box-score name matching can miss a player; unmatched rows are excluded.
- The market-implied mean depends on an assumed spread (each market's median published sigma/mu)
  and shape; it is a benchmark, not a forecast we publish.
- No result here is a betting-return or closing-line-value claim.

## Reproduce

Data (retrieved October 20, 2026): nflverse `stats_player_week_<season>.csv.gz` for 2021–2026 from
the `stats_player` release and `snap_counts_2026.csv` from the `snap_counts` release
(`https://github.com/nflverse/nflverse-data/releases`), in one folder. nflverse updates these
files, so later downloads can differ slightly; `--max-week 6` limits 2026 to the weeks graded here.
Boards come from git history up to commit `0cc78dcb` and need a full (not shallow) clone.

```
python extract_boards.py <repo> boards.csv 0cc78dcb
python production_mean_test.py boards.csv <nflverse-dir> --max-week 6
python shape_backtest.py <nflverse-dir> <repo>
python shape_backtest.py <nflverse-dir> <repo> --half-life 0.76 --prior-k 0.5
```

Saved output: `results/`.

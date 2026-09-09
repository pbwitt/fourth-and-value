# NFL totals: how the model differs from professional practice

Measured September 9, 2026, against nflverse closing lines and results. All
figures below are walk-forward: every prediction is trained only on games that
finished before the game being predicted.

## The benchmark nobody clears by accident

The closing total line, 2022–2025 regular season, 1,087 games:

| Metric | Value |
|---|---|
| MAE vs actual total | 10.19 |
| RMSE | 13.03 |
| SD of residual | 13.01 |
| Bias (actual − line) | +0.72 |
| Over hit rate | 48.8% |
| Push rate | 0.7% |

Two things follow. The market is close to unbiased, and an NFL total is
enormously noisy: a residual SD of 13 points means a *perfect* forecast of the
mean would still miss by 10 points on average. Most of what looks like model
error is irreducible variance.

## What the current model does against that benchmark

Ridge on trailing L3/L5 raw EPA, success rate, explosive rate, third-down and
red-zone rates, plus a home indicator, predicting each team's points. 863 games
evaluated walk-forward.

| Approach | MAE |
|---|---|
| Model predicts the total, market ignored | 10.51 |
| Closing line | 10.14 |

The model is **worse than quoting the market line back**. The stronger test is
whether its features explain anything the line missed — predict the residual
`actual − closing_line` instead of the total:

```
corr(predicted residual, actual residual) = +0.021
R² = -0.027
```

That is zero signal. Negative R² means the residual model is worse than
predicting "no disagreement" every time. The features carry no information the
closing line has not already absorbed, which is unsurprising: trailing EPA is
what every book models too.

This is consistent with the season backtest at real prices: 49.2% on 181 bets,
−5.8% ROI, with accuracy *decreasing* as claimed edge grew.

## Even the famous angles are priced

Wind is the most cited totals factor. Outdoor games, 2006–2025, n=3,585:

```
pearson r = -0.073   p < 0.0001
```

Real, and directionally correct — wind suppresses scoring. But it explains about
half a percent of residual variance. Worse, the bucket that looks exploitable
does not replicate:

| Wind | 2006–2015 | 2016–2025 |
|---|---|---|
| 0–10 mph | 47.4% under (n=1353) | 48.4% under (n=1245) |
| 11–15 mph | 51.6% under (n=345) | **58.2% under (n=328)** |
| 16+ mph | 55.0% under (n=180) | 51.5% under (n=134) |

The 58.2% cell is the kind of result that gets published as a system. Its own
out-of-sample counterpart is 51.6%. Slicing a residual five ways will always
produce one impressive cell.

## What professionals do differently

**They anchor to the market.** The opening and closing lines are the single best
available forecast. A sharp model predicts the *residual* against that line and
usually outputs a small number. Building a total from scratch and comparing it
to the market, which is what this pipeline does, treats an inferior forecast as
if it were independent evidence.

**They model a distribution, not a point.** Pricing an over/under requires
`P(total > line)`, which needs a spread as well as a mean. With residual SD ≈
13, a 3-point projection difference is worth roughly `Φ(3/13) − 0.5 ≈ 9`
percentage points of win probability *if the projection is right* — and nothing
at all if it is not. Reporting "model edge +6.6 points" implies a precision the
model does not have and cannot be converted to expected value.

**They decompose scoring rather than regressing on efficiency.** The standard
build is drives per team × points per drive, with pace (seconds per play,
neutral-script tempo) driving drive count and opponent-adjusted efficiency
driving points per drive. Raw EPA is schedule-contaminated; adjustment solves
offense and defense jointly. Red-zone and third-down rates get regressed hard
toward the mean because they are noisy in small samples — L3 of a red-zone rate
is close to pure noise.

**They regress to a prior, especially early.** Week 1 through 4 estimates are
blends of prior season, league average and the market's own number, with
empirical-Bayes weights. Carrying end-of-season form into Week 1 without
shrinkage, as this pipeline now does, ignores roster turnover.

**They grade on closing line value, not win rate.** A season is 272 games; a
5-point win-rate difference is inside the noise band. CLV — did you beat the
number the market closed at — converges far faster and is the standard measure
of whether a process is +EV.

**They look for information the market prices slowly**, not for a better model
of public data: injury and inactive news speed, weather updates against stale
numbers, and derivative markets (team totals, alternates) that move behind the
main line.

## What this means for this site

The line-comparison core is the defensible product, and it is already built: 11
books, exact-line matching, de-vigged prices, best price at the consensus line,
and outlier detection. That is real edge for a bettor and requires no forecast.

Ranked by value:

1. **Never publish a point-estimate edge.** If a model number goes up, publish
   `P(over)` from a distribution and compare it to the de-vigged market
   probability. An edge in points is not an edge.
2. **Log closing lines and grade CLV.** The fetcher now captures provider
   `last_update`; snapshotting it on a schedule turns into both line-movement
   history for readers and the only validation metric that converges in one
   season.
3. **Rebuild the projection market-anchored** — features plus the market line,
   target the residual. It will honestly report near-zero edge until a feature
   with real residual signal appears, which is the correct behavior.
4. **Then, if pursuing genuine signal**, replace trailing EPA with
   opponent-adjusted drive-level rates, and validate against the residual
   benchmark above rather than against a coin flip.

Line movement is the highest-value reader feature that does not require winning
a forecasting contest against the market.

# Game environment and player props — November 1, 2026

Research question: would player prop projections improve if they knew the game
environment, meaning the implied team total and the spread, taken either from the market
or from the site's own game models? This is a backtest, not a validated adjustment.
The one output is a logged-only shadow (`scripts/nfl_spread_shadow.py`), which never
changes published projections, probabilities or picks.

## What each sport uses today

- **NFL:** neither. `make_player_prop_params.py` uses recent game logs, a career baseline,
  opponent yards allowed (±15%), home/away and injury availability.
- **NBA:** the site's own game model, not market lines. Projected pace scales per-minute
  rates, and the projected margin feeds a blowout term in the minutes model.
- **NHL:** neither. The v2.4 player models take no market features. The opponent-adjusted
  kind scales by opponent shots and goals allowed. Market totals and moneylines appear only
  in `coherence.py`, which is research tracking.

## Method

Each player-market gets one mean. The adjusted mean is `mu * (1 + a + b·x)`, fit by OLS of
`actual − mu` on `mu·[1, x]` within each market group. Every week (NHL: ISO week) is
predicted from the other weeks only. The fair comparison is level-only (`a`) against
level + environment (`a + b·x`), so a general level bias can't pass for signal. NFL
probabilities are the raw Normal/Poisson ones at the consensus line, before isotonic
calibration. NHL uses the model's own negative-binomial count distribution. Pushes are
excluded.

- **NFL:** frozen pregame archives for weeks 3–7. That is 1,919 graded player-markets
  across 69 games, with actuals from the nflverse `stats_player` release. Week 7 actuals
  and projections match `review/probability_comparison.csv` exactly.
- **NHL:** the last morning snapshot before puck drop for each game, Oct 6–31. That is
  16,164 player-markets (11,371 two-sided) across 180 games, with results rebuilt from
  the archived NHL API pages.

## Findings

**NFL. Results are in `results/nfl_backtest.txt`.**
- **Team total on its own: no help,** whether from the market or from our model. Our
  totals made rushing worse.
- **Market spread: a consistent effect on passing and receiving.** Underdogs beat
  projections by about 1–1.7% per point on pass attempts, completions, receptions and
  receiving yards (game-clustered t ≈ 1.7–2.6). The slope was positive in all five weeks.
- **Market spread, held out:**
  - Passing log loss 0.7649 → 0.7592 against level-only.
  - Receiving log loss 0.7635 → 0.7609.
  - Volume props (attempts, completions, carries, receptions) 0.7755 → 0.7684.
  - Rushing got worse, 0.7718 → 0.7753.
- **Our projected spread** (from `week_predictions.csv`, correlation 0.62 with the
  market): it helps passing about as much as the market spread does, but it hurts
  receiving and rushing. Overall it is worse than level-only.
- **Context:** projections finished on the correct side of the consensus line in 50.0% of
  props. Raw log loss is 0.767 for the model against 0.691 for the market's no-vig
  probabilities. The spread closes only a small part of that gap.

**NHL. Results are in `results/nhl_backtest.txt`.**
- **Across all two-sided props, no variant moved log loss by more than 0.001.** Current
  model 0.6433, market 0.6380.
- **One diagnostic to watch:** when the market implies more team goals than our team model
  does, players beat their goal and point means (t ≈ 2.0–2.7). That did not carry over to
  held-out weeks.

## Caveats

Five NFL weeks and four NHL weeks are a small sample. Several variants were tried, so
t ≈ 2 is suggestive at best. Rows within a game are correlated. NFL probabilities are
uncalibrated, though which side of the line a projection lands on does not depend on
calibration. The shadow coefficients were fit on weeks 3–7, so only week 8 onward counts.

## Reproduce

```sh
# from this folder; stats_player_week_2026.csv.gz comes from
# https://github.com/nflverse/nflverse-data/releases/tag/stats_player
python nfl_build_dataset.py /path/to/stats_player_week_2026.csv.gz
python nfl_backtest.py
python nhl_backtest.py
```

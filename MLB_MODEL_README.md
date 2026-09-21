# MLB models: implementation and operating notes

First production version: September 21, 2026. This replaces the MLB market-only launch with independent forecasts. **Experimental picks are not a claim of profitable betting.** Market Watch remains a separate comparison of books.

## Public pages

- `/mlb/picks.html`: one qualifying offer per game/player/market, ranked by estimated return.
- `/mlb/props/`: forecasts for strikeouts, pitcher outs, hits, total bases, home runs and RBIs, alongside book comparisons.
- `/mlb/totals/`: full-game moneylines, run lines and totals. Failed model checks leave research forecasts visible but withhold picks.
- `/mlb/validation.html`: dates, Brier scores, reference scores, calibration gaps and eligibility by market.
- `/mlb/methods.html`: public explanation of inputs, distributions, pricing and limitations.

Navigation includes **MLB → Model Picks** and **MLB → Model Results**. Forecast cards show calibrated projected means, unconditional win probability, push probability, conditional fair American odds, estimated return per $100, break-even edge and selected input context. Input context is not a feature-attribution explanation.

## Files and commands

```sh
make mlb_daily PY=.venv/bin/python      # update history, train/cache models, refresh prices and publishable pages
make mlb_models PY=.venv/bin/python     # history + training only; no paid odds or commentary
.venv/bin/python scripts/mlb/train.py --cached-history  # deterministic audit from downloaded observations
.venv/bin/python scripts/mlb/refresh.py # current models + fresh prices/lineups; withholds stale model artifacts
.venv/bin/python scripts/mlb/refresh.py --validate
make mlb_test PY=.venv/bin/python
NODE_PATH=/path/to/node_modules node tests/mlb_browser.cjs
```

| File | Purpose |
| --- | --- |
| `scripts/mlb/model_data.py` | Official completed schedule, per-game boxes, normalization and cache |
| `scripts/mlb/models.py` | Prior-date feature state, baselines, distributions, pushes and game matrix |
| `scripts/mlb/train.py` | Chronological partitions, candidate selection, calibration, audits and artifacts |
| `scripts/mlb/predict.py` | Current native player matching, batting orders, forecasts, pick checks and deduplication |
| `scripts/mlb/refresh.py` | Schedule/odds refresh, model attachment, expiry, snapshot and validation |
| `scripts/mlb/site.py` | Reproducible public pages and readable validation tables |
| `docs/assets/mlb.js` | Filters, forecasts, coverage and client-side pick expiry |
| `tests/test_mlb_models.py` | Leakage, probability arithmetic, lineup, freshness and eligibility regressions |

## Historical observations and leakage protection

The official MLB Stats API supplies the prior year's August 1–November 15 games and current-year March 1 through **yesterday in America/New_York**. Only completed regular-season and postseason games scheduled for nine innings, with at least nine innings reached, untied final scores, and no resumed-game flag qualify. Per-team batting runs must reconcile to the final result, and each side must have a unique starting pitcher.

Normalized game files contain **per-game** batting/pitching stats. The API's `seasonStats` fields are deliberately discarded. The training job fails on missing expected boxes; it does not silently drop failed downloads. Cached boxes are observations as first collected; later official scoring corrections are not automatically re-fetched.

Feature construction groups games by official date. Every game on a date is forecast before any outcome from that date is added to state. `State.past` also enforces `observation_date < forecast_date`. This prevents same-game and same-day/doubleheader contamination. Current-day completed games are intentionally unavailable until the next daily cutoff.

The historical test conditions on the actual starting pitcher and original batting order from the completed box. We do not have archived pregame starter/lineup announcements, so this does **not** measure the effect of late scratches or the accuracy of pregame lineup availability. Live forecasts require current probable starters and, for hitters, a complete published starting order.

## Feature windows

All windows also have a 370-day maximum lookback.

- Team: last 40 games; runs scored/allowed, PA and K/H/HR/BB/TB rates. Bullpen RA/9 is computed from team pitching minus the starter. Bullpen workload is pitches in the previous three calendar days.
- Pitcher: last 15 starts; last-five outs and pitches, batters faced, K/BB/H/HR rates, runs per out, and rest capped at 30 days.
- Batter: last 60 appearances; rates per PA, last 20 starting appearances for PA context, original/published batting slot and a transparent projected-PA input.
- Park: last 100 games at the venue, shrunk by 40 games toward nine combined runs. This is a venue scoring indicator, not an isolated park factor; home-team quality can influence it.
- Matchup: home/away and postseason indicators; own and opponent team and starter features.

Small samples shrink toward fixed league-like priors. Missing players do not receive public league-average or 50% predictions. Teams require ten prior games, each probable starter three prior starts, and hitters 50 prior PA plus a published lineup match.

These models do not explicitly use handedness, weather forecasts, injuries, umpires, lineup-wide weighted batter projections, or tactical pinch-hit/starter-hook predictions. Hitter PA is a feature, not a simulated lineup. Recent workload and the postseason indicator provide partial context but cannot replace those missing inputs.

## Regression, distribution and calibration

Each target compares two candidates on a calibration partition:

1. A fixed, small histogram gradient-boosted regression: 100 iterations, seven leaves, learning rate 0.05, 60-row leaf minimum, L2 regularization 20, no early stopping and random seed 42. Counts use Poisson loss; outs use squared loss.
2. A transparent rolling-rate baseline. For example, pitcher Ks use expected batters faced × own K rate × the square root of opponent K rate / fixed league K rate.

The lower calibration MSE wins. No test results choose the candidate or tune model hyperparameters. In the first run, the rolling K baseline won; the other targets selected boosted regression. This is an empirical selection, not an assumption that a complex estimator must be superior.

Counts use Poisson distributions if overdispersion is negligible, otherwise negative binomial distributions with residual-estimated dispersion. Pitcher outs use a normal CDF evaluated at half-count boundaries. Isotonic calibration maps raw cumulative probabilities to observed cumulative outcomes on the calibration partition. A 2% raw-distribution mixture avoids exact certainty in unobserved tails. Finite supports collect the remaining upper tail at their endpoint; tested betting thresholds remain well below these endpoints.

Displayed means come from the calibrated mass function, rather than the raw regression mean. All team and prop probabilities preserve total mass. The joint team-run distribution is an independent outer product with tied final scores removed and the remainder normalized. This approximation does not model run correlation, starter changes or extra innings explicitly.

At decimal price D:

`EV per $1 = P(win) * (D - 1) - (1 - P(win) - P(push))`

`conditional win probability = P(win) / (1 - P(push))`

Fair American odds and break-even edge use conditional probability. Expected return includes refunded stakes. Market consensus remains independent of the model. No sportsbook price enters the predictive feature vector.

## Chronological checks and first results

At first launch there were **3,178 complete games**, with no missing expected boxes. For the current model:

- Train through July 22, 2026.
- Select candidate, estimate dispersion and calibrate through August 21.
- Test August 22–September 20: 405 distinct games for the game-line tests; player coverage varies.
- Then forecast current games using rolling inputs through September 20. Model weights remain fitted before the calibration/test dates. Thus recent results update features immediately at the next daily cutoff without falsely claiming that a model fitted on the test outcomes was tested independently on those same outcomes.

| Market | Model Brier | Reference Brier | Relative gain |
| --- | ---: | ---: | ---: |
| Pitcher strikeouts | 0.154509 | 0.170886 | +9.6% |
| Pitcher outs | 0.184883 | 0.200632 | +7.8% |
| Batter hits | 0.149956 | 0.152075 | +1.4% |
| Total bases | 0.186141 | 0.188561 | +1.3% |
| Home runs | 0.101886 | 0.103109 | +1.2% |
| RBIs | 0.151132 | 0.152176 | +0.7% |
| Game totals | 0.232490 | 0.231275 | −0.5% |
| Moneylines | 0.248323 | 0.250000 | +0.7% |
| Run lines | 0.229658 | 0.230070 | +0.2% |

These are launch observations, not constants or a guarantee. The live report is authoritative after retraining. **Regular-season totals failed the reference comparison and are research-only.** The other markets passed the initial point-estimate checks, although the gains outside pitcher props are small and statistical significance is unestablished.

The reference is the empirical count-outcome distribution from the training partition, with small smoothing. It is not the market and contains no player/matchup information. Brier tests use fixed thresholds, not archived sportsbook lines. Multiple thresholds and players from one game are correlated; the reported sample count is not a count of independent bets.

The previous postseason is tested separately with an older model trained before that postseason: training ends 21 days before the last prior-year regular-season date, calibration ends on that regular-season date, then playoff outcomes are held out. Launch coverage was 47 games (46 for pitcher props). This is a small stress test of an earlier model, not evidence that the current weights will beat upcoming playoff markets.

Initial checks require better Brier score than the reference, at least 150 regular-season games and calibration ECE ≤6 percentage points. Postseason uses at least 30 games and ECE ≤12 points. The initial design allowed a tiny Brier deterioration; after the first audit this was **tightened** to require an actual improvement, which excludes regular-season totals. Threshold changes are policy decisions, not new independent validation evidence. There is no historical ROI or closing-line-value claim. Timestamped forward snapshots start the record needed for that work.

## Live eligibility and expiry

In addition to market checks and identity/history requirements:

- Game starts within 24 hours; snapshot excludes games starting during its fetch.
- Quote age ≤90 minutes; model input check age ≤90 minutes in the browser. Comparison odds remain available up to 12 hours.
- At least two distinct paired books at the exact line; selected quote also has a valid same-book pair.
- Threshold inside the tested market range; whole-number pushes are priced explicitly.
- Estimated EV ≥3% and conditional probability edge ≥3 percentage points.
- EV above 30% is withheld for manual review.
- Best price at the same side and line; highest-EV qualifying offer per game/player/market. This does not remove correlations between different markets.

Every unavailable forecast or rejected pick carries its reason. A started game, feed error, expired quote or stale model check cannot remain a current Model Pick in the browser. Forecasts may be retained as labeled research while ordinary market quotes are still fresh.

## Automation, artifacts and recovery

The MLB workflow updates at 15:15 and 21:15 UTC and on manual dispatch. It restores raw normalized game history, summary stats and model cache, runs the MLB/shared price tests, updates history, trains if needed, then refreshes odds and published lineups. Cold cache can take several minutes and is allowed 40 minutes. Repeated same-cutoff runs reuse an artifact only if the model source signature and sklearn version match. Daily cutoffs advance the partitions and features automatically.

Training failure explicitly sets `MLB_TRAINING_FAILED=1` for refresh, suppressing forecasts even if a same-date artifact remains. Fresh market comparisons can still publish while the workflow fails visibly. A missing/stale/version-mismatched artifact also withholds forecasts. Odds failures hide offers. Publishing validation errors prevent a bad snapshot from replacing the public feed.

- `data/mlb/model_data/<game-id>.json`: normalized game observations.
- `data/mlb/model_data/manifest.json`: expected IDs, cutoff and download failures.
- `data/mlb/models/current.joblib`: locally produced model weights, calibration, reference distributions, feature state and report. Never load arbitrary third-party joblib files.
- `data/mlb/models/training_inputs.json.gz`: exact normalized observations used in the audit; SHA-256 in the public report.
- `docs/mlb/data/validation.json`: public dates, metrics, model choices, feature names, thresholds, source version and input hash.
- `docs/mlb/data/latest.json`: current snapshot with per-quote forecasts, inputs, eligibility, model version and odds timestamps.
- `data/mlb/snapshots/<UTC>.json`: timestamped forward forecasts and decisions.

GitHub retains each run's models, compressed training inputs, manifest, summary stats and snapshots for 90 days. The Actions cache accelerates the next run; it is not the permanent audit store. Preserve/download artifacts if a longer record is required. Public snapshots/reports are also committed in repository history. No OpenAI calls or paid commentary are part of this workflow; only the existing odds API usage continues.

For failures, inspect the workflow and `model_summary.error` / `unavailable_reasons` first. To rebuild from cached observations after code changes, run `train.py --cached-history`, then `refresh.py`. To recover missing historical games, run `train.py` normally. Never force a model probability or pick flag into the public feed to bypass a failing check.

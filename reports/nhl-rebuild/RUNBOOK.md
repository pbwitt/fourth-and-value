# Build, operate, review and roll back NHL v2

No merge, workflow dispatch to production or deployment was performed during this rebuild.
The draft PR is the approval boundary for publishing this change.

## Reproduce from a clean checkout, without API access

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-nhl.txt
.venv/bin/python scripts/nhl/v2/restore.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/nhl/v2/evaluate.py
.venv/bin/python scripts/nhl/v2/report.py
.venv/bin/python -m unittest discover -s tests -p 'test_nhl*.py'
.venv/bin/python scripts/nhl/refresh.py --offline
.venv/bin/python scripts/site_notices.py --scope nhl
```

The restore command validates source payload hashes and normalized history hashes. It
never extracts arbitrary archive paths or executes downloaded code. Evaluation writes
models, exact windows/metrics, locked selection, final-test predictions and grading
summaries. New ingestion/creation times and gzip container bytes may change; numerical
predictions are deterministic for a fixed source archive and dependency stack. A cached
feature file is invalidated when either source manifests or feature-builder code changes.

To fetch a new historical reconstruction, run `python scripts/nhl/v2/data.py`; it uses only
public NHL endpoints, no keys, subscriptions or MoneyPuck downloads. Keep the old raw
archive and compare hashes because official records can be revised. Do not relabel a
new reconstruction as a replay of original publication-time data.

## Daily inference (after approval and merge)

`make nhl_daily` / `scripts/daily_nhl_refresh.sh` remain the supported entrypoints. Only
the existing `NHL_ODDS_API_KEY` or `ODDS_API_KEY` environment secret is needed. `.env` is
ignored; request URLs containing credentials are never logged or archived. The current
season is fetched from NHL statistics and cached; completed seasons come from the
versioned compressed history. Daily inference does not train models. Artifacts older than 400 days are withheld until a reviewed annual retraining.

The workflow gates UTC schedules to 10:30 and 16:30 America/New_York and runs only on
main. The PR workflow is read-only and cannot deploy. GitHub cron can run late; a run
outside the intended hour is skipped and the 24-hour freshness check detects missed
updates. Manual dispatch remains an explicit later-update option. Do not dispatch the
production workflow for branch review.

Existing pages and JSON remain under `docs/nhl/`. Predictions within 48 hours include
independent/market/final probabilities, pushes, fair/minimum prices, EV, scenarios,
model and freshness metadata, evidence-based feature notes and review status. Inputs,
prices and forecasts are durably committed under `artifacts/nhl/`, with Actions artifacts
as a secondary 90-day diagnostic copy. No pick quota or wagering integration exists.

## Analyst review and prospective grading

Prepare a JSON evidence record using the fields in `scripts/nhl/v2/review.py`, including
the exact `offer_id` and `forecast_id` from a saved forecast. Then:

```bash
.venv/bin/python scripts/nhl/v2/review.py /path/to/review.json
.venv/bin/python scripts/nhl/v2/grading.py
```

Reviews append to `artifacts/nhl/reviews.jsonl`; commit reviewed records through the normal
review workflow. Source publication must precede review, and both must precede application.
An override requires an explicit method and a double-counting explanation. Re-running
the model creates a new forecast identity; old reviews never silently apply to new prices.
Grading is local and does not update Supabase or the legacy saved-bet ledger. Inspect
unresolved participation/rule cases instead of filling them with zero.

`grading.py` scores forecast outcomes and probability metrics by market and morning/later
session. A betting simulation requires an explicitly frozen selected-offer ledger; viewing
an offer is not a bet. Helpers implement one-unit staking, one offer per game, game-cluster
uncertainty, drawdown, odds distribution and 0.05-decimal worse-execution sensitivity.
CLV requires the same contract and an actually observed paired quote within 30 minutes
before start; changed lines remain incomparable. The current twice-daily schedule does
not guarantee a closing snapshot. Manual later captures or a separately approved bounded
collector are needed before claiming systematic CLV measurement.

## Checks and monitoring

- Run `make nhl_test`; shared NBA math and site-notice tests are also in CI.
- `python scripts/validate_nhl_freshness.py` checks the actual content timestamps,
  season/game type, probability mass and model provenance.
- Browser: `npm install --no-save --package-lock=false playwright@1.55.1`,
  `npx playwright install chromium`, then `NHL_BUNDLED_BROWSER=1 node tests/nhl_browser.cjs`.
  Covers all five routes, five widths, filtering and valid/missing/stale/failure/empty states.
- An odds failure publishes an explicit unavailable state, then fails the workflow.
  A model failure preserves fresh market comparisons, withholds model values, publishes
  the reason, and marks the workflow failed. Stale models cannot appear current.
- Monitor model coverage, unknown identities, unverified settlement books, feed age,
  API quota, archive size and unresolved grades. No fixed coverage threshold promotes
  forecasts into recommendations. Monthly review should compare rolling calibration
  and count log scores to the frozen baseline using whole-game clusters.

## Resources and retention

Observed evaluation data: 5,248 games, 188,883 skater appearances, 44 used raw API pages.
The normalized cold-start history is approximately 4.5 MB compressed; raw source archive
approximately 7 MB; final predictions approximately 13 MB. A first evaluation takes a
few minutes on this development machine, with feature construction dominant; allow
1 GB RAM and 10 minutes on CI-class hardware as a planning estimate. Daily inference
uses the small fitted artifact and bounded current-season requests; allow the existing
15-minute job limit. No GPU or new paid service is required.

Source pages are deduplicated, but daily/monthly source revisions and odds increase Git
storage. Check size monthly and migrate to an approved immutable store before 500 MB;
do not delete evidence or purchase storage automatically. Historical download and
training are manual/offline operations, not scheduled in the market-refresh job.

## Deployment approval and rollback

Review the PR, evaluation limitations and browser evidence. Approval to merge is also a
decision to activate the modified existing main-branch refresh on its next schedule;
no separate model promotion is implied. All recommendations remain disabled.

For a model-only problem, remove or quarantine the model artifact in a reviewed change:
inference then retains current market comparisons and explicitly withholds forecasts.
For a pipeline/site regression, revert this PR through a reviewed commit and rerun the
prior market-only refresh after approval. Preserve `artifacts/nhl/` and previous model
manifests outside the rollback diff so every published decision remains reproducible.
Never restore an old public snapshot as if it were fresh, and never re-enable the retired
leaking training scripts as a rollback.

## Fixed historical market-price diagnostic

Historical access was verified on the existing plan. Twenty-one fixed morning dates
(the 15th of October–April in three seasons) were downloaded for 630 existing credits,
plus a 10-credit entitlement probe. No upgrade was requested. The archive is committed
under `artifacts/nhl/historical-odds/`; rerunning the evaluation uses it without a key:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/nhl/v2/market_evaluate.py
```

Run `restore.py` and `evaluate.py` first to reconstruct the feature cache if absent.
Only `--download` permits new API calls, and the script skips saved dates, caps one run
at 630 credits and preserves a 2,000-credit live-operation reserve. It never buys access.
`HISTORICAL_MARKETS.md` separates timestamped quote diagnostics and shadow simulations
from realized execution and explains the small sample and historical-rule assumptions.

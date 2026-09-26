# Timestamped historical game-price diagnostic

**No betting edge is established.** Historical access is included in the existing authorized odds plan. After the core forecast evaluation, we acquired a fixed monthly sample for 630 credits, plus a separate 10-credit access probe. This supersedes the earlier finding that no usable historical price sample was available locally.

## Fixed scope and assumptions

The 15th of every month, October–April, in 2023–24, 2024–25 and 2025–26, at 10:30 America/New_York: 21 decision snapshots covering 130 regular-season games. Dates were fixed before downloading; no dates were chosen by returns. Raw provider timestamps, query times, ingestion times and checksums are archived. Only prices timestamped at or before the decision, for games not started, are eligible.

The hockey algorithms, fitted-window rules and shadow thresholds were unchanged. Each season uses parameters trained on earlier seasons. Development-season results are retrospective diagnostics of the eventually selected specification; only 2025–26 is the final-season diagnostic. Historical house-rule versions, account limits and actual execution are unverified. These are quoted-price shadow simulations, not realized betting returns or a certified executable backtest.

## Paired probability benchmark

One home/Over observation per game, market and exact line; freshest mapped quote with leave-offered-book-out paired consensus. Push results are excluded from the conditional probability score. Multiple lines remain grouped by game in uncertainty calculations. Differences below are model minus market log loss; lower is better.

| Season | Market | Observations / games | Hockey log loss | Market log loss | Difference 95% interval |
|---|---|---:|---:|---:|---|
| 20232024 | h2h | 44 / 44 | 0.67437 | 0.69635 | [-0.06105, 0.01479] |
| 20232024 | spreads | 44 / 44 | 0.71863 | 0.70646 | [-0.03972, 0.06392] |
| 20232024 | totals | 41 / 40 | 0.69001 | 0.68718 | [-0.02135, 0.02364] |
| 20242025 | h2h | 42 / 42 | 0.65232 | 0.65666 | [-0.04683, 0.03367] |
| 20242025 | spreads | 42 / 42 | 0.67764 | 0.66999 | [-0.04990, 0.07026] |
| 20242025 | totals | 45 / 40 | 0.70312 | 0.68497 | [-0.00266, 0.04453] |
| 20252026 | h2h | 44 / 44 | 0.70420 | 0.70547 | [-0.03466, 0.03151] |
| 20252026 | spreads | 44 / 44 | 0.58389 | 0.57256 | [-0.02814, 0.05739] |
| 20252026 | totals | 53 / 43 | 0.68079 | 0.68952 | [-0.02656, 0.01272] |

Every paired interval includes zero. The sample does not support a claim that the independent model beats the market. Brier scores and reliability bins are included in `historical-market-evaluation.json`. The 86 development games are below the predeclared 500-game blend-training floor, so no market blend is fitted.

## Fixed shadow price policy

At most four one-unit offers/day and one offer/game, requiring a mapped standard settlement profile, quote age ≤30 minutes, independent EV ≥2%, positive worst-scenario log-growth and an offered price at least the minimum from the ±10% rate scenarios. Actual win/loss/push outcomes only grade the frozen selections. Historical analyst approval is not fabricated; these selections are explicitly simulation-only and never enter the production recommendations list.

| Season | Count / turnover | Net units | ROI | Max drawdown | ROI 95% interval | ROI at −0.05 decimal execution |
|---|---:|---:|---:|---:|---|---:|
| 20232024 | 6 / 6 | 3.2407 | 54.01% | 1.0000 | [-41.98%, 146.75%] | 50.68% |
| 20242025 | 8 / 8 | 0.9192 | 11.49% | 3.0000 | [-54.17%, 73.51%] | 8.36% |
| 20252026 | 3 / 3 | -1.5238 | -50.79% | 1.5238 | [-100.00%, 47.62%] | -52.46% |

The final season produces only three shadow selections and loses 1.5238 units. The sample is far too small for a stable profitability estimate; the positive development returns are not a discovered strategy. No threshold was changed after these results. Odds quantiles, selection identities, raw offers and settled outcomes are retained in JSON/gzip outputs. Confidence intervals resample game clusters.

## What remains unavailable

Full daily historical price coverage, player-price histories, contemporaneous analyst evidence, historical rule versions and confirmed execution are not supplied by this sample. Closing quotes and afternoon snapshots were not acquired, so CLV and early-versus-later comparisons remain unavailable. The access probe proves entitlement, not coverage of every requested feature. A larger backfill needs an explicit quota budget; no subscription upgrade is required or was attempted.

Reproduce without network access after restoring/building the feature cache: `python scripts/nhl/v2/market_evaluate.py`. Raw snapshots live in `artifacts/nhl/historical-odds/`. This diagnostic does not enable production picks.

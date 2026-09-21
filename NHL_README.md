# NHL operations — 2026–27 relaunch

## Public pages

- `/nhl/`: overview and official upcoming regular-season schedule
- `/nhl/props/`: shots on goal, goals, assists and points
- `/nhl/totals/`: totals, puck lines and moneylines
- `/nhl/top.html`: Market Watch (same-line price differences, not model best bets)
- `/nhl/methods.html`: current methods and limitations

The previous December 29, 2025 cards have been removed from the active pages. Prior outputs and model files remain in local `data/nhl/` and Git history for research. No historical bet records are deleted or regraded by this refresh.

## Run

```sh
make nhl_daily PY=.venv/bin/python
make nhl_pages PY=.venv/bin/python   # offline; keeps original snapshot timestamps
make nhl_test PY=.venv/bin/python
.venv/bin/python scripts/validate_nhl_freshness.py
gh workflow run nhl-daily.yml --ref main
```

`make nhl_daily_pub` dispatches the GitHub workflow. `nhl_totals_daily` is a compatibility alias for the new refresh. Other legacy Makefile model shortcuts stop with an explanation rather than overwrite the current pages. The old shell wrappers build locally only; GitHub owns publishing. `.env` is read by python-dotenv rather than exported through shell word splitting. No paid commentary is run.

## Automation and credentials

`.github/workflows/nhl-daily.yml` runs at 14:30 and 20:30 UTC (10:30 AM / 4:30 PM Eastern daylight time; one hour earlier in winter). It uses `NHL_ODDS_API_KEY`, falling back to `ODDS_API_KEY`. The dedicated secret uses the verified local odds credential, without modifying other sports' secrets.

The workflow checks out current main after any queued run, tests NHL pricing/season guards, restores history cache, refreshes data, archives inputs for 90 days, commits only `docs/nhl/`, and explicitly requests a Pages rebuild. Feed failure publishes an error state and fails the job, preserving the last successful snapshot timestamp. Git conflicts stop publishing and retain the audit artifact.

## Data and season behavior

The official `api-web.nhle.com/v1/schedule/{date}` feed supplies a 45-day schedule. Only current-season `gameType=2`, future/pre-game, normally scheduled games qualify. The season rolls over in September, accommodating early opening nights. Odds events must match both team names and start within ten minutes. Ambiguous or unmatched events are excluded; the snapshot records their count. A schedule failure stops market publishing rather than admitting unverified preseason fixtures.

The Odds API uses `icehockey_nhl`, US books, and h2h/spreads/totals once per refresh. Four prop markets are requested per event only within 48 hours of puck drop, up to 16 events. Empty regular-season markets are normal and do not resurrect old odds. Quotes expire after 24 hours; the browser also hides started games and reloads boards every five minutes.

NHL Stats `/stats/rest/en/skater/summary` and `/team/summary` provide paginated regular-season summaries for current and prior seasons. Queries exclude today's games, use stable ID ordering, verify pagination completeness, and replace the cache only after all requests succeed. The cache refreshes after 12 hours. Historical references are withheld on a stats failure or when fetched more than 36 hours ago. New-season players with fewer than 20 games use the prior season only if it has at least 20 games; otherwise no reference is shown. The source season and sample size are visible.

- Public: `docs/nhl/data/latest.json`
- History cache: `data/nhl/history/current.json`
- Audit snapshots: `data/nhl/snapshots/<UTC>.json`
- Pipeline: `scripts/nhl/refresh.py`
- Pages: `scripts/nhl/site.py`, `docs/assets/nhl.js`
- Shared tested price normalization: `scripts/nba/pipeline.py` helpers, configured explicitly for NHL

## Model status — do not mistake references for picks

The old totals model selected same-game scoring, shots, save percentage and other box-score fields as predictors. Opponent and home/away summaries were computed using the full dataset. These inputs leak information that was unavailable before puck drop. The former props edge path could silently use market consensus when no player model was available; unknown players could receive 50%. Neither path is used by the current public refresh.

Player references use season per-game means and an uncalibrated Poisson distribution. Whole-number lines explicitly separate over, under and push. Shots may be overdispersed. Totals references average team scoring with opponent conceding rates, using the same reference season for both teams. No game-total win probability is published because overtime/shootout and settlement treatment need separate validation. All `model_probability` fields remain null.

To restore model picks: rebuild strictly pregame features, apply chronological train/validation splits, verify current player IDs and roles, incorporate ice time/power-play deployment and confirmed goalies, test calibration against outcomes, and evaluate returns against timestamped odds with pushes and settlement handled correctly. Market Watch needs three OTHER paired books and is explicitly a research signal; consensus is not ground truth and line differences are not arbitrage.

## Relaunch verification

On September 21, 2026, the initial fresh pull returned 250 upcoming regular-season games, 756 price quotes across ten books, and 940 prior-season player summaries plus all 32 teams. These are launch observations, not future guarantees. Player props were not yet posted for the regular-season window. Tests cover September rollover, preseason/postponed exclusion, event matching, exact-line de-vigging, Poisson pushes, season/sample thresholds, stale/ambiguous history, empty markets and paginated official data. Browser checks cover desktop/mobile navigation, filters, best prices and stale/error suppression; NBA regression tests cover the shared helpers.

```sh
NODE_PATH=/path/to/playwright/node_modules node tests/nhl_browser.cjs
```

Sources: https://www.nhl.com/schedule , https://www.nhl.com/stats/ , https://the-odds-api.com/liveapi/guides/v4/ .

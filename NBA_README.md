# NBA operations

NBA is a separate pipeline; it does not change NFL/NHL predictions or run paid commentary.

## Pages

- `/nba/`: overview and upcoming provider events
- `/nba/props/`: points, rebounds, assists, threes, PRA, PR, PA, RA, blocks, steals, turnovers
- `/nba/totals/`: totals, spreads and moneylines
- `/nba/top.html`: **Market Watch**, using independent same-line book comparisons; not model best bets
- `/nba/methods.html`: assumptions, availability and model limitations

The official 2026–27 season starts October 20. See https://www.nba.com/news/2026-27-nba-regular-season-schedule . We do not synthesize unavailable props or insert demonstration bets on the public pages.

## Run locally

Requires the existing Python 3.11 environment / requirements.txt and `ODDS_API_KEY` in `.env` or the environment. The key never enters the browser or log output.

```sh
make nba_daily PY=.venv/bin/python
make nba_pages PY=.venv/bin/python
make nba_test PY=.venv/bin/python
.venv/bin/python scripts/nba/pipeline.py --history-season 2025-26 --offline
```

`nba_pages` is a network-free rebuild from the saved snapshot. `nba_daily` calls the event list, fetches game lines for the next 45 days, and fetches 11 prop markets for at most 16 games within the next 48 hours. All odds requests use one region (`us`). There are no expensive historical-odds requests. Published quota metadata records remaining provider credits; exact billed usage depends on returned markets.

Source documentation: https://the-odds-api.com/liveapi/guides/v4/ and https://the-odds-api.com/sports-odds-data/betting-markets.html . The NBA event list is not the full season schedule. Futures are excluded.

## Automation

`.github/workflows/nba-daily.yml` runs at 15:00 and 21:00 UTC and supports manual dispatch. It uses `NBA_ODDS_API_KEY`, falling back to the repository `ODDS_API_KEY` secret. The NBA-specific secret was configured from the verified local credential after the shared workflow credential returned HTTP 401 during launch testing. Stats calls are free and optional on manual runs; failure does not invent baselines or prevent odds publishing. No OpenAI secret is used.

The workflow tests pricing behavior, restores cached regular-season history, attempts prior/current-season game logs, refreshes odds, saves 90-day artifacts, commits only `docs/nba/`, and pushes with rebase/retry. It explicitly requests a GitHub Pages rebuild after publishing because a bot-token commit does not trigger one. A failed odds fetch publishes an error state and fails the job, preserving the last success time and saved evidence. Check Actions for runner-specific feed access problems.

## Artifacts and correctness

- Public feed: `docs/nba/data/latest.json`
- Local audit snapshots: `data/nba/snapshots/YYYYMMDDTHHMMSSZ.json` (gitignored)
- Historical NBA Stats logs: `data/nba/history/YYYY-YY.json` (gitignored, cached in Actions)
- Reusable implementation: `scripts/nba/pipeline.py`, `scripts/nba/site.py`, `docs/assets/nba.js`

De-vig needs both outcomes from the same book/event/player/market/line. Spread pairing normalizes the away handicap to the home sign. Conflicting duplicate quotes are withheld. Consensus uses one vote per paired book; Market Watch excludes the evaluated book and requires three other paired books. Positive consensus price gaps are research signals, not calibrated expected profit. Quotes older than 24 hours and started games are hidden. The browser repeats that gate and refreshes open boards every five minutes.

## Prediction readiness

The initial launch includes live market infrastructure and an optional historical reference layer. **It does not claim to have a validated NBA player or game prediction model.** NBA Stats denied/timed out locally and on GitHub runners during setup; the loader and scheduled retry are in place. Never report statistics as loaded until a history file actually exists.

Player baselines require 20–30 previous regular-season appearances and show the source's last game date. They use player IDs for deduplication and suppress ambiguous name matches. The mean and smoothed historical hit rate do not adjust minutes, injuries, role changes, opponent or rest; they cannot qualify model picks.

Before activating NBA model picks:

1. Obtain and validate historical logs, schedule, player/roster IDs and current availability.
2. Model minutes separately from per-minute points/rebounds/assists; preserve joint outcomes for combinations.
3. For game totals, model pace and offensive/defensive efficiency with home/rest/travel inputs.
4. Use chronological evaluation with only inputs available before tipoff, separate preseason and postseason, and evaluate calibration against archived contemporaneous odds, including pushes.
5. Publish model and data versions, coverage, last input dates and unsupported-player reasons. Add injury adjustments only once availability inputs are verified.

No NBA bet grading, injury automation, paid summaries or model best-bet claims are silently enabled by this launch.

## Verification

`tests/test_nba.py` checks cross-line/event isolation, opposite spread signs, duplicates, leave-one-book-out comparisons, chronology/DNP/push behavior, empty slates, failures and stale prices. `tests/nba_browser.cjs` checks navigation/layout, filters, best prices, stale data and failure suppression using an in-memory fixture (never shipped).

```sh
NODE_PATH=/path/to/node_modules node tests/nba_browser.cjs
```

Use the existing local Playwright installation if needed; its location has no effect on which repository is modified.

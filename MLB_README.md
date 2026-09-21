# MLB operations and postseason readiness

## Public section

- `/mlb/`: schedule, probable pitchers and regular/postseason filter
- `/mlb/props/`: pitcher strikeouts/outs; batter hits, total bases, home runs and RBIs
- `/mlb/totals/`: full-game moneyline, run line and total
- `/mlb/top.html`: Market Watch; same-line disagreements, not model best bets
- `/mlb/methods.html`: methods, settlement caveats and prediction limitations

## Commands

```sh
make mlb_daily PY=.venv/bin/python
make mlb_pages PY=.venv/bin/python   # offline rebuild; original timestamps retained
make mlb_test PY=.venv/bin/python
.venv/bin/python scripts/mlb/refresh.py --validate
gh workflow run mlb-daily.yml --ref main
```

The local refresh reads `.env`; never print keys or request URLs containing keys. No OpenAI calls or paid commentary run automatically.

## Schedule, postseason and identity

The official MLB Stats API `/api/v1/schedule` provides the next 45 days with `probablePitcher,team` hydration. Accepted MLB game types: R regular season, F Wild Card, D Division Series, L Championship Series, W World Series. Spring training, exhibitions, All-Star games, completed/live games, delayed starts, postponed/suspended/resumed games and TBD start times do not qualify. Unconfirmed participants are excluded. Conditional postseason games retain their if-necessary label.

An odds event must match both team names and a start time within ten minutes. MLB game IDs and start times separate doubleheaders; duplicate or ambiguous provider matches are withheld. Athletics aliases are normalized explicitly. New postseason fixtures appear automatically when teams/times and provider odds are available. A failed schedule call stops market publishing rather than admitting unverified events.

Probable starters are provisional, not confirmed lineups. Both names appear on schedule and quote cards. Batting lineups are not verified. There are no automated weather, injury, bullpen or workload adjustments.

## Statistics and model status

MLB regular-season `byDateRange` statistics cover January 1 through yesterday in America/New_York; current-day and postseason results do not contaminate that context. Hitting and pitching are separate groups. The loader requires complete reported coverage and writes cache atomically after both succeed. Descriptive context is matched by normalized player name, retaining the official player ID. Ambiguous names or multiple team splits are withheld. Pitcher identification is also checked against the probable-pitcher IDs where context is available.

Batter context shows PA, AVG, OPS and counting stats. Pitcher context shows IP, starts, K, K/9 and ERA. Rates use outs, not decimal innings: 5.2 IP = 17 outs. No league-average or 50% fallback is used. Statistics refresh after 12 hours or a new Eastern cutoff date; public context expires after 36 hours and is hidden on a stats-feed failure.

**This launch has no validated MLB prediction model.** `model_probability` is always null. Regular-season stats are historical context, not postseason workload forecasts. Before model picks: verify batting order, expected plate appearances, pitcher handedness and workload, opponent and park factors, weather, bullpen availability, and postseason deployment; perform chronological outcome/odds validation and calibration. Whole-number pushes, participation, listed-pitcher and postponement rules must be handled by book/market. First-five markets are not mixed with full-game markets.

## Odds and budgets

The Odds API sport is `baseball_mlb`; one US region is used. Events are checked first, then h2h/spreads/totals in one request. Six props markets are requested per game only within 24 hours of first pitch, capped at 20 events per run. The snapshot and UI report any cap-related skipped games. Exact quota cost depends on returned markets; no paid historical-odds endpoint is used.

Quotes expire after 12 hours, or at first pitch. Boards refresh open tabs every five minutes. Games that begin during the fetch are removed before publishing. De-vigging requires paired same-book/event/player/line outcomes; best price compares the exact line and side. Market Watch requires three OTHER paired books and excludes the evaluated book. Market probability comparisons on integer lines are conditional on no push; they are not guaranteed profit estimates.

## Automation

`.github/workflows/mlb-daily.yml` runs at 15:15 and 21:15 UTC (11:15 AM and 5:15 PM Eastern daylight time; one hour earlier in winter), plus manual dispatch. It uses a dedicated `MLB_ODDS_API_KEY`, falling back to `ODDS_API_KEY`. The dedicated secret is configured from the verified local credential without changing NFL/NHL/NBA settings.

The job checks out current main after queued runs, runs MLB and shared NBA math tests, restores the statistics cache, refreshes/validates, uploads 90-day audit artifacts, commits only `docs/mlb/`, and requests a GitHub Pages rebuild. A feed failure publishes an error state and fails the job; a validation failure blocks the bad snapshot. Conflicts stop with the audit artifact retained.

- Public feed: `docs/mlb/data/latest.json`
- Statistics cache: `data/mlb/history/current.json`
- Timestamped snapshots: `data/mlb/snapshots/<UTC>.json`
- Implementation: `scripts/mlb/refresh.py`, `scripts/mlb/site.py`, `docs/assets/mlb.js`
- Shared price helpers: `scripts/nba/pipeline.py`, with explicit MLB market/sport arguments

## Verification

Initial September 21 launch pull: 1,330 quotes across eleven sportsbooks; 747 hitting and 861 pitching records; all six prop markets populated. These counts are observations, not promises about future coverage. Tests cover every playoff round, excluded statuses, doubleheaders, team aliases, duplicates, baseball innings arithmetic, missing/ambiguous stats, dated regular-season queries, incomplete responses, empty slates, quote expiry and started games. Browser checks cover navigation/mobile layout, postseason filtering, best-book selection and stale/error suppression.

```sh
NODE_PATH=/path/to/node_modules node tests/mlb_browser.cjs
```

Sources: https://www.mlb.com/schedule , https://www.mlb.com/postseason , https://www.mlb.com/stats/ , https://the-odds-api.com/sports/mlb-odds.html , https://the-odds-api.com/sports-odds-data/betting-markets.html .

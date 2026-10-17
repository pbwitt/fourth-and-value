# NBA launch plan

Working plan for bringing NBA to parity with NFL, NHL and MLB. Written October 17, 2026
(the 2026–27 regular season starts Tuesday, October 20). Read `AGENTS.md` and
`NBA_README.md` first; this file says what is done, what is next and the constraints
that apply.

## Done (PRs #145 and #146)

- **Bet Tracker** grades NBA from ESPN box scores. A player listed as not playing is a
  push (refund); a player with no readable stats stays pending instead of being graded
  on zeros. Closing-line value already handled `basketball_nba`.
- **Live Odds** (`/live/`, editors) covers NBA before and during play; the `live-odds`
  Edge Function (Supabase, version 4) takes `sport: 'NBA'`.
- **Market Analytics** (`/markets/?sport=nba`) grades each game's last pregame snapshot
  in `artifacts/nba/runs/` against ESPN finals (`scripts/build_market_results.py
  --sport nba`, run by `nba-daily.yml`).
- **Credits:** the NBA refresh is a core Odds API job; `CORE_DAILY` is 850
  (`scripts/odds_budget.py`, `REFRESH_SCHEDULE.md`).

## Check on opening night (games Tuesday evening ET)

- The tracker's ESPN parser (`docs/tracking/live-feeds.js`, `espnPlayers`) has never
  seen a real NBA box score; it was tested on fixtures only. After the first games go
  final, confirm a tracked NBA prop settles with the right stat value (not pending as
  "player not found"). If NBA bets all stay pending, compare a real
  `.../basketball/nba/summary?event=<id>` response with `ESPN_STATS` (stat keys are
  looked up as `<category name>.<key>`, and NBA categories are expected to be unnamed).
- Market Analytics first grades Tuesday's games at the Wednesday 11 AM ET NBA refresh.
- Credits: at about 9,300 left on October 17 and about 800–900 a day across all sports,
  the balance runs out around October 28 unless credits are added. Optional jobs
  (Pinnacle snapshot, editorial prices) are paused until the November 1 reset.

## Constraints

- **Data source: ESPN.** `stats.nba.com` refuses GitHub runners (and timed out locally
  during setup), and the daily refresh runs on GitHub, so production must not depend
  on it. ESPN's site API is free and reachable from runners:
  `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD&limit=100`
  and `.../summary?event=<id>`. `scripts/nba_weekly_results.py` already parses ESPN
  summaries (`_players`, labels-based, never substitutes zero for a missing stat).
- **Python 3.11.** `requirements.txt` pins pandas 1.5.3, which does not build on 3.12+.
- **No paid historical odds** beyond small, approved samples: one run may not spend more
  than 2,000 credits without the owner's approval, and the balance is tight this month.
- **Commit what production needs.** The daily refresh on GitHub must find the history
  and trained model in the repository; nothing may depend on a local machine.
- `AGENTS.md`: Eastern Time everywhere reader-facing; SEO rules for any new page;
  any change to Top Picks eligibility, research stages or freshness updates
  `scripts/research/daily_process.html`, `ANALYST_RESEARCH.md` and the generated
  `docs/research/daily-process.html` in the same change.

## Phase 1: data foundation (next)

1. **ESPN history backfill** (new script, e.g. `scripts/nba/espn_history.py`):
   - Walk the scoreboard by Eastern date for 2023–24, 2024–25 and 2025–26. Keep regular
     season (`season.type == 2`) and postseason (3) separately; drop preseason (1).
   - For each final game, fetch the summary once and parse player rows with
     `nba_weekly_results._players` (or shared code) plus team totals.
   - Pace requests politely (about 3,700 games) and cache raw responses locally
     (gitignored). Commit only compact parsed data, e.g. one gzip JSON per season in
     `artifacts/nba/history/` with a manifest (source URLs, fetched_at, row counts,
     sha256), the way `artifacts/nhl/` keeps its training sources.
2. **Switch `scripts/nba/pipeline.py` history to ESPN.** `add_baselines` and
   `player_context.nba_defense` read NBA Stats column names; emit the same fields
   from ESPN so they keep working: player rows `PLAYER_ID` (ESPN athlete id),
   `PLAYER_NAME`, `GAME_ID`, `GAME_DATE`, `MATCHUP`, `TEAM_NAME`, `MIN`, `PTS`, `REB`,
   `AST`, `FG3M`, `BLK`, `STL`, `TOV`; team rows `GAME_ID`, `GAME_DATE`, `TEAM_NAME`,
   `TEAM_ABBREVIATION` plus the totals `nba_defense` uses. Positions come from
   `athlete.position.abbreviation`. `nba-daily.yml` then adds yesterday's finals
   incrementally (reuse `artifacts/nba/results/`). This restores player baselines and
   context panels on `/nba/props/`; today `history_players` is 0.
3. **Availability:** ESPN's NBA injuries endpoint and the NBA's official injury report.
   Wire NBA into `docs/assets/injury-context.js` (its source allowlist has CBS
   `/mlb/injuries/` and `/nhl/injuries/` only) once the source is verified.

## Phase 2: NBA model v1 (shadow mode)

Follow the NHL v2 pattern (`NHL_README.md`, `reports/nhl-rebuild/MODEL_CARD.md`):
lagged inputs only, training separate from daily inference, versioned artifacts, a
dated evaluation report (e.g. `reports/nba-v1/`).

- **Game model:** possessions (pace) × points per possession from lagged offensive and
  defensive efficiency, with home court, rest, back-to-backs and travel; one joint
  score distribution prices moneyline, spread and total, with push mass on whole lines.
- **Player model:** projected minutes (recency, role, starter status, rest, blowout risk
  from the spread) × per-minute rates for points, rebounds, assists, threes, blocks,
  steals and turnovers, with overdispersion; combos (PRA, P+R, P+A, R+A) simulated
  jointly, never as independent marginals.
- **Evaluation:** chronological; train on earlier seasons, test on 2025–26; count log
  loss, Brier and calibration against simple baselines. Price-based claims wait for
  graded regular-season odds (`artifacts/nba/runs/` starts with the opener).
- **Publish as experimental** on `/nba/props/` and `/nba/totals/` with model and data
  versions, coverage and unsupported-player reasons; grade prospectively.

## Phase 3: Top Picks and research (after graded evidence)

- `docs/assets/briefing-picks.js`: add NBA to `collect`, `researchStatus` and the summary
  loops, and give it a `BLEND` weight. `comparison()` reads `final_probability`,
  `push_probability`, `market_probability` and `other_books` for non-NFL/MLB sports;
  NBA rows currently carry MLB-style `other_book_probability`, so align the fields.
- Research: the sport enums in `scripts/research_discovery.py` (`SCHEMA`, `catalog`) and
  `scripts/analyst_shortlist.cjs`; NBA instructions in `scripts/analyst_review.py`
  (injury report, minutes limits, rest, confirmed lineups); `nba.com` in the source
  allowlist (`safeSourceURL`); morning card coverage (`scripts/morning_card.py`).
- Scheduling: NBA news settles late in the day, so lean on the 4:30 PM ET afternoon
  refresh; add NBA to `morning-picks.yml` / `afternoon-refresh.yml` as needed.
- This is a Top Picks policy change: update the process page per `AGENTS.md`.

## Phase 4: more surfaces

- **Prediction Markets:** add Kalshi's NBA game-winner series to
  `config/prediction_markets.json` (currently only `KXNFLGAME`). Verify the series
  ticker against `https://external-api.kalshi.com/trade-api/v2/series/<ticker>`
  before shipping, then run the manual Prediction Market Snapshot workflow.
- NBA player pages (like `scripts/nhl/players.py`), a candidate board, an arbitrage page
  (generalize `scripts/nhl/v2/arbitrage.py` with a verified NBA settlement profile),
  NBA on the home page slate, and NBA explainers.

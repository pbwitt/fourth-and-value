# Scheduled site refreshes

- **MLB:** daily at 15:15 and 21:15 UTC (11:15 AM and 5:15 PM Eastern daylight time; 10:15 AM and 4:15 PM Eastern standard time). Existing `mlb-daily.yml` updates completed-game history, trains/calibrates models as needed, refreshes odds and published orders, validates, archives inputs, and publishes.
- **NFL:** Wednesdays at 14:00 UTC (10 AM Eastern daylight time / 9 AM Eastern standard time). `nfl-weekly.yml` selects the next scheduled regular-season week rather than computing it from a hard-coded season date. Runs outside the upcoming season window skip market requests and publishing.
- Both workflows can be dispatched manually. GitHub schedules may run later than the exact configured time.
- Paid game insights are off by default. Scheduled runs receive no OpenAI key. Only an explicit manual `generate_insights=true` approval enables that step; `make` also defaults to `SKIP_AI_INSIGHTS=1`.

The NFL workflow prepares all required files on a clean runner, caching prior-season player stats, compact play-by-play inputs and line-history snapshots. Current-season data is refreshed each run. It stops if the previous week is not yet represented in either player stats or play-by-play. All training play-by-play is restricted to seasons/weeks earlier than the selected week. The season and week are read from the schedule, including January's previous-year NFL season. Postseason automation is not enabled by this regular-season workflow.

NFL totals forecasts now calculate their live L3/L5 features from completed-game raw metrics, including the latest completed game; they no longer reuse the last historical row's already-lagged features. Historical training rows remain lagged to avoid same-game leakage. NFL line snapshots are persisted before the injury/line-movement screen runs; a first snapshot correctly has no measured movement. Paid insights and user bet-tracker grading are separate from routine site publication.

A successful NFL run publishes props, totals, injury context and approved insights (only if requested), then explicitly rebuilds GitHub Pages. Failed data, modeling or QC steps prevent publication; diagnostics and model artifacts are retained for 30 days. MLB retains its existing 90-day model audit artifacts. The configured GitHub cache accelerates runs but is not a permanent archive.

The Pirates blog post is outside these refreshes and remains unchanged.

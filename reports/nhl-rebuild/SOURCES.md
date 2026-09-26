# Sources and availability decisions

Checked 2026-09-26. No subscriptions purchased, historical-odds entitlement assumed,
restricted content bypassed, or sportsbook account accessed.

| Input | Source / coverage | Cost and refresh | Historical availability / decision |
|---|---|---|---|
| Game results, team goals and shots | [NHL statistics](https://www.nhl.com/stats/), `api.nhle.com/stats/rest/en/team/summary`, four complete seasons | Public endpoint; daily during season | Game IDs and dates are stable. Original publication and revision timestamps absent; reconstructed next-day noon UTC cutoff. Shootout awards absent from statistical GF, explicitly restored for settlement. |
| Player goals, assists, points, shots, TOI | NHL `skater/summary` per game; 188,883 appearances | Public endpoint; monthly partitions to avoid the 10,000-row cap; refresh current season only | Outcomes and lagged features, not retrospective active lineups. Stable player IDs; identity ambiguity fails closed. Current-game position excluded. |
| Special teams | Lagged team PP/PK percentage from the same reports | Included above | Available retrospectively with the same timestamp limitation. Missing game percentages shrink toward fixed priors. Not opportunity-weighted PP/PK event counts; this challenger was not selected. |
| Opponent strength and home advantage | Derived from prior game rates | No added feed | Strict lag, recency and shrinkage; selected regularized core. |
| Rest, back-to-back and congestion proxy | Previous completed game dates | Included above | Dates retrospectively known; no exact historical reschedule announcements. Lagged rest challenger tested; broader context bundle not selected. No travel-distance estimate inferred from team names. |
| Aggregate goalie contribution | Prior team goals conceded / shots allowed proxy | Included above | Includes empty-net and team defense, so it is not a goalie talent statistic. Context challenger only. |
| Starting goalies, workload and identity | Public NHL goalie reports could supply outcomes; live announcements require sourced records | No paid feed configured | No reliable historical starter announcement times. Retrospective starters would leak information. Not included in independent models; sourced analyst shadow reviews supported. |
| Shot quality / xG / attempts | [MoneyPuck download terms](https://moneypuck.com/data.htm) evaluated | Free permission is limited to non-commercial and stated journalistic uses | Not integrated into a commercial site without appropriate permission. Building a new event-level xG model from NHL play-by-play remains unimplemented; it is not silently approximated by shots. |
| Even-strength lines, PP unit, linemates | No timestamped historical provider found in the repository | No subscription added | Unavailable; live-only analyst evidence. No roster reconstruction from final-game line combinations. |
| Injuries, scratches, returns, trades, coaching, tactics | No historical timestamped archive in repository | Analyst source URL and source/review times required | Prospective only. Prior observed position/TOI is allowed; current roster/team membership is not asserted from old stats. |
| Upcoming games | `api-web.nhle.com/v1/schedule/YYYY-MM-DD` | Public; each market refresh | Official game ID/type/state, teams and start time; only regular season, upcoming, non-postponed games. Current schedule cannot certify past announced schedules. |
| Offered prices | [The Odds API NHL](https://the-odds-api.com/sports/nhl-odds.html), existing `NHL_ODDS_API_KEY` or `ODDS_API_KEY` | Existing authorized plan; unchanged bounded live requests (events + game lines + up to 16 prop events), no upgrades | Provider quote time and local ingestion time both recorded. Paired quotes and raw payloads archived. Prop market coverage varies by book and proximity to game time. |
| Historical prices, movement, closing quotes | [The Odds API historical documentation](https://the-odds-api.com/liveapi/guides/v4/#historical-odds) and local inventory | Historical paid endpoint not called | Old CSVs lack quote times and exact settlement provenance. September 2026 snapshots precede the modeled regular season. No adequate settled decision-time sample. Market benchmark, learned blending, executable ROI and early-vs-late comparison are blocked. |
| Settlement profiles | Primary sportsbook rules linked in `config/nhl_settlement.json` | Recheck before market use and when rules change | Standard full-game mapping only, with jurisdiction/header exceptions explicitly unverified. Unknown profiles do not join cross-book consensus; unknown EV/minimum price is withheld. No historical rule-version backfill assumed. |

Reliability policies: API schema, paired team scores, duplicate IDs, incomplete pagination,
report caps and source hashes fail closed. Requests have bounded timeouts/retries. Current
model inputs require a successful check within 36 hours; market quotes expire after 24
hours. A modeled offer is a conditional forecast given action, not a guarantee of player
participation or price executability. Sparse/rookie histories use explicit priors; an
unknown or ambiguous player receives no independent forecast.

Local legacy inventory: 18 game-odds CSVs and 16 prop-odds CSVs were inspected; headers
had event/start fields but no quote/publication timestamps. The saved ledger has seven
total entries, five NHL, with `sog` and inconsistently named `team_total` markets. Grading
code also matched player names and dates and could invent zero scores for missing columns.
The replacement uses stable IDs and unresolved/void states, and never mutates that ledger.

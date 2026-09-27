# Research-driven Top Picks — decision log, 2026-09-27

Branch: codex/research-driven-top-picks. Review before deployment. Target: an operational pilot before Tuesday September 29; this is not a claim of profitable or validated picks.

## Preserved contracts

`docs/assets/briefing-picks.js` remains the shared browser/Node selector. Existing sport JSON, routes, six table columns, actual book line, odds, quote timestamp, comparison semantics, and Bet Tracker payload remain supported. New metadata is additive. Model probabilities are never supplied by the language model. Exact offer/forecast identity is retained in immutable research snapshots. Market Watch stays separate.

## Audit

The inherited selector admitted only four candidates per sport and one per game before research. NFL sorted probability-minus-implied-price, MLB sorted model EV, and pass verdicts consumed slots. NFL calibrated tails can depend on very low-attempt appearances without workload adjustment; calibration extrapolation cannot repair that input problem. Evidence collection had a small article budget and reviews could not revisit material news within a session. NFL game quantity forecasts do not establish betting probabilities. MLB historical validation is not a timestamped executable-price backtest. No old performance claim is adopted.

## Implementation sequence

1. Shared $2.75 America/New_York calendar-day research ledger, including legacy charges; durable pre-call reservations, no automatic billable retries, retain unknown charges.
2. Remove selection quotas; deduplicate identical outcome/line offers at the best price, retain different markets in one game. Gate unsupported calibration tails and possible partial-workload distortions; preserve excluded rows and reasons for diagnosis. Reliability and sensitivity precede raw value ranking.
3. Independent slate discovery blinded to our probabilities/ranks, joined to current exact sportsbook offers. Merge with quantitative candidates. Independently proposed bets remain prospective research, never silently receive a model probability or EV.
4. Shared evidence collection and targeted follow-up, bounded review batches, cached stable decisions with explicit rechecks on changed prices/forecasts or new material reporting. Unreviewed candidates remain visible when budget runs out.
5. Public status, coverage, correlated exposure, operational schedules, replay tests and rendered-state checks. Draft PR, no deployment in this change.

## Decisions and constraints

- $2.75 is an application-enforced cap across NFL, MLB and NHL research, including failed calls with unknown usage. It excludes the existing odds feed subscription and unrelated OpenAI use. No subscriptions/upgrades authorized or purchased.
- Use requested gpt-6-astra. Standard API documented rates: $10/M input, $1/M cached input, $12.50/M cache writes, $50/M output; reserve at conservative $12.50/M for all input. Web search costs $0.01/call plus content tokens. Built-in search has a 128k context; isolated one-search requests need a large conservative reservation. Unknown charges retain the entire reservation. Sources: https://developers.openai.com/api/docs/models/gpt-6-astra and https://developers.openai.com/api/docs/guides/tools-web-search (checked 2026-09-27).
- No learned new blend or arbitrary qualitative probability adjustment. Independent discovery/reviews are a prospective selection experiment. No numeric LLM confidence, fabricated injury news, or inferred health from silence.
- Prefer sourced official reports; professional recommendations may suggest research but are opinions. Search results alone are not verified news: retrieve, date, match and archive the underlying evidence before publishing factual support.
- More candidates means more possible research than $2.75 can buy. Expose incomplete coverage and queue state; do not pretend every market received equal research. Batching is an API resource limit, not a published picks quota.
- NHL refreshes at 10:30 and 16:30 ET. Books have no guaranteed opening time. Main markets may precede props; absent markets remain pending and later refreshes can add candidates. Freshness limits remain enforced, not extended to fill a morning list.
- No one-per-game constraint. Shared game/player exposure is labeled; no assumption of independent bet outcomes and no automatic stake allocation.

## Unresolved validation

A larger candidate pool and independent research cannot be credited with improved ROI without prospective grading. Archive discovery origin, screening reasons, original predictions, reviewed prices, sources, exclusions and budget events for that evaluation. NFL missing full-workload history and unvalidated game probability models remain limitations, not problems an LLM can numerically repair.


## Completed implementation and final operating decisions

- Shared discovery/review orchestration now includes NFL, MLB and NHL. NHL's old paid runner is a compatibility no-op, preventing a second allowance. Review batches rotate across sports. All eligible candidates remain in the pool; the table initially displays 20 with Show all.
- Fixed NFL 3% minimum sensitivity EV and a lower-priority research tier above 30% EV. These are declared operating heuristics, not fitted selection thresholds. Matched partial-workload and calibration extrapolation failures are withheld. Integer quote keys compare numeric values, avoiding Python 5.0 versus JavaScript 5 serialization mismatches. Unknown pushes fail closed.
- MLB retains 3% EV but drops the redundant 3-percentage-point gap hurdle. No model weights, validation results or predicted quantities were changed by this policy update.
- $2.00 maximum morning spend preserves $0.75 for noon-and-later research. Both use the same $2.75 day ledger; no rollover. A conservative search reservation may prevent full slate submission despite a lower expected bill. Submitted-game counts are not claims of exhaustive research.
- Added timestamped NFL moneyline/spread/total quote export from the existing authorized feed; adding h2h uses one extra market credit per requested region. Fresh game quantity forecasts are supplied as labeled context without assigning unsupported probabilities.
- Early MLB refresh no longer depends on available editorial-writing slots. NFL's existing early refresh remains. Article selection/writing/budget and Market Watch are unchanged.
- Public operating description: `/research/daily-process.html`, linked from Research and Today's Picks. Schedule text comes from workflow configuration, numerical policy values from config, and a source fingerprint makes stale generated documentation fail PR checks. The HTML paper links to the current operating policy without changing its empirical results.
- No billable API calls or site deployments were performed on this branch. The new hosted-search integration is tested through request/response fixtures and needs a monitored production smoke run after approval. Existing Astra review transport remains the production client.

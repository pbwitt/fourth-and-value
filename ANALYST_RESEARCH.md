# Daily candidate research

The current reader-facing description is `/research/daily-process.html`, linked
from Research and Today's Picks. `scripts/build_daily_process.py` generates its
schedule and policy fields. After changing a workflow or selection policy,
review `scripts/research/daily_process.html`, regenerate the page, and run
`python scripts/build_daily_process.py --check`. PR checks detect drift.

## Flow and contracts

1. Sport refreshes publish statistics, forecasts and exact sportsbook offers.
2. Independent discovery receives games/market types, without model ranks or
   probabilities. It searches current reporting and returns research directions.
3. Directions must match real current offers by exact game, player, market and
   side; the actual offer supplies its line, book, price and timestamp. Models
   can supply contrary evidence at the subsequent review stage.
4. `docs/assets/briefing-picks.js` owns the quantitative screen in the browser and
   Node adapter. Union both discovery routes. Prefer the best book for an
   identical outcome, line and settlement. No four-candidate or one-game quota.
5. Source collection, targeted retrieval and review batches rotate across sports.
   Independent discovery ideas are researched early so a large model pool cannot
   consume the allowance before those ideas are assessed.
   Consider/wait/pass assessments precede reliability and numerical ordering.
   Changed evidence gets rechecked; unchanged price, forecast and evidence can
   reuse a review for three hours with the original timestamp preserved.
6. Publish a focused daily card of **up to 10 reviewed ideas**, aiming for a
   manageable five to ten when supported, with no required minimum. Only exact
   current consider assessments no older than three hours, or explicit analyst
   selections, qualify. Explicit analyst selections take priority. Wait/pass/unreviewed and changed-price reviews never fill
   empty slots. Consolidate alternate thresholds for the same subject/market/side
   and equivalent MLB 0.5-hit/0.5-total-base outcomes, preserving original forecasts.
   The research pool remains separate in a collapsed section, initially showing
   20 additional offers with a Show all control. It retains every eligible offer,
   including additional consider decisions beyond the card limit, for audit.
   There is no per-game or per-sport quota. Shared game exposure is labeled on
   the card; the limit is a reader workflow choice, not a validated betting policy.

Existing sport routes and JSON contracts remain supported. Additive feeds are
`/briefing/discovery.json` and `/nfl/data/quotes.json`; `/briefing/reviews.json`
retains schema version 1 with additive budget, audit and coverage metadata.
NFL moneylines/spreads/totals in the quote feed have no invented win probability.
Available fresh NFL game quantity forecasts are separate review context. The
existing authorized game-odds fetch now requests h2h alongside spreads/totals;
this uses one additional market credit per requested region, no subscription
upgrade. Raw responses retain source quote times and separate ingestion time.

The table still shows the offered book's actual threshold and price, timestamp,
model and market comparison, sourced assessment and Bet Tracker. Research-only
rows without an eligible model say Not established and cannot create a tracker
model probability/edge. Original forecasts remain archived and may be supplied
as explicitly unqualified diagnostic context. Numerical estimates are never
invented or adjusted by the language model. Market Watch and daily-article
selection/writing remain separate. Morning MLB/NFL refresh eligibility now also
runs independently of article-slot exhaustion.

## Quantitative policy and ranking

NFL props require outcome calibration, a known push probability, fresh quotes,
and at least 3% EV under the smaller of raw and calibrated probabilities where
both exist. This is a sensitivity heuristic, not a confidence interval. Matching
calibration extrapolation and possible partial-game workload distortion are
withheld from model ranking. Missing raw provenance lowers reliability; apparent
returns above 30% are placed in a lower-priority research tier. These thresholds
are operating rules, not a profitable subset learned from historical returns.

MLB keeps applicable predictive validation, supported lines, fresh inputs, two
paired books and best same-line price. Its 3% EV hurdle remains; the additional
3-percentage-point gap hurdle is removed. Apparent EV over 30% remains research
only. NHL retains its coherent scoring/opportunity forecasts, verified settlement,
2% EV and adverse-scenario minimum-price rule. NHL ranking still uses worst-case
fixed-fraction log growth; the fraction is not stake advice.

A current consider assessment ranks before pending/wait/pass across all sports,
so a reviewed MLB/NHL offer appears before unreviewed NFL rows in the initial table.
No cross-sport probability or numerical value score is introduced.
Reliability then precedes each sport's numerical score. Later review batches
prioritize previously reviewed offers needing new context/price assessment.
Multiple bets in the same game remain possible and are not assumed independent.
No new numerical model/market/qualitative blending weight is introduced.

All displayed comparison probabilities condition on no push. NFL already emits
that quantity; MLB/NHL win mass is divided by one minus push mass. Missing push
mass stays unknown. NHL/MLB other-book references exclude the offered book; the
inherited NFL prop consensus can include it and is labeled accordingly. A market
median threshold is not a mean forecast. `EV = p_win * decimal_odds + p_push - 1`;
fair decimal odds are `(1 - p_push) / p_win`. No model means no model EV claim.

## Evidence and analyst review

The source policy admits league/team reporting, ESPN and CBS. Independent search
also admits Action Network, Covers and VSiN as leads; professional recommendations
are labeled opinion and never establish an edge by themselves. Results must be
retrieved as actual dated articles before they can support published findings.
Allowed HTTPS hosts, redirect limits, response-size limits, full player/team
matching, publication/update/retrieval order and source-excerpt validation remain.
No access restriction is bypassed. Article excerpts and HTML are untrusted data.

Articles must be published within seven days and published/updated within 72
hours, before retrieval/review. Direct MLB/NHL injury tables have unknown original
publication time and retain `published_at: null`; they are live-only snapshots,
usable for 90 minutes. Full matched injury rows stay archived and expandable;
request excerpts may contain a labeled subset. Absence never proves health.
NFL injury-table parsing preserves actual rows instead of JSON-LD legends.

Free source collection remains bounded (16 articles, up to eight NFL team indexes,
and direct injury tables); targeted search leads allow up to 24 additional
retrievals. Batch prompts select compact excerpts; they do not claim every fact
in an archived table was read. Search coverage counts games submitted, not games
exhaustively researched. Missing, blocked, stale and mismatched sources remain
explicit. Positive/adverse source statuses require verified excerpts. Numeric
confidence, unsupported fields and fabricated excerpts are rejected locally.
Interpretations still require human review. Missing news alone is not a veto.

## Schedule, budget and failures

See the generated public schedule for exact times. Early morning jobs refresh
stale MLB/NFL boards, then call research before briefing publication. Standalone
successful MLB/NFL/NHL refreshes also trigger research. Additional checks occur
at 08:45, 10:45, 12:45, 15:45, 16:45 and 18:45 ET. Review windows are 05:00–12:00
and 12:00–21:00. NHL scheduled refreshes remain 10:30/16:30 ET. New props may post
later; no quote is invented. NFL/MLB quotes expire after 90 minutes; NHL Top Picks
quotes/forecasts after 30 minutes, with model inputs under 36 hours.

All Top Picks research uses `artifacts/analyst/daily-budget.json`: **$2.75 per
America/New_York calendar day**, including legacy same-day ledger charges.
Morning requests can use at most $2.00, retaining $0.75 for noon-and-later checks.
One shared allowance covers discovery, follow-up and reviews across all three
sports; unused capacity does not roll over. Existing odds-feed and independent
article-writer costs are outside it. The legacy NHL paid entry point now only
marks the shared queue; it cannot spend a second allowance.

Use gpt-6-astra, default tier, low reasoning. Reviews have no tools, at most
26,000 serialized request bytes and 4,200 output tokens. Discovery uses an
isolated web-search request with one built-in tool call, no response history,
at most 12,000 request bytes and 1,600 output tokens. Conservative search
reservations include 131,072 search-context tokens, twice serialized request
bytes, overhead, output maximum, $0.01 search fee and margin. Bounds can exceed
remaining funds even when expected actual cost is small; skip rather than guess.

CI serializes research jobs and checkpoints the reservation and exact packet to
origin before payment. Local runs use a file lock; standalone ledger mutations
are separately locked and atomic. Never run paid local experiments concurrently
with production or against an isolated test ledger. No API retries follow timeouts.
Unknown usage retains the entire reservation. Unexpected over-reservation usage
halts further spending for operator investigation. Use actual conservative input
and output usage to settle; do not assume cache discounts. Credentials remain in
environment/GitHub Secrets. No new paid feed or hosting service is purchased.

## Reproduction and evaluation

Python 3.11, Node 22, requests 2.32.5 for review jobs; modeling dependencies remain
in the existing requirements files. Browser CI uses Playwright 1.55.1. No live
API call is part of PR tests.

```sh
node tests/briefing_picks.cjs
python -m unittest discover -s tests -p 'test_research*.py'
python -m unittest discover -s tests -p 'test_analyst_review.py'
python -m unittest discover -s tests -p 'test_nhl*.py'
python -m unittest discover -s tests -p 'test_mlb*.py'
python -m unittest discover -s tests -p 'test_editorial_schedule.py'
python scripts/build_daily_process.py --check
python scripts/replay_top_picks.py
python scripts/analyst_review.py  # no paid requests; writes current status/archive
node tests/research_browser.cjs   # installed Playwright/Chromium required
```

Immutable boards, exact request/source packets, responses, discovery snapshots
and published results live under `artifacts/analyst/`; CI also uploads them for
90 days. The saved-data selection replay is `reports/top-picks/selection-replay.json`.
It proves candidate-policy behavior, not predictive improvement or profitability.
Original dataset hashes are recorded. NHL preseason/empty slates make no picks.

Bet Tracker retains executed price and stake, but does not yet save a structured
review/candidate ID or human intervention timestamp. NFL game research markets,
MLB and NHL moneyline/puck-line tracker grading are not fully connected and remain
pending. Do not claim intervention ROI without unambiguous identity joins and
complete outcomes. No private tracker records are read or written by this runner.

## Release, monitoring and rollback

Merge/deploy only after approval. Following release, refresh the sports feeds so
the new NFL quote feed and MLB EV policy exist, then run Morning Candidate Research.
Inspect the reserved/settled budget, immutable packets, public discovery/review
status and live table. First production search needs monitoring: the new hosted
search route is covered by mocked transport/schema tests, not a paid call on this
review branch. Missing sources, budget exhaustion and an empty card are valid
states, never labeled completed review. Target a supervised morning pilot before
relying on the new process for Tuesday's decisions.

To stop research, set `astra_enabled` and `discovery_enabled` false; model/feed
updates remain available. For a selection/UI rollback, first disable paid review,
then restore the prior selector and templates from commit `611e9c4` after approval.
Keep the new budget module, serialized workflow and disabled paid entry points.
Do not revert the entire change in a way that re-enables the old weekly-funded
runner. Preserve both daily and legacy ledgers and all archives. Never reset a
ledger to regain spending or retry an unknown billable request.

OpenAI references checked 2026-09-27: [model/pricing](https://developers.openai.com/api/docs/models/gpt-6-astra),
[web search](https://developers.openai.com/api/docs/guides/tools-web-search),
and [Responses API](https://developers.openai.com/api/reference/typescript/resources/responses/methods/create).

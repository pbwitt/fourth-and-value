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
selection/writing remain separate. MLB no longer has automatic early editorial
refreshes; article writing follows the coordinated 7:05 a.m. sports feeds,
with an 8:30 a.m. delivery target.

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
The research pool retains each sport's numerical order. The main ten-idea card
sorts across sports by explicit analyst selection, forecast reliability, then
expected log growth at the NHL ranker's fixed 0.0025 fraction. NFL uses the lower
of raw/calibrated probability where available; MLB uses unconditional win/push
mass; NHL retains its existing worst-scenario score. Refunds contribute zero.
This common scale prevents feed order or incomparable source ranks from filling
the card with one sport. It is an operational ordering heuristic, not a stake
recommendation, confidence score or validated cross-sport performance claim.
Missing independent forecasts receive no numerical score and rank after eligible
models, unless explicitly selected by an analyst. There are no sport quotas. Later review batches
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
All bets should be reviewed before deciding. Missing news alone is not a veto.

## Schedule, budget and failures

`Morning Picks Edition` is scheduled for **07:05 America/New_York**, with recovery
starts at **07:35, 08:05 and 08:35**. A gate reads current main before any sport
pull: a completed same-day morning edition skips all three feeds and paid research, then
verifies delivery. A missing or incomplete edition calls NFL, MLB and NHL reusable
workflows, waits for all three, then calls research. A separate short handoff
dispatches editorial with scheduled recovery semantics; writing does not block
Top Picks research or its recovery starts. Explicit test editions skip the
automatic editorial handoff.
Parent job results travel with the card. A failed sport or expired feed marks
the run incomplete; completed assessments from healthy boards remain visible.
NHL/MLB standalone schedules are **16:30 Eastern**. Existing NFL game-day updates
remain. Neither later updates nor editorial publication trigger paid research.
There is no hourly MLB refresh. The article watchdog uses scheduled recovery
eligibility, not the manual full-refresh path. Automatic articles require
post-7:05 model checks within 90 minutes and remain subject to per-story data
and source validation. Editorial targets 08:30 ET and closes automatic writing
at noon; hourly maintenance and approved-post publication continue. NBA
integration into the coordinated sports refresh remains future work.

Normal research runs 07:00–12:00 ET, after the feeds complete. Publication has no
promised minute. The whole **$2.75 daily** allowance is available to this morning
run; there is no afternoon reserve. All charges share
`artifacts/analyst/daily-budget.json`, including prior/legacy same-day charges.
No automatic intraday discovery or reassessment runs. Existing odds-feed and
article-writer costs remain outside this cap. The legacy NHL entry point only
marks the shared queue and cannot spend independently.

`scripts/morning_card.py` calls the same Node/browser selector at publication.
Quotes/forecasts must pass the normal sport-specific freshness checks, and
consider reviews must match the exact offer and be no older than three hours.
Up to ten distinct ideas are written to `/briefing/morning-card.json` (schema 1)
and an immutable `/briefing/cards/DATE-ID.json` archive. Each row preserves the
original quote, model, market comparison, review and sources. Later updates never
mutate the edition. Started games are labeled historical; previous-day editions
are explicitly labeled previous. The separate research pool still expires rows.
A temporary card-fetch failure retains the last dated card in an open browser.

`--publish-card` skips before any paid call if this Eastern date already has a
completed morning edition (including a valid empty edition). A completed test
edition does not satisfy a normal morning run: both the pre-feed gate and research
step proceed without needing `--replace-card`, retaining the test archive and
same-day spending. Explicit `--test-edition` requests still skip if either kind
is already complete, unless `--replace-card` is supplied. Incomplete editions do not
block recovery. Missing/malformed/future-dated editions cannot suppress a run. `--replace-card` is an explicit
operator override; archives and spending remain intact. `--test-edition` permits
an outside-window run and labels it Test edition. Both flags are workflow inputs.
The scheduled parent defaults both to false. A blank card is allowed after
completed research finds no qualifying offers; wait/pass/unreviewed entries never
fill it. Each card stores per-sport candidate/reviewed/pending counts, batch and
refresh statuses, plus publication-time availability. `research_incomplete` is
separate from `no_reviewed_candidates`. Missing credentials, rejected responses,
failed discovery, refresh failures and expired research are not a normal no-pick
day. The workflow publishes available diagnostics, then exits nonzero. A run that
arrives outside 07:00–12:00 without a completed edition also fails before pulling
feeds or spending. Normal bounded coverage (some completed reviews plus the
batch/budget limit) stays explicit and does not imply every candidate was assessed.

Publication uses the same frozen feed/discovery snapshot that was researched;
budget-checkpoint rebases cannot swap in newer offers. Freshness is still checked
at publication. The final workflow polls the public card for the exact edition ID,
date, kind and status, using unique query strings and no-cache requests. It tries
24 times, 20 seconds apart, with a 15-second request timeout, and fails if delivery
cannot be proven. The next recovery start verifies/rebuilds a completed edition
without paid research. Browser/CDN caches can still delay visibility for readers.

All four existing scheduled starts depend on GitHub cron. The September 28
incident confirmed that none arrived during the morning. An independent Supabase
trigger is now implemented in `supabase/morning_scheduler.sql`, with four bounded
dispatches to the same workflow and existing idempotency/budget gates. **Activation
is pending a Supabase administrator's setup and live verification**; a Git merge
alone does not install it. See [MORNING_SCHEDULER.md](MORNING_SCHEDULER.md) for the
incident evidence, installation, receipt checks and rollback. Workflow summaries
record the trigger source and gate execution time. Until activated, all automatic
starts still share GitHub's scheduler. GitHub documents timezone support and
delayed/dropped cron events:
https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule

Owner-authorized release testing on **September 27, 2026 only** uses a **$20
shared ceiling**, configured by `test_budget_override` with an exact Eastern date
and authorization reason. It includes every charge already recorded that day;
it does not reset or create another ledger. Both discovery and review use the
same override, and each reservation records the effective limit and reason.
The exception expires automatically at Eastern midnight; September 28 resumes
$2.75 without a deployment or manual reset. Increasing `daily_budget_usd` alone
cannot bypass the normal cap. This is a testing allowance, not a spending target.

Use gpt-6.1-sol (GPT-6.1 Sol), default tier, low reasoning. GPT-6 Astra (`gpt-6-astra`)
was used through October 1, 2026; reviews and ledger entries record the model used.
Rates in `scripts/nhl/v2/astra.py` (`RATES`) are conservative: Sol $2.50/M input
(list $2 plus the cache-write premium) and $10/M output. Reservations
(`astra.bounds`, discovery bounds) and `research_budget.settle` use the same
model's rates, so a normal call cannot trigger the overrun halt. The change lowers
cost; review rules, batch limits and the daily allowance are unchanged. Reviews have no tools, at most
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
review/candidate ID or human intervention timestamp. The Grade Tracked Bets
workflow (`scripts/grade_bets.cjs`, every 30 minutes) settles pending NFL, MLB, NHL
and NBA tracker bets, props and game markets, from final box scores once a game
started 4+ hours ago; unmatched players, unsupported markets and postponed games
stay pending. Do not claim intervention ROI without unambiguous identity joins and
complete outcomes. No private tracker records are read or written by this runner.

Price movement after publication: each MLB and NHL refresh runs
`scripts/line_movement.py`, which keeps, for every pick on a card published in the
last four days, the latest snapshot ingested after publication and before the start
(`docs/{mlb,nhl}/data/line-movement.json`). It records the same book's line and price
and other books' fair probability at the pick's exact line; a changed line is a
separate line move, never a same-line probability change. With refreshes at 7:05 a.m.
and 4:30 p.m. ET this is the latest pregame snapshot held, not the closing line. The
briefing summarizes it; it does not change selection, ranking or review.

## Release, monitoring and rollback

After authorized release, dispatch `Morning Picks Edition` to exercise the full
feed → review → immutable-card sequence. For an afternoon test use:

```sh
gh workflow run morning-picks.yml --ref main -f test_edition=true -f replace_card=true
```

Inspect all three feed jobs, the research result, charged/reserved usage, the
card's edition/date/coverage, original timestamps and live desktop/mobile table.
A rerun with default inputs must not replace a completed same-day edition or spend
again. Scheduled starts are 7:05, 7:35, 8:05 and 8:35 a.m.; publication depends on
feeds and research. Incomplete editions recover within the same daily ledger.
Alerts remain GitHub Actions failures; a completed no-pick day stays distinct
from incomplete research. No external alert subscription or recipient was added. The research pool shows feed coverage.
Run `python -m unittest discover -s tests -p 'test_morning*.py'` for archival
and idempotency tests. Browser tests check missing feeds, expired quotes and
retention through later updates. No tests call paid research APIs.

To stop research, set `astra_enabled` and `discovery_enabled` false; model/feed
updates remain available. For a selection/UI rollback, first disable paid review,
then revert the morning-edition code/workflow commit if required. Preserve
`docs/briefing/cards/` archives and the last card, even during a UI rollback.
Keep the new budget module, serialized workflow and disabled paid entry points.
Do not revert the entire change in a way that re-enables the old weekly-funded
runner. Preserve both daily and legacy ledgers and all archives. Never reset a
ledger to regain spending or retry an unknown billable request.

OpenAI references checked 2026-09-27 (Sol pricing checked 2026-10-01): [Astra model/pricing](https://developers.openai.com/api/docs/models/gpt-6-astra), [Sol model/pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[web search](https://developers.openai.com/api/docs/guides/tools-web-search),
and [Responses API](https://developers.openai.com/api/reference/typescript/resources/responses/methods/create).

Each research run attempts at most eight batches of at most three candidates. The broad candidate pool remains available, but the runner reports `review_limit_reached` when it stops this focused review. This is separate from the shared daily spending ceiling.


## Editorial Desk manual publication — September 30, 2026

All private `editorial_ideas` submissions (reader or editor-origin) stop at
`review` after writing. They never enter the writer's direct public-article
branch, even if legacy `publish_own` / `write_now_publish` flags are true.
Independent daily articles without an idea row retain their existing automation.
The desk uses Save idea → Generate draft (Analysis or Opinion) → review/edit → Save draft
→ Preview → Publish saved article → explicit confirmation. Featuring, changing
kind, and saving are not publication authorization. `approved` means publication
requested; hourly delivery follows, subject to `publish_on`. The publisher
requires an approved/publishing status and a stored approval fingerprint; reader
rows also require `approved_by`, whose current editor role is rechecked.
See `EDITORIAL_OPERATIONS.md` for rollout, regression checks and limitations.


## Uniform Opinion generation — September 30, 2026

Both article types use the same editor-authorized drafting and rewrite controls.
Opinion uses dated reporting from at least two source domains, a source/citation
validator and the separate factual audit. It does not depend on betting-board or
model freshness; its evidence excludes market prices and forecasts. Missing
reporting stops before spending. Unsupported historical or current claims remain
blocked. Drafts preserve article type and the editor's saved byline. Failed
rewrites preserve the previous draft; all successful drafts require manual
publication. The existing model, input/output limits, spending cap, no-paid-retry
rules, reader isolation and publication fingerprint checks still apply.

Explicit Opinion requests skip sports-model and briefing refreshes and go directly
to reporting collection. Editor-origin Opinion ideas and reader ideas accepted for
research can use available daily drafting slots; they never become standalone
public daily articles. Reader submissions alone do not authorize a paid request.
Activation requires `supabase/editorial_opinion_generation.sql` plus redeploying
`editorial-write-now`. See `EDITORIAL_OPERATIONS.md` for the release sequence.

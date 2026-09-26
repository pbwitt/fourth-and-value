# Editorial homepage and publishing

The homepage combines a curated lead, an automated market rundown, recent reporting, dated model features, opinion and results. Existing authored posts remain unchanged.

## Schedule and costs

`.github/workflows/editorial-daily.yml` collects a morning edition at **5:07 AM America/New_York**, adjusting for daylight saving time. GitHub schedules can be delayed; this is a target, not an exact-time promise. At **17 minutes past each hour**, it checks the private queue for approved articles and refreshes homepage freshness labels.

Each price edition makes one full-game totals request per sport (NFL, MLB, NBA, NHL) with existing Odds API credentials and reads four public ESPN RSS feeds. It does **not** train models or generate new prop forecasts. Existing sports/model refreshes continue separately. The authorized Astra writer then researches and directly publishes original features, using the server-side `OPENAI_API_KEY` secret. See the original-analysis operating details below.

Each edition is at `/briefing/YYYY-MM-DD.html`; `/briefing/` is the latest. Data is stored under `docs/briefing/history/`, with immutable timestamped snapshots under `snapshots/`. The report covers NFL games within seven days and other sports within 48 hours. It labels market observations as research, not recommendations. ESPN headlines are linked, limited to 24 words and not expanded into unsupported facts. The briefing links headlines; the separate feature writer reads bounded publisher excerpts privately.

## Private editorial desk

Visit `/editorial/inbox.html` on a phone or computer. Use the site's existing Supabase email sign-in. An account needs trusted `app_metadata.fv_editor = true`; ordinary users cannot read or write the queue. No service key goes into public JavaScript.

One-time database setup: execute **`supabase/editorial.sql`** in the existing project's SQL editor. This creates only the new `editorial_ideas` table, its policies and approval trigger. It does not change the bet tracker. Current REST credentials do not grant schema-management access, so this step requires a project administrator. Set the owner's editor flag using `scripts/editorial_owner.py --email ...` with the existing server-side credentials. Never put the email or credentials into public config. Sign out and sign in again after a role change so the JWT is refreshed. Ensure the Supabase Auth redirect allowlist includes `https://fourthandvalue.com/editorial/inbox.html` (or the existing site-wide wildcard).

Ideas remain private in Supabase. Statuses:

- **Submitted:** an angle to research; no public article exists.
- **Researching:** work is in progress (can be set through the trusted server client).
- **Review:** a saved article draft; preview and edit it on the desk.
- **Approved:** owner explicitly approved this saved version and byline.
- **Publishing:** the hourly job has claimed the approved version; edits are temporarily blocked. If deployment fails, the next run retries this exact version.
- **Published:** a public page exists and Git push + Pages build request succeeded. CDN deployment may take a little longer.
- **Archived:** removed from the active publication queue, retained privately.

Saving changed content, sources, byline, type, feature setting or date invalidates approval. Analysis requires source links; opinion requires explicit approval just like analysis. No idea or unapproved draft is exported to Git, job logs or artifacts. Drafts render as escaped plain text, not executable HTML. Mobile edits use an optimistic version check to avoid silently overwriting another device.

The SQL trigger fingerprints the approved content. The publisher claims it atomically before rendering and only acknowledges after publishing. It rechecks the owner's current trusted role. The workflow's single concurrency group prevents two publishers racing. If a job fails after a claim, do not manually force the status to published; rerun the workflow. A user cannot self-promote to editor by changing user metadata.

## Updating the lead and opinion

Existing features are listed in `config/editorial.json`. New published articles flow into `docs/editorial/published.json`, which contains only public metadata. A featured analysis can become the homepage lead; opinion always remains distinctly labeled in the opinion column/archive. All approved analysis also appears in the blog index. The private desk is noindex and is linked discreetly from the homepage footer.

This release intentionally keeps recently published deep dives as dated analysis, not as live picks. It does not imply that old odds remain available. The initial receipts box links to the audited Week 2 scorecard; automatic grading of new editorial picks needs structured wager records before it can be enabled safely.

## Run and verify

```bash
python -m unittest discover -s tests -p 'test_editorial.py'
python scripts/editorial.py --refresh
python scripts/editorial.py --publish-approved --receipt /tmp/editorial-receipt.json
# Commit/push only public docs before acknowledgement:
python scripts/editorial.py --ack --receipt /tmp/editorial-receipt.json
```

The local `.env` is not loaded by the publisher automatically. Supply credentials via environment variables; GitHub uses Secrets. A legacy misspelled `UPABASE_SERVICE_ROLE_KEY` secret is accepted as a fallback by the workflow, without exposing its value.

Missing odds/news fail closed per sport and remain visible in coverage notes. Started events, stale/future quotes, unpaired O/U lines and single-book markets are excluded. Movement compares medians across matched books only and is never labeled a true opening line. Public model freshness and injury feeds remain separate; news does not silently modify model estimates.

## Launch boundary

The public homepage and morning briefing can run before the private database migration. Until the SQL is installed and an owner is provisioned, the private desk clearly reports unavailable access/setup and does not pretend to save ideas. Owner email is pending user confirmation. Automated long-form analysis is authorized and runs independently of this optional private desk.

## Automated original analysis — budget revision, September 22

Enabled in `config/editorial.json`: **two published articles each morning, with a 6:30 AM Eastern delivery check**, exact model
`gpt-6-astra`, low reasoning, standard service tier. No fallback model. Runs on GitHub
Actions at **05:07 America/New_York**, without the owner's computer. GitHub may delay
scheduled jobs; publication follows research, factual checking and the Pages build.
Additional triggers at :27 and :47 from 5–9 AM recover delayed starts. Hourly runs also retry unfilled, eligible article slots; completed or uncertain paid attempts are not repeated. The independent Morning Article Delivery Watchdog checks at :23 and :53 from 5–9 AM, plus 6:33 AM and after MLB refreshes during the morning window. It dispatches recovery if no editorial run is active and reports an overdue edition as a failed check.

The allocator prefers leagues covered least recently, breaking ties in favor of
active boards and rotating ties daily. It tries other leagues if current sources
are unavailable. MLB/NFL/NBA/NHL all participate; no offseason filler is required.
Two published articles is the delivery target. Inadequate reporting, factual failures,
funding errors or budget limits must be visible as incomplete delivery, never a
successful edition. Quality and spending guards remain in force. Existing paid
failed/started slots are not blindly repeated; the delivery summary identifies
blocked states for recovery. Missing source evidence remains retryable.

The delivery check counts unique catalog URLs with existing public article files,
not writer invocations or drafts. It runs after publication so a partial edition and
its diagnostics are preserved before the workflow fails. Before 6:30 AM Eastern, an incomplete
edition is pending; from 6:30 AM it fails with the counts and blocking reasons in the
Actions summary. The workflow and watchdog also fetch the public article URLs and verify their
page titles, retrying during deployment propagation. An accepted Pages build
request alone does not count as successful public delivery.
GitHub cron is best effort; redundant triggers and a watchdog on the same platform
cannot guarantee delivery through a GitHub-wide outage.

### Research and quality

The collector reads public RSS from ESPN, CBS, Yahoo and (for MLB) MLB.com, then
fetches bounded excerpts from approved publisher hosts. Reports must be no older
than 72 hours; at least two publisher hosts are required. Blocked or unavailable
pages are skipped. General research searches up to 12 headlines per publisher,
and matchup research up to 30. If NFL feeds cannot supply two publishers, the
collector checks up to eight articles linked from the NFL's own news index,
requiring a NewsArticle publication date within 72 hours and readable article text; missing readable reporting is recorded in the ledger. There is no paid web-search tool. Two publishers repeating the
same report do not constitute independent corroboration.

Astra writes 550–750 words using those excerpts and fresh local market/model evidence.
A separate Astra request checks the draft against the original packet. Unsupported
injury facts, invented prices/model adjustments, unproven news-to-line causation and
repetitive angles fail review. News does not silently alter model estimates. A pass
or watchlist is allowed when there is no defensible bet. The automated check reduces
errors but is not a guarantee; the owner reviews the live site.

Source links and dates appear with each article. Public evidence snapshots include
our market/model data and source metadata, **not publisher excerpts**. API responses
remain in ignored `.editorial-cache/`; secrets never enter public files.

### Matchup-aware Fourth & Value evidence

Daily features now identify a target matchup when the reporting clearly points to
one, when a preview explicitly names one, or when a current eligible model pick can
be paired with matchup-specific reporting. Once a target is established, unrelated
model rows are removed before the evidence packet is compacted. Target-game market
records, eligible model picks and research-only model references are retained ahead
of unrelated/background evidence.

Eligible model evidence may come from player props, moneylines, spreads/run lines
or totals. The writer is instructed to use the most informative supported market
rather than defaulting to a total. Current model-pick records can carry the quoted
book/line, model probability, fair price, edge/EV and selected inputs; research-only
references remain clearly labeled and carry no stale price or EV.

Model availability is also recorded by event. If a target matchup has no forecast,
the packet preserves the model's explicit reason when available (for example,
waiting for both probable starters) rather than substituting a forecast from a
different game or vaguely implying that all model data are missing. Publication
validation rejects model evidence from another matchup when a target game is set.

Fresh future-game quotes have a six-hour maximum age. Background model references
must be at most six hours old and carry no stale quote or EV. Feature packets retain
model status and data cutoff dates; no validation result is inferred from missing
data. NBA/NHL research-only estimates must not be presented as established edges.

New previews leave the homepage when a referenced quoted game starts; otherwise
analysis expires from the homepage after three days. The blog retains the articles.
Opinions stay separate and are never generated in the owner's voice. Existing
Pirates prose is unchanged. A market-briefing lead fills an empty homepage safely.

### Spending guard and recovery

`docs/editorial/budget.json` enforces **$9 over a rolling seven days for this writer**.
This new policy excludes the already incurred six-story launch cost, other OpenAI
jobs, Odds API charges and taxes. It is not an organization-wide billing limit.
Prices are pinned to the official Astra pricing checked September 22: $12.50/M input
as a conservative allowance including cache writes, $50/M output. Recheck these rates
before changing models or if OpenAI pricing changes.

Each story allows one write (18,000 input UTF-8 bytes + framing allowance; 3,200
output tokens) and one audit (24,000 input bytes + framing; 1,000 output tokens).
No paid tools, retries or priority tier. A conservative full-story reservation with
10% headroom is recorded **before** the first call. CI commits and pushes the budget
and started slot before spending; checkpoint failure prevents the call. Settled
usage replaces the reservation using conservative rates without cached discounts.
Timeouts or missing usage retain the full reservation. When the next reservation
would cross $9, generation stops; price/homepage refreshes continue. This can yield
fewer than fourteen weekly articles. Actual costs are tracked, not promised.

CI checkpoints include completed public articles from earlier slots so a later crash
cannot publish only their catalog metadata. The final publication commit includes
settled budget, pages, catalogs, evidence and daily ledgers. Inspect
`docs/editorial/runs/YYYY-MM-DD.json` and Actions logs for skips or funding failures.
Never delete an uncertain reservation or started slot without checking API usage.

`writing_enabled: false` stops new paid articles. The manual workflow has an optional
`check_api` switch, default false: a tiny, budgeted live request against the actual
GitHub secret. Its reservation also persists first. A successful probe demonstrates
API access at that moment, not a guaranteed future credit balance. API prepaid
credits are separate from ChatGPT/Codex usage credits and spending-limit meters.

### Verification and manual operation

```
python -m unittest discover -s tests -p 'test_editorial*.py'
python scripts/editorial.py --refresh
python scripts/editorial_writer.py
python scripts/editorial.py
```

Tests cover freshness, escaping, citations, duplicate runs, publication/audit flow,
source handling, seven-day budget expiry, retained uncertain charges and checkpoint
failure. Local runs need environment credentials; scripts do not load `.env`.
Use the GitHub workflow for normal publication so reservations are durable before
paid calls. No paid retry is automatic.

### Historical launch

September 22's six launch features used Astra high and paid web research. Five had
a second paid audit; the sixth audit returned `credit_balance_exhausted` and was
checked directly against original reports before publication. The launch cost drove
this budget revision. The small local API probe subsequently succeeded; check the
cloud probe separately because GitHub may hold a different key.

### Verification for this release

All 25 editorial tests passed, as did desktop/mobile browser checks. Two bounded
private rehearsals cost approximately $0.15 and $0.16 under conservative accounting.
The first was rejected for calling shared-source reporting independent corroboration;
the instruction was corrected and the second completed writing, factual review and
HTML/evidence publication to `/private/tmp/fv-editorial-rehearsal/`, not the live site.
These test charges are included in the new budget. One successful rehearsal is not
a guarantee of future publication or a representative weekly cost benchmark.

Cloud verification found GitHub's old API secret returned `401 invalid_api_key`,
while the local key passed. The GitHub secret was replaced securely with the tested
local key. Explicit health-check failures now mark the workflow failed after price
publication, rather than appearing as an overall success. See the latest public
`/editorial/runs/health.json` for the confirmed cloud probe status.

## Market rundown refresh

Five price editions: 05:07, 06:37, 11:07, 16:07 and 21:07 America/New_York. Only the
two morning runs can generate articles; later editions use existing Odds API credentials
and public feeds, with no OpenAI calls. This adds up to twelve odds requests daily
above the previous single edition (provider quota still applies). Hourly renders
remove stale/started games from the latest briefing while retaining dated archives.

Cards select a mover, a book disagreement and the next distinct game, skipping
categories without evidence. Movement and disagreement rank proportionally to the
market median for cross-sport selection, not by betting value. Cards identify books
and prices at the lowest over total and highest under total; these are observed
line extremes, not EV recommendations. The briefing links recent original analysis
and shows counts of changed medians and differing book totals. No invented movement,
news causation or freshness timestamps are used to create the appearance of activity.

## Responsible-use notices

`python scripts/site_notices.py` applies static notices to public HTML without
changing authored analysis or publication dates. Use `--scope mlb`, `nba` or `nhl`
for league-only builds. All five publishing workflows run the applicator and its
coverage tests before committing public pages. The shared navigation also loads
`assets/responsible-use.js` as a fallback for new pages generated outside those jobs;
existing notices are detected to prevent duplicates. Keep its copy aligned with
`scripts/site_notices.py` when changing language. Styles are in
`assets/responsible-use.css`.

Betting tools, analysis, research and the tracker receive a concise notice near the
heading. Public pages receive a responsible-use footer linking to terms, responsible
play and NCPG support. Opinion/home/listing pages use only the footer. Redirects,
authoring templates and video-render canvases are excluded; video landing pages
are covered. Existing MP4s and off-site YouTube descriptions are not modified.

The terms explain uncertainty, stale quotes, backtests, hypothetical edges,
arbitrage/settlement risks, adult access versus local wagering eligibility, and
personal wagering responsibility. NCPG's current national resource was verified
September 22, 2026: https://www.ncpgambling.org/help-treatment/ (1-800-MY-RESET).
Privacy copy now describes private editorial submissions and distinguishes account
data from technical/provider data. AdSense remains inactive; provider-specific
privacy disclosures and applicable consent controls must be implemented before
activation. These changes are not a legal opinion or certification of enforceability.

## September 23: data before editorial publication

The earlier schedules let morning writing precede the MLB daily model update. An
input-size reducer could also remove every market row. Both are corrected. The
morning editorial job now calls the reusable MLB refresh workflow, waits for it to
finish and checks out the published main branch afterward. It then fetches the
current all-league price briefing, records per-league readiness, and only then writes.
Hourly homepage-only runs do not launch MLB or paid writing. Manual price/writer
refreshes follow the same MLB dependency. Failed data jobs do not prevent price
status publication, but cannot bypass the writer's readiness checks.

MLB also refreshes hourly at :15 from 10 a.m. through 10 p.m. America/New_York.
The 90-minute model-pick expiry is unchanged; scheduling delays can still leave
gaps, and we do not extend stale picks to disguise them. This increases Odds API
requests; OpenAI article count and the $9 rolling budget remain unchanged.

All analysis requires a fresh current-day briefing and an actual current market or
model record. MLB additionally requires successful quotes and model checks within
90 minutes, a ready model, and model history through yesterday's Eastern date.
Missing data is recorded before any paid call. Unspent waiting slots may be retried
after data recovery; previously attempted paid slots are never automatically retried.
The writer must cite at least one current evidence ID in its article. The input
reducer keeps at least one market and model row when available, reducing repeated
book quotes, background references and source excerpts first. Oversized requests
remain blocked by the existing budget limits.

NFL, NBA and NHL may use fresh price data without claiming a current or validated
forecast. The gate does not fabricate projections or force an offseason story.
`python scripts/editorial_writer.py --inspect-data` reports availability without
calling a paid API. Existing September 23 analysis remains dated; it is not silently
rewritten with later quotes.

The first morning sequence now starts at 05:07 Eastern, with a 06:07 catch-up.
Both refresh MLB before checking article readiness. The catch-up skips any slot
that already incurred a paid attempt; it only fills unattempted or data-waiting
slots within the same two-article daily maximum and budget. There is no guarantee
of an exact completion time from GitHub's scheduler. The failing cloud NFL odds
credential was replaced with the local credential verified against the NFL totals
endpoint, restoring the opportunity to cover NFL as well as MLB tomorrow.

## Recurring Thursday NFL preview

On Thursdays (America/New_York), reserve one of the two daily article slots for
the upcoming Thursday NFL matchup(s) in the odds feed. Friday UTC kickoffs count
as Thursday when appropriate in Eastern time. With multiple Thursday games, lead
with the evening game. No Thursday game means normal league rotation.

The preview uses only those games’ market snapshots and matching model records,
plus current team-specific reporting from multiple publishers. Cover verified
availability, matchup context, named book prices, model limitations, a countercase,
and a supported lean or pass. Do not manufacture a wager or injury-driven move.
Targeted collection checks up to 30 recent items per feed, matching team names in
headlines/URLs and excluding promotional offers. All usual source and data gates
remain in force. Unspent slots missing data/reporting can retry at 06:37; paid
attempts never automatically repeat. This replaces a daily slot, not an extra
paid article, and retains the rolling spending cap.

Regression checks: `python -m unittest discover -s tests -p 'test_editorial*.py'`.


## Resilient daily editorial scheduling

The daily workflow no longer decides whether to write by comparing
`github.event.schedule` to exact cron strings. The named morning schedules remain
useful target times, but GitHub scheduling delays or a missed morning event must not
silently suppress the writer.

Every scheduled run now begins with `scripts/editorial_schedule.py`, which classifies
the run from current state:

- current Eastern time
- today's published analysis count
- today's editorial slot ledger
- `funding_required`
- uncertain `started` slots
- briefing freshness
- MLB board/model freshness

Scheduled runs after 5:00 AM Eastern may fill an unattempted or
`waiting_for_data` slot. This includes the ordinary hourly maintenance schedule, so
a missed 5:07 or 6:37 event can be rescued later without changing the daily two-story
limit.

The planner does **not** automatically retry a slot that reached `started`, because
a paid request may already have occurred. It also does not bypass the global
`writing_enabled` switch, the daily publication limit, or a funding-required state.

MLB refreshes are state-aware. A retryable MLB slot or recoverable MLB data skip can
request a fresh MLB update when the current board is not already fresh. Briefing
refreshes are similarly based on freshness and writer need rather than one exact cron
expression.

### False-green guard

Whenever the planner says the writer is eligible, `editorial_writer.py` records a
writer marker in the daily ledger:

```json
{
  "last_writer_check": {
    "at": "...",
    "status": "completed",
    "counts": {
      "published": 0,
      "skipped": 0,
      "started": 0,
      "waiting_for_data": 1
    }
  }
}
```

The workflow then runs a separate verification step. If a writer was expected but a
recent **completed** marker is absent, the workflow fails instead of reporting green.

A completed marker does not mean an article had to publish. It means the writer
actually executed and reached a terminal scheduling result. A legitimate
`waiting_for_data`, factual-audit rejection, or funding failure remains visible in
the daily ledger and Actions logs.

### Diagnostics

The planner logs:

- run mode: `morning`, `catch-up`, `market-refresh`, `maintenance`, or `manual`
- Eastern planner timestamp
- whether the writer was eligible
- whether briefing/MLB refreshes were requested
- the reason the writer was or was not needed

Regression coverage explicitly tests an 8:00 AM Eastern hourly event with the
non-morning cron string and requires it to rescue an unattempted slot. It also tests
the pre-5 AM guard, retryable `waiting_for_data`, non-retryable `started` slots,
daily limits, funding state, MLB freshness, manual behavior, the writer kill switch,
and the completed-marker assertion.

## Phone ideas and reader submissions (September 24)

Owner flow: sign in at `/editorial/inbox.html`, enter a short angle and sport,
then Save idea. Title and body are optional. Submitted owner **analysis** ideas
are considered ahead of ordinary rotation when daily slots are allocated;
Thursday NFL priority remains first. One sport per daily edition still applies.
Already allocated/attempted slots are not replaced, so late ideas may wait until
the next day. Each researched idea uses one of the two existing daily paid
attempts and the same rolling $9 guard; no additional paid service is used.
A topic is not factual evidence. Fresh data, source collection and factual audit
must pass. Successful owner ideas publish directly and are archived privately.
Opinion ideas and general `Sports` ideas require personal editorial work; the
writer does not invent an owner's opinion. Submit analysis under its actual sport.

Reader flow: `/editorial/suggest.html` uses email sign-in, accepts at most three
short suggestions per account per day, and exposes no other reader's submissions.
Suggestions appear in the editor's inbox notification count. The editor can
archive one or Accept for research. Acceptance authorizes one budgeted research
attempt, **not publication**. The generated headline/body/sources are saved only
in Supabase as Review, attributed to Fourth & Value and stamped with the market
snapshot time. The editor can preview, edit and approve the exact saved draft.
The existing hourly publisher then publishes it. Reader identity and original
submission text are not exported to the public article or logs. Draft content
changes invalidate prior approval. Research failures archive the idea with a
private explanation instead of automatically charging again. A hard process crash
can leave Researching/started; inspect the ledger before manually recovering it.

### One-time activation

1. Execute `supabase/editorial_submissions.sql` in the existing project's SQL
   editor, after the already-installed `supabase/editorial.sql`. The browser page
   `/editorial/setup.html` provides an exact copy button. Existing rows are kept;
   reader origins are recorded immutably and require an editor's approval.
   Current server credentials cannot execute schema-management SQL, so the project
   administrator must perform this step. Do not rerun the old base SQL afterward.
2. Add `https://fourthandvalue.com/editorial/suggest.html` to Supabase Auth's
   allowed redirect URLs; retain the inbox URL. Sign out/in after changing roles.
3. For email, configure GitHub Actions secrets `RESEND_API_KEY` and
   `EDITORIAL_NOTIFY_FROM` (a sender address on a verified Resend domain).
   `EDITORIAL_NOTIFY_EMAIL` is already configured privately for the owner.
   Never commit the API key. Delivery uses https://api.resend.com/emails; see
   https://resend.com/docs/api-reference/emails/send-email.
   https://resend.com/docs/dashboard/emails/idempotency-keys documents the
   provider's 24-hour deduplication window. Persistent notification timestamps
   suppress ordinary repeats. A delivery followed by a failed database update
   lasting beyond that provider window can produce a duplicate alert.
4. Notification checks run with the hourly editorial workflow. Alerts contain a
   desk link, not private submission text. New reader suggestions and completed
   review drafts get separate alerts. No email is sent without configured
   credentials; the inbox counts remain available. Notification failure cannot
   block public market refreshes. This is an inbox badge, not OS push notifications.

### Verification

- `python -m unittest discover -s tests -p 'test_editorial*.py'` covers owner
  direct publication, reader draft-only routing, acceptance, privacy and email
  failure/idempotency behavior without paid API calls.
- `tests/editorial_permissions.cjs` executes both SQL files in isolated PostgreSQL
  (PGlite with pgcrypto). It verifies cross-account isolation, insert validation,
  server-enforced daily limits, editor acceptance, approval, approval invalidation,
  and service-only publishing. Set `FV_PGLITE` to the installed package directory.
- Browser checks at 390px and 1440px verify idea-only saves, the reader submission
  form, review controls and no horizontal overflow using isolated mock accounts.
- After activation, submit one real reader suggestion, check its inbox badge,
  confirm an email arrives, accept research, and verify the draft stays private
  until approved. Live email delivery is unverified until Resend is configured.

## Write now: explicit extra research, September 24

The phone desk now distinguishes Save idea (daily queue) from Write now (a
separate, explicit writing request). Select a saved submitted analysis idea under
NFL/MLB/NBA/NHL. Write now defaults to a private draft for approval. An owner can
select automatic publication for their own idea; a reader-originated idea cannot
bypass draft approval. A future earliest-publication date also prevents immediate
automatic publication. Opinion/general Sports submissions need personal editing.

An on-demand request can run after the two automatic daily slots are filled, but
still uses the shared rolling $9 budget. Separate UUID-keyed state and reservation
keys prevent duplicate paid attempts and filename collisions. Source/data checks
remain mandatory. A free data/source failure can be explicitly retried; a started
or terminal paid attempt cannot. On-demand-reserved ideas are excluded from the
automatic rotation, preventing an automatic publish while a draft was requested.
Failed dispatch keeps the idea reserved for explicit retry instead of silently
returning it to automatic publication. New drafts, including owner-requested ones,
are eligible for configured email notifications. No OS push notification is added.

### Activation

Use `/editorial/write-now-setup.html` for copy buttons and dashboard steps:
1. Run `supabase/editorial_write_now.sql` after the reader-submission migration.
2. Deploy `supabase/functions/editorial-write-now/index.ts` as the Edge Function
   `editorial-write-now`, keeping JWT verification enabled.
3. Set Edge Function secret `GITHUB_EDITORIAL_TOKEN`: a fine-grained GitHub token
   limited to this repository with Actions read/write. Supabase supplies its URL
   and anon key. Do not expose the dispatch token in HTML or JavaScript.
4. Test a saved idea from the desk, then verify the draft remains private.

Existing service-role credentials cannot deploy Edge Functions or execute SQL
schema migrations; a Supabase administrator must perform those setup steps.
Email still needs RESEND_API_KEY and EDITORIAL_NOTIFY_FROM in GitHub Actions.
The recipient is configured. Delivery is checked by the editorial workflow;
while the desk is open, inbox counts refresh every 60 seconds. GitHub queuing and
refresh processing mean Write now is a job request, not instant article output.

The function checks the caller's current Supabase editor metadata, restricts
origins to the production site, validates the saved idea, uses an optimistic
update to prevent simultaneous clicks, and keeps the GitHub token server-side.
SQL enforces a one-minute request cooldown and forces reader requests to drafts.
Only the idea UUID and publication preference reach GitHub workflow inputs.
Reference: https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event
and https://supabase.com/docs/guides/functions/secrets.

Verification includes `node tests/editorial_dispatch.cjs` (mocked network), the
PostgreSQL permission integration test with the additional migration, and Python
coverage proving explicit owner requests produce private drafts even with both
automatic daily slots already published. No live paid generation or email delivery
is claimed until the function and email credentials are configured and tested.

Write now feedback fix (September 24): action messages are also shown beside
the form buttons. Changing the immediate-publication checkbox does not mark the
article dirty; article edits still require saving, with an explicit hint. Function
errors display either the server error or message field. Browser checks at 390px
and 1440px verified default private requests, explicit owner publication, error
feedback, unsaved-edit blocking, and reader publication controls. The live dispatch was subsequently verified; see the requested-topic fix below.
A completed private draft remains unverified.

Requested-topic research fix (September 24): the first live Write now dispatch
worked, but the writer returned publish=false because a Jayden Daniels request
received Falcons–Packers / Jayden Reed evidence. No draft was produced. Submitted
ideas now retain their topic through source selection and compaction instead of
being replaced by a generic market pick. Unmatched named players use full-name
discovery hints, including hyphen-normalized URLs. No relevant two-publisher
reporting means waiting_for_data before budget reservation or a paid call. Topics
without recognizable player/team names wait for clarification. This is limited
feed discovery, not unrestricted web research; it cannot guarantee coverage of
every player. The previous paid attempt remains archived; do not clear its budget
reservation or trigger a paid retry implicitly. All 62 editorial tests pass,
including the Daniels/Reed topic regression and zero paid calls on missing sources.

Current pause/resume checklist: see EDITORIAL_HANDOFF.md (September 24, 2026).

## Private morning pipeline dashboard

Open `/editorial/diagnostics.html` from the editorial desk. It uses the existing
editor login; reports are protected by database row-level security, not merely an
unlisted URL. Sign in through the existing desk so no new Auth redirect is needed.

One-time setup: run `supabase/editorial_diagnostics.sql` in the existing Supabase
SQL editor. It adds one read-only-for-editors table. Existing idea, article and bet
tables are unchanged. Only server-side service credentials can write reports.
There are no new secrets required for the dashboard.

The editorial workflow records start and finish observations. The watchdog records
its own delivery check, including missing editions. Reports include GitHub stage
outcomes, per-sport odds/model freshness, selected and blocked articles, factual
review results, source-fallback provenance and the next eligible trigger. No private
idea text, draft bodies, publisher excerpts, model responses or secrets are stored.
An unknown or stale observation must not be displayed as a passing current check.
Earlier runs cannot retrospectively prove whether a fallback was used.

Reports are append-only across run attempts and phases, with idempotent updates for
the same observation. The date picker provides daily history; the snapshot picker
shows up to 100 observations for that day. The latest view refreshes each minute;
selecting an older snapshot pins it. If every workflow fails to start, the dashboard
shows no report or an old observation, not a healthy pipeline. A report-storage
failure appears as a failed diagnostics job; it does not block article publication.

Email summaries are deferred: `RESEND_API_KEY` and `EDITORIAL_NOTIFY_FROM` are not
currently configured. The existing recipient secret alone is insufficient to send.
Source fallback is recorded separately from story selection. The system still does
not repeat rejected/uncertain paid writing attempts or fabricate missing inputs.


## September 26: requested draft verification

The owner Write now journey has produced a private Review draft in a live run;
mobile/desktop preview and editing controls were checked using that draft in an
isolated browser. Approval and public delivery of this requested draft remain the
next owner-controlled step. See EDITORIAL_HANDOFF.md for current completion status.

Explicit NFL requests refresh their model feed even when daily slots are filled.
A completed writer without a draft/article now fails the requested-result check.
Full request size is checked before claiming a private idea or reserving money;
source excerpts are reduced within the unchanged byte caps. Paid and uncertain
attempts cannot automatically retry. A verified legacy pre-request zero-cost
failure can be explicitly released, retaining its earlier attempt in the ledger.

Requested MLB Wild Card overviews use a separate statistical format, with official
standings, home/away records and completed-game matchup history. This applies to
explicit requests only; daily matchup model requirements remain intact. No series
prices or retrospective model forecasts are invented. The historical archive scan
is limited to published editorial evidence, and this limitation accompanies the
records. Broader raw model-artifact indexing remains outstanding.

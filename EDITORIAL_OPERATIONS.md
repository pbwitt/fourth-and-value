# Editorial homepage and publishing

The homepage combines a curated lead, an automated market rundown, recent reporting, dated model features, opinion and results. Existing authored posts remain unchanged.

## Schedule and costs

`.github/workflows/editorial-daily.yml` collects a morning edition at **6:07 AM America/New_York**, adjusting for daylight saving time. GitHub schedules can be delayed; this is a target, not an exact-time promise. At **17 minutes past each hour**, it checks the private queue for approved articles and refreshes homepage freshness labels.

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

Enabled in `config/editorial.json`: **two article opportunities per day**, exact model
`gpt-6-astra`, low reasoning, standard service tier. No fallback model. Runs on GitHub
Actions at **06:07 America/New_York**, without the owner's computer. GitHub may delay
scheduled jobs; publication follows research, factual checking and the Pages build.
Hourly runs rebuild the homepage and latest briefing to remove expired games and prices, and publish approved owner drafts without paid article calls.

The allocator prefers leagues covered least recently, breaking ties in favor of
active boards and rotating ties daily. It tries other leagues if current sources
are unavailable. MLB/NFL/NBA/NHL all participate; no offseason filler is required.
Two is a maximum, not a guarantee: inadequate reporting, factual failures, funding
errors or budget limits can produce fewer articles. Existing daily slots, including
failed/started slots, never automatically repeat. Today's six launch articles count
as an already completed edition; the smaller format starts tomorrow.

### Research and quality

The collector reads public RSS from ESPN, CBS, Yahoo and (for MLB) MLB.com, then
fetches bounded excerpts from approved publisher hosts. Reports must be no older
than 72 hours; at least two publisher hosts are required. Blocked or unavailable
pages are skipped. There is no paid web-search tool. Two publishers repeating the
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

Fresh future-game quotes have a six-hour maximum age. Background model references
must be at most 24 hours old and carry no stale quote or EV. Feature packets retain
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

Four price editions: 06:07, 11:07, 16:07 and 21:07 America/New_York. Only the
morning run generates articles; later editions use existing Odds API credentials
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

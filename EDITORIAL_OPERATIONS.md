# Editorial homepage and publishing

The homepage combines a curated lead, an automated market rundown, recent reporting, dated model features, opinion and results. Existing authored posts remain unchanged.

## Schedule and costs

`.github/workflows/editorial-daily.yml` collects a morning edition at **8:07 AM America/New_York**, adjusting for daylight saving time. GitHub schedules can be delayed; this is a target, not an exact-time promise. At **17 minutes past each hour**, it checks the private queue for approved articles and refreshes homepage freshness labels.

The morning edition makes one full-game totals request per sport (NFL, MLB, NBA, NHL) with existing Odds API credentials and reads four public ESPN RSS feeds. It does **not** train models or generate new prop forecasts. Existing sports/model refreshes continue separately. The authorized Astra writer then researches and directly publishes original features, using the server-side `OPENAI_API_KEY` secret. See the original-analysis operating details below.

Each edition is at `/briefing/YYYY-MM-DD.html`; `/briefing/` is the latest. Data is stored under `docs/briefing/history/`, with immutable timestamped snapshots under `snapshots/`. The report covers NFL games within seven days and other sports within 48 hours. It labels market observations as research, not recommendations. ESPN headlines are linked, limited to 24 words and not expanded into unsupported facts. No article bodies are scraped.

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

## Original analysis — September 22 update

The owner authorized paid Astra generation and direct publication on September 22.
The automated newsroom does **not** depend on the Supabase review queue. The private
idea/opinion desk remains optional; its approval rules apply only to owner drafts.

- Exact model: `gpt-6-astra`; reasoning effort: `high`. No model fallback.
- Morning edition: 08:07 America/New_York, hosted on GitHub Actions. The computer can be off.
- Six maximum story slots: one for each NFL/MLB/NBA/NHL, plus two extra slots allocated
  to leagues with fresh upcoming totals. Currently this normally means two NFL and
  two MLB stories. NBA/NHL rotate into extra slots as their boards become active.
- Slots are opportunities, not a guaranteed article quota. Offseason coverage must
  have a real current roster, injury, schedule or market question. Unverifiable
  stories are skipped. Recent titles are supplied to discourage repetitive angles.
- Each article researches the live web, then a separate Astra call checks facts.
  Sources are displayed with section-level links and dates. At least two domains,
  one recent source, substantial depth and valid evidence references are required.
- Original analysis leads the homepage. ESPN RSS remains an input to the factual
  briefing, not the sole source for feature research.
- Only fresh future-game quotes enter an article's market packet (six-hour limit).
  MLB model rows must have `is_model_pick`; stale/unsupported predictions are omitted.
  NFL articles receive fresh totals plus dated model means from the public props export when within 24 hours. Saved model references carry no stale odds or EV; their limitations remain explicit. MLB references may similarly supply dated model inputs without a current recommendation.
  NBA/NHL unvalidated or withheld forecasts must never be called proven edges.
- Injury causation is a hypothesis unless supported by timestamped before/after
  quotes. No invented openers, adjusted probabilities or guaranteed bets.
- Evidence snapshots are public at `/editorial/evidence/`; article dates remain fixed.
  Published prices are historical observations, not a promise of current availability.

### Spend and failure controls

Only the morning edition or an explicit manual briefing refresh invokes the writer.
Hourly homepage updates make no paid writing calls. `writing_enabled` is the kill
switch in `config/editorial.json`. The existing GitHub `OPENAI_API_KEY` secret is used
server-side; it is never copied into HTML or browser JavaScript.

A daily ledger in `docs/editorial/runs/YYYY-MM-DD.json` stores the allocation,
started/published/skipped states, token usage and failure reasons. Completed or
started slots are not retried automatically, including on a same-day manual rerun.
Each story allows one research call (12,000 output tokens / 8 web tool calls) and
one audit (4,000 output tokens / 4 web tool calls). Six slots means at most twelve
model requests and 96,000 output tokens per normal edition, plus input and search
charges. These are ceilings, not expected usage or a dollar spending cap. Set an
API project budget separately if desired. No other paid insights job is enabled.

Network failures do not trigger paid retries. If the runner crashes before its ledger
is pushed, GitHub cannot recover the uncommitted ledger; inspect the API usage before
manually rerunning a failed job. Likewise, interrupted `started` slots require an
intentional ledger repair after checking usage. A failure does not publish an
unfinished article. Automated factual review reduces errors but cannot guarantee
correctness; the owner reviews the live site and can request corrections.

Local run (load the API key into the environment without printing it):

```
python scripts/editorial.py --refresh
python scripts/editorial_writer.py
python scripts/editorial.py
python -m unittest discover -s tests -p 'test_editorial*.py'
```

Publish the article HTML, evidence, ledger, catalog, homepage, blog index and sitemap
together. Never stage `.env` or unrelated local drafts. Existing Pirates analysis
is unchanged. The daily writer does not edit past articles or author opinions in
the owner's name.

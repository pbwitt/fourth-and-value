# Editorial homepage and publishing

The homepage combines a curated lead, an automated market rundown, recent reporting, dated model features, opinion and results. Existing authored posts remain unchanged.

## Schedule and costs

`.github/workflows/editorial-daily.yml` collects a morning edition at **8:07 AM America/New_York**, adjusting for daylight saving time. GitHub schedules can be delayed; this is a target, not an exact-time promise. At **17 minutes past each hour**, it checks the private queue for approved articles and refreshes homepage freshness labels.

The morning edition makes one full-game totals request per sport (NFL, MLB, NBA, NHL) with existing Odds API credentials and reads four public ESPN RSS feeds. It does **not** train models, generate new prop forecasts or call a paid writing model. Existing sports/model refreshes continue separately. `OPENAI_API_KEY` is deliberately absent from this workflow. Longer research articles are owner-reviewed; automated qualitative writing is not enabled in this release.

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

Existing features are listed in `config/editorial.json`. New approved articles flow into `docs/editorial/published.json`, which contains only public metadata. A featured analysis can become the homepage lead; opinion always remains distinctly labeled in the opinion column/archive. All approved analysis also appears in the blog index. The private desk is noindex and is linked discreetly from the homepage footer.

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

The public homepage and morning briefing can run before the private database migration. Until the SQL is installed and an owner is provisioned, the private desk clearly reports unavailable access/setup and does not pretend to save ideas. Owner email is pending user confirmation. Paid long-form drafting remains a separate future opt-in, not an incidental cost of collecting prices.

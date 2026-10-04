# Independent morning start

Status: **active since September 29, 2026** (24 of 24 morning dispatches accepted
through October 4). The 4:30 p.m. afternoon refresh uses the same mechanism; see
[Afternoon refresh](#afternoon-refresh). Merging GitHub changes never installs a
Supabase job. No new paid service or subscription is required by this setup;
existing platform quotas still apply.

## Why this exists

On September 28, 2026 no scheduled Morning Picks run appeared in the 07:05,
07:35, 08:05 or 08:35 Eastern windows. The owner dispatched run
[36424228385](https://github.com/pbwitt/fourth-and-value/actions/runs/36424228385)
at 08:48:36. Its first job started four seconds later; feeds and research succeeded,
the card was written at 08:55:32, and the run completed at 08:57:05. The only
scheduled run that day, [36466088838](https://github.com/pbwitt/fourth-and-value/actions/runs/36466088838),
was created at 14:34:02 and reused the completed morning edition. This identifies
late/missing scheduled-event delivery, not a model runtime or runner queue failure.
It does not establish GitHub's internal reason or which occurrences were dropped.

## Execution

1. GitHub keeps its existing four scheduled starts.
2. Once activated, Supabase's independent clock also requests the existing
   `morning-picks.yml` workflow at 07:05, 07:35, 08:05 and 08:35 Eastern. The job
   checks every five minutes in UTC 11:00–14:55; an Eastern-time SQL guard covers
   both daylight and standard time. A late tick requests only the current slot.
3. A private unique slot record prevents duplicate requests from repeated SQL
   execution or reinstall. There are at most four Supabase dispatches per day.
4. GitHub's existing `morning-picks-edition` concurrency group serializes starts.
   The existing gate skips feed pulls and paid research for a completed morning
   card, then verifies delivery. A missing/incomplete card follows the normal
   feed → model → discovery/review → publication sequence and the same $2.75 ledger.
5. PostgreSQL records whether the dispatch was accepted, rejected, timed out, or
   lacked a receipt. GitHub records the trigger source and gate execution time.
   An accepted dispatch does **not** mean that research or live publication passed;
   those remain checked by the existing Actions workflow.

This changes no model, ranking, freshness, research threshold or spending limit.
Neither trigger can force a test edition or replace a completed card. Additional
requests after a completed edition only run the gate/delivery checks. Failed or
incomplete research can still cause another feed pull within the normal budget.
No second automatic research edition is created.

## One-time activation (project administrator)

The available Supabase service-role credential provides application data access,
not SQL administration. Do not paste database passwords or tokens into chat.

1. Merge the PR containing the `trigger_source` workflow input. Installation
   before this merge will make GitHub reject the new input.
2. In GitHub Settings → Developer settings → Fine-grained personal access tokens,
   create a token restricted to **pbwitt/fourth-and-value**, with **Actions:
   read and write**. No Contents write, administration or other repository access
   is needed. Set an expiration you will maintain. Do not copy the broad local
   GitHub CLI credential. An existing suitably restricted token can also be used.
3. In the existing Supabase project, open Vault and add the token under the exact
   name **`fv_morning_github_token`**. Use Vault's secret form; never commit it or
   add it to a public site file. Enable **pg_cron** and **pg_net** in Integrations /
   Database Extensions if absent. Do not change the project's subscription.
4. Open SQL Editor as `postgres`, paste and run
   [supabase/morning_scheduler.sql](supabase/morning_scheduler.sql). It creates
   only its own private `fv_morning` schema and one named cron job; it does not
   change bets, articles, or existing scheduled jobs. Re-running is safe.

No Edge Function deployment is needed. This SQL sends to a fixed GitHub repository,
workflow and branch. Site visitors and application/service roles cannot invoke
the private function or read its dispatch records.

## Verify activation

Inspect the named job (do not print Vault values or HTTP request headers):

```sql
select jobname, schedule, active
from cron.job where jobname = 'fv-morning-dispatch';

select fv_morning.due_slot(now()); -- null outside 07:05–09:00 ET

select slot_at at time zone 'America/New_York' as slot_et,
       requested_at, request_id, response_status, result, received_at
from fv_morning.dispatches order by slot_at desc limit 12;
```

For an immediate end-to-end check **only after today's real morning card is
complete**, run the following once as administrator. It consumes one of today's
four request slots and uses ordinary inputs (no replacement/test/budget override):

```sql
select fv_morning.tick(
  ((now() at time zone 'America/New_York')::date + time '08:35')
  at time zone 'America/New_York'
);
```

Commit/finish that SQL call first: pg_net sends asynchronously after commit. Then,
in a separate SQL execution, run `select fv_morning.collect_receipts();` and inspect
the safe columns above. A `queued` result is not proof of acceptance. Expect a 2xx
HTTP status and a GitHub workflow summary with source `supabase`,
`edition_already_published`, skipped sports pulls, no paid calls, and successful
verification of the exact live edition. `already_requested` means that slot was
used; inspect its receipt rather than deleting it or resetting the ledger.

Finally verify the next automatic morning request and publication. Only then
update this status and the reader-facing daily-process page to say active.
Tomorrow's first full-path run cannot be proven by a completed-card smoke test.

## Failures and monitoring

`fv_morning.dispatches` keeps 90 days of safe metadata. `accepted` means GitHub
accepted the request; `rejected` with 401/403 usually requires checking token
expiry/permissions, while 422 can mean the workflow input is not deployed.
`timeout`, `transport_error` or `receipt_missing` require checking Supabase
network/pg_net health. Unknown requests are not repeated within a slot; later
slots provide bounded recovery. The raw response body and credentials are never
copied into the dispatch ledger.

Cron's SQL success only means the request was queued. Check the HTTP receipt and
the GitHub run separately. A next tick collects receipts before they expire from
pg_net; a database outage can prevent that, producing `receipt_missing` instead
of a false success. Inspect cron failures in the dashboard:

```sql
select r.start_time, r.end_time, r.status, r.return_message
from cron.job_run_details r join cron.job j using (jobid)
where j.jobname = 'fv-morning-dispatch'
order by r.start_time desc limit 12;
```

This is independent of GitHub **cron**, not of GitHub Actions/API or Supabase
availability. Paused projects, expired credentials, outages, missing odds,
incomplete research or Pages failures can still prevent publication. There is
no new email/SMS alert service. Existing Actions failures remain visible; a
failure to dispatch is recorded in Supabase. Maintain the token expiration and
project availability. Do not claim a guaranteed publication minute.

## Tests and rollback

```sh
npm install --prefix /tmp/fv-morning-db @electric-sql/pglite@0.5.8
FV_PGLITE=/tmp/fv-morning-db/node_modules/@electric-sql/pglite node tests/morning_scheduler.cjs
python -m unittest discover -s tests -p 'test_morning*.py'
python scripts/build_daily_process.py --check
```

The integration test executes the real PL/pgSQL in isolated PostgreSQL, with
pg_cron, pg_net and Vault interfaces mocked. It exercises DST, delayed ticks,
four-request bounds, deduplication/reinstall, fixed request contents, HTTP
receipts and private permissions. It does not prove hosted extension execution,
credential validity, or live dispatch; activation checks above cover those.

To disable only the independent trigger, deactivate `fv-morning-dispatch` in
Supabase Cron or run:

```sql
select cron.unschedule(jobid) from cron.job where jobname = 'fv-morning-dispatch';
```

Keep dispatch records, spending ledgers and published-card archives. Existing
GitHub schedules/manual dispatch continue unchanged. Do not uninstall shared
pg_cron/pg_net extensions or remove other jobs.

References: [Supabase Cron](https://supabase.com/docs/guides/cron/quickstart),
[pg_net](https://supabase.com/docs/guides/database/extensions/pg_net),
[Vault](https://supabase.com/docs/guides/database/vault), and
[GitHub dispatch API](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event).

## Afternoon refresh

From September 29 to October 4, 2026, GitHub's `30 16 * * *` America/New_York schedule
started the NHL and MLB afternoon refreshes between 19:11 and 20:27 Eastern, two and a
half to four hours late and after many games had started. Every run was created late,
so this is scheduled-event delivery, not a runner queue.

1. `supabase/afternoon_scheduler.sql` adds a second named job,
   `fv-afternoon-dispatch` (every five minutes, UTC 20:00–22:55). An Eastern-time
   guard requests `afternoon-refresh.yml` once for the 16:30 slot; a tick up to an
   hour late still catches it, and older slots never replay. It reuses the morning
   schema, ledger (`workflow` column), receipts, advisory lock and Vault token.
2. `Afternoon Market Refresh` calls the existing `nhl-daily.yml` and `mlb-daily.yml`
   reusable workflows. Those two no longer carry their own GitHub schedule.
3. A GitHub backup schedule at 16:45 Eastern remains. Its gate
   (`scripts/afternoon_gate.py`) skips a sport whose feed already published after
   16:00 Eastern that day and starts nothing after midnight, so a late backup does
   not spend a second round of odds credits. A manual (operator) start always runs.

Install after `afternoon-refresh.yml` is on main: open SQL Editor as `postgres` and
run [supabase/afternoon_scheduler.sql](supabase/afternoon_scheduler.sql). It requires
the morning installation and fails without changing anything if that is missing.
Re-running is safe.

```sql
select jobname, schedule, active from cron.job where jobname = 'fv-afternoon-dispatch';
select fv_morning.afternoon_slot(now()); -- null outside 16:30–17:30 ET
select slot_at at time zone 'America/New_York' as slot_et, workflow,
       requested_at, response_status, result
from fv_morning.dispatches where workflow = 'afternoon-refresh.yml'
order by slot_at desc limit 7;
```

Rollback: `select cron.unschedule('fv-afternoon-dispatch');` stops new requests and
leaves the ledger. The GitHub backup keeps running; to rely on it alone, restore the
16:30 schedule in `nhl-daily.yml` and `mlb-daily.yml` or move the backup earlier.

-- Independent afternoon trigger for the 4:30 p.m. Eastern NHL/MLB market refresh.
-- Run as postgres in the existing Supabase project, after supabase/morning_scheduler.sql
-- and once .github/workflows/afternoon-refresh.yml is on main. It reuses the morning
-- scheduler's private schema, dispatch ledger, receipts and restricted Vault token
-- (fv_morning_github_token: only this repository, Actions read/write).
-- No credentials belong in this file. No bet/editorial tables are accessed.
begin;

do $$
begin
  if to_regclass('fv_morning.dispatches') is null
     or to_regprocedure('fv_morning.collect_receipts()') is null then
    raise exception 'Install supabase/morning_scheduler.sql first';
  end if;
  if not exists (select 1 from vault.decrypted_secrets
                 where name = 'fv_morning_github_token' and length(decrypted_secret) > 20) then
    raise exception 'Create the fv_morning_github_token Vault secret first';
  end if;
end $$;

-- Which workflow each ledger row started; earlier rows are morning starts.
alter table fv_morning.dispatches add column if not exists workflow text not null default 'morning-picks.yml';

-- One slot a day at 4:30 p.m. Eastern. Postgres converts the time zone, so winter needs no
-- cron edit. A tick delayed up to an hour still catches the slot; older slots never replay.
create or replace function fv_morning.afternoon_slot(at_time timestamptz)
returns timestamptz language sql stable set search_path = '' as $$
  with local_time as (select at_time at time zone 'America/New_York' as t)
  select case when t::time >= time '16:30' and t::time < time '17:30' then
    (date_trunc('day', t) + interval '16 hours 30 minutes') at time zone 'America/New_York'
    else null end
  from local_time;
$$;

create or replace function fv_morning.afternoon_tick(at_time timestamptz default now())
returns jsonb language plpgsql set search_path = '' as $$
declare
  slot timestamptz := fv_morning.afternoon_slot(at_time);
  github_token text;
  queued_request bigint;
begin
  -- The morning lock also serializes this tick against the morning one and an operator's check.
  perform pg_advisory_xact_lock(460705, 1);
  perform fv_morning.collect_receipts();
  if slot is null then
    return jsonb_build_object('status', 'outside_window');
  end if;
  if exists (select 1 from fv_morning.dispatches where slot_at = slot) then
    return jsonb_build_object('status', 'already_requested', 'slot_at', slot);
  end if;
  select decrypted_secret into github_token from vault.decrypted_secrets
   where name = 'fv_morning_github_token';
  if github_token is null or length(github_token) <= 20 then
    raise exception 'Scheduler GitHub credential is missing; no request sent';
  end if;
  queued_request := net.http_post(
    url := 'https://api.github.com/repos/pbwitt/fourth-and-value/actions/workflows/afternoon-refresh.yml/dispatches',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || github_token,
      'Accept', 'application/vnd.github+json',
      'Content-Type', 'application/json',
      'User-Agent', 'FourthAndValue-AfternoonScheduler',
      'X-GitHub-Api-Version', '2026-03-10'),
    body := '{"ref":"main","inputs":{"trigger_source":"supabase"}}'::jsonb,
    timeout_milliseconds := 10000
  );
  insert into fv_morning.dispatches(slot_at, requested_at, request_id, workflow)
    values (slot, now(), queued_request, 'afternoon-refresh.yml');
  return jsonb_build_object('status', 'queued', 'slot_at', slot, 'request_id', queued_request);
end;
$$;

revoke all on all functions in schema fv_morning from public, anon, authenticated, service_role;

-- 36 short SQL checks a day and at most one outbound request. UTC 20-22 covers
-- 4:30-5:30 p.m. Eastern in summer and winter. The same named job is replaced on
-- reinstall. The workflow gate skips a sport that already refreshed this afternoon.
select cron.schedule('fv-afternoon-dispatch', '*/5 20-22 * * *', 'select fv_morning.afternoon_tick();');
commit;

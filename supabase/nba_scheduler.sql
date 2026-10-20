-- Independent trigger for the NBA market refresh at 6:45 a.m. and 5 p.m. Eastern.
-- Run as postgres in the existing Supabase project, after supabase/morning_scheduler.sql.
-- It reuses the morning scheduler's private schema, dispatch ledger, receipts and restricted
-- Vault token (fv_morning_github_token: only this repository, Actions read/write).
-- The request sends only the refresh_history input nba-daily.yml already accepts.
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

-- Which workflow each ledger row started (also added by supabase/afternoon_scheduler.sql).
alter table fv_morning.dispatches add column if not exists workflow text not null default 'morning-picks.yml';

-- Two slots a day: 6:45 a.m., so props are fresh in the morning and the run commits before the
-- 7:05 a.m. morning pipeline starts, and 5 p.m. Eastern. Postgres converts the time zone, so
-- winter needs no cron edit. A tick delayed up to an hour still catches the slot; older slots
-- never replay.
create or replace function fv_morning.nba_slot(at_time timestamptz)
returns timestamptz language sql stable set search_path = '' as $$
  with local_time as (select at_time at time zone 'America/New_York' as t)
  select case
    when t::time >= time '06:45' and t::time < time '07:45' then
      (date_trunc('day', t) + interval '6 hours 45 minutes') at time zone 'America/New_York'
    when t::time >= time '17:00' and t::time < time '18:00' then
      (date_trunc('day', t) + interval '17 hours') at time zone 'America/New_York'
    else null end
  from local_time;
$$;

create or replace function fv_morning.nba_tick(at_time timestamptz default now())
returns jsonb language plpgsql set search_path = '' as $$
declare
  slot timestamptz := fv_morning.nba_slot(at_time);
  github_token text;
  queued_request bigint;
begin
  -- The morning lock also serializes this tick against the morning and afternoon ones.
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
    url := 'https://api.github.com/repos/pbwitt/fourth-and-value/actions/workflows/nba-daily.yml/dispatches',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || github_token,
      'Accept', 'application/vnd.github+json',
      'Content-Type', 'application/json',
      'User-Agent', 'FourthAndValue-NBAScheduler',
      'X-GitHub-Api-Version', '2026-03-10'),
    body := '{"ref":"main","inputs":{"refresh_history":true}}'::jsonb,
    timeout_milliseconds := 10000
  );
  insert into fv_morning.dispatches(slot_at, requested_at, request_id, workflow)
    values (slot, now(), queued_request, 'nba-daily.yml');
  return jsonb_build_object('status', 'queued', 'slot_at', slot, 'request_id', queued_request);
end;
$$;

revoke all on all functions in schema fv_morning from public, anon, authenticated, service_role;

-- 60 short SQL checks a day and at most two outbound requests. UTC 10-12 covers
-- 6:45-7:45 a.m. Eastern and UTC 21-22 covers 5-6 p.m. Eastern, in summer and winter.
-- The same named job is replaced on reinstall. The workflow gate skips GitHub's late
-- backup starts once a slot has published.
select cron.schedule('fv-nba-dispatch', '*/5 10-12,21-22 * * *', 'select fv_morning.nba_tick();');
commit;

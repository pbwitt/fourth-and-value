-- Independent morning trigger. Run as postgres in the existing Supabase project.
-- Prerequisites: pg_cron + pg_net enabled, and Vault secret
-- fv_morning_github_token (only this repository, Actions read/write).
-- Install after the matching morning-picks.yml inputs are on main.
-- No credentials belong in this file. No bet/editorial tables are accessed.
begin;

do $$
begin
  if not exists (select 1 from pg_extension where extname = 'pg_cron')
     or not exists (select 1 from pg_extension where extname = 'pg_net') then
    raise exception 'Enable pg_cron and pg_net in Supabase Integrations first';
  end if;
  if not exists (select 1 from vault.decrypted_secrets
                 where name = 'fv_morning_github_token' and length(decrypted_secret) > 20) then
    raise exception 'Create the fv_morning_github_token Vault secret first';
  end if;
end $$;

create schema if not exists fv_morning;
revoke all on schema fv_morning from public, anon, authenticated, service_role;

create table if not exists fv_morning.dispatches (
  slot_at timestamptz primary key,
  requested_at timestamptz not null,
  request_id bigint unique not null,
  response_status integer,
  result text not null default 'pending',
  received_at timestamptz
);
alter table fv_morning.dispatches enable row level security;
revoke all on fv_morning.dispatches from public, anon, authenticated, service_role;

-- Timezone conversion happens in Postgres, so winter needs no cron edit.
-- A delayed tick catches the current slot once; it never replays older slots.
create or replace function fv_morning.due_slot(at_time timestamptz)
returns timestamptz language sql stable set search_path = '' as $$
  with local_time as (select at_time at time zone 'America/New_York' as t)
  select case when t::time >= time '07:05' and t::time < time '09:00' then
    (date_trunc('day', t) + interval '7 hours 5 minutes'
      + floor(extract(epoch from (t - date_trunc('day', t) - interval '7 hours 5 minutes')) / 1800)
        * interval '30 minutes') at time zone 'America/New_York'
    else null end
  from local_time;
$$;

-- Preserve only response codes/categories, never tokens, headers, or response bodies.
create or replace function fv_morning.collect_receipts()
returns void language plpgsql set search_path = '' as $$
begin
  update fv_morning.dispatches d
     set response_status = r.status_code,
         result = case when r.timed_out then 'timeout'
                       when r.error_msg is not null then 'transport_error'
                       when r.status_code between 200 and 299 then 'accepted'
                       else 'rejected' end,
         received_at = r.created
    from net._http_response r
   where r.id = d.request_id and d.result in ('pending', 'receipt_missing');
  update fv_morning.dispatches
     set result = 'receipt_missing'
   where result = 'pending' and requested_at < now() - interval '15 minutes';
  delete from fv_morning.dispatches where requested_at < now() - interval '90 days';
end;
$$;

create or replace function fv_morning.tick(at_time timestamptz default now())
returns jsonb language plpgsql set search_path = '' as $$
declare
  slot timestamptz := fv_morning.due_slot(at_time);
  github_token text;
  queued_request bigint;
begin
  -- Also serialize an administrator's check against the scheduled tick.
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
    raise exception 'Morning GitHub credential is missing; no request sent';
  end if;
  queued_request := net.http_post(
    url := 'https://api.github.com/repos/pbwitt/fourth-and-value/actions/workflows/morning-picks.yml/dispatches',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || github_token,
      'Accept', 'application/vnd.github+json',
      'Content-Type', 'application/json',
      'User-Agent', 'FourthAndValue-MorningScheduler',
      'X-GitHub-Api-Version', '2026-03-10'),
    body := '{"ref":"main","inputs":{"trigger_source":"supabase","test_edition":false,"replace_card":false}}'::jsonb,
    timeout_milliseconds := 10000
  );
  insert into fv_morning.dispatches(slot_at, requested_at, request_id)
    values (slot, now(), queued_request);
  return jsonb_build_object('status', 'queued', 'slot_at', slot, 'request_id', queued_request);
end;
$$;

revoke all on all functions in schema fv_morning from public, anon, authenticated, service_role;

-- 48 short SQL checks/day; at most four outbound requests. UTC 11-14 covers the
-- Eastern window in summer and winter, plus time to collect the final receipt.
-- The same named job is replaced on reinstall. The workflow gate prevents
-- duplicate research. Disabling this job does not alter existing GitHub starts.
select cron.schedule('fv-morning-dispatch', '*/5 11-14 * * *', 'select fv_morning.tick();');
commit;

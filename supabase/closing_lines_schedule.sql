-- pg_cron schedule for the closing-lines Edge Function. Run as postgres after
-- supabase/closing_line_value.sql and after deploying supabase/functions/closing-lines
-- (verify_jwt off). Prerequisites: pg_cron + pg_net, and a Vault secret named
-- closing_lines_secret (any long random string). No credentials belong in this file:
-- the secret is read from Vault at call time.
--
-- Live: every 5 minutes; each game is priced once, in its final 6 minutes.
-- Backfill: 10:00 UTC daily, past games still without a close (historical odds).
-- The function stops paid requests before the shared balance would fall below 2,000
-- credits and caps each run at 2,000.
begin;

do $$
begin
  if not exists (select 1 from pg_extension where extname = 'pg_cron')
     or not exists (select 1 from pg_extension where extname = 'pg_net') then
    raise exception 'Enable pg_cron and pg_net in Supabase Integrations first';
  end if;
  if not exists (select 1 from vault.decrypted_secrets
                 where name = 'closing_lines_secret' and length(decrypted_secret) > 20) then
    raise exception 'Create the closing_lines_secret Vault secret first';
  end if;
end $$;

select cron.schedule('closing-lines', '*/5 * * * *', $job$
  select net.http_post(
    url := 'https://fzjonxpzsrbdhbujbhsn.supabase.co/functions/v1/closing-lines',
    headers := jsonb_build_object('Content-Type', 'application/json',
      'x-closing-secret', (select decrypted_secret from vault.decrypted_secrets where name = 'closing_lines_secret')),
    body := '{}'::jsonb,
    timeout_milliseconds := 30000);
$job$);

select cron.schedule('closing-lines-backfill', '0 10 * * *', $job$
  select net.http_post(
    url := 'https://fzjonxpzsrbdhbujbhsn.supabase.co/functions/v1/closing-lines',
    headers := jsonb_build_object('Content-Type', 'application/json',
      'x-closing-secret', (select decrypted_secret from vault.decrypted_secrets where name = 'closing_lines_secret')),
    body := '{"mode":"backfill","limit":50}'::jsonb,
    timeout_milliseconds := 120000);
$job$);
commit;

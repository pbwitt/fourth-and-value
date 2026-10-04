-- Live Odds: the Odds API key for the live-odds Edge Function, kept in Supabase Vault.
--
-- The "Live Odds key" GitHub workflow copies the key the NHL refresh uses
-- (NHL_ODDS_API_KEY, else ODDS_API_KEY) here with the service role, and the
-- function reads it back with the service role. Browsers (anon, authenticated)
-- can call neither function. An ODDS_API_KEY Edge Function secret, if set,
-- takes precedence over this copy. Safe to re-run.

create or replace function public.set_live_odds_key(new_key text)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  existing uuid;
begin
  if coalesce(length(new_key), 0) < 10 then
    raise exception 'Invalid Odds API key';
  end if;
  select id into existing from vault.secrets where name = 'live_odds_api_key';
  if existing is null then
    perform vault.create_secret(new_key, 'live_odds_api_key', 'Odds API key for the live-odds Edge Function');
  else
    perform vault.update_secret(existing, new_key);
  end if;
end;
$$;

create or replace function public.live_odds_key()
returns text
language sql
stable
security definer
set search_path = ''
as $$
  select decrypted_secret from vault.decrypted_secrets where name = 'live_odds_api_key' limit 1
$$;

revoke all on function public.set_live_odds_key(text) from public, anon, authenticated;
revoke all on function public.live_odds_key() from public, anon, authenticated;
grant execute on function public.set_live_odds_key(text) to service_role;
grant execute on function public.live_odds_key() to service_role;

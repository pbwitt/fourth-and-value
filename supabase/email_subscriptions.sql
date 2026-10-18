-- Owner email subscriptions: the unsubscribe link in the Morning Card email.
--
-- scripts/morning_email.py (service role) creates one row per list and recipient and
-- reads `subscribed` before every send; it sends nothing when the row says false or
-- cannot be read. The email's unsubscribe link carries the row's random token to
-- /email/unsubscribe.html, which calls the two functions below with the public anon
-- key. Nobody can list, read or change rows directly: RLS is on with no policies and
-- table privileges are revoked. A function call with an unknown token changes nothing
-- and returns no row; neither function ever returns the email address.
--
-- Additive and safe to re-run. Apply in the SQL Editor as postgres (or as a migration).

create table if not exists public.email_subscriptions (
  list text not null check (list in ('morning_card')),
  email text not null check (email = lower(email) and position('@' in email) > 1),
  token uuid not null default gen_random_uuid() unique,
  subscribed boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (list, email)
);

alter table public.email_subscriptions enable row level security;
revoke all on table public.email_subscriptions from anon, authenticated;

create or replace function public.email_subscription_get(p_token uuid)
returns table (list text, subscribed boolean)
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select s.list, s.subscribed from public.email_subscriptions s where s.token = p_token;
$$;

create or replace function public.email_subscription_set(p_token uuid, p_subscribed boolean)
returns table (list text, subscribed boolean)
language sql
security definer
set search_path = public, pg_temp
as $$
  update public.email_subscriptions s
     set subscribed = p_subscribed, updated_at = now()
   where s.token = p_token and p_subscribed is not null
  returning s.list, s.subscribed;
$$;

revoke all on function public.email_subscription_get(uuid) from public;
revoke all on function public.email_subscription_set(uuid, boolean) from public;
grant execute on function public.email_subscription_get(uuid) to anon, authenticated;
grant execute on function public.email_subscription_set(uuid, boolean) to anon, authenticated;

notify pgrst, 'reload schema';

-- Fourth & Value — multi-user bet tracker schema
--
-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query)
-- after creating the project. Safe to re-run: uses "if not exists" / "or replace"
-- where possible, but the policies use "create policy" which errors if they
-- already exist - drop them first if you're re-running this after an edit.

create extension if not exists pgcrypto;

create table if not exists public.bets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),

  -- carried over 1:1 from the old bets.csv columns
  legacy_bet_id text unique,          -- original "bet_1762..." id, for one-time CSV migration only
  timestamp timestamptz,
  league text not null,
  game_date date,
  team_home text,
  team_away text,
  player text,
  market_type text,
  side text,
  line numeric,
  book text,
  odds numeric,
  stake_dollars numeric not null,
  status text not null default 'pending',   -- pending | won | lost | push
  actual_result numeric,
  payout numeric,
  graded_timestamp timestamptz,
  model_prob numeric,
  edge_bps numeric
);

create index if not exists bets_user_id_idx on public.bets(user_id);
create index if not exists bets_status_idx on public.bets(status);
create index if not exists bets_league_idx on public.bets(league);

alter table public.bets enable row level security;

-- A user can only ever see, add, change, or remove their OWN rows. This is
-- what actually enforces privacy - not app code, the database itself refuses
-- cross-user access even if the anon key is public in the site's JS.
create policy "select own bets" on public.bets
  for select using (auth.uid() = user_id);

create policy "insert own bets" on public.bets
  for insert with check (auth.uid() = user_id);

create policy "update own bets" on public.bets
  for update using (auth.uid() = user_id);

create policy "delete own bets" on public.bets
  for delete using (auth.uid() = user_id);

-- The grading scripts (grade_bets_nfl.py / grade_bets_nhl.py) run with the
-- service_role key, which bypasses RLS entirely by design - that's how they
-- update everyone's pending bets in one pass without needing per-user login.
-- The service_role key must NEVER be used in docs/ (the static site's JS) -
-- only in GitHub Actions secrets / local .env, never committed.

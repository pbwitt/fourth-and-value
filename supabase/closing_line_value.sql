-- Closing line value (CLV) for the bet tracker.
--
-- The closing-lines Edge Function fills these columns: it snapshots each bet's
-- market a few minutes before the game starts (or, for past games, from The Odds
-- API's historical endpoint) and records the market's no-vig closing price.
--
--   closing_fair_prob  median no-vig probability of the bet's side, at the bet's
--                      exact line, across US books
--   closing_odds       the bet's own book's closing American price, same line
--   closing_line       the line the market closed at for this side (equals the
--                      bet's line when captured; the consensus line when it moved)
--   clv_ev             expected ROI of the price taken against the fair close:
--                      closing_fair_prob * decimal(odds) - 1. +0.03 = +3% CLV
--   beat_line          over/under bets whose line moved: true when the close moved
--                      past the number taken (an over that closed higher)
--   clv_status         null = not yet processed; captured | line_moved | no_market
--                      | no_event | unsupported
--
-- Applied to the production project as migration 20261005102623 (closing_line_value);
-- this file is that migration's text, committed for review and rebuilds.

alter table public.bets
  add column if not exists event_id text,
  add column if not exists commence_time timestamptz,
  add column if not exists closing_line numeric,
  add column if not exists closing_odds numeric,
  add column if not exists closing_fair_prob numeric,
  add column if not exists closing_books integer,
  add column if not exists closing_captured_at timestamptz,
  add column if not exists clv_ev numeric,
  add column if not exists beat_line boolean,
  add column if not exists clv_status text,
  add column if not exists clv_note text;

alter table public.bets drop constraint if exists bets_clv_status_check;
alter table public.bets add constraint bets_clv_status_check
  check (clv_status is null or clv_status in
    ('captured', 'line_moved', 'no_market', 'no_event', 'unsupported'));

create index if not exists bets_clv_pending_idx
  on public.bets (game_date) where clv_status is null;

-- Signed-in users may edit their own bets, so this trigger keeps the closing data
-- server-owned: clients cannot write it, and changing what the bet IS (market,
-- side, line, player, game, book) clears it so the job captures it again.
-- Changing only the price or stake keeps the close and recomputes clv_ev.
-- clv_ev and beat_line are derived here on every write, from whoever.
create or replace function public.bets_clv_guard()
returns trigger
language plpgsql
set search_path = ''
as $$
declare
  dec numeric;
  reset boolean := false;
begin
  if current_user in ('anon', 'authenticated') then
    if tg_op = 'INSERT' then
      reset := true;
    elsif (new.league, new.market_type, new.side, new.line, new.player,
           new.game_date, new.team_home, new.team_away, new.book)
          is distinct from
          (old.league, old.market_type, old.side, old.line, old.player,
           old.game_date, old.team_home, old.team_away, old.book) then
      reset := true;
    else
      new.event_id := old.event_id;
      new.commence_time := old.commence_time;
      new.closing_line := old.closing_line;
      new.closing_odds := old.closing_odds;
      new.closing_fair_prob := old.closing_fair_prob;
      new.closing_books := old.closing_books;
      new.closing_captured_at := old.closing_captured_at;
      new.clv_status := old.clv_status;
      new.clv_note := old.clv_note;
    end if;
    if reset then
      new.event_id := null;
      new.commence_time := null;
      new.closing_line := null;
      new.closing_odds := null;
      new.closing_fair_prob := null;
      new.closing_books := null;
      new.closing_captured_at := null;
      new.clv_status := null;
      new.clv_note := null;
    end if;
  end if;

  dec := case
    when new.odds is null or new.odds = 0 then null
    when new.odds > 0 then 1 + new.odds / 100.0
    else 1 + 100.0 / abs(new.odds)
  end;
  new.clv_ev := case
    when new.closing_fair_prob is null or dec is null then null
    else round(new.closing_fair_prob * dec - 1, 4)
  end;
  new.beat_line := case
    when new.closing_line is null or new.line is null
      or new.closing_line = new.line then null
    when lower(new.side) = 'over' then new.closing_line > new.line
    when lower(new.side) = 'under' then new.closing_line < new.line
  end;
  return new;
end;
$$;

drop trigger if exists bets_clv_guard on public.bets;
create trigger bets_clv_guard
  before insert or update on public.bets
  for each row execute function public.bets_clv_guard();

-- Raw odds snapshots the job paid for, kept so CLV can be recomputed later with a
-- different method (sharper books, line conversion) without paying again.
-- RLS on with no policies: only the service role reads or writes it.
create table if not exists public.odds_snapshots (
  id bigint generated always as identity primary key,
  captured_at timestamptz not null default now(),
  source text not null check (source in ('live', 'historical')),
  sport_key text not null,
  event_id text not null,
  snapshot_at timestamptz,
  markets text[] not null,
  credits_used integer,
  payload jsonb not null
);
alter table public.odds_snapshots enable row level security;
create index if not exists odds_snapshots_event_idx on public.odds_snapshots (event_id);

-- The cron job calls the function with a shared secret kept in Vault; the
-- function checks it here, so no secret has to be copied into Edge Function
-- settings by hand.
create or replace function public.closing_lines_secret_ok(candidate text)
returns boolean
language sql
security definer
set search_path = ''
as $$
  select coalesce(candidate, '') <> '' and exists (
    select 1 from vault.decrypted_secrets
    where name = 'closing_lines_secret' and decrypted_secret = candidate);
$$;
revoke all on function public.closing_lines_secret_ok(text) from public, anon, authenticated;
grant execute on function public.closing_lines_secret_ok(text) to service_role;

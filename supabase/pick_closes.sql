-- Closing prices for every published pick (morning and afternoon editions), for the
-- weekly closing-line value report card in each sport's recap.
--
-- scripts/pick_closes.py registers each edition's picks once it is verified live,
-- using the same ticket fields as Bet Tracker (briefing-picks.js ticketData), so the
-- closing-lines Edge Function prices them exactly like tracked bets and shares each
-- game's snapshot with them. The closing columns mean what they mean on public.bets
-- (supabase/closing_line_value.sql); decision_market_prob is the market's no-vig chance
-- the pick was decided against, so the recap can report how far the market moved.
--
-- RLS on with no policies: only the service role reads or writes it. Published picks
-- are public, but nothing outside our jobs needs this table.

create table if not exists public.pick_closes (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  edition_id text not null,
  edition_kind text not null check (edition_kind in ('morning', 'afternoon')),
  decision_date date not null,
  pick_index integer not null,
  policy_version text,
  published_at timestamptz not null,
  league text not null,
  market_type text not null,
  side text not null,
  line numeric,
  player text,
  book text not null,
  odds numeric not null,
  game_date date not null,
  team_home text not null,
  team_away text not null,
  decision_market_prob numeric,
  final_prob numeric,
  event_id text,
  commence_time timestamptz,
  closing_line numeric,
  closing_odds numeric,
  closing_fair_prob numeric,
  closing_books integer,
  closing_captured_at timestamptz,
  clv_ev numeric,
  beat_line boolean,
  clv_status text check (clv_status is null or clv_status in
    ('captured', 'line_moved', 'no_market', 'no_event', 'unsupported')),
  clv_note text,
  unique (edition_id, pick_index)
);
alter table public.pick_closes enable row level security;
create index if not exists pick_closes_pending_idx on public.pick_closes (game_date) where clv_status is null;
create index if not exists pick_closes_date_idx on public.pick_closes (decision_date, league);

-- clv_ev and beat_line are derived on every write, as on public.bets.
create or replace function public.pick_closes_derive()
returns trigger
language plpgsql
set search_path = ''
as $$
declare
  dec numeric;
begin
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

drop trigger if exists pick_closes_derive on public.pick_closes;
create trigger pick_closes_derive
  before insert or update on public.pick_closes
  for each row execute function public.pick_closes_derive();

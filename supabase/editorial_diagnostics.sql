-- Run once in the existing Supabase project's SQL editor. Safe to rerun.
-- This adds one private table; it does not change articles, ideas or bets.
begin;
create table if not exists public.editorial_pipeline_reports (
 id text primary key,
 edition_day date not null,
 observed_at timestamptz not null,
 report jsonb not null check (jsonb_typeof(report) = 'object'),
 created_at timestamptz not null default now()
);
create index if not exists editorial_pipeline_reports_day
 on public.editorial_pipeline_reports (edition_day, observed_at desc);
alter table public.editorial_pipeline_reports enable row level security;
revoke all on public.editorial_pipeline_reports from public, anon, authenticated;
grant select on public.editorial_pipeline_reports to authenticated;
grant all on public.editorial_pipeline_reports to service_role;
drop policy if exists "editors read pipeline reports" on public.editorial_pipeline_reports;
create policy "editors read pipeline reports" on public.editorial_pipeline_reports
 for select to authenticated
 using ((auth.jwt()->'app_metadata'->>'fv_editor') = 'true');
notify pgrst, 'reload schema';
commit;

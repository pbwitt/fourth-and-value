-- One-time setup in the existing project's Supabase SQL editor.
-- No email address or private draft is committed here.
begin;
create extension if not exists pgcrypto;
create table if not exists public.editorial_ideas (
 id uuid primary key default gen_random_uuid(),
 user_id uuid not null default auth.uid() references auth.users(id),
 created_at timestamptz not null default now(),
 updated_at timestamptz not null default now(),
 kind text not null default 'analysis' check(kind in ('analysis','opinion')),
 sport text not null default 'NFL' check(sport in ('NFL','MLB','NBA','NHL','Sports')),
 idea text not null check(length(idea) between 1 and 12000),
 title text not null default '' check(length(title)<=180),
 body text not null default '' check(length(body)<=60000),
 byline text not null default '' check(length(byline)<=100),
 sources text not null default '' check(length(sources)<=12000),
 featured boolean not null default false,
 publish_on date,
 status text not null default 'submitted' check(status in ('submitted','researching','review','approved','publishing','published','archived')),
 approved_hash text,
 published_url text
);
alter table public.editorial_ideas enable row level security;
revoke all on public.editorial_ideas from anon, authenticated;
grant select,insert,update on public.editorial_ideas to authenticated;
grant all on public.editorial_ideas to service_role;
drop policy if exists "editor owns ideas" on public.editorial_ideas;
create policy "editor owns ideas" on public.editorial_ideas for all to authenticated
 using (user_id=auth.uid() and (auth.jwt()->'app_metadata'->>'fv_editor')='true')
 with check (user_id=auth.uid() and (auth.jwt()->'app_metadata'->>'fv_editor')='true');
create or replace function public.guard_editorial_approval() returns trigger language plpgsql
set search_path=public,extensions as $$
declare changed boolean; content_hash text;
begin
 new.updated_at=now();
 content_hash=encode(digest(convert_to(jsonb_build_array(new.title,new.body,new.byline,new.sources,new.kind,new.featured,new.publish_on)::text,'UTF8'),'sha256'),'hex');
 if TG_OP='INSERT' then
   new.status=case when length(trim(new.body))>0 then 'review' else 'submitted' end; new.approved_hash=null; new.published_url=null;
 else
   if old.status='publishing' and auth.role()<>'service_role' then raise exception 'Publication is in progress; retry editing after it finishes.'; end if;
   changed = (new.title,new.body,new.byline,new.sources,new.kind,new.featured,new.publish_on)
      is distinct from (old.title,old.body,old.byline,old.sources,old.kind,old.featured,old.publish_on);
   if changed then new.status='review'; new.approved_hash=null; end if;
   if new.status='approved' and old.status<>'approved' then
     if auth.uid() is distinct from new.user_id or old.status<>'review' or length(trim(new.title))=0 or length(trim(new.body))<100 or length(trim(new.byline))=0 then
       raise exception 'Save a complete draft, then approve it as its owner.';
     end if;
     new.approved_hash=content_hash;
   elsif new.status='approved' and not changed then new.approved_hash=old.approved_hash;
   end if;
   if new.status='publishing' and (auth.role()<>'service_role' or old.status not in ('approved','publishing') or changed) then
     raise exception 'Only the publisher can claim an approved version.';
   end if;
   if new.status='published' and (auth.role()<>'service_role' or old.status<>'publishing' or new.approved_hash is distinct from old.approved_hash) then
     raise exception 'Only the publisher can mark an approved draft published.';
   end if;
   if new.status not in ('approved','publishing','published') then new.approved_hash=null; end if;
 end if;
 return new;
end $$;
drop trigger if exists editorial_approval_guard on public.editorial_ideas;
create trigger editorial_approval_guard before insert or update on public.editorial_ideas
 for each row execute function public.guard_editorial_approval();
commit;

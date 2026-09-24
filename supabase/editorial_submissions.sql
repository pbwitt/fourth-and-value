-- Run once after editorial.sql. Existing private ideas/drafts are preserved.
begin;
alter table public.editorial_ideas add column if not exists research_error text;
alter table public.editorial_ideas add column if not exists requires_review boolean not null default false;
alter table public.editorial_ideas add column if not exists approved_by uuid references auth.users(id);
alter table public.editorial_ideas add column if not exists research_requested_at timestamptz;
alter table public.editorial_ideas add column if not exists notification_sent_at timestamptz;
alter table public.editorial_ideas add column if not exists draft_notification_sent_at timestamptz;
-- Backfill the trusted approving account for already-approved legacy drafts.
update public.editorial_ideas set approved_by=user_id where status in ('approved','publishing','published') and approved_by is null;
drop policy if exists "editor owns ideas" on public.editorial_ideas;
drop policy if exists "read own ideas or editor inbox" on public.editorial_ideas;
create policy "read own ideas or editor inbox" on public.editorial_ideas for select to authenticated
 using(user_id=auth.uid() or (auth.jwt()->'app_metadata'->>'fv_editor')='true');
drop policy if exists "submit own idea" on public.editorial_ideas;
create policy "submit own idea" on public.editorial_ideas for insert to authenticated
 with check(user_id=auth.uid());
drop policy if exists "edit ideas as editor" on public.editorial_ideas;
create policy "edit ideas as editor" on public.editorial_ideas for update to authenticated
 using((auth.jwt()->'app_metadata'->>'fv_editor')='true')
 with check((auth.jwt()->'app_metadata'->>'fv_editor')='true');
create or replace function public.guard_editorial_approval() returns trigger language plpgsql
set search_path=public,extensions as $$
declare changed boolean; content_hash text;
begin
 new.updated_at=now();
 content_hash=encode(digest(convert_to(jsonb_build_array(new.title,new.body,new.byline,new.sources,new.kind,new.featured,new.publish_on,new.idea,new.sport)::text,'UTF8'),'sha256'),'hex');
 if TG_OP='INSERT' then
   new.created_at=now();
   new.requires_review=coalesce((auth.jwt()->'app_metadata'->>'fv_editor')::boolean,false) is not true;
   new.approved_by=null; new.research_requested_at=null;
   new.notification_sent_at=null; new.draft_notification_sent_at=null; new.research_error=null;
   if new.requires_review then
     if length(new.idea)>2000 or new.title<>'' or new.body<>'' or new.byline<>'' or new.sources<>'' or new.featured or new.publish_on is not null then
       raise exception 'Reader submissions contain an idea, sport and type only.';
     end if;
     -- Serialise this account's submissions before checking the daily limit.
     perform pg_advisory_xact_lock(hashtextextended(auth.uid()::text,0));
     if (select count(*) from public.editorial_ideas where user_id=auth.uid() and created_at>now()-interval '24 hours')>=3 then
       raise exception 'Please limit suggestions to three per day.';
     end if;
   end if;
   new.status=case when length(trim(new.body))>0 then 'review' else 'submitted' end; new.approved_hash=null; new.published_url=null;
 else
   if new.user_id is distinct from old.user_id or new.requires_review is distinct from old.requires_review then
     raise exception 'Submission ownership and review requirement cannot change.';
   end if;
   new.approved_by=old.approved_by;
   if new.research_requested_at is distinct from old.research_requested_at then
     if coalesce(auth.jwt()->'app_metadata'->>'fv_editor','false')<>'true' then
       raise exception 'Only an editor can request research.';
     end if;
     if old.status<>'submitted' or old.kind<>'analysis' then raise exception 'Only submitted analysis ideas can enter research.'; end if;
     new.research_requested_at=now();
   end if;
   if old.status in ('publishing','researching') and auth.role()<>'service_role' then raise exception 'Publication is in progress; retry editing after it finishes.'; end if;
   changed = (new.title,new.body,new.byline,new.sources,new.kind,new.featured,new.publish_on,new.idea,new.sport)
      is distinct from (old.title,old.body,old.byline,old.sources,old.kind,old.featured,old.publish_on,old.idea,old.sport);
   if changed then new.status=case when length(trim(new.body))>0 then 'review' else 'submitted' end; new.approved_hash=null; end if;
   if new.status='approved' and old.status<>'approved' then
     if coalesce(auth.jwt()->'app_metadata'->>'fv_editor','false')<>'true' or old.status<>'review' or length(trim(new.title))=0 or length(trim(new.body))<100 or length(trim(new.byline))=0 then
       raise exception 'Save a complete draft, then approve it as its owner.';
     end if;
     new.approved_hash=content_hash; new.approved_by=auth.uid();
   elsif new.status='approved' and not changed then new.approved_hash=old.approved_hash;
   end if;
   if new.status='publishing' and (auth.role()<>'service_role' or old.status not in ('approved','publishing') or changed) then
     raise exception 'Only the publisher can claim an approved version.';
   end if;
   if new.status='published' and (auth.role()<>'service_role' or old.status<>'publishing' or new.approved_hash is distinct from old.approved_hash) then
     raise exception 'Only the publisher can mark an approved draft published.';
   end if;
   if new.status not in ('approved','publishing','published') then new.approved_hash=null; new.approved_by=null; end if;
 end if;
 return new;
end $$;
commit;

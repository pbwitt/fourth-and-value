-- Run after editorial_submissions.sql. Existing ideas are preserved.
begin;
alter table public.editorial_ideas add column if not exists write_now_requested_at timestamptz;
alter table public.editorial_ideas add column if not exists write_now_publish boolean not null default false;
create or replace function public.guard_editorial_write_request() returns trigger
language plpgsql set search_path=public,extensions as $$
begin
 if TG_OP='INSERT' then
   new.write_now_requested_at=null; new.write_now_publish=false;
 elsif (new.write_now_requested_at,new.write_now_publish) is distinct from (old.write_now_requested_at,old.write_now_publish) then
   if coalesce(auth.jwt()->'app_metadata'->>'fv_editor','false')<>'true' or old.status<>'submitted' or new.status<>'submitted' or old.kind<>'analysis' then
     raise exception 'Only an editor may request writing for a submitted analysis idea.';
   end if;
   if old.write_now_requested_at>now()-interval '1 minute' then raise exception 'Writing was just requested; wait before retrying.'; end if;
   new.write_now_requested_at=now();
   -- A reader's origin can never be converted to automatic publication.
   if old.requires_review or old.user_id is distinct from auth.uid() then new.write_now_publish=false; end if;
 end if;
 return new;
end $$;
drop trigger if exists editorial_write_request_guard on public.editorial_ideas;
create trigger editorial_write_request_guard before insert or update on public.editorial_ideas
 for each row execute function public.guard_editorial_write_request();
commit;

-- Reader story suggestions closed (November 20, 2026).
--
-- /editorial/suggest.html no longer offers a form, and the database now refuses new
-- rows from anyone but an editor, so a signed-in reader cannot submit through the API
-- either. Editors (app_metadata.fv_editor = true) keep adding their own ideas and
-- Write-now requests from the inbox; the service role is unaffected by RLS. Existing
-- submissions, the inbox's read policy and editor updates are unchanged.
--
-- Additive and safe to re-run. Apply in the SQL Editor as postgres (or as a migration),
-- after editorial_submissions.sql. To reopen reader suggestions, restore that file's
-- "submit own idea" policy.
begin;
drop policy if exists "submit own idea" on public.editorial_ideas;
drop policy if exists "editors submit ideas" on public.editorial_ideas;
create policy "editors submit ideas" on public.editorial_ideas for insert to authenticated
 with check(user_id=auth.uid() and (auth.jwt()->'app_metadata'->>'fv_editor')='true');
commit;

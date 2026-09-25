# Editorial desk handoff — September 24, 2026

## Product direction

Patrick wants a simple phone-friendly way for friends to participate in Fourth &
Value: submit a few sentences, have the site research an article, and retain owner
control over what gets published. He may also use the ChatGPT app for his own
GitHub-backed post requests. The desk is valuable because friends can contribute
without GitHub access. They should never need a title or a finished article.

User paused work until tomorrow. Do not start another paid writing attempt as
part of this handoff. No corrected paid retry has been run.

## Verified tonight

- Owner sign-in and editor access work.
- Supabase schema and Write now columns exist.
- GITHUB_EDITORIAL_TOKEN is saved in Supabase; the fine-grained token is scoped
  to this repository with Actions read/write.
- The editorial-write-now Edge Function now runs the intended code. Initially,
  starter authentication code rejected the site's key; replacing the function
  contents resolved that failure. Do not repeat token setup unnecessarily.
- The owner's Write now click successfully dispatched GitHub run 36076646069.
- That run called the writer, but produced NO draft and published nothing.
- Its idea was archived to prevent repeated paid attempts. Private error text
  now explains the actual evidence mismatch.
- UI feedback is visible beside the buttons. The publish checkbox no longer
  marks the article dirty or disables Write now. Tested mobile and desktop.

## Failure and fix

The idea asked about Jayden Daniels. Generic market targeting replaced the
research with Falcons–Packers reporting about Jayden Reed. The model returned
publish=false because it could not verify the requested player injury and market.
This was an evidence-selection bug, not exhausted API credits.

Commit 63cb155 on main fixes requested-topic selection: full-name discovery hints
for unmatched players, no automatic replacement with a generic game pick, and
waiting before budget reservation/paid writing if relevant reporting is absent.
All 62 editorial tests passed, including a no-paid-call missing-source regression.
A read-only live source collection found two Daniels-specific reports after the
fix. This DOES NOT prove an article can pass all market and factual-audit gates.
Source discovery remains limited to configured publisher feeds; arbitrary player
coverage and ambiguous/lowercase-only names are not guaranteed.

## Tomorrow: finish one end-to-end test

1. Check current git main and queue before editing. Worktree used tonight:
   /private/tmp/fv-editorial-home. The original /Users/pwitt/fourth-and-value
   checkout has unrelated user work; do not overwrite it.
2. Owner creates a new NFL idea because the original paid attempt is archived:
   “Analyze Jayden Daniels’ injury, Washington’s upcoming matchup, and whether
   current betting lines reflect his availability.”
3. Save idea, then Write now, with automatic publication UNCHECKED. This is a
   new paid attempt within the existing weekly cap; do not silently clear the
   prior budget reservation or rerun the old request.
4. Watch the new GitHub dispatch and private Supabase queue. Verify the packet
   contains relevant reporting AND the correct matchup/market before claiming
   success. Fix remaining selection issues if needed, not by weakening audits.
5. Confirm a private REVIEW draft appears, inspect it, preview it, and test owner
   approval/publication only when the content is ready and publication intended.
6. Test the friend flow with a separate signed-in account: submit, owner sees it,
   accept for research or Write now, draft requires owner approval. Verify friends
   cannot read other private ideas or publish their own submissions.

## Notifications and moderation still outstanding

- In-app queue counts refresh about every 60 seconds while the desk is open.
  This is not phone/OS push notification support.
- Notification email recipient is configured in GitHub as EDITORIAL_NOTIFY_EMAIL.
- RESEND_API_KEY and EDITORIAL_NOTIFY_FROM (verified sender) were still missing
  at last check. Email delivery has NOT been verified. Recheck secrets by name
  only; never print their values.
- Email notifications are intended for new reader submissions and ready drafts.
- Blocking unwanted users is NOT implemented. Archive is available; submission
  limits and private row policies exist. Discuss block/report moderation before
  opening participation broadly.
- Friends' drafts always require owner approval. Owner ordinary saved analysis
  ideas can enter daily automatic publishing; explicit Write now defaults to a
  private draft unless the owner selects publication. Opinion needs review.

## Budget and useful links

Two automatic writing attempts/day; rolling seven-day $9 guard. Write now is an
extra explicit attempt under the same weekly guard. Failures do not automatically
repeat paid calls. No model/effort or budget increase was made tonight.

- Owner desk: https://fourthandvalue.com/editorial/inbox.html
- Friend submissions: https://fourthandvalue.com/editorial/suggest.html
- Function setup: https://fourthandvalue.com/editorial/write-now-setup.html
- Detailed operations: EDITORIAL_OPERATIONS.md
- Original archived idea: c0fedfa6-cf60-4ac0-95cb-b99a831d20c9
- Original workflow: https://github.com/pbwitt/fourth-and-value/actions/runs/36076646069

Keep private article content in Supabase; do not publish reader emails or
unapproved drafts in git, public logs, or generated site evidence.

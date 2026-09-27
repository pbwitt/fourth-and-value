# Fourth & Value audience pilot

Prepared September 27, 2026. Repository reference: `403df3269a361cdd05d5a3dd3bc7010c1cf84fa3`.

## What is ready

The owner authorized preparing and carrying out a small organic outreach pilot.
Paid advertising is **on hold at the owner's request** while the site builds
readership. Do not pursue ad accounts, approvals, quotes, or campaigns until the
owner explicitly resumes that work. This package contains a researched venue
shortlist, original post drafts, an access-request draft, and a measurement plan.
No posts, account applications, outreach messages, ad purchases, or recurring
promotion jobs have been sent or activated. Advertising spend authorized: **$0**.
The $100 example and creative in [ADS.md](ADS.md) are retained only as deferred
reference material, separate from all existing research and editorial budgets.

This branch changes only promotion documents. It does not change the site's
morning runs, models, picks, public pages, dependencies, or spending settings.

## First pilot

1. The owner handles X. The X drafts in [OUTREACH.md](OUTREACH.md) are available
   for the owner's use; the assistant must not publish to X or pursue its app
   setup unless the owner explicitly changes that division of work.
2. Identify an existing Reddit or forum account and establish an authorized
   publishing connection and any required platform access. A public profile link
   identifies an account but does not provide publishing access.
3. Make at most one community contribution during the first week, only in a
   suitable current thread where the rules permit it. Include the full useful
   explanation and disclose the affiliation. Omit the site link unless allowed.
   A removed contribution ends that venue's pilot; do not repost it elsewhere in
   that community or use another account.
4. After seven days from the first publication, record reach, meaningful replies,
   actual spending, attributable visits where measurable, and what readers found
   useful. Decide whether to continue from that evidence. There is no automatic
   increase in posting frequency or spend.

The frequency above is our pilot choice, not a claim about a platform's limits.
There is no daily posting quota. This is a plan for execution once access is
available, not a scheduled background task.

## Files

| File | Purpose |
| --- | --- |
| [OUTREACH.md](OUTREACH.md) | Verified venue rules, post drafts, access route, and Reddit request text |
| [ADS.md](ADS.md) | Deferred paid-ad reference; no active paid-ad work |
| [activity.csv](activity.csv) | Empty activity ledger; append actual actions, not scheduled intentions |

## What still needs an external answer

| Item | Why it matters | Current state |
| --- | --- | --- |
| Reddit account/profile | Needed for community participation | No account identified |
| Reddit commercial/API approval if using an agent | Owner permission alone does not establish platform access | Not applied for; request draft prepared |
| Visitor measurement | Clicks alone cannot demonstrate retained readership | No working attribution pipeline established in this audit |

Do not request passwords in chat. Use a supported account authorization flow when
a suitable integration is chosen. Any additional account/service fees need to fit
an explicitly agreed budget; they are not part of the picks research allowance.

## Execution status

The owner explicitly renewed permission to carry out organic engagement after
putting paid ads on hold. Inspection then found the existing
`scripts/post_tweet.py` and locally configured X credentials. The read-only
`GET https://api.x.com/2/users/me` check returned HTTP 403, `Client Forbidden`,
reason `client-not-enrolled`. X's response says the app supplying the keys and
tokens must be attached to a developer Project. A second read-only request
confirmed that reason. No publishing request was sent, and no credentials were
printed or committed.

The owner subsequently said they will handle X. That instruction supersedes
the earlier request to fix the X app and the assistant's plan to publish X1.
The access-check result is retained as history, not an active assistant task.
X publishing and account setup are now the owner's responsibility. Organic
outreach to other communities remains authorized, subject to account access and
the relevant community and platform rules.

There is no running engagement monitor or scheduled posting task. Forum posting
also remains unavailable without an identified account and applicable platform
access. The activity ledger stays empty until a public contribution is actually
published; API checks are not engagement.

## Evidence and messaging

Lead with transparent sports research, exact lines and prices, and useful
explanations. The [published NHL study](https://fourthandvalue.com/research/nhl-forecasting-and-market-pricing.html)
reports improved player count predictions against a simple baseline, with
important evaluation limitations. It does **not** establish a profitable betting
edge. The outreach drafts preserve that distinction.

The [current process page](https://fourthandvalue.com/research/daily-process.html)
is the source for operational descriptions. Do not promise a publication minute,
a fixed number of picks, complete slate research, a guaranteed outcome, or proven
ROI. A price-comparison signal is not a validated prediction. Do not use a model's
extreme probability as advertising proof.

Rules were consulted on September 27 through public primary-source pages; some
community pages were cached by the search provider. Recheck the exact current
thread and rules before publishing. No community moderator has approved this
pilot. Do not portray an unverified venue as an accepted placement.

Preparation checks: relative document links resolve; both X drafts fit 280
characters with a shortened link; the hypothetical price example's arithmetic
is correct; the activity ledger has no fabricated entries. Both proposed live
landing pages returned HTTP 200 via curl, with the expected page titles and
responsible-use links. The published paper supports the draft's limited claims.
These checks do not establish publishing access, ad approval, or attribution.

## Activity and reporting

Record UTC publication time, channel, account, exact destination/post URL, draft
ID, the rules URL and check time, actual cost, attribution URL, and outcome in
`activity.csv`. Store platform credentials outside the repository. For drafts,
use these documents; an empty ledger means nothing has been published.

Use this short report after actual activity:

> Period: [dates]. Published: [count and links]. Spend: [$ actual / $ authorized].
> Reach and clicks: [measured figures, or unavailable]. Attributable visits and
> engaged/returning readers: [measured figures, or unavailable]. Useful feedback:
> [summary]. Removed/rejected items: [count and reason]. Next action: [continue,
> revise, pause, or no suitable opportunity].

Do not turn unavailable metrics into zero, combine ad clicks with verified site
visits, or infer campaign ROI without a measured business outcome.

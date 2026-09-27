# Community and owned-channel pilot

Research date: September 27, 2026. All copy below is a draft; nothing was posted.

## Venue decisions

| Venue | What the primary source establishes | Pilot decision |
| --- | --- | --- |
| [Fourth & Value on X](https://x.com/fourthandvalue) | The repository links to this handle and has a posting script with locally configured credentials. Verification returned HTTP 403, `client-not-enrolled`; the authenticated account remains unverified. | First owned-channel candidate once the app is attached to a developer Project and the account check succeeds. Use X1 and X2 on separate days. |
| [r/sportsbook](https://www.reddit.com/r/sportsbook/) | Community rules reject promotion-only contributions and require the full analysis on Reddit. Individual recurring threads impose additional requirements. | Conditional candidate for a relevant methods/discussion thread. R1 below is not a Pick of the Day entry. Check that thread's rules; no automatic permission to add a brand link. |
| [r/algobetting](https://www.reddit.com/r/algobetting/) | The rules prohibit advertising and reject short-run bragging about individual picks/days. | Excluded from promotional posting. Do not disguise a marketing post as an educational answer. |
| [r/hockey](https://www.reddit.com/r/hockey/wiki/selfpromotion/) | Its self-promotion guide requires meaningful participation and three other posts for each self-promotion post; it also bars sales. | Deferred. A new account should not begin with a betting promotion. Genuine hockey-research discussion may fit later, subject to current rules. Do not manufacture participation to satisfy a ratio. |

These are researched possibilities, not moderator endorsements. A community
being relevant to our subject does not make it available for promotion. The
r/hockey wiki's statement about a site-wide ratio is not adopted here as current
Reddit-wide policy.

## Publishing access

Reddit's [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy)
requires approval for API access and explicit written approval for commercial
uses. Its app rules cover AI agents, require transparency, and prohibit automated
spam, including substantially similar posts across communities. Consequently,
this package does not implement a Reddit scraper or posting bot. An owner using
Reddit's normal interface still has to follow the community rules; that is
distinct from obtaining access for an automated agent.

Use the existing `scripts/post_tweet.py` for the first X post once its app
enrollment is corrected and the branded account is verified. The initial
connection audit missed this repository capability; the execution check is
recorded in README.md. No replacement integration is needed to attempt that
route.

X's [automation rules](https://help.x.com/en/rules-and-policies/x-automation)
permit informational automated posts subject to its rules. They prohibit
unsolicited automated replies based on keyword searches and require explicit
prior approval for AI reply bots. This pilot therefore starts with an original
post on the owned account; it does not launch a reply bot or use website
scripting to work around the API enrollment failure.

The connected-tool directory search found no Reddit publishing plugin. It found
Metricool as an optional social-publishing integration, but that does not establish
Reddit support or usable access to our X account. Its [X access guide](https://help.metricool.com/x-twitter-access-in-metricool-2bt66)
and [X add-on guide](https://help.metricool.com/your-guide-to-the-x-twitter-add-on-wt5wy)
describe a paid-plan requirement and a $10/month per-account X add-on. No plugin,
subscription, or account was connected or purchased. Choose access after the
owner identifies the account; avoid buying a scheduler merely to publish two
posts.

### Reddit commercial access inquiry — unsent

Use the commercial-contact route linked from the Responsible Builder Policy.
Supply the owner's actual business/contact/account details through the form;
none are invented here.

> Fourth & Value publishes sports forecasting research, sportsbook price
> comparisons, and dated betting analysis at https://fourthandvalue.com. We do
> not accept wagers. We are requesting guidance and approval for a small,
> commercial, AI-assisted community participation use case.
>
> The proposed scope is to read a bounded set of relevant public discussion
> threads and, where a community allows it, publish original answers through a
> clearly identified Fourth & Value app/account. Contributions would include
> their substantive analysis on Reddit, disclose the affiliation, and include
> a site link only where permitted. We would not send unsolicited private
> messages, manipulate voting, repeat posts across communities, build user
> profiles, or train models on Reddit content.
>
> Please confirm the appropriate access route, identification requirements,
> permissible scope, retention/deletion obligations, and any commercial terms
> or fees. No automated access or posting will begin under this proposal until
> the required approval is obtained.

## Original post drafts

### X1 — methods paper

Post from the site's own branded account. Source: the linked paper, especially
its abstract, player results, and limitations. The text plus link fits a normal
short X post using X's shortened-link length.

> We tested four seasons of NHL forecasts. Player predictions improved on a
> simple baseline, but a betting edge remains unproven.
>
> What would you want to see before trusting a model?
>
> https://fourthandvalue.com/research/nhl-forecasting-and-market-pricing.html

### X2 — how the morning card works

Source: the linked process page. Check it again before posting because operating
details can change.

> What goes into Fourth & Value's morning card? Forecasts, exact sportsbook
> offers and research into the case for each candidate. Fewer than ten picks—or
> none—is a valid result. Here's the process:
>
> https://fourthandvalue.com/research/daily-process.html

### R1 — a complete native explanation

Use only in a current thread inviting this topic. There is deliberately no
outbound link in the text. This is a hypothetical pricing example, not a live
pick, a win-rate claim, or a substitute for thread-specific eligibility.

**Suggested title, if standalone methods posts are allowed:** Why liking the
outcome and liking the price are separate decisions

> Disclosure: I work on Fourth & Value, a sports-analysis site.
>
> Here's a hypothetical example of the distinction we try to make. Suppose a
> well-supported forecast gives an under a 55% chance of winning at a half-point
> line, so there is no push. At -110, a $100 stake wins $90.91 or loses $100. Its
> estimated average result is 0.55 × $90.91 − 0.45 × $100 = +$5. At -135, the same
> outcome and probability give approximately −$4.26 per $100 staked.
>
> The forecast did not change, but the offer did. That doesn't mean the 55% is
> reliable: workload, a missing injury update, or poor calibration could erase
> the entire estimated advantage. A second book posting a different line isn't
> direct confirmation of 55%, either. You need the exact line and both prices
> to make a comparable market assessment.
>
> For an integer line, a push also matters: expected profit is the probability
> of winning times the net payout, minus the probability of losing times the
> stake. A refunded push contributes zero. Comparing a non-push market
> probability with an unconditional win probability mixes two different things.
>
> Our NHL research has not established a profitable edge over sportsbooks.
> This framework is useful because it makes the assumptions visible and leaves
> room to pass when the evidence is weak.

The arithmetic above was checked directly. Do not change the illustrative 55%
into a claim about today's model. If a reader asks for a source and links are
allowed, reference the NHL paper transparently; do not insert promotional links
into unrelated replies.

### P1 — publisher sponsorship inquiry — deferred

Paid placements are on hold at the owner's request. This draft is retained for
reference; do not select a recipient or send it unless the owner resumes paid
advertising. If resumed, use only the publisher's advertising contact, not
private messages to individual community members.

> Hello — I work on Fourth & Value, which publishes sports forecasting research,
> price comparisons and dated betting analysis. We're considering a small,
> clearly labeled sponsorship introducing readers to our NHL methodology paper.
> The paper explains its limitations and does not claim a proven betting edge.
>
> Do you accept this type of advertiser? If so, could you share the placement,
> audience geography and age eligibility, rate/minimum spend, labeling policy,
> and the reporting provided? We would need to agree on the full cost and scope
> before booking anything. Thank you.

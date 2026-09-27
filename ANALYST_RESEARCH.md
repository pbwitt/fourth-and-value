# Morning candidate research

The briefing first uses each sport's existing model-and-price eligibility rules.
It then takes at most four candidates per sport, one per game, and requests a
sourced Astra critique. Nothing guarantees a daily pick or a profitable strategy.
The model estimates, market prices, AI review and human decision remain separate.

## Implementation and contracts

`docs/assets/briefing-picks.js` owns selection in both the browser and the Node
adapter `scripts/analyst_shortlist.cjs`. The Python runner does not approximate
that policy. NFL and MLB feeds and routes retain their existing contracts.
`/briefing/reviews.json` is an additive schema-version-1 feed. Expand **Astra
analysis** beneath a candidate to read sourced evidence, the countercase and
checks that could invalidate the thesis. Bet Tracker remains beside each row.
The independent price rundown is unaffected.

**Top picks analysis**, above the table, presents up to three short prose summaries of completed reviews. It starts with the leading reviewed candidate per sport, then fills any remaining space in existing model order. It repeats the stored interpretation, countercase and first open check without another AI request, changing model scores or inventing claims. Source links, quote/review times and changed-offer warnings remain visible. Each paragraph opens the full review, which contains every evidence item and open check. Missing or expired reviews produce an explicit waiting/empty state. Open reviews stay open during the 30-second freshness refresh when their offer remains on the list.

Each review retains the game, player, market, side, line, book, actual odds,
quote timestamp, forecast timestamp and probability values it assessed. The
browser binds review status to this exact identity. A refreshed price/forecast
can retain earlier same-bet context, explicitly labelled **needs recheck**.
Changed line, player, book, game or day cannot inherit that review. Sources and
review times remain visible; old reviews expire after 12 hours and games expire
at their scheduled start. Failed research does not make an old forecast current.

NFL probabilities are conditional on non-push settlement and already incorporate
market calibration. They are not independent forecasts; its consensus may
include the offered book. MLB probabilities are unconditional wins with a
separate push mass; its other-book market reference is conditional on non-push
settlement. No conversion is silently inferred when push mass is missing.
This change does not repair or revalidate the legacy NFL model.

## Evidence and analysis

The shared NHL collector reads ESPN/CBS league RSS plus official MLB/NFL news
indexes. It retrieves at most eight articles per sport, at most two relevant
sources per candidate, published within 72 hours and retrieved before the
review. Publication and retrieval times are separate. Opportunity/injury reporting is ranked before generic team coverage; betting-pick/promotional headlines are excluded. Matching uses full player
names or full team names/nicknames, never a city or player surname alone.
Article text is untrusted input. Redirects are restricted to allowed HTTPS
publishers; no subscriptions, credentials or access restrictions are bypassed.

MLB questions cover starters, batting order, opportunity, bullpen usage,
handedness, park/weather and settlement. NFL questions cover participation,
snap/route/carry role, quarterback/offensive line, matchups and weather. These
are questions, not claims that those inputs are available or useful. RSS and
news coverage is incomplete; verified lineup, injury or weather data may be
missing. MLB hitter props require published batting orders under the existing
model policy, so the morning list can omit them until a later update.

The strict response schema requires a countercase and explicit checks. Positive
or adverse research status requires supporting source evidence. Source IDs,
candidate IDs and exact short excerpts are checked locally; unsupported fields,
fabricated excerpts, future timestamps and numeric confidence are rejected.
`represented_in` flags possible double counting with model inputs/market prices.
Citation validation does not independently prove an interpretation is correct:
the analyst must verify it. Astra cannot adjust probability, EV, fair price or
stake and cannot approve a wager. No automatic support/concern betting rule is
claimed to improve results.

## Schedule, spending and failures

`Morning Candidate Research` runs after standalone MLB/NFL refreshes and at
05:45, 08:45, 10:45, 12:45 and 15:45 America/New_York. GitHub execution can be
delayed. At most one paid attempt per sport in 05:00–12:00 and a distinct later
12:00–18:00 window. Empty slates, missing sources/keys, or expired candidates
make no API call. Outside those windows only current status is published.

NHL and MLB/NFL share **$5 per rolling seven days**, counted across the existing
NHL ledger and `artifacts/analyst/budget.json`. The existing editorial article
budget is separate and unchanged. Both review workflows share a concurrency
group and local file lock. NFL/MLB budget priority alternates by date; shortlist
ranking never changes. Budget exhaustion leaves visible unreviewed candidates.

Model `gpt-6-astra`, standard service tier, low reasoning, no tools, no automatic
retries, maximum 26,000 serialized request bytes including schema/instructions,
maximum 4,200 output tokens. Conservative reservation prices all request bytes
as tokens plus overhead at $12.50/M input and $50/M output, with 10% margin.
Maximum reservation is under $0.62/call; actual usage is normally lower. A timeout
retains its reservation. CI commits/pushes reservation and exact packet before
calling the API. A failed checkpoint prevents payment. Each sport can require
roughly two minutes of bounded source retrieval plus up to four minutes API time.
No new paid feed or hosting service is required.

API details checked against official documentation on 2026-09-27:
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) and
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Reproduction and prospective evaluation

Python 3.11, requests 2.32.5 and Node 22; existing full test dependencies are in
`requirements.txt`. `OPENAI_API_KEY` is read only from environment/GitHub Secrets.
For local authorized execution `--env-file` also requires python-dotenv.

```sh
node tests/briefing_picks.cjs
python -m unittest discover -s tests -p 'test_analyst_review.py'
python -m unittest discover -s tests -p 'test_nhl_analyst.py'
python scripts/analyst_review.py          # selection/status only, no paid call
python scripts/analyst_review.py --astra  # bounded call only when eligible
```

Archives retain the original pre-review shortlist, serialized model/source
packet, response/usage, and published result as immutable JSON in
`artifacts/analyst/{boards,requests,responses,published}`. Packets include the
actual excerpts sent; public reviews omit full article excerpts. The production
archive is committed and also uploaded for 90 days. These observations start a
prospective cohort; they are not retrospective evidence of qualitative uplift.

Bet Tracker records your executed odds and stake. It does not mark research as
human-verified, and its current schema does not save a human review decision or
research ID. Match an exported bet to the archived game/player/market/line/book
and quote only when identity is unambiguous. Without a timestamped human decision
and complete outcome grading, do not claim causal improvement from intervention.
Automatic MLB and NHL moneyline/puck-line grading remain unconnected; those bets
are logged pending, as the existing tracker dialog states. No private bets are
read or written by the research runner.

## Release and rollback

Merge the tested branch, verify Pages publication and `/briefing/` at mobile and
desktop widths, then dispatch `analyst-daily.yml`. Verify its public statuses and
the budget/request archive. A zero-candidate result is valid and makes no call.
Check `review_unavailable`, `budget_exhausted`, missing reporting and upstream
feed freshness in Actions and the public feed; do not label missing analysis as
completed. Source availability and interpretive quality require ongoing review.

Disable `astra_enabled` in `config/analyst_review.json` to stop new MLB/NFL paid
calls. The model board and tracker continue functioning. To revert the feature,
revert its merge commit and rebuild Pages; retain spend and research archives so
rollback/redeployment cannot erase previous charges. Preserve both ledgers when
changing concurrency or retention. No database migration is required.

## Decision log

- Reuse the existing NHL collector, validator, client and conservative budgeting;
  keep sport-specific prompts and model semantics explicit.
- Share the browser's real screen instead of maintaining a second Python screen.
- Retain model forecasts unchanged. No historical records establish numerical
  qualitative adjustments or a superior betting rule.
- Preserve dated earlier context when price/forecast changes, with a recheck
  label; never silently transfer an approval to another offer.
- Keep the existing combined research cap, rather than adding $5 per league.
- Published batting-order requirements remain in force; no fabricated morning
  opportunity estimate substitutes for a missing lineup.

## Release validation (2026-09-27 UTC)

The real Astra API accepted the two-sport synthetic integration packet and returned two schema-valid reviews without probability changes: 1,534 input tokens, 575 output tokens, conservatively charged $0.047925 against the research ledger. This was an integration check, not analysis of actual bets. A separate read-only source connectivity check retrieved two timestamp-verified articles for each sport; that checks transport and parsing, not comprehensive injury/lineup coverage. The production-input run found no eligible current candidates and made no paid call.

Automated checks cover NHL regression behavior, the shared browser screen, MLB/NFL research identity and timestamps, source validation, spending limits, API failures and duplicate attempts. Browser fixtures cover sourced concern, changed price, unavailable research, stale feeds, expired quotes and tracker saves at 390/768/1440px. No real bets or sign-in emails were created.

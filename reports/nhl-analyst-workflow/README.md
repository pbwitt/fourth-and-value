# NHL morning research workflow

Implementation plan (2026-09-26): keep the existing forecast models and Market Watch
contract; add an independent, frozen quantitative candidate screen; collect dated
public reporting; ask GPT-6 Astra to critique those candidates against that evidence;
record a separate human decision; preserve every stage for prospective evaluation.
Integrate a new NHL Top Picks page using the existing site components. Run contract,
selection, source, API-failure, budget and browser tests before a draft PR.

## Decision log

- All current NHL forecasts remain experimental. A research candidate is neither a
  validated recommendation nor a placed bet. Model probabilities and pricing survive
  Astra and human review unchanged. There is no invented qualitative probability lift.
- Fixed morning policy: today's Eastern regular-season games, independent model,
  verified settlement, quotes/forecasts at most 30 minutes old, inputs under 36 hours,
  model EV at least 2%, offered price meets the existing worst-scenario minimum,
  positive worst-scenario log-growth rank. At most four candidates, one per game.
  These are prospective operating choices, not profit-optimized backtest thresholds.
- Market consensus remains an independent diagnostic; lack of consensus cannot
  replace a missing model. Market Watch stays a separate price-comparison board.
- Astra reviews a bounded evidence packet. No paid search tools, autonomous wagers,
  generated source URLs or unsourced news. Dated sources retrieved after the original
  forecast are available only at the later review time; their timestamps are retained.
- Shared NFL/NHL concept: model and price screen → research → analyst decision.
  NHL's stronger timestamp, settlement and exposure guards are explicit; this change
  does not assert that the older NFL screen uses identical thresholds or validation.
- A rolling seven-day $5 cap and at most one paid morning batch are proposed. A later
  quantitative refresh does not automatically trigger another paid review. No request
  on empty slates. Requests reserve their full bounded cost before submission; uncertain
  failures retain the reservation and cannot be retried automatically.
- No historical qualitative backtest is claimed. Archive the original shortlist even
  if every candidate is subsequently rejected. Evaluate analyst-selected and all-screened
  cohorts prospectively with the same settlement and flat-unit accounting.

## Data contracts and site behavior

The existing `latest.json` row contract, seven markets, filters and five NHL routes
are unchanged. Additive route `/nhl/picks.html` reads `data/candidates.json` schema 1:
`board_id`, `generated_at`, Eastern `decision_date`, `session`, `source_snapshot_id`,
`policy_version`, `policy`, `status`, `review_status`, `eligible_count`, exclusion counts,
`candidates`, and an always-empty validated `recommendations` list. Each candidate
retains the complete quote/forecast row, stable `candidate_id`, rank, research status,
optional `qualitative_review`, and distinct human decision. Market probabilities remain
conditional on no push; model win probability and EV retain their original semantics.

The new page checks snapshot identity against `latest.json`. Failed models/feeds,
another date, changed snapshots and started games cannot show current candidates.
At 30 minutes a candidate becomes a dated research note with selection disabled;
its price is never represented as executable. Client-side expiry runs every 30 seconds
and is rechecked on submission. Every new forecast needs a new review; research is
not silently transplanted between prices, lines or model runs. An afternoon refresh
archives earlier research and produces a new quantitative board without a paid call.

Market Watch now uses `conditional_price_advantage` for membership and sorting,
with three other paired books. This removes its old use of model rank and the implicit
model-push requirement for integer lines. Existing output fields, including push-aware
`consensus_ev`, are preserved. The displayed price difference is not labeled an
unconditional expected return. Supplemental model details remain available there.

The research paper is revision 1.1 with an operational extension describing this
workflow. Its trained models, empirical data, evaluation windows and results are unchanged.

## Source and API configuration

`config/nhl_analyst.json` contains the fixed policy, `gpt-6-astra`, enabled flag and
rolling seven-day cap. Existing `OPENAI_API_KEY` is read only from the environment
or an explicitly specified local env file. GitHub uses the existing secret; no key
appears in source, generated HTML, public JSON or request archives. NHL odds retain
the existing authorized feed. No new subscriptions or paid search tools are used.

The source collector checks ESPN and CBS NHL RSS plus NHL.com's news index. It
requests at most eight article pages, at most two sources per candidate. HTTPS
publisher allowlists are reapplied to redirects. Bodies are bounded to 2.5 MB;
only the excerpt used by the model, its full-page hash, publisher time and actual
retrieval time are archived. Undated or over-72-hour stories are excluded. Headline
and excerpt matching uses full player names and team names/nicknames; it does not
establish roster membership. News availability and semantic relevance still need
human judgment. Retrieval follows the quantitative cutoff and is labeled later context.

The live connectivity probe on 2026-09-26 retrieved two usable dated articles for
two test matchups after eight attempts, both from ESPN. It did not establish complete
injury, starter, deployment or tactical coverage, nor independence of multiple publishers.
See `source-probe.json`. An empty packet skips Astra and asks for manual research.

Astra uses Responses structured outputs, low reasoning effort, at most 4,200 output
tokens and 26,000 serialized request bytes including schema/instructions. Exact source
IDs, candidate IDs, excerpts, timestamp validity, quotation limits and allowed fields
are checked locally. Any invalid response fails closed for the entire batch. The
same source contributes at most 25 quoted words across the batch. These checks do
not prove that the model's interpretation follows from a quotation; the UI explicitly
marks it unverified. Prompted narrative confidence percentages and probability/EV
adjustment fields are prohibited. It has no tools, book login or wagering capability.

Budget reservations use conservative $12.50/million input and $50/million output
rates, include framing overhead and 10% headroom, and count uncertain attempts in full.
The maximum configured request reserves $0.61666; a shorter packet reserves less.
One morning attempt per Eastern date is allowed. No automatic retries, alternative
model or paid fallback. Successful usage settles at conservative token cost. Prices
must be reviewed when the provider changes rates. In CI the budget and exact packet
are committed and pushed before the API request; checkpoint failure prevents the call.
Local invocation holds a file lock. Run local production calls only from the current
ledger and publish its records before another runner is allowed to proceed.

Official API references checked 2026-09-26:
[Astra model and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Reproduce and operate

Python 3.11 and existing pinned `requirements-nhl.txt`; no additional runtime packages.
Browser checks use Node 22 and Playwright 1.55.1. Typical work is a few seconds for
screening, up to three index/eight article requests, and one bounded API response.
Request timeouts are 5/8 seconds per source connection/read and 15/240 seconds for
Astra. Training stays separate. The GitHub job allows 20 minutes including existing
data refresh. Deployment retains the current GitHub Pages infrastructure.

```bash
pip install -r requirements-nhl.txt
python -m unittest discover -s tests -p 'test_nhl*.py'
python scripts/nhl/refresh.py --offline
python scripts/nhl/analyst.py                         # prepare only; never paid
python scripts/site_notices.py --scope nhl
NODE_PATH=/path/to/node_modules node tests/nhl_browser.cjs

# Production data then optional one-per-morning sourced research:
python scripts/nhl/refresh.py --env-file /secure/path/nhl.env
python scripts/nhl/analyst.py --astra --env-file /secure/path/openai.env

# Download a review note on Top Picks, then import BEFORE kickoff to count prospectively:
python scripts/nhl/v2/decisions.py record /path/to/nhl-review-CANDIDATE.json
python scripts/nhl/v2/decisions.py grade --cached-history \
  --output artifacts/nhl/analyst/evaluation.json

# API wiring test is explicitly opt-in and uses a synthetic fixture, never a real pick:
python tests/nhl_astra_smoke.py                      # request bound only, no API
python tests/nhl_astra_smoke.py --live --env-file /secure/path/openai.env
```

An operator commits the imported decision ledger and updated public board to make
the review durable; the download alone remains on the analyst's computer. Duplicate
conflicting decisions are refused. Analyst-reported time and ingestion time stay
separate. Records ingested after kickoff are excluded from prospective comparisons.
No override is allowed in this preparation form. The separate existing explicit
override interface remains available with its stronger documentation requirements.

`artifacts/nhl/analyst/boards` preserves the pre-review shortlist; `requests` and
`responses` preserve exact evidence packets and API results; `published` preserves
versions shown to users; `decisions` preserves human records; `budget.json` is the
durable spend ledger. The daily workflow archives these and runs prospective grading.
The shadow report includes counts, flat-unit returns, drawdown, clustered uncertainty,
worse execution and non-push Brier score for screened, selected and passed cohorts.
It uses only the first morning observation/game/day. Later runs cannot be cherry-picked.
Decisions on later or afternoon boards remain archived but are explicitly excluded
from this first-morning evaluation cohort; `excluded_decisions` lists their IDs.
Missing player participation remains unresolved. Closing-price data is not collected
by this extension, so CLV is explicitly unavailable. Small selected cohorts cannot
establish calibration improvement or analyst skill.

## Validation and release

- The original 34 NHL tests passed before integration; 14 additional tests cover
  selection, correlated exposure, evidence timing/identity, unsupported output, budget
  limits, timeout reservations, duplicate requests, human imports and cohort grading.
- Browser checks cover all six NHL routes at five widths, populated cards, filtering,
  price-only Market Watch ordering at integer lines, original probabilities, sources,
  review download, missing research, expired quotes, snapshot changes and failed feeds.
- One real Astra integration test using a clearly labeled synthetic fixture passed.
  It returned `needs_information`, cited the supplied text, and applied no numerical
  adjustment. Usage was 937 input/222 output tokens; conservative cost $0.022812
  (reserved $0.32538). `astra-smoke.json` retains the result. This proves API plumbing,
  not hockey-analysis quality, profitable selections or prospective improvement.
- The initial prospective report contains zero settled candidates and null returns.
  No new performance claim is supported. All markets remain experimental.

After review and approval, merge this branch, run NHL Daily Update, verify the
new Top Picks route plus existing Market Watch and all model cards, and inspect
`review_status`, budget entries, source coverage and candidate counts. Zero candidates
is not an outage; `feed_unavailable`, `review_unavailable` or stale snapshot identity
requires investigation. Browser quotes expire independently of scheduled refreshes.

For a quick rollback set `astra_enabled` false and run prepare again; quantitative
screening continues without spending. To remove the extension, revert this PR and
regenerate the original site. Preserve the analyst archives and spend ledger when
rolling back; deleting them could allow duplicate requests and erase prospective
decisions. No model retraining or historical artifact rollback is required.

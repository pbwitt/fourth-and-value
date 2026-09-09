# Fourth & Value: NFL site review

Reviewed September 8–9, 2026. Scope: homepage → NFL discovery → props → picks → methods → pricing checks → totals → research/blog → tracking, plus the Python build and scheduled refresh. Changes are local; the live site has not been deployed from this review.

## Assessment

The site's useful core is free line comparison with transparent research. The largest problem was credibility of the displayed estimates, followed by mobile usability and discovery. Traffic acquisition should follow clearer evidence, dependable timestamps and a consistent NFL experience.

The [live props](https://fourthandvalue.com/props/index.html) and [picks pages](https://fourthandvalue.com/props/top.html) inspected at the start of the review were Week 1, 2026. The alternate `/nfl/props/` URL still contained Week 11, 2025 and a broken navigation path. NFL totals were Week 12, 2025. These are different freshness situations and should not be grouped under one “current” label.

## Findings and changes

| Priority | Finding | Change / disposition |
|---|---|---|
| Critical | Week 1 touchdown estimates fell back to generic priors yet were presented as player-specific probabilities. The live shortlist included estimated profit above $900 per $100. | Missing relevant player evidence now yields an unavailable model estimate. Unsupported estimates cannot qualify for Top Picks. Existing AI commentary is visibly archived. |
| Critical | De-vig calculation grouped a player's alternate lines together; the props UI also applied a fixed 0.9524 factor regardless of the actual opposite price. | Match event, player, market, book and exact line; normalize the two opposing implied probabilities. Missing or conflicting opposite prices remain unknown. Duplicate quotes do not get extra votes. |
| Critical | Poisson integer-line equality was included in Under as a win; a second probability implementation also mishandled Over at integer lines. Zero Poisson rates could hit `log(0)`. | Shared probability implementation uses explicit wins/pushes. Model probabilities are conditional on non-push settlement, consistent with the existing calibration labels; expected value accounts for returned stakes. Aliases for interceptions are supported. |
| Critical | Current-season logs and position pools could include games after the forecast cutoff during historical evaluation. | Enforce season/week cutoffs in recent-log loading and parameter construction, including direct backtest callers. |
| Critical | Totals rolling features sorted by week rather than chronological date across seasons, and cross-validation split team-ordered rows. | Sort features by game date and align assignments by original index. Validation splits by date, keeping both sides of games together and all training dates before test dates. Existing headline accuracy claims are withdrawn pending revalidation. |
| High | Totals edge finder used `price` (American odds) as the points line and could retain old output when no new edges qualified. | Use `point`, count distinct books, use a median, require three books for screening, choose the matching offered side and overwrite empty output. |
| High | Model home spread had the opposite sign from sportsbook home handicaps. | Use away predicted score minus home predicted score. The published 2025 totals remain an archive, not refreshed forecasts. |
| High | Upcoming kickoff was treated as proof that a sportsbook quote was fresh; `max_age_hours` was unused. | Capture provider `last_update`, enforce maximum age on upcoming offers, reject missing/invalid/future times, and distinguish build time from quote time. |
| High | Make could reuse `latest_all_props.csv` because it was a cached file target. | Scheduled full NFL builds use `--always-make`; pricing helper and calibration changes are explicit build dependencies. |
| High | Props filters showed all books when every book was unchecked; the UI created thousands of cards and table rows simultaneously. | Clear all means zero results. Render 24 cards per page, with labeled filters, reset, explicit empty states, and shared filter URLs. |
| High | Top Picks said “Consensus probability” when showing the individual book's implied price; model means were treated like probability-calibrated thresholds. | Shared props/picks rendering separates book probability, paired fair probability, exact-line consensus and model mean. Picks require documented evidence and recent quotes. |
| High | “Best” selection could combine unrelated players sharing an initial and surname or hide the best available offer at a user's selected book. | Use complete player names within an event and apply sportsbook filters before selecting the best price for an identical line/side. |
| High | Related-line ratios were called mathematically impossible and described as structural arbitrage. | Rewrite the explanation as model-dependent screening. Provide a counterexample where the suggested two legs both lose. A ratio of marginal betting lines is not a joint-outcome constraint. |
| High | Tracker grading omitted current canonical market names, could match another game date, and could settle missing actuals as pushes. | Add market aliases, require matching game dates and finite actuals, reject ambiguous player matches, and enforce date selection. Game dates on new NFL props tracking and newly fetched scores use Eastern time. No live user bets were read or changed. |
| Medium | Navigation overflowed at intermediate widths; custom button/menu roles lacked complete state handling. | Use native buttons, expand the compact-menu breakpoint, correct grid placement, add visible focus, synchronize expanded state and support Escape focus restoration. |
| Medium | Main entry page did not provide a clear NFL path or explain the tool's limitations. | Replace with an NFL-focused homepage, dedicated NFL hub, a three-step reading guide, and direct routes to comparisons, methods, research and tracking. |
| Medium | Missing descriptions/canonicals/sitemap and broken favicon references weakened discoverability and polish. | Static metadata, sitemap and robots file; reuse existing SVG favicon; remove duplicate NFL props content through a canonical redirect. Templates and stale analysis archives are excluded from indexing. |
| Medium | The documented four-game recency weights listed only three observations and the wrong newest-game weight. | Correct the explanation to 3.9%, 9.9%, 24.6%, 61.6% at decay 0.4; clarify that lower decay emphasizes recency. |
| Medium | AI summaries inferred Over/Under from edge sign and could fall back to unsupported rows. | Use the supplied side, apply shortlist evidence/timestamp gates, and emit a deterministic empty snapshot when no rows qualify. |

## Current data and model limits

The saved raw Week 1 input produced 13,091 merged offers. Of these, 4,646 had fitted-calibration estimates and 8,445 had no supported model estimate; the latter includes markets the model does not cover. The rendered board removes duplicate offers and “No Scorer,” leaving 13,043 offers.

**None of the saved raw quotes contains a provider timestamp.** Rebuilding cannot recover that provenance. The revised board therefore says quote freshness is unverified, and the generated Top Picks page has no qualifying picks. The next successful odds pull through the revised fetcher will retain provider timestamps. This review did not consume paid odds or AI API quota.

The checked-in calibration artifact covers 12,904 graded records from nine 2025 weeks. It has not been refitted during this review. There are remaining reasons not to claim an independently demonstrated betting edge:

- Calibration fitting/evaluation needs strict separation by time. Selecting the latest snapshot of a week is not proof it was captured before kickoff.
- The historical loader hardcodes exclusion of 2026 snapshot filenames; season selection and immutable snapshot provenance need redesign before the next calibration refit.
- Alternate lines and opposing sides are correlated. Report unique player-game outcomes and cluster uncertainty by game/week rather than treating every row as independent.
- Historical roster/team inference can be wrong after trades. Add an as-of roster source; do not infer a current team from a player's most recent old season alone.
- Current touchdown estimation still lacks a validated career/role-aware fallback. Showing no estimate is intentional until that model exists.
- Discrete count distributions and yardage tails need held-out comparisons; continuity correction fixes settlement math, not model fit.
- Injury status, role changes, weather and lineup confirmation are not complete input features. Do not imply they are automatically accounted for.
- Totals scripts still have 2024/2025 datasets and experimental Week 11 paths. The published totals page must remain archival until those inputs, forecasts and evaluation are modernized together.
- Grading requires dated, completed-game player logs. The weekly stats pull and grader still use different storage conventions. Unsupported/missing results now remain pending; a verified schedule-to-weekly-stat adapter and an explicit void/inactive-player policy are the next tracking improvements.

## Recommended validation before promoting model performance

1. Freeze pre-kickoff snapshots with season, week, event ID, provider quote time, model version and training cutoff.
2. Build chronological train/calibration/test periods. Never refit on the held-out evaluation period.
3. Compare each market with a same-event, same-line, paired-price market baseline using Brier score, log loss and calibration curves. Measure identical rows for both forecasts.
4. Report unique outcomes, number of games/weeks, quote availability and coverage alongside error metrics.
5. Evaluate the actual shortlist policy prospectively: odds available at publication, pushes/voids, realistic stake rules, closing-line movement and uncertainty intervals. Profit is not established by a calibration curve or model agreement.

These changes to evaluation practice follow the distinction between fitting and independent evaluation described in [scikit-learn's calibration documentation](https://scikit-learn.org/stable/modules/calibration.html). Discrete settlement calculations were checked against the distribution functions documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html).

## Traffic and general-use plan

### First two weeks: make discovery measurable

- Deploy the reviewed build after checking the visible snapshot. Verify the custom domain in Google Search Console and submit `/sitemap.xml`.
- Inspect the homepage, NFL hub and props page with Search Console's URL Inspection. Confirm selected canonical, mobile rendering and indexing eligibility.
- Establish a weekly baseline: search impressions/clicks, queries, indexed pages, landing pages and click-through rate. Separate branded searches from useful NFL informational searches.
- If adopting analytics, measure NFL hub → props, filter usage, empty results, same-line comparison, methods clicks and return visits. Choose the service and update the privacy disclosure before adding tracking. No analytics or marketing service was connected in this review.

### Next four to six weeks: publish useful NFL material

Publish one carefully checked article per week, each with a named author/reviewer, sources, publication/review dates and a direct route into the relevant tool:

1. How to compare NFL player props at different sportsbooks (same-line examples).
2. What −110 means: bookmaker margin, break-even probability and expected value.
3. Why a 55% forecast is not a “safe” bet: calibration and sample size.
4. NFL prop pushes, voids and what happens when a player does not participate (verify current book-specific rules).
5. Why early-season touchdown estimates need role and usage evidence.
6. A transparent weekly model review showing misses, missing coverage and changed assumptions.

These themes reuse the site's actual functionality. Avoid hundreds of thin player/team pages with templated text, automatic “best bets” claims, or daily AI recaps with no new verified evidence. Google emphasizes useful, original, current content and understandable crawlable structure in its [SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide).

### Retention after freshness is dependable

- Keep the shared filter links; consider remembered sportsbook preferences and a saved-player watchlist next.
- Refresh closer to NFL decision times (Thursday evening and Sunday morning) only after measuring data availability and API quota. A Monday-only snapshot is often too old for a Sunday recommendation.
- Publish a clear scheduled-refresh status with last successful run and oldest quote. Do not show “live” unless the data actually refreshes continuously.
- Build a public, versioned results page once grading and evaluation are reliable. Distinguish the public model record from each user's private bet tracker.
- Consider an opt-in weekly digest only when publication cadence and consent/privacy handling are defined. No unsolicited signup forms or mailing integration were added.

Search changes create eligibility and a better user experience; they do not guarantee rankings or traffic. Judge progress over several weeks, and improve the pages visitors actually use.

## Reproduction and verification

- Python regressions: `.venv/bin/python -m unittest discover -s tests -v`.
- Site metadata: `.venv/bin/python scripts/build_site_metadata.py`.
- Preview: `python3 -m http.server 8010 --bind 127.0.0.1 --directory docs`.
- Browser checks: `PLAYWRIGHT_MODULE=/path/to/playwright CHROME_PATH=/path/to/chrome node tests/browser_review.cjs` (or install Playwright and use its default browser).
- Rebuild local props from a merged CSV using `scripts/build_props_site.py` and `scripts/build_top_picks.py` with explicit season/week. Full scheduled builds refresh source data and run QC.

Verification results are recorded below. The pre-existing deletion of `docs/data/ai/insights_week1.json` was preserved.

### Verification results

- Nine Python regression tests passed (prices, exact-line/event identity, duplicate quotes, Poisson zero rates, integer pushes, expected value, freshness, chronological rolling features, empty totals outputs and grading dates/missing actuals).
- Full weekly QC passed with one expected warning: no 2026 weekly game statistics exist before Week 1; prior-season data supplies the baseline. The corrected QC checks 2,243 paired Over/Under estimates and their probabilities sum to one.
- The shipped props JavaScript passed snapshot interaction checks for pagination, clear-all-books, search, URL state, selected-book comparison, reset and accurate empty-shortlist freshness. This is a DOM harness, not visual browser validation.
- Python compilation and JavaScript syntax checks passed. All local HTML links/assets resolve; sitemap contains 22 canonical, indexable pages.
- The full board contains 13,043 deduplicated offers and renders 24 at a time. Its dictionary-encoded HTML is approximately 1.5 MB (205 KB when locally gzip-compressed); hosting compression was not measured. The empty shortlist is approximately 6 KB.
- Visual rendering in a real browser is **not verified**: both installed Chrome builds timed out at launch; the installed Playwright package cannot download supported Chromium/Firefox builds for macOS 12. Responsive CSS and interaction logic were reviewed; a supported-machine browser pass remains necessary before deployment.
- Email sign-in, live database policies and automated settlement against real customer records were not exercised. No live bets, subscriptions, paid data refreshes or public deployments were made.

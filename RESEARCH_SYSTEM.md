# Research, reliability and model-validation system

Central developer reference for discovery → screening → research → publication →
decision ledger → settlement. Operating rules are in
[ANALYST_RESEARCH.md](ANALYST_RESEARCH.md); the reader-facing flow is
`/research/daily-process.html` (source `scripts/research/daily_process.html`).
Earlier decisions: [reports/top-picks/IMPLEMENTATION.md](reports/top-picks/IMPLEMENTATION.md).
Last updated October 5, 2026 (audit of `66e7b0a`).

Three different things are kept apart throughout:

1. **Software correctness**: tests pass, records are immutable, failures are visible.
2. **Predictive validation**: probability accuracy against outcomes and baselines.
3. **Evidence of a betting edge**: performance at recorded prices against the market.

No part of this change establishes (3) for any sport. NFL fails (2) at book lines; by
owner decision it stays in Top Picks with an explicit not-validated label.

## 1. Findings and what changed

The audited commit was HEAD, so every finding was current.

| # | Finding | Verified cause | Change | State |
|---|---|---|---|---|
| 1 | 8:41 a.m. Oct 5 card: five “Consider” ideas, empty evidence, `needs_information` | Confirmed (`docs/briefing/cards/2026-10-05-466e6f14…`). “Consider” meant only “no blocker named”; the label implied corroboration. | research-state-1 separates the five concepts; labels “Consider · model case only”, “Consider · verified context”, “Waiting on a material fact”, “Pass · verified adverse fact”, “Research failed”, “Research stale”. Deterministic gates for verified adverse and unresolved consequential facts. | Fixed |
| 2a | 20 of 38 reviewed | `max_review_batches = 8` × 3 per run (24 slots) is the binding limit; two of the eight 8:37 batches failed. Budget was not the constraint. | Coverage summary per sport (reviewed, reused, failed with categories, not attempted, retried). Limit unchanged. | Fixed (diagnostics) |
| 2b | Failed MLB batches | 8:09 ET: validator false positive on “not a guaranteed workload” rejected all three paid reviews. 8:39 ET: one `research_support` without a supporting item rejected the whole batch. | Negation fix; per-candidate validation; one bounded retry of rejected candidates. Replayed on the recorded responses (`tests/test_research_replay.py`). | Fixed |
| 2c | Failed NFL batch | 8:39:53 ET: reservation committed, no response, never settled. Settlement sits in a `finally` around the API call, so the exception came from the durable checkpoint (git rebase/push racing a concurrent editorial push at 8:39:52); no request was sent. The $0.121 stayed `reserved`. | Checkpoint retries rebase/push 3 times; on failure the entry is `released_not_sent` at zero actual cost with the reserved amount kept. Discovery gets the same handling. | Fixed |
| 2d | NHL refresh failure | A date-dependent NHL player-page test (fixed in `d3e5564`) failed “Verify NHL math”, so NHL never refreshed. Each edition was therefore `research_incomplete`, and the 7:35, 8:05 and 8:35 recovery starts re-pulled NFL/MLB and re-ran paid discovery and review for sports that had completed at 7:08 (11/11 NFL, 8/8 MLB). The 7:11 card had 10 rows; the 8:41 card had 5. | Recovery scope: a healthy sport with a feed ≤60 minutes old is reused (no refresh, so unchanged offers keep their reviews); discovery reruns only for new games/questions. Card shows run URL and categories. | Fixed |
| 2e | ~$1.89 remaining | $0.8627 charged or reserved across four runs, including the $0.121 never sent. | `ledger_summary` splits settled, uncertain, outstanding and released amounts. | Fixed |
| 3 | NFL calibration not refitted/validated after cutoff fixes | Confirmed; artifact had no provenance. | Point-in-time reconstruction, cutoff verification, separated windows, refit, exact-line market comparison (`reports/nfl-validation/2026-10-05/`). Result: model worse than market at book lines; no artifact installed. Owner decision: NFL stays in Top Picks, labelled “not validated for the current model” (`config/nfl_calibration.json`). | Fixed (labelling) / blocked on evidence (skill) |
| 4 | MLB postseason pitcher-outs passes despite calibration error | The postseason report is a separate 2025 fit (training through 2025-09-07, calibration through 2025-09-28), not the live artifact. It overpredicts outs by 1.87 (16.17 vs 14.30); the 0.85 bin observes 0.55 (n = 49), but aggregate ECE 0.118 ≤ 0.12 passes. The regular-season live artifact also overpredicts outs by 6%. | Side/range/bias gates (`scripts/mlb/gates.py`), clustered intervals and provenance in training reports. Today it would withhold pitcher-outs Overs and 3 of 8 postseason picks. | Fixed (intervals arrive with the next training run) |
| 5 | NHL lines, power play, goalies not numerical | Confirmed: team-level save rate, “Line and power-play assignment not verified”. | Facts captured; shadow ice-time pilot; goalie facts captured, effect not validated. | Mitigated (shadow) |
| 6 | Extend existing tracking | NHL decisions/grading and line movement existed. | Cross-sport decision ledger + grading reuse their definitions. | Done |

## 2. Architecture and data flow

```
sport refresh (nfl-weekly, mlb-daily, nhl-daily)
  └─ feeds: docs/props/top-picks.json, docs/mlb/data/latest.json, docs/nhl/data/{latest,candidates}.json
gate (morning_operations.gate → recovery_scope)        decides per-sport refresh/reuse
discovery (research_discovery.run)                      model-blind; leads only
selection (docs/assets/briefing-picks.js via analyst_shortlist.cjs)
  └─ collect(): numerical screens, freshness, best price, research_state
review (analyst_review.prepare → review_batches → run_queue → review)
  ├─ evidence.collect / targeted (free retrieval, categorized failures)
  ├─ astra.payload → bounds → research_budget.reserve → astra.checkpoint → call_api
  ├─ astra.parse_response(partial=True) → research_facts.build_facts
  └─ docs/briefing/reviews.json (+ immutable archives in artifacts/analyst/)
publication (morning_card.publish_card)
  ├─ card: docs/briefing/morning-card.json + immutable docs/briefing/cards/DATE-ID.json
  ├─ NHL pilot: artifacts/research/pilot/nhl-shots/DATE-ID.json (shadow)
  └─ ledger: artifacts/research/decisions/DATE-ID.json (decision-ledger-1)
late check (late_research.py, disabled)                 docs/briefing/reassessments*.json
settlement (research_outcomes.py → research_grading.py) artifacts/research/grades/*.json
```

## 3. File map

| File | Responsibility |
|---|---|
| `docs/assets/briefing-picks.js` | Source-of-truth selector (browser + Node): screens, `researchState()`, card gate, labels, `lateFor()` |
| `scripts/analyst_shortlist.cjs` | Node adapter; adds `research_state`, `_card_value`, review keys |
| `scripts/analyst_review.py` | Review runner, prompt `sports-research-7`, batch records, retries, coverage summary |
| `scripts/nhl/v2/astra.py` | Request bounds, `ReviewError`/`CheckpointError` categories, checkpoint retry, partial validation |
| `scripts/nhl/v2/evidence.py` | Allow-listed retrieval, freshness, `failure_category()` |
| `scripts/research_facts.py` | research-fact-1 records, verification overrides, overlap guard |
| `scripts/research_budget.py` | Daily ledger; `release_unsent()`, `ledger_summary()` |
| `scripts/research_discovery.py` | Model-blind discovery; same-day idempotency |
| `scripts/morning_operations.py` | Gate and `recovery_scope()` |
| `scripts/morning_card.py` | Edition, research health diagnostics, `freeze_decisions()` |
| `scripts/decision_ledger.py` | decision-ledger-1 freeze/verify |
| `scripts/nhl/v2/deployment_pilot.py` | NHL shots ice-time scenarios (pure Python) |
| `scripts/late_research.py`, `.github/workflows/late-research.yml` | Late check (disabled) |
| `scripts/research_outcomes.py`, `scripts/research_grading.py` | Outcome adapters and grading |
| `scripts/nfl_validation.py` | NFL reconstruction, calibration fit, evaluation, install guard |
| `scripts/make_props_edges.py` | `calibration_status()` compatibility label |
| `scripts/make_player_prop_params.py` | `MODEL_VERSION` (bump when forecasting logic changes) |
| `scripts/mlb/gates.py`, `scripts/mlb/train.py`, `scripts/mlb/predict.py`, `scripts/mlb/site.py` | MLB gates, intervals, provenance, Model Results |

## 4. Schemas and compatibility

All changes are additive; readers ignore unknown fields. Historical editions,
reviews and archives are never rewritten.

* **reviews.json** (schema 1, unchanged version): boards gain `batches[]`
  (`index, sport, status, category, candidate_ids, request_id, reservation_key,
  reserved_usd, charged_usd, ledger_status, request_sent, accepted, rejected[], http_status`),
  `coverage_summary`, and candidates gain `research_failure {category, stage, request_id, at}`.
  `qualitative_review.facts[]` holds research-fact-1 records.
* **Evidence items** (prompt `sports-research-7`): add `assumption`, `materiality`,
  `verification`, `effect`, `applies_to`. Older prompts lack them; their facts are
  `not_classified` and never gate deterministically.
* **research-fact-1**: `fact_id, candidate_id, offer_id, forecast_id, sport, event{game,
  game_id, commence_time, player, home_team, away_team, market, side, line},
  source{source_id, url, title, publisher, source_kind, published_at, updated_at,
  retrieved_at, publication_basis, content_sha256}, excerpt, interpretation, kind,
  direction, represented_in, assumption, materiality, verification,
  classification_basis, effect, applies_to, usage, conflicts_with[], freshness{…},
  probability_adjustment: null, recorded_at`.
* **research-state-1** (computed, also frozen into card rows/ledgers):
  `model, evidence, direction, material, completion, gate, label, facts, failure`.
* **morning-card.json** (schema 1): `research.sports[s]` gains `batches`,
  `coverage_summary`, `source_diagnostics`; `research.run_url`; top-level
  `decision_ledger {status, ledger_id, universe, counts, path, pilot_records, pilot_path}`.
  Rows gain `research_state`.
* **decision-ledger-1**: `ledger_id, edition_id, kind, decision_date, frozen_at,
  universe_count, rules, code{selector_sha, ledger_sha, prompt_version}, counts,
  entries[], entries_sha256`. Entry: identity, exact offer, `probabilities{model_win_conditional,
  push, market_conditional, market_books, market_basis, raw_probability, break_even}`,
  `expected_value`, `research{state, verdict, reviewed_at, request_id, prompt_version,
  fact_ids, failure}`, `decisions{baseline, research_filtered, adjusted}`.
* **decision-grades-1**: per ledger, never inside it; verifies `entries_sha256`.
* **late-reassessment-1** and `docs/briefing/reassessments.json` (schema 1).
* **nhl-deployment-pilot-1** records; **nfl-validation-1** report.
* **NFL calibration artifact**: unchanged keys plus optional `_provenance`
  (model_version, code_sha256, run_id, windows, evaluation) and `_market_counts`.

## 5. Status meanings

**research-state-1 gates** (only the first two can enter the card):

| gate | meaning |
|---|---|
| `model_case_only` | Consider verdict, no outside evidence verified. Not corroboration. |
| `verified_context` | Consider verdict with verified evidence that does not defeat the case. |
| `material_question` | Wait verdict, or a consequential fact that is unresolved, conflicting, or contradicted within the review. |
| `adverse_fact` | Verified (official/original/secondary report, not opinion), consequential concern about this game. Pass regardless of verdict. |
| `pass` | Reviewer judged the case not defensible. |
| `stale` | Older than 3 hours or no longer the exact offer. |
| `assessment_pending` | Legacy reporting-only review. |
| `failed` / `not_reviewed` | Attempt failed (category shown) / not attempted. |

**Batch statuses**: `completed`, `partially_completed` (some candidates rejected),
`review_unavailable` (batch failed), `budget_exhausted`, `budget_halted`,
`already_attempted*`, `expired_during_research`, `api_key_unavailable`,
`outside_review_window`, `disabled`, `no_candidates`.

**Failure categories**: batch-level `response_incomplete`, `response_refusal`,
`response_invalid_json`, `response_schema_invalid`, `candidate_identity_mismatch`,
`api_http_error` (+`http_status`), `api_timeout`, `api_connection_error`,
`checkpoint_failed`, `request_bounds_exceeded`, `evidence_collection_failed:*`,
`internal_error:*`; candidate-level `citation_unsupported`, `excerpt_not_found`,
`quote_limit_exceeded`, `unsupported_positive_status`, `unsupported_adverse_status`,
`consider_with_blockers`, `wait_without_blocker`, `numeric_confidence`.
Retried once: candidate-level categories and `checkpoint_failed`. Never retried:
transport/service errors (possibly billed).

**Ledger entry statuses**: `reserved` (outstanding), `settled`,
`uncertain_reservation_retained` (full reservation counts), `released_not_sent`
(provably unsent; zero actual), `reservation_overrun_stop` (halts spending).

## 6. Model and calibration provenance; validation gates

* **NFL** (`reports/nfl-validation/2026-10-05/`): model `nfl-props-2026-10-04`,
  code hash recorded; inputs nflverse weekly stats, snap counts and schedules
  (hashes in the report). Cutoff verified by supplying future rows. Windows:
  hyperparameters from 2025 development; calibration fit 2024 → evaluated 2025
  (fixed grids); deployment candidate fit 2024+2025 → scored at exact 2026 week 2–4
  book lines (provider-timestamped, seen weeks) → prospective from week 5.
  Grid 2025: raw 0.1864, refit 0.1861, legacy 0.2283, empirical reference 0.2099.
  Book lines (46 games, 1,547 forecasts): market 0.2474, legacy 0.2513, refit 0.2724,
  raw 0.2775; constant 50% 0.2500. **No artifact installed.** `calibration_status()`
  labels the legacy file as not validated for the current model. Under the owner
  policy (`config/nfl_calibration.json`) the label still starts with “Calibration
  fitted”, so NFL stays in Top Picks, and rows show “Calibration not validated for
  this model”. With the policy off, NFL props fail the selector's requirement.
  Requalification: a book-line calibration fit on earlier timestamped weeks must,
  on ≥60 later games, score below a constant 50% with the 95% interval of
  (model − market) Brier entirely below +0.002.
* **MLB**: regular-season report = the live bundle on a held-out 30-day window;
  postseason report = a separate earlier fit, never live. Gates per pick: aggregate
  checks; skill interval excludes zero (when present); count bias ≤5% or interval
  includes zero, else the favoured side is blocked; the pick's side/range bin has
  ≥30 outcomes and is not overstated by >3 points beyond its Wilson interval.
  Accuracy is against an empirical baseline only; no historical prices exist, so
  market performance is unavailable. Calibration gaps are never subtracted from EV.
* **NHL**: v2.3 unchanged; evaluation in `reports/nhl-v2.3/`.

Brier: 0 is perfect; 0.25 is the constant-50% score on binary outcomes, not a floor.

## 7. Research adjustment rules and double counting

* The language model classifies facts; it never supplies probabilities or effect sizes.
* A fact may affect numbers only through code (today only the NHL pilot), only if it
  is about this game, verified (not opinion/conflicting), not marked as represented in
  model features, and published after the model's feature cutoff.
* NHL pilot mapping: excerpt states minutes (one plausible number, 5–30) → point
  scenario `supported_adjustment`; deployment fact without a number → range from the
  model's projected ice time to the player's own p10/p90 of the last ten appearances
  (`scenario_only`, no decision change); unresolved/conflicting → full p10–p90
  (`unresolved_fact`). Goalie and participation facts → `unvalidated_context`.
  The baseline must reproduce the published forecast to 1e-6 or the record fails
  closed (`baseline_mismatch`). Verified on all 346 Oct 5 shots rows.

## 8. Scheduling, budget, retries and freshness

* Morning: 07:05 start, recoveries 07:35/08:05/08:35 (scheduled starts, not
  publication times). Recovery reuse: NFL/MLB feeds ≤60 minutes old with successful
  refresh and research; NHL always refreshes (30-minute quote window).
* Budget: $2.75/day shared by discovery, review and (when enabled) late checks;
  `later_reserve_usd` holds part of it for later checks (currently 0).
* Freshness unchanged: articles ≤72 h, live injury tables 90 min, reviews 3 h for
  the card, quotes 90/90/30 minutes (NFL/MLB/NHL).

## 9. Commands

```sh
# Tests (no network, no paid calls)
python -m unittest discover -s tests
node tests/briefing_picks.cjs && node tests/research_state.cjs
python scripts/build_daily_process.py --check
# NFL validation
python scripts/nfl_validation.py fetch
python scripts/nfl_validation.py run --run-id YYYY-MM-DD     # ~10 minutes
python scripts/nfl_validation.py books --run-id YYYY-MM-DD   # re-score book lines only
python scripts/nfl_validation.py install --run-id YYYY-MM-DD # only after the gate passes
# Grading
python scripts/research_outcomes.py --sport mlb --out data/research/outcomes-mlb.json
python scripts/research_grading.py --outcomes data/research/outcomes-mlb.json
# Late check (dry run never pays)
python scripts/late_research.py
# Inspect failures
python -c "import json;c=json.load(open('docs/briefing/morning-card.json'));print(json.dumps(c['research'],indent=1))"
```

## 10. Rollout and rollback

* Merge → next morning run uses prompt 7, new gates and ledgers. Verify: card
  `decision_ledger.status == 'frozen'`, labels, batch diagnostics, no duplicate spend
  on recovery starts.
* NFL: after the next NFL refresh, `docs/props/top-picks.json` rows read
  “Calibration fitted for a model version before the forecast-cutoff fixes; not
  validated for the current model”; NFL stays in Top Picks with that label. To make
  NFL research only, set `allow_incompatible_for_top_picks` to false.
* MLB gates take effect at the next MLB refresh (retrain adds intervals). Rollback:
  remove the `reasons()` call in `mlb/predict.pick_reason`.
* Late check: set `late_check.enabled: true` (and optionally `later_reserve_usd`),
  then dispatch `Late Research Check` once and inspect the reassessment and ledger.
  Rollback: set it false.
* Research gates: rollback by restoring the `shortlist()` condition; keep all archives.
* Never reset a ledger or delete archives, ledgers or grades.

## 11. Known limitations and blockers

* No untouched historical holdout exists for NFL; 2026 weeks 2–4 were seen.
  Prospective NFL book-line evaluation needs archived offers each week (already
  produced by the weekly archive) and the snap-count release.
* 2025 NFL lines in git are truncated, edge-sorted page subsets without provider
  timestamps; unusable for fitting (rejected, see log).
* MLB clustered intervals appear only after the next training run (the MLB API is
  not reachable from this development container; CI can reach it).
* MLB/NHL grading adapters depend on the refresh workflows' local history caches.
* The late check and the NHL pilot have not run against live paid reviews.
* `max_review_batches = 8` still bounds coverage (deliberately unchanged).

**Promotion criteria for experiments** (fixed now, before results):
* NHL pilot to live adjustments: ≥300 settled `supported_adjustment` records across
  ≥100 games; adjusted log loss better than baseline on the same entries with a
  game-clustered 95% interval excluding zero; no adjusted calibration bin with n ≥ 30
  off by more than 3 points beyond its interval; owner approval via PR.
* Research gates: compare research-filtered vs baseline on ≥500 settled ledger
  entries; report returns at recorded prices with clustered intervals; no promotion
  claim from fewer.
* Late check: enable after one supervised dispatch shows correct triggers, spending
  and immutable reassessments.

## 12. Decision log

| Date | Decision | Reason | Rejected alternatives |
|---|---|---|---|
| 2026-10-05 | Separate five research concepts; deterministic adverse/unresolved gates | “Consider” implied corroboration; adverse facts must be able to force a pass | Require a news story for every bet (suppresses valid model cases) |
| 2026-10-05 | Per-candidate validation + one retry | A single bad review discarded paid work | Retry whole batches (double spend); loosen validators (unsafe) |
| 2026-10-05 | Release unsent reservations at zero | Checkpoint failure precedes the request, so no charge can exist | Keep as `reserved` forever (misstates spend); retry API on timeouts (may double bill) |
| 2026-10-05 | Recovery reuses healthy sports | An unrelated NHL failure triggered three paid reruns that produced a worse card | Treat any incomplete edition as fully re-runnable (status quo) |
| 2026-10-05 | NFL calibration on fixed grids for fitting/evaluation; exact lines for market comparison | 2025 page lines are selected on the old model's edge, untimestamped | Fit on 2025 page subsets (selection bias) |
| 2026-10-05 | Install no NFL artifact; label NFL not validated | Every version loses to the market at book lines; legacy ≈ constant 50% | Install grid refit (worse at book lines); keep the old “fitted; not prospectively validated” label (implied it was fitted for this model) |
| 2026-10-05 | Owner: keep NFL in Top Picks (`config/nfl_calibration.json`) | Product continuity while the book-line calibration is rebuilt; labels disclose the evidence | Make NFL research-only (initial proposal, overruled by owner) |
| 2026-10-05 | MLB side/range/bias gates with a-priori thresholds | Aggregate ECE hid one-sided errors | Subtract ECE from EV (different quantities); tune thresholds on returns (no prices; overfitting) |
| 2026-10-05 | NHL pilot shadow-only, minutes only when stated in the source | No validated minutes-per-promotion effect | Assume fixed TOI bumps for line changes |
| 2026-10-05 | Late check shipped disabled | New paid path not exercised live from here | Enable by default |
| 2026-10-05 | Ledger baseline uses the card's own ranking units | Same universe and ordering isolates the research effect | Separate baseline ranker (confounds comparison) |

# NHL rebuild: implementation plan and decisions

Started 2026-09-26. Branch `codex/nhl-model-rebuild`. No merge or deployment authorized.

1. Freeze public routes, fields, market semantics, filters and failure behavior in contract tests.
2. Collect official regular-season team and skater game records with immutable raw responses,
   identifiers, retrieval times, checksums and explicit historical availability assumptions.
3. Build strictly lagged features; compare historical-rate, shrunk opponent strength,
   regularized Poisson and boosting candidates. Compare player count distributions and
   opportunity projections. Preserve full score and goal/assist dependence.
4. Lock selection on seasons through 2024–25; reserve 2025–26 for final evaluation.
   No betting ROI without contemporaneous, executable prices. No market blend without
   a past-only outcome/quote validation sample.
5. Add audited quote pairing, push-aware pricing, sourced analyst reviews and prospective grading.
6. Adapt to existing feed/pages, archive decisions, test failure states, provide operating docs,
   actual evaluation artifacts and a draft PR.

## Audit decisions

- Existing public routes are `/nhl/`, `/nhl/props/`, `/nhl/totals/`, `/nhl/top.html`,
  `/nhl/methods.html`; these replace the obsolete routes proposed in the 2025 plan.
- Public feed: `docs/nhl/data/latest.json`; source is `scripts/nhl/refresh.py`.
  Market keys: player_shots_on_goal, player_goals, player_assists, player_points,
  totals, h2h, spreads. Existing historical references remain a separate nullable fallback.
- Legacy `nhl_build_features.py` uses full-sample opponent/home-away aggregation;
  `nhl_train_model.py` selects same-game shots, points, PP goals and goalie save percentage.
  Its row split does not establish grouped chronological evaluation. Its results are rejected.
- `nhl/nhl_models.py` uses rolling rates, Normal/Poisson assumptions, fits isotonic
  regression to consensus (not outcomes), and returns 0.5 for an unknown player.
  None of these artifacts will be loaded by the replacement.
- Current refresh explicitly publishes historical references, not model predictions.
  The existing comparison function is shared with NBA; NHL corrections will be isolated.
- Existing summaries and old model/odds files are ignored in Git. Local records will be
  inventoried separately, not promoted to timestamped historical evidence by filename alone.
- Work is in an isolated local checkout because the user's workspace contains uncommitted
  editorial changes and a newer live NHL snapshot. Those files will not be changed.
- No applicable AGENTS.md found in the repository or parent directories.

## Evaluation protocol (locked before final-test inspection)

Use 2022–23 as initial training history; 2023–24 and 2024–25 as expanding, season-aware
validation folds. Candidate selection uses mean predictive log score across those folds.
2025–26 is the final untouched season. All sides and player records of a game share a fold.
Morning decision time is 10:30 America/New_York; optional update is 16:30. Previous-day
results are eligible only after the documented next-day availability cutoff. Final-test
outcomes can update rolling histories after that cutoff but cannot select algorithms,
thresholds, calibration or hyperparameters. Playoffs are excluded and unvalidated.

Public statistics do not provide original publication/revision timestamps. Historical
evaluation is therefore a **reconstructed predictive test**, with a conservative lag,
not a certified replay of vintage data. Prospective snapshots provide actual ingestion times.
No unrecorded injuries, starters, lines, trades or closing odds enter historical features.
All recommendations remain disabled until timestamped executable-price evidence exists.

## Implementation decisions and remaining limitations

- Official game summaries are complete for all four requested seasons (1,312 games each).
  NHL GF excludes shootout awards. Normalization reconstructs regulation and final settlement
  scores explicitly, including the OT empty-net standings exception. Monthly partitions
  avoid silent 10,000-row truncation.
- Validation selected `poisson_core`, `opportunity_nb` shots and `opportunity_nb` scoring.
  More team context and boosting were rejected for unstable validation gains; a hurdle
  player challenger did not improve scoring. No final-test-driven algorithm change.
- A post-evaluation correctness audit removed target-game position from rookie priors.
  The unchanged protocol was rerun and selection stayed the same. The test was opened
  before that correction; EVALUATION.md discloses this instead of claiming a pristine rerun.
- No reliable timestamped historical odds or qualitative archive was found. Live consensus,
  independent forecasts, future blend training utilities, sourced analyst reviews and
  prospective grading are implemented; market blending and validated picks remain off.
- MoneyPuck's free-use terms do not establish authorization for this commercial site.
  No xG data were downloaded. Starter identity, line units, injuries, travel distance and
  teammate effects remain unsupported model inputs, with explicit analyst uncertainty.
- Settlement mapping is limited to primary sources in `config/nhl_settlement.json`.
  Unmapped books retain their quotes but do not join other-book consensus or receive EV.
- Current main was fetched and the isolated branch rebased before site integration,
  preserving the newer responsible-use notices and other unrelated upstream changes.
- Source/statistics and forecast snapshots are durably archived in Git with content hashes;
  Actions retention alone was rejected because it expires after 90 days.

- Final entitlement check confirmed that the existing odds key includes historical access.
  A fixed monthly sample (the 15th, October–April across 2023–24 through 2025–26) was
  declared before download and acquired for 630 credits, leaving over 14,000. The
  locked model and shadow policy are unchanged. The earlier missing-local-odds finding
  is superseded by HISTORICAL_MARKETS.md; full daily and player-price coverage are still absent.

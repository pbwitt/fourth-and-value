# NHL v2.1 model card

Purpose: independent hockey forecasts and push-aware price research for analysts. All
markets remain experimental for betting. The engine does not select a mandatory number
of picks, generate a narrative edge, place wagers, or alter sportsbook accounts.

## Selected specifications

Team: regularized Poisson regression (`alpha=0.3`) on home indicator, lagged attacking
rate, and opponent conceding rate. Features shrink toward fixed three-goal league priors
with 12 equivalent games and 90-day half-life, using at most 164 past games. Standardization
and model coefficients are fit on training data only. Independent regulation goal counts
form one score matrix. Each diagonal mass receives one OT/shootout goal using a training-only
home-win tendency shrunk by 20 games. All full-game markets integrate this same matrix.
The prior means are documented modeling choices, not an inferred 50% outcome fallback.

This is an interpretable partially pooled rate construction, not a full Bayesian
hierarchical model. Empty-net scoring is present in target counts but has no explicit
state-dependent mechanism. Regulation scoring dependence, goalie changes and changing
league style can therefore be misrepresented. Score support truncation is checked.
Overtime and shootout winner identity are not predicted separately; both have the same
one-goal game-market settlement transform. Shootout statistics never enter player counts.

Player: separate recency-weighted projected TOI (30-day half-life, five prior appearances)
and per-minute production (120-day half-life, 12 prior appearances). Last observed position
selects a fixed position prior; unseen positions use the fixed unknown/forward prior.
Shots have a separately estimated negative-binomial dispersion. Goals/assists share a
latent gamma intensity; conditional binomial allocation of total points preserves their
joint distribution, with points exactly their sum. Scoring dispersion is estimated from
training residual moments. No player-market independence is asserted for joint bets.

The player model intentionally averages historical per-game per-minute rates by recency,
rather than pooling all minutes, so short appearances can be noisy. Shrinkage and explicit role uncertainty
mitigate but do not eliminate that limitation. No line-unit or teammate effect is claimed.

## Selection and calibration

Five team candidates: historical attack rate; multiplicative attack/opponent/home;
regularized core; regularized context; shallow Poisson boosting. Four player candidates:
rate Poisson; opportunity Poisson; opportunity negative binomial; opportunity hurdle.
Shots select on count log loss; scoring selects on mean goals/assists/points count log
loss to preserve a coherent scoring family. Selection uses the two development seasons.
There is no independently fitted probability calibrator. Training-only distribution
parameters and held-out calibration diagnostics are provided; no claim of perfect
calibration is made. Fixed diagnostic lines are not actual offer coverage.

The team context bundle (shots, save proxy, PP/PK and rest) and boosting were rejected
for inconsistent validation. Opportunity separation improved player validation;
overdispersion helped shots most. Detailed metrics and paired uncertainty are in
[EVALUATION.md](EVALUATION.md). No model is selected by ROI.

## Market and pricing components

Two-sided multiplicative de-vig, with additive-method sensitivity recorded. Both sides
must agree on game/player/market/line/settlement and be within five minutes. Other-book
consensus is a median of distinct paired books within fifteen minutes, excluding the
offered book. Unknown settlement profiles are isolated by book. Consensus requires
three other books before a Market Watch price comparison is eligible.

Market probabilities are conditional on non-push. Independent/final probabilities are
unconditional win probabilities given action. EV = p(win) × (decimal odds − 1) − p(loss).
Fair decimal odds = (1 − p(push)) / p(win). Integer-line comparison EV additionally
depends on the independent push estimate and is labeled accordingly. Market blend weight
is zero: `fit_blend` is a prospective training utility and does not qualify an artifact
for inference. No market consensus ever replaces a missing independent forecast.

Minimum price uses a 2% EV buffer and the worst required price across ±10% rate scenarios, including each scenario’s own push probability.
These are transparent operational sensitivity assumptions, **not estimated confidence
bounds or a historically optimized policy**. Ranking uses worst-scenario expected log
growth for a fixed 0.25% bankroll fraction; it is only research ordering. The selector
requires an explicit validated-executable status and analyst review, limits exposure to
one unit per game and four units/day, and currently returns an empty list.

## Failure modes and scope

- Regular season only. Playoffs, alternate settlement headers, specials and parlays are
  unsupported; do not extrapolate this model to them.
- Missing or ambiguous player identity, unknown teams, stale model inputs, future-trained
  artifacts, bad checksums and unsupported profiles fail closed. Unknown model values stay null.
- The same player name is never fuzzily assigned to another ID. Book names are mapped to
  stable IDs only when unique. Current participation is unconfirmed; missing appearance
  at grading remains unresolved unless a sourced DNP record supports a void.
- Retrospective corrected statistics may differ from originally published values. Historical
  ingestion in 2026 is never passed off as original pregame availability.
- The final test contains all skaters with positive TOI, not a historical prop-offer universe.
- Daily forecasts are withheld beyond 48 hours; offseason shrinkage and rookies can be
  particularly uncertain. New injury/goalie/lineup information invalidates analyst preparation.
- The optional live roster argument can enforce team membership, but no daily roster feed
  is wired yet; the UI explicitly says lineup and current deployment are unconfirmed.

## Versioned artifacts and analyst layer

`models/nhl/v2/manifest.json` records versions, selected specifications, training dates,
environment, model hash and history hash. The frozen artifact is small, with a compressed
four-season history for cold starts. Raw sources are in `artifacts/nhl/training-sources.tar.gz`.
Daily archives contain the offered prices, feature vectors, source pages and predictions.

Reviews record forecast/offer identity, analyst, kind, source URL, source and review times,
reason, overlap with existing features/market, and optional explicit manual probability.
The override does not overwrite the original forecast or model probability. New snapshots
require new review because forecast identity changes. Reviews are prospective shadow data;
no numerical goalie adjustment is inferred from prose. Local grading compares model,
market and reviewed outcomes without creating a historical betting ledger.

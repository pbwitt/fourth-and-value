# Week 1, 2026: analysis and article handoff

Scope: NFL games September 9–14, 2026. Prepared September 17. Publication authorized by the site owner after reviewing the findings. No social post is included in this release.

## Deliverables

- `docs/blog/week-1-2026-model-review.html`: 3,522-word article in the existing editorial style, responsive tables, section links, and expandable ledger for all 16 games.
- `docs/blog/week-1-2026-model-review.md`: editable article source.
- `docs/blog/week-1-2026-review-data.json`: summary, source hashes, selected tickets, paired forecast observations and game outcomes.
- `scripts/review_week1_2026.py`: reproducible audit using archived published pages, never refitting the model.
- `reports/week1-2026/`: summary and detailed local CSVs. CSVs are ignored by the repository's existing rules; the public JSON preserves the key row-level evidence.
- Blog index and sitemap include the new article.

## Findings

Archived Top Picks: 848 offers, 815 graded, 365–450, −58.74 units, −7.2% ROI. Many offers repeat the same outcomes across books and lines.

Illustrative one-ticket-per-player/game/market selection: choose highest published expected return, with deterministic book/line/side tie breaks, before inspecting the result. 242 selected, 232 graded, 97–135, −27.43 units, −11.8% ROI; ten pending. This was defined for the audit and is not asserted to have been a deployed betting strategy.

Rushing attempts: 15–26, −10.97 units, −26.8%. Receptions: 48–68, −10.78 units, −9.3%. Together they account for about 79% of the net loss. Passing yards were +3.70 units on just two correlated, +185 unders; one involved Darnold's early injury. All 436 published passing-yard model probabilities were exactly 50%.

Probability comparison on 530 identical player/game/market observations: model Brier 0.25089 versus exact-line fair market consensus 0.24791. Market has lower point estimates in every modeled category, but the game-block bootstrap difference interval crosses zero. Raw mean absolute errors also trail the representative line in every modeled market.

Totals MAE: raw model 12.47, published shrunk model 11.79, same-time market 11.78, closing line 11.69. Raw directional leans went 6–9–1. The only calibrated positive-EV total was the losing NE–SEA over.

Closing favorites: 12–4 outright, 9–6–1 ATS. Closing overs: 9–7, versus 8–7–1 at our Wednesday snapshot. Broad prop-market winners include passing-TD overs, interception overs, receptions overs, rushing-yard overs and passing-yard unders, under the article's representative-line/best-price methodology.

## Reproduction

From repository root:

```sh
.venv/bin/python scripts/review_week1_2026.py
python3 -m http.server 8010 --bind 127.0.0.1 --directory docs
```

Open `/blog/week-1-2026-model-review.html`. The analysis requires the existing saved schedule, player stats and Week 1 totals inputs listed and hashed in the output. Published props come from Git revision `6d420cf`. No API keys, quota, private bet records or current model artifacts are needed.

## Interpretation boundaries

- Returns are hypothetical stakes, not customer settlements. Ordinary action after participation is assumed; injury-protection rules are not verified per sportsbook.
- Missing outcomes and players without offensive usage evidence remain unresolved, not inferred zeros, losses, wins or voids. Reported ROI excludes those stakes.
- The quote snapshot predates all games, but Wednesday prices are not proof of Sunday availability. There is no closing prop archive in this analysis and no claim of measured prop closing-line value.
- Repeated books and alternate lines are excluded from the one-ticket comparison. Different markets on one player and different players in one game remain correlated. The 530 probability observations represent 192 player-game identities and 16 games.
- Anytime/first/last TD settlement is outside the audit. Passing touchdowns and interceptions have market results but no published model forecasts.
- Market returns mean returns to the specified betting side, not sportsbook revenue or hold realized from unknown handle.
- Main article recommendations are proposals for later validation, not model changes implemented in this task.

## Verification

- Completed the audit against both archived published pages and all 16 completed games. Assertions check pregame publication, quote timestamps, season/week, both teams, local game dates, unambiguous player matching and game-score totals.
- Independently recalculated offer and one-ticket P&L with Python's `decimal` arithmetic, reconciling exactly to reported ROI within floating-point tolerance.
- Checked uniqueness of selected player/game/market identities, market subtotal reconciliation, weighted Brier aggregation, representative known outcomes and unresolved Tua Tagovailoa rows.
- Validated HTML element nesting, five tables including the expandable game ledger, unique anchors, all local article links, structured article metadata, blog discovery and sitemap inclusion.
- Validated the public JSON against the internal summary and its 242 selected tickets, 530 paired probability observations and 16 game records.
- Python compilation and `git diff --check` passed. The site metadata build changed only the intended sitemap entry in addition to the blog index edit.
- Browser visual rendering was not exercised. Responsive table overflow, native disclosure controls and keyboard focus styles are included; no browser-rendering claim is made.

# Weekly recaps by sport: plan and handoff

Living record of the "weekly recaps for every sport" project, so any session
(Claude, Codex or a person) can pick up where the last one stopped. Update the
status list whenever a step lands.

## What the owner asked for (2026-10-06)

- Do **not** headline a win–loss record or units on the home page. Transparency is
  good, but a bad stretch should not be showcased as a scoreboard.
- Instead, publish **weekly recaps by sport**, in the style of the NFL weekly
  review (`scripts/nfl_weekly_review.py`, e.g. `docs/blog/week-3-recap-week-4-preview-2026.html`):
  detailed grading that shows where we were strong and weak that week, plus a
  preview. Readers who want the detail can open them.
- The home page links to each sport's latest recap (and Market Results) instead of
  the old "Keep the receipts" price-movement box, which was hard to follow and
  depended on refresh timing.

## What the NFL recap does (the template to copy)

`nfl_weekly_review.py` (runs Wednesdays 10:00 ET in `.github/workflows/nfl-weekly.yml`):

1. `snapshot`: freezes the week's pregame evidence (props board, top picks, totals
   predictions, lines, injuries) into `reports/nfl-weekly/<season>/week-<n>/pregame/`
   with a manifest and `snapshot_at`. Immutable once written.
2. `review`: grades the completed week against official stats:
   - published shortlist ("tickets"): record and units by market;
   - whole board: one representative line per prop (most-offered line, best price),
     overs/unders/pushes by market;
   - probability quality: model vs market Brier on the same outcomes, with a
     game-block bootstrap 95% interval; model mean vs line absolute error;
   - totals: model vs market mean absolute error, model direction record and units
     at the best executable same-line price (unpriced games excluded, disclosed);
   - games that kicked off before the freeze are excluded and disclosed;
   - preview of the next week: biggest model-vs-market total gaps and prop watch;
   - SVG charts, a blog article, `review-data.json`, and the blog index entry.

## Evidence available per sport

| Sport | Pregame evidence | Results source | Status |
|---|---|---|---|
| NFL | Weekly frozen archive (above) | nflverse via existing code | Recap automated (Wed 10:00 ET) |
| NHL | `artifacts/nhl/runs/*.json.gz`: every refresh, with official stats pages archived alongside (see `scripts/build_market_results.py` `nhl_pregame_quotes`, `nhl_results`) | archived NHL stats pages | Evidence exists; recap generator not built |
| MLB | **None until this project:** `docs/mlb/data/latest.json` is overwritten every refresh | statsapi.mlb.com box scores (reachable from Actions, not from the Claude container) | Archive added in step 2 below |
| NBA | `docs/nba/data/latest.json` only; season starts 2026-10-20 | ESPN / NBA feeds | Later |

Published card picks for every sport: `docs/briefing/cards/<date>-<edition>.json`
(read rows with `editionRows` in `docs/assets/briefing-picks.js`; several editions
can share a date: use the last `kind: "morning"` edition per `decision_date`, never
`test` editions, and record each pick's `policy_version`; AGENTS.md forbids
relabeling older results as evidence for a newer selection policy).

## Plan and status

- [x] 0. Handoff file (this document).
- [x] 1. Home page: replace "Keep the receipts" with links to each sport's latest
      recap and Market Results (`scripts/editorial_templates/home.html`,
      `latest_recaps()` in `scripts/editorial.py`). Recap file names it looks for:
      NFL `docs/blog/week-<n>-recap-week-<n+1>-preview-<season>.html`; other sports
      `docs/blog/<sport>-recap-<YYYY-MM-DD>.html` (week's last day). The Next up card
      was removed from the home page at the owner's request in the same PR.
- [x] 2. MLB pregame archive: every MLB refresh writes a compact gzip snapshot of
      the pregame rows to `artifacts/mlb/runs/<UTC stamp>.json.gz`
      (`scripts/mlb/archive.py`, called from `scripts/mlb/refresh.py`;
      `mlb-daily.yml` commits `artifacts/mlb/`). Grading should take, for each game,
      the last snapshot saved before first pitch.
- [ ] 3. MLB weekly recap generator (`scripts/mlb_weekly_review.py`), modeled on the
      NFL review: published picks graded (by market, by policy version); board-wide
      main-line over/under hit rates by market; model vs market Brier on the same
      outcomes with a game-block bootstrap; pitcher outs/strikeouts mean error vs
      line; team totals model vs market error; next-week preview. Results from
      statsapi box scores (`/game/{gamePk}/boxscore`, as in
      `docs/tracking/live-feeds.js` `mlbPlayers`). First full week of archives:
      Oct 7–13, 2026, so the first recap can run Monday Oct 13 (postseason only: few
      games; say so in the article).
- [ ] 4. NHL weekly recap generator from `artifacts/nhl/runs` (reuse
      `build_market_results.py` helpers), same sections.
- [ ] 5. Schedule both (e.g. Monday morning ET) in their workflows, add the blog index
      entries, link them from the home page box (step 1 picks them up by file name),
      and update `scripts/research/daily_process.html` + rebuild (AGENTS.md).
- [ ] 6. NBA once its season has a week of archived evidence (after Oct 20).

## How to run and check

- Tests: `python -m unittest discover -s tests -p 'test_editorial*.py'`,
  `python -m unittest tests.test_mlb_archive`, `node tests/briefing_picks.cjs`,
  `python scripts/build_daily_process.py --check` (required whenever a workflow or
  `briefing-picks.js` changes: edit `scripts/research/daily_process.html`, then run
  `python scripts/build_daily_process.py`).
- SEO: `python scripts/seo_check.py --changed` for any new public page (SEO_POLICY.md).
- The Claude container cannot reach statsapi.mlb.com, espn.com or nhle.com; test
  feed parsing with fixtures and let GitHub Actions do live fetches.

## Session log

- 2026-10-06 (Claude): steps 0–2 shipped in one PR. Earlier the same day: news feed
  fixed to ESPN's API (#96), home page redesign (#97–#101). NFL Week 4 recap /
  Week 5 preview is due from the Wed Oct 7 10:00 ET run; its Week 4 pregame archive
  was frozen 2026-09-30 18:56 UTC, before Thursday kickoff.
- 2026-10-07 (Claude): steps 0–2 merged as #102. Also published the Chris Sale
  pitcher-outs methods post (`docs/blog/chris-sale-pitcher-outs-2026.html`, data in
  `docs/blog/chris-sale-pitcher-outs-2026/`), featured on the home page through
  Oct 12 via `config/editorial.json`. Next: step 3 (MLB weekly recap generator);
  the MLB archive starts filling from the first `mlb-daily` run after #102.

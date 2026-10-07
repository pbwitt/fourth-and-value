# Weekly recaps: operating handoff

Updated October 7, 2026. Owner's cadence: **MLB Monday, NHL Tuesday, NFL
Wednesday, NBA Thursday**, in America/New_York. The homepage links to reviews;
it does not headline a win–loss record or units.

## Delivery and recovery

| Sport | Publication day | Review window | Evidence and results |
|---|---|---|---|
| MLB | Monday | Previous Monday–Sunday | Compact pregame snapshots; published morning cards; official StatsAPI final games and boxscores |
| NHL | Tuesday | Previous Tuesday–Monday | Durable forecast runs and hash-verified official NHL result archives |
| NFL | Wednesday | Completed football week, next-week preview | Immutable weekly pregame archive and nflverse completed results |
| NBA | Thursday | Previous Thursday–Wednesday | Durable refresh snapshots; exact fixture matching to official ESPN final scoreboards/boxscores |

Primary recovery for MLB/NHL/NBA runs after the existing morning pipeline,
including mornings when the card gate skips new research. `recap_schedule.py`
computes the latest due week for each sport, so a missed day is caught up.
Daily 10 a.m., noon and 6 p.m. ET GitHub starts are backups. Scheduled starts can
be delayed; they are not guaranteed publication times.

The delivery gate requires both the correctly dated HTML and matching JSON
summary. Complete reports are skipped. Waiting checks back off two hours;
published unresolved reports retry once per Eastern date through three days
after the period end. Manual dated reruns can incorporate later official
corrections. NFL has its own completion guard, Wednesday 10 a.m./noon/6 p.m.
starts, and missing-report recovery in Wednesday/Thursday morning refreshes.
No new NFL archive can be frozen after kickoff.

Both recap workflows use `weekly-recaps` concurrency, `queue: max`, and no
in-progress cancellation. This serializes their shared homepage/blog/sitemap
writes without replacing pending jobs. Other site writers retain their rebase
protection; conflicting publication fails visibly with saved artifacts.

## Implementation status

- [x] Homepage latest-per-sport discovery (`latest_recaps` in `editorial.py`).
- [x] MLB immutable compact archives (`scripts/mlb/archive.py`), including saved
      quote, model and capture timestamps where available.
- [x] MLB weekly published-pick and available full-board grading.
- [x] NHL weekly grading from existing archived predictions/results.
- [x] NBA durable archives (`scripts/recap_archive.py`) committed by NBA refresh.
- [x] NBA results adapter (`scripts/nba_weekly_results.py`) and Thursday routing.
      Actual articles wait for sufficient saved pregame evidence and final results.
- [x] Staggered delivery, idempotent morning recovery, backup checks, status output.
- [x] NFL recovery guard, current-homepage refresh and article SEO metadata.
- [x] Grading, archive, schedule and adapter tests; PR checks and process docs.

The shared engine is `scripts/sport_weekly_review.py`. The operational workflow is
`.github/workflows/sport-weekly-recaps.yml`; its PR test workflow is separate.
NFL remains in `scripts/nfl_weekly_review.py` and `nfl-weekly.yml`.

## Grading contracts

- Read the canonical `editionRows` view in `docs/assets/briefing-picks.js`, taking
  the final valid morning edition per decision date. Exclude test editions and
  retain the original `policy_version`. Never rerank old predictions with today's
  selector or label earlier returns as evidence for a new selection policy.
- Published picks retain the exact offered line, side, book and price. Stake one
  unit per pick; pushes return the stake and remain in units-risked ROI. Missing
  participation/results/statistics remain unresolved and out of ROI.
- Full-board forecasts are a separate population: last valid pregame snapshot
  per game; most-offered same-book paired line; median tie break; average
  de-vigged probability across paired books. Do not select on results.
- Reject recorded quote, ingestion, model or capture timestamps at/after kickoff.
  Preserve the offered time and record a stricter official cutoff separately.
- Compare model and market probabilities on matched non-push outcomes only,
  conditioning model probabilities on non-push. Unknown push estimates at integer
  lines remain missing. Game-block bootstrap reports probability uncertainty.
- Mean errors require a matching forecast mean actually saved. NHL archived
  regulation goal means cannot substitute for a full-game total mean. MLB team
  totals are absent from the supported archive; do not fabricate them.
- NHL explicit unverified settlement contracts stay unresolved. MLB pitchers
  need confirmed starts; NBA props need confirmed positive minutes. Ambiguous
  names/fixtures and unsupported markets remain unresolved.

## Outputs, coverage and reproducibility

Public HTML/JSON/SVG: `docs/blog/<sport>-recap-YYYY-MM-DD.*`; date is period end.
NFL keeps `week-N-recap-week-N+1-preview-YYYY.html`. Each report has source paths,
exact graded rows, coverage limitations, policy/market breakdowns, model versus
market comparisons where available, and dated future-total research watchlists.
No fresh watchlist is invented when the archive lacks one.

`docs/recaps/status.json` records expected period and delivery state. Workflow
failure is retained if a sport errors, after preserving other successful outputs.
HTML, SVG, JSON/status and official response artifacts are retained for 90 days.
MLB/NBA official final responses are also committed under
`artifacts/<sport>/results/` for offline reproduction. NHL results are hash-checked
against the content-addressed archive.

MLB board archiving began October 7. Older published cards can be graded, but
cannot supply a full historical board. The October 12 report has partial board
coverage October 7–11; October 19 can cover the first complete archived calendar
week, October 12–18. October 13 is Tuesday (an error in the earlier handoff).

Initial implementation reports: MLB September 28–October 4 (published cards
only); NHL September 29–October 5 (available archived games). NBA remains in a
waiting state until its stored forecasts and completed results can be matched.

## Run and verify

```sh
python scripts/recap_schedule.py
python scripts/sport_weekly_review.py --sport mlb --end 2026-10-04 --offline
python scripts/sport_weekly_review.py --sport nhl --end 2026-10-05
python -m unittest discover -s tests -p 'test_sport_weekly_review.py'
python -m unittest discover -s tests -p 'test_recap*.py'
python -m unittest discover -s tests -p 'test_nba_weekly_results.py'
python -m unittest discover -s tests -p 'test_nfl*.py'
node tests/briefing_picks.cjs
python scripts/build_daily_process.py --check
python scripts/seo_check.py --changed
```

Use an explicit dated rerun to incorporate late corrections. Investigate failed
status/Actions jobs; never edit a pregame archive to make grading pass. If any
workflow, freshness or grading rule changes, update `ANALYST_RESEARCH.md`, the
source `scripts/research/daily_process.html` and its generated public page.

## Session history

- October 6–7, Claude: PR #102 introduced homepage recap links, MLB compact
  pregame archives, and the original handoff. MLB/NHL generators remained open.
- October 7, Codex: reviewed and published the Chris Sale methods post (#103),
  correcting small-sample claims and calculation assumptions; no selection changes.
- October 7, Codex: implemented this handoff and revised delivery to the owner's
  Monday/Tuesday/Wednesday/Thursday schedule. At noon the standalone Wednesday
  NFL scheduled job had not started (latest direct run October 5); recovery now
  also uses the morning pipeline and explicit completion gates.

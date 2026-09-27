# September 27, 2026 test refresh

Local preview only; no merge, site deployment or wagers. Snapshot at **1:05:49 p.m. ET**.

Fresh authorized Odds API quotes were fetched around 12:53–12:55 p.m. ET.
NFL probabilities were recomputed at those offered lines from the latest successful
Week 3 production parameters (12:06 p.m.; history through September 21).
MLB inference used the verified September 26 history/model bundle and newly
retrieved official probable starters and published batting orders. The local
runtime matched production scikit-learn **1.9.1**; the initial version mismatch
correctly withheld MLB forecasts until corrected. Models were not retrained.

## Results

- Before kickoff: **474 NFL + 158 MLB candidates**, not recommendations.
- At 1:05:49 p.m.: **162 NFL + 158 MLB candidates**; started games were removed.
- **18 current-offer assessments completed during the run:** 13 consider, 2 wait,
  3 pass. Nine NFL reviews were already historical by the final snapshot because
  those games had started. The remaining nine reviewed offers are shown below.
- **NHL: no candidates.** Its existing feed was expired; NHL inference was not
  refreshed or tested by this NFL/MLB run.
- Research charge: **$0.785862**; recorded Eastern-day total **$1.729202 / $2.75**.
  The preview imposed a $2.25 reservation ceiling to retain $0.50 for the still-live
  legacy workflow. Subsequent batches stopped when their maximum reservations
  would exceed that ceiling; unused reservation capacity is not actual spending.
- **Independent discovery did not execute.** The remaining allowance could not
  cover its conservative maximum-cost reservation. No games were submitted to
  hosted search; this stage still needs its separate live smoke test on a fresh
  daily allowance. Direct sourced critiques did execute.

These are frozen test offers, not a continuously refreshed betting card.
Model probabilities remain experimental; no ROI or validated betting edge is established.

| Candidate | Price / book | Quote time ET | Model win | Other-book win | Assessment |
|---|---|---|---|---|---|
| Jonathan Aranda Over 0.5 Batter RBIs | +331 / DraftKings | 12:54:19 PM | 30.0% | 26.9% | consider |
| Tampa Bay Rays @ Philadelphia Phillies Over 6.5 Game total | +100 / LowVig.ag | 12:53:46 PM | 63.2% | 48.9% | consider |
| Junior Caminero Over 0.5 Batter RBIs | +239 / DraftKings | 12:54:19 PM | 35.6% | 30.2% | consider |
| Jonny Deluca Over 0.5 Batter hits | -125 / BetMGM | 12:54:10 PM | 66.8% | 53.8% | consider |
| Victor Mesa Jr. Over 0.5 Batter hits | -142 / DraftKings | 12:54:19 PM | 66.8% | 56.8% | consider |
| Jonny Deluca Over 0.5 Batter total bases | -125 / BetMGM | 12:54:10 PM | 63.2% | 52.7% | consider |
| Nick Fortes Over 0.5 Batter total bases | +101 / Caesars | 12:53:31 PM | 56.2% | 47.7% | consider |
| Nick Martinez Over 14.5 Pitcher outs | -165 / Bovada | 12:52:42 PM | 77.1% | 59.4% | wait |
| Los Angeles Angels @ Seattle Mariners Moneyline | +144 / FanDuel | 12:54:21 PM | 49.5% | 40.8% | wait |

The seven current “consider” rows are all related to Rays–Phillies. They are not
independent confirmations and should not be treated as a seven-bet portfolio.
Research scheduling and this small budget concentrated reviews on the earlier
MLB game; most of the full candidate pool remains unreviewed. “Consider” means a
case worth human assessment, not approval or proof it is among the day's best bets.

## Examples of the actual critique

- **Rays–Phillies over 6.5, +100:** the model projects 8.284 runs, with both orders
  published. The assessment identifies the scoring case, the large disagreement
  with ten other paired books, and weather uncertainty. It does not raise the
  model probability based on the narrative.
- **Angels moneyline +144:** wait for the remaining batting order; the near-even
  scoring projection is sensitive to personnel. The earlier +160 review was not
  reused as a review of +144.
- **Nick Martinez over 14.5 outs, -165:** wait because interruption timing matters
  directly to completing five innings.
- **Jared Goff over 0.5 rushing yards, +128 (now started):** pass; the positive
  mean lacked a defensible carry/scramble distribution at a near-zero threshold.

## Defects found and corrected

1. Completed MLB/NHL consider assessments could appear below twenty unreviewed
   NFL offers. Current assessment status now orders the combined table, retaining
   sport-specific numerical order; no cross-sport probability score is invented.
2. The summary counted earlier changed-price reviews as current reviews. It now
   separates current reviews from earlier analysis needing a recheck.
3. The certainty parser rejected “not guaranteed volume.” Only this explicit
   negation is allowed; positive guarantee claims and numeric confidence remain
   rejected. The affected saved response was revalidated without another API call.

## Verification and audit

- Browser: actual preview at desktop and mobile widths, expanded pool, quote/model
  fields, review links and started-game removal; no JavaScript errors or page overflow.
- Regression checks: shared selector, research browser, 14 NHL analyst tests,
  19 shared analyst tests, research/budget tests and generated-process-page check.
- Structured result and source/input hashes: [test-refresh-2026-09-27.json](test-refresh-2026-09-27.json).
- Exact input files, original API responses, source excerpts and revalidation record
  remain in the isolated local test workspace `/private/tmp/fv-testrefresh-20260927`.
- The settled test charges are retained in `artifacts/analyst/daily-budget.json`
  on the review branch so a same-day rollout does not grant a fresh allowance.

The selection/review changes remain in draft PR #20. No live feed was replaced
with test outputs. Forecast quality, the unexecuted discovery stage and broad
slate research coverage cannot be validated by this refresh alone.

## Follow-up: manageable daily card

After the user clarified a five-to-ten daily target, the preview was changed to
show **at most 10** current reviewed ideas, with no minimum. Wait, pass and pending
research remain in a separate collapsed pool. Alternate versions are consolidated.
At 1:23 p.m. ET the same frozen inputs produced **6 distinct card entries** and
**314 additional research offers**. All six card entries share Rays–Phillies
exposure; the header and individual rows disclose that concentration. No new
research calls, probability changes or wagers were made. Desktop/mobile rendering,
empty cards, changed/expired reviews, tracker identity and the complete pool were
checked. The process page includes the simple flow and this presentation policy.

# Does fading a sportsbook line outlier work?

Reviewed September 17, 2026, using the frozen September 9 Week 1 board and completed September 9–14 results. This is a new local analysis; the published blog and production model are unchanged.

The user's idea is coherent: if four books are at 30 and another at 35, investigate the under at 35; reverse the direction for a low outlier. At equal prices and settlement terms, the more favorable threshold dominates the less favorable one. Different prices can erase that advantage. A hit alone does not show that the outlier was wrong: the result may also win at the consensus line.

## What the existing logic does

The live build uses `scripts/market_math.py:add_market_comparisons`, called by both `make_props_edges.py` and `build_props_site.py`.

- `consensus_line` takes each book's median offered line, then the median across books. This reasonably prevents a book listing many alternate lines from getting many independent votes. It is a descriptive median, not a count of books agreeing at an exact threshold; it can be an unoffered midpoint.
- `consensus_prob` is the median of paired, de-vigged prices at the exact event, player, market, line and side. Matching exact lines is appropriate.
- `book_count` counts books with usable paired probabilities at that specific line. It does **not** report how many books support `consensus_line`.
- The target book participates in the displayed aggregates. At a unique outlier line, the fair consensus can simply equal that one book's own de-vigged probability. It is then not independent corroboration from other books.
- Top Picks qualifies on a fitted model, freshness and positive model-versus-price edge; it is not a consensus-outlier selection system.

The older standalone `make_consensus.py` still combines alternate-line prices when removing vig and omits event identity. It is not called by the current live path inspected here. The older `backtest_fade_outlier_lines.py` also groups by player/week/market/book and takes the first line, potentially pairing that line with a side quote from another alternate threshold. Its historical selector lacks a per-event pregame cutoff. Its prior output should not be treated as an audited answer to this question.

## Test definition

1. Use the already-audited Week 1 offers, actual outcomes and pending-result treatment from `review_week1_2026.py`; no model estimate affects selection.
2. For each book/player/game/market, choose one central offered Over/Under pair, using the paired fair probability closest to 50/50. Tie-break by lower hold, then lower line. This is a main-line proxy, since the feed does not distinguish every alternate within a raw market label.
3. Exclude the target book from the consensus. Require at least four other distinct books at exactly the same line, a strict majority of the other books, and no more than 30 minutes of quote-time skew among the agreeing peers and target.
4. Bet Under at a higher target line, Over at a lower target line. Use the actual price for that exact book, line and side. No hindsight price threshold or model filter is applied.
5. For the strict test, require **all** other available books to agree: one lone outlier versus four or more peers. This naturally gives at most one ticket per player/game/market.
6. For the broader majority test, keep one ticket per player/game/market by largest absolute gap, then best payout, then alphabetical book. The rule uses no outcomes.
7. Compare with the **same side on the same player/game**, bet at the best price among the agreeing books at the consensus threshold. This baseline distinguishes wins caused by the extra room from wins both thresholds would have produced.

## Results

All returns risk one unit per graded ticket. Pending results are excluded; no selected pushes occurred. These are hypothetical price-snapshot results, not confirmed sportsbook settlements or evidence of execution at later times.

| Definition | Graded | W–L | Win rate | Net units | ROI | Same-side consensus W–L | Consensus ROI |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| Lone outlier, all 4+ peers agree | 57 | 35–22 | 61.4% | +3.78 | +6.6% | 29–28 | +2.5% |
| Exactly four peers versus one outlier | 32 | 19–13 | 59.4% | −0.26 | −0.8% | 15–17 | −0.8% |
| Broader 4+ majority, one selected ticket per player/market | 126 | 65–61 | 51.6% | −10.35 | −8.2% | 57–69 | −10.2% |

The strict test selected 60 tickets, with three unresolved. The exact four-versus-one subset selected 33, with one unresolved. The broader test selected 133, with seven unresolved. These are overlapping definitions, not independent samples.

The strict outlier threshold turned six consensus losses into wins, adding 2.37 units versus the paired consensus-price baseline after accounting for different prices. In the exact four-versus-one subset, four extra wins produced only about 0.007 units of net improvement: paying more for the better number almost completely offset the improved hit rate.

The strict result is encouraging but uncertain: a 5,000-draw game-block bootstrap across its 15 represented games gives an approximate ROI interval of −11.8% to +24.6%. The paired ROI improvement interval also crosses zero (approximately −7.7 to +18.7 percentage points). This one week is not evidence of a stable profitable rule.

## High outlier versus low outlier

Within the strict test:

| Direction | W–L | Net units | ROI | Same-side consensus ROI |
| --- | --- | ---: | ---: | ---: |
| Under at unusually high line | 16–7 | +4.03 | +17.5% | +18.8% |
| Over at unusually low line | 19–15 | −0.24 | −0.7% | −8.5% |

The high-line Unders won three additional tickets, but the prices were sufficiently worse that they earned slightly less than their consensus-line counterparts. The low-line Overs also gained three wins, materially reducing their losses. This is why hit rate and threshold quality cannot replace a price-aware comparison.

## Real examples

- **Zay Flowers receptions:** five other books at 4.5; BetMGM at 5.5, Under −160. He caught five passes. Under 5.5 won; Under 4.5 lost. The best consensus Under was +128, so this was better protection at a substantially lower payout.
- **Stefon Diggs receptions:** four other books at 3.5; BetMGM Under 4.5 at −210. Four catches turned a consensus Under loss into an outlier Under win. The corresponding best consensus Under was +113.
- **RJ Harvey rushing yards:** four other books at 18.5; BetMGM Over 16.5 at −120. His 18 yards won at the lower threshold and lost at consensus; the consensus Over price was −107.
- **Derrick Henry rushing yards:** five other books at 79.5; FanDuel Under 80.5 at −114. He ran for 144 yards. Both unders lost; the extra yard did not fix the directional miss.

## Which markets looked best?

Among strict lone-outlier opportunities, receptions went 13–3 for +4.30 units (+26.9%); five outcomes flipped from consensus losses to outlier wins. Passing attempts went 5–0 for +3.98 units, but all five also won at consensus, where better payouts would have earned +4.59 units. Receiving yards went 12–10, +0.25 units; rushing yards went 4–6, −2.54 units. Passing completions went 1–2, −1.20 units; the only passing-TD selection lost. No passing-yard or rushing-attempt opportunities met the strict rule. These small market slices are descriptive, not reliable ranking estimates.

## Assessment

Keep the idea as a price-aware line-shopping signal. The current median is a useful starting point, but the user's exact hypothesis needs an explicit count of agreeing independent books, the target excluded from its own reference, a central-line rule, timestamp alignment, and exact-side pricing. The better number demonstrably changed outcomes in Week 1. It did not automatically create profit, and broadening the definition made the observed record substantially worse.

Existing archived 2025 snapshots provide potential additional research data. A longer test should reconstruct main lines and pregame provenance per event before using those weeks; the older backtest's latest-file-per-week shortcut is insufficient. No historical profit claim beyond the audited 2026 Week 1 sample is made here.

## Reproduce and inspect

```sh
.venv/bin/python scripts/review_week1_2026.py
.venv/bin/python scripts/analyze_consensus_outliers_week1_2026.py
```

The second script writes `reports/consensus-outliers-week1-2026/audit.json` with summaries and every selected quote, price, result, peer count and paired consensus control. Local CSVs include all central book lines, all qualifying outlier offers and the broader one-ticket selection. The raw board comes from frozen Git revision `6d420cf`; quote and result provenance is retained in the original Week 1 audit.

# NBA paper evidence (FV-2026-05)

`evidence.json` holds every number in the technical report
[NBA Forecasting and Market Pricing](https://fourthandvalue.com/research/nba-forecasting-and-market-pricing.html).
It is written by `scripts/research/nba/study.py` from the committed ESPN season archives
(`artifacts/nba/history/`), the production model (`artifacts/nba/model/nba-v1.json`) and the
v1 evaluation (`reports/nba-v1/evaluation.json`). The market section also reads the DraftKings
lines embedded in cached ESPN game summaries (`data/nba/espn/summary/`, gitignored), which
`scripts/nba/espn_history.py --season 2025-26` restores.

Protocol: select on 2024-25 with a 2023-24 fit, score 2025-26 once with a fit on both earlier
seasons. Intervals are 95% percentile intervals from 500 whole-game bootstrap replicates.

Rebuild:

```
.venv/bin/python scripts/research/nba/study.py        # about an hour on a laptop CPU
python3 scripts/research/build_nba_paper.py           # HTML + evidence JSON in docs/research/
node scripts/research/render_nba_paper.cjs            # PDF and browser checks (Playwright 1.55.1)
```

The builder asserts that the figures quoted in the manuscript text match this file.

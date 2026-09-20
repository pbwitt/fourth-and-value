# September 20, 2026: Colts–Chiefs prop article

Published article: `docs/blog/colts-chiefs-jones-passing-attempts-2026.html`.
Recommended wording: **conditional model lean**, Daniel Jones under 31.5 attempts at -110 or better; DraftKings tied with two books. Do not turn this into a high-confidence recommendation: removing the one-game defense adjustment eliminates the calibrated edge.

## Evidence

- Existing Week 2 model generated 2026-09-20 13:53:50 UTC. The article did not retrain the model.
- Fresh event-specific Odds API requests: passing attempts and receptions, plus spreads/totals; retrieved approximately 20:43 UTC (4:43 p.m. ET). No OpenAI API calls were used.
- `fresh-props.csv` preserves the returned prop sample. `game-odds.json` preserves the game lines and retrieval timestamp. These contain no API credentials.
- Public CSVs and JSON in `docs/blog/colts-chiefs-jones-2026/` preserve Jones's exact quotes, parameters, morning model rows, 2025 logs, and derived figures. Their timestamps distinguish the model build from the afternoon market refresh.
- Schedule: nflverse `nfldata/data/games.csv`, downloaded September 20, 2026. Cached `data/schedule_2025.csv` has midnight placeholders and must not be used for night splits. The 13 relevant verified rows are preserved in the article's game-log CSV.
- No 2025 Jones games in the verified sample started at/after 19:00 ET. Berlin is neutral site, not an Indianapolis home game. Removing his Week 14 Jacksonville early injury exit changes the road under count from 3/6 to 2/5.

## Reproduce

From this repository with the original local stats and parameters still present:

```sh
.venv/bin/python scripts/analyze_snf_jones_2026_week2.py --schedule docs/blog/colts-chiefs-jones-2026/jones-2025-game-log.csv
```

The analysis accepts the preserved game-log CSV as its schedule input because it includes the needed nflverse columns. It performs no network calls. It asserts reconstruction of the stored production parameters and raw/calibrated probabilities. If local Week 2 parameters or historical source files change, use the published snapshots to audit the original article rather than silently rewriting it with newer inputs.

Charts require matplotlib, numpy and scipy, and read only the saved analysis JSON:

```sh
python scripts/plot_snf_jones_2026_week2.py
```

## Editorial limits and checks

The baseline is 29.94 attempts, lowered to 26.99 by a 0.90136 defensive multiplier based on one game. No road, night, weather, spread or play-count adjustment is used for this market. Calibration has flat regions; 31.5 and 32.5 receive the same 63.6% final estimate. The article explains why that is a calibration-resolution limitation.

Official Colts/Chiefs sources support the Week 1 results, current injury designations, pressure context, recovery update and historical injury exit. Sources are linked beside the relevant claims. The weather source is a timestamped forecast, not a modeled input.

Verification: numerical reconstruction and sensitivity calculations; local links and asset existence; browser checks at 390, 768 and 1440 px for overflow, charts, anchors, blog search and navigation to the post. SVG and PNG charts inspected visually. CSV downloads must be force-added because this repository ignores CSVs globally.

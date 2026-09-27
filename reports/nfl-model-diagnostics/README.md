# Murray passing-yards audit — September 27, 2026

This is a pregame calculation audit, not a model validation or betting recommendation.
The dated [machine-readable record](2026-09-27-murray.json) preserves the original
forecast, offer, input trace, source hashes and provenance. The morning workflow
artifact supplied its immutable parameters and quotes. Weekly player history was
retrieved at 12:09:42 UTC for this reconstruction; it was not in that artifact.
Only seasons 2020–2025 and 2026 weeks before Week 3 enter the calculation.

## Findings

- Murray's only current-season appearance supplied five attempts, three completions
  and 18 yards. Attempts and yards per completion each give this recent sample 20%
  weight. The completion-rate input is 0.6, without career shrinkage. Appearance
  totals do not distinguish injury exits from normal starter workloads.
- The actual components are 26.0253706 attempts × 0.6 × 9.5947485 yards per completion
  = 149.824131 yards. The opponent adjustment raises this to 153.990676; the away
  multiplier reduces it to the published **144.7512355**. Sigma is **71.4467432**.
- At under 242.5, the raw Normal probability is **91.4365%**. Isotonic calibration
  sends it to **98%**, the upper endpoint beyond the fitted raw range. The artifact's
  12,904 graded rows cover all markets; market/tail counts and independent-game
  counts are not supplied. That total cannot validate this extreme probability.
- Bovada offers under 212.5 at −115, 232.5 at −185 and 242.5 at −240 in the same
  snapshot. This is an alternate-line ladder. Its central line is close to other
  books' 210.5–212.5 thresholds. The exact 242.5 paired probability, 66%, comes from
  Bovada alone, not independent agreement by multiple books at that threshold.
- Under −240 requires **70.5882%** wins without pushes. Holding sigma fixed in an
  explicitly hypothetical, uncalibrated Normal model, the mean reaches break-even
  at **203.8191** yards. Using 211.5 as a hypothetical mean yields **66.7816%**.
  A book's median threshold is not its expected mean; neither scenario is a new
  forecast or a validated adjustment.

The [Vikings' September 24 report](https://www.vikings.com/news/kyler-murray-return-to-lineup-reunion-with-baker-mayfield)
confirms the concussion ended Murray's opener on his 11th snap and that he cleared
protocol to resume his starting role. This directly challenges treating that short
appearance as normal opportunity. It does not establish his eventual passing total,
a new win probability, or a restricted workload on his return.

## Reproduce

Download the `nfl-refresh-36313509435` artifact from
[the morning workflow](https://github.com/pbwitt/fourth-and-value/actions/runs/36313509435).
Obtain `stats_player_week_YEAR.parquet` for 2020–2026 from the public
[nflverse player-stat release](https://github.com/nflverse/nflverse-data/releases/tag/stats_player),
saving as `weekly_player_stats_YEAR.parquet`. Compare hashes with the audit record
before expecting a bit-identical result; the upstream release may be revised.
With repository dependencies installed:

```sh
python reports/nfl-model-diagnostics/reproduce.py --artifact-dir /path/to/artifact --history-dir /path/to/history
python -m unittest discover -s tests -p 'test_nfl_prop_diagnostics.py'
```

The reproduction prints the components and asserts mean and sigma against the
archived forecast. It does not fetch prices, invoke AI, refresh publication dates,
or write to a model feed. Tests cover forecast cutoff, CSV trace retention,
exact-offer identity, paired quote timing, alternate lines, endpoint behavior and
push-aware omission. There is no claim of improved forecast accuracy or returns.

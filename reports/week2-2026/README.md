# NFL Week 2 audit / Week 3 editorial handoff

Prepared September 22, 2026. Article: `/blog/week-2-recap-week-3-preview-2026.html`.

## Reproduce

From the repository root:

```bash
.venv/bin/python scripts/review_week2_2026.py
.venv/bin/python scripts/build_week2_week3_editorial_charts.py
```

No API key or paid insight generation is used. Historical git objects `13c730d` and `5a4f3bf` must be available (a shallow clone needs those objects fetched).

Frozen results are in `source/`; weekly player stats were downloaded from the nflverse stats_player release on September 22. Schedule results are the saved 2026 schedule. Source hashes and pregame page hashes are in `summary.json`.

The prop audit takes the latest of the two pregame snapshots per game: Thursday for Detroit–Buffalo, Sunday for the other 15. All quote/build timestamps are checked against kickoff. Grade only supported markets with an unambiguous player/game match and offensive-usage evidence. Missing outcomes remain unresolved; actual sportsbook void/protection rules may differ.

The representative-line market analysis and highest-EV deduplicated shortlist are distinct result-blind selection rules. The latter is an illustrative portfolio, not placed bets. Both the deduplicated and all-offer results are disclosed. Whole-game totals use the separate saved September 17 snapshot, with nflverse's recorded close as a second benchmark.

Outputs include offers, shortlist, deduplicated tickets, representative lines, probability comparisons, games and summary. Public row-level JSON is `docs/blog/week-2-2026-review-data.json`. Correlated bets are not independent trials; the probability comparison uses a 5,000-replicate game-block bootstrap.

## Main findings

- Closing totals: 10 unders / 6 overs; 647 total points; favorites 8–8 ATS.
- Deduplicated shortlist: 189–245, +5.1657 units, 434 graded; 13 unresolved.
- All displayed offers: 779–1,037, −118.9702 units, 1,816 graded; 45 unresolved.
- Rushing yards +23.5568 units; receptions −15.1983.
- Matched Brier: model .250473 versus consensus .249117, 609 observations; difference interval spans zero.
- Raw totals directions: 5–11; raw MAE 12.8582 versus saved market 11.0.
- Featured Jones under31.5 −110: win with31. Featured Skattebo under1.5 +118: loss with4. Combined −.0909u.

## Week 3 evidence and caveats

`reports/week3-2026-preview/` preserves relevant files from successful GitHub run `35712180278`, artifact `nfl-refresh-35712180278`. The run manifest reports history through September21, captured September22 around09:46UTC. Public preview JSON contains all16 raw totals, individual total quotes, and the Kraft reception offers.

**The automated Week 3 injury CSV has zero rows.** All16 injury totals rows consequently have zero adjustments. Do not interpret this as zero injury impact. The article explicitly flags missing coverage and uses official team/league reports manually. This editorial task does not repair the injury ingestion pipeline. Follow-up needed: investigate source/week availability, add a coverage/freshness status, and avoid a “no injury edge” interpretation when reports are missing. Do not fabricate quantified QB adjustments.

Confirmed news cited in the article includes Mariota starting, Murray cleared, Penix starting, Williams week-to-week, Reed estimated DNP, Dart's early exit, Pierce expected absence and Nacua's Monday inactivity. Team reports and NFL links appear next to each claim. No announcement-linked causal injury estimate is claimed.

Kraft is conditional research, not a released wager: plus120 or better at3.5, subject to Reed availability/route role. Receptions were the weakest model market. Totals gaps are uncalibrated baselines; article does not issue stakes or claim positive expected returns from them.

## Video

Script, JSON and YouTube metadata are under `content/videos/week-2-recap-week-3-preview-2026*`. User explicitly approved immediate creation and publication on September22, superseding the earlier script-review preference. Voiceover uses the existing Cedar setup. Keep voiceover and omit on-screen production labeling. Landscape master is 1920×1080, rendered with the new dedicated editorial renderer; old videos are unchanged.

Do not change the published Pirates post or unrelated NHL workspace edits.

## Final production verification

MP4 rendered at1920×1080, duration388.370567seconds, 74,966,667bytes. Chrome decoded and played the export; all15 narration segments have nonzero RMS in the exported audio track. Open/middle/end seeks succeeded. All15 scene layouts were inspected; article checks passed at1440px and390px without horizontal overflow. All local article/video links resolve. Thumbnail1280×720. Technical details are in the video directory render-info.json and playback-verification.json.

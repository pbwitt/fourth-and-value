# Low-scoring situations: pre-registered hypotheses (FV-2026-04)

Registered October 15, 2026 (ET), before any factor's outcome was computed. The only
earlier result in this area is the NFL wind check in
`reports/TOTALS_METHODOLOGY_2026-09-09.md`, which this study repeats on purpose.
Changes after this commit are listed under **Amendments** with a date and reason; a
factor is never dropped or redefined because of its result.

## Question

Do non-statistical circumstances (weather, schedule, travel, officials, stakes) lead
to games scoring **less than the closing betting total**, not merely less than average?

## Outcomes, per game

- `actual` = final combined score (NHL: shootout goal not counted; overtime counted,
  as sportsbooks grade totals).
- `line` = closing (or latest pre-game) total for the game.
- `residual` = `actual − line`. Negative means the game went under.
- `under` = 1 if `actual < line`, 0 if `actual > line`; pushes excluded from rates.

For each factor we report three numbers against all other games in the same sport and
period:

1. **Effect on scoring**: mean `actual` in factor games minus other games.
2. **Market adjustment**: mean `line` in factor games minus other games.
3. **Residual**: mean `residual` in factor games, with a 95% interval, and the under
   rate with a Wilson 95% interval.

## Pass rules (fixed in advance)

- **Discovery** and **replication** periods are separate, consecutive season blocks.
  Every factor is evaluated in both. Factor thresholds are fixed here and are not tuned.
- A factor **replicates** if its mean residual has the predicted sign in both periods
  and the replication one-sided p-value is below 0.05 (two-sided factors: same sign in
  both periods and two-sided replication p below 0.05).
- A factor is **bettable at standard prices** only if it also wins at least 52.4% of
  non-push bets in the replication period (break-even at −110), in the replicated
  direction.
- Across the whole study, Benjamini–Hochberg q-values (false discovery rate) are
  reported for the full-sample residual tests. A result with q ≥ 0.10 is described as
  not distinguishable from chance even if its own p-value is small.
- Factors with fewer than 100 qualifying games in a period are reported as
  descriptive only and cannot pass.
- Officials (referees, umpires) are classified using **discovery-period games only**:
  each official with at least 30 discovery games is ranked by mean residual and the
  lowest third is labelled "low-scoring". The test is whether that label still predicts
  unders in replication-period games. No official is named as low-scoring from
  full-sample data.

## Periods

- **NFL** (nflverse closing totals): discovery 2006–2015, replication 2016–2025
  regular season and playoffs. The 2026 season to date (Weeks 1–6) is a separate
  out-of-sample check, reported but not part of the pass rule.
- **NHL, NBA, MLB**: periods are set from source coverage before outcomes are computed:
  the earliest half of available seasons is discovery, the latest half replication. The
  exact seasons are recorded in an amendment when the data are assembled.

## Hypotheses

Direction: **U** predicted under, **O** predicted over, **2** two-sided.

### NFL

| ID | Factor | Definition | Dir |
|---|---|---|---|
| NFL-1 | Strong wind | Outdoor or open roof, wind ≥ 15 mph | U |
| NFL-2 | Freezing | Outdoor or open roof, temperature ≤ 32°F | U |
| NFL-3 | Precipitation | Outdoor or open roof, game weather text mentions rain, snow, showers, drizzle or sleet | U |
| NFL-4 | Snow | As NFL-3, snow only | U |
| NFL-5 | Dome team in the cold | A team whose home stadium is a dome or closed roof plays outdoors at ≤ 40°F | U |
| NFL-6 | Thursday night | Thursday game after Week 1, both teams on ≤ 4 days' rest | U |
| NFL-7 | Short week off Monday | Either team played the previous Monday and plays on Sunday (6 days' rest) | U |
| NFL-8 | Off a bye | Either team on ≥ 13 days' rest (regular season) | 2 |
| NFL-9 | Prime time | Kickoff 8:00 p.m. ET or later | U |
| NFL-10 | Divisional game | Division opponents | U |
| NFL-11 | Late divisional rematch | Second divisional meeting of the season, Week 10 or later | U |
| NFL-12 | Low-scoring referee | Referee in the lowest third by discovery-period residual (see pass rules) | U |
| NFL-13 | West Coast team, early kickoff | Pacific-time road team in an Eastern-time stadium, kickoff before 2:00 p.m. ET | U |
| NFL-14 | Late-season bad team | Week 15 or later, either team's win rate ≤ .300 entering the game | 2 |
| NFL-15 | New starting quarterback | Either team's starting QB differs from its starter in the previous game, which that QB had started at least 3 of the last 4 | U |
| NFL-16 | Altitude | Game in Denver | 2 |
| NFL-17 | Playoffs | Postseason game | U |
| NFL-18 | Neutral or international site | Neutral-site game (including international), excluding the Super Bowl | 2 |

### NHL

| ID | Factor | Definition | Dir |
|---|---|---|---|
| NHL-1 | Back-to-back | Either team played the previous day | O |
| NHL-2 | Both on back-to-back | Both teams played the previous day | 2 |
| NHL-3 | Rest mismatch | One team on a back-to-back, the other on ≥ 2 days' rest | O |
| NHL-4 | Backup goalie starts | Either team's starting goalie is not its season-to-date leader in starts (team has ≥ 10 games) | O |
| NHL-5 | Three in four nights | Either team playing its third game in four days | 2 |
| NHL-6 | Long road trip | Road team playing its 4th or later consecutive road game | 2 |
| NHL-7 | Homecoming | Home team's first home game after ≥ 4 straight road games | 2 |
| NHL-8 | Time-zone travel | Road team's previous game was ≥ 2 time zones away, within 2 days | 2 |
| NHL-9 | Divisional game | Division opponents | U |
| NHL-10 | Playoffs | Postseason game | U |
| NHL-11 | Low-scoring referee | Referee (either of two) in lowest third by discovery residual | U |
| NHL-12 | Outdoor game | Winter Classic, Stadium Series, Heritage Classic and similar | U |
| NHL-13 | Rust after a break | Either team on ≥ 4 days' rest (regular season) | U |
| NHL-14 | Opening month | October regular-season game | 2 |
| NHL-15 | Season's final two weeks | Last 14 days of the regular season | 2 |

### NBA

| ID | Factor | Definition | Dir |
|---|---|---|---|
| NBA-1 | Back-to-back | Either team played the previous day | U |
| NBA-2 | Both on back-to-back | Both teams played the previous day | U |
| NBA-3 | Tired visitor | Road team on a back-to-back, home team rested | U |
| NBA-4 | Three in four nights | Either team playing its third game in four days | U |
| NBA-5 | Altitude | Road game at Denver or Utah | 2 |
| NBA-6 | Time-zone travel | Road team's previous game was ≥ 2 time zones away, within 2 days | U |
| NBA-7 | Long road trip | Road team playing its 5th or later consecutive road game | U |
| NBA-8 | After the All-Star break | Either team's first game after the break | 2 |
| NBA-9 | Season's final two weeks | Last 14 days of the regular season | 2 |
| NBA-10 | Playoffs | Postseason game | U |
| NBA-11 | Low-scoring crew chief | Crew chief in lowest third by discovery residual | U |
| NBA-12 | Opening two weeks | First 14 days of the regular season | 2 |
| NBA-13 | Rest advantage | One team has ≥ 2 more days' rest than the other | 2 |

### MLB

| ID | Factor | Definition | Dir |
|---|---|---|---|
| MLB-1 | Cold | Outdoor (or open roof), temperature ≤ 50°F | U |
| MLB-2 | Heat | Outdoor (or open roof), temperature ≥ 85°F | O |
| MLB-3 | Wind blowing in | ≥ 10 mph blowing in from left, center or right field | U |
| MLB-4 | Wind blowing out | ≥ 10 mph blowing out to left, center or right field | O |
| MLB-5 | Wrigley wind in | MLB-3 at Wrigley Field | U |
| MLB-6 | Wrigley wind out | MLB-4 at Wrigley Field | O |
| MLB-7 | Roof closed | Retractable roof closed or dome | 2 |
| MLB-8 | Rain | Drizzle or rain recorded at game time | U |
| MLB-9 | Day after night | Day game where either team played a night game the previous day | U |
| MLB-10 | Getaway day | Either team's next game is the following day in another city | U |
| MLB-11 | Doubleheader | Either game of a doubleheader | 2 |
| MLB-12 | Low-scoring plate umpire | Home-plate umpire in lowest third by discovery residual | U |
| MLB-13 | Twilight start | Local first pitch between 4:00 and 6:30 p.m. | U |
| MLB-14 | September and later | Regular-season games from September 1 | 2 |

## Not testable with public data (stated in advance)

- Weather forecasts that changed after the line opened (no timestamped forecast and
  opening-line archive).
- NBA star rest and late scratches (no historical injury-report archive with
  timestamps).
- NHL goalie confirmation timing (we know who started, not when it was announced).
- NFL coaching changes and play-calling tendencies (statistical, out of scope here).

## Amendments

**A1 — October 15, 2026, 9:31 p.m. EDT, before any outcome was computed.**

1. *NHL scoring.* The registered text said the shootout goal is not counted. That is
   wrong for settlement: U.S. sportsbooks grade NHL totals on the official final score,
   which awards the shootout winner one goal. `actual` is the official final score.
2. *Test statistic.* "Mean residual has the predicted sign" is made precise as the
   **excess residual**: mean residual of factor games minus mean residual of the other
   games in the same universe and period. Its p-value is Welch's two-sample t-test. This
   removes era-wide drift (for example, NFL overs running slightly ahead of the line).
   Bettability still uses the factor games' own under (or over) rate.
3. *Universes.* Weather factors compare against other outdoor (or open-roof) games
   where the same measurement is recorded. NFL-8 and NFL-14 compare within the regular
   season. Officials compare against games worked by other officials rated in discovery.
   Every other factor compares against all games in the period.
4. *Lines by source.* NFL: nflverse `total_line` (closing). NHL and NBA through 2022–23:
   the SportsbookReviewsOnline closing total (preserved by the Internet Archive). NHL and
   NBA from 2023–24, and all MLB: the median of the books' final pre-game totals in
   ESPN's odds feed, excluding live, opening and retired feeds.
5. *Officials.* The NBA crew chief is the first-listed official in ESPN's data. NHL
   games have two referees; the factor applies if either is rated low-scoring. Because
   the discovery period selects the officials, discovery-period figures for NFL-12,
   NHL-11, NBA-11 and MLB-12 are in-sample by construction and only the replication
   period counts.
6. *Season labels.* NHL and NBA seasons are labelled by the year they start (2019 =
   2019–20, including the 2020 bubble playoffs).

**A2 — October 15, 2026, 9:54 p.m. EDT, after the game tables were assembled and before any
NHL, NBA or MLB factor outcome was computed.**

1. *Periods.* Source coverage: NHL and NBA 2007–08 through 2025–26 (19 seasons), MLB 2010
   through 2025 (16 seasons), plus the 2026 MLB season and the first 107 games of the 2026–27
   NHL season. Discovery / replication: NHL and NBA 2007–2015 / 2016–2025 (season labels by
   start year); MLB 2010–2017 / 2018–2025. The 2026 MLB season (the first with the
   ball-strike challenge system) is a separate out-of-sample check, reported but not part of
   the pass rule, like the 2026 NFL season. The 107 NHL games of 2026–27 are too few and are
   not reported.
2. *Data-quality filters* (applied identically to every factor): games matched to the odds
   archive are dropped when the archive's final score disagrees with the official score
   (NHL 12 of 19,286; NBA 19 of 19,736), since the match may be wrong; NBA All-Star and
   exhibition games are excluded; lines outside plausible ranges are treated as missing
   (NBA outside 150–300 points, MLB outside 5–15 runs, NHL outside 4–8 goals), which removes
   four MLB games and no NHL or NBA league games.

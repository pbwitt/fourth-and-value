# Matchup backtest: player vs opponent and home/away

**Question.** Bettors often say a player "owns" an opponent (a quarterback who always shreds one
defense, a hitter who always hits one pitcher) or is much better at home. Once a forecast already
knows the player's form, the opponent's defense and the venue, does adding the player's own history
against that opponent, or his own home/away split, make it more accurate?

**Answer: no, in all three sports we model.** Matchup history and personal home/away splits added
nothing measurable out of sample. When the backtest was allowed to choose how much weight to give
them, it chose zero or close to zero. Two other things did help: league-wide home/away effects of
the right size, and platoon (batter side vs pitcher hand) in baseball.

| Sport | Data and test | Player vs this opponent | Player's own home/away split | What did help |
|---|---|---|---|---|
| NFL | nflverse 2012 to 2026 wk 4; 7 markets; test 2024 to 2026 | Zero weight in 4 of 7 markets. Elsewhere no gain; rushing yards were slightly worse, within noise | No gain in any market | The league home/away gap is about ±1–1.5% for passing and receiving yards, not production's fixed ±6%. Fitted values gave clearly better 2024–26 passing-yard and receiving-yard forecasts |
| NHL | nhl-v2.3 history 2022–23 to 2025–26; test 2025–26 (47,230 player-games) | At most 2% weight after 8 meetings; no change on the test | No change | Correcting for players the model consistently over- or under-projects (shots and points) |
| MLB | Retrosheet, 2.46M plate appearances 2012–2025; test 2023–25 (549,195 PA) | After 30 PA: 3% weight for hits and HR, 9% for K and walks. Changes are tiny; none at all among 25+ PA matchups | No gain | Platoon: same-handed matchups have 12% fewer HR, 13% fewer walks and 8% more strikeouts than opposite-handed ones. The production model has no handedness |

**"Owning" an opponent does not carry over.** Each line takes players whose earlier record against
an opponent was far above or below our forecast, then shows how they did *the next time* against
that same opponent (actual ÷ forecast):

- NHL shots (4+ earlier games): those 49% above came in at 1.025 next time; those 42% below at 1.021.
  All players with that much history averaged 1.015.
- MLB hits (20+ earlier PA vs the pitcher): those 62% above came in at 1.053; those 49% below at
  1.025. All such pairs averaged 1.003. A 110-point gap in history became a 3-point gap.
- NFL receiving yards (4+ earlier games): those 39% above came in at 0.953; those 30% below at 0.941.
  The reference group averaged 0.946.
- Aaron Judge against pitchers he had faced 15+ times (2023–25, 222 PA): 53 hits, against 54.1
  forecast without any matchup history.

**Why.** The samples are small, and the matchups that look decisive are mostly noise:
- 20 plate appearances against one pitcher carry about ±.100 of noise in batting average (one standard deviation).
- An NFL player meets a non-division opponent about once every three or four years, and the
  rosters and coordinators change in between.

What good matchup modeling uses instead are the *mechanisms* behind matchups, measured on large
samples:
- handedness;
- the opponent's current defense against the position;
- park and venue;
- game script (implied team totals and spreads);
- weather;
- lineups and injuries.

Our models already use the opponent's defense, park and venue. They lack platoon in MLB and implied
team totals in NFL props.

**Candidate changes** (not implemented; each needs its own validation in the production pipeline):
1. NFL: replace the fixed home/away multipliers with fitted ones. The fitted gap is about
   ±1.1% for passing yards and ±1.5% for receiving yards, against ±6% now.
2. MLB: add platoon (batter side vs pitcher hand) to the batter and pitcher prop models.
3. NHL: add a shrunk correction for players the model consistently over- or under-projects. A
   league home/away factor (shots ±2%, scoring ±4%) is a smaller, not yet significant gain.
4. Do not add head-to-head or personal home/away splits to any model. If useful, show
   "vs this opponent" history in the player snapshot as context labeled as not used by the model.

**Limitations.**
- These are tests of predictive accuracy (log loss, squared error, Brier score). They are not
  betting returns: archived prop lines for these seasons are unavailable.
- The NFL baseline rebuilds the production recipe from public data. It is not production's saved
  weekly forecasts.
- The MLB baseline is a plate-appearance model built for this test (batter and pitcher rates,
  park, home). It is not the production game-level model.
- "Especially at home" (matchup × venue) could not be tested reliably. Splitting already-thin
  matchup samples in half leaves almost nothing, and the matchup effect alone is about zero.

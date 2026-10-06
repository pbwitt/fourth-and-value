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

# Details

Research only. No production model, pick rule or published page reads these files. Every adjustment uses games before the one forecast; shrinkage is chosen on validation seasons and scored once on later test seasons. Scripts: `scripts/research/matchups/`; regenerate this page with `python scripts/research/matchups/report.py`.

## NHL player props

Baseline: nhl-v2.3 opportunity_nb_opp, refit per fold on earlier seasons. Test: 2025-26, 47,230 player-games in 1,312 games. Count log loss (lower is better); change vs. production with a 95% interval from resampling games.

| Variant | Shots log loss | vs production | Points log loss | vs production |
|---|---|---|---|---|
| production | 1.5407 | +0.0000 (+0.0000 to +0.0000) | 0.8385 | +0.0000 (+0.0000 to +0.0000) |
| + league home/away | 1.5404 | -0.0002 (-0.0005 to +0.0000) | 0.8385 | -0.0000 (-0.0005 to +0.0004) |
| + player overall (control) | 1.5379 | -0.0027 (-0.0035 to -0.0019) | 0.8379 | -0.0006 (-0.0012 to -0.0001) |
| + player home/away split | 1.5379 | -0.0027 (-0.0035 to -0.0019) | 0.8379 | -0.0006 (-0.0012 to -0.0000) |
| + player vs opponent | 1.5379 | -0.0027 (-0.0035 to -0.0019) | 0.8379 | -0.0006 (-0.0012 to -0.0001) |
| + both splits | 1.5379 | -0.0027 (-0.0035 to -0.0019) | 0.8379 | -0.0006 (-0.0012 to -0.0000) |
| naive: league home + vs opponent only | 1.5403 | -0.0003 (-0.0006 to -0.0001) | 0.8385 | -0.0000 (-0.0005 to +0.0004) |

Splits measured against the control (the same forecast without the split):

| Split | Shots | Goals | Assists | Points |
|---|---|---|---|---|
| + player home/away split | no detectable change +0.0000 (+0.0000 to +0.0000) | no detectable change +0.0000 (-0.0000 to +0.0000) | no detectable change -0.0000 (-0.0001 to +0.0000) | no detectable change -0.0000 (-0.0000 to +0.0000) |
| + player vs opponent | no detectable change +0.0000 (-0.0000 to +0.0001) | no detectable change +0.0000 (+0.0000 to +0.0000) | no detectable change +0.0000 (+0.0000 to +0.0000) | no detectable change +0.0000 (+0.0000 to +0.0000) |
| + both splits | no detectable change +0.0000 (-0.0000 to +0.0001) | no detectable change +0.0000 (-0.0000 to +0.0000) | no detectable change -0.0000 (-0.0001 to +0.0000) | no detectable change -0.0000 (-0.0000 to +0.0000) |

League home/away factor (training seasons): {'shots home': 1.0224, 'shots away': 0.9776, 'scoring home': 1.0408, 'scoring away': 0.9592}.
Chosen shrinkage (expected-count units of prior; 'none' = the history is ignored): {'shots': {'overall': '100', 'home/away split': 'none', 'vs opponent': '1000'}, 'scoring': {'overall': '100', 'home/away split': '1000', 'vs opponent': 'none'}}.
Weight the history earns: {'shots': {'vs opponent after 8 games': 0.019, 'home/away split after 40 games': 0.0}, 'scoring': {'vs opponent after 8 games': 0.0, 'home/away split after 40 games': 0.022}}.

**Matchup streaks** (test season, next game vs the same opponent):

- shots: everyone with 4+ prior games vs this opponent (reference): 34,964 games; earlier ratio 0.967; next game actual ÷ forecast 1.015
- shots: beat expectation by 30%+ vs this opponent (4+ prior games): 4,929 games; earlier ratio 1.492; next game actual ÷ forecast 1.025
- shots: fell 23%+ short vs this opponent (4+ prior games): 9,170 games; earlier ratio 0.581; next game actual ÷ forecast 1.021
- points: everyone with 4+ prior games vs this opponent (reference): 34,964 games; earlier ratio 0.954; next game actual ÷ forecast 0.994
- points: beat expectation by 30%+ vs this opponent (4+ prior games): 8,443 games; earlier ratio 1.715; next game actual ÷ forecast 0.989
- points: fell 23%+ short vs this opponent (4+ prior games): 13,492 games; earlier ratio 0.41; next game actual ÷ forecast 0.986

## NFL player props

Baseline: production recipe: EWMA(4) + career blend, opponent-defense rating, fixed home multipliers. Seasons: {'train': '2012-2021', 'validation': '2022-2023', 'test': '2024-2026 wk4'}. Squared error of the projection (lower is better) and Brier score at a half-point line on our production number.

| Market | Test rows | Fitted home ÷ away (production) | Home/away fitted vs fixed | Player home/away split | Player vs opponent | Naive vs opponent |
|---|---|---|---|---|---|---|
| pass_yds | 1,147 | 1.011 / 0.989 (1.06 / 0.94) | better -99.56 (-198.88 to -7.78) | no detectable change +0.28 (-15.83 to +13.05) | no detectable change +0.00 (+0.00 to +0.00) | better -99.56 (-198.88 to -7.78) |
| pass_attempts | 1,147 | 0.995 / 1.005 (1.00 / 1.00) | no detectable change +0.05 (-0.12 to +0.22) | no detectable change +0.14 (-0.03 to +0.29) | no detectable change +0.00 (+0.00 to +0.00) | no detectable change +0.05 (-0.12 to +0.22) |
| completions | 1,147 | 1.004 / 0.996 (1.00 / 1.00) | no detectable change -0.04 (-0.11 to +0.03) | no detectable change +0.19 (-0.04 to +0.39) | no detectable change +0.00 (+0.00 to +0.00) | no detectable change -0.04 (-0.11 to +0.03) |
| rush_yds | 1,409 | 1.032 / 0.969 (1.04 / 0.96) | no detectable change -1.77 (-3.75 to +0.33) | no detectable change +0.00 (+0.00 to +0.00) | no detectable change +5.26 (-0.40 to +11.11) | no detectable change +3.49 (-2.54 to +9.63) |
| rush_attempts | 1,409 | 1.018 / 0.983 (1.02 / 0.98) | no detectable change -0.00 (-0.03 to +0.02) | no detectable change +0.00 (+0.00 to +0.00) | no detectable change -0.01 (-0.05 to +0.04) | no detectable change -0.01 (-0.06 to +0.04) |
| recv_yds | 4,267 | 1.015 / 0.985 (1.06 / 0.94) | better -5.32 (-9.62 to -1.18) | no detectable change +1.43 (-0.01 to +2.68) | no detectable change +0.00 (+0.00 to +0.00) | better -5.32 (-9.62 to -1.18) |
| receptions | 4,267 | 1.008 / 0.992 (1.03 / 0.97) | no detectable change -0.01 (-0.02 to +0.00) | no detectable change +0.00 (+0.00 to +0.00) | no detectable change +0.00 (-0.00 to +0.01) | no detectable change -0.01 (-0.02 to +0.01) |

Chosen shrinkage, in games of prior (none = ignored), and Brier score at the line:

| Market | k overall / venue / vs opponent | Brier production | Brier + vs opponent | Brier + home/away split |
|---|---|---|---|---|
| pass_yds | none / 300 / none | 0.2500 | 0.2489 | 0.2491 |
| pass_attempts | 300 / 300 / none | 0.2495 | 0.2496 | 0.2498 |
| completions | 300 / 100 / none | 0.2490 | 0.2492 | 0.2502 |
| rush_yds | none / none / 30 | 0.2499 | 0.2504 | 0.2497 |
| rush_attempts | none / none / 100 | 0.2499 | 0.2499 | 0.2500 |
| recv_yds | none / 300 / none | 0.2501 | 0.2498 | 0.2503 |
| receptions | none / none / 100 | 0.2469 | 0.2467 | 0.2466 |

**Matchup streaks** (test seasons, next game vs the same opponent):

- pass_yds: everyone with 4+ prior games vs this opponent (reference): 290 games; earlier ratio 1.028; next game actual ÷ forecast 0.964
- pass_yds: beat expectation by 15%+ vs this opponent (4+ prior games): 41 games; earlier ratio 1.254; next game actual ÷ forecast 0.959
- pass_yds: fell 13%+ short vs this opponent (4+ prior games): 26 games; earlier ratio 0.814; next game actual ÷ forecast 1.038
- pass_attempts: everyone with 4+ prior games vs this opponent (reference): 290 games; earlier ratio 1.031; next game actual ÷ forecast 0.968
- pass_attempts: beat expectation by 15%+ vs this opponent (4+ prior games): 37 games; earlier ratio 1.238; next game actual ÷ forecast 0.984
- pass_attempts: fell 13%+ short vs this opponent (4+ prior games): 14 games; earlier ratio 0.79; next game actual ÷ forecast 0.976
- completions: everyone with 4+ prior games vs this opponent (reference): 290 games; earlier ratio 1.037; next game actual ÷ forecast 0.967
- completions: beat expectation by 15%+ vs this opponent (4+ prior games): 41 games; earlier ratio 1.249; next game actual ÷ forecast 0.987
- completions: fell 13%+ short vs this opponent (4+ prior games): 18 games; earlier ratio 0.781; next game actual ÷ forecast 1.038
- rush_yds: everyone with 4+ prior games vs this opponent (reference): 223 games; earlier ratio 1.083; next game actual ÷ forecast 1.014
- rush_yds: beat expectation by 15%+ vs this opponent (4+ prior games): 88 games; earlier ratio 1.339; next game actual ÷ forecast 0.936
- rush_yds: fell 13%+ short vs this opponent (4+ prior games): 43 games; earlier ratio 0.721; next game actual ÷ forecast 1.077
- rush_attempts: everyone with 4+ prior games vs this opponent (reference): 223 games; earlier ratio 1.058; next game actual ÷ forecast 0.992
- rush_attempts: beat expectation by 15%+ vs this opponent (4+ prior games): 65 games; earlier ratio 1.281; next game actual ÷ forecast 0.921
- rush_attempts: fell 13%+ short vs this opponent (4+ prior games): 36 games; earlier ratio 0.788; next game actual ÷ forecast 0.952
- recv_yds: everyone with 4+ prior games vs this opponent (reference): 718 games; earlier ratio 1.11; next game actual ÷ forecast 0.946
- recv_yds: beat expectation by 15%+ vs this opponent (4+ prior games): 306 games; earlier ratio 1.394; next game actual ÷ forecast 0.953
- recv_yds: fell 13%+ short vs this opponent (4+ prior games): 159 games; earlier ratio 0.702; next game actual ÷ forecast 0.941
- receptions: everyone with 4+ prior games vs this opponent (reference): 718 games; earlier ratio 1.088; next game actual ÷ forecast 0.957
- receptions: beat expectation by 15%+ vs this opponent (4+ prior games): 279 games; earlier ratio 1.337; next game actual ÷ forecast 0.968
- receptions: fell 13%+ short vs this opponent (4+ prior games): 134 games; earlier ratio 0.73; next game actual ÷ forecast 0.976

**Patrick Mahomes vs DEN** (passing yards, every regular-season meeting in the data):

| Season | Week | Venue | Yards | Forecast | Earlier meetings | Earlier actual ÷ forecast |
|---|---|---|---|---|---|---|
| 2018 | 4 | away | 304 | 275 | 0 | 1.0 |
| 2018 | 8 | home | 303 | 288 | 1 | 1.106 |
| 2019 | 7 | away | 76 | 269 | 2 | 1.078 |
| 2019 | 15 | home | 340 | 266 | 3 | 0.821 |
| 2020 | 7 | away | 200 | 276 | 4 | 0.931 |
| 2020 | 13 | home | 318 | 300 | 5 | 0.89 |
| 2021 | 13 | home | 184 | 283 | 6 | 0.921 |
| 2021 | 18 | away | 270 | 267 | 7 | 0.881 |
| 2022 | 14 | away | 352 | 259 | 8 | 0.897 |
| 2022 | 17 | home | 328 | 271 | 9 | 0.945 |
| 2023 | 6 | home | 306 | 305 | 10 | 0.971 |
| 2023 | 8 | away | 240 | 321 | 11 | 0.974 |
| 2024 | 10 | home | 266 | 251 | 12 | 0.953 |
| 2025 | 11 | away | 276 | 243 | 13 | 0.96 |
| 2026 | 1 | home | 184 | 239 | 14 | 0.971 |

## MLB plate appearances

Source: Retrosheet event files 2012-2025 via Chadwick Bureau (regular season). 2,457,250 plate appearances. Seasons: {'train': '2012-2019', 'validation': '2021-2022', 'test': '2023-2025'}. Log loss per plate appearance (lower is better), change vs. the production-like baseline.

| Variant | Hit | Strikeout | Walk/HBP | Home run |
|---|---|---|---|---|
| production-like (rates, park, home) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) |
| + platoon (bat side vs pitcher hand) | no detectable change -0.2 (-0.5 to +0.0) | no detectable change -0.4 (-0.9 to +0.3) | better -1.8 (-2.4 to -1.2) | better -0.9 (-1.2 to -0.5) |
| + player overall (control) | no detectable change +0.4 (-0.1 to +0.9) | better -3.9 (-5.2 to -2.6) | better -2.1 (-2.8 to -1.4) | better -1.0 (-1.7 to -0.4) |
| + batter home/away split | no detectable change +0.4 (-0.1 to +0.9) | better -3.9 (-5.2 to -2.6) | better -2.1 (-2.8 to -1.4) | better -1.0 (-1.7 to -0.4) |
| + batter vs this pitcher | no detectable change +0.5 (-0.1 to +0.9) | better -4.2 (-5.5 to -2.8) | better -2.3 (-3.0 to -1.5) | better -1.0 (-1.6 to -0.4) |
| + pitcher vs this team | no detectable change +0.4 (-0.1 to +0.9) | better -3.7 (-5.0 to -2.3) | better -2.1 (-2.8 to -1.4) | better -1.0 (-1.7 to -0.4) |
| + batter's own platoon split | no detectable change +0.4 (-0.1 to +0.9) | better -4.9 (-6.3 to -3.4) | better -2.1 (-2.8 to -1.4) | better -1.0 (-1.7 to -0.4) |
| naive: platoon + batter vs pitcher only | no detectable change -0.2 (-0.5 to +0.1) | better -0.8 (-1.4 to -0.1) | better -2.0 (-2.6 to -1.4) | better -0.8 (-1.2 to -0.5) |

Interval values are ×10⁻⁴ log loss per plate appearance.

Matchup additions measured against the control:

| Addition | Hit | Strikeout | Walk/HBP | Home run |
|---|---|---|---|---|
| + batter home/away split | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) |
| + batter vs this pitcher | no detectable change +0.0 (-0.0 to +0.1) | better -0.3 (-0.4 to -0.2) | better -0.2 (-0.3 to -0.0) | no detectable change +0.0 (-0.0 to +0.1) |
| + pitcher vs this team | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.2 (-0.0 to +0.4) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) |
| + batter's own platoon split | no detectable change +0.0 (+0.0 to +0.0) | better -1.0 (-1.4 to -0.6) | no detectable change +0.0 (+0.0 to +0.0) | no detectable change +0.0 (+0.0 to +0.0) |
| batter vs pitcher, only matchups with 25+ earlier PA | no detectable change -0.2 (-1.6 to +1.4) | no detectable change +2.0 (-2.5 to +6.8) | no detectable change -0.2 (-4.4 to +3.9) | no detectable change -1.0 (-2.7 to +0.5) |

Chosen shrinkage per outcome, in plate appearances of prior (none = ignored): {'overall': {'hit': '1000', 'k': '100', 'bb': 'none', 'hr': '1000'}, 'pitcher': {'hit': 'none', 'k': 'none', 'bb': '1000', 'hr': 'none'}, 'venue': {'hit': 'none', 'k': 'none', 'bb': 'none', 'hr': 'none'}, 'bvp': {'hit': '1000', 'k': '300', 'bb': '300', 'hr': '1000'}, 'platoon_own': {'hit': 'none', 'k': '1000', 'bb': 'none', 'hr': 'none'}, 'pvt': {'hit': 'none', 'k': '1000', 'bb': 'none', 'hr': 'none'}}.
Weight batter-vs-pitcher history earns after 30 PA: {'hit': 0.029, 'k': 0.091, 'bb': 0.091, 'hr': 0.029}.
League home factor: {'hit': {'home': 1.012, 'away': 0.9885}, 'k': {'home': 0.9788, 'away': 1.0203}, 'bb': {'home': 1.0359, 'away': 0.9655}, 'hr': {'home': 1.0146, 'away': 0.9859}}.
Platoon factor: {'hit': {'same hand': 0.9801, 'opposite hand': 1.0173}, 'k': {'same hand': 1.0414, 'opposite hand': 0.9638}, 'bb': {'same hand': 0.9228, 'opposite hand': 1.0641}, 'hr': {'same hand': 0.9337, 'opposite hand': 1.0603}}.

**Matchup streaks** (2023-2025, the next plate appearance against the same pitcher):

- every pair with 20+ PA (reference): 11,731 PA; earlier ratio 1.055; hits actual ÷ forecast 1.003
- 20+ PA vs this pitcher, hits 40%+ above expectation: 1,925 PA; earlier ratio 1.622; hits actual ÷ forecast 1.053
- 20+ PA vs this pitcher, hits 30%+ below expectation: 1,888 PA; earlier ratio 0.512; hits actual ÷ forecast 1.025
- Aaron Judge against pitchers he had faced 15+ times: 222 PA, earlier hit ratio 0.936, 53 hits vs 54.1 forecast without matchup history

MLB data: The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at "www.retrosheet.org". NFL data: nflverse. NHL data: the frozen nhl-v2.3 history.

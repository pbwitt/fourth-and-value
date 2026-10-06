# Model improvement plan

Owner: pbwitt. Started 2026-10-05. Update this file as each phase finishes.

## Status

| Phase | State | Notes |
|---|---|---|
| 0. Audit | Done 2026-10-05 | Findings and proposed plan below. Step 1 (housekeeping) merged and deployed 2026-10-05. |
| 1. News and lineup data | In progress | NHL projected starting goalies (start chance and shrunk save rate from official box scores) as context only, 2026-10-05. Confirmed starters still need a permitted source. |
| 2. Model fixes | Not started | |
| 3. Market blend before betting | In progress | Step 2 (blend live on Top Picks at w = 0.25, 3% EV) merged 2026-10-05 in pbwitt/fourth-and-value#87. Step 2b (2026-10-06): 1% floor with high (≥ 3%) and moderate (1–3%) confidence tiers, and the two-book rule counts the offered book for every sport. Pinnacle reference not yet added. |
| 4. Backtest and calibration | Not started | |
| 5. Monitoring and site | Not started | |

---

## The brief (as given, 2026-10-05)

Read the whole brief, start with Phase 0, and check in with the owner before making big
changes to the models.

### Context

- Our models produce `model_prob` and `edge_bps` for picks in NFL, NHL and MLB (player props,
  sides, totals). NBA starts soon and has to be supported.
- Picks go out in the morning through `morning-picks.yml`, plus `afternoon-refresh.yml`. Both are
  dispatched by pg_cron in Supabase (project ref `fzjonxpzsrbdhbujbhsn`, "fourthandvaluetracker").
- Odds come from The Odds API. The key is in Supabase Vault and readable by the service role
  through `public.live_odds_key()`. About 15,700 credits are left. Never let any job drop us below
  2,000, and ask the owner before any single task spends more than 2,000.

### Already built, so extend it rather than rebuild it

- Migration `closing_line_value` adds closing-line columns to `bets`: `event_id`,
  `commence_time`, `closing_line`, `closing_odds`, `closing_fair_prob`, `closing_books`,
  `clv_ev`, `beat_line`, `clv_status`, `clv_note`. A guard trigger keeps clients from writing
  those fields. The migration also creates `odds_snapshots`, which stores the raw Odds API
  payloads.
- `clv_ev = closing_fair_prob × decimal(odds) − 1`, where `closing_fair_prob` is the median
  no-vig probability across US books at the bet's exact line.
- Edge function `closing-lines` runs every 5 minutes and prices each game in the final 6 minutes
  before it starts. A daily backfill runs at 10:00 UTC. Market mappings, including NBA, live in
  `clv.mjs`.

### What week 1 showed (28 bets, $5 flat stakes)

- Results: 15–13, +$9.35, ROI +6.7%. But average CLV was −5.3%. At the bettor's own book, the
  line moved our way on 7 bets, held on 5, and moved against us on 10.
- The model averaged 61% on these bets; the market closed them at 47.6%. The model expected 15
  wins out of 25 and we got 12. The closing market expected 11.5 out of 24 and we got 12. In every
  market, the model was 10–19 points more confident than the market.
- Worst case was NHL shots on goal: model 71% vs market 52%. All 7 SOG bets and all 3 NHL points
  bets were unders, which suggests the distributions are too narrow.
- NFL went 5–1 (+$28.50), but at plus-money prices the market rated at about 42%, and its CLV
  was negative.
- Data problems: NFL `rush_yds` bets show `model_prob` exactly 0.50, which looks like a
  placeholder. Several bets were Bovada alternate lines. NHL rows with `market_type`
  "team_total" carry lines like 6.5 and 7.5, which look like game totals; ask before relying on
  them. `bets.timestamp` (placement time) is empty on every row. Team names must always be full
  names.

### Phase 0: Audit (report back before changing anything)

For each market: how is the probability produced (distribution, inputs, calibration step), when
does it run, and where could the 0.50 placeholders come from? Give a short summary and a plan.

### Phase 1: News and lineup data (top priority)

We need injuries, goalie changes, lineups and other key information in every prediction.

- NHL: confirmed starting goalies, line combinations and power-play units, scratches and
  injuries, time-on-ice trends. Use the NHL API (api-web.nhle.com) where possible.
- NFL: official injury reports (practice participation, game status), inactives about 90 minutes
  before kickoff, depth charts, snap, route and target shares (nflverse), weather for outdoor
  games.
- NBA: the official injury report, starting lineups and rest days, minutes trends, back-to-backs.
- MLB: probable pitchers, lineups, weather and park.
- Prefer official or free APIs. Where we have to scrape, check robots.txt and terms, cache
  results, rate-limit, and use an identifying user agent.
- Store every input with an as-of timestamp, so backtests only use information we actually had
  at bet time.
- When a player is out, redistribute his usage to teammates (targets, carries, ice time,
  minutes).
- Add news-timed runs on top of the morning run: after NHL goalie and lineup confirmations, after
  NFL inactives for each kickoff window, and close to NBA tip-off after injury report updates. If
  a published pick's inputs change, re-price it and flag it.

### Phase 2: Model fixes

- Model opportunity first, then rate. NHL SOG = (even-strength + power-play ice time) × shot
  rate. NFL receptions = routes or target share × catch rate. NBA = minutes × per-minute rates.
- Use overdispersed distributions: negative binomial for counts (SOG, points, strikeouts,
  receptions, pass attempts) and a skewed distribution or simulation for yards. No Poisson or
  normal where they don't fit.
- Early in a season, shrink player rates toward last season and career priors, and weight this
  season's games up as they accumulate.
- Never publish a default probability. If a market can't be priced, don't publish a pick for it.

### Phase 3: Blend with the market before betting

- At decision time, get the market's no-vig probability: the median across US books, plus
  Pinnacle (Odds API region "eu") as a sharp reference where it's posted.
- Final probability = sigmoid(w·logit(model) + (1−w)·logit(market)). Fit w per market in
  Phase 4; until then use w = 0.25.
- Edge = final probability × decimal(best available price) − 1. Bet only when edge clears a
  per-market threshold; start at 2–3%.
- Example: an SOG under at −140 where the model says 71% and the market says 52% blends to about
  57%, below the 58.3% break-even, so no bet.
- Always shop the best price across books, and flag any book that's stale against consensus.
- Record raw model probability, market probability, final probability, price, book and decision
  time on every pick, and populate `bets.timestamp` with when the bet was placed.

### Phase 4: Backtest and calibration

- For each market, score the raw model, the market and the blend with walk-forward splits, using
  log loss, Brier score and calibration by decile. Fit w and the thresholds out of sample.
- Historical odds come from `odds_snapshots` going forward, plus The Odds API's historical
  endpoint. Props go back to 2023 and cost about 10 credits per market per game. Start with NFL
  receptions, about 2,700 credits for a season.
- Success means the blend beats the market alone out of sample, and live CLV turns positive by
  market.

### Phase 5: Monitoring and the site

- Weekly CLV report by league and market: bet count, average `clv_ev`, percent of bets that beat
  the close, and the gap between model and close. Flag any market still negative after about 100
  bets.
- Tracker page: add a CLV card (average CLV, percent beating the close, breakdown by market) and
  a confidence band on ROI of about ±100%/√n.
- NBA: `closing-lines` already recognizes the `market_type` labels points/pts, rebounds/reb,
  assists/ast, threes/3pm, blocks, steals, turnovers, pra, pr, pa, ra and stocks. Use those labels
  in the tracker, or update `MARKETS` in `clv.mjs`.

### Ground rules

Work on branches in small, tested PRs. Don't break the live site or the scheduled workflows. Keep
secrets in Vault or GitHub secrets and never commit them. Ask the owner when something is
ambiguous instead of guessing.

---

## Phase 0 findings (2026-10-05)

Sources: repository at `a2b716e`, the deployed `closing-lines` function (v1), the `bets` table
(32 rows, 28 settled) and `cron.job`. Week-1 totals match the brief: 28 settled, 15–13, +$9.35,
average model probability 0.601, average `clv_ev` −5.30% over 24 captured closes.

### How each market is priced

| League | Markets | Distribution | Inputs | Calibration | Runs (ET) |
|---|---|---|---|---|---|
| NFL | rush/rec/pass yards, receptions, rush/pass attempts, completions | **Normal** for all of these, including the counts | nflverse weekly stats: EWMA of the last 4 games (α 0.4) blended with an exponentially decayed career baseline shrunk to a position pool (`career_baseline.py`). Volume × efficiency latents. Opponent-defense, fixed home/away multipliers, injury-status multiplier, manual overrides | Isotonic curve per market, fitted on 2025 weeks 4–14 (`models/nfl_prop_calibration.json`) | Daily 07:05 via Morning Picks; Wed 10:00 weekly rebuild; Thu 17:00; Sun 08:00, 11:00, 12:30, 15:55, 20:00 |
| NFL | pass TDs, INTs | Poisson | as above | pass TDs use the pooled isotonic curve | as above |
| NFL | anytime TD | — | — | withheld (no estimate published) | — |
| NHL | SOG, goals, assists, points (`nhl-v2.3`) | **Negative binomial** (α shots 0.065, scoring 0.007, so scoring is close to Poisson). Goals and assists split points by binomial allocation | NHL API game logs. Projected all-situations TOI × per-minute rate, history aged by games played (half-lives: TOI 14, rate 110, base 82.5), shrunk to a fixed position prior. Opponent shots/goals-allowed factor | None ("identity") | 07:05 and 16:30 |
| NHL | moneyline, puck line, total | Independent Poisson regulation goals (`poisson_core`) with an OT/SO allocation | Team attack/defense, recency-weighted | None | 07:05 and 16:30 |
| MLB | pitcher Ks, outs, hits/TB/HR/RBI, game lines | Ks: rolling baseline (BF × K rate × √(opp K / league)); others: gradient boosting. NB or Poisson for counts; **Normal** for outs | MLB Stats API boxes, probable starters, published lineups only | Isotonic on the CDF, chronological split | 07:05 and 16:30 |
| NBA | 11 prop markets, game lines | **No model.** A historical hit rate is shown when NBA Stats logs load (they are blocked on runners) | — | — | 11:00 and 17:00 (GitHub cron) |

Selection: `docs/assets/briefing-picks.js` builds Top Picks. NFL candidates need
`screening_ev = (1 − push) × (min(calibrated, raw) × decimal − 1) ≥ 3%`, ranked by that EV. MLB
needs `is_model_pick` (EV ≥ 3% and probability edge ≥ 3 points, ≥ 2 paired books, best price).
NHL comes from the analyst board ranked by worst-case log growth. **No sport blends in the market
probability before ranking.**

### Where the 0.50s come from

They are not a missing-value default. They are flat plateaus in the NFL isotonic calibration:

| Market | Raw probability range mapped to exactly 0.50 |
|---|---|
| rush_yds | 0.334–0.666 |
| pass_yds | 0.128–0.872 |
| pass_completions | 0.30–0.70 |
| pass_tds (pooled curve) | 0.334–0.666 |
| rush_attempts, receptions | narrow plateaus around 0.50 |
| recv_yds | every input maps to 0.484–0.516 |

The isotonic fit learned that the raw model has almost no information in those ranges, so it
returns 0.50 regardless of the line. Combined with an EV-ranked selector, a coin-flip estimate
makes the longest plus-money price look best, so the screen picks alternate lines. Example from
the Oct 4 card: Croskey-Merritt rush_yds Under 49.5 at Bovada +155. Raw model 0.615, calibrated
0.50, screening EV 27.5%. The main line was 60, and the market's fair price at 49.5 was 0.367.
The three NFL rush_yds bets and the Croskey-Merritt receptions candidate (raw 0.503, calibrated
0.50, +135) all came through this path.

### Other findings

1. **NHL week-1 bets used v2.1**, which aged player history by calendar days. After the
   offseason, last season had almost no weight, so stars were pulled toward the position prior.
   Examples: Werenski SOG 3.5 (season average 3.47) priced at 0.826 Under against a 0.54 market;
   Pastrnak 0.786 against 0.548. v2.2 and v2.3 (Oct 4) fixed most of the level bias. v2.2's
   check on live props was still slightly worse than market consensus in every market, with
   high-volume assists and points about 5–7% low; v2.3 improved shots but has no new live-prop
   comparison.
2. **Winner's curse.** Every sport ranks by the size of the model–market disagreement with no
   market blend, so the largest model errors are the ones selected.
3. **NFL injury handling biases toward Unders.** `apply_availability_adjustment` multiplies the
   mean by P(plays), for example 0.65 for Questionable. Props on a player who doesn't play are
   voided, so the forecast should be conditional on playing. Usage is never redistributed to
   teammates.
4. **NFL count spreads are too narrow.** Receptions σ = catch rate × σ(targets) ignores
   catch-level variance (floor 0.5). Counts and yards use Normal distributions.
5. **The NHL "team_total" label comes from the bet ticket.** `ticketData` in
   `briefing-picks.js` maps NHL `totals` to `team_total`, so those 6.5 and 7.5 rows are game
   totals labeled at entry. `clv.mjs` already prices them as game totals.
6. **`bets.timestamp` is never written.** The tracker insert (`docs/tracking/bet-tracking.js`)
   doesn't set it. `created_at` is not a safe stand-in: the Padres–Cubs total was entered about
   24 hours after its game.
7. **`edge_bps` is a probability gap at the vig-inclusive price, not EV.** Formula:
   (p − 1/decimal) × 10⁴. Phase 3 should store EV explicitly.
8. **Credit floor gaps.** `closing-lines` stops paid calls at 500 credits (`CLOSING_LINES_RESERVE`
   default), not 2,000. The NFL, NHL, MLB and NBA refresh scripts record the remaining quota but
   have no floor at all.
9. **`closing-lines` source isn't in the repository.** The deployed function refers to
   `tests/closing_lines.cjs` and `supabase/closing_lines_schedule.sql`, and the migration exists
   only in Supabase. None of these files are in `main` or any remote branch.
10. **Current news inputs.** NFL: weekly nflverse designations only (no inactives, snaps or
    routes). NHL: CBS injury listings for analyst context only; goalie and lineup are explicitly
    "unconfirmed". MLB: probable starters and published lineups; no weather, handedness or
    injuries. NBA: none.

### Proposed plan

Each step is a small PR with tests. Steps marked ✋ change pick output or model behavior and wait
for owner approval.

1. **Housekeeping (no model change).** Commit the deployed `closing-lines` source, migration and
   schedule. Raise the `closing-lines` reserve to 2,000. Add one shared credit floor (2,000) to
   every Odds API caller. Make the ticket send `timestamp` (placement time) and `totals` for NHL
   game totals.
2. ✋ **Stop the bleeding.** Remove the NFL calibration plateaus from pick eligibility: an offer
   whose calibrated probability sits on a flat segment is not priceable. Add a guard against
   alternate lines (require the offer line within a set distance of the consensus line, or
   require ≥ 2 books at that line). Apply the Phase 3 blend at w = 0.25 with a 3% EV threshold,
   using `consensus_prob` already in each feed. This is a Top Picks policy change, so it also
   updates `daily_process.html` and `ANALYST_RESEARCH.md` per `AGENTS.md`.
3. **Phase 1, NHL first** (it plays daily): NHL API starting goalies, lines, PP units and
   scratches, stored with as-of timestamps; then NFL (nflverse injuries plus inactives, snaps,
   routes, weather); NBA before Oct 20; MLB weather and park.
4. ✋ **Phase 2:** NHL ES/PP TOI split; NFL NB counts and simulated yards conditional on playing;
   NBA minutes × per-minute rates.
5. **Phase 4,** starting with no-cost data (`odds_snapshots` and archived NHL/MLB runs); then NFL
   receptions history after owner approval (about 2,700 credits).
6. **Phase 5** CLV report and tracker card.

### Open questions for the owner

1. NHL "team_total" rows: evidence says they are game totals. Relabel the three settled rows (and
   today's pending one) to `totals`, or leave history as is and fix only new tickets?
2. Step 2 changes what Top Picks publishes and will cut the card sharply in every sport (see the
   blend check below). Go ahead before the Phase 4 backtest, or shadow-run it alongside the current
   selector for a week first? The owner wants variety on the card: proposed one slot per sport,
   caps per sport and market, and an optional labeled "Leans" row that is not bet or tracked.
3. `bets.timestamp` for existing rows: backfill from `created_at` where it precedes
   `commence_time`, and leave the rest null?
4. The odds keys used by the GitHub workflows (`NFL_ODDS_API_KEY`, `MLB_ODDS_API_KEY`,
   `NHL_ODDS_API_KEY`, `NBA_ODDS_API_KEY`, `ODDS_API_KEY`): do they all draw on the same 15,700
   credits as the Vault key? The 2,000 floor has to be enforced against the shared balance.

### Follow-up checks (2026-10-05)

**Market blend by sport.** w = 0.25 and EV ≥ 3% at the best price, applied to each sport's latest
committed board. It thins every sport rather than leaving mostly NFL:

| Board | Raw model, EV ≥ 3% | Blended, EV ≥ 3% |
|---|---|---|
| NHL, Oct 4 afternoon run (110 priced outcomes) | 29 | 2 (moneyline, puck line) |
| MLB, Oct 5 morning run (38) | 13 | 4 (outs, run line, 2 totals) |
| NFL, Oct 5 pre-screened feed only (14) | 11 | 2 (receptions) |

**NHL shots on goal, every graded line.** 530 SOG lines from Sept 29 to Oct 3, all priced by
v2.1, against official results (`docs/markets/data/nhl.json`, last pregame snapshot):

| | Over hit rate |
|---|---|
| Actual | 44.9% |
| Market expected | 49.0% |
| Model expected | 42.1% |

Log loss: model 0.711, market 0.691, coin flip 0.693. Unders did hit a little more often than the
market expected in week 1, but on the 176 lines where the model was 10+ points below the market on
the Over, Overs hit 48.9% against a market 48.8% and a model 31.8%. The model's strongest Under
calls, which are the ones the selector picks, carried no information beyond the market. The
current v2.3 board (Oct 4 afternoon) still leans Under on SOG: on 80 lines it favors the Under by
more than 2 points on 48, the Over on 11, with an average gap of −3.3 points on the Over. This is
much smaller than v2.1's gap; v2.2's re-forecast of Sept 29 to Oct 2 put the median model mean at
0.958 of the market's, against 0.846 for v2.1.

More current-season games will mostly update ice time and roles (14-game half-life). They will not
remove the remaining lean: shrinkage toward a position-wide prior pulls the high-volume shooters
books post props on downward, and ranking by disagreement selects the largest errors. The fixes
are the market blend (Phase 3), each player's own prior instead of the position average, and the
even-strength/power-play ice-time split (Phase 2).

**NFL record is mostly luck.** Our six NFL bets went 5–1. By closing-market probabilities we
expected 2.4 wins (chance of 5+: about 4%); by the model's own probabilities, 3.1 (about 13%).
Three of the six had a calibrated probability of exactly 0.50, so the model was not claiming an
edge. The broader record agrees: the Week 3 review
(`reports/nfl-weekly/2026/week-3/review/summary.json`) graded the highest-EV offer per player,
game and market from the archived Top Picks. Result: 258 graded, 108–150, ROI −12.9%;
receptions 52–73, −12.6%. Model Brier 0.2554 against market 0.2512.

**NHL starting goalies, candidate sources (first list, before checking terms).**
| Source | What it has | Cost and catch |
|---|---|---|
| NHL.com daily "projected lineups" previews | Projected goalies, lines, scratches per game | Free and official; terms and structure to check |
| Daily Faceoff starting goalies | Confirmed / Likely / Unconfirmed with reporter | Free page; terms may forbid automated use |
| MySportsFeeds | NHL feeds, lineups likely | From $25/month commercial, but the non-live tier may not cover pregame lineups |
| SportsDataIO | "Starting Goaltenders by Date", projected and confirmed | Self-serve $99–149/month is next-day and personal use; real-time commercial is quoted, estimated $500–1,000+/month |
| RotoWire | Starting goalies and lineups | Custom quote through sales |
| NHL API | Who started each past game | Free; no pregame announcement |

Plan regardless of source: goalie game logs from the NHL API → save rate shrunk toward league
average and a start probability (workload, back-to-backs) → expected goals against for the
opponent. Confirmations then replace the start probability when a permitted source has them.

### Goalie source check results (2026-10-05)

Robots.txt and terms were read from a GitHub runner (one-off workflow, since removed).

| Source | Robots.txt | Terms | Verdict |
|---|---|---|---|
| NHL.com projected lineups | Allows `/news/` | Bans "unauthorized spidering, scraping, or harvesting"; content is for "non-commercial, informational, personal use" | Not usable automatically |
| RotoWire | Allows most paths | Bans automated access without written consent, and bans AI tools from reading or storing its data | Not usable; licensed feed by quote only. Its goalie page was fetched by the check but deliberately not read |
| Left Wing Lock | Allows all | Personal, non-commercial viewing only; names Starting Goalies as data it bans scraping | Not usable |
| Daily Faceoff | Allows `/starting-goalies` (disallows `/api/`, `/cms/`) | None on dailyfaceoff.com, but The Nation Network's Terms of Service (published at oilersnation.com/terms-of-service) cover all its brands and bar "any commercial use of ... any content, materials, or databases from our network" and copying or republishing without written permission (checked 2026-10-05) | Not usable automatically without permission or a license. Owner chose not to ask (2026-10-05). The page carries structured data (goalie IDs, Confirmed/Likely status, reporter) |
| NHL API (`api-web.nhle.com`) | n/a | Covered by the NHL terms above | Pregame it lists each team's goalies with season stats, not the starter; `right-rail` has a `scratches` field (empty at 9 AM). The site already relies on this API, so the NHL terms question applies to existing pipelines too |

Paid alternatives remain SportsDataIO (Daily Faceoff's data carries FantasyData/SportsDataIO player IDs) and
MySportsFeeds. The NHL-API-based start-probability model needs no new source.

### Step 1 status: done (2026-10-05)

Merged in pbwitt/fourth-and-value#86.


- Shared 2,000-credit floor and 2,000 per-run cap (`scripts/odds_budget.py`) wired into every scheduled
  Odds API caller. `closing-lines` v2 deployed after the merge: 2,000 reserve (a lower setting is
  ignored), cost-aware check, 2,000 per-run cap. Its 5-minute runs returned 200 from 14:05 to 15:00 UTC.
  The deployed `clv.mjs` writes the accent-stripping regex range as literal characters rather than
  `\u0300-\u036f` escapes; it matches the same characters.
- `closing-lines` source, migration and schedule committed from Supabase.
- Bet tickets record NHL game totals as `totals` and write `bets.timestamp` on save. Existing rows unchanged
  (open questions 1 and 3).
- Found while testing: a date-dependent NHL player-page test failed every Oct 5 Morning Picks NHL job.
  Fixed separately in pbwitt/fourth-and-value#85.

### Step 2 status: done (2026-10-05): market blend on Top Picks

Owner approved going ahead ("go ahead and finish"). Live on Top Picks, not a shadow run. Merged in
pbwitt/fourth-and-value#87.

- `docs/assets/briefing-picks.js`: every model candidate is blended,
  `f = sigmoid(0.25·logit(model) + 0.75·logit(market))`, with `market` the median no-vig
  probability at the exact line. It needs at least two books at that line (alternate lines with
  one book have no market) and `(1 − push)(f·decimal − 1) ≥ 3%` at its own price. NFL keeps the
  smaller of raw and calibrated model chance and withholds calibrated 50% estimates (the curve's
  flat centre). Card order uses the blended chance for every sport.
- Variety: each sport's best eligible bet is taken first, then at most 4 per sport and 2 per sport
  and market. Each sport's best positive-but-below-3% offer is listed as a lean (not a pick, not
  tracked) for an outcome not already picked.
- Tickets record `market_prob`, `final_prob`, `blend_weight`, `expected_value` (at the price taken)
  and `decision_at`; migration `bet_blend_fields` applied 2026-10-05.
- Editions now carry `morning-edition-2`; earlier editions keep `morning-edition-1`.
- Replay on the Oct 5, 8:36 AM ET feeds: 2 NFL picks (receptions overs at +150/+160 where the
  calibrated 54.8% is a plateau level) and 3 MLB picks (two totals, one run line); 44 NFL offers
  below 3% after the blend, 25 at a calibrated 50%, 23 with fewer than two books at the line.
  The NFL receptions picks show the 0.25 weight is generous for NFL given its Week 3 record; Phase 4
  should fit a lower NFL weight if the out-of-sample scores agree.
- Not done: the Pinnacle (`eu` region) sharp reference, which costs extra credits per request.

### Step 2b (2026-10-06): confidence tiers so the card is not empty

The first `morning-edition-2` card (Oct 6) published no picks. MLB's feed had six model picks at
7 a.m. (850 hitter props were waiting on batting orders) and all six blended below 3%; the
best was the Dodgers −1.5 at +174, +1.3%. Every one of the 32 NHL candidates was withheld as a
thin market, because the code counted MLB/NHL `other_books` (which excludes the offered book)
against the two-book minimum while NFL's `book_count` includes it. MLB/NHL therefore needed three
books, not the two this plan specifies; most NHL player props are posted by two.

Owner asked for more picks at a sensible level. Changes (edition policy `morning-edition-3`):

- Two books at the exact line, counting the offered one, for every sport (MLB/NHL
  `other_books ≥ 1`). An MLB/NHL consensus can now rest on one other book.
- A pick needs blended EV ≥ 1%. ≥ 3% is labeled **high confidence**, 1–3% **moderate
  confidence**; high ranks before moderate on the card, after analyst selection and forecast
  reliability. Leans are the best positive offer below 1%. The weight stays 0.25.
- Caps unchanged: 10 per card, 4 per sport, 2 per sport and market.

Replays with the same selector (before qualitative review and card caps):

| | 3%, MLB/NHL 3 books (Oct 5 rule) | 1%, 2 books (this step) |
|---|---|---|
| MLB, Sept 28 to Oct 6 morning feeds (9 mornings, 6 with model picks) | 11 | 28 (11 high, 17 moderate) |
| Card rows published Sept 29 to Oct 5 (143) | 16 | 57 (MLB 29, NFL 23, NHL 5) |
| Oct 6 morning feeds | 0 | 5 (MLB 1 moderate; NHL 3 high, 1 moderate) |

MLB floors on the same nine mornings: 3% → 11, 2% → 21, 1.5% → 26, 1% → 28, 0.5% → 34,
above 0 → 39. Most of the gain comes by 1.5%; below 1% adds offers the blend prices near break-even.
Three of the four Oct 6 NHL additions are shots-on-goal unders priced against one other book, the market
where week 1 showed the model most overconfident, so track moderate and NHL picks separately.
The tier is derivable from `expected_value` on each ticket; Phase 4 should fit the floors with
the weight.

### Phase 1, NHL goalies: projected starters as context (2026-10-05)

- `scripts/nhl/v2/goalies.py` reads the official per-game goalie report (`api.nhle.com` stats
  `goalie/summary`, the source family the skater history already uses; 2,768 rows for 2025-26).
  Last season is collected once, the current season at most every 12 hours, checksummed in the
  cached history.
- For each game in the next 48 hours: each team's start chances from recency-weighted starts over
  its last 20 games (10-game half-life), last night's starter cut to 35% of his share on a
  back-to-back. A goalie whose latest box score is for another team is dropped, and the list is
  limited to the official current roster (`api-web.nhle.com` `roster/{team}/current`) so offseason
  moves don't leak in. Save rate is shrunk toward .903 with 1,000 shots of prior weight.
- Live dry run on a GitHub runner (Oct 5, 11:46 AM ET, nothing published): 13 games in the next
  48 hours, row text 135–205 characters, e.g. "Boston Bruins: likely Jeremy Swayman 88% (sv 0.906),
  Michael DiPietro 12%". All 25 roster requests answered 200 in a separate check; an earlier
  burst run got 19, which led to spacing and one retry on 429/5xx.
- Point in time: only box scores available by the decision (next day 12:00 UTC, the history's
  convention) count, so the 7:05 AM run does not yet see last night's starter; the 4:30 PM
  refresh does.
- Output: `goalie_assumption` on every row in that window ("Projected, not confirmed", the
  opposing goalie for a skater, both for a game line) and `goalie_projections` with `as_of` on the
  snapshot, archived with each run. **Context only**: no probability, price or eligibility changes.
  A fetch failure leaves the generic text and records `goalie_error`.
- Next: test whether the opposing goalie's expected save rate improves goals/points/SOG forecasts
  out of sample (Phase 4) before it enters the model. Confirmed starters still need a permitted
  source: Daily Faceoff is ruled out by The Nation Network's terms unless it grants permission, so a paid
  feed (SportsDataIO, MySportsFeeds) or a manual confirmation is the remaining route.

### Data fixes (2026-10-05)

- The four NHL `team_total` rows (three settled, one pending) were relabeled `totals`.
- `bets.timestamp` was backfilled from `created_at` for the 30 rows saved before their game
  started; two rows saved after the start stay empty. Captured closes were unchanged
  (average CLV still −5.30% over 24 settled bets).
- Shared odds balance: unverified. The NHL feed reported 16,033 credits left at 7:21 PM ET Oct 4 and
  the MLB feed 15,655 at 8:06 AM ET Oct 5, consistent with one shared balance; the floor applies per
  key either way.

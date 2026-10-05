# Model improvement plan

Owner: pbwitt. Started 2026-10-05. Update this file as each phase finishes.

## Status

| Phase | State | Notes |
|---|---|---|
| 0. Audit | Report delivered 2026-10-05; waiting on owner sign-off | Findings and proposed plan below. No model code changed. |
| 1. News and lineup data | Not started | |
| 2. Model fixes | Not started | |
| 3. Market blend before betting | Not started | |
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
2. Step 2 changes what Top Picks publishes and will likely cut the card sharply, mostly NFL
   plus-money props. Go ahead before the Phase 4 backtest, or shadow-run it alongside the current
   selector for a week first?
3. `bets.timestamp` for existing rows: backfill from `created_at` where it precedes
   `commence_time`, and leave the rest null?
4. The odds keys used by the GitHub workflows (`NFL_ODDS_API_KEY`, `MLB_ODDS_API_KEY`,
   `NHL_ODDS_API_KEY`, `NBA_ODDS_API_KEY`, `ODDS_API_KEY`): do they all draw on the same 15,700
   credits as the Vault key? The 2,000 floor has to be enforced against the shared balance.

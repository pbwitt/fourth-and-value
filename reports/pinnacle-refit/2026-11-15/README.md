# Pinnacle vs the US-book median, and the model-weight refit (November 15, 2026)

Step 2d of `docs/model-improvement-plan.md`, the follow-up that was due November 11. It set
policy `morning-edition-8`: Pinnacle enters the market term while its morning snapshot is fresh,
and MLB player props move to the 0.10 model weight. Research only until that policy's editions are
graded; earlier editions are not regraded as evidence for it.

Reproduce with `python scripts/pinnacle_refit.py --stats <nflverse weekly stats 2026> --out
reports/pinnacle-refit/2026-11-15` (the script's docstring covers the git history it needs).
`lines.csv` has every scored line; `results.json` has every figure below.

## What was compared

For each line, Pinnacle's no-vig chance from the morning snapshot (about 7:10 a.m. ET) is paired
with the US-book median from the same morning's published feed, recovered from git history, and
with the last feed published before the game (8–10 hours later on most lines).

- **NFL props**: 855 lines in 55 games, Week 6 to Week 10 Thursday; the model's side as listed on
  the Top Picks board (`docs/props/top-picks.json`), graded with nflverse weekly stats. The grading
  matched all 247 overlapping weekly-review tickets. The Week 6 Sunday lines use the Saturday
  snapshot because the credit reserve refused snapshots from October 18 to 23.
- **NHL and NBA**: every line the recap boards graded and Pinnacle priced, one side per line
  (Over, or the home team): NHL 1,430 props and 361 game lines; NBA 1,354 props and 168 game lines.

Check: the November 15 card's Javonte Williams under 2.5 receptions comes out exactly as published
(Pinnacle 45.3%, US median 42.8%, model 54.1%, +136 at DraftKings).

## Results

| Set | Lines | Log loss Pinnacle / US | Pinnacle better in | US moved toward Pinnacle (95%) |
|---|---|---|---|---|
| NFL props | 855 | 0.6791 / 0.6787 | 36% of resamples | 0.18 (0.09–0.28) |
| NHL props | 1,430 | 0.6779 / 0.6783 | 63% | 0.20 (0.15–0.25) |
| NBA props | 1,354 | 0.6916 / 0.6912 | 37% | 0.24 (0.15–0.33) |
| NHL game lines | 361 | 0.6444 / 0.6449 | 77% | 0.54 (0.29–0.81) |
| NBA game lines | 168 | 0.5730 / 0.5729 | 47% | 0.27 (−0.27–0.79) |

**On results, a tie.** No difference is distinguishable from zero. If Pinnacle were exactly right,
it would beat the US median by only about 0.0005 log loss on NFL props (0.0010 NHL props, 0.0004
NBA props); the standard error is about twice that. Separating them on outcomes would take roughly
15,000 graded NFL lines.

**On line movement, Pinnacle leads.** The last column is the share of the morning gap the US
median closed by the last pregame refresh, fitted with the model's own gap as a control. The
model's gap predicts no NFL movement (0.003), so the result is not the US median correcting its
own noise; NHL and NBA models predict a little (0.08 and 0.07). Pinnacle's morning price carries
information the US books take in later, about a fifth of the gap on props and about half on NHL
game lines.

**Betting Pinnacle's own EV does not work (NFL, best US price).**

| Lines | Count | Expected by Pinnacle | Realized (95%) |
|---|---|---|---|
| All board lines | 855 | −3.4% | −7.0% (−14.7% to +0.4%) |
| Pinnacle EV > 0 | 150 | +2.0% | −7.7% (−25.5% to +10.3%) |
| Pinnacle EV > 2% | 61 | +3.4% | −19.4% (−47.2% to +10.2%) |
| Blend ≥ 1% (US median) | 82 | −1.4% | −3.0% (−25.9% to +20.3%) |
| Blend ≥ 1% and Pinnacle EV > 0 | 33 | +2.1% | −1.0% (−41.0% to +39.7%) |

Small samples, wide intervals, but nothing supports treating Pinnacle as the truth.

## Decision: market term

`market = sigmoid(s·logit(Pinnacle) + (1−s)·logit(US median))`, with s = **0.20** for player props
and **0.50** for game lines, where the same-day snapshot is under three hours old and posts the exact
line; otherwise the US median alone. The shares are rounded movement estimates, below the 50/50
props split proposed in Step 2c. Afternoon editions, price checks and board sorting use the US
median alone. Tickets and closing-line value keep the US median as `market_prob`.

On the November 15 Williams under: market 43.3% instead of 42.8%, blend 44.4% instead of 43.9%,
expected value +4.7% instead of +3.6% at +136.

## Decision: model weights

Refit on lines graded after October 12 (out of sample for the October 14 fit), game-clustered 90%
ranges:

| Lines | Count | Games | Log loss market / w 0.10 / model | Best w | 90% range |
|---|---|---|---|---|---|
| NHL | 12,895 | 197 | 0.6423 / 0.6424 / 0.6491 | 0.00 | 0.00–0.16 |
| NFL props (weeks 6–9 tickets) | 458 | 44 | 0.6862 / 0.6871 / 0.7199 | 0.00 | 0.00–0.22 |
| MLB | 554 | 11 | 0.6456 / 0.6467 / 0.6656 | 0.00 | 0.00–0.00 |

On the 855 NFL lines above, the 10% blend scored 0.6796 against the US median's 0.6787. The model
has not yet shown value beyond the market in any sport. MLB player props move from 0.25 to 0.10 (no
effect until the 2027 season); NHL, NFL and NBA keep 0.10, inside every range. A zero weight would
publish no model picks; that is the owner's decision and was not made here.

## Cost and coverage

The snapshot cost 62 credits per morning on average (median 41, most 141) across 26 paid mornings.
Ten named bookmakers billed as one region on every probe, so Pinnacle could later join the regular
US pulls at no extra cost. Pinnacle posts main lines only; across the snapshots it matched 24% of our
NFL prop outcomes, 10% NHL and 4% NBA (our feeds carry many alternate lines), 46–63% of NFL and NHL
game-line outcomes and 35% of NBA's.

## Limits

NFL lines are the model's side on the Top Picks board only, so they lean toward lines the model
likes. NHL and NBA use the recap boards' grading. The movement test measures where US prices went,
not who was right; it is the better-powered test here, not a proof. Refit when another month of
snapshots is graded.

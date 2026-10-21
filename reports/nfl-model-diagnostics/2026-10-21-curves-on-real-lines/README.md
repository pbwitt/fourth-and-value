# Skewed curves on real NFL lines, and where the picks' return comes from — October 21, 2026

Follow-up to [the curve-shape audit](../2026-10-21-shape-and-level/), which found on 2024–25 box
scores, with lines placed at multiples of a simple forecast, that a gamma curve fits receiving
yards and a negative binomial fits receptions better than the Normal. Before switching production,
this report tests the curves where it matters: on the lines books actually posted for 2026 Weeks 1–6
(93 games), using the shipped means from [the pass-volume change](../2026-10-21-pass-volume-recency/).

**Decision: the curve switch is not adopted.** On real lines the skewed curves did not price better
than the Normal for receiving yards or receptions, and every raw curve lost clearly to the market.
The published probabilities stay as they are. No model, probability or pick changes in this report.

## 1. Curves on real lines (`curves.py`)

One row per game, player, market and half-point line with 2+ books, Over side, at the last board
before kickoff; 1,014 receptions and 1,015 receiving-yard main lines. "Predictive" uses a spread
fitted on 2024–25 forecast errors (`var = a·m + b·m²`); it is close to the model's own sigma, so the
spread is not the problem.

Log loss on main lines (lower is better; 0.693 is a coin flip):

| Market | Market no-vig | Published (calibrated) | Normal | Skewed |
|---|---|---|---|---|
| Receptions | 0.688 | 0.693 | 0.736 | 0.745 |
| Receiving yards | 0.693 | 0.693 | 0.740 | 0.759 |
| Rushing yards | 0.694 | 0.698 | 0.756 | 0.737 |

(Normal and skewed with the predictive spread; the model's own sigma gives the same picture.)

- Every raw curve is worse than the market at the market's own line, by 0.04–0.07, with intervals
  well clear of zero. At that line the market prices about 50%; our mean is noisier than the
  market's information, so any raw probability away from 50% loses. The calibration curve, which
  flattens receiving yards to about 50% everywhere, is doing the right thing.
- The skewed curve helps rushing yards and hurts receiving yards and receptions. Its mean Over
  chance runs about 5 points below the 50% actual rate, the mirror image of the Normal's Over lean.
- Lines that only some books post (`other books' lines` in `results/curves.txt`) tell the same story.

## 2. Where the candidate pool's return comes from (`picks.py`)

Offers are the best price per game, player, market, side and half-point line with 2+ books
(9,058 offers). This is the candidate pool under the Top Picks gate, not the published card.

| Rule | Bets | Won | ROI [95%, resampling games] |
|---|---|---|---|
| Blend gate (production) | 144 | 52.8% | +22.6% [+1.7%, +42.7%] |
| Price edge only, model ignored | 89 | 53.9% | +13.5% [−10.3%, +37.3%] |
| Model at or above market, no price edge | 5,097 | 48.8% | −4.8% [−7.7%, −1.9%] |

Most of the blend gate's bets and return are receptions (115 bets, +30%). Holding the market price
fixed, the model's gap from the market does not predict results (logistic coefficient on the gap:
receptions +0.10, standard error 0.31; other markets −0.04, standard error 0.18). The receptions
return comes from prices instead: **receptions Overs that the market priced under 40%** (average line
1.96, so Over 1.5 or 2.5 at plus money) won 52.0% against a no-vig 37.7%, an ROI of +33.6%
[+6.7%, +59.5%] across 102 offers, 62 games and 77 players. The matching Unders, priced over 60%,
won 48.0%. The calibrated receptions probability stays near 50%, so the blend gate picks up exactly
those underdog Overs.

## What this means

- Improving the curve's shape would barely change published probabilities, because calibration
  flattens them, and on real lines it did not price better anyway.
- The model's probability carries no detectable information beyond the market so far this season;
  the picks' return comes from the price. Improvements are more likely on that side: a sharper fair
  price (book weighting, starting with the Pinnacle refit due October 28) and finding systematic
  mispricing like the low receptions Overs.
- **The receptions pattern is a hypothesis, not an established edge.** It was found by looking, in
  six weeks of one season, and offers in one game are correlated. It should be tracked prospectively
  under a rule fixed now: receptions Overs at half-point lines of 2.5 or lower, with 2+ books and a
  no-vig chance under 40%, graded weekly from Week 7.
- Calibration should be refitted on the new means once enough graded weeks exist; it still uses
  the 2025 fit.

## Limitations

- Six weeks of one season; each player-game contributes several correlated lines.
- Means are the shipped settings replayed for 2026; published probabilities came from the old means.
- The candidate pool ignores the card's per-game caps and analyst review, so it is not the scorecard.

## Reproduce

Inputs: `boards.csv` from `../2026-10-21-shape-and-level/extract_boards.py <repo> boards.csv 0cc78dcb`,
`graded.csv` from `../2026-10-21-shape-and-level/production_mean_test.py boards.csv <nflverse> --max-week 6 --out graded.csv`,
and `replay.csv` from `../2026-10-21-pass-volume-recency/replay_means.py <data> <repo> replay.csv starts_recent17_hl6`.

```
python curves.py boards.csv graded.csv replay.csv
python picks.py boards.csv graded.csv
```

Saved output: `results/`.

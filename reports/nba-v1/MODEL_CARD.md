# NBA v1 model card

Purpose: independent NBA game and player forecasts for research, published in **shadow
mode**. Rows carry model fields, but the feed never sets `model_checked_at`, so NBA stays
ineligible for editorial articles and Top Picks until graded 2026–27 prices support it
(`NBA_LAUNCH_PLAN.md`, Phase 3). The site shows the probability and projection with an
"experimental" label and does not show an expected value. The engine selects no picks and
writes no narrative.

Code: `scripts/nba/model.py` (states, distributions), `scripts/nba/train.py` (selection,
evaluation, fitting), `scripts/nba/inference.py` (daily forecasts). Parameters:
`artifacts/nba/model/nba-v1.json` (fit on every regular season through 2025–26; daily
runs never retrain). Metrics: [EVALUATION.md](EVALUATION.md), `evaluation.json`.

## Data

ESPN box scores, 2023–24 through 2025–26 (`artifacts/nba/history/`, manifest with
sha256). Model states use regular-season and NBA Cup final games only. Playoff and play-in
games total 12–17 fewer points than the regular season and would otherwise be the last
games every team carries into October. Training targets are regular-season games; for
team features, training starts December 1, 2023, because earlier games have no
prior-season state.

## Game model

- **State:** per team, exponentially weighted sums by game (selected half-life 10 games) of
  possessions per 48 minutes (FGA − OREB + TOV + 0.44·FTA, averaged across both teams),
  points scored and allowed per possession, and opponent box-score totals per possession.
  A new season multiplies last season's weights by the selected carry (0.25). Ratings
  shrink toward lagged league averages with three equivalent games.
- **Expectation:** pace = home pace × away pace ÷ league pace; points per possession =
  own offense × opponent defense ÷ league. Two ridge regressions map those expected
  regulation points, rest days (capped at 4), back-to-backs and a neutral-site flag to
  regulation points for each team. The intercepts carry home court.
- **Distribution:** a bivariate normal on integer regulation scores, with the residual
  covariance from training. Tied regulation mass plays empirical overtime periods
  (observed five-minute score pairs, up to four overtimes), so no final is tied.
  Moneyline, spread and total all come from one score matrix, with push mass on whole
  lines.
- **Not modeled:** travel distance, altitude, injuries or lineups, in-season trades
  beyond what results show, and rest decisions announced on game day.

## Player model

- **Minutes:** ridge regression on recency-weighted minutes (5- and 20-appearance
  half-lives), last game, recent start rate, experience, a team change, a layoff of more
  than 10 days, back-to-back, and blowout risk (the game model's expected margin ×
  minutes share). The residual SD comes from training, by 4-minute bins of the
  projection. Minutes follow a normal distribution truncated to (0.5, 48].
- **Rates:** per-minute PTS, REB, AST, 3PM, BLK, STL and TOV, with a 20-appearance
  half-life, shrunk to the position-group prior (G/F/C) with 100 prior minutes.
  Adjustments: game pace ÷ own team pace, and the opponent's allowed rate ÷ league
  (weight 1.0, selected). A training-fit linear blend then combines this structural mean
  with the player's last-10 and last-30 averages for each stat.
- **Counts:** gamma-Poisson given minutes. Each stat's gamma shape is fit by maximum
  likelihood on the full predictive mixture, so minutes uncertainty is included.
- **Combos (PRA, P+R, P+A, R+A):** simulated jointly, 20,000 seeded draws per player-game.
  Each draw has one shared minutes value and correlated gamma intensities (a Gaussian
  copula, with correlations implied by training residual covariances). Combos are never
  built from independent marginals. Turnovers fit as nearly Poisson (shape about 2,000),
  so their copula correlations have no practical effect.
- **Calibration:** one logit slope per market, fit on 2024–25 predictions from a model
  trained on 2023–24 only, at pseudo-lines. Points and points combos shrink toward 50%
  (slopes about 0.75–0.85); the other stats stay near 1. It applies to the over/under
  decision, with push mass left as simulated.
- **Scope:** forecasts are conditional on the player appearing, because books void
  props for DNPs. A player needs three NBA appearances in the history. Players listed
  Out on ESPN's injury report are withheld. Other statuses (questionable, day-to-day)
  are shown but not adjusted for. Teammate absences do not change minutes or usage in
  v1.

## Identity

Odds API names match ESPN current rosters, fetched for teams with props. Accents,
punctuation and suffixes (Jr., III) are normalized. If a roster fetch fails, a unique
history name whose last team is in the game is used. Ambiguous or unmatched players get
no forecast, and the reason is published (`model_unsupported_reason`).

## Known limitations

- **Points tails:** points and points combos are still somewhat overconfident at the
  tails on the test season after calibration. See the calibration lines in
  EVALUATION.md. They are most overconfident where the model disagrees sharply with a
  player's recent form, which is where sportsbook lines tend to sit.
- **Opening weeks:** opening-week team states come from last season, shrunk heavily.
  The market knows about offseason roster moves and the model does not, so large early
  disagreements with the market are expected and are not evidence of value.
- **Pseudo-lines:** they show discrimination and calibration, not profitability. Price
  claims wait for archived 2026–27 odds graded in Market Analytics.

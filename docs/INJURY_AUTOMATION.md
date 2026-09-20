# Injury automation

Fourth & Value now consumes the weekly nflverse injury report before building
player parameters. The fetcher stores the raw report at
`data/injuries/injuries_<season>.csv` and a target-week normalized snapshot at
`data/injuries/injuries_week<week>.csv`, with a JSON audit beside it.

The normalizer keeps the player, team, position, report status, practice status,
availability estimate, and UTC retrieval time. Status estimates are deliberately
conservative:

| Report evidence | Availability input |
| --- | ---: |
| Out / IR | 0.00 |
| Doubtful | 0.15 |
| Questionable | 0.65 |
| Probable | 0.90 |
| No report status, limited practice | 0.95 |
| No report status, did not practice | 0.85 |

The weekly props builder multiplies the player’s distribution mean or Poisson
rate by that availability input and writes `injury_status` and
`injury_availability` into the parameter snapshot. Missing or delayed injury
data is a no-op. The manual `data/player_adjustments.json` file remains an
emergency override, but it is not the normal source of injury information.

The totals screen is produced by `scripts/build_injury_totals_signal.py`. It
uses conservative position priors, caps the total impact, and compares the
injury-adjusted projection with captured total-line movement. It reports
`possible_downward_overreaction` when the market has moved down materially more
than the model’s estimated injury impact, and `possible_underreaction` when the
market has not moved as much. If an opening or prior quote is missing, the signal
is explicitly `insufficient_market_history` rather than a bet recommendation.

These position effects are starting priors, not validated coefficients. Before
promoting a totals reaction to Top Picks, capture line snapshots before and
after injury announcements, join them to final active/inactive status, and fit
the coefficients using walk-forward data. The signal should also be evaluated
against closing totals and actual game totals, with separate checks for
quarterbacks, skill players, offensive linemen, and defensive players.

The source is nflverse’s weekly injury-report release, which documents the data
as API-collected weekly reports. The NFL injury page remains the confirmation
source for the latest designation. Neither source should be treated as a
guarantee of a player’s final snap count.

# Prediction markets: observation and paper execution

The first integration covers Kalshi NFL game-winner contracts, configured in
`config/prediction_markets.json`. It is an observation board plus an owner-only
local paper ledger. No credentials are needed for observation or paper execution.
Phase 2 adds a separate [private demo client](KALSHI_DEMO.md) for authenticated
mock-funds order rehearsal, disabled by default. There is no production order
endpoint or real-money enable switch. The public site never reads private ledgers.

## Observe and preview

```sh
python scripts/refresh_prediction_markets.py
python scripts/build_prediction_markets.py
python -m http.server 8010 --bind 127.0.0.1 --directory docs
```

Visit `/prediction-markets/`. Markets > Prediction Markets links to the board.
`--max-markets 6` is useful for a bounded smoke check. The checked-in config caps
observations at 40 contracts and discovery at ten 100-result pages per series;
the client caps all requests at 250 and does not retry failures. Observations
are sorted by market close for sampling; close/expiration is never called kickoff.
The sample can contain in-play markets because the exchange close time is not a
verified game start. This is stated on the board and paper intents separately
require a verified future start.

The optional **Prediction Market Snapshot** Actions workflow is manual and only
runs on `main`. It fetches the data, archives the observation for 30 days, commits
the public snapshot (including explicit failure status), then explicitly requests
a GitHub Pages rebuild. Dispatch time is not publication time. There is no
scheduled refresh or always-on worker yet. A browser rereads the saved feed every
five minutes and marks observations older than 15 minutes historical. It retains
an explicitly historical copy if a later fetch fails.

Public data is in `docs/prediction-markets/snapshot.json`. Private observation
archives and paper ledgers live under ignored `data/prediction-markets/`.
Generated JSON writes are atomic. A failed observation never relabels old prices
as freshly retrieved. Contract-level retrieval failures, missing fees and bounded
discovery remain explicit; returned-contract counts are not research coverage.

## Pricing and comparison

`scripts/prediction_markets/pricing.py` supplies calculations to both public
estimates and paper execution. The browser selects precomputed 1/10/100-contract
estimates and does not maintain a second pricing implementation.

* Preserve fixed-point prices and hundredths of contracts with Decimal.
* Derive YES asks from `1 - NO bid`, and NO asks from `1 - YES bid`.
* Sweep actual observed depth; no fill beyond the available quantity or limit.
* Estimate taker fees from the current series fee type/multiplier and latest
  effective event override. Future event changes expire the fee estimate.
  Unsupported fee types, missing multiplier or failed override history withhold
  total cost and equivalent odds. No default zero-fee assumption is allowed.
* Round the total debit up to cents at each price level. This is deliberately a
  conservative estimate: exchange direct-member precision, within-level fills,
  rounding accumulators and rebates can differ. The demo client captures actual
  fees from its fills; public and paper estimates are not actual charges.
* Equivalent American odds describe a $1 terminal payout, not a probability.
  Intermediate/fair-price settlements and ties follow the preserved contract
  rules. An NFL tie paying $0.50 is not a refund of the entry price.

Source rules, metadata-update time and order-book observation time remain separate.
The rule fingerprint also covers contract identity, payout size and strike fields.
No match is inferred from a team-name similarity. A manually reviewed `mappings`
entry must contain exact `ticker`, contract `side`, `rules_hash`, `verified_at`,
`evidence_url`, `settlement_verified: true`, `settlement_profile`, and a
`sportsbook_outcome` with `sport`, `game_id`, `player`, `market`, `side`, `line`,
`commence_time`. The source book row must independently have the same verified
settlement profile, exact outcome fields and a fresh timestamp. Mappings expire
after seven days; rule changes invalidate them immediately. The current NFL quote
feed does not establish equivalent settlement, so it cannot produce a verified
comparison simply by matching a team. The initial map is intentionally empty.

No exchange price supplies a missing model probability. These observations do not
enter the shared Top Picks selector, change Market Watch, or reclassify historical
research as support for a new selection strategy.

## Private paper orders

Create an ignored JSON file under `data/prediction-markets/` containing an array
of intents. Replace the placeholders with a real exact contract, the rule hash
from its snapshot, and verified future timestamps. This is a schema example,
not a suggested wager or forecast:

```json
[
  {
    "id": "owner-paper-test-001",
    "mode": "paper",
    "ticker": "EXACT-KALSHI-TICKER",
    "side": "yes",
    "contracts": "1",
    "limit_price_dollars": "0.50",
    "rules_hash": "EXACT-RULE-HASH-FROM-SNAPSHOT",
    "commence_time": "VERIFIED-FUTURE-GAME-START-WITH-TIMEZONE",
    "start_time_source": "URL-OF-GAME-SCHEDULE",
    "expires_at": "INTENT-EXPIRY-WITH-TIMEZONE"
  }
]
```

```sh
python scripts/paper_prediction_markets.py run --intents data/prediction-markets/intents.json
python scripts/paper_prediction_markets.py settle
```

`run` retrieves only the requested configured-series contracts before simulating,
so unrelated discovery cannot age their quotes. `--snapshot PATH` allows
offline replay but still enforces the 30-second quote, snapshot and fee age gates.
Each intent is one immediate-or-cancel attempt; unfilled quantity is canceled,
never assumed to rest. Repeating the same ID returns its recorded outcome;
changing its contents requires a new ID. Already consumed liquidity cannot be
filled twice within one snapshot. Across different observations, liquidity can
reappear: paper fills are an optimistic observation-based simulation, not proof
of executable returns. There is no queue, latency or adverse-selection model.

Paper-only limits are $10 per order, $100 gross daily purchases (Eastern date),
$25 open cost per event and $100 total open cost. Settlement releases open
exposure but does not reset gross daily spend. These are simulated-dollar test
limits, not authorization to spend real money or stake recommendations. A file
lock covers check-and-write; ledger files have owner-only permissions. Ledgers
cannot be written into this worktree's public `docs/` tree.

No forecast is required to rehearse execution. Expected profit stays null unless
the operator explicitly supplies an optional `forecast` with matching `rules_hash`,
`side`, a `source`, fresh `at`, `includes_partial_settlements: true`, and
`expected_payout_per_contract` in [0,1]. This is recorded input, not an approved
Top Pick or a probability produced by the integration. `settle` uses only explicit
finalized market settlement values, including intermediate payouts; a missing
value never becomes a zero or a win inferred from a title.

## Before real trading

The demo CLI supports order lifecycle, actual fills/fees, timeout reconciliation
and cancellation. An account-level smoke test still needs a Kalshi demo account
and locally held credentials; fixture tests alone are not that validation.
Live activation additionally needs the owner's account access,
explicit capital/exposure limits, a validated strategy and deployment choice.
This foundation does not implement live trading or multi-user account custody.
Existing sportsbook Bet Tracker records remain separate from paper simulations.

## Verification and sources

```sh
python -m unittest discover -s tests -p 'test_prediction_markets.py'
node tests/prediction_markets_browser.cjs
python scripts/build_prediction_markets.py --check
python scripts/build_daily_process.py --check
python scripts/seo_check.py --changed
```

PR tests use fixtures only; they never call Kalshi or an account API.

Official API references checked October 4, 2026:

* https://docs.kalshi.com/getting_started/orderbook_responses
* https://docs.kalshi.com/api-reference/market/get-series
* https://docs.kalshi.com/api-reference/events/get-event-fee-changes
* https://docs.kalshi.com/getting_started/fee_rounding
* https://kalshi.com/docs/kalshi-fee-schedule.pdf
* https://docs.kalshi.com/getting_started/demo_env

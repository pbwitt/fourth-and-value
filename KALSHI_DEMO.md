# Kalshi demo order rehearsal

Phase 2 provides an owner-operated CLI for authenticated **mock-funds** orders.
It is separate from the public observation board, paper simulator, Bet Tracker
and Top Picks. No credentials, journal or order submission is exposed in the
browser. There is no scheduled trader, automatic strategy selection or live
activation setting. Fixture checks validate code paths; an authenticated demo
smoke test still needs the owner's demo account and a suitable open demo market.

## Account setup

Create a [Kalshi demo account](https://demo.kalshi.co/) and a demo API key. Demo
and production accounts/keys are separate. Keep the private key outside this
repository, with owner-only permissions (`chmod 600`). Set these variables in
your local shell or secret manager, never in a committed file or chat:

* `KALSHI_DEMO_API_KEY_ID`: demo API key identifier.
* `KALSHI_DEMO_PRIVATE_KEY_PATH`: absolute path to the private PEM file.

The client supports RSA-PSS/SHA-256 and Ed25519. It has a fixed
`https://external-api.demo.kalshi.co` origin, rejects redirects and unsupported
endpoints, and has no production credential fallback. HTTP failures expose only
the status code, never response bodies, headers or key material. The initial
account scope is subaccount zero. Keep one durable journal for this integration;
do not switch journals to bypass an outstanding reservation or spending limit.
Rotating the API key requires explicit journal migration because its fingerprint
binds the journal to the key that created the orders.

```sh
python3 -m venv .venv-kalshi-demo
.venv-kalshi-demo/bin/python -m pip install -r requirements-kalshi-demo.txt
.venv-kalshi-demo/bin/python scripts/kalshi_demo.py inspect --ticker EXACT-DEMO-TICKER
```

Only configured series (initially `KXNFLGAME`) are accepted. Demo listings and
liquidity can differ from production; do not reuse a production snapshot or
broaden the allowlist just to force a test. `inspect` obtains the exact demo
rules/hash, price ticks, current event fee overrides and order book. For submission,
all observations must still be within 30 seconds after account-balance retrieval.
An imminent fee change, missing fee metadata or missing tick grid blocks orders.

## Submit one reviewed test intent

Copy `config/kalshi_demo.json` to an ignored local file, such as
`data/prediction-markets/demo-config.json`, and change its `enabled` to `true`.
The committed default remains disabled. The limits are mock dollars: $2 per
order, $20 gross purchases per Eastern day, $5 open cost per event and $10 total
open cost. These are execution test limits, not stake recommendations.

Save one JSON object to `data/prediction-markets/demo-intent.json`:

```json
{
  "id": "owner-demo-001",
  "mode": "demo",
  "ticker": "EXACT-DEMO-TICKER",
  "side": "yes",
  "contracts": "1.00",
  "limit_price_dollars": "0.50",
  "rules_hash": "REVIEWED-HASH-FROM-DEMO-INSPECT",
  "commence_time": "VERIFIED-FUTURE-GAME-START-WITH-TIMEZONE",
  "start_time_source": "URL-OF-GAME-SCHEDULE",
  "expires_at": "SHORT-FUTURE-INTENT-EXPIRY-WITH-TIMEZONE"
}
```

This is a schema example, not a selected contract. Market close is not game start.
The operator supplies a verified future start and reviews the exact rules. A
model forecast is not inferred from the exchange price.

```sh
.venv-kalshi-demo/bin/python scripts/kalshi_demo.py --config data/prediction-markets/demo-config.json submit --intent data/prediction-markets/demo-intent.json
.venv-kalshi-demo/bin/python scripts/kalshi_demo.py status
.venv-kalshi-demo/bin/python scripts/kalshi_demo.py reconcile
.venv-kalshi-demo/bin/python scripts/kalshi_demo.py cancel --id owner-demo-001
```

All commands use the same default private journal, under ignored
`data/prediction-markets/demo-journal.json`. `--journal` can choose a persistent
path outside the repository. Keep it backed up; deleting it loses exposure and
daily spending history. Cancellation/reconciliation still work when submissions
are disabled. Exit status 0 means inspection or fully reconciled state; 1 means
an outstanding/uncertain order; 2 means the operation stopped with an error.
`status` is a local journal view; `reconcile` reads current exchange state.

## Execution and accounting

The execution order is: lock/read journal; reject reused IDs with changed
instructions or any unresolved earlier order; read demo metadata, fees and book;
validate rules, verified start, intent expiry and limit ticks; reserve the full
requested quantity against spending/exposure limits; check available demo balance;
recheck freshness and limits; persist/fsync the reservation; recheck freshness;
POST once; persist the acknowledgement; retrieve the exchange order and all fill
pages; save the reconciled result. Concurrent processes share an exclusive file
lock. Journals and lock files have owner-only permissions.

Orders use the V2 event endpoint with immediate-or-cancel, fixed-point quantity
and price, self-trade prevention and cancellation on exchange pause. A YES buy
is `bid` at its YES limit; a NO buy is `ask` at `1 - NO limit`, because this API
quotes the YES leg. Any unfilled IOC quantity is canceled. Observed liquidity
must exist, but the reservation covers the **whole** request because liquidity
may deepen before arrival. The bound includes the maximum quadratic taker fee
and one cent of rounding per potential .01-contract fill; it can greatly exceed
the preview estimate. This is intentionally conservative for small demo tests.

Each durable intent ID receives a deterministic client order ID. Repeating it
returns the recorded attempt; it never sends another POST. A timeout, malformed
acknowledgement, interrupted process, incomplete pagination or inconsistent
fill record retains the reservation and blocks new submissions. Reconciliation
searches all available order pages for that client ID. An absent order is not
proof of rejection: no resend or automatic reservation release occurs. If the
exchange never exposes the order, leave submissions stopped for manual review.
Do not clear the journal or invent a replacement intent to work around ambiguity.

Only terminal exchange status plus complete, deduplicated, identity-checked fill
counts replaces the reservation with actual purchase-equivalent premium plus
reported `fee_cost`. Missing fees never become zero. Cancellation can race fills,
so its acknowledgement alone never releases money. Known fills may not disappear
or mutate on later reconciliation. Actual fills must respect the limit.

Filled cost remains open until a finalized market exposes an explicit per-YES
settlement value. Intermediate payouts, including 50 cents, are preserved.
Settlement releases open exposure, but not gross daily spending. This journal
accounts only for these demo orders; existing positions, opposite-side netting,
rebates and other account activity can make account cash movements differ.
Its purchase-equivalent profit is not an audited account P&L.

## Verification and promotion

```sh
.venv-kalshi-demo/bin/python -m unittest discover -s tests -p 'test_kalshi_demo.py'
```

Tests use generated ephemeral keys and simulated responses, never account access.
Before calling this account-tested, complete one open-market demo inspection,
submit one reviewed small IOC, reconcile the exchange's actual fill/fee fields,
repeat the same ID to verify no duplicate, and reconcile any remaining order.
Exercise cancellation if an order unexpectedly remains open. Record the demo
results privately. Unsupported or changed API responses retain reservations.

Live trading remains a separate implementation and decision: account eligibility,
strategy validation, explicit real capital limits, persistent hosting and recovery
must be settled before connecting production order endpoints. This CLI cannot
activate real-money orders by changing a URL or environment flag.

Official references checked October 4, 2026:

* [Demo environment](https://docs.kalshi.com/getting_started/demo_env)
* [Authenticated requests](https://docs.kalshi.com/getting_started/quick_start_authenticated_requests)
* [Order direction](https://docs.kalshi.com/getting_started/order_direction)
* [Create order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2)
* [Get order](https://docs.kalshi.com/api-reference/orders/get-order)
* [Get orders](https://docs.kalshi.com/api-reference/orders/get-orders)
* [Cancel order V2](https://docs.kalshi.com/api-reference/orders/cancel-order-v2)
* [Get fills](https://docs.kalshi.com/api-reference/portfolio/get-fills)

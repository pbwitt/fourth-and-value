# Arbitrage: scans, Scan now and the plan

The Arbitrage section (`/arbitrage/`; off the site menu for now, reached by URL) lists pairs of
bets at two sportsbooks that pay back more than the total staked whichever side
wins, for the NFL, NBA, NHL and MLB. The reader-facing plan, history and rules
are on `/arbitrage/plan.html`. Nothing here feeds Top Picks or Market Watch.

## Pieces

| File | Role |
| --- | --- |
| `docs/assets/arbitrage.js` | The engine, shared by the page, its Scan now button and the scheduled scans. |
| `config/arbitrage_settlement.json` | Sportsbook kinds (licensed, offshore, sweepstakes) and NBA, NFL and MLB settlement profiles. |
| `config/nhl_settlement.json` | NHL profiles, read as they are, so `/nhl/arbitrage.html` and `/arbitrage/` agree. |
| `scripts/arbitrage/build.py` | Writes `docs/arbitrage/data/rules.json` and `docs/arbitrage/plan.html` (from `scripts/arbitrage/plan.html`). `--check` in PRs. |
| `scripts/arbitrage/scan.cjs` | Scheduled scan for one league: `docs/arbitrage/data/<sport>.json` and the paper ledger `docs/arbitrage/data/ledger/<sport>.json`. |
| `scripts/arbitrage/replay.cjs` | One-time replay over git history: `docs/arbitrage/data/replay.json`. |
| `docs/arbitrage/index.html`, `finder.js`, `arbitrage.css` | The finder page. |
| `supabase/functions/live-odds/handler.mjs` | Editor-only odds relay used by Live Odds and Scan now. |

## What counts

For each event the engine pairs bets on the same subject (the game's margin, its
total, or one player's market): both sides of one line, two lines that overlap
(a middle), or a moneyline against the other team's spread. Each pair is graded
on every final result that matters (margins, or counts from zero), so pushes and
overlaps are exact; a pair is dropped if any result loses both bets or pushes one
and loses the other. Moneylines push on a tie only in the NFL.

Best prices are tried across every pair of distinct books. A pair is **locked**
when both quotes are within five minutes of each other, before the start, and both
books have the same settlement profile on file for that market group (game,
player, or for MLB batter and pitcher). An assumed profile counts but is labeled.
Other pairs under 100% are **check** leads with reasons (`rules_unknown`,
`rules_differ`, `quotes_apart`). Two-way pairs from 100% to 101% are near misses.
Exact offered lines and prices are kept; started games and contradictory quotes
are dropped.

Stakes split a total in proportion to each side's break-even probability. A
result exactly on a whole-number line refunds both bets; profits are quoted for
every other result. The after-tax estimate treats each winning bet's profit as
gambling income and each losing stake as a loss, 90% deductible when itemizing
(IRC 165(d) from tax year 2026), in the pair's worst result. It ignores state tax
and annual limits and is not tax advice.

## Schedule and spending

`nba-daily.yml`, `nhl-daily.yml`, `mlb-daily.yml` and `nfl-weekly.yml` run
`node scripts/arbitrage/scan.cjs --sport <SPORT>` after their refresh saves the
snapshot and before they publish. Each writes only its own league's files, so
parallel refreshes do not collide. A rescan of a snapshot already in the ledger
does not count twice. Scans make no requests and spend nothing.

Scan now calls `live-odds` with the editor's session:

* `{sport, scan: 'lines'}`: moneylines, spreads and totals for every game in the
  next 8 days in one request, 3 credits;
* `{sport, event}`: one game with game lines and four props, up to 7 credits;
* `more: true` adds the `us2` region at twice the cost. Those books have no rules
  on file, so their pairs start as checks.

Prices are reused for 60 seconds and paid requests stop below `LIVE_ODDS_RESERVE`
(default 2,000 credits). **After changing `handler.mjs`, redeploy the `live-odds`
function in Supabase**; until then the page explains that Scan now needs it.

## Changing rules

Edit `config/arbitrage_settlement.json` (or `config/nhl_settlement.json`), then run
`python scripts/arbitrage/build.py` and commit `docs/arbitrage/data/rules.json`
and `docs/arbitrage/plan.html`. Removing a book's group sends its pairs to checks.
Replace each `assumed` entry with a `source` once the book's published rules are
read. If `/nhl/arbitrage.html` and `/arbitrage/` ever disagree on an NHL lock, the
parity test in `tests/test_arbitrage.py` should fail first.

## Replay

```sh
git fetch --unshallow origin main        # or --deepen; the replay needs history
node scripts/arbitrage/replay.cjs
python scripts/arbitrage/build.py
```

It rescans every distinct committed snapshot with today's rules at its own time.
It is a paper exercise on saved quotes, not bets placed; do not relabel it as
evidence for any pick policy. The plan page reports the replay committed with it.

## Checks

```sh
node tests/arbitrage.cjs
python -m unittest tests/test_arbitrage.py
node tests/live_odds.cjs
python scripts/arbitrage/build.py --check
python scripts/build_daily_process.py --check
python scripts/seo_check.py --changed
```

## Not modeled

Bet limits, account restrictions, state availability, obvious-error cancellations,
whole-dollar rounding, feed delay and partial fills. Offshore books are not
licensed by any US state; the finder labels them and can hide them.

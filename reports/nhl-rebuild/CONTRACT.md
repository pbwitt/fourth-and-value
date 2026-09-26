# Public NHL compatibility contract

Routes and DOM: existing five routes, shared navigation/style, cards, search, market/book/game
filters, best-price checkbox, reset, load-more and query parameters (`q`, `market`, `book`,
`game`) remain. Top is Market Watch, not a forced-picks page.

Feed envelope retains sport, season, status, checked_at, last_success_at, events, rows,
matched_events, excluded_events, props_events_skipped, history_checked_at, history_through_date,
history_error, history_player_count, requests, quota_remaining, model_status.

Event fields: nhl_game_id, season, game_type, commence_time, home_team, away_team.
Quote fields: event_id, commence_time, home_team, away_team, game, book, book_label, market,
market_label, player (empty for games), side, line (null for h2h), price (American),
book_probability (break-even conditional on settlement), quoted_at (UTC ISO).
Comparison fields: fair_probability, consensus_probability, paired_books, best_price,
other_book_probability, other_books, consensus_ev (percentage units).
Historical fields: baseline_mean, model_probability, model_status and optional
baseline_probability, baseline_push, baseline_games, baseline_season, baseline_source.

Null means unavailable. Never replace missing independent probabilities with 0.5 or consensus.
Comparison probabilities are conditional on a non-push; model_probability is unconditional
win probability with explicit push/loss alongside it. The existing baseline probability is
also unconditional. New fields are additive and distinguish these probability bases.

Necessary correction: the old consensus_ev treated whole-number lines as push-free.
Without an independent push estimate it will be null for push-capable lines; a separate
conditional price advantage remains available. Pairing must also enforce settlement
profile and quote-time proximity. Tightening pairing may reduce eligible comparisons.

Failure and freshness: feed errors, stale snapshots, started games and old/future quotes
are hidden. A valid empty slate is successful. Model failures retain current market quotes
with an explicit unavailable model state; old predictions never replace missing current ones.

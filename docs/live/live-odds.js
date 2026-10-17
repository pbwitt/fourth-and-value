/* Cross-book comparison for one NHL or NBA game, before or during play. Shared by the
   private Live Odds page and Node tests (tests/live_odds.cjs).

   Same rules as the scheduled comparison (scripts/nba/pipeline.py flatten and
   scripts/nhl/v2/pricing.py compare): each book's two sides are de-vigged
   multiplicatively; a quote is compared with the median fair probability of the
   OTHER books, and only when at least three of them price the same line. The
   difference is time. Quotes from games in progress are kept, and a comparison
   only uses quotes close together in time, because a price from before a goal or
   basket is a different bet. NBA prices move with nearly every possession, so its
   live window is tighter. A quote far behind the newest one in the response is
   shown but never compared.

   Exact offered lines and prices are kept. A side without its pair has no fair
   probability rather than a guessed one. Whole-number lines can push, so their
   probabilities are conditional on no push. One-way bets (milestones such as
   "to score", "3+ shots" or "25+ points" often come without the other side) have no margin to
   remove, so they are only compared on price: how the best price stands against
   the other books' median price, margins included. Nothing here feeds Top Picks or
   Market Watch; a bet you track goes to Bet Tracker as an ordinary ticket. */
(function (global) {
'use strict';

// Markets per league, in display order: game markets, player props, then the
// milestones books post during games (the function asks for them once play starts).
// `window`: reference quotes must be this close in time to the quote they judge;
// `stale`: a quote this far behind the newest one in the response is too old to compare.
const SPORTS = {
  NHL: { key: 'icehockey_nhl', start: 'puck drop',
    markets: { h2h: 'Moneyline', spreads: 'Puck line', totals: 'Game total',
      player_shots_on_goal: 'Shots on goal', player_goals: 'Goals', player_assists: 'Assists', player_points: 'Points',
      player_goal_scorer_anytime: 'Anytime goal scorer', player_shots_on_goal_alternate: 'Shots on goal (alternate)',
      player_points_alternate: 'Points (alternate)' },
    window: { live: 2 * 60e3, pre: 15 * 60e3 }, stale: { live: 5 * 60e3, pre: 24 * 3600e3 } },
  NBA: { key: 'basketball_nba', start: 'tipoff',
    markets: { h2h: 'Moneyline', spreads: 'Spread', totals: 'Game total',
      player_points: 'Points', player_rebounds: 'Rebounds', player_assists: 'Assists', player_threes: 'Threes made',
      player_points_alternate: 'Points (alternate)', player_rebounds_alternate: 'Rebounds (alternate)',
      player_assists_alternate: 'Assists (alternate)', player_threes_alternate: 'Threes made (alternate)' },
    window: { live: 60e3, pre: 15 * 60e3 }, stale: { live: 3 * 60e3, pre: 24 * 3600e3 } },
};
const BY_KEY = Object.fromEntries(Object.entries(SPORTS).map(([sport, s]) => [s.key, sport]));
const YES_NO = new Set(['player_goal_scorer_anytime']);
// The grader's contract for each milestone: anytime scorer = goals over 0.5; "3+ shots" = shots over 2.5;
// "25+ points" = points over 24.5.
const BASE = { player_goal_scorer_anytime: 'player_goals', player_shots_on_goal_alternate: 'player_shots_on_goal',
  player_points_alternate: 'player_points', player_rebounds_alternate: 'player_rebounds',
  player_assists_alternate: 'player_assists', player_threes_alternate: 'player_threes' };
const MIN_OTHER_BOOKS = 3;   // as consensus_ev in the scheduled pipeline
const FLAG_EDGE = 2;         // percent; the scheduled pipeline's minimum EV target
const GAP_FLAG = 10;         // percent; one-way price gaps worth listing
const PAIR_WINDOW = 5 * 60e3;

// The league an Odds API response belongs to; responses without one are NHL, the original sport.
const sportOf = event => BY_KEY[event?.sport_key] || 'NHL';

const median = values => {
  const v = [...values].sort((a, b) => a - b), m = v.length >> 1;
  return v.length ? (v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2) : null;
};

function implied(price) {
  const p = Number(price);
  if (!Number.isFinite(p) || Math.abs(p) < 100) return NaN;
  return p > 0 ? 100 / (p + 100) : -p / (100 - p);
}

// Fair American odds for a probability; null when there is none.
function american(probability) {
  if (!(probability > 0 && probability < 1)) return null;
  const dec = 1 / probability;
  return Math.round(dec >= 2 ? 100 * (dec - 1) : -100 / (dec - 1));
}

const signed = n => n == null ? '—' : (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n);
const point = n => (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n);

// Every usable quote in one Odds API event-odds response.
function quotes(event, sport = sportOf(event)) {
  const out = [];
  if (!event || !Array.isArray(event.bookmakers) || !SPORTS[sport]) return out;
  const home = event.home_team, away = event.away_team, labels = SPORTS[sport].markets;
  for (const book of event.bookmakers) {
    for (const market of book?.markets || []) {
      const key = market?.key;
      if (!Object.hasOwn(labels, key)) continue;
      const at = Date.parse(market.last_update || book.last_update);
      if (!Number.isFinite(at)) continue;
      for (const o of market.outcomes || []) {
        const probability = implied(o?.price);
        if (!Number.isFinite(probability)) continue;
        const prop = key.startsWith('player_'), yesNo = YES_NO.has(key), side = String(o.name ?? '');
        const player = prop ? String(o.description ?? '').trim() : '';
        let line = null;
        if (key !== 'h2h' && !yesNo) {
          line = o.point == null || o.point === '' ? NaN : Number(o.point);
          if (!Number.isFinite(line)) continue;
        }
        if (prop && !player) continue;
        if (yesNo ? side !== 'Yes' && side !== 'No' : (prop || key === 'totals') && side !== 'Over' && side !== 'Under') continue;
        if ((key === 'h2h' || key === 'spreads') && side !== home && side !== away) continue;
        out.push({ sport, book: String(book.key), book_label: String(book.title || book.key), market: key,
          market_label: labels[key], player, side, line, price: Number(o.price), book_probability: probability,
          home_team: home, away_team: away, at, quoted_at: new Date(at).toISOString() });
      }
    }
  }
  return out;
}

// Home -1.5 pairs with away +1.5, never with away -1.5.
function offerKey(r) {
  const line = r.market === 'spreads' && r.side === r.away_team ? -r.line : r.line;
  return [r.market, r.player, line].join('|');
}

function compare(rows, mode, sport = rows[0]?.sport || 'NHL') {
  const { window: WINDOW, stale: STALE } = SPORTS[sport] || SPORTS.NHL;
  const unique = new Map(), conflicts = new Set();
  for (const row of rows) {
    const id = [offerKey(row), row.book, row.side].join('|'), prev = unique.get(id);
    if (prev) {
      if (row.at < prev.at) continue;
      // Contradictory prices at the same moment fail closed.
      if (row.at === prev.at && row.price !== prev.price) conflicts.add(id);
    }
    unique.set(id, { ...row });
  }
  const kept = [...unique].filter(([id]) => !conflicts.has(id)).map(([, r]) => r);
  const newest = kept.reduce((m, r) => Math.max(m, r.at), -Infinity);
  const group = (keyOf) => {
    const m = new Map();
    for (const r of kept) { const k = keyOf(r); if (!m.has(k)) m.set(k, []); m.get(k).push(r); }
    return [...m.values()];
  };
  for (const pair of group(r => offerKey(r) + '|' + r.book)) {
    const m = pair[0].market;
    const expected = m === 'h2h' || m === 'spreads' ? [pair[0].home_team, pair[0].away_team]
      : YES_NO.has(m) ? ['Yes', 'No'] : ['Over', 'Under'];
    const valid = pair.length === 2 && expected.every(s => pair.some(r => r.side === s))
      && Math.abs(pair[0].at - pair[1].at) <= PAIR_WINDOW;
    const total = pair.reduce((s, r) => s + r.book_probability, 0);
    for (const r of pair) {
      r.fair_probability = valid ? r.book_probability / total : null;
      r.stale = newest - r.at > STALE[mode];
    }
  }
  for (const side of group(r => offerKey(r) + '|' + r.side)) {
    for (const r of side) {
      const refs = side.filter(q => q.book !== r.book && q.fair_probability != null && !q.stale
        && Math.abs(q.at - r.at) <= WINDOW[mode]);
      r.other_probability = median(refs.map(q => q.fair_probability));
      r.other_books = refs.length;
      r.advantage = !r.stale && refs.length >= MIN_OTHER_BOOKS ? 100 * (r.other_probability / r.book_probability - 1) : null;
      // Without enough fair prices, compare prices only: the other books' margins stay in.
      const raw = side.filter(q => q.book !== r.book && !q.stale && Math.abs(q.at - r.at) <= WINDOW[mode]);
      r.other_price_probability = median(raw.map(q => q.book_probability));
      r.price_books = raw.length;
      r.price_gap = r.advantage == null && !r.stale && raw.length >= MIN_OTHER_BOOKS
        ? 100 * (r.other_price_probability / r.book_probability - 1) : null;
    }
  }
  return { rows: kept, newest: Number.isFinite(newest) ? newest : null };
}

function label(r) {
  if (r.market === 'h2h') return r.side;
  if (r.market === 'spreads') return `${r.side} ${point(r.line)}`;
  if (YES_NO.has(r.market)) return r.side === 'Yes' ? 'To score' : 'Not to score';
  if (BASE[r.market] && r.side === 'Over' && r.line % 1 === 0.5) return `Over ${r.line} (${r.line + 0.5}+)`;
  return `${r.side} ${r.line}`;
}

/* One line per offered bet (market, player, line and side), best fresh price first.
   Flagged lines beat the other books' median fair price by FLAG_EDGE percent or more;
   gap-flagged one-way lines beat the other books' median price by GAP_FLAG percent. */
function board(event, now = Date.now(), sport = sportOf(event)) {
  const start = Date.parse(event?.commence_time);
  const mode = Number.isFinite(start) && start <= now ? 'live' : 'pre';
  const order = Object.keys(SPORTS[sport]?.markets || {});
  const { rows, newest } = compare(quotes(event, sport), mode, sport);
  const sides = new Map();
  for (const r of rows) {
    const k = offerKey(r) + '|' + r.side;
    if (!sides.has(k)) sides.set(k, []);
    sides.get(k).push(r);
  }
  const lines = [...sides.values()].map(qs => {
    const fresh = qs.filter(q => !q.stale);
    const pool = fresh.length ? fresh : qs;
    // Best price = lowest implied probability; on a tie, the newer quote.
    const best = pool.reduce((b, q) => q.book_probability < b.book_probability
      || (q.book_probability === b.book_probability && q.at > b.at) ? q : b);
    const fair = median(fresh.filter(q => q.fair_probability != null).map(q => q.fair_probability));
    const r = qs[0];
    return {
      id: offerKey(r) + '|' + r.side, market: r.market, market_label: r.market_label, player: r.player,
      side: r.side, line: r.line, label: label(r), push_possible: r.market !== 'h2h' && Number.isInteger(r.line),
      best, fair_probability: fair, fair_odds: american(fair),
      fair_books: fresh.filter(q => q.fair_probability != null).length,
      flagged: best.advantage != null && best.advantage >= FLAG_EDGE,
      gap_flagged: best.advantage == null && best.price_gap != null && best.price_gap >= GAP_FLAG,
      quotes: [...qs].sort((a, b) => a.book_probability - b.book_probability || a.book_label.localeCompare(b.book_label)),
    };
  });
  const rank = l => l.flagged ? 2 : l.gap_flagged ? 1 : 0;
  lines.sort((a, b) => rank(b) - rank(a)
    || (rank(a) === 2 ? b.best.advantage - a.best.advantage : rank(a) === 1 ? b.best.price_gap - a.best.price_gap : 0)
    || order.indexOf(a.market) - order.indexOf(b.market) || a.player.localeCompare(b.player)
    || (a.line ?? 0) - (b.line ?? 0) || a.side.localeCompare(b.side));
  return { sport, mode, newest: newest == null ? null : new Date(newest).toISOString(),
    books: [...new Set(rows.map(r => r.book_label))].sort(), lines };
}

// ---- Bet Tracker -----------------------------------------------------------------

const LIVE_SETTLEMENT = 'Full-game market: it settles on the whole game, including play before you bet. '
  + 'Verify your sportsbook’s live rules and market exceptions.';

// The offer row Bet Tracker's shared dialog expects (docs/assets/offer-tracker.js).
// No model probability: a comparison between books is not a forecast. Milestones are
// saved as the grader's base contract: anytime scorer = goals "Yes" with no line.
function ticket(game, q) {
  return { sport: game.sport || q.sport || 'NHL', event_id: game.id, commence_time: game.commence_time,
    game: `${game.away_team} @ ${game.home_team}`, home_team: game.home_team, away_team: game.away_team,
    player: q.player || '', market: BASE[q.market] || q.market, market_label: q.market_label, side: q.side,
    line: YES_NO.has(q.market) ? null : q.line,
    book: q.book, book_label: q.book_label, price: q.price, quoted_at: q.quoted_at, settlement_scope: LIVE_SETTLEMENT };
}

// ---- Matching the league scoreboard (docs/tracking/live-feeds.js) -----------------

const squash = s => String(s || '').normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]/g, '');

// "Montreal Canadiens" (odds) is the scoreboard's "Montréal Canadiens" or "Canadiens";
// "Los Angeles Clippers" is ESPN's "LA Clippers" or "Clippers".
function sameTeam(full, team) {
  const f = squash(full), short = squash(team?.short || team?.name);
  return !!f && !!short && (f === squash(team?.name) || f.endsWith(short));
}

function findGame(games, event) {
  return (games || []).find(g => sameTeam(event?.home_team, g.home) && sameTeam(event?.away_team, g.away)) || null;
}

// League schedule dates are Eastern: a 10 pm ET start is 02:00 UTC the next day.
function easternDate(when) {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' })
    .format(new Date(when));
}

const api = { SPORTS, MIN_OTHER_BOOKS, FLAG_EDGE, GAP_FLAG, sportOf, implied, american, signed, quotes, compare,
  board, ticket, findGame, easternDate };
if (typeof module === 'object' && module.exports) module.exports = api;
else global.FVLiveOdds = api;
})(typeof window === 'undefined' ? globalThis : window);

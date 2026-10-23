/* Cross-book arbitrage for every league. Shared by the Arbitrage page (docs/arbitrage/),
   its Scan now button and the scheduled scans (scripts/arbitrage/scan.cjs), so one set of
   rules decides what counts.

   A pair is two bets at two books where every final result pays at least the total staked:
   both sides of one line, two different lines that overlap (a middle: both can win), or a
   moneyline against the other team's spread. Instead of trusting a formula per shape, each
   pair is graded on every result that matters (the final margin, or the player's or game's
   count), so pushes and overlaps are exact. Staked in proportion to each side's break-even
   probability, a single winning side returns total / (p1 + p2) either way.

   Exact offered lines and prices are kept. A pair is locked only when both quotes were
   taken within PAIR_WINDOW of each other, before the game started, at two different books
   whose rules for that market are on file and match (docs/arbitrage/data/rules.json, built
   from config/arbitrage_settlement.json and config/nhl_settlement.json). Anything else under
   100% is a lead to check, never a lock. Limits, voids, palpable-error rules and account
   restrictions are not modeled. Nothing here feeds Top Picks or Market Watch. */
(function (global) {
'use strict';

const PAIR_WINDOW = 5 * 60e3;   // as scripts/nhl/v2/pricing.py
const NEAR = 1.01;              // two-way pairs this close to 100% are listed as near misses
const NEAR_LIMIT = 10;
const DEDUCTIBLE = 0.9;         // share of gambling losses deductible from tax year 2026 (IRC 165(d))
// Leagues whose games can end level after full time; there a moneyline pushes on a tie.
const TIES = new Set(['NFL']);
const KEYS = { NFL: 'americanfootball_nfl', NBA: 'basketball_nba', NHL: 'icehockey_nhl', MLB: 'baseball_mlb' };
const BY_KEY = Object.fromEntries(Object.entries(KEYS).map(([s, k]) => [k, s]));
const GAME = new Set(['h2h', 'spreads', 'totals']);
// Alternate and milestone markets settle as their base market at the offered line.
const BASE = { alternate_spreads: 'spreads', alternate_totals: 'totals', player_goal_scorer_anytime: 'player_goals' };
const LABELS = { h2h: 'Moneyline', totals: 'Game total', player_shots_on_goal: 'Shots on goal', player_threes: 'Threes made',
  batter_rbis: 'RBIs', batter_total_bases: 'Total bases', batter_hits_runs_rbis: 'Hits + runs + RBIs',
  player_points_rebounds_assists: 'Points + rebounds + assists', player_points_rebounds: 'Points + rebounds',
  player_points_assists: 'Points + assists', player_rebounds_assists: 'Rebounds + assists', player_blocks_steals: 'Blocks + steals',
  player_pass_yds: 'Passing yards', player_rush_yds: 'Rushing yards', player_reception_yds: 'Receiving yards',
  player_pass_tds: 'Passing TDs', player_pass_completions: 'Completions', player_pass_attempts: 'Pass attempts' };
const SPREAD_LABEL = { NHL: 'Puck line', MLB: 'Run line' };

function marketLabel(sport, market) {
  if (market === 'spreads') return SPREAD_LABEL[sport] || 'Spread';
  if (LABELS[market]) return LABELS[market];
  const words = (market.startsWith('pitcher_') ? 'pitcher ' : '') + market.replace(/^(player|batter|pitcher)_/, '').replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function implied(price) {
  const p = Number(price);
  if (!Number.isFinite(p) || Math.abs(p) < 100) return NaN;
  return p > 0 ? 100 / (p + 100) : -p / (100 - p);
}

// Which of a book's rule sets governs a market: game bets, or player props (MLB splits batters and pitchers).
function groupOf(sport, market) {
  if (GAME.has(market)) return 'game';
  if (sport === 'MLB' && market.startsWith('pitcher_')) return 'pitcher';
  if (sport === 'MLB' && market.startsWith('batter_')) return 'batter';
  return 'player';
}

const num = v => v == null || v === '' ? NaN : Number(v);

// One usable quote, or null. Rows from any of our snapshots or the Odds API share this shape.
function quote(sport, r) {
  const market = BASE[r.market] || r.market, side = String(r.side ?? '');
  const at = Date.parse(r.quoted_at), start = Date.parse(r.commence_time), probability = implied(r.price);
  if (!market || !r.event_id || !r.book || !Number.isFinite(at) || !Number.isFinite(start) || !Number.isFinite(probability)) return null;
  const q = { sport, event_id: String(r.event_id), game: r.game || `${r.away_team} @ ${r.home_team}`,
    home_team: r.home_team, away_team: r.away_team, commence_time: new Date(start).toISOString(), start,
    book: String(r.book), book_label: String(r.book_label || r.book), market,
    market_label: r.market === market && r.market_label ? String(r.market_label) : marketLabel(sport, market),
    player: String(r.player || '').trim(), side, line: null, price: Number(r.price), probability, at,
    quoted_at: new Date(at).toISOString() };
  if (market === 'h2h') return side === q.home_team || side === q.away_team ? q : null;
  q.line = num(r.line);
  if (!Number.isFinite(q.line)) return null;
  if (market === 'spreads') return side === q.home_team || side === q.away_team ? q : null;
  if (market !== 'totals' && !q.player) return null;
  return side === 'Over' || side === 'Under' ? q : null;
}

// Rows from docs/<sport>/data/latest.json or docs/nfl/data/quotes.json.
function fromSnapshot(rows, sport) {
  return (rows || []).map(r => quote(sport, r)).filter(Boolean);
}

// An Odds API odds response: one event or a list of them.
function fromOddsApi(events, sport) {
  const out = [];
  for (const event of Array.isArray(events) ? events : [events]) {
    const league = sport || BY_KEY[event?.sport_key];
    if (!league || !Array.isArray(event?.bookmakers)) continue;
    for (const book of event.bookmakers) {
      for (const market of book?.markets || []) {
        const key = String(market?.key || ''), milestone = key === 'player_goal_scorer_anytime';
        const base = BASE[key] || key.replace(/_alternate$/, '');
        for (const o of market.outcomes || []) {
          let side = String(o?.name ?? ''), line = o?.point;
          if (milestone) {
            if (side !== 'Yes' && side !== 'No') continue;
            side = side === 'Yes' ? 'Over' : 'Under'; line = 0.5;
          }
          const q = quote(league, { event_id: event.id, home_team: event.home_team, away_team: event.away_team,
            commence_time: event.commence_time, book: book.key, book_label: book.title, market: base,
            player: o?.description || '', side, line, price: o?.price, quoted_at: market.last_update || book.last_update });
          if (q) out.push(q);
        }
      }
    }
  }
  return out;
}

/* Settlement for one quote's market at its book: { profile, assumed } or null when the book's
   rules for that market are not on file. */
function settlement(rules, sport, book, market) {
  const entry = rules?.sports?.[sport]?.books?.[book], group = groupOf(sport, market);
  const profile = entry?.[group];
  return profile ? { profile, assumed: (entry.assumed || []).includes(group) } : null;
}

const bookInfo = (rules, book) => rules?.books?.[book] || null;

// win/push/loss of one bet at one final result. Margins are the home team's.
function result(q, outcome) {
  let x;
  if (q.market === 'h2h' || q.market === 'spreads') {
    const margin = q.side === q.home_team ? outcome : -outcome;
    x = margin + (q.market === 'spreads' ? q.line : 0);
  } else {
    x = q.side === 'Over' ? outcome - q.line : q.line - outcome;
  }
  return x > 0 ? 'win' : x === 0 ? 'push' : 'loss';
}

// Every final result that can separate these bets: home margins, or counts from zero up.
function outcomes(sport, legs) {
  const margin = legs[0].market === 'h2h' || legs[0].market === 'spreads';
  const lines = legs.map(q => q.line == null ? 0 : q.line);
  const lo = margin ? -Math.ceil(Math.max(...lines.map(Math.abs))) - 2 : Math.max(0, Math.floor(Math.min(...lines)) - 1);
  const hi = margin ? -lo : Math.ceil(Math.max(...lines)) + 1;
  const list = [];
  for (let v = lo; v <= hi; v++) if (!(margin && v === 0 && !TIES.has(sport))) list.push(v);
  return list;
}

/* Distinct result patterns for a pair, e.g. [['win','loss'],['loss','win'],['win','win']], or
   null when some final result loses both bets or pushes one and loses the other. */
function coverage(sport, legs) {
  const seen = new Map();
  for (const v of outcomes(sport, legs)) {
    const r = legs.map(q => result(q, v));
    if (!r.includes('win') && r.includes('loss')) return null;
    const k = r.join('|');
    if (!seen.has(k)) seen.set(k, { results: r, values: [] });
    seen.get(k).values.push(v);
  }
  return [...seen.values()];
}

// Where both bets win, described from the first leg's subject: counts, or that team's margin.
function middleRange(legs, patterns) {
  const both = patterns.find(p => p.results.every(r => r === 'win'));
  if (!both) return null;
  const q = legs[0], margin = q.market === 'h2h' || q.market === 'spreads';
  const team = margin ? (q.side === q.home_team ? q.home_team : q.away_team) : null;
  const values = margin ? both.values.map(v => team === q.home_team ? v : -v) : both.values;
  return { team, from: Math.min(...values), to: Math.max(...values) };
}

// The bet a quote describes, without book or price: the identity a pair is tracked under.
const offer = q => [q.market, q.player, q.side, q.line ?? ''].join('|');

/* Best pairs in one list of quotes. Options: rules (published rules.json), sport, now (ms;
   games starting by then are skipped). Returns { locked, check, near, stats }. */
function scan(input, { rules = null, sport = input[0]?.sport, now = Date.now() } = {}) {
  // Newest quote per book and offer; contradictory prices at one moment fail closed.
  const latest = new Map(), conflicts = new Set();
  for (const q of input) {
    if (q.start <= now || q.at >= q.start) continue;
    const id = [q.event_id, offer(q), q.book].join('#'), prev = latest.get(id);
    if (prev && q.at < prev.at) continue;
    if (prev && q.at === prev.at && q.price !== prev.price) conflicts.add(id);
    latest.set(id, q);
  }
  const quotes = [...latest].filter(([id]) => !conflicts.has(id)).map(([, q]) => q);
  // Subjects: one event's margin (moneylines and spreads), its total, or one player's market.
  const subjects = new Map();
  for (const q of quotes) {
    const k = [q.event_id, q.market === 'h2h' || q.market === 'spreads' ? 'margin' : q.market, q.player].join('#');
    if (!subjects.has(k)) subjects.set(k, new Map());
    const bets = subjects.get(k), b = offer(q);
    if (!bets.has(b)) bets.set(b, []);
    bets.get(b).push(q);
  }
  const locked = [], check = [], near = [];
  for (const bets of subjects.values()) {
    const list = [...bets.values()];
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const [a, b] = [list[i][0], list[j][0]];
        if (a.side === b.side) continue;
        const patterns = coverage(sport, [a, b]);
        if (!patterns) continue;
        const middle = middleRange([a, b], patterns);
        const best = { locked: null, other: null };
        for (const qa of list[i]) {
          for (const qb of list[j]) {
            if (qa.book === qb.book) continue;
            const total = qa.probability + qb.probability;
            if (total >= (middle ? 1 : NEAR)) continue;
            const pair = describe([qa, qb], total, patterns, middle, rules, sport);
            const slot = pair.status === 'locked' ? 'locked' : 'other';
            if (!best[slot] || total < best[slot].implied_total) best[slot] = pair;
          }
        }
        if (best.locked) locked.push(best.locked);
        const other = best.other;
        if (other && (!best.locked || other.implied_total < best.locked.implied_total)) {
          (other.implied_total < 1 ? check : near).push(other);
        }
      }
    }
  }
  const byReturn = (x, y) => x.implied_total - y.implied_total || x.start - y.start;
  locked.sort(byReturn); check.sort(byReturn); near.sort(byReturn);
  return { locked, check, near: near.slice(0, NEAR_LIMIT),
    stats: { quotes: quotes.length, events: new Set(quotes.map(q => q.event_id)).size, subjects: subjects.size,
      books: [...new Set(quotes.map(q => q.book))].sort(),
      newest: quotes.length ? new Date(Math.max(...quotes.map(q => q.at))).toISOString() : null } };
}

function describe(legs, total, patterns, middle, rules, sport) {
  const rule = legs.map(q => settlement(rules, sport, q.book, q.market));
  const reasons = [];
  rule.forEach((r, i) => { if (!r) reasons.push({ code: 'rules_unknown', book: legs[i].book_label, market: legs[i].market_label }); });
  if (rule.every(Boolean) && rule[0].profile !== rule[1].profile) reasons.push({ code: 'rules_differ' });
  const spread = Math.abs(legs[0].at - legs[1].at);
  if (spread > PAIR_WINDOW) reasons.push({ code: 'quotes_apart', seconds: Math.round(spread / 1000) });
  const tier = reasons.some(r => r.code.startsWith('rules')) ? 'unknown' : rule.some(r => r.assumed) ? 'assumed' : 'verified';
  const status = total < 1 && !reasons.length ? 'locked' : total < 1 ? 'check' : 'near';
  const ordered = [...legs].sort((x, y) => offer(x).localeCompare(offer(y)));
  const q = legs[0];
  return {
    id: [sport, q.event_id, ...ordered.map(offer)].join('#'), sport, status, tier, reasons,
    kind: middle ? 'middle' : 'two_way', cross_market: legs[0].market !== legs[1].market,
    event_id: q.event_id, game: q.game, home_team: q.home_team, away_team: q.away_team,
    commence_time: q.commence_time, start: q.start, player: q.player,
    subject: q.player ? `${q.player} · ${q.market_label}` : legs[0].market === legs[1].market ? q.market_label : 'Moneyline and spread',
    implied_total: total, return_pct: 100 * (1 / total - 1), middle,
    push_refund: patterns.some(p => p.results.every(r => r === 'push')),
    outcomes: patterns.map(p => p.results),
    quote_spread_seconds: Math.round(spread / 1000),
    licensed: legs.every(l => bookInfo(rules, l.book)?.licensed === true),
    legs: legs.map((l, i) => ({ book: l.book, book_label: bookInfo(rules, l.book)?.label || l.book_label,
      licensed: bookInfo(rules, l.book)?.licensed ?? null, market: l.market, market_label: l.market_label, player: l.player,
      side: l.side, line: l.line, price: l.price, probability: l.probability, quoted_at: l.quoted_at,
      settlement_profile: rule[i]?.profile || null, settlement_assumed: !!rule[i]?.assumed })),
  };
}

const decimal = price => 1 / implied(price);
// A result that refunds both bets returns the stake; profits are quoted for every other result.
const settled = pair => pair.outcomes.filter(results => !results.every(r => r === 'push'));

/* Stakes that pay the same whichever single side wins, for `total` dollars: the worst and best
   profit over results that settle (a result exactly on a whole-number line refunds both).
   Cents are kept; books that only take whole dollars shift it slightly. */
function stakes(pair, total) {
  const T = pair.legs.reduce((s, l) => s + implied(l.price), 0);
  const legs = pair.legs.map(l => ({ ...l, stake: total * implied(l.price) / T, payout: total / T }));
  const returns = settled(pair).map(results => results.reduce((s, r, i) =>
    s + (r === 'win' ? legs[i].stake * decimal(legs[i].price) : r === 'push' ? legs[i].stake : 0), 0));
  return { legs, payout: total / T, profit: Math.min(...returns) - total, best: Math.max(...returns) - total };
}

/* Profit after federal income tax in the worst result. Each winning bet's profit is gambling
   income; a losing bet is a loss, deductible only when itemizing and then only DEDUCTIBLE of it
   (from tax year 2026). An estimate, not tax advice: state tax and the annual limits are ignored. */
function afterTax(pair, total, { rate = 0.24, itemize = true, deductible = DEDUCTIBLE } = {}) {
  const { legs } = stakes(pair, total);
  const values = settled(pair).map(results => {
    let wins = 0, losses = 0;
    results.forEach((r, i) => {
      if (r === 'win') wins += legs[i].stake * (decimal(legs[i].price) - 1);
      else if (r === 'loss') losses += legs[i].stake;
    });
    const taxable = Math.max(0, itemize ? wins - deductible * losses : wins);
    return wins - losses - rate * taxable;
  });
  return Math.min(...values);
}

const api = { PAIR_WINDOW, NEAR, DEDUCTIBLE, KEYS, implied, marketLabel, groupOf, settlement, fromSnapshot,
  fromOddsApi, coverage, scan, stakes, afterTax };
if (typeof module === 'object' && module.exports) module.exports = api;
else global.FVArbitrage = api;
})(typeof window === 'undefined' ? globalThis : window);

// Pure closing-line logic, free of I/O so tests/closing_lines.cjs can run it in Node.

export const SPORTS = {
  NFL: 'americanfootball_nfl',
  NCAAF: 'americanfootball_ncaaf',
  NHL: 'icehockey_nhl',
  MLB: 'baseball_mlb',
  NBA: 'basketball_nba',
  WNBA: 'basketball_wnba',
};

// Tracker market_type -> Odds API market key and how its outcomes are shaped.
// kind: player (Over/Under with description = player), total (Over/Under),
// h2h (outcome name = team), spread (outcome name = team, point = spread).
const P = key => ({ key, kind: 'player' });
const GAME = {
  h2h: { key: 'h2h', kind: 'h2h' },
  moneyline: { key: 'h2h', kind: 'h2h' },
  spreads: { key: 'spreads', kind: 'spread' },
  spread: { key: 'spreads', kind: 'spread' },
  totals: { key: 'totals', kind: 'total' },
  total: { key: 'totals', kind: 'total' },
};
export const MARKETS = {
  NFL: {
    ...GAME,
    receptions: P('player_receptions'),
    rec_yds: P('player_reception_yds'),
    rush_yds: P('player_rush_yds'),
    rush_attempts: P('player_rush_attempts'),
    pass_yds: P('player_pass_yds'),
    pass_attempts: P('player_pass_attempts'),
    pass_completions: P('player_pass_completions'),
    pass_tds: P('player_pass_tds'),
  },
  NHL: {
    ...GAME,
    sog: P('player_shots_on_goal'),
    points: P('player_points'),
    assists: P('player_assists'),
    goals: P('player_goals'),
    // Legacy label: bet tickets saved before 2026-10-05 recorded NHL game totals as
    // "team_total" (the ticket mapped the feed's `totals` market to that name).
    // Those rows carry full-game lines (6.5, 7.5) and are priced as game totals.
    // New tickets record `totals`.
    team_total: { key: 'totals', kind: 'total' },
  },
  MLB: {
    ...GAME,
    pitcher_strikeouts: P('pitcher_strikeouts'),
    pitcher_outs: P('pitcher_outs'),
    pitcher_hits_allowed: P('pitcher_hits_allowed'),
  },
};
MARKETS.NBA = {
  ...GAME,
  points: P('player_points'),
  pts: P('player_points'),
  rebounds: P('player_rebounds'),
  reb: P('player_rebounds'),
  assists: P('player_assists'),
  ast: P('player_assists'),
  threes: P('player_threes'),
  '3pm': P('player_threes'),
  blocks: P('player_blocks'),
  steals: P('player_steals'),
  turnovers: P('player_turnovers'),
  pra: P('player_points_rebounds_assists'),
  pts_reb_ast: P('player_points_rebounds_assists'),
  points_rebounds_assists: P('player_points_rebounds_assists'),
  pr: P('player_points_rebounds'),
  pts_reb: P('player_points_rebounds'),
  points_rebounds: P('player_points_rebounds'),
  pa: P('player_points_assists'),
  pts_ast: P('player_points_assists'),
  points_assists: P('player_points_assists'),
  ra: P('player_rebounds_assists'),
  reb_ast: P('player_rebounds_assists'),
  rebounds_assists: P('player_rebounds_assists'),
  stocks: P('player_blocks_steals'),
  blocks_steals: P('player_blocks_steals'),
};
MARKETS.WNBA = MARKETS.NBA;
MARKETS.NCAAF = MARKETS.NFL;

export function specFor(bet) {
  const league = String(bet.league || '').toUpperCase();
  if (!SPORTS[league]) return { error: `League ${bet.league} is not tracked for CLV.` };
  const spec = MARKETS[league]?.[String(bet.market_type || '').toLowerCase()];
  if (!spec) return { error: `Market ${bet.market_type} has no closing-line mapping yet.` };
  if (spec.kind === 'player' && !bet.player) return { error: 'Player prop without a player name.' };
  if (spec.kind !== 'h2h' && (bet.line == null || bet.line === '')) return { error: 'Bet has no line.' };
  return { sport: SPORTS[league], ...spec };
}

export function norm(s) {
  return String(s ?? '')
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[.'’`]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\b(jr|sr|ii|iii|iv)\b/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

const BOOKS = { caesars: 'williamhill_us', williamhill: 'williamhill_us', betonline: 'betonlineag',
  mgm: 'betmgm', dk: 'draftkings', fd: 'fanduel', espn: 'espnbet', espn_bet: 'espnbet',
  hardrock: 'hardrockbet', hard_rock: 'hardrockbet' };
export function bookKey(book) {
  const k = String(book ?? '').toLowerCase().trim().replace(/[\s-]+/g, '_');
  return BOOKS[k] || k;
}

const dateIn = (ms, timeZone) => new Intl.DateTimeFormat('en-CA',
  { timeZone, year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(ms));
export const etDate = ms => dateIn(ms, 'America/New_York');

// Same two teams (either order), starting on the bet's date in Eastern or
// Pacific time (so a late West Coast start still matches). Full names first;
// if that finds nothing, nicknames alone ("LA Clippers" = "Los Angeles Clippers").
const nickname = s => { const w = norm(s).split(' '); return w[w.length - 1] || ''; };
export function matchEvent(bet, events) {
  const day = String(bet.game_date || '').slice(0, 10);
  const onDay = (events || []).filter(e => {
    const t = Date.parse(e.commence_time);
    return Number.isFinite(t) && (etDate(t) === day || dateIn(t, 'America/Los_Angeles') === day);
  });
  for (const key of [norm, nickname]) {
    const want = [key(bet.team_home), key(bet.team_away)].sort().join('|');
    const hits = onDay.filter(e => [key(e.home_team), key(e.away_team)].sort().join('|') === want);
    if (hits.length === 1) return { event: hits[0] };
    if (hits.length > 1) return { error: 'More than one game matched (doubleheader?).' };
  }
  return { error: `No ${bet.team_away} at ${bet.team_home} game found on ${day}.` };
}

export const implied = american => {
  const a = Number(american);
  if (!Number.isFinite(a) || a === 0) return null;
  return a > 0 ? 100 / (a + 100) : -a / (-a + 100);
};
const same = (a, b) => a != null && b != null && Math.abs(Number(a) - Number(b)) < 1e-6;
const median = xs => {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const r4 = x => Math.round(x * 1e4) / 1e4;

// The bet's closing price from one event-odds payload. Fair probability is the
// proportional no-vig price per book, at the bet's exact line, then the median
// across books. If no book still offers that line, the consensus line the side
// moved to is reported instead.
export function closeForBet(bet, spec, event) {
  const side = norm(bet.side);
  const line = bet.line == null || bet.line === '' ? null : Number(bet.line);
  const player = spec.kind === 'player' ? norm(bet.player) : null;
  const mineBook = bookKey(bet.book);
  const fairs = [], otherPoints = [];
  let closingOdds = null, exactSeen = false;

  for (const book of event?.bookmakers || []) {
    const market = (book.markets || []).find(m => m.key === spec.key);
    if (!market) continue;
    let outs = market.outcomes || [];
    if (player) outs = outs.filter(o => norm(o.description) === player);
    if (!outs.length) continue;

    let mine, opp;
    if (spec.kind === 'h2h') {
      if (outs.length !== 2) continue;
      mine = outs.find(o => norm(o.name) === side);
      opp = outs.find(o => o !== mine);
    } else if (spec.kind === 'spread') {
      mine = outs.find(o => norm(o.name) === side && same(o.point, line));
      opp = mine && outs.find(o => o !== mine && same(o.point, -line));
      if (!mine) for (const o of outs) if (norm(o.name) === side && o.point != null) otherPoints.push(Number(o.point));
    } else {
      mine = outs.find(o => norm(o.name) === side && same(o.point, line));
      opp = mine && outs.find(o => norm(o.name) !== side && same(o.point, line));
      if (!mine) for (const o of outs) if (norm(o.name) === side && o.point != null) otherPoints.push(Number(o.point));
    }
    if (!mine) continue;
    exactSeen = true;
    if (book.key === mineBook) closingOdds = Number(mine.price);
    const pm = implied(mine.price), po = opp && implied(opp.price);
    if (pm != null && po != null) fairs.push(pm / (pm + po));
  }

  if (fairs.length) {
    return { clv_status: 'captured', closing_fair_prob: r4(median(fairs)), closing_books: fairs.length,
      closing_line: line, closing_odds: closingOdds, clv_note: null };
  }
  if (exactSeen) {
    return { clv_status: 'no_market', closing_fair_prob: null, closing_books: 0, closing_line: line,
      closing_odds: closingOdds, clv_note: 'Only one side was posted at this line, so there is no fair price.' };
  }
  if (otherPoints.length) {
    return { clv_status: 'line_moved', closing_fair_prob: null, closing_books: 0,
      closing_line: median(otherPoints), closing_odds: null,
      clv_note: `No book closed at ${line}; consensus moved to ${median(otherPoints)}.` };
  }
  return { clv_status: 'no_market', closing_fair_prob: null, closing_books: 0, closing_line: null,
    closing_odds: closingOdds, clv_note: 'Market not posted by US books at the close.' };
}

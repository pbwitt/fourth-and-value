// Arbitrage: the shared engine (docs/assets/arbitrage.js), the scheduled scan and ledger
// (scripts/arbitrage/scan.cjs) and the Scan now modes of the live-odds relay. Fixtures only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const A = require('../docs/assets/arbitrage.js');
const { run } = require('../scripts/arbitrage/scan.cjs');

const ROOT = path.resolve(__dirname, '..');
const RULES = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs/arbitrage/data/rules.json'), 'utf8'));
const NOW = Date.parse('2026-10-22T15:00:00Z');
const HOME = 'Philadelphia 76ers', AWAY = 'Cleveland Cavaliers';
const row = (book, market, side, price, line = null, extra = {}) => ({ event_id: 'e1', game: `${AWAY} @ ${HOME}`,
  home_team: HOME, away_team: AWAY, commence_time: '2026-10-22T23:10:00Z', player: '', book, book_label: book,
  market, side, price, line, quoted_at: '2026-10-22T14:58:00Z', ...extra });
const prop = (book, side, price, line = 21.5, extra = {}) => row(book, 'player_points', side, price, line, { player: 'Evan Mobley', ...extra });
const scan = (rows, sport = 'NBA', options = {}) => A.scan(A.fromSnapshot(rows, sport), { rules: RULES, sport, now: NOW, ...options });
const near = (a, b, msg) => assert(Math.abs(a - b) < 1e-9, `${msg}: ${a} vs ${b}`);
const q = (rows, sport = 'NBA') => A.fromSnapshot(rows, sport);
const shapes = (legs, sport = 'NBA') => A.coverage(sport, q(legs, sport))?.map(p => p.results.join('/')).sort();

// ---- Prices and parsing --------------------------------------------------------------
near(A.implied(-110), 110 / 210, 'favorite'); near(A.implied(150), 0.4, 'underdog');
assert(Number.isNaN(A.implied(50)) && Number.isNaN(A.implied('x')));
{
  const rows = q([prop('fanduel', 'Over', -110), prop('fanduel', 'Yes', 110), prop('fanduel', 'Over', -110, null),
    row('fanduel', 'h2h', 'Somebody', 120), row('fanduel', 'spreads', HOME, -110, -3.5), row('fanduel', 'h2h', HOME, 50),
    { ...prop('fanduel', 'Under', -110), quoted_at: 'never' }, row('fanduel', 'player_points', 'Over', -110, 21.5)]);
  assert.deepEqual(rows.map(r => [r.market, r.side, r.line]), [['player_points', 'Over', 21.5], ['spreads', HOME, -3.5]],
    'unpaired names, missing lines, bad prices or times and nameless props are dropped');
  assert.equal(A.marketLabel('NHL', 'spreads'), 'Puck line'); assert.equal(A.marketLabel('MLB', 'pitcher_outs'), 'Pitcher outs');
  assert.equal(A.groupOf('MLB', 'batter_hits'), 'batter'); assert.equal(A.groupOf('NBA', 'totals'), 'game');
}
{
  // Odds API responses: alternates settle as their base market; anytime scorer is goals over 0.5.
  const event = { id: 'e9', sport_key: 'icehockey_nhl', commence_time: '2026-10-22T23:00:00Z', home_team: 'Buffalo Sabres',
    away_team: 'Chicago Blackhawks', bookmakers: [{ key: 'draftkings', title: 'DraftKings', last_update: '2026-10-22T14:59:00Z', markets: [
      { key: 'alternate_totals', outcomes: [{ name: 'Over', price: 120, point: 7.5 }] },
      { key: 'player_goal_scorer_anytime', outcomes: [{ name: 'Yes', price: 250, description: 'Tage Thompson' }, { name: 'Maybe', price: 100, description: 'X' }] },
      { key: 'player_points_alternate', outcomes: [{ name: 'Over', price: 300, point: 1.5, description: 'Tage Thompson' }] }] }] };
  assert.deepEqual(A.fromOddsApi([event]).map(r => [r.sport, r.market, r.side, r.line, r.player, r.book_label]), [
    ['NHL', 'totals', 'Over', 7.5, '', 'DraftKings'], ['NHL', 'player_goals', 'Over', 0.5, 'Tage Thompson', 'DraftKings'],
    ['NHL', 'player_points', 'Over', 1.5, 'Tage Thompson', 'DraftKings']]);
  assert.deepEqual(A.fromOddsApi({ bookmakers: [] }), [], 'unknown league');
}

// ---- Which pairs pay on every result -----------------------------------------------------
assert.deepEqual(shapes([prop('a', 'Over', 100), prop('b', 'Under', 100)]), ['loss/win', 'win/loss'], 'both sides of a half line');
assert.equal(shapes([prop('a', 'Over', 100, 25.5), prop('b', 'Under', 100, 24.5)]), undefined, 'a 25 loses both');
assert.deepEqual(shapes([prop('a', 'Over', 100, 24.5), prop('b', 'Under', 100, 26.5)]), ['loss/win', 'win/loss', 'win/win'], 'middle');
assert.deepEqual(shapes([prop('a', 'Over', 100, 25), prop('b', 'Under', 100, 25)]), ['loss/win', 'push/push', 'win/loss'], 'whole line');
assert.equal(shapes([prop('a', 'Over', 100, 25), prop('b', 'Under', 100, 24.5)]), undefined, 'a 25 pushes one and loses the other');
assert.deepEqual(shapes([prop('a', 'Over', 100, 24), prop('b', 'Under', 100, 24.5)]), ['loss/win', 'push/win', 'win/loss'], 'push beside a win');
assert.deepEqual(shapes([row('a', 'h2h', HOME, 100), row('b', 'h2h', AWAY, 100)]), ['loss/win', 'win/loss'], 'NBA has no ties');
assert.deepEqual(shapes([row('a', 'h2h', HOME, 100), row('b', 'h2h', AWAY, 100)], 'NFL'), ['loss/win', 'push/push', 'win/loss'], 'NFL tie refunds both');
assert.deepEqual(shapes([row('a', 'h2h', HOME, 100), row('b', 'spreads', AWAY, 100, 1.5)], 'NHL'), ['loss/win', 'win/loss', 'win/win'],
  'moneyline and the other team +1.5: a one-goal win wins both');
assert.equal(shapes([row('a', 'h2h', HOME, 100), row('b', 'spreads', AWAY, 100, -1.5)], 'NHL'), undefined);
assert.deepEqual(shapes([row('a', 'spreads', HOME, 100, -2.5), row('b', 'spreads', AWAY, 100, 3.5)]), ['loss/win', 'win/loss', 'win/win']);
assert.equal(shapes([row('a', 'spreads', HOME, 100, -3.5), row('b', 'spreads', AWAY, 100, 2.5)]), undefined);

// ---- Scanning ----------------------------------------------------------------------------
{
  // Over +102 at BetRivers, Under +110 at Bovada: 97.12%. Both have NBA player rules on file (assumed).
  const rows = [prop('betrivers', 'Over', 102), prop('betrivers', 'Under', -130), prop('bovada', 'Over', -140), prop('bovada', 'Under', 110),
    prop('draftkings', 'Over', -105), prop('draftkings', 'Under', -115)];
  const { locked, check, near: close, stats } = scan(rows);
  assert.equal(locked.length, 1); assert.equal(check.length, 0); assert.equal(close.length, 0);
  const p = locked[0];
  assert.deepEqual(p.legs.map(l => [l.book, l.side, l.price]), [['betrivers', 'Over', 102], ['bovada', 'Under', 110]]);
  assert.equal(p.status, 'locked'); assert.equal(p.tier, 'assumed'); assert.equal(p.kind, 'two_way'); assert.equal(p.licensed, false);
  assert.equal(p.id, 'NBA#e1#player_points|Evan Mobley|Over|21.5#player_points|Evan Mobley|Under|21.5');
  near(p.return_pct, 100 * (1 / (100 / 202 + 100 / 210) - 1), 'return');
  assert.equal(stats.quotes, 6); assert.deepEqual(stats.books, ['betrivers', 'bovada', 'draftkings']);
  const st = A.stakes(p, 1000);
  near(st.legs[0].stake + st.legs[1].stake, 1000, 'stakes add to the total');
  near(st.legs[0].stake * 2.02, st.payout, 'Over pays the payout'); near(st.legs[1].stake * 2.1, st.payout, 'Under pays the payout');
  near(st.profit, st.payout - 1000, 'profit'); near(st.best, st.profit, 'no middle');
  // Tax: the winning bet's profit is income; 90% of the losing stake is deductible when itemizing.
  const worst = Math.min(...[[0, 1], [1, 0]].map(([w, l]) => {
    const win = st.legs[w].stake * ([2.02, 2.1][w] - 1), loss = st.legs[l].stake;
    return win - loss - 0.24 * (win - 0.9 * loss);
  }));
  near(A.afterTax(p, 1000), worst, 'after tax, itemizing at 24%');
  const plain = Math.min(...[[0, 1], [1, 0]].map(([w, l]) => {
    const win = st.legs[w].stake * ([2.02, 2.1][w] - 1); return win - st.legs[l].stake - 0.24 * win;
  }));
  near(A.afterTax(p, 1000, { itemize: false }), plain, 'standard deduction');
  assert(A.afterTax(p, 1000, { itemize: false }) < 0 && A.afterTax(p, 1000) > 0);
}
{
  // The cheapest pair has a book with no rules on file; a slightly worse pair at two books on file is still locked.
  const rows = [prop('betrivers', 'Over', 110), prop('mybookieag', 'Under', 105), prop('bovada', 'Under', 100), prop('fanduel', 'Under', -125)];
  const { locked, check } = scan(rows);
  assert.deepEqual(locked.map(p => p.legs.map(l => l.book)), [['betrivers', 'bovada']]);
  assert.deepEqual(check.map(p => p.legs.map(l => l.book)), [['betrivers', 'mybookieag']], 'the better unverified pair is a lead');
  assert.deepEqual(check[0].reasons, [{ code: 'rules_unknown', book: 'mybookieag', market: 'Points' }]);
  assert.equal(check[0].tier, 'unknown');
}
{
  // Quotes far apart, the same book on both sides, started games and contradictory quotes.
  const apart = scan([prop('betrivers', 'Over', 110), prop('bovada', 'Under', 105, 21.5, { quoted_at: '2026-10-22T14:40:00Z' })]);
  assert.equal(apart.locked.length, 0); assert.deepEqual(apart.check[0].reasons, [{ code: 'quotes_apart', seconds: 1080 }]);
  assert.deepEqual(scan([prop('betrivers', 'Over', 110), prop('betrivers', 'Under', 105)]).check, [], 'one book is not a pair');
  const started = [prop('betrivers', 'Over', 110), prop('bovada', 'Under', 105)].map(r => ({ ...r, commence_time: '2026-10-22T14:59:00Z' }));
  assert.deepEqual(scan(started).locked, [], 'games already started are skipped');
  const conflict = scan([prop('betrivers', 'Over', 110), prop('betrivers', 'Over', -120), prop('bovada', 'Under', 105)]);
  assert.deepEqual(conflict.locked, [], 'two prices for one offer at one moment fail closed');
  const newest = scan([prop('betrivers', 'Over', 110, 21.5, { quoted_at: '2026-10-22T14:50:00Z' }), prop('betrivers', 'Over', -120), prop('bovada', 'Under', 105)]);
  assert.deepEqual(newest.locked, [], 'the newest quote replaces an older one');
}
{
  // MLB props have no rules on file; offshore game lines don't either. NHL uses its own config.
  const mlb = (book, market, side, price, line) => row(book, market, side, price, line, { player: market === 'h2h' || market === 'totals' ? '' : 'J.P. Crawford',
    home_team: 'Seattle Mariners', away_team: 'Los Angeles Angels', game: 'Los Angeles Angels @ Seattle Mariners' });
  const props = scan([mlb('betrivers', 'batter_rbis', 'Over', 370, 0.5), mlb('betmgm', 'batter_rbis', 'Under', -295, 0.5)], 'MLB');
  assert.equal(props.locked.length, 0); assert.equal(props.check[0].reasons.length, 2);
  const lines = scan([mlb('draftkings', 'totals', 'Over', 105, 8.5), mlb('fanduel', 'totals', 'Under', 105, 8.5)], 'MLB');
  assert.equal(lines.locked.length, 1); assert.equal(lines.locked[0].licensed, true); assert.equal(lines.locked[0].legs[0].settlement_profile, 'mlb_game_action');
  const offshore = scan([mlb('draftkings', 'totals', 'Over', 105, 8.5), mlb('bovada', 'totals', 'Under', 105, 8.5)], 'MLB');
  assert.equal(offshore.check.length, 1);
  const nhl = scan([row('draftkings', 'player_points', 'Over', 105, 0.5, { player: 'A Skater' }), row('betmgm', 'player_points', 'Under', 105, 0.5, { player: 'A Skater' }),
    row('draftkings', 'h2h', HOME, 110), row('betonlineag', 'h2h', AWAY, 110)], 'NHL');
  assert.deepEqual(nhl.locked.map(p => p.subject), ['A Skater · Points']);
  assert.deepEqual(nhl.check[0].reasons, [{ code: 'rules_unknown', book: 'betonlineag', market: 'Moneyline' }], 'BetOnline NHL game rules are not on file');
}
{
  // Middles and near misses.
  const { locked } = scan([prop('fanduel', 'Over', 100, 24.5), prop('draftkings', 'Under', 102, 26.5)]);
  assert.equal(locked[0].kind, 'middle'); assert.deepEqual(locked[0].middle, { team: null, from: 25, to: 26 });
  const st = A.stakes(locked[0], 100); assert(st.best > st.profit && st.profit > 0);
  const nfl = scan([row('draftkings', 'h2h', HOME, 105), row('fanduel', 'h2h', AWAY, -102)], 'NFL');
  assert.equal(nfl.locked.length, 1); assert.equal(nfl.locked[0].push_refund, true);
  near(A.stakes(nfl.locked[0], 100).profit, 100 * (1 / nfl.locked[0].implied_total - 1), 'a tie refunds both; profit is for settled results');
  const close = scan([prop('fanduel', 'Over', -105), prop('draftkings', 'Under', 102)]);
  assert.equal(close.locked.length + close.check.length, 0); assert.equal(close.near[0].status, 'near');
  assert.deepEqual(scan([prop('fanduel', 'Over', -105, 24.5), prop('draftkings', 'Under', 100, 26.5)]).near, [], 'middles over 100% are not near misses');
}

// ---- Scheduled scan and paper ledger ----------------------------------------------------
{
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fv-arb-'));
  fs.mkdirSync(path.join(root, 'docs/arbitrage/data'), { recursive: true }); fs.mkdirSync(path.join(root, 'docs/nba/data'), { recursive: true });
  fs.writeFileSync(path.join(root, 'docs/arbitrage/data/rules.json'), JSON.stringify(RULES));
  const feed = (at, rows) => fs.writeFileSync(path.join(root, 'docs/nba/data/latest.json'), JSON.stringify({ status: 'ready', last_success_at: at, rows }));
  const pair = [prop('betrivers', 'Over', 102), prop('bovada', 'Under', 110)];
  feed('2026-10-22T14:58:30Z', pair);
  const out = run({ sport: 'NBA', root, now: NOW });
  assert.equal(out.locked.length, 1); assert.equal(out.locked[0].start, undefined, 'internal times are not published');
  const saved = JSON.parse(fs.readFileSync(path.join(root, 'docs/arbitrage/data/nba.json'), 'utf8'));
  assert.equal(saved.sport, 'NBA'); assert.equal(saved.snapshot_at, '2026-10-22T14:58:30Z'); assert.equal(saved.scanned_at, '2026-10-22T15:00:00Z');
  const ledger = () => JSON.parse(fs.readFileSync(path.join(root, 'docs/arbitrage/data/ledger/nba.json'), 'utf8'));
  assert.equal(ledger().entries[0].snapshots, 1);
  run({ sport: 'NBA', root, now: NOW + 60e3 });
  assert.equal(ledger().entries[0].snapshots, 1, 'the same snapshot is never counted twice');
  feed('2026-10-22T15:30:00Z', pair.map(r => ({ ...r, quoted_at: '2026-10-22T15:29:00Z', price: r.side === 'Over' ? 105 : 110 })));
  run({ sport: 'NBA', root, now: Date.parse('2026-10-22T15:31:00Z') });
  const e = ledger().entries[0];
  assert.deepEqual([e.snapshots, e.first_seen, e.last_seen], [2, '2026-10-22T14:58:30Z', '2026-10-22T15:30:00Z']);
  assert.equal(e.first_legs[0].price, 102, 'first-seen prices are kept'); assert(e.best_return_pct > e.first_return_pct);
  run({ sport: 'NBA', root, now: Date.parse('2027-06-01T00:00:00Z') });
  feed('2027-06-01T00:00:00Z', []); run({ sport: 'NBA', root, now: Date.parse('2027-06-01T00:00:00Z') });
  assert.deepEqual(ledger().entries, [], 'entries older than 180 days are dropped');
  fs.rmSync(path.join(root, 'docs/nba/data/latest.json'));
  assert.equal(run({ sport: 'NBA', root, now: NOW }).feed_status, 'missing');
  assert.throws(() => run({ sport: 'WNBA', root }), /Unknown sport/);
  fs.rmSync(root, { recursive: true, force: true });
}

// ---- Scan now through the live-odds relay ---------------------------------------------------
(async () => {
  const { createHandler, SPORTS, LINES } = await import('../supabase/functions/live-odds/handler.mjs');
  const ID = 'c'.repeat(32), calls = [];
  let left = 9000;
  const fetchImpl = async (url) => {
    if (url.startsWith('https://db.example/auth/v1/user')) return new Response(JSON.stringify({ app_metadata: { fv_editor: true } }), { status: 200 });
    calls.push(url);
    left -= 3;
    return new Response(JSON.stringify([{ id: ID, sport_key: 'basketball_nba', commence_time: '2026-10-22T23:10:00Z', home_team: HOME, away_team: AWAY, bookmakers: [] },
      { id: 'bad' }]), { status: 200, headers: { 'x-requests-remaining': String(left), 'x-requests-last': '3' } });
  };
  let clock = NOW;
  const handle = createHandler({ fetchImpl, now: () => clock, env: n => ({ SUPABASE_URL: 'https://db.example', SUPABASE_ANON_KEY: 'anon', ODDS_API_KEY: 'k'.repeat(16) })[n] });
  const req = body => handle(new Request('https://edge/', { method: 'POST', headers: { origin: 'https://fourthandvalue.com', authorization: 'Bearer t' }, body: JSON.stringify(body) }));
  for (const bad of [{ scan: 'all' }, { scan: 'lines', event: ID }, { scan: 'lines', more: 'yes' }, { sport: 'WNBA', scan: 'lines' }])
    assert.equal((await req(bad)).status, 400, JSON.stringify(bad));
  assert.equal(calls.length, 0);
  const first = await (await req({ sport: 'NBA', scan: 'lines' })).json();
  const u = new URL(calls[0]);
  assert.equal(u.pathname, '/v4/sports/basketball_nba/odds');
  assert.deepEqual(['regions', 'markets', 'oddsFormat', 'commenceTimeFrom', 'commenceTimeTo'].map(k => u.searchParams.get(k)),
    ['us', LINES.join(','), 'american', '2026-10-22T15:00:00Z', '2026-10-30T15:00:00Z']);
  assert.deepEqual([first.sport, first.scan, first.regions, first.cost, first.reused, first.remaining], ['NBA', 'lines', 'us', 3, false, 8997]);
  assert.deepEqual(first.events.map(e => e.id), [ID], 'only valid event ids');
  assert.equal((await (await req({ sport: 'NBA', scan: 'lines' })).json()).reused, true); assert.equal(calls.length, 1, 'reused for 60 seconds');
  const more = await (await req({ sport: 'NBA', scan: 'lines', more: true })).json();
  assert.equal(new URL(calls[1]).searchParams.get('regions'), 'us,us2'); assert.equal(more.regions, 'us,us2');
  await req({ sport: 'NFL', event: ID, more: true });
  const ev = new URL(calls[2]);
  assert.equal(ev.pathname, `/v4/sports/americanfootball_nfl/events/${ID}/odds`); assert.equal(ev.searchParams.get('regions'), 'us,us2');
  assert.equal(ev.searchParams.get('markets'), SPORTS.NFL.markets.join(','));
  for (const s of ['NFL', 'MLB']) assert.equal(SPORTS[s].markets.length, 7, `${s}: at most 7 credits a game`);
  await req({ sport: 'MLB', scan: 'lines' }); assert.equal(new URL(calls[3]).pathname, '/v4/sports/baseball_mlb/odds');
  left = 1500; clock += 61e3;
  const paused = await req({ sport: 'NHL', scan: 'lines' });
  assert.equal(paused.status, 200, 'the reserve is learned from a response');
  clock += 61e3;
  assert.equal((await req({ sport: 'NHL', scan: 'lines' })).status, 429, 'paid scans pause below the reserve');
  console.log('PASS: arbitrage engine, scheduled scan and ledger, and Scan now relay.');
})().catch(error => { console.error(error); process.exit(1); });

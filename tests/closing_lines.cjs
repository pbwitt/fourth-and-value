// closing-lines: market mapping, exact-line closing prices, and the credit floor.
// Fixtures mirror Odds API and PostgREST shapes; nothing leaves the process.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const load = file => import(pathToFileURL(path.join(__dirname, '../supabase/functions/closing-lines', file)).href);
const near = (a, b, msg) => assert(Math.abs(a - b) < 1e-9, `${msg}: ${a} vs ${b}`);

(async () => {
  const C = await load('clv.mjs');
  const H = await load('handler.mjs');

  // ---- Market mapping ------------------------------------------------------------
  assert.deepEqual(C.specFor({ league: 'NHL', market_type: 'totals', line: 6.5 }),
    { sport: 'icehockey_nhl', key: 'totals', kind: 'total' }, 'new NHL tickets record game totals as totals');
  assert.deepEqual(C.specFor({ league: 'NHL', market_type: 'team_total', line: 6.5 }),
    { sport: 'icehockey_nhl', key: 'totals', kind: 'total' }, 'legacy team_total rows are game totals');
  assert.equal(C.specFor({ league: 'NHL', market_type: 'sog', player: 'Tage Thompson', line: 3.5 }).key, 'player_shots_on_goal');
  for (const [label, key] of Object.entries({ points: 'player_points', pts: 'player_points', reb: 'player_rebounds',
    '3pm': 'player_threes', pra: 'player_points_rebounds_assists', stocks: 'player_blocks_steals' })) {
    assert.equal(C.specFor({ league: 'NBA', market_type: label, player: 'A B', line: 1.5 }).key, key, label);
  }
  assert.match(C.specFor({ league: 'NHL', market_type: 'sog', line: 2.5 }).error, /without a player/);
  assert.match(C.specFor({ league: 'NFL', market_type: 'receptions', player: 'A B' }).error, /no line/);
  assert.match(C.specFor({ league: 'CFL', market_type: 'h2h' }).error, /not tracked/);
  assert.equal(C.specFor({ league: 'MLB', market_type: 'h2h' }).kind, 'h2h', 'moneylines need no line');

  // ---- Names, books and games -------------------------------------------------------
  assert.equal(C.norm('Alexis Lafrenière'), 'alexis lafreniere');
  assert.equal(C.norm('J.T. Miller'), 'jt miller');
  assert.equal(C.norm('Marvin Harrison Jr.'), 'marvin harrison');
  assert.equal(C.bookKey('Caesars'), 'williamhill_us');
  assert.equal(C.bookKey('DraftKings'), 'draftkings');
  const events = [
    { id: 'e1', commence_time: '2026-10-04T00:10:00Z', home_team: 'Buffalo Sabres', away_team: 'Chicago Blackhawks' },
    { id: 'e2', commence_time: '2026-10-04T02:30:00Z', home_team: 'Los Angeles Clippers', away_team: 'Denver Nuggets' },
  ];
  assert.equal(C.matchEvent({ game_date: '2026-10-03', team_home: 'Buffalo Sabres', team_away: 'Chicago Blackhawks' }, events).event.id,
    'e1', 'an 8:10 PM Eastern start belongs to the bet date');
  assert.equal(C.matchEvent({ game_date: '2026-10-03', team_home: 'LA Clippers', team_away: 'Denver Nuggets' }, events).event.id,
    'e2', 'nicknames match when full names differ');
  assert.match(C.matchEvent({ game_date: '2026-10-05', team_home: 'Buffalo Sabres', team_away: 'Chicago Blackhawks' }, events).error, /No Chicago/);

  // ---- Closing prices at the bet's exact line ---------------------------------------------
  const sog = (key, over, under, point = 3.5, player = 'Tage Thompson') => ({ key, markets: [{ key: 'player_shots_on_goal',
    outcomes: [{ name: 'Over', description: player, price: over, point }, { name: 'Under', description: player, price: under, point }] }] });
  const spec = C.specFor({ league: 'NHL', market_type: 'sog', player: 'Tage Thompson', line: 3.5 });
  const bet = { side: 'under', line: 3.5, player: 'Tage Thompson', book: 'fanduel', odds: -128 };
  const close = C.closeForBet(bet, spec, { bookmakers: [sog('fanduel', 100, -120), sog('draftkings', -110, -110), sog('betmgm', 105, -125)] });
  assert.equal(close.clv_status, 'captured');
  assert.equal(close.closing_books, 3);
  assert.equal(close.closing_odds, -120, "the bettor's own book's closing price");
  // Under no-vig per book: FanDuel 0.5217, DraftKings 0.5, BetMGM 0.5325; the median is FanDuel's.
  const fair = (u, o) => C.implied(u) / (C.implied(u) + C.implied(o));
  near(close.closing_fair_prob, Math.round(fair(-120, 100) * 1e4) / 1e4, 'median of per-book no-vig probabilities');
  const moved = C.closeForBet(bet, spec, { bookmakers: [sog('fanduel', -110, -110, 2.5), sog('draftkings', -110, -110, 2.5)] });
  assert.equal(moved.clv_status, 'line_moved');
  assert.equal(moved.closing_line, 2.5);
  const oneSided = C.closeForBet(bet, spec, { bookmakers: [{ key: 'fanduel', markets: [{ key: 'player_shots_on_goal',
    outcomes: [{ name: 'Under', description: 'Tage Thompson', price: -120, point: 3.5 }] }] }] });
  assert.equal(oneSided.clv_status, 'no_market');
  assert.equal(C.closeForBet(bet, spec, { bookmakers: [] }).clv_status, 'no_market');

  // ---- Credit floor and per-run cap ----------------------------------------------------------
  assert.equal(H.CREDIT_FLOOR, 2000);
  assert.equal(H.RUN_CAP, 2000);
  assert.equal(H.estimateCost('sports/icehockey_nhl/events', {}, false), 0, 'live event lists are free');
  assert.equal(H.estimateCost('historical/sports/icehockey_nhl/events', { date: 'x' }, true), 1);
  assert.equal(H.estimateCost('sports/icehockey_nhl/events/e1/odds', { markets: 'player_points,totals', regions: 'us' }, true), 2);
  assert.equal(H.estimateCost('historical/sports/icehockey_nhl/events/e1/odds', { markets: 'totals', regions: 'us' }, true), 10);

  const NOW = Date.parse('2026-10-05T23:05:00Z');
  const game = (id, eventId, home, away) => ({ id, league: 'NHL', market_type: 'sog', side: 'under', line: 3.5,
    player: 'Tage Thompson', book: 'fanduel', odds: -128, game_date: '2026-10-05', team_home: home, team_away: away,
    event_id: eventId, commence_time: '2026-10-05T23:10:00Z' });
  const twoGames = [game('b1', 'e9', 'Buffalo Sabres', 'Chicago Blackhawks'), game('b2', 'e8', 'Boston Bruins', 'New York Rangers')];
  // Two games start within 6 minutes, so one run makes two paid requests (1 credit each).
  function harness({ remaining, last = 1, env = {} }) {
    const calls = { odds: 0, patches: [], snapshots: 0 };
    const json = (body, headers = {}) => new Response(JSON.stringify(body), { status: 200, headers });
    const fetchImpl = async (url, init = {}) => {
      if (url.includes('/rest/v1/rpc/closing_lines_secret_ok')) return json(true);
      if (url.includes('/rest/v1/rpc/live_odds_key')) return json('test-key');
      if (url.includes('/rest/v1/bets?select=')) return json(twoGames);
      if (url.includes('/rest/v1/bets?id=')) { calls.patches.push(JSON.parse(init.body)); return new Response(null, { status: 204 }); }
      if (url.includes('/rest/v1/odds_snapshots')) { calls.snapshots++; return new Response(null, { status: 201 }); }
      if (url.startsWith('https://api.the-odds-api.com/')) {
        calls.odds++;
        const id = url.match(/events\/([^/]+)\/odds/)[1];
        return json({ id, bookmakers: [sog('fanduel', 100, -120), sog('draftkings', -110, -110)] },
          { 'x-requests-remaining': String(remaining), 'x-requests-last': String(last) });
      }
      throw new Error(`unexpected request ${url}`);
    };
    const handle = H.createHandler({ fetchImpl, now: () => NOW,
      env: name => ({ SUPABASE_URL: 'https://db.example', SUPABASE_SERVICE_ROLE_KEY: 'service', ...env })[name] });
    const post = async body => (await handle(new Request('https://fn.example/closing-lines',
      { method: 'POST', headers: { 'x-closing-secret': 's'.repeat(32) }, body: JSON.stringify(body) }))).json();
    return { calls, post };
  }

  // A healthy balance: both games priced, both closes written.
  {
    const { calls, post } = harness({ remaining: 15000 });
    const out = await post({});
    assert.equal(calls.odds, 2);
    assert.equal(calls.snapshots, 2);
    assert.equal(out.results.captured, 2);
    assert.equal(out.credits_left, 15000);
    assert.equal(out.stopped, undefined);
  }
  // At 2,000 left the next 1-credit request would cross the floor, so it is refused.
  {
    const { calls, post } = harness({ remaining: 2000 });
    const out = await post({});
    assert.equal(calls.odds, 1, 'the first request reveals the balance; the second is refused');
    assert.match(out.stopped, /below the 2000 reserve/);
  }
  // A lower reserve setting is ignored; a higher one is honored.
  {
    const { calls, post } = harness({ remaining: 2000, env: { CLOSING_LINES_RESERVE: '100' } });
    assert.match((await post({})).stopped, /below the 2000 reserve/);
    assert.equal(calls.odds, 1);
  }
  {
    const { calls, post } = harness({ remaining: 2500, env: { CLOSING_LINES_RESERVE: '3000' } });
    assert.match((await post({})).stopped, /below the 3000 reserve/);
    assert.equal(calls.odds, 1);
  }
  // One run may not spend more than 2,000 credits.
  {
    const { calls, post } = harness({ remaining: 15000, last: 2000 });
    assert.match((await post({})).stopped, /per-run cap is 2000/);
    assert.equal(calls.odds, 1);
  }
  // Unauthorized callers never reach the odds service.
  {
    const handle = H.createHandler({ fetchImpl: async url => {
      if (url.includes('closing_lines_secret_ok')) return new Response('false', { status: 200 });
      throw new Error('no other request expected');
    }, env: name => ({ SUPABASE_URL: 'https://db.example', SUPABASE_SERVICE_ROLE_KEY: 'service' })[name] });
    const res = await handle(new Request('https://fn.example', { method: 'POST', headers: { 'x-closing-secret': 'wrong' }, body: '{}' }));
    assert.equal(res.status, 401);
  }

  console.log('closing_lines: all checks passed');
})().catch(error => { console.error(error); process.exit(1); });

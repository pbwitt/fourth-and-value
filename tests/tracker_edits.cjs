const assert = require('node:assert/strict');
const E = require('../docs/tracking/edit-bet.js');
const { payout, gradeTarget } = require('../scripts/grade_bets.cjs');
const original = { id: 'ticket', user_id: 'owner', league: 'MLB', game_date: '2026-10-08', team_home: 'SD', team_away: 'CHC',
  player: null, market_type: 'totals', side: 'under', line: 7.5, odds: -110, stake_dollars: 25, book: 'DraftKings',
  status: 'won', actual_result: 5, payout: 47.73, graded_timestamp: '2026-10-09T06:00:00Z', model_prob: 0.6, edge_bps: 760 };

// The motivating correction: twice the stake, same winning bet, corrected return.
let row = E.buildUpdate(original, { stake_dollars: '50' });
assert.equal(row.stake_dollars, 50); assert.equal(row.status, 'won'); assert.equal(row.payout, 95.45);
assert.equal(row.actual_result, 5); assert(!Object.hasOwn(row, 'graded_timestamp'));
assert(!Object.hasOwn(row, 'model_prob')); assert(!Object.hasOwn(row, 'edge_bps'));
for (const status of ['pending', 'won', 'lost', 'push']) {
  for (const odds of [-115, 150]) {
    row = E.buildUpdate({ ...original, status }, { stake_dollars: 50, odds });
    assert.equal(row.payout, status === 'pending' ? null : payout(50, odds, status));
    assert.equal(row.status, status);
  }
}
assert.equal(E.buildUpdate(original, { odds: 150 }).payout, 62.5);
assert.equal(E.buildUpdate(original, { odds: 150 }).edge_bps, null);
assert(!Object.hasOwn(E.buildUpdate(original, { odds: 150 }), 'model_prob'));
assert.equal(E.buildUpdate({ ...original, payout: 48 }, { book: 'FanDuel' }).payout, 48, 'keep custom return on unrelated edits');
assert.equal(E.buildUpdate(original, { stake_dollars: 50, payout: 100 }).payout, 100, 'explicit sportsbook payout');
assert.equal(E.buildUpdate(original, { status: 'push', payout: 100 }).payout, 25);
assert.equal(E.buildUpdate(original, { status: 'lost', payout: 100 }).payout, 0);
row = E.buildUpdate(original, { status: 'pending' });
assert.equal(row.payout, null); assert.equal(row.actual_result, null); assert.equal(row.graded_timestamp, null);
row = E.buildUpdate({ ...original, status: 'pending', payout: null, actual_result: null }, { status: 'won' }, 'now');
assert.equal(row.payout, 47.73); assert.equal(row.graded_timestamp, 'now'); assert.equal(row.actual_result, null);

// Changed selection loses the old result and model; zero lines and no-line bets survive.
for (const change of [{ line: 0 }, { side: 'over' }, { player: 'Player' }, { game_date: '2026-10-07' },
  { league: 'NHL' }, { market_type: 'spreads' }, { team_home: 'NYY' }, { team_away: 'BOS' }]) {
  row = E.buildUpdate(original, change);
  assert.equal(row.status, 'pending'); assert.equal(row.payout, null); assert.equal(row.actual_result, null);
  assert.equal(row.graded_timestamp, null); assert.equal(row.model_prob, null); assert.equal(row.edge_bps, null);
}
assert.equal(E.buildUpdate(original, { line: 0 }).line, 0);
assert.equal(E.buildUpdate({ ...original, player_team: 'SD' }, { player: 'A different player' }).player_team, null);
assert(!Object.hasOwn(E.buildUpdate({ ...original, player_team: 'SD' }, { stake_dollars: 50 }), 'player_team'));
row = E.buildUpdate({ ...original, market_type: 'h2h', line: null }, { line: '', stake_dollars: 50 });
assert.equal(row.line, null); assert.equal(row.status, 'won');
for (const change of [{ stake_dollars: 0 }, { stake_dollars: -10 }, { stake_dollars: 'NaN' }, { stake_dollars: 0.001 },
  { odds: 50 }, { odds: 110.5 }, { odds: '' }, { payout: -1 }, { line: 'nope' }, { actual_result: 'nope' },
  { status: 'invalid' }, { game_date: '2026-02-30' }, { game_date: 'bad' }]) {
  assert.throws(() => E.buildUpdate(original, change), Error, JSON.stringify(change));
}
row = E.buildUpdate(original, { id: 'other', user_id: 'intruder', model_prob: 1, created_at: 'now' });
for (const key of ['id', 'user_id', 'model_prob', 'created_at']) assert(!Object.hasOwn(row, key));

// The grader's atomic filter rejects changed stakes, odds and selections, including nullable fields.
const target = new URL(gradeTarget(original), 'https://example.com/').searchParams;
for (const key of ['league', 'game_date', 'team_home', 'team_away', 'market_type', 'side', 'line', 'odds', 'stake_dollars']) {
  assert.equal(target.get(key), `eq.${original[key]}`);
}
assert.equal(target.get('status'), 'eq.pending'); assert.equal(target.get('player'), 'is.null');
assert(!target.has('player_team'), 'work before the optional player-team migration');
assert.equal(new URL(gradeTarget({ ...original, player_team: null }), 'https://example.com/').searchParams.get('player_team'), 'is.null');
assert(!new URL(gradeTarget(original, null), 'https://example.com/').searchParams.has('status'), 'backfill supplies its own settled-status guard');
const special = new URL(gradeTarget({ ...original, player: 'A & B + C' }), 'https://example.com/');
assert.equal(special.searchParams.get('player'), 'eq.A & B + C');

(async () => {
  const state = { user: { id: 'owner' }, writes: [], data: original, error: null, filters: [] };
  global.getCurrentUser = async () => state.user;
  const query = {
    eq(key, value) { state.filters.push([key, value]); return query; },
    is(key, value) { state.filters.push([key, value]); return query; },
    select() { return query; },
    async maybeSingle() { return { data: state.data, error: state.error }; },
  };
  global.supabaseClient = { from: () => ({ update(row) { state.writes.push(row); return query; } }) };
  state.user = null; assert.equal((await E.save(original, { stake_dollars: 50 })).ok, false);
  state.user = { id: 'intruder' }; assert.equal((await E.save(original, { stake_dollars: 50 })).ok, false);
  assert.equal(state.writes.length, 0);
  state.user = { id: 'owner' };
  assert.equal((await E.save(original, { stake_dollars: 0 })).ok, false); assert.equal(state.writes.length, 0);
  assert.equal((await E.save(original, { stake_dollars: 50 })).ok, true);
  assert.equal(state.writes[0].payout, 95.45);
  for (const key of Object.keys(original)) assert(state.filters.some(([k, v]) => k === key && v === original[key]), key);
  state.data = null; assert.equal((await E.save(original, { stake_dollars: 50 })).conflict, true);
  state.error = { message: 'network unavailable' }; assert.equal((await E.save(original, { stake_dollars: 50 })).ok, false);
  console.log('PASS: bet edits validate money, recalculate returns, preserve ownership, clear stale results and guard concurrent edits/grading.');
})().catch(error => { console.error(error); process.exitCode = 1; });

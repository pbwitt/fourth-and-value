#!/usr/bin/env node
/* Settle pending Bet Tracker bets from final box scores, for every league the
   tracker's live stats support (NHL, MLB, NFL, NBA).

   Uses the same feed adapters and verdicts as the live view
   (docs/tracking/live-feeds.js, live-stats.js), so a bet is graded exactly
   as the tracker showed it at the final horn.

   A bet is settled only when its game is final and started at least
   SETTLE_AFTER_HOURS ago (time for stat corrections). Anything that cannot be
   matched with certainty (unknown market, player not in the box score,
   postponed game, several possible games) stays pending.

   The repository is public, so logs carry counts only: never players,
   stakes or account ids.

   Settling a player bet also records the player's team from the box score.
   --backfill-teams instead adds the team to already-settled player bets.

   Usage: SUPABASE_SERVICE_ROLE_KEY=... node scripts/grade_bets.cjs [--dry-run] [--backfill-teams]
*/
'use strict';
const { execFile } = require('node:child_process');
const feeds = require('../docs/tracking/live-feeds.js');
const L = require('../docs/tracking/live-stats.js');

const SUPABASE_URL = (process.env.SUPABASE_URL || 'https://fzjonxpzsrbdhbujbhsn.supabase.co').replace(/\/$/, '');
const SETTLE_AFTER_HOURS = 4;
const LOOKBACK_DAYS = 14;
const NHL = 'https://api-web.nhle.com/v1';

// Feeds are read with curl: ESPN refuses Node's built-in HTTP client.
function curlJSON(url) {
  return new Promise((resolve, reject) => {
    execFile('curl', ['-sSfL', '--compressed', '--max-time', '25', '-H', 'Accept: application/json', url],
      { maxBuffer: 20e6 }, (error, stdout) => {
        if (error) return reject(new Error(`Feed request failed (${new URL(url).host})`));
        try { resolve(JSON.parse(stdout)); } catch { reject(new Error(`Feed returned invalid JSON (${new URL(url).host})`)); }
      });
  });
}

// The browser reaches NHL through the live-stats relay; a server can call it directly.
const nhlDirect = body => curlJSON(body.game != null ? `${NHL}/gamecenter/${body.game}/boxscore` : `${NHL}/score/${body.date}`);

function payout(stake, odds, tone) {
  stake = Number(stake); odds = Number(odds);
  if (tone === 'push') return stake;
  if (tone === 'lost') return 0;
  const decimal = odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
  return Math.round(stake * decimal * 100) / 100;
}

// Pure: one bet + its final game/box -> the fields to write, or a skip reason.
function gradeBet(bet, game, box, now) {
  if (!game) return { skip: 'game not found' };
  if (game.state === 'off') return { skip: 'postponed' };
  const view = L.evaluate(bet, game, box);
  if (view.status !== 'final') return { skip: 'not final' };
  const started = Date.parse(view.game.start);
  if (!Number.isFinite(started) || now - started < SETTLE_AFTER_HOURS * 3600e3) return { skip: 'waiting for corrections' };
  if (!['won', 'lost', 'push'].includes(view.tone)) {
    return { skip: view.kind === 'unsupported' ? 'unsupported market' : view.tone === 'missing' ? 'player not found' : 'cannot grade' };
  }
  const odds = Number(bet.odds);
  if (!Number.isFinite(odds) || Math.abs(odds) < 100) return { skip: 'invalid odds' };
  const actual = view.value ?? view.margin;
  const update = {
    status: view.tone,
    actual_result: Number.isFinite(actual) ? actual : null,
    payout: payout(bet.stake_dollars, odds, view.tone),
    graded_timestamp: new Date(now).toISOString(),
  };
  // The box score shows which side the player was on; keep it for the tracker.
  if (bet.player && !bet.player_team && view.playerTeam) update.player_team = view.playerTeam;
  return { update };
}

// Pure: a settled player bet + its final game/box -> the player's team code, or a skip reason.
function teamFor(bet, game, box) {
  if (!game) return { skip: 'game not found' };
  if (game.state !== 'final') return { skip: 'not final' };
  const player = box && L.findPlayer(bet.player, box.players);
  const team = player && L.teamCode(game[player.side]);
  return team ? { team } : { skip: 'player not found' };
}

// Older settled bets predate the player_team column: find each player's team
// in the final box score. No lookback limit; the feeds keep past seasons.
async function backfillTeams({ bets, fetchJSON, proxy }) {
  const run = plan => Promise.all(plan.requests.map(r => r.proxy ? proxy(r.proxy) : fetchJSON(r.url))).then(plan.build);
  const results = [], boards = new Map(), boxes = new Map();
  for (const bet of bets) {
    const date = String(bet.game_date || '');
    if (!feeds.LEAGUES.includes(bet.league)) { results.push({ bet, skip: 'league not supported' }); continue; }
    if (!date || !bet.player) { results.push({ bet, skip: 'no player or date' }); continue; }
    const dayKey = `${bet.league}|${date}`;
    if (!boards.has(dayKey)) boards.set(dayKey, run(feeds.scoreboardPlan(bet.league, date)).catch(() => null));
    const games = await boards.get(dayKey);
    if (!games) { results.push({ bet, skip: 'feed unavailable' }); continue; }
    const game = L.findGame(bet, games);
    let box = null;
    if (game && game.state === 'final') {
      const key = `${game.league}:${game.id}`;
      if (!boxes.has(key)) boxes.set(key, run(feeds.boxPlan(game.league, game.id)).catch(() => null));
      box = await boxes.get(key);
      if (!box) { results.push({ bet, skip: 'feed unavailable' }); continue; }
    }
    results.push({ bet, ...teamFor(bet, game, box) });
  }
  return results;
}

async function gradeAll({ bets, fetchJSON, proxy, now }) {
  const run = plan => Promise.all(plan.requests.map(r => r.proxy ? proxy(r.proxy) : fetchJSON(r.url))).then(plan.build);
  const oldest = L.day(now - LOOKBACK_DAYS * 864e5), today = L.day(now);
  const results = [], boards = new Map(), boxes = new Map();

  for (const bet of bets) {
    const date = String(bet.game_date || '');
    if (!feeds.LEAGUES.includes(bet.league)) { results.push({ bet, skip: 'league not supported' }); continue; }
    if (!date || date > today) { results.push({ bet, skip: 'not played yet' }); continue; }
    if (date < oldest) { results.push({ bet, skip: 'older than lookback' }); continue; }

    const dayKey = `${bet.league}|${date}`;
    if (!boards.has(dayKey)) boards.set(dayKey, run(feeds.scoreboardPlan(bet.league, date)).catch(() => null));
    const games = await boards.get(dayKey);
    if (!games) { results.push({ bet, skip: 'feed unavailable' }); continue; }
    const game = L.findGame(bet, games);

    let box = null;
    if (game && game.state === 'final' && bet.player) {
      const key = `${game.league}:${game.id}`;
      if (!boxes.has(key)) boxes.set(key, run(feeds.boxPlan(game.league, game.id)).catch(() => null));
      box = await boxes.get(key);
      if (!box) { results.push({ bet, skip: 'feed unavailable' }); continue; }
    }
    results.push({ bet, ...gradeBet(bet, game, box, now) });
  }
  return results;
}

async function supabase(path, init = {}) {
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const res = await fetch(`${SUPABASE_URL}/rest/v1/${path}`, { ...init, headers: {
    apikey: key, Authorization: `Bearer ${key}`, 'Content-Type': 'application/json', ...init.headers } });
  if (!res.ok) {
    // PostgREST names a missing column in its message; nothing else is kept.
    const body = await res.json().catch(() => ({}));
    const error = new Error(`Supabase ${res.status}`);
    error.missingTeamColumn = /player_team/.test(String(body.message || ''));
    throw error;
  }
  return res.status === 204 ? null : res.json();
}

const BET_FIELDS = 'id,league,game_date,team_home,team_away,player,market_type,side,line,odds,stake_dollars';
const MIGRATION_HINT = 'The bets table has no player_team column yet: run the alter table line in supabase/schema.sql.';

async function backfillMain(dryRun) {
  let bets;
  try {
    bets = await supabase(`bets?status=in.(won,lost,push)&player=not.is.null&player_team=is.null&select=${BET_FIELDS}`);
  } catch (error) {
    throw error.missingTeamColumn ? new Error(MIGRATION_HINT) : error;
  }
  const results = await backfillTeams({ bets, fetchJSON: curlJSON, proxy: nhlDirect });
  let written = 0;
  const tally = {};
  for (const r of results) {
    const k = `${r.bet.league} ${r.team ? 'team found' : 'skipped: ' + r.skip}`;
    tally[k] = (tally[k] || 0) + 1;
    if (!r.team || dryRun) continue;
    const rows = await supabase(`bets?id=eq.${encodeURIComponent(r.bet.id)}&player_team=is.null`,
      { method: 'PATCH', headers: { Prefer: 'return=representation' }, body: JSON.stringify({ player_team: r.team }) });
    written += rows.length;
  }
  console.log(`${dryRun ? 'Dry run: ' : ''}${bets.length} settled player bets without a team checked.`);
  for (const [k, n] of Object.entries(tally).sort()) console.log(`  ${k}: ${n}`);
  if (!dryRun) console.log(`Added the team to ${written}.`);
}

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) throw new Error('SUPABASE_SERVICE_ROLE_KEY is not set.');
  if (process.argv.includes('--backfill-teams')) return backfillMain(dryRun);
  const now = Date.now();
  // Before the player_team migration, grade exactly as before and save no team.
  let teamColumn = true, bets;
  try { bets = await supabase(`bets?status=eq.pending&select=${BET_FIELDS},player_team`); }
  catch (error) {
    if (!error.missingTeamColumn) throw error;
    teamColumn = false;
    console.log(MIGRATION_HINT);
    bets = await supabase(`bets?status=eq.pending&select=${BET_FIELDS}`);
  }
  const results = await gradeAll({ bets, fetchJSON: curlJSON, proxy: nhlDirect, now });
  if (!teamColumn) for (const r of results) if (r.update) delete r.update.player_team;

  let written = 0, raced = 0;
  const tally = {};
  for (const r of results) {
    const k = `${r.bet.league} ${r.update ? r.update.status : 'pending: ' + r.skip}`;
    tally[k] = (tally[k] || 0) + 1;
    if (!r.update || dryRun) continue;
    // Only a still-pending row changes, so a manual edit is never overwritten.
    const rows = await supabase(`bets?id=eq.${encodeURIComponent(r.bet.id)}&status=eq.pending`,
      { method: 'PATCH', headers: { Prefer: 'return=representation' }, body: JSON.stringify(r.update) });
    if (rows.length) written++; else raced++;
  }
  console.log(`${dryRun ? 'Dry run: ' : ''}${bets.length} pending bets checked.`);
  for (const [k, n] of Object.entries(tally).sort()) console.log(`  ${k}: ${n}`);
  if (!dryRun) console.log(`Settled ${written}${raced ? `; ${raced} changed elsewhere first` : ''}.`);
  const unavailable = results.filter(r => r.skip === 'feed unavailable').length;
  if (unavailable && unavailable === results.filter(r => !['league not supported', 'not played yet', 'older than lookback'].includes(r.skip)).length) {
    throw new Error('Every feed request failed; nothing could be graded.');
  }
}

if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });
module.exports = { gradeBet, gradeAll, teamFor, backfillTeams, payout, SETTLE_AFTER_HOURS };

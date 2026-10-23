#!/usr/bin/env node
/* One-time replay of the arbitrage scan over every league snapshot committed to git:

     git fetch --unshallow origin main   # the replay needs history, not a shallow clone
     node scripts/arbitrage/replay.cjs [--ref origin/main]

   Each distinct snapshot is scanned with today's engine and rules as of its own time, and the
   pairs are folded into a ledger exactly as the scheduled scan does (scan.cjs record). Writes
   docs/arbitrage/data/replay.json, which the plan page reports (scripts/arbitrage/build.py).

   This is a paper replay: saved quotes, not bets placed. A pair seen once may have been gone
   minutes later; one seen in two or more snapshots lasted at least until the next refresh.
   Profits use the prices when each pair was first seen and stakes that pay the same either way. */
'use strict';
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const A = require('../../docs/assets/arbitrage.js');
const { record, FEEDS } = require('./scan.cjs');

const ROOT = path.resolve(__dirname, '../..');
const git = (...args) => execFileSync('git', ['-C', ROOT, ...args], { encoding: 'utf8', maxBuffer: 1 << 30 });
const STAKES = [250, 1000];

function snapshots(ref, file) {
  const seen = new Set(), out = [];
  for (const commit of git('log', ref, '--format=%H', '--', file).split('\n').filter(Boolean).reverse()) {
    let feed;
    try { feed = JSON.parse(git('show', `${commit}:${file}`)); } catch { continue; }
    const at = feed.last_success_at || feed.generated_at;
    if (!at || seen.has(at) || !Array.isArray(feed.rows)) continue;
    seen.add(at);
    out.push({ at, rows: feed.rows });
  }
  return out;
}

const profit = (pairs, stake) => Math.round(pairs.reduce((s, p) => s + stake * p.first_return_pct / 100, 0));
const mean = pairs => pairs.length ? pairs.reduce((s, p) => s + p.first_return_pct, 0) / pairs.length : 0;

function main() {
  const argv = process.argv.slice(2), i = argv.indexOf('--ref'), ref = i >= 0 ? argv[i + 1] : 'origin/main';
  const rules = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs/arbitrage/data/rules.json'), 'utf8'));
  const sports = {}, entries = [], firstPair = new Map();
  let from = null, to = null;
  for (const sport of Object.keys(FEEDS)) {
    let ledger = { entries: [] }, withPairs = 0, count = 0;
    for (const snap of snapshots(ref, FEEDS[sport])) {
      const now = Date.parse(snap.at);
      const { locked, check } = A.scan(A.fromSnapshot(snap.rows, sport), { rules, sport, now });
      const pairs = [...locked, ...check];
      for (const p of pairs) if (!firstPair.has(p.id)) firstPair.set(p.id, p);
      if (pairs.length) withPairs++;
      count++;
      ledger = { entries: record(ledger, pairs, snap.at, now) };
      from = !from || snap.at < from ? snap.at : from;
      to = !to || snap.at > to ? snap.at : to;
    }
    const books = new Map();
    for (const e of ledger.entries) {
      const k = e.first_legs.map(l => l.book_label).sort().join('|');
      books.set(k, (books.get(k) || 0) + 1);
    }
    sports[sport] = { snapshots: count, snapshots_with_pairs: withPairs, pairs: ledger.entries.length,
      locked: ledger.entries.filter(e => e.status === 'locked').length,
      persisted: ledger.entries.filter(e => e.snapshots >= 2).length,
      top_book_pairs: [...books].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, n]) => [...k.split('|'), n]) };
    entries.push(...ledger.entries);
    console.log(`${sport}: ${count} snapshots, ${ledger.entries.length} distinct pairs`);
  }
  const scenario = (label, note, keep) => {
    const pairs = entries.filter(keep), noMlb = pairs.filter(p => p.sport !== 'MLB');
    return { label, note, pairs: pairs.length, mean_return_pct: mean(pairs),
      profit: Object.fromEntries(STAKES.map(s => [s, profit(pairs, s)])),
      profit_no_mlb: Object.fromEntries(STAKES.map(s => [s, profit(noMlb, s)])) };
  };
  const twice = e => e.snapshots >= 2;
  const scenarios = [
    scenario('A. Every pair under 100%', 'Any book, any rules status, gone or not by the next refresh.', () => true),
    scenario('B. Still there at the next refresh', 'Seen in two or more snapshots, so there was time to act.', twice),
    scenario('C. Licensed books only', 'Both books licensed by US states.', e => e.licensed),
    scenario('D. Licensed and still there', 'Both conditions.', e => e.licensed && twice(e)),
    scenario('E. Rules on file for both books', 'Reached Locked at least once; checked or assumed settlement rules.', e => e.status === 'locked'),
  ];
  // Federal tax on scenario B at $1,000 per pair, worst result of each pair.
  const b = entries.filter(twice).map(e => firstPair.get(e.id)).filter(Boolean);
  const taxCase = (label, options) => {
    const values = b.map(p => options ? A.afterTax(p, 1000, options) : A.stakes(p, 1000).profit);
    return { label, profit: Math.round(values.reduce((s, v) => s + v, 0)), pairs: values.length,
      pairs_positive: values.filter(v => v > 0).length };
  };
  const tax = { stake: 1000, cases: [
    taxCase('Before tax', null),
    taxCase('After tax: 24% bracket, itemizing (90% of losses deductible)', { rate: 0.24, itemize: true }),
    taxCase('After tax: 32% bracket, itemizing', { rate: 0.32, itemize: true }),
    taxCase('After tax: 24% bracket, standard deduction (losses not deductible)', { rate: 0.24, itemize: false }),
  ] };
  // What the pairs that lasted were made of: an offshore book, an MLB prop (rules not on file), or neither.
  const mix = { offshore: 0, mlb_props: 0, licensed_other: 0 };
  for (const e of entries.filter(twice)) {
    const p = firstPair.get(e.id);
    if (!p.licensed) mix.offshore++;
    else if (p.sport === 'MLB' && p.legs.some(l => A.groupOf('MLB', l.market) !== 'game')) mix.mlb_props++;
    else mix.licensed_other++;
  }
  const examples = [...entries].sort((x, y) => y.first_return_pct - x.first_return_pct).slice(0, 8)
    .map(e => ({ sport: e.sport, game: e.game, subject: e.subject, first_seen: e.first_seen, snapshots: e.snapshots,
      return_pct: e.first_return_pct, status: e.status, licensed: e.licensed, legs: e.first_legs }));
  const out = { schema_version: 1, generated_by: 'scripts/arbitrage/replay.cjs', ref, rules_version: rules.version,
    window: { from, to }, sports, scenarios, persisted_mix: mix, tax, examples };
  fs.writeFileSync(path.join(ROOT, 'docs/arbitrage/data/replay.json'), JSON.stringify(out, null, 1) + '\n');
  console.log(JSON.stringify({ scenarios, tax }, null, 1));
}

main();

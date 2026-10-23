#!/usr/bin/env node
/* Scheduled arbitrage scan for one league, run by its refresh workflow after the snapshot is saved:

     node scripts/arbitrage/scan.cjs --sport NBA

   Reads that league's published quotes (FEEDS) and docs/arbitrage/data/rules.json, runs the
   shared engine (docs/assets/arbitrage.js) and writes docs/arbitrage/data/<sport>.json for the
   Arbitrage page. Every locked or check-first pair is also kept in a paper ledger
   (docs/arbitrage/data/ledger/<sport>.json): when it was first and last seen, in how many
   snapshots, and the prices when first seen. A snapshot already in the ledger is not counted
   twice. Each league writes only its own files, so refreshes never collide. No requests. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const A = require('../../docs/assets/arbitrage.js');

const FEEDS = { NFL: 'docs/nfl/data/quotes.json', NBA: 'docs/nba/data/latest.json',
  NHL: 'docs/nhl/data/latest.json', MLB: 'docs/mlb/data/latest.json' };
const KEEP_DAYS = 180;
const OUT = 'docs/arbitrage/data';

const iso = ms => new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');

function read(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (error) {
    if (fallback !== undefined && error.code === 'ENOENT') return fallback;
    throw error;
  }
}

// Atomic: a reader never sees half a file.
function write(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value, null, 1) + '\n');
  fs.renameSync(tmp, file);
}

const publicPair = ({ start, ...pair }) => pair;
const legSummary = pair => pair.legs.map(l => ({ book: l.book, book_label: l.book_label, market_label: l.market_label,
  side: l.side, line: l.line, price: l.price }));

/* Fold one scan into the ledger. A pair keeps the prices from when it was first seen; its
   status is the best it reached (locked beats check). */
function record(ledger, pairs, seenAt, now) {
  const entries = new Map((ledger.entries || []).map(e => [e.id, e]));
  for (const pair of pairs) {
    const e = entries.get(pair.id);
    if (!e) {
      entries.set(pair.id, { id: pair.id, sport: pair.sport, game: pair.game, commence_time: pair.commence_time,
        subject: pair.subject, kind: pair.kind, cross_market: pair.cross_market, licensed: pair.licensed,
        status: pair.status, tier: pair.tier, first_seen: seenAt, last_seen: seenAt, snapshots: 1,
        first_return_pct: pair.return_pct, best_return_pct: pair.return_pct, first_legs: legSummary(pair) });
      continue;
    }
    e.last_seen = seenAt;
    e.snapshots += 1;
    e.best_return_pct = Math.max(e.best_return_pct, pair.return_pct);
    if (pair.status === 'locked' && e.status !== 'locked') Object.assign(e, { status: 'locked', tier: pair.tier });
    e.licensed = e.licensed || pair.licensed;
  }
  const cutoff = now - KEEP_DAYS * 86400e3;
  return [...entries.values()].filter(e => Date.parse(e.commence_time) >= cutoff)
    .sort((a, b) => b.first_seen.localeCompare(a.first_seen) || a.id.localeCompare(b.id));
}

function run({ sport, root = path.resolve(__dirname, '../..'), now = Date.now() }) {
  if (!FEEDS[sport]) throw new Error(`Unknown sport ${sport}; use one of ${Object.keys(FEEDS).join(', ')}`);
  const rules = read(path.join(root, OUT, 'rules.json'));
  const feed = read(path.join(root, FEEDS[sport]), null);
  const snapshotAt = feed?.last_success_at || feed?.generated_at || null;
  const { locked, check, near, stats } = A.scan(A.fromSnapshot(feed?.rows, sport), { rules, sport, now });
  const result = { schema_version: 1, sport, scanned_at: iso(now), snapshot_at: snapshotAt,
    feed_status: feed ? feed.status || 'unknown' : 'missing', rules_version: rules.version,
    pair_window_seconds: A.PAIR_WINDOW / 1000, near_threshold: A.NEAR, stats,
    locked: locked.map(publicPair), check: check.map(publicPair), near: near.map(publicPair) };
  write(path.join(root, OUT, `${sport.toLowerCase()}.json`), result);

  const ledgerFile = path.join(root, OUT, 'ledger', `${sport.toLowerCase()}.json`);
  const ledger = read(ledgerFile, { schema_version: 1, sport, entries: [] });
  if (snapshotAt && ledger.last_snapshot !== snapshotAt) {
    write(ledgerFile, { schema_version: 1, sport, updated_at: iso(now), last_snapshot: snapshotAt,
      entries: record(ledger, [...locked, ...check], snapshotAt, now) });
  }
  return result;
}

if (require.main === module) {
  const argv = process.argv.slice(2), opt = name => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : undefined; };
  const sport = String(opt('sport') || '').toUpperCase(), when = opt('now');
  const result = run({ sport, ...(opt('root') ? { root: path.resolve(opt('root')) } : {}), ...(when ? { now: Date.parse(when) } : {}) });
  console.log(`${sport} arbitrage: ${result.locked.length} locked, ${result.check.length} to check, ${result.near.length} near misses`
    + ` from ${result.stats.quotes} quotes at ${result.stats.books.length} books (feed ${result.feed_status}).`);
}

module.exports = { run, record, FEEDS };

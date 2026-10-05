// HTTP handler for closing-lines. Kept free of Deno APIs so tests/closing_lines.cjs
// can run it with stubbed fetches; index.ts only wires it to Deno.serve.
//
// Called by pg_cron every 5 minutes (supabase/closing_lines_schedule.sql) with the
// shared secret from Vault in an x-closing-secret header, checked through
// public.closing_lines_secret_ok. The Odds API key is the same one live-odds uses:
// an ODDS_API_KEY secret, else the Vault copy read through public.live_odds_key.
//
// POST {}                          live: match pending bets (yesterday..tomorrow ET)
//                                  to Odds API events (free), then for games starting
//                                  within 6 minutes take one priced snapshot per game,
//                                  only for the markets bet on (1 credit per market).
// POST {mode:'backfill', limit:50} past games still without a close: historical
//                                  events (1 credit per sport and day) and historical
//                                  odds one minute before the start (10 credits per
//                                  market per game).
//
// Every paid snapshot is stored whole in public.odds_snapshots. A paid request is
// refused when it could take the credits left below CLOSING_LINES_RESERVE (default
// and minimum 2000, the owner's floor for the shared Odds API balance; the variable
// can only raise it), or when one run would spend more than RUN_CAP credits.

import { specFor, matchEvent, closeForBet, etDate } from './clv.mjs';

const ODDS = 'https://api.the-odds-api.com/v4';
const WINDOW_MS = 6 * 60e3;
const HOUR = 3600e3, DAY = 24 * HOUR;
export const CREDIT_FLOOR = 2000;
export const RUN_CAP = 2000;   // larger jobs need the owner's approval

// Upper bound on what one request can cost: markets x regions, 10x for historical
// odds; historical event lists cost 1 and live event lists are free.
export function estimateCost(path, params, paid) {
  if (!paid) return 0;
  if (/\/events$/.test(path)) return 1;
  const markets = String(params.markets || '').split(',').filter(Boolean).length || 1;
  const regions = String(params.regions || 'us').split(',').filter(Boolean).length || 1;
  return markets * regions * (path.startsWith('historical/') ? 10 : 1);
}
const FIELDS = 'id,league,market_type,side,line,player,book,odds,game_date,team_home,team_away,event_id,commence_time';

class Failure extends Error {
  constructor(status, message, { abort = false } = {}) { super(message); this.status = status; this.abort = abort; }
}

const iso = ms => new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');
const numeric = v => v != null && v !== '' && Number.isFinite(Number(v)) ? Number(v) : null;
function groupBy(items, keyOf) {
  const groups = new Map();
  for (const item of items) {
    const k = keyOf(item);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(item);
  }
  return groups;
}

export function createHandler({ fetchImpl = fetch, now = () => Date.now(), env = () => undefined } = {}) {
  let vaultKey = null;

  function database(base, key) {
    const headers = { apikey: key, Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };
    async function call(path, init = {}) {
      const res = await fetchImpl(`${base}/rest/v1/${path}`,
        { ...init, headers: { ...headers, ...(init.headers || {}) } });
      if (!res.ok) throw new Failure(502, `Database request failed (HTTP ${res.status}).`, { abort: true });
      const text = await res.text();
      return text ? JSON.parse(text) : null;
    }
    const minimal = { Prefer: 'return=minimal' };
    return {
      rpc: (fn, args = {}) => call(`rpc/${fn}`, { method: 'POST', body: JSON.stringify(args) }),
      upcoming: (from, to) => call(`bets?select=${FIELDS}&clv_status=is.null`
        + `&game_date=gte.${from}&game_date=lte.${to}&order=game_date`),
      past: (today, nowIso, limit) => call(`bets?select=${FIELDS}&clv_status=is.null`
        + `&or=(game_date.lt.${today},commence_time.lt.${nowIso})&order=game_date&limit=${limit}`),
      patch: (id, fields) => call(`bets?id=eq.${encodeURIComponent(id)}`,
        { method: 'PATCH', headers: minimal, body: JSON.stringify(fields) }),
      snapshot: row => call('odds_snapshots', { method: 'POST', headers: minimal, body: JSON.stringify(row) }),
    };
  }

  function oddsApi(db, state) {
    const reserve = Math.max(CREDIT_FLOOR, numeric(env('CLOSING_LINES_RESERVE')) ?? CREDIT_FLOOR);
    async function key() {
      const direct = env('ODDS_API_KEY');
      if (direct) return direct;
      if (!vaultKey) {
        const k = await db.rpc('live_odds_key');
        if (typeof k === 'string' && k) vaultKey = k;
      }
      return vaultKey;
    }
    return async function get(path, params, paid) {
      const estimate = estimateCost(path, params, paid);
      if (paid && state.remaining != null && state.remaining - estimate < reserve) {
        throw new Failure(429, `Paused: ${state.remaining} credits left; this request could take the balance below the ${reserve} reserve.`, { abort: true });
      }
      if (paid && state.spent + estimate > RUN_CAP) {
        throw new Failure(429, `Paused: this run has spent ${state.spent} credits; the per-run cap is ${RUN_CAP}.`, { abort: true });
      }
      const apiKey = await key();
      if (!apiKey) throw new Failure(503, 'No Odds API key: set ODDS_API_KEY or run the Live Odds key workflow.', { abort: true });
      let res;
      try {
        res = await fetchImpl(`${ODDS}/${path}?${new URLSearchParams({ ...params, apiKey })}`,
          { headers: { Accept: 'application/json' } });
      } catch {
        throw new Failure(502, 'The odds service could not be reached.');   // never echo the URL: it has the key
      }
      const left = numeric(res.headers.get('x-requests-remaining'));
      if (left != null) state.remaining = left;
      const cost = numeric(res.headers.get('x-requests-last'));
      if (cost) state.spent += cost;
      if (res.status === 401 || res.status === 403) {
        vaultKey = null;
        throw new Failure(503, 'The odds service refused the key, or the plan is out of credits.', { abort: true });
      }
      if (res.status === 404) return null;
      if (res.status === 422) throw new Failure(422, 'The odds service rejected the request (unsupported market or date).');
      if (res.status === 429) throw new Failure(429, 'The odds service is rate limiting.', { abort: true });
      if (!res.ok) throw new Failure(502, `The odds service returned HTTP ${res.status}.`);
      return { body: await res.json(), cost };
    };
  }

  async function run(mode, limit, db) {
    const t = now();
    const state = { remaining: null, spent: 0 };
    const odds = oddsApi(db, state);
    const out = { mode, bets: 0, matched: 0, results: {}, errors: [] };
    const count = s => { out.results[s] = (out.results[s] || 0) + 1; };

    const bets = mode === 'backfill'
      ? await db.past(etDate(t), iso(t), limit)
      : await db.upcoming(etDate(t - DAY), etDate(t + DAY));
    out.bets = bets.length;

    // Leagues and markets with no mapping are marked once and skipped after.
    const usable = [];
    for (const bet of bets) {
      const spec = specFor(bet);
      if (spec.error) {
        await db.patch(bet.id, { clv_status: 'unsupported', clv_note: spec.error });
        count('unsupported');
      } else usable.push({ bet, spec });
    }

    // Price one game's markets, store the raw snapshot, and write each bet's close.
    async function capture(sport, eventId, group, snapshotDate) {
      const path = snapshotDate ? `historical/sports/${sport}/events/${eventId}/odds`
        : `sports/${sport}/events/${eventId}/odds`;
      const params = { regions: 'us', oddsFormat: 'american', dateFormat: 'iso',
        ...(snapshotDate ? { date: snapshotDate } : {}) };
      const all = [...new Set(group.map(x => x.spec.key))];
      // One request for every market; if the service rejects the set, retry each
      // market alone so one bad market cannot cost the others their close.
      let batches = [all];
      for (let attempt = 0; attempt < 2; attempt++) {
        const rejected = [];
        for (const markets of batches) {
          let res;
          try {
            res = await odds(path, { ...params, markets: markets.join(',') }, true);
          } catch (error) {
            if (error.abort) throw error;
            if (error.status === 422 && markets.length > 1 && attempt === 0) { rejected.push(...markets); continue; }
            for (const x of group.filter(x => markets.includes(x.spec.key))) {
              if (error.status === 422) {
                await db.patch(x.bet.id, { clv_status: 'no_market', clv_note: 'The odds service does not offer this market here.' });
                count('no_market');
              }
            }
            if (error.status !== 422) out.errors.push(`${eventId}: ${error.message}`);
            continue;
          }
          const event = snapshotDate ? res?.body?.data : res?.body;
          const snapshotAt = snapshotDate ? res?.body?.timestamp : iso(t);
          const bets = group.filter(x => markets.includes(x.spec.key));
          if (!event || event.id !== eventId) {
            for (const x of bets) {
              await db.patch(x.bet.id, { clv_status: 'no_market', clv_note: 'The game was not in the odds snapshot.' });
              count('no_market');
            }
            continue;
          }
          await db.snapshot({ source: snapshotDate ? 'historical' : 'live', sport_key: sport, event_id: eventId,
            snapshot_at: snapshotAt, markets, credits_used: res.cost, payload: event });
          for (const x of bets) {
            const close = closeForBet(x.bet, x.spec, event);
            await db.patch(x.bet.id, { ...close, closing_captured_at: snapshotAt,
              event_id: eventId, commence_time: x.bet.commence_time });
            count(close.clv_status);
          }
        }
        if (!rejected.length) break;
        batches = rejected.map(m => [m]);
      }
    }

    try {
      // 1. Find each bet's game. Live uses the free events list; backfill pays
      // 1 credit for a historical list per sport and day, taken at 12:00 UTC.
      const unmatched = usable.filter(x => !x.bet.event_id);
      const keyOf = x => mode === 'backfill' ? `${x.spec.sport}|${String(x.bet.game_date).slice(0, 10)}` : x.spec.sport;
      for (const [k, group] of groupBy(unmatched, keyOf)) {
        const [sport, day] = k.split('|');
        let events = [];
        if (mode === 'backfill') {
          const res = await odds(`historical/sports/${sport}/events`, { dateFormat: 'iso', date: `${day}T12:00:00Z` }, true);
          events = Array.isArray(res?.body?.data) ? res.body.data : [];
        } else {
          const res = await odds(`sports/${sport}/events`, { dateFormat: 'iso',
            commenceTimeFrom: iso(t - 12 * HOUR), commenceTimeTo: iso(t + 48 * HOUR) }, false);
          events = Array.isArray(res?.body) ? res.body : [];
        }
        for (const x of group) {
          const m = matchEvent(x.bet, events);
          if (m.event) {
            x.bet.event_id = m.event.id;
            x.bet.commence_time = m.event.commence_time;
            await db.patch(x.bet.id, { event_id: m.event.id, commence_time: m.event.commence_time });
            out.matched++;
          } else if (mode === 'backfill') {
            // Live leaves misses for the next run; backfill is the last chance.
            await db.patch(x.bet.id, { clv_status: 'no_event', clv_note: m.error });
            count('no_event');
          }
        }
      }

      // 2. Take the close.
      const ready = usable.filter(x => {
        const c = Date.parse(x.bet.commence_time);
        if (!x.bet.event_id || !Number.isFinite(c)) return false;
        return mode === 'backfill' ? c <= t : c > t && c - t <= WINDOW_MS;
      });
      for (const [eventId, group] of groupBy(ready, x => x.bet.event_id)) {
        const start = Date.parse(group[0].bet.commence_time);
        await capture(group[0].spec.sport, eventId, group, mode === 'backfill' ? iso(start - 60e3) : null);
      }
    } catch (error) {
      if (!(error instanceof Failure) || !error.abort) throw error;
      out.stopped = error.message;
    }

    out.credits_used = state.spent;
    out.credits_left = state.remaining;
    return out;
  }

  return async function handle(req) {
    const reply = (status, body) => new Response(JSON.stringify(body),
      { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });
    if (req.method !== 'POST') return reply(405, { error: 'POST required' });

    const base = env('SUPABASE_URL'), service = env('SUPABASE_SERVICE_ROLE_KEY');
    if (!base || !service) return reply(503, { error: 'Server connection is not configured.' });
    const db = database(base, service);

    try {
      const secret = req.headers.get('x-closing-secret') || '';
      if (!secret || (await db.rpc('closing_lines_secret_ok', { candidate: secret })) !== true) {
        return reply(401, { error: 'Unauthorized' });
      }
      const raw = await req.text();
      if (raw.length > 200) return reply(400, { error: 'Request too large' });
      let input;
      try { input = raw ? JSON.parse(raw) : {}; } catch { return reply(400, { error: 'Invalid request' }); }
      if (!input || typeof input !== 'object' || Array.isArray(input)) return reply(400, { error: 'Invalid request' });
      const mode = input.mode == null ? 'live' : input.mode;
      if (mode !== 'live' && mode !== 'backfill') return reply(400, { error: 'mode must be live or backfill' });
      const limit = Math.min(Math.max(Math.trunc(numeric(input.limit) ?? 50), 1), 200);
      return reply(200, await run(mode, limit, db));
    } catch (error) {
      if (error instanceof Failure) return reply(error.status, { error: error.message });
      return reply(500, { error: 'Closing lines failed.' });
    }
  };
}

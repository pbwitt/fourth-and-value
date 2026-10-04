// HTTP handler for live-odds. Kept free of Deno APIs so tests/live_odds.cjs can
// run it with stubbed fetches; index.ts only wires it to Deno.serve.
//
// One press of Run now on the private Live Odds page (docs/live/) is at most one
// paid Odds API request, for one NHL game. Only editor accounts
// (app_metadata.fv_editor) may spend it, and the key never leaves the server: an
// ODDS_API_KEY Edge Function secret if set, otherwise the copy the "Live Odds key"
// workflow keeps in Supabase Vault (supabase/live_odds_key.sql), read with the
// service role.
//
// POST {}             -> NHL games from 6 hours ago to 24 hours ahead (free endpoint)
// POST {event:'<id>'} -> that game's prices from US books: moneyline, puck line,
//                        total and four player props. The Odds API charges one
//                        credit per market returned, so 7 at most.
//
// Guards: a game's prices are reused for 60 seconds, so repeat presses cost
// nothing; paid requests stop while the credits left are below the reserve kept
// for the scheduled refreshes (LIVE_ODDS_RESERVE, default 2000).

const ORIGINS = new Set(['https://fourthandvalue.com', 'https://www.fourthandvalue.com']);
const ODDS = 'https://api.the-odds-api.com/v4/sports/icehockey_nhl';
export const MARKETS = ['h2h', 'spreads', 'totals',
  'player_shots_on_goal', 'player_goals', 'player_assists', 'player_points'];
const TTL = { odds: 60e3, events: 300e3, error: 10e3 };
const EVENT_ID = /^[0-9a-f]{32}$/;
const MAX_BYTES = 3e6;
const MAX_ENTRIES = 100;

class Failure extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

const iso = ms => new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');

export function createHandler({ fetchImpl = fetch, now = () => Date.now(), env = () => undefined,
  allowOrigin = o => ORIGINS.has(o) } = {}) {
  const cache = new Map();
  let remaining = null;   // credits left, from the most recent Odds API response
  let vaultKey = null;

  function reserve() {
    const raw = env('LIVE_ODDS_RESERVE'), n = raw == null || raw === '' ? NaN : Number(raw);
    return Number.isFinite(n) && n >= 0 ? n : 2000;
  }

  // Concurrent callers share one in-flight request; failures are kept briefly.
  function cached(key, ttl, load) {
    const hit = cache.get(key);
    if (hit && hit.expires > now()) return { value: hit.value, reused: true };
    const value = load().then(
      result => { cache.set(key, { value, expires: now() + ttl }); return result; },
      error => { cache.set(key, { value, expires: now() + TTL.error }); throw error; });
    cache.set(key, { value, expires: now() + TTL.error });
    if (cache.size > MAX_ENTRIES) {
      for (const [k, v] of cache) if (cache.size > MAX_ENTRIES || v.expires <= now()) cache.delete(k);
    }
    return { value, reused: false };
  }

  // Returns null for no valid session, false for a signed-in non-editor.
  async function editor(authorization) {
    const base = env('SUPABASE_URL'), key = env('SUPABASE_ANON_KEY');
    if (!base || !key) throw new Failure(503, 'Live odds need their server connection configured.');
    const res = await fetchImpl(`${base}/auth/v1/user`, { headers: { apikey: key, Authorization: authorization } });
    if (!res.ok) return null;
    const user = await res.json();
    return user?.app_metadata?.fv_editor === true;
  }

  async function oddsKey() {
    const direct = env('ODDS_API_KEY');
    if (direct) return direct;
    if (vaultKey) return vaultKey;
    const base = env('SUPABASE_URL'), service = env('SUPABASE_SERVICE_ROLE_KEY');
    if (!base || !service) return null;
    try {
      const res = await fetchImpl(`${base}/rest/v1/rpc/live_odds_key`, { method: 'POST', body: '{}',
        headers: { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' } });
      const key = res.ok ? await res.json() : null;
      if (typeof key === 'string' && key) vaultKey = key;
    } catch { /* reported below as a missing key */ }
    return vaultKey;
  }

  async function upstream(path, params) {
    const key = await oddsKey();
    if (!key) throw new Failure(503, 'Live odds have no Odds API key yet. Run the Live Odds key workflow, or add ODDS_API_KEY to the Edge Function secrets.');
    let res;
    try {
      res = await fetchImpl(`${ODDS}/${path}?${new URLSearchParams({ ...params, apiKey: key })}`,
        { headers: { Accept: 'application/json' } });
    } catch {
      // Never echo the request URL: it carries the key.
      throw new Failure(502, 'The odds service could not be reached.');
    }
    const left = res.headers.get('x-requests-remaining');
    if (left != null && left !== '' && Number.isFinite(Number(left))) remaining = Number(left);
    if (res.status === 401 || res.status === 403) {
      vaultKey = null;   // a rotated key is read again on the next press
      throw new Failure(503, 'The odds service refused the key, or the plan is out of credits.');
    }
    if (res.status === 404) throw new Failure(404, 'The odds service no longer lists that game.');
    if (res.status === 429) throw new Failure(429, 'The odds service is busy. Wait a few seconds and try again.');
    if (!res.ok) throw new Failure(502, `The odds service returned an error (HTTP ${res.status}).`);
    const text = await res.text();
    if (text.length > MAX_BYTES) throw new Failure(502, 'The odds response was too large.');
    let body;
    try { body = JSON.parse(text); } catch { throw new Failure(502, 'The odds service sent an unreadable response.'); }
    const last = res.headers.get('x-requests-last');
    return { body, cost: last != null && last !== '' && Number.isFinite(Number(last)) ? Number(last) : null };
  }

  async function games() {
    const { value } = cached('events', TTL.events, async () => {
      const t = now();
      const { body } = await upstream('events', { dateFormat: 'iso',
        commenceTimeFrom: iso(t - 6 * 3600e3), commenceTimeTo: iso(t + 24 * 3600e3) });
      if (!Array.isArray(body)) throw new Failure(502, 'The odds service listed games in an unexpected format.');
      return body.filter(e => EVENT_ID.test(String(e?.id)))
        .map(e => ({ id: e.id, commence_time: e.commence_time, home_team: e.home_team, away_team: e.away_team }));
    });
    return { games: await value, remaining };
  }

  async function prices(event) {
    const key = 'odds:' + event, hit = cache.get(key);
    if (!(hit && hit.expires > now()) && remaining != null && remaining < reserve()) {
      throw new Failure(429, `Live odds are paused: ${remaining} credits left, below the ${reserve()} kept for the scheduled refreshes.`);
    }
    const { value, reused } = cached(key, TTL.odds, async () => {
      const fetched = now();
      const { body, cost } = await upstream(`events/${event}/odds`,
        { regions: 'us', markets: MARKETS.join(','), oddsFormat: 'american', dateFormat: 'iso' });
      if (!body || typeof body !== 'object' || body.id !== event) {
        throw new Failure(502, 'The odds service returned a different game than requested.');
      }
      return { event: body, fetched_at: iso(fetched), cost };
    });
    const result = await value;
    return { ...result, cost: reused ? 0 : result.cost, reused, remaining };
  }

  return async function handle(req) {
    const origin = req.headers.get('origin') || '';
    const cors = { 'Access-Control-Allow-Origin': allowOrigin(origin) ? origin : 'https://fourthandvalue.com',
      'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
      'Access-Control-Allow-Methods': 'POST, OPTIONS', Vary: 'Origin' };
    const reply = (status, body) => new Response(JSON.stringify(body),
      { status, headers: { ...cors, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });
    if (!allowOrigin(origin)) return reply(403, { error: 'Unsupported origin' });
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
    if (req.method !== 'POST') return reply(405, { error: 'POST required' });
    const authorization = req.headers.get('authorization') || '';
    if (!authorization.startsWith('Bearer ')) return reply(401, { error: 'Sign in first.' });

    const raw = await req.text();
    if (raw.length > 200) return reply(400, { error: 'Request too large' });
    let input;
    try { input = raw ? JSON.parse(raw) : {}; } catch { return reply(400, { error: 'Invalid request' }); }
    if (!input || typeof input !== 'object' || Array.isArray(input)) return reply(400, { error: 'Invalid request' });
    if (input.event != null && !EVENT_ID.test(String(input.event))) return reply(400, { error: 'Invalid game' });

    try {
      const allowed = await editor(authorization);
      if (allowed === null) return reply(401, { error: 'Sign in again.' });
      if (!allowed) return reply(403, { error: 'Live odds are limited to editor accounts.' });
      return reply(200, input.event == null ? await games() : await prices(String(input.event)));
    } catch (error) {
      if (error instanceof Failure) return reply(error.status, { error: error.message, remaining });
      return reply(502, { error: 'Live odds are unavailable right now.', remaining });
    }
  };
}

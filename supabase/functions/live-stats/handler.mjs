// HTTP handler for live-stats. Kept free of Deno APIs so tests/live_stats.cjs
// can run it with a stubbed fetch; index.ts only wires it to Deno.serve.
//
// The NHL feed sends no CORS header, so Bet Tracker cannot read it directly.
// This relays two fixed NHL endpoints unchanged; the browser normalizes them
// with docs/tracking/live-feeds.js like every other league.
//
// POST {league:'NHL', date:'YYYY-MM-DD'} -> api-web.nhle.com/v1/score/{date}
// POST {league:'NHL', game:'2026010053'} -> .../gamecenter/{game}/boxscore
//
// Responses are cached per isolate so every tracker watching a game shares one
// upstream request per interval. Callers only choose a date or a numeric id.

const ORIGINS = new Set(['https://fourthandvalue.com', 'https://www.fourthandvalue.com']);
const NHL = 'https://api-web.nhle.com/v1';
const TTL = { live: 15e3, idle: 60e3, error: 10e3 };
const MAX_ENTRIES = 400;
const MAX_BYTES = 2e6;

export function createHandler({ fetchImpl = fetch, now = () => Date.now(), allowOrigin = o => ORIGINS.has(o) } = {}) {
  const cache = new Map();

  function cached(key, load) {
    const hit = cache.get(key);
    if (hit && hit.expires > now()) return hit.value;
    const value = load().then(
      text => { cache.set(key, { value, expires: now() + (/"gameState":"(LIVE|CRIT)"/.test(text) ? TTL.live : TTL.idle) }); return text; },
      error => { cache.set(key, { value, expires: now() + TTL.error }); throw error; });
    // Concurrent callers share the in-flight request.
    cache.set(key, { value, expires: now() + TTL.error });
    if (cache.size > MAX_ENTRIES) {
      for (const [k, v] of cache) if (cache.size > MAX_ENTRIES || v.expires <= now()) cache.delete(k);
    }
    return value;
  }

  async function upstream(url) {
    const res = await fetchImpl(url, { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`Upstream ${res.status}`);
    const text = await res.text();
    if (text.length > MAX_BYTES) throw new Error('Upstream response too large');
    JSON.parse(text);
    return text;
  }

  return async function handle(req) {
    const origin = req.headers.get('origin') || '';
    const cors = { 'Access-Control-Allow-Origin': allowOrigin(origin) ? origin : 'https://fourthandvalue.com',
      'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
      'Access-Control-Allow-Methods': 'POST, OPTIONS', Vary: 'Origin' };
    const reply = (status, body) => new Response(typeof body === 'string' ? body : JSON.stringify(body),
      { status, headers: { ...cors, 'Content-Type': 'application/json' } });
    if (!allowOrigin(origin)) return reply(403, { error: 'Unsupported origin' });
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
    if (req.method !== 'POST') return reply(405, { error: 'POST required' });

    const raw = await req.text();
    if (raw.length > 200) return reply(400, { error: 'Request too large' });
    let input;
    try { input = JSON.parse(raw); } catch { return reply(400, { error: 'Invalid request' }); }
    if (input?.league !== 'NHL') return reply(400, { error: 'Unsupported league' });

    let url;
    if (input.game != null) {
      if (!/^\d{1,12}$/.test(String(input.game))) return reply(400, { error: 'Invalid game' });
      url = `${NHL}/gamecenter/${input.game}/boxscore`;
    } else {
      const date = String(input.date || '');
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date))) return reply(400, { error: 'Invalid date' });
      url = `${NHL}/score/${date}`;
    }
    try { return reply(200, await cached(url, () => upstream(url))); }
    catch { return reply(502, { error: 'Live stats are unavailable right now.' }); }
  };
}

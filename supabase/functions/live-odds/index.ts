// Deploy as live-odds. Supabase provides SUPABASE_URL, SUPABASE_ANON_KEY and
// SUPABASE_SERVICE_ROLE_KEY. The Odds API key comes from an ODDS_API_KEY secret or
// the Vault copy kept by the Live Odds key workflow. LIVE_ODDS_RESERVE is optional.
// Logic lives in handler.mjs so tests/live_odds.cjs can run it in Node.
import { createHandler } from './handler.mjs';

Deno.serve(createHandler({ env: (name: string) => Deno.env.get(name) }));

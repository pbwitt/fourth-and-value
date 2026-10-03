// Deploy as live-odds. Needs ODDS_API_KEY in the Edge Function secrets; Supabase
// provides SUPABASE_URL and SUPABASE_ANON_KEY. LIVE_ODDS_RESERVE is optional.
// Logic lives in handler.mjs so tests/live_odds.cjs can run it in Node.
import { createHandler } from './handler.mjs';

Deno.serve(createHandler({ env: (name: string) => Deno.env.get(name) }));

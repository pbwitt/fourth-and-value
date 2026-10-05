// Deploy as closing-lines with verify_jwt off: pg_cron calls it without a user
// session, and the handler checks the Vault secret in x-closing-secret instead.
// Supabase provides SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. ODDS_API_KEY and
// CLOSING_LINES_RESERVE are optional. Logic lives in handler.mjs and clv.mjs so
// tests/closing_lines.cjs can run it in Node.
import { createHandler } from './handler.mjs';

Deno.serve(createHandler({ env: (name: string) => Deno.env.get(name) }));

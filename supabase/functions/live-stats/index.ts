// Deploy as live-stats. No secrets needed: it relays the public NHL feed only.
// Logic lives in handler.mjs so tests/live_stats.cjs can run it in Node.
import { createHandler } from './handler.mjs';

Deno.serve(createHandler());

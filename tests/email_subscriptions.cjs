/* supabase/email_subscriptions.sql in real PostgreSQL through PGlite. No network
 * requests or production database access. The public roles reach rows only through
 * the two token functions, which never return the address. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {PGlite} = require(process.env.FV_PGLITE || '@electric-sql/pglite');

(async () => {
  const db = new PGlite();
  // Supabase's default privileges: new tables and functions in public are granted to
  // all three roles, so the SQL itself must take table access away from the public ones.
  await db.exec(`create role anon; create role authenticated; create role service_role bypassrls;
    grant usage on schema public to anon, authenticated, service_role;
    alter default privileges in schema public grant all on tables to anon, authenticated, service_role;
    alter default privileges in schema public grant all on functions to anon, authenticated, service_role;`);
  const sql = fs.readFileSync(path.join(__dirname, '../supabase/email_subscriptions.sql'), 'utf8')
    .replace("notify pgrst, 'reload schema';", '');  // PostgREST is not running here
  await db.exec(sql);
  await db.exec(sql);  // re-running is safe
  const owner = (await db.query(`insert into public.email_subscriptions(list, email) values ('morning_card', 'owner@example.com')
    returning token::text, subscribed`)).rows[0];
  assert.equal(owner.subscribed, true);
  assert.match(owner.token, /^[0-9a-f-]{36}$/);
  for (const bad of [`('morning_card', 'Owner@Example.com')`, `('morning_card', 'not-an-address')`, `('weekly', 'x@example.com')`])
    await assert.rejects(db.query(`insert into public.email_subscriptions(list, email) values ${bad}`), /check constraint/);

  async function as(role, query, params) {
    await db.exec(`set role ${role}`);
    try { return (await db.query(query, params)).rows; } finally { await db.exec('reset role'); }
  }
  for (const role of ['anon', 'authenticated']) {
    await assert.rejects(as(role, 'select * from public.email_subscriptions'), /permission denied/, `${role} cannot read rows`);
    await assert.rejects(as(role, `update public.email_subscriptions set subscribed = false`), /permission denied/);
    await assert.rejects(as(role, `insert into public.email_subscriptions(list, email) values ('morning_card', 'x@example.com')`), /permission denied/);
  }
  assert.deepEqual(await as('anon', 'select * from public.email_subscription_get($1)', [owner.token]),
    [{list: 'morning_card', subscribed: true}], 'the address is never returned');
  assert.deepEqual(await as('anon', 'select * from public.email_subscription_set($1, false)', [owner.token]),
    [{list: 'morning_card', subscribed: false}]);
  assert.deepEqual(await as('anon', 'select * from public.email_subscription_get($1)', [owner.token]),
    [{list: 'morning_card', subscribed: false}]);
  const unknown = '11111111-2222-4333-8444-555555555555';
  assert.deepEqual(await as('anon', 'select * from public.email_subscription_set($1, true)', [unknown]), [], 'unknown tokens change nothing');
  assert.deepEqual(await as('anon', 'select * from public.email_subscription_set($1, null)', [owner.token]), [], 'null is not a state');
  assert.deepEqual(await as('authenticated', 'select * from public.email_subscription_set($1, true)', [owner.token]),
    [{list: 'morning_card', subscribed: true}]);
  // The sender (service role) reads and creates rows directly.
  assert.deepEqual(await as('service_role', `select subscribed from public.email_subscriptions where list = 'morning_card' and email = 'owner@example.com'`),
    [{subscribed: true}]);
  await assert.rejects(as('service_role', `insert into public.email_subscriptions(list, email) values ('morning_card', 'owner@example.com')`),
    /duplicate key/, 'one row per list and address');
  console.log('PASS: email subscriptions: no public table access, token-only functions without addresses, unknown tokens inert, constraints and re-runnable SQL.');
})().catch(error => { console.error(error); process.exitCode = 1; });

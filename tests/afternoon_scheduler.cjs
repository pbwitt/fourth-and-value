/* Real PostgreSQL/PLpgSQL through PGlite; pg_net/cron/Vault transport mocked.
 * No network requests or production database access. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {PGlite} = require(process.env.FV_PGLITE || '@electric-sql/pglite');

(async () => {
  const db = new PGlite();
  await db.exec(`
    create role anon; create role authenticated; create role service_role bypassrls;
    create schema vault; create schema net; create schema cron;
    create table vault.decrypted_secrets(name text primary key, decrypted_secret text);
    insert into vault.decrypted_secrets values ('fv_morning_github_token','fake-restricted-token-for-tests');
    create table net.requests(id bigserial primary key, url text, headers jsonb, body jsonb, timeout_ms int);
    create table net._http_response(id bigint, status_code int, timed_out boolean, error_msg text, created timestamptz default now());
    create function net.http_post(url text, body jsonb, headers jsonb, timeout_milliseconds int)
      returns bigint language sql as $$
        insert into net.requests(url,headers,body,timeout_ms) values ($1,$3,$2,$4) returning id;
      $$;
    create table cron.job(jobid bigserial primary key, jobname text unique, schedule text, command text);
    create function cron.schedule(job_name text, schedule text, command text)
      returns bigint language sql as $$
        insert into cron.job(jobname,schedule,command) values ($1,$2,$3)
        on conflict (jobname) do update set schedule=$2, command=$3 returning jobid;
      $$;
  `);
  // Extensions aren't loadable in PGlite; substitute only their presence query.
  const morning = fs.readFileSync(path.join(__dirname, '../supabase/morning_scheduler.sql'), 'utf8')
    .replaceAll('from pg_extension', "from (values ('pg_cron'), ('pg_net')) as installed(extname)");
  const afternoon = fs.readFileSync(path.join(__dirname, '../supabase/afternoon_scheduler.sql'), 'utf8');
  const workflow = fs.readFileSync(path.join(__dirname, '../.github/workflows/afternoon-refresh.yml'), 'utf8');
  assert.ok(workflow.includes("cron: '45 16 * * *'\n      timezone: America/New_York"), 'GitHub schedule is only a later backup');
  assert.ok(workflow.includes('options: [operator, supabase]'));
  assert.ok(workflow.includes('group: afternoon-market-refresh\n  cancel-in-progress: false'));
  assert.ok(workflow.includes('python scripts/afternoon_gate.py --source "$AFTERNOON_TRIGGER_SOURCE"'));
  for (const name of ['nhl-daily.yml', 'mlb-daily.yml']) {
    const sport = fs.readFileSync(path.join(__dirname, '../.github/workflows', name), 'utf8');
    assert.ok(!/^\s*schedule:/m.test(sport), `${name} has no late GitHub schedule of its own`);
    assert.ok(sport.includes('workflow_call:') && sport.includes('workflow_dispatch:'));
  }

  await assert.rejects(() => db.exec(afternoon), /morning_scheduler\.sql first/, 'requires the morning ledger');
  await db.exec('rollback');
  await db.exec(morning);
  // An earlier morning row predates the workflow column and keeps its meaning.
  await db.exec("select fv_morning.tick('2026-09-29T11:05:00Z');");
  await db.exec(afternoon);
  await db.exec(afternoon);
  const jobs = (await db.query('select jobname, schedule, command from cron.job order by jobname')).rows;
  assert.deepEqual(jobs.map(j => [j.jobname, j.schedule]), [['fv-afternoon-dispatch','*/5 20-22 * * *'],['fv-morning-dispatch','*/5 11-14 * * *']]);
  assert.equal((await db.query("select workflow from fv_morning.dispatches")).rows[0].workflow, 'morning-picks.yml');

  async function slot(time) {
    return (await db.query('select fv_morning.afternoon_slot($1)::text as slot', [time])).rows[0].slot;
  }
  async function tick(time) {
    return (await db.query('select fv_morning.afternoon_tick($1) as result', [time])).rows[0].result;
  }
  // Daylight and standard time, including both transition days.
  for (const day of ['2026-10-05','2026-12-01','2027-03-14','2026-11-01']) {
    const winter = ['2026-12-01','2026-11-01'].includes(day);
    const hour = winter ? 21 : 20;
    assert.equal(await slot(`${day}T${hour}:29:59Z`), null);
    assert.equal(Date.parse(await slot(`${day}T${hour}:30:00Z`)), Date.parse(`${day}T${hour}:30:00Z`));
    assert.equal(Date.parse(await slot(`${day}T${hour + 1}:29:59Z`)), Date.parse(`${day}T${hour}:30:00Z`));
    assert.equal(await slot(`${day}T${hour + 1}:30:00Z`), null);
  }
  // Every five-minute cron tick for a full day: exactly one request.
  let queued = 0;
  for (let h = 20; h <= 22; h++) {
    for (let m = 0; m < 60; m += 5) {
      const result = await tick(`2026-10-05T${h}:${String(m).padStart(2,'0')}:00Z`);
      queued += result.status === 'queued';
      assert.ok(!JSON.stringify(result).includes('fake-restricted-token'));
    }
  }
  assert.equal(queued, 1);
  const requests = (await db.query("select * from net.requests where url like '%afternoon-refresh.yml%'")).rows;
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, 'https://api.github.com/repos/pbwitt/fourth-and-value/actions/workflows/afternoon-refresh.yml/dispatches');
  assert.deepEqual(requests[0].body, {ref:'main',inputs:{trigger_source:'supabase'}});
  assert.equal(requests[0].timeout_ms, 10000);
  assert.equal(requests[0].headers.Authorization, 'Bearer fake-restricted-token-for-tests');
  const row = (await db.query("select slot_at, workflow from fv_morning.dispatches where workflow='afternoon-refresh.yml'")).rows[0];
  assert.equal(Date.parse(row.slot_at.toISOString()), Date.parse('2026-10-05T20:30:00Z'));
  // Morning ticks are unaffected and the two never share a slot.
  assert.equal((await db.query("select fv_morning.tick('2026-10-05T11:05:00Z') as r")).rows[0].r.status, 'queued');
  assert.equal((await tick('2026-10-05T15:00:00Z')).status, 'outside_window');
  // Reinstalling does not replay the day's request; a late first tick sends one current request.
  await db.exec(afternoon);
  assert.equal((await tick('2026-10-05T21:10:00Z')).status, 'already_requested');
  assert.equal((await tick('2026-10-06T21:25:00Z')).status, 'queued');
  assert.equal((await db.query("select * from net.requests where url like '%afternoon-refresh.yml%'")).rows.length, 2);
  // Receipts are recorded by the shared collector without retaining transport details.
  const id = (await db.query("select request_id from fv_morning.dispatches where workflow='afternoon-refresh.yml' order by slot_at limit 1")).rows[0].request_id;
  await db.exec(`insert into net._http_response(id,status_code,timed_out,error_msg) values (${id},204,false,null);`);
  await tick('2026-10-07T12:00:00Z');
  assert.equal((await db.query(`select result from fv_morning.dispatches where request_id=${id}`)).rows[0].result, 'accepted');
  await db.exec("delete from vault.decrypted_secrets;");
  await assert.rejects(() => db.exec("select fv_morning.afternoon_tick('2026-10-08T20:30:00Z');"));
  for (const role of ['anon','authenticated','service_role']) {
    await db.exec('set role ' + role);
    await assert.rejects(() => db.exec('select fv_morning.afternoon_tick()'));
    await assert.rejects(() => db.exec('select * from fv_morning.dispatches'));
    await db.exec('reset role');
  }
  console.log('Afternoon scheduler passed: ET/DST, one dispatch a day, install/repeat deduplication, morning coexistence, request identity, receipts, missing credentials, and private permissions. Transport/extensions remain mocked.');
  await db.close();
})().catch(e => { console.error(e.message); process.exitCode = 1; });

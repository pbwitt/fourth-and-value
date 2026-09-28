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
  const raw = fs.readFileSync(path.join(__dirname, '../supabase/morning_scheduler.sql'), 'utf8');
  const workflow = fs.readFileSync(path.join(__dirname, '../.github/workflows/morning-picks.yml'), 'utf8');
  assert.ok(workflow.includes("cron: '5,35 7,8 * * *'"));
  assert.ok(workflow.includes('timezone: America/New_York'));
  assert.ok(workflow.includes('group: morning-picks-edition\n  cancel-in-progress: false'));
  assert.ok(workflow.includes('options: [operator, supabase]'));
  // Extensions aren't loadable in PGlite; substitute only their presence query.
  // All scheduling, permission, deduplication and receipt code runs unchanged.
  const sql = raw.replaceAll('from pg_extension', "from (values ('pg_cron'), ('pg_net')) as installed(extname)");
  await db.exec(sql);
  await db.exec(sql);
  assert.equal((await db.query('select * from cron.job')).rows.length, 1);
  assert.equal((await db.query('select schedule from cron.job')).rows[0].schedule, '*/5 11-14 * * *');

  async function slot(time) {
    return (await db.query('select fv_morning.due_slot($1)::text as slot', [time])).rows[0].slot;
  }
  async function tick(time) {
    return (await db.query('select fv_morning.tick($1) as result', [time])).rows[0].result;
  }
  async function rejected(sql) {
    await assert.rejects(() => db.exec(sql));
  }
  for (const day of ['2026-09-29','2026-12-01','2027-03-14','2026-11-01']) {
    const winter = ['2026-12-01','2026-11-01'].includes(day);
    const startHour = winter ? 12 : 11;
    assert.equal(await slot(`${day}T${startHour}:04:59Z`), null);
    assert.equal(await slot(`${day}T${startHour + 2}:00:00Z`), null);
    assert.equal(Date.parse(await slot(`${day}T${startHour}:05:00Z`)), Date.parse(`${day}T${startHour}:05:00Z`));
    assert.equal(Date.parse(await slot(`${day}T${startHour}:29:00Z`)), Date.parse(`${day}T${startHour}:05:00Z`));
    assert.equal(Date.parse(await slot(`${day}T${startHour + 1}:59:59Z`)), Date.parse(`${day}T${startHour + 1}:35:00Z`));
  }
  // Simulate every five-minute cron tick for a full day (48 eligible UTC ticks).
  let queued = 0;
  for (let h = 11; h <= 14; h++) {
    for (let m = 0; m < 60; m += 5) {
      const result = await tick(`2026-09-29T${h}:${String(m).padStart(2,'0')}:00Z`);
      queued += result.status === 'queued';
      assert.ok(!JSON.stringify(result).includes('fake-restricted-token'));
    }
  }
  assert.equal(queued, 4);
  const requests = (await db.query('select * from net.requests')).rows;
  assert.equal(requests.length, 4);
  for (const request of requests) {
    assert.equal(request.url, 'https://api.github.com/repos/pbwitt/fourth-and-value/actions/workflows/morning-picks.yml/dispatches');
    assert.deepEqual(request.body, {ref:'main',inputs:{trigger_source:'supabase',test_edition:false,replace_card:false}});
    assert.equal(request.timeout_ms, 10000);
    assert.equal(request.headers.Authorization, 'Bearer fake-restricted-token-for-tests');
  }
  assert.equal((await tick('2026-09-29T12:55:00Z')).status, 'already_requested');
  assert.equal((await tick('2026-09-29T16:05:00Z')).status, 'outside_window');
  // Restart the installation without replaying already-issued slots.
  await db.exec(sql);
  assert.equal((await tick('2026-09-29T12:35:00Z')).status, 'already_requested');
  // A late first tick sends one current-slot request, not a backlog of starts.
  assert.equal((await tick('2026-09-30T12:40:00Z')).status, 'queued');
  assert.equal((await db.query('select * from net.requests')).rows.length, 5);

  await db.exec(`insert into net._http_response(id,status_code,timed_out,error_msg) values
    (1,204,false,null),(2,401,false,null),(3,null,true,null),(4,null,false,'do not retain arbitrary transport details');
    select fv_morning.collect_receipts();`);
  const receipts = (await db.query('select * from fv_morning.dispatches order by request_id')).rows;
  assert.deepEqual(receipts.slice(0,4).map(r=>r.result), ['accepted','rejected','timeout','transport_error']);
  assert.ok(!JSON.stringify(receipts).includes('fake-restricted-token'));
  assert.ok(!JSON.stringify(receipts).includes('do not retain'));
  await db.exec("update fv_morning.dispatches set requested_at=now()-interval '16 minutes' where request_id=5; select fv_morning.collect_receipts();");
  assert.equal((await db.query('select result from fv_morning.dispatches where request_id=5')).rows[0].result,'receipt_missing');
  await db.exec("insert into net._http_response(id,status_code,timed_out) values (5,204,false); select fv_morning.collect_receipts();");
  assert.equal((await db.query('select result from fv_morning.dispatches where request_id=5')).rows[0].result,'accepted');
  await db.exec("delete from vault.decrypted_secrets;");
  await rejected("select fv_morning.tick('2026-10-01T11:05:00Z');");
  assert.equal((await db.query('select * from net.requests')).rows.length, 5);
  // No site role, including its service role, can trigger or inspect this job.
  for (const role of ['anon','authenticated','service_role']) {
    await db.exec('set role ' + role);
    await rejected('select * from fv_morning.dispatches');
    await rejected('select fv_morning.tick()');
    await db.exec('reset role');
  }
  console.log('Morning scheduler passed: ET/DST, four bounded dispatches, repeat/install deduplication, request identity, receipts, missing credentials, and private permissions. Transport/extensions remain mocked.');
  await db.close();
})().catch(e => { console.error(e.message); process.exitCode = 1; });

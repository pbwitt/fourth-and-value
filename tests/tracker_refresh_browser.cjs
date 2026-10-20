// An open tracker picks up grades written by the grading job without a reload, and says
// what finished-but-ungraded bets would add. All external requests are intercepted.
const assert = require('node:assert/strict'), fs = require('node:fs'), http = require('node:http'), path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..', 'docs');
const server = http.createServer((req, res) => {
  let name = decodeURIComponent(req.url.split('?')[0]); if (name.endsWith('/')) name += 'index.html';
  const file = path.resolve(root, '.' + name);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (error, data) => {
    if (error) { res.writeHead(404); return res.end(); }
    res.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' })[path.extname(file)] || 'application/octet-stream');
    res.end(data);
  });
});

(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  let browser;
  try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } }), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    await page.route('**/nav.js*', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    await page.route('https://cdn.jsdelivr.net/**', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    await page.clock.install();
    await page.addInitScript(() => {
      const user = { id: 'owner', email: 'reader@example.com' };
      const bet = (id, extra) => ({ id, user_id: 'owner', league: 'NHL', game_date: '2026-06-01', team_home: 'BOS', team_away: 'TOR',
        market_type: 'totals', side: 'over', line: 5.5, player: null, book: 'DraftKings', odds: -110, stake_dollars: 10,
        status: 'pending', payout: null, actual_result: null, created_at: '2026-06-01T12:00:00Z', ...extra });
      window.db = { rows: [bet('won', { status: 'won', payout: 19.09, actual_result: 7 }), bet('open'), bet('also-open', { odds: 150 })],
        selects: 0, delays: [] };
      const copy = value => JSON.parse(JSON.stringify(value));
      window.supabase = { createClient: () => ({
        auth: { getSession: async () => ({ data: { session: { user } } }), onAuthStateChange: () => {} },
        functions: { invoke: async () => ({ data: { games: [] }, error: null }) },
        from: () => ({
          // The rows as they were when the request was sent, answered after any queued delay.
          select: () => ({ order: async () => {
            const rows = copy(db.rows), delay = db.delays.shift() || 0; db.selects++;
            if (delay) await new Promise(resolve => setTimeout(resolve, delay));
            return { data: rows, error: null };
          } }),
        }),
      }) };
    });
    await page.goto(base + '/tracking/');
    await page.locator('#betsTableBody [data-edit-bet="open"]').waitFor();
    // Polls from Node: the page's timers run on Playwright's fake clock.
    const until = async (check, what) => {
      for (let i = 0; i < 100; i++) { if (await check()) return; await new Promise(resolve => setTimeout(resolve, 20)); }
      throw new Error('Timed out: ' + what);
    };
    const pl = () => page.locator('#profitLoss').textContent();
    const grade = (id, status, payout) => page.evaluate(([id, status, payout]) => Object.assign(db.rows.find(r => r.id === id), { status, payout }), [id, status, payout]);
    const show = () => page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
    assert.equal(await pl(), '+$9.09');

    // Returning to the tab picks up a grade without a reload.
    await grade('open', 'lost', 0);
    await show();
    await until(async () => await pl() === '$-0.91', 'graded loss shown');
    assert.equal(await page.locator('#totalReturned').textContent(), '$19.09');
    assert.equal(await page.locator('#totalStaked').textContent(), '$20.00');

    // While a bet is pending the ledger reloads every five minutes; an unchanged ledger is not redrawn.
    const before = await page.evaluate(() => db.selects);
    await page.evaluate(() => { document.querySelector('#betsTableBody tr').dataset.marker = '1'; });
    await page.clock.runFor(5 * 60e3 + 1000);
    await until(() => page.evaluate(n => db.selects > n, before), 'scheduled reload');
    assert(await page.evaluate(() => document.querySelector('#betsTableBody tr').dataset.marker === '1'), 'unchanged ledger keeps the DOM');
    await grade('also-open', 'won', 25);
    await page.clock.runFor(5 * 60e3 + 1000);
    await until(async () => await page.locator('#totalReturned').textContent() === '$44.09', 'second grade shown');
    assert.match(await page.locator('#betsTableBody').textContent(), /\$25/);

    // With nothing pending there is no polling.
    const settledSelects = await page.evaluate(() => db.selects);
    await page.clock.runFor(15 * 60e3);
    assert.equal(await page.evaluate(() => db.selects), settledSelects);

    // A slow earlier load never overwrites a newer one.
    await page.evaluate(() => {
      Object.assign(db.rows.find(r => r.id === 'also-open'), { status: 'pending', payout: null });
      db.delays.push(400);
      window.slowLoad = loadBets();                  // sees also-open pending
      Object.assign(db.rows.find(r => r.id === 'also-open'), { status: 'won', payout: 30 });
      return loadBets();                             // sees it won at $30
    });
    assert.equal(await page.locator('#totalReturned').textContent(), '$49.09');
    await page.clock.runFor(1000);
    await page.evaluate(() => window.slowLoad);
    assert.equal(await page.locator('#totalReturned').textContent(), '$49.09');

    // Finished bets the live view has seen but the grader has not written yet.
    assert.equal(await page.locator('#ungradedNote').textContent(), '');
    await page.evaluate(() => {
      Object.assign(db.rows.find(r => r.id === 'open'), { status: 'pending', payout: null });
      Object.assign(db.rows.find(r => r.id === 'also-open'), { status: 'pending', payout: null });
      return loadBets();
    });
    await page.evaluate(() => {
      liveViews = new Map([['open', { status: 'final', tone: 'lost' }], ['also-open', { status: 'final', tone: 'won' }]]);
      updateUngradedNote();
    });
    assert.equal(await page.locator('#ungradedNote').textContent(), '2 finished bets not graded yet: +$5.00 unofficial');
    assert.equal(await pl(), '+$9.09', 'unofficial results stay out of P/L');
    await page.evaluate(() => { liveViews = new Map([['open', { status: 'live', tone: 'lost' }]]); updateUngradedNote(); });
    assert.equal(await page.locator('#ungradedNote').textContent(), '');
    assert(await page.locator('#ungradedNote').isHidden());

    assert.deepEqual(errors, []);
    console.log('PASS: tracker reloads grades on tab return and while pending, keeps the newest load, and notes ungraded finals.');
  } finally { if (browser) await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });

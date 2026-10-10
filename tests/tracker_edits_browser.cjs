// Exercise the real tracker UI against an in-memory, owner-scoped database.
// All external requests are intercepted; no account or saved bet is touched.
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
    const page = await browser.newPage({ viewport: { width: 390, height: 900 } }), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    await page.route('**/nav.js*', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    await page.route('https://cdn.jsdelivr.net/**', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    await page.addInitScript(() => {
      const user = { id: 'owner', email: 'reader@example.com' };
      const bet = { id: 'win', user_id: 'owner', league: 'MLB', game_date: '2026-06-01', team_home: 'SD', team_away: 'CHC',
        market_type: 'totals', side: 'under', line: 7.5, player: null, book: 'DraftKings', odds: -110, stake_dollars: 25,
        status: 'won', payout: 47.73, actual_result: 5, graded_timestamp: '2026-06-02T06:00:00Z', model_prob: 0.6, edge_bps: 760 };
      window.db = { rows: JSON.parse(sessionStorage.getItem('test-bets') || 'null') || [bet], updates: [], fail: false, delay: 0, user };
      const copy = value => JSON.parse(JSON.stringify(value));
      window.supabase = { createClient: () => ({
        auth: { getSession: async () => ({ data: { session: db.user ? { user: db.user } : null } }), onAuthStateChange: () => {} },
        functions: { invoke: async () => ({ data: { games: [] }, error: null }) },
        from: () => ({
          select: () => ({ order: async () => ({ data: copy(db.rows), error: null }) }),
          update: values => {
            db.updates.push(copy(values)); const filters = [];
            const query = {
              eq(key, value) { filters.push([key, value]); return query; },
              is(key, value) { filters.push([key, value]); return query; },
              select() { return query; },
              async maybeSingle() {
                await new Promise(resolve => setTimeout(resolve, db.delay));
                if (db.fail) return { error: { message: 'offline' }, data: null };
                const row = db.rows.find(r => r.user_id === db.user?.id && filters.every(([key, value]) => (r[key] ?? null) === value));
                if (!row) return { data: null, error: null };
                Object.assign(row, values); sessionStorage.setItem('test-bets', JSON.stringify(db.rows));
                return { data: copy(row), error: null };
              },
            };
            return query;
          },
        }),
      }) };
    });
    await page.goto(base + '/tracking/');
    const mobileEdit = page.locator('#betCards [data-edit-bet="win"]');
    const dialog = page.locator('#editBetDialog');
    await mobileEdit.waitFor();

    // Cancel (including Escape) never saves. Exact original values are prefilled.
    await mobileEdit.click();
    assert.equal(await page.locator('#eb-stake_dollars').inputValue(), '25');
    assert.equal(await page.locator('#eb-status').inputValue(), 'won');
    assert.equal(await page.locator('#eb-payout').inputValue(), '47.73');
    assert.equal(await page.locator('#eb-line').inputValue(), '7.5');
    await page.locator('#eb-stake_dollars').fill('75'); await page.locator('#eb-cancel').click();
    assert.equal(await page.evaluate(() => db.updates.length), 0);
    await mobileEdit.click(); assert.equal(await page.locator('#eb-stake_dollars').inputValue(), '25');
    await page.keyboard.press('Escape'); assert(await dialog.isHidden());
    assert.equal(await page.evaluate(() => document.activeElement.dataset.editBet), 'win');

    // Doubling a winning bet updates that row, its return and all totals.
    await mobileEdit.click(); await page.locator('#eb-stake_dollars').fill('50');
    assert.equal(await page.locator('#eb-payout').inputValue(), '95.45');
    assert.equal(await page.locator('#eb-status').inputValue(), 'won');
    await page.evaluate(() => { db.delay = 300; });
    await page.locator('#eb-save').click();
    assert(await page.locator('#eb-stake_dollars').isDisabled());
    await page.keyboard.press('Escape'); assert(await dialog.isVisible());
    await page.locator('#eb-form').dispatchEvent('submit'); // duplicate submit while saving
    await dialog.waitFor({ state: 'hidden' });
    assert.equal(await page.evaluate(() => db.updates.length), 1);
    assert.equal(await page.evaluate(() => db.rows.length), 1);
    assert.equal(await page.locator('#totalBets').textContent(), '1');
    assert.equal(await page.locator('#totalStaked').textContent(), '$50.00');
    assert.equal(await page.locator('#totalReturned').textContent(), '$95.45');
    assert.equal(await page.locator('#profitLoss').textContent(), '+$45.45');
    assert.match(await page.locator('#betCards').textContent(), /Stake \$50.00/);
    assert.match(await page.locator('#betsTableBody').textContent(), /\$95.45/);
    await page.reload(); await mobileEdit.waitFor();
    assert.equal(await page.locator('#totalReturned').textContent(), '$95.45', 'saved value survives reload');
    const downloadWait = page.waitForEvent('download'); await page.locator('#f-download').click();
    const download = await downloadWait, stream = await download.createReadStream();
    let csv = ''; for await (const chunk of stream) csv += chunk;
    assert.match(csv, /50\.00,won,5,95\.45,45\.45/);
    await page.locator('#betCards [data-share-bet="win"]').click();
    await page.locator('#shareSheet').waitFor({ state: 'visible' });
    assert.match(await page.locator('#shareImage').getAttribute('src'), /^blob:/);
    await page.getByRole('button', { name: 'Close', exact: true }).click();

    // Validation and failed requests keep the draft and never change the displayed ledger.
    await mobileEdit.click(); await page.locator('#eb-odds').fill('50');
    await page.locator('#eb-save').click(); assert.match(await page.locator('#eb-feedback').textContent(), /valid American odds/);
    assert.equal(await page.evaluate(() => db.updates.length), 0);
    await page.locator('#eb-odds').fill('150'); assert.equal(await page.locator('#eb-payout').inputValue(), '125.00');
    await page.evaluate(() => { db.fail = true; });
    await page.locator('#eb-save').click(); await page.waitForFunction(() => document.getElementById('eb-feedback').textContent.includes('could not be saved'));
    assert(await dialog.isVisible()); assert.equal(await page.locator('#eb-odds').inputValue(), '150');
    assert.equal(await page.locator('#totalReturned').textContent(), '$95.45');
    await page.evaluate(() => { db.fail = false; });
    await page.locator('#eb-save').click(); await dialog.waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#totalReturned').textContent(), '$125.00');

    // A grade/edit in another tab causes a conflict and reloads the latest row.
    await mobileEdit.click(); await page.locator('#eb-stake_dollars').fill('60');
    await page.evaluate(() => { db.rows[0].book = 'FanDuel'; });
    await page.locator('#eb-save').click();
    await page.waitForFunction(() => document.getElementById('eb-feedback').textContent.includes('changed or was deleted'));
    assert.equal(await page.evaluate(() => db.rows[0].stake_dollars), 50);
    await page.locator('#eb-cancel').click(); await mobileEdit.click();
    assert.equal(await page.locator('#eb-book').inputValue(), 'FanDuel');
    await page.locator('#eb-cancel').click();

    // Desktop editing and manual settlement; resetting a selection clears the old grade.
    await page.setViewportSize({ width: 1440, height: 1000 });
    const desktopEdit = page.locator('#betsTableBody [data-edit-bet="win"]');
    await desktopEdit.click(); await page.locator('#eb-status').selectOption('push');
    assert.equal(await page.locator('#eb-payout').inputValue(), '50.00'); assert(await page.locator('#eb-payout').isDisabled());
    await page.locator('#eb-save').click(); await dialog.waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#profitLoss').textContent(), '+$0.00');
    await desktopEdit.click(); await page.locator('#eb-details summary').click();
    await page.locator('#eb-line').fill('0');
    assert(await page.locator('#eb-reset').isVisible()); assert.equal(await page.locator('#eb-status').inputValue(), 'pending');
    await page.locator('#eb-save').click(); await dialog.waitFor({ state: 'hidden' });
    const resetBet = await page.evaluate(() => db.rows[0]);
    assert.equal(resetBet.line, 0); assert.equal(resetBet.status, 'pending'); assert.equal(resetBet.payout, null);
    assert.equal(resetBet.actual_result, null); assert.equal(resetBet.model_prob, null);
    await desktopEdit.click(); await page.locator('#eb-status').selectOption('won');
    await page.locator('#eb-payout').fill('130'); await page.locator('#eb-save').click(); await dialog.waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#totalReturned').textContent(), '$130.00');

    // Undoing a pick change restores a custom result instead of replacing its return.
    await desktopEdit.click(); await page.locator('#eb-payout').fill('135');
    await page.locator('#eb-details summary').click(); await page.locator('#eb-line').fill('1');
    assert.equal(await page.locator('#eb-status').inputValue(), 'pending');
    await page.locator('#eb-line').fill('0');
    assert.equal(await page.locator('#eb-status').inputValue(), 'won');
    assert.equal(await page.locator('#eb-payout').inputValue(), '135');
    await page.locator('#eb-cancel').click();

    // Moneyline and legacy market names can be edited without inventing a line or changing the pick.
    await page.evaluate(async () => {
      Object.assign(db.rows[0], { market_type: 'moneyline', side: 'SD', line: null });
      await loadBets();
    });
    await desktopEdit.click();
    assert.equal(await page.locator('#eb-line').inputValue(), '');
    assert.equal(await page.locator('#eb-market_type').inputValue(), 'moneyline');
    await page.locator('#eb-book').fill('BetMGM');
    await page.locator('#eb-save').click(); await dialog.waitFor({ state: 'hidden' });
    assert.equal(await page.evaluate(() => db.rows[0].line), null);
    assert.equal(await page.evaluate(() => db.rows[0].status), 'won');
    assert.equal(await page.evaluate(() => db.rows[0].payout), 130);

    // Narrow-screen fit with the expanded form, and no runtime errors.
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await (width < 821 ? mobileEdit : desktopEdit).click(); await page.locator('#eb-details summary').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `page fits ${width}`);
      assert.equal(await dialog.evaluate(el => el.scrollWidth > el.clientWidth + 1), false, `dialog fits ${width}`);
      if (process.env.TRACKER_EDIT_SCREENSHOTS) await page.screenshot({ path: path.join(process.env.TRACKER_EDIT_SCREENSHOTS, `tracker-edit-${width}.png`) });
      await page.locator('#eb-cancel').click();
    }
    assert.deepEqual(errors, []);
    console.log('PASS: desktop/mobile edit, cancel, duplicate submission, validation, errors, conflicts, settlement, persistence, CSV and totals.');
  } finally { if (browser) await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });

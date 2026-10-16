// Every chart gets a Share button; the image is the branded card with the chart's explanation.
const assert = require('node:assert/strict'), fs = require('node:fs'), http = require('node:http'), path = require('node:path');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '../docs');
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png'};
const server = http.createServer((req, res) => {
  let name = decodeURIComponent(req.url.split('?')[0]);
  if (name.endsWith('/')) name += 'index.html';
  const file = path.resolve(root, '.' + name);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (e, b) => {
    if (e) { res.writeHead(404); return res.end(); }
    res.setHeader('Content-Type', TYPES[path.extname(file)] || 'application/octet-stream');
    res.end(b);
  });
});
const STUDY = '/research/low-scoring-games-situational-factors.html';

// The card is not blank: a mint bar on top and many colours where the chart is drawn.
async function cardStats(page) {
  return page.evaluate(async () => {
    const img = document.querySelector('.fv-cs-preview img');
    const c = document.createElement('canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const bar = [...ctx.getImageData(40, 4, 1, 1).data].slice(0, 3);
    const colours = new Set();
    for (let y = Math.round(c.height * 0.3); y < c.height * 0.75; y += 24) {
      const row = ctx.getImageData(0, y, c.width, 1).data;
      for (let i = 0; i < row.length; i += 12) colours.add(`${row[i] >> 3},${row[i + 1] >> 3},${row[i + 2] >> 3}`);
    }
    return {w: c.width, h: c.height, bar, colours: colours.size};
  });
}

async function openCard(page, selector) {
  await page.locator(`${selector} .fv-share-btn`).click();
  await page.waitForFunction(() => document.querySelector('.fv-cs-preview img[src]') || /couldn/.test(document.querySelector('.fv-cs-status').textContent), null, {timeout: 20000});
  assert.equal(await page.locator('.fv-cs-status').textContent(), '', 'the image was drawn');
  return cardStats(page);
}

(async () => {
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({headless: true, ...(process.env.CHROME_PATH ? {executablePath: process.env.CHROME_PATH} : {})});
  try {
    for (const width of [390, 1440]) {
      const page = await browser.newPage({viewport: {width, height: 900}}), errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      // Fonts come from Google; the test does not depend on that network.
      await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
      await page.goto(base + STUDY);
      await page.waitForFunction(() => document.querySelectorAll('.fv-share-btn').length === 5);
      assert.deepEqual(await page.$$eval('[data-fv-share=ready]', els => els.map(e => e.id)), ['figure-1', 'figure-2', 'figure-3', 'figure-4', 'figure-5']);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'no horizontal scroll');

      const m = await page.evaluate(() => window.__fvChartShare.describe(document.getElementById('figure-1')));
      assert.equal(m.title, 'What makes NFL games go under?');
      assert.match(m.subtitle, /^Of 18 situations, one held up in both decades: Strong wind \(15\+ mph\) → under won 55\.9% \(113–89\) in 2016–2025\./);
      assert.match(m.text, /^Each row is a situation/);
      assert.equal(m.notes, "NFL: share of bets won on each situation's predicted side. Blue rows met the pre-registered replication rule. Values beyond the axis are drawn at its edge.");
      assert.match(m.source, /^Data: 5,431 NFL games/);
      assert.match(m.from, /^What Makes Games Go Under\? Weather/);
      assert.equal(m.date, 'October 16, 2026');
      assert.equal(m.label, 'Research · FV-2026-04');
      assert.equal(m.url, `https://fourthandvalue.com${STUDY}#figure-1`);

      const card = await openCard(page, '#figure-1');
      assert.equal(card.w, 2400);
      assert.ok(card.h > 1800 && card.h < 4000, `card height ${card.h}`);
      assert.deepEqual(card.bar, [124, 226, 189], 'mint brand bar');
      assert.ok(card.colours > 12, `chart drawn (${card.colours} colours)`);
      if (width === 390) {
        assert.match(await page.locator('[data-act=save]').getAttribute('download'), /^fourth-and-value-low-scoring-games-situational-factors-figure-1\.png$/);
        await page.evaluate(() => { window.__opened = []; window.open = u => { window.__opened.push(u); return null; }; });
        await page.locator('[data-social=x]').click();
        await page.locator('[data-social=bluesky]').click();
        const [x, bsky] = await page.evaluate(() => window.__opened);
        assert.match(x, /^https:\/\/twitter\.com\/intent\/tweet\?text=What%20makes%20NFL/);
        assert.ok(decodeURIComponent(x).endsWith(`${STUDY}?utm_source=x&utm_medium=social&utm_campaign=chart_share#figure-1`));
        assert.ok(decodeURIComponent(bsky.split('text=')[1]).length <= 300, 'Bluesky post fits 300 characters');
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('.fv-cs-sheet.open').count(), 0);
        assert.equal(await page.evaluate(() => document.body.style.overflow), '');
      }
      await page.screenshot({path: `/tmp/fv-chart-share-${width}.png`});
      assert.deepEqual(errors, []);
      await page.close();
    }

    // Blog charts explain themselves from the page: section heading, SVG title, lead paragraph, caption.
    const page = await browser.newPage({viewport: {width: 1280, height: 900}}), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
    await page.goto(base + '/blog/chris-sale-pitcher-outs-2026.html#chart-2');
    await page.waitForFunction(() => document.querySelectorAll('.fv-share-btn').length === 4);
    await page.waitForFunction(() => document.getElementById('chart-2').classList.contains('fv-share-target'));
    const b = await page.evaluate(() => window.__fvChartShare.describe(document.getElementById('chart-2')));
    assert.equal(b.title, 'Step 3: Get three opinions on the same start');
    assert.match(b.subtitle, /^Chance of 17 or fewer, exactly 18, and 19 or more outs/);
    assert.match(b.text, /^We split the start into three results/);
    assert.match(b.notes, /quoting 17\.5 and 18\.5/);
    assert.equal(b.label, 'Analysis');
    assert.equal(b.date, 'October 7, 2026');
    const blog = await openCard(page, '#chart-2');
    assert.equal(blog.w, 2400);
    assert.ok(blog.colours > 12);
    // The page's chart CSS (svg at width:100%) must not reach the button's icon; one copy of the script runs.
    const btn = await page.evaluate(() => {
      const b = document.querySelector('.fv-share-btn'), r = b.getBoundingClientRect(), i = b.querySelector('svg').getBoundingClientRect();
      return {h: r.height, icon: i.width, scripts: document.querySelectorAll('script[src*="chart-share.js"]').length};
    });
    assert.ok(btn.h <= 40 && btn.icon === 15, `button ${btn.h}px, icon ${btn.icon}px`);
    assert.equal(btn.scripts, 1);

    // A browser still holding an old nav.js (Cloudflare can keep it for hours) gets the buttons from the page's own tag.
    const stale = await browser.newPage();
    stale.on('pageerror', e => errors.push(e.message));
    await stale.route('**/nav.js*', r => r.fulfill({contentType: 'text/javascript', body: ''}));
    await stale.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
    await stale.goto(base + '/blog/week-5-recap-week-6-preview-2026.html');
    await stale.waitForFunction(() => document.querySelectorAll('.fv-share-btn').length === 3);
    await stale.goto(base + STUDY);
    await stale.waitForFunction(() => document.querySelectorAll('.fv-share-btn').length === 5);
    // The study links to its pre-registration on this site, and to nothing on GitHub.
    assert.equal(await stale.locator('a[href*="github.com"]').count(), 0);
    const prereg = await stale.locator('.download-bar a', {hasText: 'Pre-registration'}).getAttribute('href');
    assert.equal(prereg, './low-scoring-games-situational-factors-preregistration.html');
    await stale.goto(new URL(prereg, base + STUDY).href);
    assert.match(await stale.locator('h1').textContent(), /pre-registered hypotheses \(FV-2026-04\)/);
    await stale.close();

    // A chart page without nav.js, and Market Results, which keeps its own share sheet.
    await page.goto(base + '/blog/mlb-playoff-strikeouts-2026.html');
    await page.waitForFunction(() => document.querySelectorAll('.fv-share-btn').length === 5);
    await page.goto(base + '/markets/?sport=nfl');
    await page.waitForFunction(() => document.querySelectorAll('.share-chart').length > 0);
    assert.equal(await page.locator('.fv-share-btn').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS: every chart shares a branded image with its headline, takeaway, explanation, notes and source; links and posts point back to the chart.');
  } finally {
    await browser.close();
    server.close();
  }
})().catch(e => { console.error(e); server.close(); process.exitCode = 1; });

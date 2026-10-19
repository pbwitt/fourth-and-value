// Exercise the real tracker sign-in screen against a stubbed Supabase Auth.
// All external requests are intercepted; no email is sent and no account is touched.
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
    const page = await browser.newPage({ viewport: { width: 960, height: 1000 } }), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    await page.route('**/nav.js*', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    await page.route('https://cdn.jsdelivr.net/**', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    await page.addInitScript(() => {
      window.auth = { user: null, otp: [], verify: [], listeners: [] };
      window.supabase = { createClient: () => ({
        auth: {
          getSession: async () => ({ data: { session: auth.user ? { user: auth.user } : null } }),
          onAuthStateChange: cb => { auth.listeners.push(cb); return { data: { subscription: { unsubscribe() {} } } }; },
          signInWithOtp: async args => { auth.otp.push(JSON.parse(JSON.stringify(args))); return { data: {}, error: null }; },
          verifyOtp: async args => {
            auth.verify.push({ ...args });
            if (args.token !== '123456') return { data: {}, error: { code: 'otp_expired', status: 403, message: 'Token has expired or is invalid' } };
            auth.user = { id: 'owner', email: args.email };
            auth.listeners.forEach(cb => cb('SIGNED_IN', { user: auth.user }));
            return { data: { session: { user: auth.user } }, error: null };
          },
        },
        functions: { invoke: async () => ({ data: { games: [] }, error: null }) },
        from: () => ({ select: () => ({ order: async () => ({ data: [], error: null }) }) }),
      }) };
    });
    const status = page.locator('#signInStatus'), codeStatus = page.locator('#codeStatus');

    // A failed link explains itself and its error fragment is cleared from the address.
    await page.goto(base + '/tracking/#error=access_denied&error_code=otp_expired&error_description=Email+link+is+invalid+or+has+expired&sb=');
    await page.locator('#signInScreen').waitFor();
    assert.match(await status.textContent(), /already used or has expired/);
    assert.equal(await page.evaluate(() => location.hash), '');

    // The emailed link returns to this page without any fragment, keeping the query.
    await page.goto(base + '/tracking/?from=tv#stale');
    await page.locator('#emailInput').fill('reader@example.com');
    await page.locator('#sendLinkBtn').click();
    await page.waitForFunction(() => auth.otp.length === 1);
    assert.deepEqual(await page.evaluate(() => auth.otp[0]), { email: 'reader@example.com', options: { emailRedirectTo: base + '/tracking/?from=tv' } });
    assert.match(await status.textContent(), /enter its 6-digit code below/);

    // Code entry validates locally before calling Supabase.
    await page.locator('#codeInput').fill('12ab');
    await page.locator('#verifyCodeBtn').click();
    assert.match(await codeStatus.textContent(), /6-digit code/);
    await page.locator('#emailInput').fill('');
    await page.locator('#codeInput').fill('123456');
    await page.locator('#verifyCodeBtn').click();
    assert.match(await codeStatus.textContent(), /email address the code was sent to/);
    assert.equal(await page.evaluate(() => auth.verify.length), 0);

    // A wrong or used code is explained and leaves the reader signed out.
    await page.locator('#emailInput').fill('reader@example.com');
    await page.locator('#codeInput').fill('654321');
    await page.locator('#verifyCodeBtn').click();
    await page.waitForFunction(() => /incorrect, expired or already used/.test(document.getElementById('codeStatus').textContent));
    assert(await page.locator('#betTracker').isHidden());

    // The right code (spaces allowed, Enter submits) signs in this browser.
    await page.locator('#codeInput').fill(' 123 456 ');
    await page.locator('#codeInput').press('Enter');
    await page.locator('#betTracker').waitFor();
    assert.deepEqual(await page.evaluate(() => auth.verify.at(-1)), { email: 'reader@example.com', token: '123456', type: 'email' });
    assert(await page.locator('#signInScreen').isHidden());
    assert.equal(await page.locator('#accountEmail').textContent(), 'reader@example.com');

    assert.deepEqual(errors, []);
    console.log('PASS: failed-link message, clean redirect, code validation, wrong code and code sign-in.');
  } finally { if (browser) await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });

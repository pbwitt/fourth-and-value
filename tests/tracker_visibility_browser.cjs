// Bet Tracker is unlisted: anyone with the /tracking/ link can use it, but only a browser
// signed in to it sees its entry points (nav link, Track bet buttons and columns,
// .fv-tracker-only links). Story suggestions are closed and collect nothing.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..','docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
const SESSION='sb-fzjonxpzsrbdhbujbhsn-auth-token';
// The shapes the boards render: NFL totals (.track-cell, Track header last), NHL player
// pages ([data-offer], Track header last), Today's Picks (.track-pick) and tagged links.
const FIXTURE=`<!doctype html><html lang="en"><head><title>Fixture</title></head><body><div id="nav-root"></div><script src="/nav.js?v=47"></script>
<table id="totals"><thead><tr><th>Book</th><th>Total</th><th>Track</th></tr></thead><tbody><tr><td>DraftKings</td><td>43.5</td><td class="track-cell"><button type="button" data-fv-ticket="a">Over 43.5</button></td></tr></tbody></table>
<table id="players"><thead><tr><th>Model</th><th>Track</th></tr></thead><tbody><tr><td>55%</td><td><span data-offer="0"><button type="button" data-fv-ticket="b">Track bet</button></span></td></tr></tbody></table>
<div class="prop-card"><div class="actions fv-track-actions"><button type="button" data-fv-ticket="c">Track bet</button></div></div>
<p><button type="button" class="track-pick" id="pick">Track bet</button></p>
<p class="fv-tracker-only" id="note">Bet Tracker settles bets.</p><footer>Fourth &amp; Value</footer></body></html>`;

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`,browser=await chromium.launch({headless:true});
  try{
    for(const signedIn of [false,true]){
      const context=await browser.newContext({viewport:{width:1280,height:900}});
      await context.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
      await context.route('**/fixture.html',route=>route.fulfill({contentType:'text/html',body:FIXTURE}));
      if(signedIn)await context.addInitScript(key=>localStorage.setItem(key,'{"access_token":"test"}'),SESSION);
      const page=await context.newPage(),errors=[];page.on('pageerror',error=>errors.push(error.message));
      const who=signedIn?'signed-in browser':'signed-out visitor';
      const shown=async selector=>page.locator(selector).first().isVisible();

      await page.goto(base+'/fixture.html');await page.waitForSelector('.fv-nav');
      assert.equal(await page.locator('.fv-nav a[href$="/tracking/"]').count(),signedIn?1:0,`nav Bet Tracker link for a ${who}`);
      for(const selector of ['#totals th:last-child','#totals .track-cell','#players th:last-child','#players td:last-child','.fv-track-actions','#pick','#note'])
        assert.equal(await shown(selector),signedIn,`${selector} for a ${who}`);
      for(const selector of ['#totals th:first-child','#totals td:nth-child(2)','#players td:first-child'])
        assert.equal(await shown(selector),true,`${selector} always shows`);
      assert.deepEqual(errors,[]);

      // Home footer: the editorial sign-in link is tracker-only; story suggestions are gone.
      await page.goto(base+'/');await page.waitForSelector('.fv-nav');
      assert.equal(await page.locator('a[href="/editorial/suggest.html"]').count(),0,'no Suggest a story link');
      assert.equal(await shown('footer a[href="/editorial/inbox.html"]'),signedIn,`Editorial desk link for a ${who}`);

      // Arbitrage: Live odds and Bet Tracker links and the editors' email sign-in panel.
      await page.goto(base+'/arbitrage/');await page.waitForSelector('.fv-nav');
      assert.equal(await shown('a[href="/tracking/"]'),signedIn,`arbitrage Bet Tracker link for a ${who}`);
      assert.equal(await shown('a[href="/live/"]'),signedIn,`arbitrage Live odds link for a ${who}`);
      assert.equal(await shown('#scan-panel'),signedIn,`Scan now sign-in panel for a ${who}`);

      // The tracker itself always works from its link, with its nav entry.
      await page.goto(base+'/tracking/');await page.waitForSelector('.fv-nav');
      assert.equal(await page.locator('.fv-nav a[href$="/tracking/"]').count(),1,'Bet Tracker nav link on the tracker');
      assert.equal(await page.evaluate(()=>document.documentElement.classList.contains('fv-tracker-off')),false,'the tracker page is never gated');
      assert.match(await page.locator('meta[name=robots]').getAttribute('content'),/noindex/);

      // Story suggestions: closed, nothing to fill in, out of search.
      await page.goto(base+'/editorial/suggest.html');
      assert.equal(await page.locator('h1').textContent(),'Suggestions are closed');
      assert.equal(await page.locator('form, input, textarea').count(),0,'no suggestion form');
      assert.match(await page.locator('meta[name=robots]').getAttribute('content'),/noindex/);
      await context.close();
    }
    const sitemap=fs.readFileSync(path.join(root,'sitemap.xml'),'utf8');
    assert.doesNotMatch(sitemap,/\/tracking\/|editorial\/suggest\.html/,'unlisted pages stay out of the sitemap');
    console.log('PASS: Bet Tracker entry points show only to signed-in browsers; the tracker works from its link; suggestions are closed.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exit(1);});

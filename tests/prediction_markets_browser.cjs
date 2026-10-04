const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../docs');
const original=JSON.parse(execFileSync(process.env.PYTHON||'python3',[path.join(__dirname,'test_prediction_markets.py'),'--fixture'],{encoding:'utf8'}));
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  try {
    const page=await browser.newPage(),errors=[];let snapshot=structuredClone(original),fail=false;
    page.on('pageerror',e=>errors.push(e.message));
    await page.clock.install({time:new Date('2026-10-04T16:00:00Z')});
    await page.route('**/snapshot.json',r=>fail?r.fulfill({status:503,body:'unavailable'}):r.fulfill({json:snapshot}));
    await page.route('**/trade-api/**',()=>{throw Error('Browser must not call exchange APIs');});
    const url=`http://127.0.0.1:${server.address().port}/prediction-markets/`;
    for(const width of [320,768,1440]) {
      await page.setViewportSize({width,height:1000});await page.goto(url);
      await page.waitForSelector('.contract');
      assert.match(await page.locator('#snapshot-status').textContent(),/Saved snapshot/);
      assert.match(await page.locator('.contract').textContent(),/\$6.05/);
      assert.equal(await page.locator('h1').count(),1);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'no page overflow');
      await page.selectOption('#contract-count','100');
      assert.match(await page.locator('tbody').textContent(),/100/);
      await page.locator('.contract summary').click();
      assert.match(await page.locator('.contract details').textContent(),/Ties settle at \$0.50/);
      assert.match(await page.locator('.contract details').textContent(),/No verified equivalent sportsbook/);
      await page.evaluate(()=>window.scrollTo(0,0));
      if(width===1440)await page.screenshot({path:'/tmp/fv-kalshi-desktop.png',fullPage:true});
      if(width===320)await page.screenshot({path:'/tmp/fv-kalshi-mobile.png',fullPage:true});
    }
    await page.clock.fastForward(16000);
    assert(await page.locator('.contract details').evaluate(e=>e.open),'rules remain open during freshness update');
    await page.clock.fastForward(901000);
    assert.match(await page.locator('#snapshot-status').textContent(),/Historical snapshot/);
    assert.match(await page.locator('.badge').textContent(),/Historical quote/);
    fail=true;await page.clock.fastForward(300000);
    await page.waitForFunction(()=>document.getElementById('snapshot-status').textContent.includes('Refresh failed'));
    assert.equal(await page.locator('.contract').count(),1,'retain dated snapshot after failure');
    fail=false;snapshot=structuredClone(original);snapshot.rows[0].title='<img src=x onerror=alert(1)>';
    snapshot.rows[0].fee=null;
    for(const estimates of Object.values(snapshot.rows[0].estimates))for(const q of Object.values(estimates)){
      q.fee_estimate_dollars=null;q.total_estimate_dollars=null;q.cost_per_contract_dollars=null;q.payout_equivalent_american=null;
    }
    await page.goto(url);await page.waitForSelector('.contract');
    assert.match(await page.locator('.contract').textContent(),/Fees unavailable/);
    assert.equal(await page.locator('.contract img').count(),0,'API content must be escaped');
    snapshot={...original,rows:[],status:'error'};await page.goto(url);
    await page.waitForFunction(()=>document.getElementById('contracts').textContent.includes('feed is unavailable'));
    fail=true;await page.goto(url);
    await page.waitForFunction(()=>document.getElementById('snapshot-status').textContent.includes('unavailable'));
    assert.equal(await page.locator('.contract').count(),0);
    assert.deepEqual(errors,[]);
    console.log('Prediction-market browser checks passed');
  } finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const {fixture}=require('./briefing_picks.cjs');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  try {
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    const now=Date.parse('2026-09-27T12:00:00Z');await page.clock.install({time:now});let feeds=fixture(now);
    const urls={'/props/top-picks.json':'NFL','/mlb/data/latest.json':'MLB','/nhl/data/latest.json':'NHL','/nhl/data/candidates.json':'NHLBoard'};
    for(const [url,key] of Object.entries(urls))await page.route('**'+url,r=>feeds[key]?r.fulfill({json:feeds[key]}):r.fulfill({status:503,body:'Unavailable'}));
    for(const width of [390,768,1440]) {
      await page.setViewportSize({width,height:1000});await page.goto(base+'/briefing/');
      await page.waitForFunction(()=>document.getElementById('picks-status').textContent.startsWith('3 candidates'));
      assert.equal(await page.locator('#daily-picks-rows tr').count(),3);
      assert.deepEqual((await page.locator('h2').allTextContents()).slice(0,2),['The price rundown',"Today's picks"]);
      assert.equal(await page.getByText('What changed and what’s next',{exact:true}).count(),0);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      assert.match(await page.locator('#daily-picks-rows').textContent(),/7:59:00 AM ET/);
      if(await page.locator('.fv-burger').isVisible())await page.locator('.fv-burger').click();
      await page.locator('.nhl-sport button').click();
      assert(await page.locator('.nhl-sport').getByRole('link',{name:'Top Picks',exact:true}).isVisible());
      assert.equal(await page.locator('.nfl-sport').getByRole('link',{name:'Insights archive',exact:true}).count(),0);
      await page.locator('.nhl-sport button').click();
      if(await page.locator('.fv-burger').isVisible())await page.locator('.fv-burger').click();
      await page.locator('#daily-picks').scrollIntoViewIfNeeded();
      await page.screenshot({path:`/tmp/fv-briefing-picks-${width}.png`,fullPage:true});
    }
    // An open page withdraws expired quotes without a manual reload.
    await page.clock.fastForward(31*60e3);await page.waitForFunction(()=>document.getElementById('picks-status').textContent.startsWith('2 candidates'));
    feeds.MLB=null;await page.reload();await page.waitForFunction(()=>document.getElementById('picks-coverage').textContent.includes('MLB: Current model list unavailable'));
    assert.equal(await page.locator('#daily-picks-rows tr').count(),1);
    feeds={};await page.reload();await page.waitForFunction(()=>document.getElementById('picks-status').textContent.startsWith('0 candidates'));
    assert.match(await page.locator('#daily-picks-rows').textContent(),/No current bets qualify/);
    await page.screenshot({path:'/tmp/fv-briefing-empty.png',fullPage:true});
    await page.goto(base+'/props/insights.html');await page.waitForURL(base+'/nfl/');
    await page.goto(base+'/briefing/2026-09-26.html');assert.equal(await page.locator('#daily-picks').count(),0,'dated archive must never show current picks');
    assert.deepEqual(errors,[]);console.log('PASS: briefing tables, navigation, desktop/mobile, quote expiry, failures and archive redirect.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});

// Today's Picks can show one sportsbook at a time: rows hide without renumbering, the choice
// lasts for the Eastern day, and an emailed pick at another book still opens.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const {fixture}=require('./briefing_picks.cjs');
const {collect,shortlist,reviewKey,reviewBetKey,bookOptions}=require('../docs/assets/briefing-picks.js');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});

// Unit: books with published picks, most first; before a card exists, the pool's books.
{
  const r=(book,label)=>({book,book_label:label});
  assert.deepEqual(bookOptions([r('b','Book B'),r('a','Book A'),r('b','Book B')],[r('c','Book C')]).map(o=>[o.key,o.picks]),[['b',2],['a',1]]);
  assert.deepEqual(bookOptions([],[r('c','Book C'),r('a','Book A'),r('c','Book C')]).map(o=>[o.key,o.offers]),[['c',2],['a',1]]);
  assert.equal(bookOptions([{book:'draftkings',bookmaker_title:'DraftKings'}],[])[0].label,'DraftKings');
}

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  try {
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    const now=Date.parse('2026-09-27T12:00:00Z');await page.clock.install({time:now});
    let feeds=fixture(now);
    const original=collect(feeds,now).selected.find(r=>r.sport==='MLB');
    const reviewed={...original,review_key:reviewKey(original),review_bet_key:reviewBetKey(original),offer_id:'o',forecast_id:'f',qualitative_review:{
      offer_id:'o',forecast_id:'f',status:'needs_information',reviewed_at:new Date(now).toISOString(),countercase:'Original risk.',open_checks:[],evidence:[],
      assessment:{verdict:'consider',reason:'Original assessment.',model_case:'Original projection.',price_case:'Original quote.',context_case:'No verified news.',blocking_checks:[]}}};
    feeds.Reviews={schema_version:1,sports:{MLB:{decision_date:'2026-09-27',candidates:[reviewed],sources:[]}}};
    const first=shortlist(collect(feeds,now).selected,now)[0];
    // Three published picks: one at Book A, two at Book B.
    const rows=[first,{...first,player:'Second player',book:'b',book_label:'Book B',game_id:'g2'},{...first,player:'Third player',book:'b',book_label:'Book B',game_id:'g3'}];
    const card={schema_version:1,kind:'morning',decision_date:'2026-09-27',published_at:new Date(now).toISOString(),edition_id:'edition-a',status:'published',rows};
    feeds={Card:card,Afternoon:{schema_version:1,kind:'afternoon',decision_date:'2026-09-27',published_at:new Date(now).toISOString(),edition_id:'edition-p',
      morning_edition_id:'edition-a',status:'published',rows:[{...first,player:'Afternoon player',market:'pitcher_outs',market_label:'Outs',game_id:'g4'}]}};
    for(const [url,k] of Object.entries({'/briefing/morning-card.json':'Card','/briefing/afternoon-card.json':'Afternoon'}))
      await page.route('**'+url,r=>feeds[k]?r.fulfill({json:feeds[k]}):r.fulfill({status:503,body:'Unavailable'}));
    await page.route('**/*.json',r=>r.fallback());
    for(const width of [320,1440]) {
      await page.setViewportSize({width,height:900});
      await page.goto(base+'/briefing/');await page.evaluate(()=>localStorage.clear());await page.reload();
      await page.waitForSelector('#picks-book-filter button');
      assert.deepEqual(await page.locator('#picks-book-filter button').allTextContents(),['All books','Book A · 2','Book B · 2']);
      assert.equal(await page.locator('#picks-book-filter [aria-pressed="true"]').textContent(),'All books');
      assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),3);
      // One book: Book B's two morning picks keep their original Track bet numbers; the afternoon
      // pick at Book A gives way to a clear message, not an empty table.
      await page.locator('#picks-book-filter button',{hasText:'Book B'}).click();
      assert.deepEqual(await page.locator('#daily-picks-rows [data-track-pick]').evaluateAll(b=>b.map(x=>x.dataset.trackPick)),['1','2']);
      assert.match(await page.locator('#picks-book-status').textContent(),/^Showing Book B only: 2 of 4 published picks today/);
      assert.match(await page.locator('#afternoon-picks-rows').textContent(),/No afternoon picks at Book B\. Show all books/);
      assert.equal(await page.locator('#picks-book-filter [aria-pressed="true"]').textContent(),'Book B · 2');
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'no sideways scrolling');
      if(width===320)assert((await page.locator('#picks-book-filter button').first().boundingBox()).height>=30,'chips stay tappable');
      await page.locator('#daily-picks-rows [data-track-pick="2"]').click();
      assert.match(await page.locator('#track-bet-description').textContent(),/Third player · .* · Book B$/);
      await page.locator('#track-cancel').click();
      // Switching to the sportsbook app and back can reload the page: the choice holds for the day.
      await page.reload();await page.waitForSelector('#picks-book-filter [aria-pressed="true"]');
      assert.equal(await page.locator('#picks-book-filter [aria-pressed="true"]').textContent(),'Book B · 2');
      assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),2);
      await page.locator('#afternoon-picks-rows [data-book=""]').click();
      assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),3);
      assert(await page.locator('#picks-book-status').isHidden());
      await page.screenshot({path:`/tmp/fv-book-filter-${width}.png`,fullPage:true});
    }
    // A ?book= link selects a book; a saved choice from another day is ignored.
    await page.goto(base+'/briefing/?book=a#daily-picks');await page.waitForSelector('#picks-book-filter [aria-pressed="true"]');
    assert.equal(await page.locator('#picks-book-filter [aria-pressed="true"]').textContent(),'Book A · 2');
    await page.evaluate(()=>localStorage.setItem('fv-picks-book',JSON.stringify({book:'b',day:'2026-09-26'})));
    await page.goto(base+'/briefing/');await page.waitForSelector('#picks-book-filter button');
    assert.equal(await page.locator('#picks-book-filter [aria-pressed="true"]').textContent(),'All books');
    // An emailed pick at Book A opens even while Book B is chosen.
    await page.locator('#picks-book-filter button',{hasText:'Book B'}).click();
    await page.goto(base+'/briefing/?edition=edition-a&pick=0&stake=2#daily-picks');
    await page.waitForSelector('#pick-tracker[open]');
    assert.match(await page.locator('#track-bet-description').textContent(),/Book A$/);
    // One book on the card: no filter to show.
    card.rows=[first];delete feeds.Afternoon;
    await page.goto(base+'/briefing/');await page.waitForSelector('#daily-picks-rows [data-track-pick]');
    assert(await page.locator('#picks-book-filter').isHidden());
    assert.deepEqual(errors,[]);
    console.log('PASS: Today’s Picks filters by sportsbook without renumbering, remembers the day’s choice and still opens emailed picks.');
  } finally {await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exit(1);});

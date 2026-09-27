const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const {fixture}=require('./briefing_picks.cjs');
const {collect,shortlist,reviewKey,reviewBetKey}=require('../docs/assets/briefing-picks.js');
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
    await page.addInitScript(()=>{
      window.trackerTest={user:{id:'test-user'},writes:[],error:null,rows:[{id:'b7c9ba55-1234-4234-8234-123456789abc',league:'MLB',player:'<img src=x onerror=alert(1)>',status:'pending',stake_dollars:25,odds:110}]};
      window.supabase={createClient:()=>({auth:{getSession:async()=>({data:{session:window.trackerTest.user?{user:window.trackerTest.user}:null}}),onAuthStateChange:()=>{}},
        from:()=>({insert:async row=>{window.trackerTest.writes.push(row);return {error:window.trackerTest.error};},select:()=>({order:async()=>({data:window.trackerTest.rows,error:null})})})})};
    });
    await page.route('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2',r=>r.fulfill({contentType:'text/javascript',body:'/* fake auth client installed by this test */'}));
    await page.route('**/*.supabase.co/**',r=>{throw Error('Test must never call a real tracker database');});
    async function openReview(){const pool=page.locator('#picks-research-pool');if(await pool.locator('.pick-research').count())await pool.evaluate(e=>e.open=true);const d=page.locator('.pick-research').first();if(!await d.evaluate(e=>e.open))await d.locator(':scope > summary').click();}
    const urls={'/briefing/morning-card.json':'Card','/props/top-picks.json':'NFL','/mlb/data/latest.json':'MLB','/nhl/data/latest.json':'NHL','/nhl/data/candidates.json':'NHLBoard','/briefing/reviews.json':'Reviews'};
    for(const [url,key] of Object.entries(urls))await page.route('**'+url,r=>feeds[key]?r.fulfill({json:feeds[key]}):r.fulfill({status:503,body:'Unavailable'}));
    for(const width of [320,390,768,1440]) {
      await page.setViewportSize({width,height:1000});await page.goto(base+'/briefing/');
      await page.waitForFunction(()=>document.getElementById('research-pool-label').textContent.includes('3 additional'));
      assert.match(await page.locator('#picks-status').textContent(),/0 reviewed picks/);
      await page.locator('#picks-research-pool').evaluate(e=>e.open=true);
      assert.equal(await page.locator('#research-picks-rows tr').count(),3);
      assert.deepEqual(await page.locator('.picks-table').first().locator('th').allTextContents(),['Bet / game','Model prediction','Market consensus','Book line / price','Price time (ET)','Book']);
      assert.equal(await page.locator('#research-picks-rows tr').first().locator('td').count(),6);
      if(width<=820){
        const row=page.locator('#research-picks-rows .pick-offer-row').first();
        assert(await row.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'mobile offer must fit without sideways scrolling');
        const offer=await row.locator('.pick-offer').boundingBox(),model=await row.locator('.pick-model').boundingBox();
        assert(offer.y<model.y,'show the offered line and price before forecast details on mobile');
        assert((await row.locator('.track-pick').boundingBox()).height>=44,'tracker remains easy to tap');
      }
      assert.match(await page.locator('#research-picks-rows').textContent(),/53.0%/);
      assert.equal(await page.locator('.book-offer-line').first().textContent(),'Over 2.5');
      assert.match(await page.locator('#research-picks-rows').textContent(),/Projected Points: 3.4/i);
      assert(!(await page.locator('#daily-picks').textContent()).includes('Astra'));
      assert.deepEqual((await page.locator('main > h2, #daily-picks-heading').allTextContents()).slice(0,2),["Today's picks",'The price rundown']);
      assert.equal(await page.getByText('What changed and what’s next',{exact:true}).count(),0);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      assert.match(await page.locator('#research-picks-rows').textContent(),/7:59:00 AM ET/);
      assert.equal(await page.locator('#top-picks-analysis, #picks-analysis-text').count(),0);
      await page.locator('[data-track-pick="1"]').click();
      assert(await page.locator('#pick-tracker').isVisible());
      assert.equal(await page.locator('#track-odds').inputValue(),'110');
      assert.match(await page.locator('#track-quote').textContent(),/7:59:00 AM ET/);
      assert.match(await page.locator('#track-review').textContent(),/Qualitative review needed/);
      assert.match(await page.locator('#track-grading').textContent(),/automatic result grading is not connected/);
      assert.equal(await page.evaluate(()=>window.trackerTest.writes.length),0);
      await page.locator('#pick-tracker').screenshot({path:`/tmp/fv-briefing-tracker-${width}.png`});
      assert.equal(await page.locator('#pick-tracker').evaluate(e=>e.scrollWidth>e.clientWidth+1),false);
      await page.locator('#track-cancel').click();
      if(await page.locator('.fv-burger').isVisible())await page.locator('.fv-burger').click();
      await page.locator('.nhl-sport button').click();
      assert(await page.locator('.nhl-sport').getByRole('link',{name:'Top Picks',exact:true}).isVisible());
      assert.equal(await page.locator('.nfl-sport').getByRole('link',{name:'Insights archive',exact:true}).count(),0);
      await page.locator('.nhl-sport button').click();
      if(await page.locator('.fv-burger').isVisible())await page.locator('.fv-burger').click();
      await page.locator('#daily-picks').scrollIntoViewIfNeeded();
      await page.screenshot({path:`/tmp/fv-briefing-picks-${width}.png`,fullPage:true});
    }
    // Expanded research, changed-price rechecks and source failures on all sizes.
    const selected=collect(feeds,now).selected.find(r=>r.sport==='MLB');
    feeds.Reviews={schema_version:1,sports:{MLB:{decision_date:'2026-09-27',review_status:'completed',sources:[{
      source_id:'s1',url:'https://www.espn.com/mlb/story/synthetic',title:'Synthetic lineup report',published_at:new Date(now-3600e3).toISOString()}],candidates:[{
      ...selected,review_key:reviewKey(selected),review_bet_key:reviewBetKey(selected),offer_id:'o',forecast_id:'f',qualitative_review:{
        offer_id:'o',forecast_id:'f',status:'concern',reviewed_at:new Date(now).toISOString(),countercase:'A lineup change could reduce projected opportunity.',
        open_checks:['Verify the announced batting order before deciding.'],evidence:[{source_id:'s1',direction:'concern',interpretation:'Check whether the expected role still applies.',represented_in:'model_features'}]}}]}}};
    const reviewedRow=feeds.Reviews.sports.MLB.candidates[0];
    reviewedRow.candidate_id='injury-candidate';reviewedRow.qualitative_review.candidate_id='injury-candidate';
    feeds.Reviews.sports.MLB.sources.push({source_id:'injury1',source_kind:'live_injury_table',candidate_ids:['injury-candidate'],
      url:'https://www.cbssports.com/mlb/injuries/',title:'Synthetic MLB injury listing',published_at:null,retrieved_at:new Date(now).toISOString(),
      missing_teams:['Missing team'],injury_rows:[{player:'<img src=x onerror=alert(1)>',team:'Boston Red Sox',position:'SP',injury:'Shoulder',status:'15-day injured list',reported_update:'Sat, Sep 26'}]});
    for(const width of [320,390,768,1440]) {
      await page.setViewportSize({width,height:1000});await page.reload();await page.waitForSelector('.pick-research',{state:'attached'});
      assert.equal(await page.locator('#top-picks-analysis, #picks-analysis-text').count(),0);
      await openReview();
      assert.match(await page.locator('.pick-research').textContent(),/A lineup change could reduce projected opportunity/);
      await openReview();
      assert(await page.locator('.pick-research').evaluate(el=>el.open));
      assert.match(await page.locator('.pick-research').textContent(),/Case against:/);
      await page.locator('.pick-injuries summary').click();
      assert.match(await page.locator('.pick-injuries').textContent(),/15-day injured list/);
      assert.match(await page.locator('.pick-injuries').textContent(),/Publication time unknown/);
      assert.match(await page.locator('.pick-injuries').textContent(),/Team coverage unavailable: Missing team/);
      assert.equal(await page.locator('.pick-injuries img').count(),0);
      assert.match(await page.locator('.pick-research > summary').textContent(),/Fourth & Value analysis/);
      assert.equal(await page.locator('.pick-research-row > td').getAttribute('colspan'),'6');
      assert.match(await page.locator('#research-picks-rows').textContent(),/Sourced concern/);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      await page.locator('#daily-picks').screenshot({path:`/tmp/fv-astra-review-${width}.png`});
    }
    feeds.Reviews.sports.MLB.sources=feeds.Reviews.sports.MLB.sources.filter(s=>s.source_id!=='injury1');
    reviewedRow.injury_context={status:'unavailable'};
    await page.reload();await page.waitForSelector('.pick-research',{state:'attached'});
    assert.match(await page.locator('.pick-research').textContent(),/Injury table: unavailable for this review/);
    // Diagnostics remain separate from the original prediction and escape source text.
    feeds.Reviews.sports.MLB.candidates[0].model_diagnostics={projection:{attempts:26,completion_rate:.6,yards_per_completion:9.6,
      current_sample:[{season:2026,week:1,attempts:5,completions:3,passing_yards:18}],recent_mean_weight:.2,yards_per_completion_recent_weight:.2},
      calibration:{raw_probability:.914365,outside_fitted_range:true},other_books_at_exact_line:0,
      offered_book_nearby_quotes:[{name:'<img src=x onerror=alert(1)>',point:242.5,price:-240}],
      raw_distribution_stress:{market_centered_mean:211.5,probability:.6678}};
    for(const width of [320,390,768,1440]) {
      await page.setViewportSize({width,height:1000});await page.reload();await page.waitForSelector('.pick-diagnostics',{state:'attached'});
      await openReview();await page.locator('.pick-diagnostics summary').click();
      assert.match(await page.locator('.pick-diagnostics').textContent(),/5 attempts, 3 completions, 18 yards/);
      assert.match(await page.locator('.pick-diagnostics').textContent(),/91.4% before historical calibration/);
      assert.match(await page.locator('.pick-diagnostics').textContent(),/not a new forecast/);
      assert.equal(await page.locator('.pick-diagnostics img').count(),0);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      await page.locator('.pick-research').screenshot({path:`/tmp/fv-model-diagnostic-${width}.png`});
    }
    delete feeds.Reviews.sports.MLB.candidates[0].model_diagnostics;
    // A full model-and-price assessment renders with no news and no layout change.
    const q=feeds.Reviews.sports.MLB.candidates[0].qualitative_review;
    q.status='needs_information';q.evidence=[];
    for(const verdict of ['consider','wait','pass']) {
      q.assessment={verdict,reason:'The model case needs a defensible opportunity estimate.',model_case:'Projected role drives the estimate.',
        price_case:'The reviewed quote clears the numerical screen.',context_case:'No relevant reporting verified.',
        blocking_checks:verdict==='wait'?['Verify expected playing time.']:[]};
      feeds.Card={schema_version:1,kind:'morning',decision_date:'2026-09-27',published_at:new Date(now).toISOString(),rows:shortlist(collect(feeds,now).selected,now)};
      for(const width of [320,390,768,1440]) {
        await page.setViewportSize({width,height:1000});await page.reload();await page.waitForSelector('.pick-research',{state:'attached'});
        assert.equal(await page.locator('#top-picks-analysis, #picks-analysis-text').count(),0);
        if(verdict==='consider'){
          assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),1);
          await page.locator('#daily-picks-rows [data-track-pick]').click();
          assert.match(await page.locator('#track-bet-description').textContent(),/MLB/);
          await page.locator('#track-cancel').click();
          await page.locator('#daily-picks-rows .pick-research > summary').click();
        }else{
          assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),0);
          await openReview();
        }
        assert.match(await page.locator('.pick-research').textContent(),/Model case:/);
        if(verdict==='wait'){
          const row=page.locator('#daily-picks tr').filter({has:page.locator('a[href^="/mlb/picks.html"]')});
          assert.match(await row.textContent(),/Experimental · Needs review/);
          assert.match(await row.textContent(),/Why: The model case needs a defensible opportunity estimate/);
        }
        assert.match(await page.locator('.pick-research').textContent(),/No relevant reporting was verified/);
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
        await page.locator('#daily-picks').screenshot({path:`/tmp/fv-assessment-${verdict}-${width}.png`});
      }
    }
    await page.clock.fastForward(31000);
    assert(await page.locator('.pick-research').evaluate(el=>el.open),'reading a review must survive the expiry timer');
    feeds.MLB.rows[0].price=120;await page.reload();await page.waitForSelector('.pick-research',{state:'attached'});
    assert.match(await page.locator('#research-picks-rows').textContent(),/Price or forecast changed/);
    assert.equal(await page.locator('#top-picks-analysis, #picks-analysis-text').count(),0);
    feeds.MLB.rows[0].price=110;feeds.Reviews=null;await page.reload();
    await page.waitForFunction(()=>document.getElementById('research-pool-label').textContent.includes('3 additional'));
    assert.match(await page.locator('#picks-status').textContent(),/0 reviewed picks/);
    await page.locator('#picks-research-pool').evaluate(e=>e.open=true);
    // Real shared helper against a local fake client: no auth emails or real bets.
    await page.locator('[data-track-pick="1"]').click();
    await page.locator('#track-odds').fill('-120');await page.locator('#track-stake').fill('25');await page.locator('#track-confirm').check();
    await page.evaluate(()=>window.trackerTest.user=null);await page.locator('#track-save').click();
    await page.waitForSelector('#track-signin:not([hidden])');assert.equal(await page.evaluate(()=>window.trackerTest.writes.length),0);
    await page.evaluate(()=>{window.trackerTest.user={id:'test-user'};window.trackerTest.error={code:'XX000'};});
    await page.locator('#track-save').click();await page.waitForFunction(()=>document.getElementById('track-feedback').textContent.includes('could not be saved'));
    assert.equal(await page.locator('[data-track-pick="1"]').textContent(),'Track bet');
    await page.evaluate(()=>window.trackerTest.error=null);await page.locator('#track-save').dblclick();
    await page.waitForFunction(()=>document.getElementById('track-feedback').textContent.startsWith('Saved to your'));
    const saved=await page.evaluate(()=>window.trackerTest.writes);
    assert.equal(saved.length,2,'one failed write and one confirmed write; double click cannot duplicate');
    assert.equal(saved[0].id,saved[1].id);assert.equal(saved[1].odds,-120);assert.equal(saved[1].stake_dollars,25);
    assert.equal(saved[1].league,'MLB');assert.equal(saved[1].user_id,'test-user');
    assert.equal(await page.locator('[data-track-pick="1"]').textContent(),'Tracked');
    assert.match(await page.locator('#research-picks-rows').textContent(),/Qualitative review needed/);
    await page.locator('#track-cancel').click();
    // Background refresh/expiry must not replace a draft with a different offer.
    await page.locator('[data-track-pick="2"]').click();await page.locator('#track-stake').fill('12');
    // An open page withdraws expired quotes without a manual reload.
    await page.clock.fastForward(31*60e3);await page.waitForFunction(()=>document.getElementById('research-pool-label').textContent.includes('2 additional'));
    assert.equal(await page.locator('#track-stake').inputValue(),'12');assert.match(await page.locator('#track-quote').textContent(),/expired or changed/);
    await page.locator('#track-cancel').click();
    feeds.MLB=null;await page.reload();await page.waitForFunction(()=>document.getElementById('picks-coverage').textContent.includes('MLB: Current model list unavailable'));
    assert.equal(await page.locator('#research-picks-rows tr').count(),1);
    feeds={};await page.reload();await page.waitForFunction(()=>document.getElementById('research-pool-label').textContent.includes('0 additional'));
    assert.match(await page.locator('#daily-picks-rows').textContent(),/No published morning picks/);
    assert.equal(await page.locator('#top-picks-analysis, #picks-analysis-text').count(),0);
    await page.screenshot({path:'/tmp/fv-briefing-empty.png',fullPage:true});
    feeds=fixture(now);
    const original=collect(feeds,now).selected.find(r=>r.sport==='MLB');
    const reviewed={...original,review_key:reviewKey(original),review_bet_key:reviewBetKey(original),offer_id:'o',forecast_id:'f',qualitative_review:{
      offer_id:'o',forecast_id:'f',status:'needs_information',reviewed_at:new Date(now).toISOString(),countercase:'Original risk.',open_checks:[],evidence:[],
      assessment:{verdict:'consider',reason:'Original assessment.',model_case:'Original projection.',price_case:'Original quote.',context_case:'No verified news.',blocking_checks:[]}}};
    feeds.Reviews={schema_version:1,sports:{MLB:{decision_date:'2026-09-27',candidates:[reviewed],sources:[]}}};
    const card={schema_version:1,kind:'test',decision_date:'2026-09-27',published_at:new Date(now).toISOString(),rows:shortlist(collect(feeds,now).selected,now)};
    feeds={Card:card};await page.reload();await page.waitForSelector('#daily-picks-rows [data-track-pick]');
    await page.clock.fastForward(6*3600e3);
    assert.match(await page.locator('#picks-status').textContent(),/Test edition for 2026-09-27/);
    assert.match(await page.locator('#daily-picks-rows').textContent(),/Game started · historical assessment/);
    assert.match(await page.locator('#daily-picks-rows').textContent(),/Original assessment/);
    assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),1);
    await page.locator('#daily-picks-rows [data-track-pick]').click();
    assert.match(await page.locator('#track-quote').textContent(),/Historical edition quote/);
    await page.locator('#track-cancel').click();
    delete feeds.Card;await page.clock.fastForward(5*60e3);await page.waitForTimeout(100);
    assert.equal(await page.locator('#daily-picks-rows [data-track-pick]').count(),1,'temporary endpoint failure retains dated card');
    await page.clock.fastForward(24*3600e3);
    assert.match(await page.locator('#picks-status').textContent(),/Previous edition/);
    await page.screenshot({path:'/tmp/fv-morning-edition-persisted.png',fullPage:true});
    await page.goto(base+'/props/insights.html');await page.waitForURL(base+'/nfl/');
    await page.goto(base+'/briefing/2026-09-26.html');assert.equal(await page.locator('#daily-picks').count(),0,'dated archive must never show current picks');
    await page.goto(base+'/tracking/');await page.waitForSelector('#betTracker:visible');
    await page.waitForFunction(()=>document.getElementById('betsTableBody').textContent.includes('<img'));
    assert.equal(await page.locator('#betsTableBody img').count(),0,'saved source text is rendered safely');
    assert.equal(await page.locator('#profitLoss').textContent(),'+$0.00','pending bets must not appear as losses');
    assert.deepEqual(errors,[]);console.log('PASS: briefing tables, navigation, desktop/mobile, quote expiry, failures and archive redirect.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});

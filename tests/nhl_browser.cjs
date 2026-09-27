// NODE_PATH=/path/to/node_modules node tests/nhl_browser.cjs
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,...(process.env.NHL_BUNDLED_BROWSER?{}:{executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'})});
  try{
    const errors=[];
    for(const width of [390,768,1280,1440,1920]){
      const p=await browser.newPage({viewport:{width,height:1000}});p.on('pageerror',e=>errors.push(e.message));
      for(const route of ['/nhl/','/nhl/props/','/nhl/totals/','/nhl/picks.html','/nhl/top.html','/nhl/methods.html','/nba/','/']){
        await p.goto(base+route);if(route.startsWith('/nhl'))await p.waitForFunction(()=>!document.getElementById('feed-status').textContent.includes('Enable JavaScript'));
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`Overflow ${width} ${route}`);
        assert.equal(await p.locator('.nhl-sport').count(),1);
        if(!await p.locator('.fv-burger').isVisible()){
          const logo=await p.locator('.fv-logo').boundingBox(),links=await p.locator('.fv-links').boundingBox();
          assert(logo.x+logo.width<=links.x,`Nav logo overlaps links at ${width}`);
        }
      }
      if(await p.locator('.fv-burger').isVisible()){
        assert((await p.locator('.fv-burger span').first().boundingBox()).height>=2,'Mobile menu icon must remain visible');
        await p.locator('.fv-burger').click();
      }
      await p.locator('.nhl-sport button').click();await p.getByRole('link',{name:'NHL Overview',exact:true}).click();assert(p.url().endsWith('/nhl/'));
      if(width===390||width===1440)await p.screenshot({path:`/tmp/fv-nhl-${width}.png`,fullPage:true});
      await p.close();
    }
    // Exercise a populated props board even when the real slate has no props.
    const p=await browser.newPage({viewport:{width:390,height:900}});
    const now=new Date(),future=new Date(+now+3600e3).toISOString();
    const row={event_id:'g',game:'Away @ Home',commence_time:future,quoted_at:now.toISOString(),player:'Example Player',market:'player_shots_on_goal',market_label:'Shots on goal',line:20.5,side:'Over',price:-110,book:'a',book_label:'Book A',book_probability:.5238,fair_probability:.5,consensus_probability:.5,paired_books:2,baseline_mean:null};
    let fixture={status:'ready',last_success_at:now.toISOString(),events:[],rows:[row,{...row,book:'b',book_label:'Book B',price:100,book_probability:.5}]};
    await p.route('**/nhl/data/latest.json',r=>r.fulfill({json:fixture}));
    await p.goto(base+'/nhl/props/');await p.waitForSelector('.prop-card');assert.equal(await p.locator('.prop-card').count(),1);
    assert((await p.locator('.prop-card').textContent()).includes('Book B'));
    await p.locator('#book').selectOption('a');assert((await p.locator('.prop-card').textContent()).includes('Book A'));
    await p.locator('#search').fill('missing');assert.equal(await p.locator('.prop-card').count(),0);
    await p.locator('#reset').click();assert.equal(await p.locator('.prop-card').count(),1);
    // Additive model fields, with explicit units, uncertainty and missing/stale safeguards.
    fixture.model_status='Experimental independent forecasts; recommendations disabled';
    fixture.rows=fixture.rows.map(r=>({...r,model_data_checked_at:now.toISOString(),independent_probability:.56,final_probability:.56,
      market_probability:.5,push_probability:0,fair_odds:-127.27,estimated_ev:.08,minimum_acceptable_odds:110,
      model_status:'Experimental independent forecast; no validated betting edge',model_version:'nhl-v2.1',validation_status:'experimental',
      analyst_status:'unreviewed',key_drivers:['Projected ice time 18.0 minutes'],uncertainties:['Goalie unconfirmed'],
      sensitivity:{win_min:.50,win_max:.61,assumption:'Rate ±10%; not a confidence interval'},
      invalidation_conditions:['Price changes'],signal_type:'combined_signal_unvalidated'}));
    await p.reload();await p.waitForSelector('.nhl-forecast');await p.locator('.nhl-forecast summary').click();
    assert((await p.locator('.nhl-forecast').textContent()).includes('Minimum acceptable price'));
    assert((await p.locator('.nhl-forecast').textContent()).includes('56.0%'));
    assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'Forecast panel mobile overflow');
    await p.evaluate(()=>window.scrollTo(0,0));await p.screenshot({path:'/tmp/fv-nhl-model-mobile.png',fullPage:true});
    await p.setViewportSize({width:1440,height:1000});await p.screenshot({path:'/tmp/fv-nhl-model-desktop.png',fullPage:true});
    fixture.rows=fixture.rows.map(r=>({...r,model_data_checked_at:new Date(+now-37*3600e3).toISOString()}));
    await p.reload();await p.waitForSelector('.prop-card');assert.equal(await p.locator('.nhl-forecast').count(),0);
    assert((await p.locator('.prop-card').textContent()).includes('forecast hidden'));
    fixture.rows=fixture.rows.map(r=>({...r,independent_probability:null,final_probability:null,model_status:'Independent model unavailable'}));
    await p.reload();await p.waitForSelector('.prop-card');assert.equal(await p.locator('.nhl-forecast').count(),0);
    assert((await p.locator('.prop-card').textContent()).includes('Independent model unavailable'));
    fixture={...fixture,status:'feed_error'};await p.reload();await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('failed'));assert.equal(await p.locator('.prop-card').count(),0);
    fixture={...fixture,status:'ready',last_success_at:new Date(+now-25*3600e3).toISOString()};await p.reload();await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('needs a refresh'));assert.equal(await p.locator('.prop-card').count(),0);
    fixture={...fixture,status:'waiting_for_markets',last_success_at:now.toISOString(),rows:[]};await p.reload();
    await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('Waiting for NHL markets'));assert.equal(await p.locator('.prop-card').count(),0);
    // Pure Market Watch: integer lines need no independent model, and model rank cannot change ordering.
    fixture={status:'ready',last_success_at:now.toISOString(),events:[],rows:[
      {...row,event_id:'a',player:'Price leader',line:6,other_books:3,conditional_price_advantage:12,consensus_ev:null,rank_score:-99},
      {...row,event_id:'b',player:'Model leader',line:6,other_books:3,conditional_price_advantage:5,consensus_ev:50,rank_score:99}]};
    await p.goto(base+'/nhl/top.html');await p.waitForSelector('.prop-card');assert.equal(await p.locator('.prop-card').count(),2);
    assert((await p.locator('.prop-card').first().textContent()).includes('Price leader'));
    // Synthetic analyst candidate. No synthetic records are published to the real feed.
    const candidate={...row,candidate_id:'candidate1',candidate_rank:1,offer_id:'offer1',forecast_id:'forecast1',decision_at:now.toISOString(),model_data_checked_at:now.toISOString(),
      independent_probability:.6,final_probability:.6,market_probability:.53,push_probability:0,fair_odds:-150,estimated_ev:.145,minimum_acceptable_odds:-115,
      model_version:'nhl-v2.1',validation_status:'experimental',other_books:3,human_decision:'unreviewed',
      key_drivers:['Projected ice time 18 minutes'],uncertainties:['Power-play role unconfirmed'],invalidation_conditions:['Price or role changes'],
      sensitivity:{win_min:.55,win_max:.65,assumption:'Rate ±10%; not a confidence interval'},
      qualitative_review:{assessment:{verdict:'wait',reason:'Resolve the role assumption.',model_case:'Opportunity drives the estimate.',price_case:'Quote passes sensitivity screen.',context_case:'Role needs checking.',blocking_checks:['Verify role.']},offer_id:'offer1',forecast_id:'forecast1',status:'needs_information',reviewed_at:now.toISOString(),countercase:'Role assumptions may change.',open_checks:['Verify participation.'],
        evidence:[{source_id:'s1',excerpt:'The goalie will be announced later.',interpretation:'Wait for confirmation; do not adjust the probability.',kind:'goalie',direction:'context',represented_in:'unknown'}]}};
    fixture={status:'ready',last_success_at:now.toISOString(),snapshot_id:'snapshot1',events:[],rows:[candidate]};
    const decisionDate=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(now);
    let board={schema_version:1,board_id:'board1',decision_date:decisionDate,generated_at:now.toISOString(),session:'morning',source_snapshot_id:'snapshot1',status:'ready',review_status:'completed',eligible_count:1,candidates:[candidate],
      sources:[{source_id:'s1',title:'Synthetic source fixture',url:'https://www.nhl.com/news/',published_at:now.toISOString(),retrieved_at:now.toISOString()}]};
    board.sources.push({source_id:'injury1',source_kind:'live_injury_table',candidate_ids:['candidate1'],
      url:'https://www.cbssports.com/nhl/injuries/',title:'Synthetic NHL injury listing',published_at:null,retrieved_at:now.toISOString(),
      injury_rows:[{player:'Example Goalie',team:'Washington Capitals',position:'G',injury:'Lower body',status:'Day-to-day',reported_update:'Sat, Sep 26'}]});
    await p.route('**/nhl/data/candidates.json',r=>r.fulfill({json:board}));
    p.on('pageerror',e=>errors.push(e.message));
    for(const width of [390,1440]){
      await p.setViewportSize({width,height:1000});await p.goto(base+'/nhl/picks.html');await p.waitForSelector('[data-candidate]');
      assert((await p.locator('[data-candidate]').textContent()).includes('60.0%'));
      assert((await p.locator('[data-candidate]').textContent()).includes('Our assessment'));
      assert((await p.locator('[data-candidate]').textContent()).includes('Model case:'));
      await p.locator('.pick-injuries summary').click();
      assert.match(await p.locator('.pick-injuries').textContent(),/Example Goalie/);
      assert.match(await p.locator('.pick-injuries').textContent(),/Publication time unknown/);
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'Candidate mobile/desktop overflow');
      await p.screenshot({path:`/tmp/fv-nhl-candidates-${width}.png`,fullPage:true});
    }
    await p.locator('#candidate-search').fill('absent');assert.equal(await p.locator('[data-candidate]').count(),0);await p.locator('#candidate-search').fill('');
    await p.getByText('Prepare an analyst review',{exact:true}).click();
    await p.locator('[name="analyst"]').fill('Test Analyst');await p.locator('[name="reason"]').fill('Synthetic test; wait for goalie confirmation.');
    await p.locator('[name="double_counting_check"]').fill('Unknown; no forecast adjustment.');
    const download=p.waitForEvent('download');await p.getByRole('button',{name:'Download review note'}).click();
    const file=await (await download).path();const note=JSON.parse(fs.readFileSync(file,'utf8'));assert.equal(note.offer_id,'offer1');assert.equal(note.decision,'watch');
    assert((await p.locator('.review-feedback').textContent()).includes('still needs to be imported'));
    board.candidates=[{...candidate,quoted_at:new Date(+now-31*60e3).toISOString()}];await p.reload();await p.waitForSelector('[data-candidate]');
    assert((await p.locator('[data-candidate] .notice').textContent()).includes('expired'));assert(await p.locator('option[value="select"]').evaluate(e=>e.disabled), await p.locator('option[value="select"]').evaluate(e=>e.outerHTML));
    board.candidates=[{...candidate,qualitative_review:null}];board.review_status='no_usable_reporting';await p.reload();await p.waitForSelector('[data-candidate]');
    assert((await p.locator('#research-status').textContent()).includes('unavailable'));
    fixture.status='feed_error';await p.reload();await p.waitForFunction(()=>document.getElementById('candidate-summary').textContent.includes('research candidates'));assert.equal(await p.locator('[data-candidate]').count(),0);
    fixture.status='ready';board.source_snapshot_id='different';await p.reload();await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('snapshot changed'));assert.equal(await p.locator('[data-candidate]').count(),0);
    board.source_snapshot_id='snapshot1';board.candidates=[];board.status='no_candidates';board.review_status='no_candidates';await p.reload();await p.waitForFunction(()=>document.getElementById('research-status').textContent.includes('no analysis was requested'));assert.equal(await p.locator('[data-candidate]').count(),0);
    assert.deepEqual(errors,[]);console.log('PASS: NHL routes at five widths; forecasts, price-only Market Watch, analyst shortlist, evidence, download, expired/missing/failure and empty states.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});

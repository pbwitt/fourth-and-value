// Local fixtures only: exercise actual page scripts and shared saves without account/database writes.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`,browser=await chromium.launch({headless:true});
  try{
    const p=await browser.newPage({viewport:{width:390,height:900}}),errors=[];
    p.on('pageerror',error=>errors.push(error.message));
    await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
    await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await p.addInitScript(()=>{
      window.db={user:{id:'test-owner'},rows:[],attempts:[],error:null,delay:0};
      window.supabase={createClient:()=>({auth:{getSession:async()=>({data:{session:db.user?{user:db.user}:null}})},from:()=>({
        insert:async row=>{db.attempts.push(row);if(db.delay)await new Promise(r=>setTimeout(r,db.delay));if(db.error)return {error:{code:'TEST'}};db.rows.push(row);return {error:null};}
      })})};
    });
    const now=new Date(),future=new Date(+now+3600e3).toISOString();
    const row={event_id:'g',game:'Montreal Canadiens @ Toronto Maple Leafs',commence_time:future,quoted_at:now.toISOString(),
      player:'Auston Matthews',market:'player_shots_on_goal',market_label:'Shots on goal',line:3.5,side:'Over',price:-110,
      book:'caesars',book_label:'Caesars',book_probability:.5238,fair_probability:.5,consensus_probability:.5,paired_books:4,
      baseline_mean:null,other_books:3,conditional_price_advantage:3,model_data_checked_at:now.toISOString(),
      independent_probability:.6,final_probability:.6,push_probability:0,market_probability:.5,estimated_ev:.145,
      model_status:'Experimental',model_version:'nhl-v2',validation_status:'experimental'};
    let fixture={status:'ready',last_success_at:now.toISOString(),snapshot_id:'snapshot',events:[],rows:[row,{...row,book:'fanduel',book_label:'FanDuel',price:100,book_probability:.5}]};
    const candidate={...row,candidate_id:'c1',candidate_rank:1,offer_id:'o1',forecast_id:'f1',decision_at:now.toISOString(),human_decision:'watch',qualitative_review:null};
    let board={schema_version:1,board_id:'board',decision_date:new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(now),
      generated_at:now.toISOString(),session:'morning',source_snapshot_id:'snapshot',status:'ready',review_status:'not_requested',eligible_count:1,candidates:[candidate],sources:[]};
    await p.route('**/nhl/data/latest.json',route=>route.fulfill({json:fixture}));
    await p.route('**/nhl/data/candidates.json',route=>route.fulfill({json:board}));
    const open=async()=>{await p.locator('[data-fv-ticket]').first().click();await p.waitForSelector('#fv-bet-tracker[open]');};
    const fill=async()=>{await p.locator('#fv-track-odds').fill('-120');await p.locator('#fv-track-stake').fill('25');await p.locator('#fv-track-confirm').check();};
    const save=async()=>p.locator('#fv-track-save').click();
    const saved=async()=>p.waitForFunction(()=>document.getElementById('fv-track-feedback').textContent.startsWith('Saved to'));
    await p.goto(base+'/nhl/props/');await p.waitForSelector('[data-fv-ticket]');
    assert.equal(await p.locator('[data-fv-ticket]').count(),1,'best-price filter still applies');
    await p.locator('#book').selectOption('caesars');await open();
    assert.match(await p.locator('#fv-track-description').textContent(),/Caesars/);
    assert.equal(await p.locator('#fv-track-odds').inputValue(),'-110');await fill();
    await p.evaluate(()=>{db.user=null;});await save();await p.waitForSelector('#fv-track-signin:not([hidden])');
    assert.equal(await p.evaluate(()=>db.rows.length),0);assert.equal(await p.locator('#fv-track-stake').inputValue(),'25');
    await p.evaluate(()=>{db.user={id:'test-owner'};db.error=true;});await save();
    await p.waitForFunction(()=>document.getElementById('fv-track-feedback').textContent.includes('could not be saved'));
    const firstID=await p.evaluate(()=>db.attempts[0].id);
    await p.evaluate(()=>{db.error=null;db.delay=250;});await save();await saved();
    const ticket=await p.evaluate(()=>db.rows[0]);assert.equal(ticket.id,firstID);assert.equal(ticket.user_id,'test-owner');
    assert.equal(ticket.market_type,'sog');assert.equal(ticket.line,3.5);assert.equal(ticket.book,'caesars');assert.equal(ticket.odds,-120);assert.equal(ticket.stake_dollars,25);
    assert(await p.locator('#fv-track-save').isDisabled());
    for(const width of [390,1440]){
      await p.setViewportSize({width,height:900});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      assert.equal(await p.locator('#fv-bet-tracker').evaluate(el=>el.scrollWidth>el.clientWidth+1),false);
      await p.screenshot({path:`/tmp/fv-offer-tracker-nhl-${width}.png`,fullPage:true});
    }
    await p.locator('#fv-track-close').click();assert(await p.locator('[data-fv-ticket]').isDisabled());
    await p.locator('#search').fill('absent');assert.equal(await p.locator('[data-fv-ticket]').count(),0);
    await p.locator('#search').fill('');assert(await p.locator('[data-fv-ticket]').isDisabled(),'saved state survives filtering');
    // All seven supported market payloads reach the actual save helper from their rendered cards.
    for(const [market,type,side,line] of [['player_goals','goals','Over',.5],['player_assists','assists','Under',1.5],['player_points','points','Over',1.5],
      ['totals','totals','Under',6],['spreads','spreads','Toronto Maple Leafs',-1.5],['h2h','h2h','Montreal Canadiens',null]]){
      fixture.rows=[{...row,market,market_label:market,side,line,player:market.startsWith('player_')?row.player:''}];
      await p.goto(base+(market.startsWith('player_')?'/nhl/props/':'/nhl/totals/'));await p.waitForSelector('[data-fv-ticket]');
      await open();await fill();await save();await saved();const result=await p.evaluate(()=>db.rows[0]);
      assert.equal(result.market_type,type);assert.equal(result.line,line);assert.equal(result.side,['Over','Under'].includes(side)?side.toLowerCase():side);
    }
    fixture.rows=[row];await p.goto(base+'/nhl/top.html');await p.waitForSelector('[data-fv-ticket]');await open();await fill();await save();await saved();
    await p.goto(base+'/nhl/picks.html');await p.waitForSelector('[data-fv-ticket]');await open();await fill();await save();await saved();
    await p.locator('#fv-track-close').click();assert.match(await p.locator('[data-candidate]').textContent(),/Recorded status: watch/);
    assert.equal(await p.locator('form[data-review]').count(),1,'analyst preparation remains separate');
    // Expired research quotes can be logged as actual wagers without enabling shadow selection.
    board.candidates=[{...candidate,quoted_at:new Date(+now-31*60e3).toISOString(),model_data_checked_at:new Date(+now-37*3600e3).toISOString()}];
    await p.reload();await p.waitForSelector('[data-fv-ticket]');assert(await p.locator('option[value=select]').evaluate(el=>el.disabled));
    await open();await fill();await save();await saved();assert.equal(await p.evaluate(()=>db.rows[0].model_prob),null);
    // Page reload must not destroy an open tracking form.
    await p.clock.install();await p.goto(base+'/nhl/props/');await p.waitForSelector('[data-fv-ticket]');await open();
    await p.clock.fastForward(301000);assert(await p.locator('#fv-bet-tracker').evaluate(el=>el.open));
    await p.clock.resume();
    fixture.status='feed_error';await p.goto(base+'/nhl/totals/');await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('failed'));
    assert.equal(await p.locator('[data-fv-ticket]').count(),0);
    fixture.status='ready';board.source_snapshot_id='changed';await p.goto(base+'/nhl/picks.html');await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('snapshot changed'));
    assert.equal(await p.locator('[data-fv-ticket]').count(),0);
    assert.deepEqual(errors,[]);console.log('PASS: NHL props, lines, Market Watch and candidates; auth, retries, exact saves, filtering, expired/failed states and responsive dialog.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

// Live Odds page in a real browser with a stubbed Supabase client: editor gate,
// game list, Run now, scoreline, flags, Bet Tracker tickets, filters, safe rendering, errors, mobile layout.
const {chromium}=require(process.env.FV_PLAYWRIGHT||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.join(__dirname,'../docs');

const HOME='Buffalo Sabres',AWAY='Chicago Blackhawks',ID='a'.repeat(32),LATER='c'.repeat(32);
const stub=`
const iso=s=>new Date(Date.now()-s*1000).toISOString();
const book=(key,title,age,markets)=>({key,title,last_update:iso(age),markets:Object.entries(markets).map(([k,outs])=>({key:k,last_update:iso(age),
  outcomes:outs.map(([name,price,point,description])=>({name,price,...(point==null?{}:{point}),...(description?{description}:{})}))}))});
const sog=(o,u,player)=>({player_shots_on_goal:[['Over',o,2.5,player],['Under',u,2.5,player]]});
const scorer=price=>({player_goal_scorer_anytime:[['Yes',price,null,'Tage Thompson']]});
const even=(k,t)=>book(k,t,20,{...sog(-110,-110,'Tage Thompson'),totals:[['Over',-110,6.5],['Under',-110,6.5]],h2h:[['${HOME}',-150],['${AWAY}',130]],
  ...{player_points:[['Over',-120,0.5,'<img src=x onerror="window.hacked=1">'],['Under',-110,0.5,'<img src=x onerror="window.hacked=1">']]},...scorer(150)});
window.mode='editor';window.calls=[];window.fail=null;window.detail='2nd Int';
window.supabaseClient={
  auth:{getUser:async()=>({data:{user:window.mode==='signedout'?null:{email:'e@x.com',app_metadata:window.mode==='editor'?{fv_editor:true}:{}}}}),
    onAuthStateChange:fn=>{window.authChanged=fn;}},
  functions:{invoke:async(name,{body})=>{
    window.calls.push([name,body]);
    if(name==='live-stats')return {data:{games:[{id:2026020022,startTimeUTC:'2026-10-03T23:00:00Z',gameState:'LIVE',gameScheduleState:'OK',
      periodDescriptor:{number:2,periodType:'REG'},clock:{timeRemaining:'00:00',secondsRemaining:0,inIntermission:window.detail==='2nd Int'},
      awayTeam:{abbrev:'CHI',name:{default:'Blackhawks'},score:1},homeTeam:{abbrev:'BUF',name:{default:'Sabres'},score:2}}]},error:null};
    if(window.fail)return {data:null,error:{message:'Edge Function returned a non-2xx status code',context:{json:async()=>({error:window.fail})}}};
    if(!body.event)return {data:{games:[{id:'${LATER}',commence_time:new Date(Date.now()+3*3600e3).toISOString(),home_team:'Boston Bruins',away_team:'New York Rangers'},
      {id:'${ID}',commence_time:new Date(Date.now()-40*60e3).toISOString(),home_team:'${HOME}',away_team:'${AWAY}'}],remaining:17185},error:null};
    return {data:{event:{id:body.event,commence_time:new Date(Date.now()-40*60e3).toISOString(),home_team:'${HOME}',away_team:'${AWAY}',
      bookmakers:[even('fanduel','FanDuel'),even('betmgm','BetMGM'),even('williamhill_us','Caesars'),
        book('draftkings','DraftKings',15,{...sog(110,-140,'Tage Thompson'),totals:[['Over',-110,6.5],['Under',-110,6.5]],...scorer(200)})]},
      fetched_at:new Date().toISOString(),cost:7,reused:false,remaining:17178},error:null};
  }}};
window.signInWithEmail=async email=>{window.signedInWith=email;return {ok:true};};
window.signInErrorMessage=e=>e?.message||'error';window.signOut=async()=>{window.mode='signedout';};
window.saved=[];window.saveTrackedBet=async t=>{window.saved.push(t);return {ok:true,id:t.id};};`;

(async()=>{
  const browser=await chromium.launch({headless:true,...(process.env.FV_CHROME?{executablePath:process.env.FV_CHROME}:{})});
  const page=await browser.newPage({viewport:{width:390,height:844}});
  await page.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'application/javascript',body:''}));
  await page.route('https://fourthandvalue.com/**',route=>{
    const p=new URL(route.request().url()).pathname.replace(/\/$/,'/index.html');
    if(p==='/tracking/bet-tracking.js')return route.fulfill({contentType:'application/javascript',body:stub});
    try{return route.fulfill({contentType:p.endsWith('.js')?'application/javascript':p.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(root+p)});}
    catch{return route.fulfill({status:404,body:'Not found'});}
  });
  const errors=[];page.on('pageerror',e=>errors.push(e.message));

  // Editor: games load, the game in progress is preselected.
  await page.goto('https://fourthandvalue.com/live/');
  await page.waitForFunction(()=>document.getElementById('message').textContent.includes('2 NHL games'));
  assert.ok((await page.locator('#message').textContent()).includes('17,185 Odds API credits left'));
  assert.equal(await page.locator('#game').inputValue(),ID,'the game in progress is the default');
  assert.equal(await page.locator('#result').isVisible(),false);

  // Run now: one paid call for that game plus the free scoreboard.
  await page.locator('#run').click();
  await page.waitForFunction(()=>!document.getElementById('result').hidden);
  assert.match(await page.locator('#message').textContent(),/This run cost 7 credits\. 17,178 Odds API credits left\./);
  const calls=await page.evaluate(()=>window.calls);
  assert.deepEqual(calls.filter(c=>c[0]==='live-odds').map(c=>c[1]),[{},{event:ID,live:true}],'a game in progress also asks for milestones');
  assert.deepEqual(calls.find(c=>c[0]==='live-stats')[1],{league:'NHL',date:await page.evaluate(()=>FVLiveOdds.easternDate(new Date(Date.now()-40*60e3)))});
  const scoreline=await page.locator('#scoreline').textContent();
  assert.ok(scoreline.includes('CHI 1 – 2 BUF')&&scoreline.includes('2nd Int'),scoreline);
  assert.equal(await page.locator('#play-notice').isVisible(),false,'intermission: no live-play warning');
  const flags=await page.locator('#flags li').allTextContents();
  assert.equal(flags.length,2);
  assert.ok(flags[0].includes('Tage Thompson · Shots on goal · Over 2.5')&&flags[0].includes('+110 at DraftKings')&&flags[0].includes('+5.0%'),flags[0]);
  // A one-way milestone: listed as a price gap, never as an edge.
  assert.ok(flags[1].includes('Tage Thompson · Anytime goal scorer · To score')&&flags[1].includes('+200 at DraftKings')
    &&flags[1].includes('+20.0%')&&flags[1].includes('where to shop, not whether the bet is good'),flags[1]);
  assert.ok((await page.locator('#lines tr',{hasText:'To score'}).textContent()).includes('vs. prices, margins in'));
  assert.equal(await page.locator('#lines tr.flagged').count(),1);
  assert.equal(await page.locator('#lines img').count(),0,'player names are escaped');
  assert.equal(await page.evaluate(()=>window.hacked),undefined);
  assert.ok((await page.locator('#asof').textContent()).includes('4 books'));

  // Track a live bet from the flag: the shared dialog saves the price received and the stake.
  await page.locator('#flags button.track').first().click();
  const dialog=page.locator('#fv-bet-tracker');await dialog.waitFor({state:'visible'});
  const description=await page.locator('#fv-track-description').textContent();
  assert.ok(description.includes('Tage Thompson')&&description.includes('DraftKings')&&description.includes('Over'),description);
  assert.equal(await page.locator('#fv-track-odds').inputValue(),'110','prefilled with the quote, editable to the price received');
  await page.locator('#fv-track-odds').fill('105');await page.locator('#fv-track-stake').fill('10');
  await page.locator('#fv-track-confirm').check();await page.locator('#fv-track-save').click();
  await page.waitForFunction(()=>document.getElementById('fv-track-feedback').textContent.includes('Saved to your Bet Tracker'));
  const [saved]=await page.evaluate(()=>window.saved);
  assert.deepEqual({...saved,id:undefined,game_date:undefined},{league:'NHL',game_date:undefined,team_home:HOME,team_away:AWAY,
    player:'Tage Thompson',market_type:'sog',side:'over',line:2.5,book:'draftkings',odds:105,stake_dollars:10,model_prob:null,edge_bps:null,id:undefined});
  assert.match(saved.id,/^[0-9a-f-]{36}$/);
  assert.equal(await page.locator('#fv-track-save').isDisabled(),true,'one quote cannot be saved twice');
  await page.locator('#fv-track-close').click();
  // Any book's quote can be tracked from the expanded list.
  const totalRow=page.locator('#lines tr',{hasText:'Over 6.5'});
  await totalRow.locator('summary').click();
  await totalRow.locator('li',{hasText:'Caesars'}).locator('button.track').click();
  await dialog.waitFor({state:'visible'});
  assert.ok((await page.locator('#fv-track-description').textContent()).includes('Caesars'));
  await page.locator('#fv-track-close').click();
  // A milestone is saved as the grader's base contract: anytime scorer = goals "Yes", no line.
  await page.locator('#flags button.track').nth(1).click();
  await dialog.waitFor({state:'visible'});
  assert.ok((await page.locator('#fv-track-description').textContent()).includes('Anytime goal scorer'));
  await page.locator('#fv-track-stake').fill('5');await page.locator('#fv-track-confirm').check();await page.locator('#fv-track-save').click();
  await page.waitForFunction(()=>window.saved.length===2);
  const scorerTicket=(await page.evaluate(()=>window.saved))[1];
  assert.deepEqual([scorerTicket.market_type,scorerTicket.side,scorerTicket.line,scorerTicket.odds,scorerTicket.book],['goals','Yes',null,200,'draftkings']);
  await page.locator('#fv-track-close').click();

  // Filters.
  const total=await page.locator('#lines tr').count();
  await page.locator('#market').selectOption('totals');
  assert.equal(await page.locator('#lines tr').count(),2);
  await page.locator('#market').selectOption('');
  await page.locator('#player').fill('thomp');
  assert.equal(await page.locator('#lines tr').count(),3);
  await page.locator('#player').fill('nobody');
  assert.ok((await page.locator('#lines').textContent()).includes('No lines match'));
  await page.locator('#player').fill('');assert.equal(await page.locator('#lines tr').count(),total);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true,'no sideways page scroll on a phone');
  if(process.env.FV_SCREENSHOT)await page.screenshot({path:process.env.FV_SCREENSHOT,fullPage:true});

  // Play under way: the warning shows. Server errors are shown as written.
  await page.evaluate(()=>{window.detail='2nd 12:40';});
  await page.locator('#run').click();await page.waitForFunction(()=>!document.getElementById('play-notice').hidden);
  await page.evaluate(()=>{window.fail='Live odds are paused: 1500 credits left, below the 2000 kept for the scheduled refreshes.';});
  await page.locator('#run').click();await page.waitForFunction(()=>document.getElementById('message').textContent.includes('paused'));
  assert.equal(await page.locator('#run').isDisabled(),false,'the button comes back after an error');

  // Changing the game clears the old prices.
  await page.locator('#game').selectOption(LATER);
  assert.equal(await page.locator('#result').isVisible(),false);

  // A signed-in reader sees the gate and spends nothing.
  const reader=await browser.newPage({viewport:{width:1280,height:900}});
  await reader.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'application/javascript',body:''}));
  await reader.route('https://fourthandvalue.com/**',route=>{
    const p=new URL(route.request().url()).pathname.replace(/\/$/,'/index.html');
    if(p==='/tracking/bet-tracking.js')return route.fulfill({contentType:'application/javascript',body:stub.replace("window.mode='editor'","window.mode='reader'")});
    try{return route.fulfill({contentType:p.endsWith('.js')?'application/javascript':p.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(root+p)});}
    catch{return route.fulfill({status:404,body:'Not found'});}
  });
  await reader.goto('https://fourthandvalue.com/live/');
  await reader.waitForFunction(()=>document.getElementById('message').textContent.includes('editor access'));
  assert.equal(await reader.locator('#desk').isVisible(),false);assert.equal(await reader.locator('#login').isVisible(),true);
  assert.deepEqual(await reader.evaluate(()=>window.calls),[]);
  await reader.locator('#email').fill('owner@example.com');await reader.locator('#login-form button').click();
  await reader.waitForFunction(()=>document.getElementById('message').textContent.includes('Check your email'));
  assert.equal(await reader.evaluate(()=>window.signedInWith),'owner@example.com');

  // Signing out hides the desk.
  await page.locator('#signout').click();await page.waitForFunction(()=>!document.getElementById('login').hidden);
  assert.equal(await page.locator('#desk').isVisible(),false);
  assert.deepEqual(errors,[]);
  await browser.close();
  console.log('Live Odds page passed: editor gate, game list, Run now, flags, tracking, filters, escaping, errors and phone layout.');
})().catch(e=>{console.error(e);process.exit(1);});

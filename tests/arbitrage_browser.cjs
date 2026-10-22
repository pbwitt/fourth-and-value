// Arbitrage finder and plan in a real browser with stubbed data and Supabase: scheduled pairs,
// stakes and after-tax estimates, filters, Track tickets, safe rendering, Scan now (editor gate,
// game-lines and one-game scans, an outdated function), the paper record and the phone layout.
const {chromium}=require(process.env.FV_PLAYWRIGHT||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const A=require('../docs/assets/arbitrage.js');
const root=path.join(__dirname,'../docs');
const RULES=JSON.parse(fs.readFileSync(path.join(root,'arbitrage/data/rules.json'),'utf8'));

const NOW=Date.now(),START=new Date(NOW+6*3600e3).toISOString(),QUOTED=new Date(NOW-120e3).toISOString();
const HOME='Philadelphia 76ers',AWAY='Cleveland Cavaliers',EVIL='<img src=x onerror="window.hacked=1">';
const row=(book,market,side,price,line=null,player='')=>({event_id:'e'.repeat(32),game:`${AWAY} @ ${HOME}`,home_team:HOME,away_team:AWAY,
  commence_time:START,book,book_label:book,market,side,price,line,player,quoted_at:QUOTED});
const scheduled=(sport,rows)=>{
  const r=A.scan(A.fromSnapshot(rows,sport),{rules:RULES,sport,now:NOW});
  return {schema_version:1,sport,scanned_at:new Date(NOW-60e3).toISOString(),snapshot_at:QUOTED,feed_status:'ready',rules_version:RULES.version,
    stats:r.stats,locked:r.locked,check:r.check,near:r.near};
};
const DATA={
  nba:scheduled('NBA',[row('betrivers','player_points','Over',102,21.5,EVIL),row('bovada','player_points','Under',110,21.5,EVIL),
    row('draftkings','totals','Over',-105,237.5),row('betrivers','totals','Under',106,237.5),
    row('fanduel','player_rebounds','Over',-105,12.5,'Nikola Jokic'),row('draftkings','player_rebounds','Under',110,12.5,'Nikola Jokic'),
    row('betrivers','player_assists','Over',110,6.5,'Ajay Mitchell'),row('mybookieag','player_assists','Under',105,6.5,'Ajay Mitchell')]),
  nhl:scheduled('NHL',[]),mlb:scheduled('MLB',[]),
  nfl:{...scheduled('NFL',[]),feed_status:'failed'},
};
const ledger=sport=>({schema_version:1,sport,entries:sport==='NBA'?[{id:'x',sport,game:`${AWAY} @ ${HOME}`,commence_time:START,subject:'Evan Mobley · Points',
  kind:'two_way',licensed:false,status:'locked',tier:'assumed',first_seen:new Date(NOW-86400e3).toISOString(),last_seen:new Date(NOW-80000e3).toISOString(),
  snapshots:2,first_return_pct:2.9,best_return_pct:2.9,first_legs:[{book:'betrivers',side:'Over',line:21.5,price:102},{book:'bovada',side:'Under',line:21.5,price:110}]}]:[]});

const LINES_EVENT=JSON.stringify({id:'f'.repeat(32),sport_key:'americanfootball_nfl',commence_time:START,home_team:'Detroit Lions',away_team:'Green Bay Packers',
  bookmakers:[{key:'draftkings',title:'DraftKings',last_update:QUOTED,markets:[{key:'h2h',last_update:QUOTED,outcomes:[{name:'Detroit Lions',price:-150},{name:'Green Bay Packers',price:130}]}]},
    {key:'fanduel',title:'FanDuel',last_update:QUOTED,markets:[{key:'h2h',last_update:QUOTED,outcomes:[{name:'Detroit Lions',price:-120},{name:'Green Bay Packers',price:150}]},
      {key:'player_pass_yds',last_update:QUOTED,outcomes:[{name:'Over',price:110,point:245.5,description:'Jared Goff'}]}]},
    {key:'betmgm',title:'BetMGM',last_update:QUOTED,markets:[{key:'player_pass_yds',last_update:QUOTED,outcomes:[{name:'Under',price:105,point:245.5,description:'Jared Goff'}]}]}]});
const stub=`
window.mode='editor';window.calls=[];window.oldFunction=false;
window.supabaseClient={
  auth:{getUser:async()=>({data:{user:window.mode==='signedout'?null:{email:'e@x.com',app_metadata:window.mode==='editor'?{fv_editor:true}:{}}}}),
    onAuthStateChange:fn=>{window.authChanged=fn;}},
  functions:{invoke:async(name,{body})=>{
    window.calls.push([name,body]);
    const ev=${LINES_EVENT};
    if(body.scan==='lines'&&window.oldFunction)return {data:{sport:body.sport,games:[]},error:null};
    // The game-lines endpoint returns only moneylines, spreads and totals.
    const lines={...ev,bookmakers:ev.bookmakers.map(b=>({...b,markets:b.markets.filter(m=>['h2h','spreads','totals'].includes(m.key))}))};
    if(body.scan==='lines')return {data:{sport:body.sport,scan:'lines',regions:body.more?'us,us2':'us',events:[lines],fetched_at:new Date().toISOString(),cost:body.more?6:3,reused:false,remaining:9000},error:null};
    if(body.event)return {data:{event:ev,fetched_at:new Date().toISOString(),cost:7,reused:false,remaining:8990},error:null};
    return {data:{sport:body.sport,games:body.sport==='NFL'?[{id:ev.id,commence_time:ev.commence_time,home_team:ev.home_team,away_team:ev.away_team}]:[],remaining:9010},error:null};
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
    const data=p.match(/^\/arbitrage\/data\/(nba|nhl|mlb|nfl)\.json$/),led=p.match(/^\/arbitrage\/data\/ledger\/(nba|nhl|mlb|nfl)\.json$/);
    if(data)return route.fulfill({contentType:'application/json',body:JSON.stringify(DATA[data[1]])});
    if(led)return route.fulfill({contentType:'application/json',body:JSON.stringify(ledger(led[1].toUpperCase()))});
    try{return route.fulfill({contentType:p.endsWith('.js')?'application/javascript':p.endsWith('.css')?'text/css':p.endsWith('.json')?'application/json':'text/html',body:fs.readFileSync(root+p)});}
    catch{return route.fulfill({status:404,body:'Not found'});}
  });
  const errors=[];page.on('pageerror',e=>errors.push(e.message));

  await page.goto('https://fourthandvalue.com/arbitrage/');
  await page.evaluate(()=>localStorage.clear());await page.reload();
  await page.waitForFunction(()=>document.getElementById('count-locked').textContent!=='—');
  // Locked: Mobley (BetRivers + Bovada), the total (DraftKings + BetRivers), Jokic (FanDuel + DraftKings). MyBookie has no rules on file.
  assert.equal(await page.locator('#count-locked').textContent(),'3');
  assert.equal(await page.locator('#count-check').textContent(),'1');
  assert.match(await page.locator('#check .pair').textContent(),/MyBookie’s rules for Assists are not on file|mybookieag’s rules for Assists are not on file/);
  const first=page.locator('#locked .pair').first();
  assert.match(await first.textContent(),/<img src=x onerror="window\.hacked=1"> · Points/,'player names render as text');
  assert.equal(await page.evaluate(()=>window.hacked),undefined);
  assert.match(await first.textContent(),/offshore: Bovada/);assert.match(await first.textContent(),/rules assumed/);
  // $100 split so both sides pay the same; after tax at 24%, itemizing.
  const stakes=await first.locator('tbody td:nth-child(4)').allTextContents();
  assert.deepEqual(stakes,['$50.97','$49.03']);
  assert.match(await first.locator('.result').textContent(),/Pays \$102\.96 whichever side wins: \$2\.96 \(\+2\.96%\)\. After federal tax at 24%, itemizing: \$1\.03\./);
  await page.fill('#stake','1000');
  assert.deepEqual(await first.locator('tbody td:nth-child(4)').allTextContents(),['$509.71','$490.29']);
  await page.uncheck('#itemize');
  assert.match(await first.locator('.result').textContent(),/standard deduction: −\$99\.83/);
  await page.check('#taxpositive');
  assert.equal(await page.locator('#count-locked').textContent(),'0','every pair loses money after tax without itemizing');
  await page.uncheck('#taxpositive');await page.check('#itemize');
  await page.check('#licensed');
  assert.equal(await page.locator('#count-locked').textContent(),'2','the Bovada pair is hidden');
  await page.uncheck('#licensed');
  await page.reload();await page.waitForFunction(()=>document.getElementById('count-locked').textContent!=='—');
  assert.equal(await page.inputValue('#stake'),'1000','settings are remembered');

  // Scans and the failed NFL feed are stated; the paper record reads the ledger.
  assert.match(await page.locator('#scans').textContent(),/The NFL feed’s last refresh failed/);
  assert.match(await page.locator('#record').textContent(),/NBA1101\$29\.61\$10\.27/);

  // Track opens Bet Tracker's dialog for one side of the pair.
  await page.locator('#locked .pair').first().locator('.track-cell button').first().click();
  await page.waitForSelector('#fv-bet-tracker[open]');
  assert.match(await page.locator('#fv-track-description').textContent(),/NBA · Cleveland Cavaliers @ Philadelphia 76ers · .* · Over · 21\.5 · Points · BetRivers/);
  assert.equal(await page.inputValue('#fv-track-odds'),'102');
  await page.locator('#fv-track-close').click();

  // Scan now: the editor's panel opens; a game-lines scan replaces that league's game lines.
  await page.waitForFunction(()=>!document.getElementById('scan-desk').hidden);
  assert.equal(await page.locator('#scan-panel').getAttribute('open'),'');
  await page.selectOption('#scan-sport','NFL');
  await page.waitForFunction(()=>document.querySelectorAll('#scan-game option[value]:not([value=""])').length===1,null,{timeout:5000}).catch(async e=>{console.log(await page.evaluate(()=>[document.getElementById('scan-game').outerHTML,JSON.stringify(window.calls),document.getElementById('scan-message').textContent]));throw e;});
  await page.check('#scan-more');
  assert.match(await page.locator('#scan-lines-cost').textContent(),/6 credits/);
  await page.locator('#scan-lines').click();
  await page.waitForFunction(()=>/game lines: 1 locked/.test(document.getElementById('scan-message').textContent));
  assert.match(await page.locator('#scan-message').textContent(),/This scan cost 6 credits\. 9,000 credits left\./);
  assert.deepEqual((await page.evaluate(()=>window.calls.filter(c=>c[1].scan)))[0][1],{sport:'NFL',scan:'lines',more:true});
  assert.equal(await page.locator('#count-locked').textContent(),'4');
  assert.match(await page.locator('#scans').textContent(),/NFL: fresh game-lines scan .* with more books/);
  // One game with props: the Jared Goff pair is graded by the same engine.
  await page.locator('#scan-event').click();
  await page.waitForFunction(()=>/Green Bay Packers @ Detroit Lions: 2 locked/.test(document.getElementById('scan-message').textContent));
  assert.match(await page.locator('#locked').textContent(),/Jared Goff · Passing yards/);
  assert.equal(await page.locator('#count-locked').textContent(),'5');

  // An outdated function lists games for a scan request; the page says what to do.
  await page.evaluate(()=>{window.oldFunction=true;});
  await page.locator('#scan-lines').click();
  await page.waitForFunction(()=>/Redeploy it in Supabase/.test(document.getElementById('scan-message').textContent));

  // Non-editors and signed-out readers see no scan controls.
  await page.evaluate(()=>{window.mode='reader';window.authChanged();});
  await page.waitForFunction(()=>document.getElementById('scan-desk').hidden&&/does not have editor access/.test(document.getElementById('scan-message').textContent));
  await page.evaluate(()=>{window.mode='signedout';window.authChanged();});
  await page.waitForFunction(()=>!document.getElementById('scan-login').hidden);

  // Phone layout: nothing wider than the screen.
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'no horizontal scroll at 390px');

  // The plan renders its tables and the history it reports.
  await page.goto('https://fourthandvalue.com/arbitrage/plan.html');
  assert.equal(await page.locator('h1').count(),1);
  assert.match(await page.locator('#history').evaluate(e=>e.parentElement.textContent),/Still there at the next refresh/);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'plan: no horizontal scroll at 390px');

  assert.deepEqual(errors,[]);
  await browser.close();
  console.log('PASS: arbitrage finder and plan in the browser.');
})().catch(error=>{console.error(error);process.exit(1);});

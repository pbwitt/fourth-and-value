// Bet Tracker's manual form against local feed fixtures: league -> date -> game fills
// teams and markets, the grading check shows before saving, and no database is touched.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..','docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});

const mlbTeam=(name,abbreviation,teamName,score)=>({team:{name,abbreviation,teamName},score});
const mlbGame=(gamePk,away,home,gameDate)=>({gamePk,gameDate,scheduledInnings:9,linescore:{currentInning:9},
  status:{abstractGameState:'Final',codedGameState:'F',detailedState:'Final'},teams:{away,home}});
const schedule={dates:[{games:[
  mlbGame(776001,mlbTeam('Chicago Cubs','CHC','Cubs',1),mlbTeam('San Diego Padres','SD','Padres',4),'2026-09-30T23:10:00Z'),
  mlbGame(776002,mlbTeam('Boston Red Sox','BOS','Red Sox',2),mlbTeam('New York Yankees','NYY','Yankees',9),'2026-09-30T17:05:00Z')]}]};
const boxscore={teams:{
  home:{batters:[1],pitchers:[],players:{ID1:{person:{id:1,fullName:'Fernando Tatis Jr.'},stats:{batting:{hits:2,totalBases:3}}}}},
  away:{batters:[2],pitchers:[],players:{ID2:{person:{id:2,fullName:'Ian Happ'},stats:{batting:{hits:0}}}}}}};

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`,browser=await chromium.launch({headless:true});
  try{
    const p=await browser.newPage({viewport:{width:390,height:900}}),errors=[];
    p.on('pageerror',error=>errors.push(error.message));
    await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
    await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await p.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await p.route('https://statsapi.mlb.com/**',route=>{
      const url=route.request().url();
      route.fulfill({json:url.includes('/boxscore')?boxscore:url.includes('gamePk=776001')?{dates:[{games:[schedule.dates[0].games[0]]}]}:schedule});
    });
    await p.addInitScript(()=>{
      window.db={rows:[]};const user={id:'test-owner',email:'reader@example.com'};
      const query={order:async()=>({data:db.rows,error:null})};
      window.supabase={createClient:()=>({
        auth:{getSession:async()=>({data:{session:{user}}}),onAuthStateChange:()=>{}},
        functions:{invoke:async()=>({data:{games:[]},error:null})},
        from:()=>({select:()=>query,insert:async row=>{db.rows.push(row);return {error:null};}})})};
    });
    await p.goto(base+'/tracking/');
    await p.waitForSelector('#betTracker',{state:'visible'});

    // League -> date -> game: teams come from the schedule, markets from the league.
    await p.locator('#mb-league').selectOption('MLB');
    await p.locator('#mb-date').fill('2026-09-30');await p.locator('#mb-date').dispatchEvent('change');
    await p.waitForFunction(()=>document.querySelectorAll('#mb-game option').length===3);
    const labels=await p.locator('#mb-game option').allTextContents();
    assert.equal(labels[1],'BOS @ NYY · Final 2–9');assert.equal(labels[2],'CHC @ SD · Final 1–4');
    await p.locator('#mb-game').selectOption('776001');
    await p.locator('#mb-market').selectOption('totals');
    assert.deepEqual(await p.locator('#mb-side option').allTextContents(),['Over','Under']);
    assert(await p.locator('#mb-player-group').isHidden());
    await p.locator('#mb-side').selectOption('under');await p.locator('#mb-line').fill('7.5');
    await p.locator('#mb-book').fill('DraftKings');await p.locator('#mb-odds').fill('-110');await p.locator('#mb-stake').fill('25');
    assert.match(await p.locator('#mb-check').textContent(),/^✓ Grades automatically/);
    await p.locator('#mb-save').click();
    await p.waitForFunction(()=>document.getElementById('mb-feedback').textContent==='Saved.');
    let row=await p.evaluate(()=>db.rows.at(-1));
    assert.deepEqual([row.league,row.game_date,row.team_away,row.team_home,row.market_type,row.side,row.line,row.book,row.odds,row.stake_dollars,row.status],
      ['MLB','2026-09-30','CHC','SD','totals','under',7.5,'DraftKings',-110,25,'pending']);
    assert.match(row.id,/^[0-9a-f-]{36}$/);
    assert.equal(await p.locator('#mb-line').inputValue(),'');

    // Moneyline: pick a team, no line field.
    await p.locator('#mb-market').selectOption('h2h');
    assert.deepEqual(await p.locator('#mb-side option').allTextContents(),['CHC','SD']);
    assert(await p.locator('#mb-line-group').isHidden());
    await p.locator('#mb-side').selectOption('home');await p.locator('#mb-odds').fill('-150');await p.locator('#mb-stake').fill('15');
    await p.locator('#mb-save').click();
    await p.waitForFunction(()=>db.rows.length===2);
    row=await p.evaluate(()=>db.rows.at(-1));assert.deepEqual([row.market_type,row.side,row.line],['h2h','SD',null]);

    // Player props: names come from the final box score and a misspelled name is flagged before saving.
    await p.locator('#mb-market').selectOption('batter_hits');
    assert(await p.locator('#mb-player-group').isVisible());
    await p.waitForFunction(()=>document.querySelectorAll('#mb-players option').length===2);
    await p.locator('#mb-player').fill('Fernando Tattis');
    assert.match(await p.locator('#mb-check').textContent(),/^⚠ .*isn’t in this game’s box score/);
    await p.locator('#mb-player').fill('Fernando Tatis Jr.');
    assert.match(await p.locator('#mb-check').textContent(),/^✓/);

    // Incomplete entries explain themselves instead of saving.
    await p.locator('#mb-line').fill('');await p.locator('#mb-save').click();
    assert.match(await p.locator('#mb-feedback').textContent(),/Enter the line/);
    assert.equal(await p.evaluate(()=>db.rows.length),2);

    // A player prop saves the player's team from the box score, and the bet list shows it beside the name.
    await p.locator('#mb-line').fill('1.5');await p.locator('#mb-odds').fill('+120');await p.locator('#mb-stake').fill('10');
    await p.locator('#mb-save').click();
    await p.waitForFunction(()=>db.rows.length===3);
    row=await p.evaluate(()=>db.rows.at(-1));assert.deepEqual([row.player,row.player_team],['Fernando Tatis Jr.','SD']);
    await p.waitForFunction(()=>document.querySelector('.bet-card-player .team-tag')?.textContent==='SD');
    assert.equal(await p.locator('#betsTableBody .team-tag').first().textContent(),'SD');

    for(const width of [390,1440]){await p.setViewportSize({width,height:900});
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`no overflow at ${width}`);}
    assert.deepEqual(errors,[]);
    console.log('PASS: manual form fills games and markets from the schedule, previews grading, saves offer-identical rows and shows player teams.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

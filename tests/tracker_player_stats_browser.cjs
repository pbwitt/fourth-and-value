// Real tracker + shared popover, with public stats and the private ledger mocked.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
const nhlPage='<section id="recent"><table><tbody>'+[
  ['2026-10-08','@ BOS','20:30',4,1,0,1],['2026-10-06','vs NYR','19:00',2,0,1,1],['2026-10-04','@ NJD','18:20',3,0,0,0],
].map(cells=>'<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table></section>';
const mlb={rows:[{player:'Example Pitcher',market:'pitcher_strikeouts',model_mean:99,model_probability:.99,side:'Over',line:2.5,
  player_context:{schema_version:1,source:'MLB completed-game box scores',through:'2026-10-07',sample_label:'starts',stat_label:'K',workload_unit:'IP',
    recent:[{games:5,mean:6.4,workload:5.67,pitches:92}],games:[{date:'2026-10-07',opp:'@ NYY',k:7}],game_columns:[['date','Date'],['opp','Opp'],['k','K']],game_focus:'k',
    trend:{label:'K',rows:[['2026-10-01',5,null,'vs BOS',5],['2026-10-07',7,null,'@ NYY',6]]},inputs:[{label:'Should not show',value:99}]}}]};

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
    const base=`http://127.0.0.1:${server.address().port}`,p=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],requests=[];
    p.on('pageerror',error=>errors.push(error.message));p.on('request',request=>requests.push(new URL(request.url()).pathname));
    await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
    await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await p.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await p.route('**/nhl/players/players.json',route=>route.fulfill({json:{players:{'example-skater':{name:'Example Skater'},'other-skater':{name:'Other Skater'}}}}));
    await p.route('**/nhl/players/example-skater/',route=>route.fulfill({contentType:'text/html',body:nhlPage}));
    await p.route('**/nhl/players/other-skater/',route=>route.fulfill({contentType:'text/html',body:nhlPage}));
    let failMLB=true,mlbDelay=0;
    await p.route('**/mlb/data/latest.json',async route=>{
      if(mlbDelay)await new Promise(resolve=>setTimeout(resolve,mlbDelay));
      await route.fulfill(failMLB?{status:503,body:'Unavailable'}:{json:mlb});
    });
    await p.route('**/props/model-context.json',route=>route.fulfill({json:{groups:{'["g","Example Back","rush_yds"]':{projection:{family:'rush',
      current_sample:[{season:2026,week:1,rushing_yards:60,carries:15}],carries:15,yards_per_carry:4,mean_stages:{final:60}}}}}}));
    await p.route('**/nba/data/latest.json',route=>route.fulfill({json:{rows:[]}}));
    await p.addInitScript(()=>{
      const user={id:'owner',email:'reader@example.com'},common={user_id:'owner',game_date:'2026-06-01',team_away:'Away',team_home:'Home',
        status:'won',odds:-110,stake_dollars:25,payout:47.73,line:3,side:'under',book:'Test book'};
      window.db={rows:[{...common,id:'nhl',league:'NHL',player:'Example Skater',market_type:'sog'},
        {...common,id:'mlb',league:'MLB',player:'Example Pitcher',market_type:'pitcher_strikeouts',line:6},
        {...common,id:'nfl',league:'NFL',player:'Example Back',market_type:'rush_yds',line:60.5},
        {...common,id:'nba',league:'NBA',player:'Missing Player',market_type:'player_points'},
        {...common,id:'game',league:'NHL',player:null,market_type:'totals'}]};
      window.supabase={createClient:()=>({auth:{getSession:async()=>({data:{session:{user}}}),onAuthStateChange:()=>{}},
        functions:{invoke:async()=>({data:{games:[]},error:null})},
        from:()=>({select:()=>({order:async()=>({data:structuredClone(db.rows),error:null})})})})};
    });
    await p.goto(base+'/tracking/');await p.locator('#betsTableBody .pc-name').first().waitFor();
    assert.equal(requests.some(url=>/model-context|data\/latest|players\.json/.test(url)),false,'stats load only on interaction');
    assert.equal(await p.locator('#betsTableBody .pc-name').count(),4,'game markets stay plain text');
    const pop=p.locator('.pc-pop'),nhl=p.locator('#betsTableBody .pc-name',{hasText:'Example Skater'});
    await nhl.scrollIntoViewIfNeeded();await nhl.hover();await pop.waitFor({state:'visible'});
    await pop.getByRole('heading',{name:'Recent games',exact:true}).waitFor();
    assert.match(await pop.textContent(),/Latest published player stats.*after this bet/s);
    assert.match(await pop.textContent(),/1 of 2 cleared Under 3/,'push is excluded from the win count');
    assert(await pop.locator('.pc-trend .pc-push').count()>0,'push remains visible on the chart');
    assert.match(await pop.textContent(),/through Oct 8, 2026/);
    assert.doesNotMatch(await pop.textContent(),/Model projection|How it works|Track record|NaN|undefined/);
    assert.equal(await pop.locator('.pc-share').count(),0,'private tracker has no broken public snapshot share link');
    await pop.hover();await p.waitForTimeout(250);assert(await pop.isVisible());
    await p.mouse.move(1,1);await pop.waitFor({state:'hidden'});

    // Keyboard pins the snapshot; live redraws keep it open and attached.
    await nhl.focus();await p.keyboard.press('Enter');await pop.waitFor({state:'visible'});
    await p.evaluate(()=>renderBets());assert(await pop.isVisible());
    assert.equal(await nhl.getAttribute('aria-expanded'),'true');
    await p.keyboard.press('Escape');await pop.waitFor({state:'hidden'});
    assert.equal(await nhl.evaluate(el=>el===document.activeElement),true);
    assert.equal(requests.filter(url=>url==='/nhl/players/example-skater/').length,1,'reopening/repainting reuses stats');

    // A failed stats feed is retryable, and a late response cannot replace another player.
    const pitcher=p.locator('#betsTableBody .pc-name',{hasText:'Example Pitcher'});
    await pitcher.click();await pop.getByText(/could not load/).waitFor();await pop.locator('.pc-close').click();
    failMLB=false;mlbDelay=250;
    await pitcher.click();await pop.getByText('Loading player stats…').waitFor();
    await nhl.click();await p.waitForTimeout(350);
    assert.equal(await pop.locator('.pc-pop-head strong').textContent(),'Example Skater');
    await pop.locator('.pc-close').click();await pitcher.click();
    await pop.getByRole('heading',{name:'Recent games',exact:true}).waitFor();
    assert.match(await pop.textContent(),/5⅔/);assert.match(await pop.textContent(),/cleared Under 6/);
    assert.doesNotMatch(await pop.textContent(),/99|Should not show|Model projection/);
    await pop.locator('.pc-close').click();
    await p.locator('#betsTableBody .pc-name',{hasText:'Example Back'}).click();
    await pop.getByRole('heading',{name:'Recent games',exact:true}).waitFor();
    assert.match(await pop.textContent(),/Rush yds \/ game60/);await pop.locator('.pc-close').click();
    await p.locator('#betsTableBody .pc-name',{hasText:'Missing Player'}).click();
    await pop.getByText('No published stats for this player and market yet.').waitFor();await pop.locator('.pc-close').click();

    // The desktop live-bet panel uses the same name trigger, including after polling redraws.
    await p.evaluate(()=>{
      liveViews=new Map([['nhl',{status:'live',kind:'prop',tone:'ahead',value:2,line:3,label:'1 to spare',progress:.66,
        game:{id:'g',league:'NHL',state:'live',detail:'2nd period',home:{abbrev:'HOM',score:1},away:{abbrev:'AWY',score:0}}}]]);
      renderLivePanel();
    });
    const liveName=p.locator('#livePanel .pc-name');await liveName.click();
    await pop.waitFor({state:'visible'});await p.evaluate(()=>{renderBets();renderLivePanel();});
    assert(await pop.isVisible());assert.equal(await liveName.getAttribute('aria-expanded'),'true');
    await p.keyboard.press('Escape');await pop.waitFor({state:'hidden'});
    assert.equal(await liveName.evaluate(el=>el===document.activeElement),true);

    for(const width of [320,390,768,1440]){
      await p.setViewportSize({width,height:900});
      const name=p.locator((width<=820?'#betCards':'#betsTableBody')+' .pc-name',{hasText:'Example Skater'});
      await name.click();await pop.waitFor({state:'visible'});
      if(width<=600)assert.match(await pop.getAttribute('class'),/pc-sheet/);
      if(process.env.TRACKER_STATS_SCREENSHOTS)await p.screenshot({path:path.join(process.env.TRACKER_STATS_SCREENSHOTS,`tracker-player-stats-${width}.png`)});
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`page fits ${width}`);
      assert.equal(await pop.evaluate(el=>el.scrollWidth>el.clientWidth+1),false,`snapshot fits ${width}`);
      await p.keyboard.press('Escape');await pop.waitFor({state:'hidden'});
    }
    await p.setViewportSize({width:390,height:900});
    await p.locator('#betCards .pc-name',{hasText:'Example Skater'}).click();
    await p.selectOption('#f-sport','MLB');await pop.waitFor({state:'hidden'});
    assert.equal(await p.locator('#betCards [data-edit-bet]').count(),1,'edit control retained');
    assert.equal(await p.evaluate(()=>db.rows[0].stake_dollars),25,'stats never write to bets');
    assert.deepEqual(errors,[]);
    console.log('PASS: tracker hover/click/keyboard stats, mobile sheets, live redraws, caching, retry, stale responses, filters and exact saved lines.');
  }finally{if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

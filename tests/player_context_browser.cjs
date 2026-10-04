// Offline fixtures: exercise the published board components without paid APIs.
const assert=require('node:assert/strict');
const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
require('../docs/assets/player-context.js');
const {rowHTML}=require('../docs/assets/briefing-picks.js');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true,...(process.env.FV_BUNDLED_BROWSER?{}:{executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'})});
  const base=`http://127.0.0.1:${server.address().port}`,now=new Date().toISOString(),future=new Date(Date.now()+3600e3).toISOString();
  const errors=[];
  const common={event_id:'g',game:'Example Away @ Example Home',commence_time:future,quoted_at:now,player:'Example Player',side:'Over',line:5.5,price:110,
    book:'draftkings',book_label:'DraftKings',book_probability:1/2.1,fair_probability:.5,consensus_probability:.5,paired_books:3,other_books:3,
    model_status:'Experimental independent forecast',model_version:'fixture',model_data_checked_at:now,independent_probability:.55,final_probability:.55,
    market_probability:.5,push_probability:0,model_probability:.55,model_push_probability:0,model_mean:5.7,model_ev_pct:15.5,model_edge_pp:7.4,model_fair_price:-122,is_model_pick:true};
  const pitcher={...common,market:'pitcher_strikeouts',market_family:'props',market_label:'Pitcher strikeouts',game_type:'R',phase:'Regular season',
    player_context:{schema_version:1,source:'MLB completed-game box scores',through:'2026-09-30',sample_games:15,sample_label:'starts',stat_label:'K',workload_label:'Innings / start',workload_unit:'IP',
      recent:[{games:5,mean:5.8,workload:5.67},{games:10,mean:6.1,workload:5.4},{games:15,mean:5.9,workload:5.53}],
      inputs:[{label:'Recent innings / start',value:5.62,unit:'IP',used:true,detail:'Last 5 starts; prior-adjusted'},
        {label:'Recent pitches / start',value:88,used:true,detail:'Last 5 starts; prior-adjusted'},
        {label:'Pitcher strikeout rate',value:25.1,unit:'%',used:true,detail:'Up to 15 starts; K / batters faced'},
        {label:'Opponent strikeout rate',value:22.6,unit:'%',used:true,detail:'Up to 40 games; K / PA'}],
      note:'Model rates include fixed priors; observed averages do not. IP averages are decimal innings (5.50 = 5½ innings).'},
    stat_context:{group:'pitching',innings:'160.2',starts:28,strikeouts:172,k_per_nine:9.63,era:3.42}};
  const nhl={...common,market:'player_shots_on_goal',market_label:'Shots on goal',line:2.5,projected_mean:3.1,projected_toi:20.4,
    player_context:{schema_version:1,source:'NHL completed-game logs',through:'2026-09-30',sample_games:164,sample_label:'appearances',stat_label:'SOG',workload_label:'Ice time',workload_unit:'min',
      recent:[{games:5,mean:3.4,workload:21.2},{games:10,mean:3.1,workload:20.9},{games:20,mean:3.2,workload:20.7}],
      inputs:[{label:'Projected ice time',value:20.4,unit:'min',used:true,detail:'Prior-adjusted workload estimate'},
        {label:'Weighted SOG / 60',value:9.12,used:true,detail:'Prior-adjusted production per 60 minutes'}],
      note:'Newer games carry more weight. Ice time has a 30-day half-life; production per minute has a 120-day half-life. Observed averages are unweighted.'}};
  try{
    const p=await browser.newPage();p.on('pageerror',e=>errors.push(e.message));
    for(const [sport,row] of [['mlb',pitcher],['nhl',nhl]]){
      let data={status:'ready',last_success_at:now,history_checked_at:now,model_checked_at:now,history_through_date:'2026-09-30',season:2026,events:[],rows:[row]};
      await p.route(`**/${sport}/data/latest.json`,route=>route.fulfill({json:data}));
      for(const width of [320,390,768,1440]){
        await p.setViewportSize({width,height:1050});await p.goto(base+`/${sport}/props/`);await p.waitForSelector('.player-context');
        assert.equal(await p.locator('.player-context').first().locator('.pc-stats .pc-stat').count(),3);
        await p.locator('.pc-more summary').click();
        assert(await p.getByRole('table').first().isVisible());
        assert.match(await p.locator('.player-context').first().textContent(),/Observed recent form/);
        assert.doesNotMatch(await p.locator('.prop-card').textContent(),/NaN|undefined/);
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${sport} overflow at ${width}`);
        assert(await p.locator('.prop-card button').count()>0,'Tracking control retained');
        if(width===390||width===1440){await p.evaluate(()=>scrollTo(0,0));await p.locator('.prop-card').screenshot({path:`/tmp/fv-player-card-${sport}-${width}.png`});await p.screenshot({path:`/tmp/fv-player-${sport}-${width}.png`,fullPage:true});}
      }
      if(sport==='nhl'){
        data={...data,rows:[{...row,model_data_checked_at:new Date(Date.now()-37*3600e3).toISOString()}]};
        await p.reload();await p.waitForSelector('.prop-card');assert.equal(await p.locator('.player-context').count(),0,'Expired NHL model context hidden');
      }
    }
    for(const width of [320,390,768,1440]){
      await p.setViewportSize({width,height:1000});await p.goto(base+'/briefing/');
      await p.waitForFunction(()=>!document.getElementById('picks-status').textContent.includes('Loading'));
      await p.evaluate(html=>document.getElementById('daily-picks-rows').innerHTML=html,
        rowHTML({...pitcher,sport:'MLB',url:'/mlb/props/',card_snapshot_at:now}));
      await p.locator('#daily-picks-rows .pc-more summary').click();
      const table=p.locator('#daily-picks-rows .pc-recent table');
      assert.equal(await table.locator('thead').evaluate(e=>getComputedStyle(e).display),'table-header-group');
      assert.equal(await table.locator('tbody tr').first().evaluate(e=>getComputedStyle(e).display),'table-row');
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`Daily picks context overflow at ${width}`);
    }
    assert.deepEqual(errors,[]);
    console.log('PASS: player context, source dates, expandable inputs and tracking at four widths; stale NHL context hidden.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});

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
      recent:[{games:5,mean:5.8,workload:5.67,pitches:91.6},{games:10,mean:6.1,workload:5.4,pitches:90.2},{games:15,mean:5.9,workload:5.53,pitches:89.7}],
      games:[{date:'2026-09-28',opp:'@ BOS',ip:'6⅓',pitches:98,k:8,bb:1,er:2},{date:'2026-09-22',opp:'vs TOR',ip:'5⅔',pitches:94,k:6,bb:2,er:3},
        {date:'2026-09-16',opp:'vs BAL',ip:'5',pitches:88,k:4,bb:3,er:2},{date:'2026-09-10',opp:'@ TB',ip:'6',pitches:91,k:7,bb:1,er:1},{date:'2026-09-04',opp:'vs SEA',ip:'5⅓',pitches:87,k:4,bb:2,er:4}],
      game_columns:[['date','Date'],['opp','Opp'],['ip','IP'],['pitches','Pitches'],['k','K'],['bb','BB'],['er','ER']],game_focus:'k',
      trend:{label:'K',note:'Every start in the window counts equally: up to 15 for rates and the last 5 for workload.',
        rows:[['2026-08-17',5,null,'vs NYY',92],['2026-08-23',7,null,'@ TOR',97],['2026-08-29',3,null,'vs BAL',85],['2026-09-04',4,null,'vs SEA',87],['2026-09-10',7,null,'@ TB',91],
          ['2026-09-16',4,null,'vs BAL',88],['2026-09-22',6,null,'vs TOR',94],['2026-09-28',8,null,'@ BOS',98],['2026-09-30',6,null,'vs NYY',96],['2026-10-02',5,null,'@ TOR',90]]},
      build:{steps:[{label:'Batters faced per start',value:23.4,unit:'',op:null},{label:'Strikeout rate',value:25.1,unit:'%',op:'×'},{label:'Opponent adjustment',value:1.02,unit:'×',op:'×'},
        {label:'Simple estimate',value:5.99,unit:'K',op:'='},{label:'Adjusted to past results',value:5.7,unit:'K',op:'→'}],note:'This market uses the simple estimate directly.'},
      distribution:{start:1,p:[.02,.05,.1,.15,.18,.17,.13,.09,.06,.03,.02],low:true,high:true},
      blend:[{label:'Strikeout rate',own:.78,detail:'352 batters faced in his last 15 starts'},{label:'Outs per start',own:.71,detail:'His last 5 starts'}],
      opponent:{team:'BOS',label:'Opposing lineup',items:[{label:'Strikeout rate',value:23.9,unit:'%',league:22.4,rank:'7th highest of 30',used:true},{label:'Runs per game',value:4.61,unit:'',league:4.4,rank:'9th most of 30',used:false}]},
      missing:['Weather and umpire','Today’s actual batting order (team rates are used)','Announced pitch limits and injuries'],
      inputs:[{label:'Recent innings / start',value:5.62,unit:'IP',used:true,detail:'Last 5 starts; prior-adjusted'},
        {label:'Recent pitches / start',value:88,used:true,detail:'Last 5 starts; prior-adjusted'},
        {label:'Pitcher strikeout rate',value:25.1,unit:'%',used:true,detail:'Up to 15 starts; K / batters faced'},
        {label:'Opponent strikeout rate',value:22.6,unit:'%',used:true,detail:'Up to 40 games; K / PA'}],
      note:'Model rates include fixed priors; observed averages do not. Innings are shown in thirds (5⅔ = five innings and two outs).'},
    stat_context:{group:'pitching',innings:'160.2',starts:28,strikeouts:172,k_per_nine:9.63,era:3.42},model_conditional_probability:.55,consensus_probability:.5,player_position:'SP'};
  // The NHL track record shows only for the running model version; match the published file.
  const nhlVersion=JSON.parse(fs.readFileSync(path.join(root,'nhl/data/track-record.json'),'utf8')).model_version;
  const nhl={...common,model_version:nhlVersion,market:'player_shots_on_goal',market_label:'Shots on goal',line:2.5,projected_mean:3.1,projected_toi:20.4,conditional_probability:.55,market_probability:.5,player_position:'C',
    player_context:{schema_version:1,source:'NHL completed-game logs',through:'2026-09-30',sample_games:164,sample_label:'appearances',stat_label:'SOG',workload_label:'Ice time',workload_unit:'min',
      recent:[{games:5,mean:3.4,workload:21.2},{games:10,mean:3.1,workload:20.9},{games:20,mean:3.2,workload:20.7}],
      games:[{date:'2026-04-16',opp:'vs CHI',toi:'21:05',shots:4,goals:1,assists:0,points:1},{date:'2026-04-14',opp:'@ TOR',toi:'20:41',shots:3,goals:0,assists:1,points:1}],
      game_columns:[['date','Date'],['opp','Opp'],['toi','TOI'],['shots','SOG'],['goals','G'],['assists','A'],['points','P']],game_focus:'shots',
      trend:{label:'SOG',note:'Faded games count less. A game’s weight halves every 120 days for production and every 30 days for ice time.',
        rows:[['2026-03-28',2,.36,'@ MTL',19.5],['2026-03-31',5,.37,'vs OTT',21.2],['2026-04-02',3,.37,'vs DET',20.4],['2026-04-04',1,.38,'@ BOS',18.9],['2026-04-07',4,.39,'vs TOR',22.0],
          ['2026-04-09',2,.39,'@ NYR',19.8],['2026-04-11',6,.40,'vs FLA',21.7],['2026-04-14',3,.40,'@ TOR',20.7],['2026-04-16',4,.41,'vs CHI',21.1],['2026-10-02',3,.98,'vs CHI',20.3]]},
      build:{steps:[{label:'Projected ice time',value:20.4,unit:'min',op:null},{label:'SOG',value:9.12,unit:'per 60 min',op:'×'},{label:'Expected shots',value:3.1,unit:'SOG',op:'='}],
        note:'Ice time × production per 60 minutes, both recency-weighted and blended with a position average.'},
      distribution:{start:0,p:[.05,.15,.22,.22,.17,.1,.05,.04],low:false,high:true},
      blend:[{label:'Production rate',own:.86,detail:'Recency-weighted games; the rest is the position average'},{label:'Ice time',own:.41,detail:'Recency-weighted games; the rest is the position average'}],
      opponent:{team:'CHI',label:'Opposing defense',items:[{label:'Shots allowed per game',value:31.9,unit:'',league:29.8,rank:'4th most of 32',used:false},{label:'Regulation goals allowed per game',value:3.21,unit:'',league:2.9,rank:'3rd most of 32',used:false}]},
      missing:['Opponent defense and goalie','Linemates and power-play role','Injuries and late lineup changes'],
      inputs:[{label:'Projected ice time',value:20.4,unit:'min',used:true,detail:'Prior-adjusted workload estimate'},
        {label:'Weighted SOG / 60',value:9.12,used:true,detail:'Prior-adjusted production per 60 minutes'}],
      note:'Newer games carry more weight. Ice time has a 30-day half-life; production per minute has a 120-day half-life. Observed averages are unweighted.'}};
  try{
    const p=await browser.newPage();p.on('pageerror',e=>errors.push(e.message));
    await p.route('**/nhl/players/players.json',route=>route.fulfill({json:{players:{'example-player':{name:'Example Player',player_id:null},'other':{name:'Other Player',player_id:8}}}}));
    for(const [sport,row] of [['mlb',pitcher],['nhl',nhl]]){
      let data={status:'ready',last_success_at:now,history_checked_at:now,model_checked_at:now,history_through_date:'2026-09-30',season:2026,events:[],rows:[row]};
      await p.route(`**/${sport}/data/latest.json`,route=>route.fulfill({json:data}));
      for(const width of [320,390,768,1440]){
        await p.setViewportSize({width,height:1050});await p.goto(base+`/${sport}/props/`);await p.waitForSelector('.prop-card');
        assert.equal(await p.locator('.prop-card .player-context').count(),0,'Context moved off the card into the name pop-up');
        const name=p.locator('.prop-card .pc-name').first(),pop=p.locator('.pc-pop');
        assert.equal(await name.getAttribute('aria-expanded'),'false');
        assert.equal(await p.locator('.prop-card .pc-pos').first().textContent(),sport==='mlb'?'SP':'C','position beside the name');
        await name.click();await pop.waitFor({state:'visible'});
        assert.equal(await name.getAttribute('aria-expanded'),'true');
        assert.equal(await pop.getAttribute('role'),'dialog');
        assert(await pop.locator('.pc-stats .pc-stat').count()>=3,'snapshot tiles');
        assert.equal(await pop.locator('.pc-log tbody tr').count(),sport==='mlb'?5:2,'recent games');
        const text=await pop.textContent();
        assert.match(text,/Recent games/);assert.match(text,/What goes into the forecast/);assert.doesNotMatch(text,/NaN|undefined|null/);
        if(sport==='mlb'){assert.match(text,/Pitches \/ start/);assert.match(text,/6⅓/);assert.match(text,/28 starts · 160⅔ IP/);}
        else{assert.match(text,/21:12/);await pop.locator('.pc-pop-link a[href="/nhl/players/example-player/"]').waitFor({state:'visible'});}
        const box=await pop.boundingBox(),nameBox=await name.boundingBox();
        if(width<=600){assert.match(await pop.getAttribute('class'),/pc-sheet/);assert(Math.abs(box.y+box.height-1050)<=1,'bottom sheet');}
        else{assert(box.y>=nameBox.y+nameBox.height||box.y+box.height<=nameBox.y,'anchored beside the name');assert(box.x+box.width<=width,'inside the viewport');}
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${sport} overflow at ${width}`);
        if(width===390||width===1440)await p.screenshot({path:`/tmp/fv-player-pop-${sport}-${width}.png`});
        if(width===390||width===1440){
          for(const tab of ['model','record']){
            if(!await pop.locator(`[data-tab=${tab}]`).count())continue;
            await pop.locator(`[data-tab=${tab}]`).click();await pop.locator(`#pc-panel-${tab}`).waitFor({state:'visible'});
            if(tab==='record')await pop.locator('.pc-record .pc-plot').waitFor({state:'visible'});
            await p.screenshot({path:`/tmp/fv-player-pop-${sport}-${width}-${tab}.png`});
          }
          await pop.locator('[data-tab=form]').click();
        }
        if(width===1440){
          // Keyboard tabs, and a chart readout that follows the pointer.
          await pop.locator('[data-tab=form]').focus();await p.keyboard.press('ArrowRight');
          assert.equal(await pop.locator('[data-tab=model]').getAttribute('aria-selected'),'true');
          assert(await pop.locator('#pc-panel-model').isVisible());assert.equal(await pop.locator('#pc-panel-form').isVisible(),false);
          assert.match(await pop.locator('#pc-panel-model').textContent(),/How the number is built/);
          await p.keyboard.press('ArrowLeft');
          const out=pop.locator('.pc-trend .pc-readout'),before=await out.textContent();
          await pop.locator('.pc-trend .pc-target').last().hover();
          assert.notEqual(await out.textContent(),before,'hovering a bar shows that game');
          assert.match(await out.textContent(),sport==='mlb'?/Oct 2 · @ TOR · 5 K/:/Oct 2 · vs CHI · 3 SOG · 20:18 TOI · counts 98%/);
        }
        await p.keyboard.press('Escape');await pop.waitFor({state:'hidden'});
        assert.equal(await p.evaluate(()=>document.activeElement?.classList.contains('pc-name')),true,'focus returns to the name');
        await name.click();await pop.waitFor({state:'visible'});await p.mouse.click(2,2);await pop.waitFor({state:'hidden'});
        await name.click();await pop.locator('.pc-close').click();await pop.waitFor({state:'hidden'});
        if(width===1440){
          await name.hover();await pop.waitFor({state:'visible'});assert.equal(await name.getAttribute('aria-expanded'),'true','hover preview');
          await pop.hover();await p.waitForTimeout(400);assert(await pop.isVisible(),'moving into the snapshot keeps it open');
          await p.mouse.move(2,1000);await pop.waitFor({state:'hidden'});
        }
        assert(await p.locator('.prop-card button.track-offer,.prop-card .fv-track-actions button').count()>0,'Tracking control retained');
      }
      if(sport==='nhl'){
        data={...data,rows:[{...row,model_data_checked_at:new Date(Date.now()-37*3600e3).toISOString()}]};
        await p.reload();await p.waitForSelector('.prop-card');assert.equal(await p.locator('.pc-name').count(),0,'Expired NHL model context hidden');
      }
    }
    // NFL: rushing and receiving traces from the saved params, with the forecast's bell curve.
    const fields=['game_id','game','player','bookmaker','book_label','market_std','market_label','name','point','price','mu','model_prob','push_prob','mkt_prob','prob_devig',
      'consensus_prob','consensus_line','book_count','edge_bps','ev_per_100','model_status','last_update','commence_time','kick_et','home_team','away_team','projection_diagnostics','player_position'];
    const back={version:'nfl-projection-trace-1',family:'rush',carries:16.2,yards_per_carry:4.4,recent_mean_weight:.5,sigma:30,home:true,
      current_sample:[{season:2026,week:1,opponent_team:'MIA',carries:15,rushing_yards:61},{season:2026,week:2,opponent_team:'NYJ',carries:18,rushing_yards:92},
        {season:2026,week:3,opponent_team:'NE',carries:12,rushing_yards:40},{season:2026,week:4,opponent_team:'KC',carries:20,rushing_yards:111}],
      mean_stages:{before_adjustments:71.28,after_defense:68.1,after_venue:72.2,final:72.2},
      opponent:{team:'WAS',kind:'rush',rating:1.15,rank:6,of:32,allowed:98.4,league:112.3,games:4,season:2026}};
    const nflRow=o=>fields.map(f=>o[f]??null);
    const nflBase={game_id:'nfl-g',game:'Example Away @ Example Home',bookmaker:'draftkings',book_label:'DraftKings',price:-110,mkt_prob:.524,prob_devig:.5,consensus_prob:.5,consensus_line:70.5,book_count:3,
      edge_bps:-500,ev_per_100:-5,model_status:'Calibration fitted; not prospectively validated',last_update:now,commence_time:future,kick_et:'Sun 1:00 PM ET',home_team:'Example Home',away_team:'Example Away'};
    const payload={fields,dictionary:{},topOnly:false,root:'..',snapshotUpcoming:2,snapshotVerified:true,lastKickoff:future,rows:[
      nflRow({...nflBase,player:'Example Back',player_position:'RB',market_std:'rush_yds',market_label:'Rushing yards',name:'under',point:70.5,mu:72.2,model_prob:.47,push_prob:0,projection_diagnostics:JSON.stringify(back)}),
      nflRow({...nflBase,player:'Example Receiver',market_std:'receptions',market_label:'Receptions',name:'over',point:4.5,mu:4.69,model_prob:.52,push_prob:0,
        projection_diagnostics:JSON.stringify({...back,family:'receive',targets:7.1,catch_rate:.66,yards_per_reception:11.8,sigma:2.1,home:false,opponent:{team:'BUF',kind:'pass',rating:.8,rank:25,of:32},
          current_sample:[{season:2026,week:4,opponent_team:'MIA',targets:6,receptions:4,receiving_yards:52}],mean_stages:{before_adjustments:4.69,after_defense:4.69,after_venue:4.69,final:4.69}})})]};
    const propsHTML=fs.readFileSync(path.join(root,'props/index.html'),'utf8')
      .replace(/(<script type="application\/json" id="props-data">)[\s\S]*?(<\/script>)/,(_,a,b)=>a+JSON.stringify(payload).replace(/</g,'\\u003c')+b);
    await p.route(base+'/props/',route=>route.fulfill({contentType:'text/html',body:propsHTML}));
    for(const width of [390,1440]){
      await p.setViewportSize({width,height:1050});await p.goto(base+'/props/');await p.waitForSelector('.prop-card .pc-name');
      const name=p.locator('.prop-card .pc-name',{hasText:'Example Back'}),pop=p.locator('.pc-pop');
      await name.click();await pop.waitFor({state:'visible'});
      assert.equal(await p.locator('.prop-card h2',{hasText:'Example Back'}).locator('.pc-pos').textContent(),'RB');
      assert.match(await pop.locator('.pc-pop-head span').textContent(),/^RB · Example Away @ Example Home$/,'position in the snapshot header');
      const form=await pop.textContent();
      assert.match(form,/2 of 4 cleared Under 70\.5/);assert.match(form,/Rush yds \/ game76/);assert.doesNotMatch(form,/NaN|undefined|null/);
      await pop.locator('[data-tab=model]').click();await pop.locator('.pc-dist .pc-plot').waitFor({state:'visible'});
      const model=await pop.locator('#pc-panel-model').textContent();
      assert.match(model,/Opposing run defense0\.96×/);assert.match(model,/Home game1\.06×/);assert.match(model,/Rushing yards allowed \/ game98\.4/);
      assert.match(model,/Under 70\.5: 47\.7% on this curve · 47% after calibration/);
      const out=pop.locator('.pc-dist .pc-readout');await pop.locator('.pc-dist .pc-target').nth(5).hover();
      assert.match(await out.textContent(),/71–85 rush yds: 19\.4%/,'hovering a bin shows its chance');
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`NFL pop-up overflow at ${width}`);
      await p.screenshot({path:`/tmp/fv-player-pop-nfl-${width}-model.png`});
      await pop.locator('[data-tab=form]').click();await p.keyboard.press('Escape');await pop.waitFor({state:'hidden'});
      await p.locator('.prop-card .pc-name',{hasText:'Example Receiver'}).click();await pop.waitFor({state:'visible'});
      await pop.locator('[data-tab=model]').click();
      assert.match(await pop.locator('#pc-panel-model').textContent(),/Expected targets7\.1×Catch rate66%=Before matchup4\.7 receptions/);
      assert.equal(await pop.locator('.pc-dist .pc-ticks span').count(),12,'one bin per catch count, 0 to 11+');
      await pop.locator('[data-tab=form]').click();await p.keyboard.press('Escape');await pop.waitFor({state:'hidden'});
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
    console.log('PASS: name pop-up (click, hover, Escape, outside click, close), game logs, inputs and tracking at four widths; stale NHL context hidden; NFL rushing and receiving curves; daily-picks context.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});

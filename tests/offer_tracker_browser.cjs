// Local fixtures only: MLB, NBA and NFL boards save exact offers through the shared dialog without account/database writes.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),os=require('node:os'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require('playwright');
const repo=path.resolve(__dirname,'..'),root=path.join(repo,'docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
const day=value=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(value));

// The NFL game-totals page is generated from CSVs that are not committed; build it from a tiny fixture.
function nflTotalsPage(kickoff,quoted){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'fv-nfl-totals-'));
  fs.writeFileSync(path.join(dir,'consensus.csv'),'game,home_team,away_team,market,consensus_line,num_books\nCAR @ ATL,ATL,CAR,total,43.5,2\nCAR @ ATL,ATL,CAR,spread,-3.0,2\n');
  fs.writeFileSync(path.join(dir,'lines.csv'),'game,commence_time,home_team,away_team,book,total_over_line,total_over_price,total_under_price,spread_home_line,spread_home_price,spread_away_price,totals_last_update,spreads_last_update\n'+
    `CAR @ ATL,${kickoff},ATL,CAR,draftkings,43.5,-110,-105,-3.0,-115,-105,${quoted},${quoted}\n`);
  const out=path.join(dir,'index.html');
  execFileSync('python3',['scripts/nfl_build_totals_page.py','--week','4','--predictions',path.join(dir,'none.csv'),'--consensus',path.join(dir,'consensus.csv'),
    '--edges',path.join(dir,'none.csv'),'--lines',path.join(dir,'lines.csv'),'--output',out],{cwd:repo,stdio:'pipe'});
  return fs.readFileSync(out,'utf8');
}

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`,browser=await chromium.launch({headless:true});
  try{
    const p=await browser.newPage({viewport:{width:390,height:900}}),errors=[];
    p.on('pageerror',error=>errors.push(error.message));
    await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
    await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    await p.addInitScript(()=>{
      window.db={user:{id:'test-owner'},rows:[]};
      window.supabase={createClient:()=>({auth:{getSession:async()=>({data:{session:db.user?{user:db.user}:null}})},from:()=>({
        insert:async row=>{db.rows.push(row);return {error:null};}
      })})};
    });
    const now=new Date(),future=new Date(+now+3*3600e3).toISOString(),quoted=now.toISOString();
    const open=async(nth=0)=>{await p.locator('[data-fv-ticket]:not([disabled])').nth(nth).click();await p.waitForSelector('#fv-bet-tracker[open]');};
    async function save(odds,stake){
      await p.locator('#fv-track-odds').fill(String(odds));await p.locator('#fv-track-stake').fill(String(stake));await p.locator('#fv-track-confirm').check();
      await p.locator('#fv-track-save').click();
      await p.waitForFunction(()=>document.getElementById('fv-track-feedback').textContent.startsWith('Saved to'));
      const row=await p.evaluate(()=>db.rows.at(-1));await p.locator('#fv-track-close').click();return row;
    }
    const noOverflow=async()=>assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);

    // MLB props and Model Picks: exact offer, grader market name and fresh non-push model probability.
    const mlbRow={event_id:'m1',mlb_game_id:1,game:'Philadelphia Phillies @ Atlanta Braves',home_team:'Atlanta Braves',away_team:'Philadelphia Phillies',
      commence_time:future,quoted_at:quoted,phase:'Wild Card',game_type:'F',player:'Bryce Harper',market:'batter_hits',market_label:'Hits',market_family:'props',
      side:'Over',line:.5,book:'fanduel',book_label:'FanDuel',price:-150,book_probability:.6,fair_probability:.58,consensus_probability:.58,paired_books:5,
      other_books:4,other_book_probability:.6,consensus_ev:.01,model_probability:.6,model_push_probability:.1,model_mean:1.1,model_fair_price:-200,
      model_ev_pct:4,model_edge_pp:6,is_model_pick:true,model_status:'Model pick',model_version:'mlb-v1',away_pitcher:null,home_pitcher:null};
    const mlb={status:'ready',last_success_at:quoted,model_checked_at:quoted,history_checked_at:null,events:[],model_summary:{},rows:[mlbRow]};
    await p.route('**/mlb/data/latest.json',route=>route.fulfill({json:mlb}));
    for(const page of ['/mlb/props/','/mlb/picks.html']){
      await p.goto(base+page);await p.waitForSelector('[data-fv-ticket]');await open();
      assert.match(await p.locator('#fv-track-description').textContent(),/^MLB · .*Bryce Harper · Over · 0.5 · Hits · FanDuel$/);
      assert.equal(await p.locator('#fv-track-odds').inputValue(),'-150');
      const t=await save(-145,20);
      assert.deepEqual({league:t.league,game_date:t.game_date,player:t.player,market_type:t.market_type,side:t.side,line:t.line,book:t.book,odds:t.odds,stake:t.stake_dollars},
        {league:'MLB',game_date:day(future),player:'Bryce Harper',market_type:'batter_hits',side:'over',line:.5,book:'fanduel',odds:-145,stake:20});
      assert(Math.abs(t.model_prob-.6/.9)<1e-12,'MLB model probability excludes pushes');
      assert(await p.locator('[data-fv-ticket]').first().isDisabled(),'saved offer is marked Tracked');await noOverflow();
    }
    // Expired MLB model inputs never become a saved probability.
    mlb.model_checked_at=new Date(+now-2*3600e3).toISOString();
    await p.goto(base+'/mlb/totals/');mlb.rows=[{...mlbRow,player:'',market:'h2h',market_label:'Moneyline',market_family:'lines',side:'Atlanta Braves',line:null,price:120}];
    await p.reload();await p.waitForSelector('[data-fv-ticket]');await open();
    let t=await save(120,10);assert.equal(t.model_prob,null);assert.equal(t.line,null);assert.equal(t.market_type,'h2h');

    // NBA game lines: no published model, so no model probability.
    const nbaRow={event_id:'n1',game:'Boston Celtics @ Detroit Pistons',home_team:'Detroit Pistons',away_team:'Boston Celtics',commence_time:future,quoted_at:quoted,
      player:'',market:'spreads',market_label:'Spread',side:'Boston Celtics',line:-2.5,book:'draftkings',book_label:'DraftKings',price:-110,
      book_probability:.52,fair_probability:.5,consensus_probability:.5,paired_books:4,other_books:3,other_book_probability:.5,consensus_ev:.01,model_probability:.8,baseline_mean:null};
    await p.route('**/nba/data/latest.json',route=>route.fulfill({json:{status:'ready',last_success_at:quoted,events:[],rows:[nbaRow]}}));
    await p.goto(base+'/nba/totals/');await p.waitForSelector('[data-fv-ticket]');await open();
    t=await save(-110,15);
    assert.deepEqual([t.league,t.market_type,t.side,t.line,t.book,t.model_prob,t.edge_bps],['NBA','spreads','Boston Celtics',-2.5,'draftkings',null,null]);

    // NFL props: the previous prompt/alert flow is replaced by the same dialog; anytime TD has no line.
    const fields=['game_id','game','player','bookmaker','book_label','market_std','market_label','name','point','price','mu','model_prob','push_prob','mkt_prob',
      'prob_devig','consensus_prob','consensus_line','book_count','edge_bps','ev_per_100','model_status','last_update','commence_time','kick_et','home_team','away_team'];
    const nfl=[['g1','Carolina Panthers @ Atlanta Falcons','Bijan Robinson','draftkings','DraftKings','rush_yds','Rush yards','over',74.5,-115,80,.56,0,.53,.5,.5,74.5,6,300,4.8,'Calibration fitted (isotonic)',quoted,future,'Sun 1:00 PM','Atlanta Falcons','Carolina Panthers'],
      ['g1','Carolina Panthers @ Atlanta Falcons','Bijan Robinson','fanduel','FanDuel','anytime_td','Anytime TD','yes',null,-120,null,null,null,.55,.5,.5,null,5,null,null,'Historical rate only',quoted,future,'Sun 1:00 PM','Atlanta Falcons','Carolina Panthers'],
      ['g0','Old @ Game','Past Player','draftkings','DraftKings','receptions','Receptions','over',4.5,-110,4,.5,0,.52,.5,.5,4.5,5,0,0,'Calibration fitted',quoted,new Date(+now-3600e3).toISOString(),'Earlier','Game','Old']];
    const props=fs.readFileSync(path.join(root,'props/index.html'),'utf8').replace(/(<script type="application\/json" id="props-data">)[\s\S]*?(<\/script>)/,
      (_,a,b)=>a+JSON.stringify({fields,dictionary:{},rows:nfl,topOnly:false,root:'..',snapshotUpcoming:2,snapshotVerified:true,lastKickoff:future})+b);
    await p.route('**/props/',route=>route.fulfill({contentType:'text/html',body:props}));
    await p.goto(base+'/props/');await p.waitForSelector('[data-fv-ticket]');
    assert.equal(await p.locator('button[data-track]').count(),0,'legacy tracking buttons removed');
    await p.locator('#q').fill('Bijan');await p.locator('#sort').selectOption('player');
    await open(0);t=await save(-112,10);
    const first=t;await open(0);t=await save(-118,10);
    const [yards,td]=[first,t].sort((a,b)=>a.market_type.localeCompare(b.market_type)).reverse();
    assert.deepEqual([yards.league,yards.market_type,yards.side,yards.line,yards.model_prob],['NFL','rush_yds','over',74.5,.56]);
    assert.deepEqual([td.market_type,td.side,td.line,td.model_prob],['anytime_td','yes',null,null]);
    await p.locator('#history').check();await p.locator('#q').fill('Past');
    assert.equal(await p.locator('[data-fv-ticket]').textContent(),'Game started');assert(await p.locator('[data-fv-ticket]').isDisabled());

    // NFL game totals and spreads: every sportsbook row offers each exact quote, saved as game totals (not team totals).
    await p.route('**/nfl/totals/',route=>route.fulfill({contentType:'text/html',body:nflTotalsPage(future,quoted)}));
    await p.goto(base+'/nfl/totals/');await p.locator('details.book-lines summary').click();
    assert.deepEqual(await p.locator('.track-cell [data-fv-ticket]').allTextContents(),['Over 43.5','Under 43.5','ATL -3','CAR +3']);
    await open(1);t=await save(-105,25);
    assert.deepEqual([t.league,t.market_type,t.side,t.line,t.book,t.team_home,t.team_away,t.game_date],['NFL','totals','under',43.5,'draftkings','ATL','CAR',day(future)]);
    await open(2);t=await save(-115,25);assert.deepEqual([t.market_type,t.side,t.line],['spreads','CAR',3]);
    for(const width of [390,1440]){await p.setViewportSize({width,height:900});await noOverflow();}
    assert.deepEqual(errors,[]);console.log('PASS: MLB, NBA, NFL props and NFL game lines save exact offers through the shared Track bet dialog.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

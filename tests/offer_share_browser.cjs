// Local fixtures only: every sport's Player Props cards offer Share, which draws the card's exact offer as an image.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..','docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
const day=value=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(value));

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`,browser=await chromium.launch({headless:true});
  try{
    const p=await browser.newPage({viewport:{width:390,height:900}}),errors=[];
    p.on('pageerror',error=>errors.push(error.message));
    await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
    await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
    const now=new Date(),future=new Date(+now+3*3600e3).toISOString(),quoted=now.toISOString(),past=new Date(+now-1800e3).toISOString();
    const noOverflow=async()=>assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    // Opens the nth card's Share and returns what the dialog shows.
    async function share(card){
      const actions=await card.locator('.actions button').allTextContents();
      assert.deepEqual(actions.slice(-2),['Track bet','Share'],'Share sits beside Track bet');
      await card.locator('.fv-share-button').click();await p.waitForSelector('#fv-share[open]');
      await p.waitForFunction(()=>document.getElementById('fv-share-image').complete&&document.getElementById('fv-share-image').naturalWidth>0);
      const shown=await p.evaluate(()=>{const img=document.getElementById('fv-share-image'),save=document.getElementById('fv-share-save');
        return {alt:img.alt,width:img.naturalWidth,height:img.naturalHeight,src:img.src,href:save.href,file:save.download};});
      assert.match(shown.src,/^blob:/);assert.equal(shown.href,shown.src);assert.deepEqual([shown.width,shown.height],[1200,624]);
      await noOverflow();
      await p.locator('#fv-share-close').click();assert.equal(await p.locator('#fv-share[open]').count(),0);
      return shown;
    }
    const eastern=value=>new Date(value).toLocaleString('en-US',{timeZone:'America/New_York',hour:'numeric',minute:'2-digit',timeZoneName:'short'});

    // NBA Player Props.
    const nbaRow={event_id:'n1',game:'Charlotte Hornets @ Washington Wizards',home_team:'Washington Wizards',away_team:'Charlotte Hornets',commence_time:future,quoted_at:quoted,
      player:'Trae Young',market:'player_assists',market_label:'Assists',side:'Over',line:9.5,book:'draftkings',book_label:'DraftKings',price:-119,
      book_probability:.54,fair_probability:.52,consensus_probability:.52,paired_books:4,other_books:3,other_book_probability:.52,consensus_ev:.01,baseline_mean:null};
    await p.route('**/nba/data/latest.json',route=>route.fulfill({json:{status:'ready',last_success_at:quoted,events:[],rows:[nbaRow]}}));
    await p.goto(base+'/nba/props/');await p.waitForSelector('.fv-share-button');
    let shown=await share(p.locator('.prop-card').first());
    assert.match(shown.alt,new RegExp(`^NBA player prop: Trae Young, Over 9\\.5 Assists at -119 \\(DraftKings\\)\\. Charlotte Hornets @ Washington Wizards, .*${eastern(future)}\\. Quoted .*${eastern(quoted)}; confirm the current price\\.$`));
    assert.equal(shown.file,`fourth-and-value-nba-prop-trae-young-${day(future)}.png`);
    // Game Lines keep Track bet only.
    await p.goto(base+'/nba/totals/');await p.waitForSelector('#result-count');
    assert.equal(await p.locator('.fv-share-button').count(),0);

    // NHL Player Props, including a started game's last pregame quote.
    const nhlRow={event_id:'h1',game:'New Jersey Devils @ Vegas Golden Knights',home_team:'Vegas Golden Knights',away_team:'New Jersey Devils',commence_time:future,quoted_at:quoted,
      player:'Jack Eichel',market:'player_assists',market_label:'Assists',side:'Over',line:.5,book:'draftkings',book_label:'DraftKings',price:-150,
      book_probability:.6,fair_probability:.58,consensus_probability:.58,paired_books:5,other_books:4,other_book_probability:.58,conditional_price_advantage:0};
    await p.route('**/nhl/data/latest.json',route=>route.fulfill({json:{status:'ready',last_success_at:quoted,events:[],
      rows:[nhlRow,{...nhlRow,event_id:'h0',game:'Boston Bruins @ Ottawa Senators',player:'David Pastrnak',commence_time:past}]}}));
    await p.goto(base+'/nhl/props/');await p.waitForSelector('.fv-share-button');
    shown=await share(p.locator('.prop-card').first());
    assert.match(shown.alt,/^NHL player prop: Jack Eichel, Over 0\.5 Assists at -150 \(DraftKings\)\./);
    assert.doesNotMatch(shown.alt,/Game started/);
    await p.locator('#history').check();
    shown=await share(p.locator('.prop-card',{hasText:'David Pastrnak'}));
    assert.match(shown.alt,/^NHL player prop: David Pastrnak, .* Game started\. Quoted /);

    // MLB Player Props.
    const mlbRow={event_id:'m1',mlb_game_id:1,game:'Philadelphia Phillies @ Atlanta Braves',home_team:'Atlanta Braves',away_team:'Philadelphia Phillies',
      commence_time:future,quoted_at:quoted,phase:'Wild Card',game_type:'F',player:'Bryce Harper',market:'batter_hits',market_label:'Hits',market_family:'props',
      side:'Over',line:.5,book:'fanduel',book_label:'FanDuel',price:-150,book_probability:.6,fair_probability:.58,consensus_probability:.58,paired_books:5,
      other_books:4,other_book_probability:.6,consensus_ev:.01,model_status:'Research forecast',away_pitcher:null,home_pitcher:null};
    await p.route('**/mlb/data/latest.json',route=>route.fulfill({json:{status:'ready',last_success_at:quoted,model_checked_at:quoted,history_checked_at:null,events:[],model_summary:{},rows:[mlbRow]}}));
    await p.goto(base+'/mlb/props/');await p.waitForSelector('.fv-share-button');
    shown=await share(p.locator('.prop-card').first());
    assert.match(shown.alt,/^MLB player prop: Bryce Harper, Over 0\.5 Hits at -150 \(FanDuel\)\./);

    // NFL Player Props: lineless yes/no props keep their exact selection.
    const fields=['game_id','game','player','bookmaker','book_label','market_std','market_label','name','point','price','mu','model_prob','push_prob','mkt_prob',
      'prob_devig','consensus_prob','consensus_line','book_count','edge_bps','ev_per_100','model_status','last_update','commence_time','kick_et','home_team','away_team'];
    const nfl=[['g1','Carolina Panthers @ Atlanta Falcons','Bijan Robinson','draftkings','DraftKings','rush_yds','Rush yards','over',74.5,-115,80,.56,0,.53,.5,.5,74.5,6,300,4.8,'Calibration fitted (isotonic)',quoted,future,'Sun 1:00 PM','Atlanta Falcons','Carolina Panthers'],
      ['g1','Carolina Panthers @ Atlanta Falcons','Bijan Robinson','fanduel','FanDuel','anytime_td','Anytime TD','yes',null,120,null,null,null,.45,.5,.5,null,5,null,null,'Historical rate only',quoted,future,'Sun 1:00 PM','Atlanta Falcons','Carolina Panthers']];
    const props=fs.readFileSync(path.join(root,'props/index.html'),'utf8').replace(/(<script type="application\/json" id="props-data">)[\s\S]*?(<\/script>)/,
      (_,a,b)=>a+JSON.stringify({fields,dictionary:{},rows:nfl,topOnly:false,root:'..',snapshotUpcoming:2,snapshotVerified:true,lastKickoff:future})+b);
    await p.route('**/props/',route=>route.fulfill({contentType:'text/html',body:props}));
    await p.goto(base+'/props/');await p.waitForSelector('.fv-share-button');
    await p.locator('#sort').selectOption('player');
    shown=await share(p.locator('.prop-card',{hasText:'Rush yards'}));
    assert.match(shown.alt,/^NFL player prop: Bijan Robinson, Over 74\.5 Rush yards at -115 \(DraftKings\)\./);
    shown=await share(p.locator('.prop-card',{hasText:'Anytime TD'}));
    assert.match(shown.alt,/^NFL player prop: Bijan Robinson, Anytime TD: Yes at \+120 \(FanDuel\)\./);
    // The board's one-minute refresh rebuilds the cards with their Share buttons.
    await p.evaluate(()=>document.getElementById('sort').dispatchEvent(new Event('change')));
    assert.equal(await p.locator('.fv-share-button').count(),2);

    for(const width of [390,1440]){await p.setViewportSize({width,height:900});await noOverflow();}
    assert.deepEqual(errors,[]);console.log('PASS: NBA, NHL, MLB and NFL Player Props cards share the exact offer as an image.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

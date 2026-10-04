// Live Odds: quote parsing, cross-book comparison, scoreboard matching and the
// editor-only live-odds relay. Fixtures mirror Odds API shapes; nothing leaves the process.
const assert=require('node:assert/strict');
const O=require('../docs/live/live-odds.js');

const NOW=Date.parse('2026-10-03T23:40:00Z');
const ago=s=>new Date(NOW-s*1000).toISOString();
const HOME='Buffalo Sabres',AWAY='Chicago Blackhawks';
// markets: {key:[[name,price,point,description]]}; one last_update per book here.
const book=(key,title,age,markets)=>({key,title,last_update:ago(age),markets:Object.entries(markets).map(([k,outs])=>(
  {key:k,last_update:ago(age),outcomes:outs.map(([name,price,point,description])=>({name,price,...(point==null?{}:{point}),...(description?{description}:{})}))}))});
const sog=(over,under,line=2.5,player='Tage Thompson')=>({player_shots_on_goal:[['Over',over,line,player],['Under',under,line,player]]});
const event=(bookmakers,commence='2026-10-03T23:10:00Z')=>({id:'a'.repeat(32),sport_key:'icehockey_nhl',commence_time:commence,home_team:HOME,away_team:AWAY,bookmakers});
const near=(a,b,msg)=>assert(Math.abs(a-b)<1e-9,`${msg}: ${a} vs ${b}`);
const line=(b,id)=>b.lines.find(l=>l.id===id);

// ---- Prices -----------------------------------------------------------------
near(O.implied(-110),110/210,'favorite');near(O.implied(150),0.4,'underdog');
assert(Number.isNaN(O.implied(50)),'American odds below 100 in size are invalid');
assert.equal(O.american(0.5),100);assert.equal(O.american(0.6),-150);assert.equal(O.american(0.4),150);
assert.equal(O.american(null),null);assert.equal(O.american(1),null);
assert.equal(O.signed(-115),'−115');assert.equal(O.signed(120),'+120');assert.equal(O.signed(null),'—');

// ---- Parsing keeps exact offers and drops unusable ones ------------------------
{
  const rows=O.quotes(event([book('draftkings','DraftKings',10,{
    h2h:[[HOME,-150],[AWAY,130]],
    totals:[['Over',-110,6.5],['Under',-110,6.5],['Over',-110,null]],
    player_shots_on_goal:[['Over',120,2.5,'Tage Thompson'],['Under',-150,2.5,'Tage Thompson'],['Over',110,1.5,''],['Yes',110,0.5,'Tage Thompson']],
    player_points:[['Over',50,0.5,'Rasmus Dahlin']],
    alternate_totals:[['Over',200,7.5]],
    spreads:[['Nobody',-110,1.5]],
  })]));
  assert.deepEqual(rows.map(r=>[r.market,r.side,r.line,r.price]),[
    ['h2h',HOME,null,-150],['h2h',AWAY,null,130],['totals','Over',6.5,-110],['totals','Under',6.5,-110],
    ['player_shots_on_goal','Over',2.5,120],['player_shots_on_goal','Under',2.5,-150]]);
  assert.equal(rows[4].player,'Tage Thompson');assert.equal(rows[0].book_label,'DraftKings');
  assert.deepEqual(O.quotes(null),[]);assert.deepEqual(O.quotes({bookmakers:'x'}),[]);
}

// ---- Cross-book comparison -------------------------------------------------------
{
  // Three books at -110/-110 (fair 50%) and one outlier at +110 on the Over.
  const even=k=>book(k,k.toUpperCase(),20,sog(-110,-110));
  const b=O.board(event([even('fanduel'),even('betmgm'),even('williamhill_us'),book('draftkings','DraftKings',15,sog(110,-140))]),NOW);
  assert.equal(b.mode,'live');assert.deepEqual(b.books,['BETMGM','DraftKings','FANDUEL','WILLIAMHILL_US']);
  const over=line(b,'player_shots_on_goal|Tage Thompson|2.5|Over');
  assert.equal(over.best.book,'draftkings');assert.equal(over.best.price,110);
  near(over.best.other_probability,0.5,'median of the other books, never its own price');
  near(over.best.advantage,100*(0.5/(100/210)-1),'+5% against the other books');
  assert.equal(over.best.other_books,3);assert.equal(over.flagged,true);
  assert.equal(b.lines[0].id,over.id,'flagged lines come first');
  near(over.fair_probability,0.5,'fair = median of every fresh paired book');assert.equal(over.fair_odds,100);assert.equal(over.fair_books,4);
  assert.equal(over.label,'Over 2.5');assert.equal(over.push_possible,false);
  assert.deepEqual(over.quotes.map(q=>q.book),['draftkings','betmgm','fanduel','williamhill_us'],'best price first');
  const dk=over.quotes[0];near(dk.fair_probability,(100/210)/(100/210+140/240),'own pair de-vigged multiplicatively');
  const under=line(b,'player_shots_on_goal|Tage Thompson|2.5|Under');
  assert.equal(under.flagged,false);assert(under.best.advantage<0);

  // Two other books are too few to compare.
  const few=O.board(event([even('fanduel'),even('betmgm'),book('draftkings','DraftKings',15,sog(110,-140))]),NOW);
  const o2=line(few,'player_shots_on_goal|Tage Thompson|2.5|Over');
  assert.equal(o2.best.other_books,2);assert.equal(o2.best.advantage,null);assert.equal(o2.flagged,false);

  // A side without its pair keeps a missing fair probability but can still be judged.
  const lone=O.board(event([even('fanduel'),even('betmgm'),even('williamhill_us'),
    book('draftkings','DraftKings',15,{player_shots_on_goal:[['Over',110,2.5,'Tage Thompson']]})]),NOW);
  const o3=line(lone,'player_shots_on_goal|Tage Thompson|2.5|Over');
  assert.equal(o3.quotes.find(q=>q.book==='draftkings').fair_probability,null);assert.equal(o3.flagged,true);
}
{
  // During play a quote far behind the newest is shown, never compared or used as a reference.
  const b=O.board(event([book('fanduel','FanDuel',10,sog(-110,-110)),book('betmgm','BetMGM',10,sog(-110,-110)),
    book('williamhill_us','Caesars',10,sog(-110,-110)),book('draftkings','DraftKings',10+360,sog(150,-190))]),NOW);
  const over=line(b,'player_shots_on_goal|Tage Thompson|2.5|Over');
  const stale=over.quotes.find(q=>q.book==='draftkings');
  assert.equal(stale.stale,true);assert.equal(stale.advantage,null);
  assert.notEqual(over.best.book,'draftkings','the best price comes from fresh quotes');
  assert.equal(over.flagged,false);assert.equal(over.fair_books,3);
  // Every quote stale: still listed, never flagged.
  const old=O.board(event([book('fanduel','FanDuel',10,{totals:[['Over',-110,6.5],['Under',-110,6.5]]}),
    book('draftkings','DraftKings',700,sog(150,-190))]),NOW);
  const o=line(old,'player_shots_on_goal|Tage Thompson|2.5|Over');
  assert.equal(o.best.book,'draftkings');assert.equal(o.best.stale,true);assert.equal(o.fair_probability,null);assert.equal(o.flagged,false);
}
{
  // During play references must be within two minutes; before the game, fifteen.
  const books=[book('fanduel','FanDuel',10,sog(-110,-110)),book('betmgm','BetMGM',10,sog(-110,-110)),
    book('williamhill_us','Caesars',10+200,sog(-110,-110)),book('draftkings','DraftKings',10,sog(110,-140))];
  const live=line(O.board(event(books),NOW),'player_shots_on_goal|Tage Thompson|2.5|Over');
  assert.equal(live.best.other_books,2);assert.equal(live.flagged,false);
  const pre=O.board(event(books,'2026-10-04T00:10:00Z'),NOW);
  assert.equal(pre.mode,'pre');
  assert.equal(line(pre,'player_shots_on_goal|Tage Thompson|2.5|Over').best.other_books,3);
}
{
  // Puck line: home -1.5 pairs with away +1.5, not with away -1.5.
  const b=O.board(event([book('draftkings','DraftKings',10,{spreads:[[HOME,150,-1.5],[AWAY,-180,1.5]]}),
    book('fanduel','FanDuel',10,{spreads:[[HOME,155,-1.5],[AWAY,-190,-1.5]]})]),NOW);
  const dkHome=line(b,'spreads||-1.5|'+HOME).quotes.find(q=>q.book==='draftkings');
  near(dkHome.fair_probability,0.4/(0.4+180/280),'matched pair de-vigged');
  assert.equal(line(b,'spreads||-1.5|'+HOME).quotes.find(q=>q.book==='fanduel').fair_probability,null,'mismatched lines are no pair');
  assert.equal(line(b,'spreads||-1.5|'+HOME).label,HOME+' −1.5');assert.equal(line(b,'spreads||-1.5|'+AWAY).label,AWAY+' +1.5');
  // Whole-number totals can push; moneylines cannot.
  const t=O.board(event([book('draftkings','DraftKings',10,{totals:[['Over',-105,6],['Under',-115,6]],h2h:[[HOME,-150],[AWAY,130]]})]),NOW);
  assert.equal(line(t,'totals||6|Over').push_possible,true);assert.equal(line(t,'h2h|||'+HOME).push_possible,false);
}
{
  // Contradictory prices from one book at the same moment fail closed; a newer quote replaces an older one.
  const rows=[...O.quotes(event([book('draftkings','DraftKings',10,sog(110,-140))])),
    ...O.quotes(event([book('draftkings','DraftKings',10,sog(120,-140))]))];
  const {rows:kept}=O.compare(rows,'live');
  assert.deepEqual(kept.map(r=>r.side),['Under']);
  const newer=O.compare([...O.quotes(event([book('draftkings','DraftKings',60,sog(110,-140))])),
    ...O.quotes(event([book('draftkings','DraftKings',10,sog(120,-150))]))],'live').rows;
  assert.deepEqual(newer.map(r=>r.price),[120,-150]);
}

// ---- In-game milestones: anytime scorer and alternate lines -----------------------------
{
  const scorer=outs=>({player_goal_scorer_anytime:outs.map(([name,price])=>[name,price,null,'Tage Thompson'])});
  const ladder=(market,rungs)=>({[market]:rungs.map(([line,price,side='Over'])=>[side,price,line,'Tage Thompson'])});
  // Parsing: the anytime scorer has no line and only Yes/No; alternates keep their exact lines.
  const rows=O.quotes(event([book('draftkings','DraftKings',10,{...scorer([['Yes',200],['No',-260],['Over',150]]),
    ...ladder('player_shots_on_goal_alternate',[[2.5,-130],[3.5,160],[null,300]])})]));
  assert.deepEqual(rows.map(r=>[r.market,r.side,r.line,r.price]),[['player_goal_scorer_anytime','Yes',null,200],
    ['player_goal_scorer_anytime','No',null,-260],['player_shots_on_goal_alternate','Over',2.5,-130],['player_shots_on_goal_alternate','Over',3.5,160]]);
  // A book offering both sides of "to score" is de-vigged like any pair.
  const paired=O.board(event([book('draftkings','DraftKings',10,scorer([['Yes',200],['No',-260]]))]),NOW);
  const yes=line(paired,'player_goal_scorer_anytime|Tage Thompson||Yes');
  near(yes.quotes[0].fair_probability,(1/3)/(1/3+260/360),'Yes/No pair');
  assert.deepEqual([yes.label,line(paired,'player_goal_scorer_anytime|Tage Thompson||No').label,yes.push_possible],['To score','Not to score',false]);
  // One-way bets: no margin to remove, so only a price gap against the other books' median price.
  const oneWay=k=>book(k,k.toUpperCase(),20,{...scorer([['Yes',150]]),...ladder('player_shots_on_goal_alternate',[[3.5,180]])});
  const b=O.board(event([oneWay('fanduel'),oneWay('betmgm'),oneWay('williamhill_us'),
    book('draftkings','DraftKings',15,{...scorer([['Yes',200]]),...ladder('player_shots_on_goal_alternate',[[3.5,190]])})]),NOW);
  const top=line(b,'player_goal_scorer_anytime|Tage Thompson||Yes');
  assert.deepEqual([top.best.book,top.best.advantage,top.flagged,top.gap_flagged],['draftkings',null,false,true]);
  near(top.best.price_gap,100*(.4/(1/3)-1),'+20% against the median of three prices');
  assert.equal(top.best.price_books,3);near(top.best.other_price_probability,.4,'median price, margins in');
  assert.equal(b.lines.findIndex(l=>l.id===top.id),0,'gap-flagged lines lead when no line beats the fair price');
  const rung=line(b,'player_shots_on_goal_alternate|Tage Thompson|3.5|Over');
  assert.equal(rung.label,'Over 3.5 (4+)');
  assert(rung.best.price_gap<O.GAP_FLAG&&!rung.gap_flagged,'a small gap is shown, not listed');
  const two=O.board(event([oneWay('fanduel'),oneWay('betmgm'),book('draftkings','DraftKings',15,scorer([['Yes',200]]))]),NOW);
  assert.equal(line(two,'player_goal_scorer_anytime|Tage Thompson||Yes').best.price_gap,null,'two other books are too few');
  const old=O.board(event([oneWay('fanduel'),oneWay('betmgm'),book('williamhill_us','Caesars',20+400,scorer([['Yes',150]])),
    book('draftkings','DraftKings',15,scorer([['Yes',200]]))]),NOW);
  assert.equal(line(old,'player_goal_scorer_anytime|Tage Thompson||Yes').best.price_gap,null,'an old quote is no reference');
  // Tickets: the grader's base contracts.
  const T=require('../docs/assets/offer-tracker.js');
  const game={id:'a'.repeat(32),commence_time:'2026-10-03T23:10:00Z',home_team:HOME,away_team:AWAY};
  const t1=T.ticketData(O.ticket(game,top.best),200,5,NOW);
  assert.deepEqual([t1.market_type,t1.side,t1.line,t1.player],['goals','Yes',null,'Tage Thompson'],'anytime scorer = goals "Yes", no line');
  const t2=T.ticketData(O.ticket(game,rung.best),190,5,NOW);
  assert.deepEqual([t2.market_type,t2.side,t2.line],['sog','over',3.5]);
  const pts=O.board(event([book('draftkings','DraftKings',10,ladder('player_points_alternate',[[1.5,240]]))]),NOW);
  assert.equal(T.ticketData(O.ticket(game,pts.lines[0].best),240,5,NOW).market_type,'points');
}

// ---- Bet Tracker tickets: the shared dialog's ledger fields ----------------------------
{
  const T=require('../docs/assets/offer-tracker.js');
  const game={id:'a'.repeat(32),commence_time:'2026-10-04T02:10:00Z',home_team:HOME,away_team:AWAY};
  const b=O.board({...event([book('fanduel','FanDuel',20,{...sog(-110,-110),totals:[['Over',-105,6],['Under',-115,6]],
    h2h:[[HOME,-150],[AWAY,130]],spreads:[[HOME,160,-1.5],[AWAY,-190,1.5]]})]),commence_time:game.commence_time},NOW);
  const q=id=>line(b,id).best;
  const ticket=(id,price,stake)=>T.ticketData(O.ticket(game,q(id)),price,stake,NOW);
  assert.deepEqual(ticket('player_shots_on_goal|Tage Thompson|2.5|Over',-105,25),{league:'NHL',game_date:'2026-10-03',
    team_home:HOME,team_away:AWAY,player:'Tage Thompson',market_type:'sog',side:'over',line:2.5,book:'fanduel',odds:-105,
    stake_dollars:25,model_prob:null,edge_bps:null},'the price you received and no invented model probability');
  const total=ticket('totals||6|Under',-115,10);
  assert.deepEqual([total.market_type,total.side,total.line,total.player],['team_total','under',6,null],'the NHL ledger names the game total team_total');
  const ml=ticket('h2h|||'+AWAY,130,10);assert.deepEqual([ml.market_type,ml.side,ml.line],['h2h',AWAY,null]);
  const pl=ticket('spreads||-1.5|'+AWAY,-190,10);assert.deepEqual([pl.market_type,pl.side,pl.line],['spreads',AWAY,1.5]);
  const row=O.ticket(game,q('player_shots_on_goal|Tage Thompson|2.5|Over'));
  assert.equal(row.quoted_at,new Date(NOW-20e3).toISOString());assert.match(row.settlement_scope,/whole game, including play before you bet/);
  assert.equal(T.identity(row),T.identity(O.ticket(game,q('player_shots_on_goal|Tage Thompson|2.5|Over'))),'one quote keeps one ticket reference');
}

// ---- Scoreboard matching -------------------------------------------------------------
{
  const g=(home,away,id)=>({id,home:{name:home,short:home.split(' ').slice(-1)[0]},away:{name:away,short:away.split(' ').slice(-1)[0]}});
  const games=[{id:'1',home:{name:'Canadiens',short:'Canadiens'},away:{name:'Blues',short:'Blues'}},
    g('Montréal Canadiens','St. Louis Blues','2'),{id:'3',home:{name:'Sabres',short:'Sabres'},away:{name:'Blackhawks',short:'Blackhawks'}}];
  assert.equal(O.findGame(games,{home_team:'Montreal Canadiens',away_team:'St Louis Blues'}).id,'1');
  assert.equal(O.findGame(games.slice(1),{home_team:'Montreal Canadiens',away_team:'St Louis Blues'}).id,'2');
  assert.equal(O.findGame(games,{home_team:HOME,away_team:AWAY}).id,'3');
  assert.equal(O.findGame(games,{home_team:AWAY,away_team:HOME}),null,'home and away must both match');
  assert.equal(O.findGame([{id:'4',home:{name:'Rangers',short:'Rangers'},away:{name:'Islanders',short:'Islanders'}}],
    {home_team:'New York Islanders',away_team:'New York Rangers'}),null);
  assert.equal(O.easternDate('2026-10-04T02:00:00Z'),'2026-10-03','a 10 pm ET start is the previous Eastern date');
  assert.equal(O.easternDate('2026-10-03T23:00:00Z'),'2026-10-03');
}

// ---- The editor-only relay -----------------------------------------------------------
(async()=>{
  const {createHandler,MARKETS,LIVE_MARKETS}=await import('../supabase/functions/live-odds/handler.mjs');
  const KEY='secret-odds-key',ID='b'.repeat(32);
  const ENV={SUPABASE_URL:'https://db.example',SUPABASE_ANON_KEY:'anon',ODDS_API_KEY:KEY};
  function setup({env=ENV,user={app_metadata:{fv_editor:true}},odds,vault=null}={}){
    let clock=NOW,left=17000;const calls=[],vaultReads=[];
    const fetchImpl=async(url,init)=>{
      if(url.startsWith('https://db.example/auth/v1/user')){
        assert.equal(init.headers.Authorization,'Bearer user-token');assert.equal(init.headers.apikey,'anon');
        return user?new Response(JSON.stringify(user),{status:200}):new Response('{}',{status:401});
      }
      if(url==='https://db.example/rest/v1/rpc/live_odds_key'){
        vaultReads.push(init);
        return new Response(JSON.stringify(typeof vault==='function'?vault():vault),{status:200});
      }
      calls.push(url);
      if(odds)return odds(url);
      const u=new URL(url);
      if(u.pathname.endsWith('/events')){
        return new Response(JSON.stringify([{id:ID,commence_time:'2026-10-03T23:10:00Z',home_team:HOME,away_team:AWAY,sport_key:'icehockey_nhl'},{id:'not-an-id'}]),
          {status:200,headers:{'x-requests-remaining':String(left),'x-requests-last':'0'}});
      }
      left-=7;
      return new Response(JSON.stringify({...event([book('draftkings','DraftKings',10,sog(110,-140))]),id:ID}),
        {status:200,headers:{'x-requests-remaining':String(left),'x-requests-last':'7'}});
    };
    const handle=createHandler({fetchImpl,now:()=>clock,env:n=>env[n]});
    const req=(body,{origin='https://fourthandvalue.com',method='POST',auth='Bearer user-token'}={})=>handle(new Request('https://edge/',
      {method,headers:{origin,...(auth?{authorization:auth}:{})},body:method==='POST'?(typeof body==='string'?body:JSON.stringify(body)):undefined}));
    return {req,calls,vaultReads,tick:ms=>{clock+=ms;},setLeft:n=>{left=n;}};
  }

  {
    const {req,calls}=setup();
    assert.equal((await req({},{origin:'https://evil.example'})).status,403);
    assert.equal((await req(null,{method:'OPTIONS'})).status,204);
    assert.equal((await req(null,{method:'GET'})).status,405);
    assert.equal((await req({},{auth:''})).status,401,'no session, no request');
    assert.equal((await req({event:'../../sports'})).status,400);
    assert.equal((await req('[1]')).status,400);assert.equal((await req('{bad')).status,400);
    assert.equal((await req({event:ID,pad:'x'.repeat(300)})).status,400,'request too large');
    assert.equal(calls.length,0,'nothing reaches the odds service before validation and sign-in');
  }
  {
    assert.equal((await setup({user:null}).req({})).status,401,'expired or invalid session');
    const reader=setup({user:{app_metadata:{}}});
    const r=await reader.req({event:ID});assert.equal(r.status,403);assert.equal(reader.calls.length,0,'readers cannot spend credits');
    assert.equal((await setup({env:{...ENV,SUPABASE_URL:''}}).req({})).status,503);
    const nokey=await setup({env:{...ENV,ODDS_API_KEY:''}}).req({event:ID});
    assert.equal(nokey.status,503);assert.match((await nokey.json()).error,/no Odds API key yet.*Live Odds key workflow/);
  }
  {
    // Game list: the free events endpoint, a 6-hour-back/24-hour-ahead window, ids validated.
    const {req,calls}=setup();
    const r=await req({});assert.equal(r.status,200);
    const body=await r.json();
    assert.deepEqual(body.games,[{id:ID,commence_time:'2026-10-03T23:10:00Z',home_team:HOME,away_team:AWAY}]);
    assert.equal(body.remaining,17000);
    const u=new URL(calls[0]);
    assert.equal(u.origin+u.pathname,'https://api.the-odds-api.com/v4/sports/icehockey_nhl/events');
    assert.equal(u.searchParams.get('commenceTimeFrom'),'2026-10-03T17:40:00Z');
    assert.equal(u.searchParams.get('commenceTimeTo'),'2026-10-04T23:40:00Z');
    assert.equal(u.searchParams.get('apiKey'),KEY);
    assert(!JSON.stringify(body).includes(KEY),'the key never reaches the browser');
    await req({});assert.equal(calls.length,1,'game list cached');
  }
  {
    // One game's prices: one paid call, then free reuse for 60 seconds.
    const {req,calls,tick}=setup();
    const [a,b]=await Promise.all([req({event:ID}),req({event:ID})]);
    const first=await a.json(),second=await b.json();
    assert.equal(calls.length,1,'simultaneous presses share one request');
    const u=new URL(calls[0]);
    assert.equal(u.pathname,`/v4/sports/icehockey_nhl/events/${ID}/odds`);
    assert.deepEqual([u.searchParams.get('regions'),u.searchParams.get('markets'),u.searchParams.get('oddsFormat')],['us',MARKETS.join(','),'american']);
    assert.equal(MARKETS.length,7);
    assert.deepEqual([first.reused,first.cost,first.remaining],[false,7,16993]);
    assert.deepEqual([second.reused,second.cost],[true,0]);
    assert.equal(first.event.id,ID);assert.equal(first.fetched_at,'2026-10-03T23:40:00Z');
    assert(!JSON.stringify(first).includes(KEY));
    tick(59e3);const again=await (await req({event:ID})).json();
    assert.deepEqual([again.reused,again.cost,again.fetched_at],[true,0,'2026-10-03T23:40:00Z']);assert.equal(calls.length,1);
    tick(2e3);const fresh=await (await req({event:ID})).json();
    assert.deepEqual([fresh.reused,fresh.cost],[false,7]);assert.equal(calls.length,2);
  }
  {
    // During a game the milestones are added, with their own reuse window.
    const {req,calls}=setup();
    assert.equal((await req({event:ID,live:'yes'})).status,400);
    const live=await (await req({event:ID,live:true})).json();
    assert.equal(new URL(calls[0]).searchParams.get('markets'),[...MARKETS,...LIVE_MARKETS].join(','));
    assert.deepEqual(LIVE_MARKETS,['player_goal_scorer_anytime','player_shots_on_goal_alternate','player_points_alternate']);
    assert.deepEqual([live.reused,live.cost],[false,7]);
    const pre=await (await req({event:ID,live:false})).json();
    assert.equal(calls.length,2,'pregame and live prices are separate requests');
    assert.equal(new URL(calls[1]).searchParams.get('markets'),MARKETS.join(','));
    assert.equal(pre.reused,false);
    assert.equal((await (await req({event:ID,live:true})).json()).reused,true);assert.equal(calls.length,2);
    // If the odds service ever refuses a milestone market, the standard prices still come back.
    const refused=setup({odds:async url=>new URL(url).searchParams.get('markets').includes('alternate')
      ?new Response('{"message":"Invalid markets"}',{status:422})
      :new Response(JSON.stringify({...event([]),id:ID}),{status:200,headers:{'x-requests-last':'3'}})});
    const fallback=await refused.req({event:ID,live:true});
    assert.equal(fallback.status,200);assert.equal(refused.calls.length,2);
    assert.equal(new URL(refused.calls[1]).searchParams.get('markets'),MARKETS.join(','));
    const other=setup({odds:async()=>new Response('{}',{status:500})});
    assert.equal((await other.req({event:ID,live:true})).status,502);assert.equal(other.calls.length,1,'only a refused market retries');
  }
  {
    // Paid calls stop below the reserve; saved prices are still served.
    const {req,calls,tick,setLeft}=setup({env:{...ENV,LIVE_ODDS_RESERVE:'16990'}});
    setLeft(16995);
    assert.equal((await req({event:ID})).status,200);   // 16988 left afterwards
    assert.equal((await req({event:ID})).status,200,'reused prices cost nothing, so the reserve does not block them');
    tick(61e3);
    const blocked=await req({event:ID});assert.equal(blocked.status,429);
    const msg=await blocked.json();assert.match(msg.error,/16988 credits left, below the 16990/);assert.equal(msg.remaining,16988);
    assert.equal(calls.length,1);
    const dflt=setup();dflt.setLeft(1999);await dflt.req({});   // learns 1999 from the free game list
    assert.equal((await dflt.req({event:ID})).status,429,'default reserve is 2000');
    assert.equal(dflt.calls.length,1);
  }
  {
    // Without an ODDS_API_KEY secret the key comes from Vault through the service role, once.
    const VAULT='vault-odds-key-123',SERVICE={...ENV,ODDS_API_KEY:undefined,SUPABASE_SERVICE_ROLE_KEY:'service-role'};
    const v=setup({env:SERVICE,vault:VAULT});
    const list=await v.req({});const prices=await v.req({event:ID});
    assert.equal(list.status,200);assert.equal(prices.status,200);
    assert.equal(v.vaultReads.length,1,'read once, then kept for the isolate');
    assert.deepEqual([v.vaultReads[0].method,v.vaultReads[0].headers.apikey,v.vaultReads[0].headers.Authorization],['POST','service-role','Bearer service-role']);
    assert(v.calls.every(u=>new URL(u).searchParams.get('apiKey')===VAULT));
    assert(!JSON.stringify(await prices.json()).includes(VAULT),'the Vault key never reaches the browser');
    // A secret takes precedence over the Vault copy.
    const both=setup({env:{...SERVICE,ODDS_API_KEY:KEY},vault:VAULT});await both.req({});
    assert.equal(both.vaultReads.length,0);assert.equal(new URL(both.calls[0]).searchParams.get('apiKey'),KEY);
    // No key anywhere: a plain message, no odds request.
    const none=setup({env:SERVICE,vault:null});const r=await none.req({event:ID});
    assert.equal(r.status,503);assert.match((await r.json()).error,/Live Odds key workflow/);assert.equal(none.calls.length,0);
    // A refused key is read again on the next press, so a rotated key takes effect.
    let current='old-vault-key-123';
    const rotated=setup({env:SERVICE,vault:()=>current,odds:async url=>new Response('{}',{status:new URL(url).searchParams.get('apiKey')===current?200:401,
      headers:{'x-requests-remaining':'9000'}})});
    assert.equal((await rotated.req({})).status,502,'old key accepted; body shape is checked separately');
    current='new-vault-key-456';rotated.tick(301e3);
    assert.equal((await rotated.req({})).status,503,'the cached old key is refused once');
    rotated.tick(11e3);await rotated.req({});
    assert.equal(rotated.vaultReads.length,2);assert.equal(new URL(rotated.calls.at(-1)).searchParams.get('apiKey'),'new-vault-key-456');
  }
  {
    // Odds service failures become plain messages without the key.
    const status=code=>setup({odds:async()=>new Response('{"message":"x"}',{status:code})}).req({event:ID});
    const refused=await status(401);assert.equal(refused.status,503);assert.match((await refused.json()).error,/refused the key|out of credits/);
    assert.equal((await status(404)).status,404);assert.equal((await status(429)).status,429);
    const err=await status(500);assert.equal(err.status,502);assert.match((await err.json()).error,/HTTP 500/);
    const down=await setup({odds:async()=>{throw new Error('connect failed '+KEY);}}).req({event:ID});
    const text=await down.text();assert.equal(down.status,502);assert(!text.includes(KEY),'network errors never echo the URL');
    const wrong=await setup({odds:async()=>new Response(JSON.stringify({id:'c'.repeat(32)}),{status:200})}).req({event:ID});
    assert.equal(wrong.status,502);
    const junk=await setup({odds:async()=>new Response('not json',{status:200})}).req({event:ID});
    assert.equal(junk.status,502);
    // A failure is kept for ten seconds, then retried.
    let n=0;const flaky=setup({odds:async()=>{n++;return new Response('{}',{status:500});}});
    await flaky.req({event:ID});await flaky.req({event:ID});assert.equal(n,1);
    flaky.tick(11e3);await flaky.req({event:ID});assert.equal(n,2);
  }
  console.log('PASS: live odds parsing, comparison, scoreboard matching and the editor-only relay.');
})().catch(e=>{console.error(e);process.exitCode=1;});

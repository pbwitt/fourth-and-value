// Live stats for Bet Tracker: feed parsing, NHL relay, matching, verdicts and
// the polling loop. Fixtures mirror the real feed shapes; nothing leaves the process.
const assert=require('node:assert/strict');
const feeds=require('../docs/tracking/live-feeds.js');
const L=require('../docs/tracking/live-stats.js');

const run=(plan,responses)=>plan.build(plan.requests.map((r,i)=>responses[i]));

// ---- Feed fixtures (trimmed real shapes) ------------------------------------
const nhlScore={games:[
  {id:2026020002,startTimeUTC:'2026-09-29T23:00:00Z',gameState:'LIVE',gameScheduleState:'OK',period:2,
   periodDescriptor:{number:2,periodType:'REG'},clock:{timeRemaining:'12:40',secondsRemaining:760,inIntermission:false},
   awayTeam:{abbrev:'MTL',name:{default:'Canadiens'},score:1},homeTeam:{abbrev:'TOR',name:{default:'Maple Leafs'},score:2}},
  {id:2026020003,startTimeUTC:'2026-09-30T00:00:00Z',gameState:'FUT',gameScheduleState:'OK',
   awayTeam:{abbrev:'NYR',name:{default:'Rangers'}},homeTeam:{abbrev:'BOS',name:{default:'Bruins'}}},
]};
const nhlBox={...nhlScore.games[0],awayTeam:{abbrev:'MTL',commonName:{default:'Canadiens'},placeName:{default:'Montréal'},score:1},
  homeTeam:{abbrev:'TOR',commonName:{default:'Maple Leafs'},placeName:{default:'Toronto'},score:2},
  playerByGameStats:{
    homeTeam:{forwards:[{name:{default:'A. Matthews'},position:'C',goals:1,assists:0,points:1,sog:3,hits:1,blockedShots:0,pim:0,powerPlayGoals:0},
                        {name:{default:'J. Tavares'},goals:0,assists:1,points:1,sog:1,hits:0,blockedShots:1,pim:2,powerPlayGoals:0},
                        {name:{default:'W. Nylander'},position:'R',goals:0,assists:0,points:0,sog:2,hits:0,blockedShots:0,pim:0,powerPlayGoals:0}],
              defense:[{name:{default:'M. Rielly'},position:'D',goals:0,assists:0,points:0,sog:1,hits:2,blockedShots:3,pim:0,powerPlayGoals:0}],goalies:[{name:{default:'J. Woll'},saves:14,shotsAgainst:15,goalsAgainst:1}]},
    awayTeam:{forwards:[{name:{default:'N. Suzuki'},goals:1,assists:0,points:1,sog:2,hits:0,blockedShots:0,pim:0,powerPlayGoals:1}],defense:[],goalies:[]}}};
const mlbSched={dates:[{games:[
  {gamePk:1,gameDate:'2026-09-29T17:05:00Z',status:{abstractGameState:'Final',codedGameState:'F',detailedState:'Final'},scheduledInnings:9,
   linescore:{currentInning:9},teams:{away:{team:{name:'Boston Red Sox',abbreviation:'BOS',teamName:'Red Sox'},score:3},home:{team:{name:'New York Yankees',abbreviation:'NYY',teamName:'Yankees'},score:4}}},
  {gamePk:2,gameDate:'2026-09-29T23:05:00Z',status:{abstractGameState:'Live',codedGameState:'I',detailedState:'In Progress'},
   linescore:{currentInning:7,currentInningOrdinal:'7th',inningState:'Middle'},teams:{away:{team:{name:'Boston Red Sox',abbreviation:'BOS',teamName:'Red Sox'},score:1},home:{team:{name:'New York Yankees',abbreviation:'NYY',teamName:'Yankees'},score:0}}},
]}]};
const mlbBox={teams:{home:{batters:[10],pitchers:[11,13],players:{
    ID10:{person:{id:10,fullName:'Aaron Judge'},position:{abbreviation:'RF'},stats:{batting:{hits:2,doubles:1,triples:0,homeRuns:1,totalBases:6,rbi:2,runs:1,baseOnBalls:0,stolenBases:0,strikeOuts:1},pitching:{}}},
    ID11:{person:{id:11,fullName:'Gerrit Cole'},position:{abbreviation:'P'},stats:{batting:{},pitching:{strikeOuts:7,outs:18,hits:4,earnedRuns:1,baseOnBalls:2,numberOfPitches:95}}},
    ID13:{person:{id:13,fullName:'Luke Weaver'},position:{abbreviation:'P'},stats:{batting:{},pitching:{strikeOuts:1,outs:3,hits:0,earnedRuns:0,baseOnBalls:0,numberOfPitches:14}}},
    ID12:{person:{id:12,fullName:'Bench Player'},stats:{batting:{},pitching:{}}}}},
  away:{batters:[],pitchers:[],players:{}}}};
const espnComp=(state,extra={})=>({date:'2026-09-27T17:00Z',status:{period:3,clock:300,type:{name:'STATUS_IN_PROGRESS',state,shortDetail:'5:00 - 3rd'}},
  competitors:[{homeAway:'home',score:'24',team:{id:'2',displayName:'Buffalo Bills',abbreviation:'BUF',shortDisplayName:'Bills'}},
               {homeAway:'away',score:'16',team:{id:'24',displayName:'Los Angeles Chargers',abbreviation:'LAC',shortDisplayName:'Chargers'}}],...extra});
const espnBoard={events:[{id:'401',date:'2026-09-27T17:00Z',competitions:[espnComp('in')]}]};
const espnSummary={header:{competitions:[espnComp('in')]},boxscore:{players:[
  {team:{id:'24'},statistics:[
    {name:'passing',keys:['completions/passingAttempts','passingYards','passingTouchdowns','interceptions'],athletes:[{athlete:{displayName:'Justin Herbert',position:{abbreviation:'QB'}},stats:['20/34','226','1','1']}]},
    {name:'rushing',keys:['rushingAttempts','rushingYards','rushingTouchdowns'],athletes:[{athlete:{displayName:'Justin Herbert'},stats:['3','-2','0']}]},
    {name:'receiving',keys:['receptions','receivingYards','receivingTouchdowns','receivingTargets'],athletes:[{athlete:{displayName:"Tre' Harris"},stats:['6','76','1','7']}]},
    {name:'interceptions',keys:['interceptions'],athletes:[{athlete:{displayName:'Derwin James Jr.'},stats:['1']}]}]},
  {team:{id:'2'},statistics:[{name:'rushing',keys:['rushingAttempts','rushingYards','rushingTouchdowns'],athletes:[
    {athlete:{displayName:'James Cook'},stats:['15','88','1']},{athlete:{displayName:'Scratched Back'},didNotPlay:true,stats:[]}]}]}]}};

// ---- Feed parsing -------------------------------------------------------------
{
  const [live,pre]=run(feeds.scoreboardPlan('NHL','2026-09-29'),[nhlScore]);
  assert.deepEqual(feeds.scoreboardPlan('NHL','2026-09-29').requests,[{proxy:{league:'NHL',date:'2026-09-29'}}],'NHL goes through the relay');
  assert.equal(live.state,'live');assert.equal(live.detail,'2nd 12:40');
  assert(Math.abs(live.elapsed-(1200+440)/3600)<1e-9);
  assert.equal(pre.state,'pre');assert.equal(pre.home.score,null);
  const box=run(feeds.boxPlan('NHL','2026020002'),[nhlBox]);
  assert.equal(box.game.home.name,'Toronto Maple Leafs');
  assert.equal(box.players.find(p=>p.name==='A. Matthews').stats.sog,3);
  assert.equal(box.players.find(p=>p.name==='J. Woll').stats.saves,14);
  const pos=n=>box.players.find(p=>p.name===n).position;
  assert.deepEqual(['A. Matthews','W. Nylander','M. Rielly','J. Tavares','J. Woll'].map(pos),['C','RW','D','F','G'],
    'NHL wingers read as LW/RW; a forward without one is F');
  const ot=run(feeds.scoreboardPlan('NHL','x'),[{games:[{...nhlScore.games[0],gameState:'OFF',gameOutcome:{lastPeriodType:'OT'}}]}])[0];
  assert.equal(ot.detail,'Final/OT');assert.equal(ot.elapsed,1);
  const ppd=run(feeds.scoreboardPlan('NHL','x'),[{games:[{...nhlScore.games[1],gameScheduleState:'PPD'}]}])[0];
  assert.equal(ppd.state,'off');assert.equal(ppd.detail,'Postponed');
}
{
  const [final,live]=run(feeds.scoreboardPlan('MLB','2026-09-29'),[mlbSched]);
  assert.match(feeds.scoreboardPlan('MLB','2026-09-29').requests[0].url,/^https:\/\/statsapi\.mlb\.com\//);
  assert.equal(final.state,'final');assert.equal(live.detail,'Mid 7th');assert.equal(live.elapsed,null);
  const box=run(feeds.boxPlan('MLB','2'),[{dates:[{games:[mlbSched.dates[0].games[1]]}]},mlbBox]);
  const judge=box.players.find(p=>p.name==='Aaron Judge'),cole=box.players.find(p=>p.name==='Gerrit Cole');
  assert.equal(judge.stats.total_bases,6);assert.equal(judge.stats.singles,0);assert.equal(judge.stats.pitcher_strikeouts,undefined);
  assert.equal(cole.stats.pitcher_strikeouts,7);assert.equal(cole.stats.hits,undefined,'pitching hits allowed are not batter hits');
  assert.equal(box.players.find(p=>p.name==='Bench Player').played,false);
  assert.deepEqual(['Aaron Judge','Gerrit Cole','Luke Weaver','Bench Player'].map(n=>box.players.find(p=>p.name===n).position),['RF','SP','RP',''],
    'the first pitcher listed is the starter');
  assert.equal(cole.stats.pitches,95);
}
{
  const [g]=run(feeds.scoreboardPlan('NFL','2026-09-27'),[espnBoard]);
  assert.match(feeds.scoreboardPlan('NFL','2026-09-27').requests[0].url,/scoreboard\?dates=20260927$/);
  assert.equal(g.state,'live');assert.equal(g.home.score,24);assert(Math.abs(g.elapsed-(2*900+600)/3600)<1e-9);
  const box=run(feeds.boxPlan('NFL','401'),[espnSummary]);
  const herbert=box.players.find(p=>p.name==='Justin Herbert');
  assert.deepEqual([herbert.side,herbert.stats.pass_completions,herbert.stats.pass_attempts,herbert.stats.pass_yds,herbert.stats.rush_yds],['away',20,34,226,-2]);
  assert.equal(herbert.stats.pass_interceptions,1);
  assert.equal(herbert.position,'QB','position survives categories that omit it');
  assert.equal(box.players.find(p=>p.name==="Tre' Harris").position,'');
  assert.equal(box.players.find(p=>p.name==='Derwin James Jr.').stats.def_interceptions,1,'defensive picks never count as thrown');
  assert.equal(box.players.find(p=>p.name==='Scratched Back').played,false);
  assert.equal(feeds.boxPlan('NBA','9').requests[0].url,'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=9');
  const nba=run(feeds.boxPlan('NBA','9'),[{header:{competitions:[espnComp('post',{status:{type:{state:'post',shortDetail:'Final'}}})]},boxscore:{players:[{team:{id:'2'},statistics:[
    {keys:['minutes','points','threePointFieldGoalsMade-threePointFieldGoalsAttempted','rebounds','assists'],athletes:[{athlete:{displayName:'Guard One'},stats:['33','21','4-9','5','7']}]}]}]}}]);
  assert.deepEqual(nba.players[0].stats,{points:21,threes:4,rebounds:5,assists:7});assert.equal(nba.game.state,'final');
}

// ---- Matching -------------------------------------------------------------------
{
  const games=run(feeds.scoreboardPlan('NHL','x'),[nhlScore]),[live]=games;
  assert.equal(L.findGame({league:'NHL',team_home:'Toronto Maple Leafs',team_away:'Montreal Canadiens'},games),live);
  assert.equal(L.findGame({league:'NHL',team_home:'TOR',team_away:'MON'},games),live,'alias abbreviations');
  assert.equal(L.findGame({league:'NHL',team_home:'MTL',team_away:'TOR'},games),live,'home/away swapped still matches');
  assert.equal(L.findGame({league:'NHL',team_home:'Boston Bruins',team_away:'Montreal Canadiens'},games),null,'both teams must match');
  const mlb=run(feeds.scoreboardPlan('MLB','x'),[mlbSched]);
  assert.equal(L.findGame({league:'MLB',team_home:'New York Yankees',team_away:'Boston Red Sox'},mlb).id,'2','doubleheader prefers the live game');
  assert(!L.teamMatches('Chicago White Sox',{name:'Boston Red Sox',short:'Red Sox',abbrev:'BOS'},'MLB'));

  const nhl=run(feeds.boxPlan('NHL','x'),[nhlBox]).players;
  assert.equal(L.findPlayer('Auston Matthews',nhl).name,'A. Matthews');
  assert.equal(L.findPlayer('Austin Matthews',nhl).name,'A. Matthews','initial + last name');
  assert.equal(L.findPlayer('Mitch Marner',nhl),null);
  assert.equal(L.findPlayer('Adam Matthews',[...nhl,{name:'A. Matthews',stats:{}}]),null,'ambiguous initials are no match');
  const nfl=run(feeds.boxPlan('NFL','x'),[espnSummary]).players;
  assert.equal(L.findPlayer('Tre Harris',nfl).name,"Tre' Harris");
  assert.equal(L.findPlayer('Derwin James',nfl).name,'Derwin James Jr.');
  assert.equal(L.findPlayer('José Ramírez',[{name:'Jose Ramirez'}]).name,'Jose Ramirez');
  // Initials as a first name are a full name, in either feed style.
  for(const box of ['J.T. Miller','J. Miller','JT Miller','J. T. Miller'])
    assert.equal(L.findPlayer('J.T. Miller',[{name:box},{name:'A. Matthews'}])?.name,box,`J.T. Miller matches ${box}`);
  assert.equal(L.findPlayer('TJ Oshie',[{name:'T.J. Oshie'}]).name,'T.J. Oshie');
  assert.equal(L.findPlayer('A.J. Brown',[{name:'AJ Brown'}]).name,'AJ Brown');
  assert.equal(L.findPlayer('J.T. Miller',[{name:'J.T. Compher'}]),null);
  assert.equal(L.findPlayer('J.T. Miller',[{name:'J. Miller'},{name:'J. Miller'}]),null,'ambiguous initials are no match');

  assert.deepEqual(L.marketSpec('NHL','player_shots_on_goal').stats,['sog']);
  assert.deepEqual(L.marketSpec('NHL','Shots on Goal').stats,['sog']);
  assert.deepEqual(L.marketSpec('MLB','batter_strikeouts').stats,['batter_strikeouts']);
  assert.deepEqual(L.marketSpec('MLB','pitcher_strikeouts').stats,['pitcher_strikeouts']);
  assert.deepEqual(L.marketSpec('NFL','player_rush_reception_yds').stats,['rush_yds','recv_yds']);
  assert.deepEqual(L.marketSpec('NBA','player_points_rebounds_assists').stats,['points','rebounds','assists']);
  assert.deepEqual(L.marketSpec('NFL','spreads'),{game:'spread'});
  assert.equal(L.marketSpec('NHL','mystery'),null);
}

// ---- Verdicts ----------------------------------------------------------------------
{
  const box=run(feeds.boxPlan('NHL','x'),[nhlBox]),game=box.game;
  const bet=o=>({league:'NHL',team_home:'TOR',team_away:'MTL',player:'Auston Matthews',market_type:'sog',side:'over',line:2.5,...o});
  let v=L.evaluate(bet(),game,box);
  assert.deepEqual([v.value,v.tone,v.label,v.progress],[3,'won','Hit',1]);
  assert.equal(v.playerTeam,'TOR','the box score side gives the player team');
  assert.equal(v.playerPosition,'C');assert.equal(v.pitches,undefined,'only pitcher markets carry a pitch count');
  assert.equal(L.evaluate(bet({player:'Nobody Here'}),game,box).playerTeam,undefined);
  assert.equal(L.teamCode({abbrev:'',short:'Leafs',name:'Toronto Maple Leafs'}),'Leafs');assert.equal(L.teamCode(undefined),null);
  v=L.evaluate(bet({line:3.5}),game,box);
  assert.deepEqual([v.tone,v.label,v.progress],['alive','Needs 1',0.75]);
  assert.equal(v.pace,6.6,'pace projects from the share of regulation played (3 SOG at 46%)');
  v=L.evaluate(bet({line:3}),game,box);assert.equal(v.label,'Needs 1','whole-number over needs to clear the line');
  assert.equal(L.evaluate(bet(),game,box).pace,null,'a hit over has no pace');
  v=L.evaluate(bet({side:'under',line:2.5}),game,box);assert.deepEqual([v.tone,v.label,v.pace],['lost','Dead',null]);
  v=L.evaluate(bet({side:'under',line:4.5}),game,box);assert.deepEqual([v.tone,v.label,v.pace],['alive','1 to spare',6.6],'unders show pace too');
  const early={...game,elapsed:0.1};assert.equal(L.evaluate(bet({line:3.5}),early,{...box,game:early}).pace,null,'too early for a pace');
  assert.equal(L.evaluate(bet({line:3.5}),{...game,elapsed:1,detail:'OT 3:00'},{...box,game:{...game,elapsed:1}}).pace,null,'no pace in overtime');
  const tot=o=>L.evaluate({league:'NHL',team_home:'TOR',team_away:'MTL',player:null,market_type:'totals',...o},game,null);
  assert.deepEqual([tot({side:'under',line:5.5}).value,tot({side:'under',line:5.5}).pace],[3,6.6],'game totals show pace');
  assert.equal(tot({side:'over',line:6.5}).pace,6.6);
  assert.equal(tot({side:'over',line:2.5}).pace,null,'a cleared over has no pace');
  assert.equal(L.evaluate({league:'NHL',team_home:'TOR',team_away:'MTL',market_type:'totals',side:'under',line:5.5},{...game,state:'final',elapsed:1},null).pace,null);
  assert.equal(L.evaluate({league:'NHL',team_home:'TOR',team_away:'MTL',market_type:'h2h',side:'TOR'},game,null).pace,undefined,'no pace on sides');
  v=L.evaluate(bet({side:'under',line:3.5}),game,box);assert.equal(v.label,'None to spare');
  v=L.evaluate(bet({market_type:'player_goals',side:'Yes',line:null}),game,box);assert.deepEqual([v.line,v.tone],[0.5,'won'],'anytime props');
  v=L.evaluate(bet({player:'Nobody Here'}),game,box);assert.deepEqual([v.tone,v.label],['missing','Not in the box score yet']);
  v=L.evaluate(bet({market_type:'faceoffs'}),game,box);assert.equal(v.kind,'unsupported');
  const final={...game,state:'final',elapsed:1};
  v=L.evaluate(bet({line:3}),final,{game:final,players:box.players});assert.deepEqual([v.tone,v.label],['push','Push']);
  v=L.evaluate(bet({line:3.5}),final,{game:final,players:box.players});assert.deepEqual([v.tone,v.note],['lost','Final · awaiting official grade']);
  assert.equal(L.evaluate(bet(),{...game,state:'pre',start:game.start},null).tone,'pre');
  assert.equal(L.evaluate(bet(),null,null).status,'nogame');

  // Yardage can fall, so an over is never locked before the final.
  const nfl=run(feeds.boxPlan('NFL','x'),[espnSummary]);
  const yd=o=>L.evaluate({league:'NFL',team_home:'Buffalo Bills',team_away:'Los Angeles Chargers',player:'Justin Herbert',market_type:'pass_yds',side:'over',line:220.5,...o},nfl.game,nfl);
  assert.deepEqual([yd().tone,yd().label,yd().pace],['ahead','Over the line',339],'yardage paces to whole yards');
  assert.deepEqual([yd({side:'under'}).tone,yd({side:'under'}).pace],['behind',339]);

  // ESPN's summary may omit the clock: the scoreboard's share played fills in,
  // and a period with no clock is unknown, not the end of the period.
  const board=run(feeds.scoreboardPlan('NFL','x'),[espnBoard])[0];
  const bare=s=>({...espnSummary,header:{competitions:[espnComp('in',{status:{type:{state:'in'},...s}})]}});
  const noClock=run(feeds.boxPlan('NFL','401'),[bare({period:3})]);
  assert.equal(noClock.game.elapsed,null);
  assert.equal(run(feeds.boxPlan('NFL','401'),[bare({})]).game.elapsed,null);
  const fill=L.evaluate({league:'NFL',team_home:'BUF',team_away:'LAC',player:'Justin Herbert',market_type:'pass_yds',side:'under',line:300.5},board,noClock);
  assert.deepEqual([fill.tone,fill.pace],['alive',339],'scoreboard clock fills in for the box score');
  assert(Math.abs(run(feeds.boxPlan('NFL','401'),[bare({period:3,displayClock:'5:00'})]).game.elapsed-(2*900+600)/3600)<1e-9,'displayClock when clock is missing');
  const nbaLate=run(feeds.boxPlan('NBA','9'),[{header:{competitions:[espnComp('in',{status:{period:4,displayClock:'45.3',type:{state:'in'}}})]},boxscore:{players:[]}}]);
  assert(Math.abs(nbaLate.game.elapsed-(3*720+720-45.3)/2880)<1e-9,'sub-minute NBA clock');

  const mlb=run(feeds.boxPlan('MLB','2'),[{dates:[{games:[mlbSched.dates[0].games[1]]}]},mlbBox]);
  assert.equal(L.evaluate({league:'MLB',team_home:'NYY',team_away:'BOS',player:'Aaron Judge',market_type:'batter_total_bases',side:'over',line:6.5},mlb.game,mlb).pace,null,'MLB has no game clock yet: no pace');

  // A player listed without an appearance is void at the final, never a zero:
  // books refund props on players who do not play.
  const nbaComp=state=>({date:'2026-10-21T23:30Z',status:{period:4,clock:0,type:{name:state==='post'?'STATUS_FINAL':'STATUS_IN_PROGRESS',state,shortDetail:state==='post'?'Final':'0:00 - 4th'}},
    competitors:[{homeAway:'home',score:'112',team:{id:'5',displayName:'Detroit Pistons',abbreviation:'DET',shortDisplayName:'Pistons'}},
                 {homeAway:'away',score:'108',team:{id:'2',displayName:'Boston Celtics',abbreviation:'BOS',shortDisplayName:'Celtics'}}]});
  const nbaBox=state=>run(feeds.boxPlan('NBA','401'),[{header:{competitions:[nbaComp(state)]},boxscore:{players:[{team:{id:'5'},statistics:[{name:'',
    keys:['minutes','points','rebounds','assists','threePointFieldGoalsMade-threePointFieldGoalsAttempted'],athletes:[
      {athlete:{displayName:'Cade Cunningham',position:{abbreviation:'G'}},stats:['36','31','6','9','3-8']},
      {athlete:{displayName:'Bench Guard',position:{abbreviation:'G'}},didNotPlay:true,reason:"DNP-COACH'S DECISION",stats:[]}]}]}]}}]);
  const nba=(o,state='post')=>{const b=nbaBox(state);return L.evaluate({league:'NBA',team_home:'Detroit Pistons',team_away:'Boston Celtics',
    player:'Bench Guard',market_type:'player_points',side:'Under',line:8.5,...o},b.game,b);};
  assert.deepEqual([nba().tone,nba().label,nba().value],['push','Did not play · void',null],'DNP under is void, not a win');
  assert.deepEqual([nba({side:'Over'}).tone,nba({market_type:'player_points_rebounds_assists',side:'Over',line:12.5}).tone],['push','push'],'DNP over and combos are void');
  assert.deepEqual([nba({},'in').tone,nba({},'in').value],['alive',0],'during play a bench player has simply not scored yet');
  assert.deepEqual([nba({player:'Cade Cunningham',side:'Over',line:29.5}).tone,nba({player:'Cade Cunningham',market_type:'player_threes',side:'Over',line:2.5}).value],['won',3]);
  // A box score whose stat columns we cannot read leaves the bet pending, never graded on zeros.
  const odd=run(feeds.boxPlan('NBA','401'),[{header:{competitions:[nbaComp('post')]},boxscore:{players:[{team:{id:'5'},statistics:[{name:'starters',
    keys:['points','rebounds'],athletes:[{athlete:{displayName:'Cade Cunningham'},stats:['31','6']}]}]}]}}]);
  const unread=L.evaluate({league:'NBA',team_home:'DET',team_away:'BOS',player:'Cade Cunningham',market_type:'player_points',side:'Under',line:29.5},odd.game,odd);
  assert.deepEqual([unread.tone,unread.label,unread.value],['missing','No readable stats for this player',undefined]);
  const nflFinal={...nfl,game:{...nfl.game,state:'final'}};
  assert.equal(L.evaluate({league:'NFL',team_home:'BUF',team_away:'LAC',player:'Scratched Back',market_type:'rush_yds',side:'under',line:20.5},nflFinal.game,nflFinal).tone,'push','NFL DNP is void');
  const mlbFinal={...mlb,game:{...mlb.game,state:'final'}};
  assert.equal(L.evaluate({league:'MLB',team_home:'NYY',team_away:'BOS',player:'Bench Player',market_type:'batter_hits',side:'under',line:0.5},mlbFinal.game,mlbFinal).tone,'push','MLB bench player is void');

  const g=o=>L.evaluate({league:'NFL',team_home:'Buffalo Bills',team_away:'Los Angeles Chargers',player:null,...o},nfl.game,null);
  assert.equal(g({market_type:'h2h',side:'Buffalo Bills'}).label,'Leading');
  assert.equal(g({market_type:'h2h',side:'Los Angeles Chargers'}).label,'Trailing');
  assert.equal(g({market_type:'spreads',side:'Buffalo Bills',line:-8}).label,'On the number');
  assert.equal(g({market_type:'spreads',side:'LAC',line:+9.5}).label,'Covering');
  assert.deepEqual([g({market_type:'totals',side:'over',line:44.5}).value,g({market_type:'totals',side:'over',line:44.5}).label],[40,'Needs 5']);
  assert.equal(g({market_type:'team_total',side:'over',line:20.5}).tone,undefined,'no team on a team total: score only');
  assert.equal(L.marketSpec('NHL','team_total').game,'total','NHL team_total means the game total');
  assert.equal(L.marketSpec('NHL','totals').game,'total','new NHL tickets record the game total as totals');
  const fin={...nfl.game,state:'final'};
  assert.equal(L.evaluate({league:'NFL',team_home:'BUF',team_away:'LAC',market_type:'spreads',side:'BUF',line:-6.5},fin,null).tone,'won');
}

// ---- Polling loop --------------------------------------------------------------------
(async()=>{
  const now=Date.parse('2026-09-29T23:40:00Z');
  const bets=[
    {id:'a',league:'NHL',game_date:'2026-09-29',team_home:'TOR',team_away:'MTL',player:'Auston Matthews',market_type:'sog',side:'over',line:3.5,status:'pending'},
    {id:'b',league:'NHL',game_date:'2026-09-29',team_home:'BOS',team_away:'NYR',player:'Artemi Panarin',market_type:'points',side:'over',line:0.5,status:'pending'},
    {id:'c',league:'MLB',game_date:'2026-09-29',team_home:'NYY',team_away:'BOS',player:'Gerrit Cole',market_type:'pitcher_strikeouts',side:'over',line:6.5,status:'pending'},
    {id:'old',league:'NHL',game_date:'2026-09-20',team_home:'TOR',team_away:'MTL',player:'Auston Matthews',market_type:'sog',side:'over',line:2.5,status:'pending'},
    {id:'won',league:'NHL',game_date:'2026-09-29',team_home:'TOR',team_away:'MTL',player:'Auston Matthews',market_type:'sog',side:'over',line:2.5,status:'won'},
  ];
  assert.deepEqual(L.liveCandidates(bets,now).map(b=>b.id),['a','b','c'],'only today/yesterday pending bets');

  const calls=[];let failMlbBox=false,updates=[];
  const proxy=async body=>{calls.push(JSON.stringify(body));return body.game?nhlBox:nhlScore;};
  const fetchJSON=async url=>{calls.push(url);
    if(url.includes('/boxscore')){if(failMlbBox)throw Error('down');return mlbBox;}
    if(url.includes('gamePk=2'))return {dates:[{games:[mlbSched.dates[0].games[1]]}]};
    return mlbSched;};
  const timers=[];
  const t=L.createLiveTracker({fetchJSON,proxy,now:()=>now,onUpdate:(v,m)=>updates.push([v,m]),setTimer:(fn,ms)=>{timers.push(ms);return timers.length;},clearTimer:()=>{}});
  t.pause();t.setBets(bets);assert.equal(calls.length,0,'paused tracker makes no requests');
  await t.refresh();
  let [views,meta]=updates.at(-1);
  assert.equal(views.get('a').label,'Needs 1');
  assert.equal(views.get('b').tone,'pre');
  assert.equal(views.get('c').label,'Hit');
  assert.deepEqual([views.get('c').pitches,views.get('c').playerPosition],[95,'SP'],'pitcher props show the live pitch count');
  assert.equal(meta.live,2);assert.equal(meta.errors,0);
  assert(!calls.some(c=>c.includes('2026020003')),'no box score fetched for games not started');
  assert.equal(calls.filter(c=>c.includes('"date"')).length,1,'one scoreboard per league and date');

  failMlbBox=true;await t.refresh();[views,meta]=updates.at(-1);
  assert.equal(views.get('c').label,'Hit','a failed refresh keeps the last good view');assert.equal(meta.errors,1);

  t.resume();await new Promise(r=>setImmediate(r));
  assert.equal(timers.at(-1),60e3,'poll every minute while a game is live');

  updates=[];t.setBets(bets.map(b=>({...b,status:'won'})));
  assert.equal(updates.at(-1)[0].size,0,'settling every bet clears the live view');

  // ---- NHL relay (edge function) ---------------------------------------------------
  const {createHandler}=await import('../supabase/functions/live-stats/handler.mjs');
  let clock=0;const upstream=[];
  const handle=createHandler({now:()=>clock,fetchImpl:async url=>{upstream.push(url);return new Response(JSON.stringify(url.includes('boxscore')?nhlBox:nhlScore),{status:200});}});
  const req=(body,origin='https://fourthandvalue.com',method='POST')=>handle(new Request('https://edge/',{method,headers:{origin},body:method==='POST'?JSON.stringify(body):undefined}));
  assert.equal((await req({league:'NHL',date:'2026-09-29'},'https://evil.example')).status,403);
  assert.equal((await req(null,undefined,'OPTIONS')).status,204);
  assert.equal((await req({league:'MLB',date:'2026-09-29'})).status,400,'relay is NHL only');
  assert.equal((await req({league:'NHL',game:'1/../../x'})).status,400);
  assert.equal((await req({league:'NHL',date:'../score'})).status,400);
  const [r1,r2]=await Promise.all([req({league:'NHL',game:2026020002}),req({league:'NHL',game:'2026020002'})]);
  assert.equal(r1.status,200);assert.deepEqual(await r2.json(),nhlBox);
  assert.deepEqual(upstream,['https://api-web.nhle.com/v1/gamecenter/2026020002/boxscore'],'concurrent callers share one upstream fetch');
  clock=14e3;await req({league:'NHL',game:'2026020002'});assert.equal(upstream.length,1,'live game cached ~15s');
  clock=16e3;await req({league:'NHL',game:'2026020002'});assert.equal(upstream.length,2);
  const down=createHandler({fetchImpl:async()=>new Response('nope',{status:503})});
  assert.equal((await down(new Request('https://edge/',{method:'POST',headers:{origin:'https://fourthandvalue.com'},body:'{"league":"NHL","date":"2026-09-29"}'}))).status,502);

  console.log('PASS: live feeds, NHL relay, bet matching, verdicts and polling.');
})().catch(e=>{console.error(e);process.exitCode=1;});

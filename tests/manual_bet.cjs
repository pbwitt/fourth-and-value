// Manual Bet Tracker entry: every market the form offers saves the same ledger row
// as a tracked offer, and the pre-save check agrees with the grader.
const assert=require('node:assert/strict');
const M=require('../docs/tracking/manual-bet.js');
const {leagues,manualGrade}=require('../docs/assets/offer-tracker.js');
const live=require('../docs/tracking/live-stats.js');
const {gradeBet}=require('../scripts/grade_bets.cjs');

const team=(abbrev,name,short,score)=>({name,abbrev,short,score});
const game=(o={})=>({id:'776001',league:'MLB',start:'2026-09-30T23:10:00Z',state:'final',detail:'Final',elapsed:1,
  away:team('CHC','Chicago Cubs','Cubs',1),home:team('SD','San Diego Padres','Padres',4),...o});
const sd=game(),games=[game({id:'776002',away:team('BOS','Boston Red Sox','Red Sox',2),home:team('NYY','New York Yankees','Yankees',9)}),sd];
const form=o=>({league:'MLB',date:'2026-09-30',game:sd,market:'totals',side:'under',line:'7.5',player:'',book:'DraftKings',odds:'-110',stake:'25',...o});
const now=Date.parse('2026-10-01T12:00:00Z');
const otherGame=()=>game({id:'776009'});

// The bet that prompted this form: Cubs @ Padres under 7.5 settles as a win on 5 runs.
let t=M.buildTicket(form());
assert.deepEqual({...t},{league:'MLB',game_date:'2026-09-30',team_home:'SD',team_away:'CHC',player:null,market_type:'totals',side:'under',
  line:7.5,book:'DraftKings',odds:-110,stake_dollars:25,model_prob:null,edge_bps:null});
assert.deepEqual(gradeBet(t,live.findGame(t,games),null,now).update.status,'won');
assert.equal(gradeBet({...t,side:'over'},sd,null,now).update.status,'lost');
assert.equal(gradeBet({...t,line:5},sd,null,now).update.status,'push');
assert(M.gradeCheck(t,sd,games,null,now).ok);

// Moneyline and spread store the team code the schedule uses, with the reader's line.
t=M.buildTicket(form({market:'h2h',side:'home',line:''}));
assert.equal(t.side,'SD');assert.equal(t.line,null);assert.equal(gradeBet(t,sd,null,now).update.status,'won');
t=M.buildTicket(form({market:'spreads',side:'away',line:'+1.5'}));
assert.equal(t.side,'CHC');assert.equal(t.line,1.5);assert.equal(gradeBet(t,sd,null,now).update.status,'lost');
t=M.buildTicket(form({market:'spreads',side:'home',line:'-2.5'}));
assert.equal(gradeBet(t,sd,null,now).update.status,'won');

// Player props grade from the box score by name.
const box={game:sd,players:[{name:'Fernando Tatis Jr.',side:'home',played:true,stats:{hits:2,total_bases:3}}]};
t=M.buildTicket(form({market:'batter_hits',side:'over',line:'1.5',player:'Fernando Tatis Jr.'}));
assert.equal(t.market_type,'batter_hits');assert.equal(gradeBet(t,sd,box,now).update.status,'won');
assert(M.gradeCheck(t,sd,games,box,now).ok);
assert.equal(t.player_team,undefined,'no box score, no team');
t=M.buildTicket(form({market:'batter_hits',side:'over',line:'1.5',player:'Fernando Tatis Jr.',box}));
assert.equal(t.player_team,'SD','the loaded box score gives the player team');
assert.equal(gradeBet(t,sd,box,now).update.player_team,undefined,'a saved team is kept');
assert.equal(M.buildTicket(form({market:'batter_hits',side:'over',line:'1.5',player:'Manny Machado',box})).player_team,undefined);
assert.equal(M.buildTicket(form({box})).player_team,undefined,'game markets have no player team');
assert.equal(M.buildTicket(form({market:'batter_hits',side:'over',line:'1.5',player:'Fernando Tatis Jr.',game:otherGame(),box})).player_team,undefined,
  'another game\'s box score is ignored');
assert.equal(M.gradeCheck({...t,player:'Fernando Tatis Sr'},sd,games,box,now).ok,true,'suffixes are ignored by the matcher');
assert.equal(M.gradeCheck({...t,player:'Manny Machado'},sd,games,box,now).ok,false,'a name missing from a final box score is flagged');

// Every market each league offers builds a ticket the grader understands, except NFL's hand-graded markets.
const nhl=game({league:'NHL',away:team('MTL','Montréal Canadiens','Canadiens'),home:team('TOR','Toronto Maple Leafs','Maple Leafs'),state:'pre'});
for(const league of Object.keys(leagues)){
  const g={...nhl,league};
  const options=M.marketOptions(league);
  assert.deepEqual(options.map(o=>o.key).sort(),Object.keys(leagues[league].markets).sort());
  assert.equal(options[0].group,'Game');
  for(const o of options){
    const sides=M.sideOptions(o.kind,g);assert(sides.length);
    const ticket=M.buildTicket({league,date:'2026-10-08',game:g,market:o.key,side:sides[0][0],line:'2.5',player:'Auston Matthews',book:'FanDuel',odds:'+120',stake:'10'});
    assert.equal(ticket.player,M.needsPlayer(o.kind)?'Auston Matthews':null,`${league} ${o.key} player`);
    assert.equal(ticket.line,M.needsLine(o.kind)?2.5:null,`${league} ${o.key} line`);
    assert.equal(o.manual,manualGrade({sport:league,market:o.key}));
    const check=M.gradeCheck(ticket,g,[g],null,Date.parse('2026-10-08T12:00:00Z'));
    assert.equal(check.ok,!o.manual,`${league} ${o.key}: ${check.text}`);
    if(!o.manual)assert(live.marketSpec(league,ticket.market_type),`${league} ${o.key} gradeable`);
  }
}
assert.equal(M.marketOptions('MLB').find(o=>o.key==='totals').label,'Total runs');
assert.equal(M.marketOptions('NFL').find(o=>o.key==='anytime_td').kind,'yesno');

// Validation is reader-facing and nothing incomplete is saved.
for(const [change,msg] of [[{game:null},/game/],[{line:''},/line/],[{line:'abc'},/line/],[{book:' '},/sportsbook/],
  [{market:'batter_hits',player:''},/player/],[{market:'h2h',side:'over'},/pick/],[{odds:'50'},/odds/],[{stake:'0'},/stake/],[{market:'nope'},/market/]]){
  assert.throws(()=>M.buildTicket(form(change)),msg,JSON.stringify(change));
}

// Doubleheaders, old dates and postponements are flagged before saving.
const second=game({id:'776003',start:'2026-10-01T02:40:00Z'});
assert.equal(M.gradeCheck(M.buildTicket(form({game:second})),second,[sd,second],null,now).ok,false);
assert.equal(M.gradeCheck(M.buildTicket(form()),sd,games,null,Date.parse('2026-10-20T12:00:00Z')).ok,false);
const off=game({state:'off'});
assert.equal(M.gradeCheck(M.buildTicket(form({game:off})),off,[off],null,now).ok,false);

console.log('PASS: manual entry builds offer-identical tickets for every market, grades the SD/CHC total, and flags ungradeable bets.');

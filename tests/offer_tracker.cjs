// Verify every sport's offer payloads against the existing ledger and grading contract.
const assert=require('node:assert/strict');
const {ticketData,identity,manualGrade,leagues}=require('../docs/assets/offer-tracker.js');
const live=require('../docs/tracking/live-stats.js');
const {ticketData:briefingTicket}=require('../docs/assets/briefing-picks.js');
const now=Date.parse('2026-10-08T20:00:00Z');
const row={sport:'NHL',event_id:'game',game:'Montreal Canadiens @ Toronto Maple Leafs',commence_time:'2026-10-09T02:00:00Z',
  quoted_at:new Date(now).toISOString(),model_data_checked_at:new Date(now).toISOString(),
  player:'Auston Matthews',market:'player_shots_on_goal',side:'Over',line:3,book:'caesars',price:-110,
  independent_probability:.45,final_probability:.45,push_probability:.1};
for(const [market,type] of Object.entries({player_goals:'goals',player_assists:'assists',player_points:'points',player_shots_on_goal:'sog',totals:'totals',h2h:'h2h',spreads:'spreads'})){
  const r={...row,market,player:market.startsWith('player_')?row.player:'',side:['h2h','spreads'].includes(market)?'Toronto Maple Leafs':'Under',line:market==='h2h'?null:market==='spreads'?-1.5:6};
  const ticket=ticketData(r,-120,25,now);
  assert.equal(ticket.market_type,type);assert.equal(ticket.game_date,'2026-10-08','ET date, not UTC date');
  assert.equal(ticket.model_prob,.5,'conditional probability excludes pushes');
  assert.deepEqual(ticket,briefingTicket(r,-120,25),'same ledger contract as the existing briefing');
  assert.equal(ticket.line,r.line);assert.equal(ticket.odds,-120);assert.equal(ticket.book,'caesars');
}
assert.equal(ticketData({...row,line:0},100,10,now).line,0);
assert.equal(ticketData({...row,final_probability:0},100,10,now).model_prob,0);
for(const change of [{final_probability:null},{push_probability:null},{independent_probability:null},{model_withheld:'Unavailable'},
  {model_data_checked_at:new Date(now-37*3600e3).toISOString()},{model_data_checked_at:new Date(now+1).toISOString()},
  {final_probability:.95,push_probability:.1}]){
  const ticket=ticketData({...row,...change},100,10,now);assert.equal(ticket.model_prob,null);assert.equal(ticket.edge_bps,null);
}
for(const [price,stake] of [[50,10],[-110,0],[-110,1.001],[-110,NaN]])assert.throws(()=>ticketData(row,price,stake,now));
for(const change of [{line:null},{book:''},{game:''},{commence_time:'bad'},{market:'unknown'},{side:''},{sport:'NCAAF'}])assert.throws(()=>ticketData({...row,...change},-110,10,now));
for(const change of [{line:3.5},{book:'fanduel'},{price:-120},{event_id:'other'},{settlement_profile:'different'},{sport:'MLB'}])assert.notEqual(identity(row),identity({...row,...change}));

// MLB: same payload as the briefing when the model is fresh; expired model inputs never become a probability.
const mlb={sport:'MLB',event_id:'m',game:'Philadelphia Phillies @ Atlanta Braves',home_team:'Atlanta Braves',away_team:'Philadelphia Phillies',
  commence_time:'2026-10-01T23:08:00Z',quoted_at:new Date(now).toISOString(),model_checked_at:new Date(now-30*60e3).toISOString(),
  player:'Bryce Harper',market:'batter_hits',side:'Over',line:.5,book:'fanduel',price:-150,model_probability:.62,model_push_probability:0};
for(const market of Object.keys(leagues.MLB.markets)){
  const r={...mlb,market,player:['h2h','spreads','totals'].includes(market)?'':mlb.player,side:['h2h','spreads'].includes(market)?'Atlanta Braves':'Over',line:market==='h2h'?null:mlb.line};
  const ticket=ticketData(r,-150,20,now);
  assert.deepEqual(ticket,briefingTicket(r,-150,20),`MLB ${market} matches the briefing`);
  assert(live.marketSpec('MLB',ticket.market_type),`MLB ${market} is gradeable`);
}
assert.equal(ticketData({...mlb,model_checked_at:new Date(now-91*60e3).toISOString()},-150,20,now).model_prob,null,'stale MLB model');
assert.equal(ticketData({...mlb,model_checked_at:undefined},-150,20,now).model_prob,null,'missing MLB model check');
assert.equal(ticketData({...mlb,model_push_probability:null},-150,20,now).model_prob,null);
// NBA: no published model yet, so a model probability is never saved.
const nba={sport:'NBA',event_id:'n',game:'Boston Celtics @ Detroit Pistons',home_team:'Detroit Pistons',away_team:'Boston Celtics',
  commence_time:'2026-10-21T23:00:00Z',quoted_at:new Date(now).toISOString(),player:'',market:'totals',side:'Over',line:221.5,book:'fanduel',price:-110,model_probability:.7};
for(const market of Object.keys(leagues.NBA.markets)){
  const ticket=ticketData({...nba,market,player:market.startsWith('player_')?'Jayson Tatum':'',side:['h2h','spreads'].includes(market)?'Boston Celtics':'Under',line:market==='h2h'?null:nba.line},-110,10,now);
  assert.equal(ticket.model_prob,null);assert.equal(ticket.league,'NBA');assert(live.marketSpec('NBA',ticket.market_type),`NBA ${market} is gradeable`);
}
assert.equal(ticketData({...nba,market:'h2h',side:'Boston Celtics',line:null},125,10,now).line,null);
// NFL: fitted calibration only; anytime TD has no line; manual markets are saved but flagged.
const nfl={sport:'NFL',event_id:'f',game:'Carolina Panthers @ Atlanta Falcons',home_team:'Atlanta Falcons',away_team:'Carolina Panthers',
  commence_time:'2026-10-04T17:00:00Z',quoted_at:new Date(now).toISOString(),player:'Bijan Robinson',market:'rush_yds',side:'over',line:74.5,
  book:'draftkings',price:-115,model_prob:.56,model_status:'Calibration fitted (isotonic)'};
for(const market of Object.keys(leagues.NFL.markets)){
  const r={...nfl,market,...(['h2h','spreads','totals'].includes(market)?{player:'',side:market==='totals'?'Under':'Atlanta Falcons',line:market==='h2h'?null:44.5}:{}),
    ...(['anytime_td','first_td','last_td'].includes(market)?{side:'yes',line:null}:{})};
  const ticket=ticketData(r,-115,10,now);
  assert.equal(ticket.model_prob,.56);assert.equal(ticket.market_type,market);
  assert.equal(!!live.marketSpec('NFL',market),!manualGrade(r),`NFL ${market}: automatic grading matches the manual flag`);
}
const td=ticketData({...nfl,market:'anytime_td',side:'yes',line:null},250,10,now);assert.equal(td.line,null);assert.equal(td.side,'yes');
assert.equal(ticketData({...nfl,model_status:'Historical rate only'},-115,10,now).model_prob,null,'uncalibrated NFL estimate');
assert.throws(()=>ticketData({...nfl,side:'over',line:null},-115,10,now),'over/under needs a line');
console.log('PASS: NHL, MLB, NBA and NFL markets, exact offers, ET dates, push probabilities, missing models, grading contract and validation.');

// Verify NHL offer payloads against the existing ledger and grading contract.
const assert=require('node:assert/strict');
const {ticketData,identity}=require('../docs/assets/nhl-tracker.js');
const {ticketData:briefingTicket}=require('../docs/assets/briefing-picks.js');
const now=Date.parse('2026-10-08T20:00:00Z');
const row={event_id:'game',game:'Montreal Canadiens @ Toronto Maple Leafs',commence_time:'2026-10-09T02:00:00Z',
  quoted_at:new Date(now).toISOString(),model_data_checked_at:new Date(now).toISOString(),
  player:'Auston Matthews',market:'player_shots_on_goal',side:'Over',line:3,book:'caesars',price:-110,
  independent_probability:.45,final_probability:.45,push_probability:.1};
for(const [market,type] of Object.entries({player_goals:'goals',player_assists:'assists',player_points:'points',player_shots_on_goal:'sog',totals:'team_total',h2h:'h2h',spreads:'spreads'})){
  const r={...row,market,player:market.startsWith('player_')?row.player:'',side:['h2h','spreads'].includes(market)?'Toronto Maple Leafs':'Under',line:market==='h2h'?null:market==='spreads'?-1.5:6};
  const ticket=ticketData(r,-120,25,now);
  assert.equal(ticket.market_type,type);assert.equal(ticket.game_date,'2026-10-08','ET date, not UTC date');
  assert.equal(ticket.model_prob,.5,'conditional probability excludes pushes');
  assert.deepEqual(ticket,briefingTicket({...r,sport:'NHL'},-120,25),'same ledger contract as the existing briefing');
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
for(const change of [{line:null},{book:''},{game:''},{commence_time:'bad'},{market:'unknown'},{side:''}])assert.throws(()=>ticketData({...row,...change},-110,10,now));
for(const change of [{line:3.5},{book:'fanduel'},{price:-120},{event_id:'other'},{settlement_profile:'different'}])assert.notEqual(identity(row),identity({...row,...change}));
console.log('PASS: all seven NHL markets, exact offers, ET dates, push probabilities, missing models and validation.');

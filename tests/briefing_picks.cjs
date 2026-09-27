const assert=require('node:assert/strict');
const {collect,rowHTML,day,ticketData}=require('../docs/assets/briefing-picks.js');
const now=Date.parse('2026-09-27T12:00:00Z'), iso=t=>new Date(t).toISOString();
function fixture(t=now) {
  const base={game:'Away @ Home',commence_time:iso(t+3600e3),player:'Example player',side:'Over',line:2.5,price:110,book:'a',book_label:'Book A',market:'player_points',market_label:'Points',quoted_at:iso(t-60e3)};
  return {
    NFL:{schema_version:1,status:'ready',generated_at:iso(t),rows:[{...base,game_id:'nfl1',bookmaker:'a',market_std:'receptions',name:'Over',point:2.5,last_update:base.quoted_at,model_prob:.6,model_status:'Calibration fitted · historical',edge_bps:100}]},
    MLB:{status:'ready',last_success_at:iso(t),model_checked_at:iso(t),rows:[{...base,event_id:'mlb1',is_model_pick:true,model_probability:.6,model_ev_pct:8}]},
    NHL:{status:'ready',snapshot_id:'snap',last_success_at:iso(t)},
    NHLBoard:{schema_version:1,status:'ready',source_snapshot_id:'snap',generated_at:iso(t),decision_date:day(t),candidates:[{...base,nhl_game_id:'nhl1',candidate_rank:1,independent_probability:.6,final_probability:.6,decision_at:iso(t),model_data_checked_at:iso(t),offer_id:'offer',forecast_id:'forecast',human_decision:'unreviewed'}]}
  };
}
assert.equal(collect(fixture(),now).selected.length,3);
// Missing independent model never inherits a consensus or 50% estimate.
let f=fixture();f.NFL.rows[0].model_prob=null;f.MLB.rows[0].model_probability=null;f.NHLBoard.candidates[0].independent_probability=null;
assert.equal(collect(f,now).selected.length,0);
f=fixture();f.NFL.rows[0].model_status='Legacy estimate';f.MLB.rows[0].is_model_pick=false;f.NHLBoard.candidates=[];
assert.equal(collect(f,now).selected.length,0);
// Different expiry policies, no future quotes, and no page-build quote substitution.
for(const [sport,limit] of [['NFL',48*3600e3],['MLB',90*60e3],['NHL',30*60e3]]) {
  for(const delta of [limit+1,-1]) {
    f=fixture();const r=sport==='NHL'?f.NHLBoard.candidates[0]:f[sport].rows[0];
    r[sport==='NFL'?'last_update':'quoted_at']=iso(now-delta);
    assert(!collect(f,now).selected.some(r=>r.sport===sport),`${sport} invalid quote ${delta}`);
  }
}
f=fixture();f.MLB.model_checked_at=iso(now-91*60e3);f.NHLBoard.candidates[0].decision_at=iso(now-31*60e3);
assert.deepEqual(collect(f,now).selected.map(r=>r.sport),['NFL']);
f=fixture();f.NHLBoard.source_snapshot_id='old';assert(!collect(f,now).selected.some(r=>r.sport==='NHL'));
f=fixture();f.NHL.model_error='failed';f.MLB.status='feed_error';f.NFL=null;assert.equal(collect(f,now).selected.length,0);
f=fixture();f.MLB=null;assert.equal(collect(f,now).selected.length,2,'one feed failure must not hide other sports');
// NHL review status is bound to the exact offer and forecast; a pass stays excluded.
f=fixture();f.NHLBoard.candidates[0].human_decision='pass';assert.equal(collect(f,now).selected.length,2);
f=fixture();f.NHLBoard.candidates[0].qualitative_review={status:'research_support',offer_id:'wrong',forecast_id:'forecast'};
assert.equal(collect(f,now).selected.at(-1).review,'Context review needed');
f.NHLBoard.candidates[0].qualitative_review.offer_id='offer';assert.match(collect(f,now).selected.at(-1).review,/analyst review needed/);
// ET day, including next-UTC-day evening games and DST boundaries.
f=fixture(Date.parse('2026-09-27T02:00:00Z'));assert.equal(collect(f,Date.parse('2026-09-27T02:00:00Z')).selected.length,3);
assert.equal(day('2026-11-01T05:30:00Z'),day('2026-11-01T06:30:00Z'));
f=fixture();f.NFL.rows[0].commence_time='2026-09-28T12:00:00Z';assert.equal(collect(f,now).selected.length,2);
assert.equal(collect(fixture(),now+3600e3).selected.length,0,'started games expire while open');
// Max four per league, one per game, no cross-sport score ranking or payout ranking.
f=fixture();const r=f.MLB.rows[0];f.MLB.rows=Array.from({length:7},(_,i)=>({...r,event_id:'m'+i,model_ev_pct:8+i}));
f.MLB.rows.push({...f.MLB.rows[6],book:'better',price:120});
const picks=collect(f,now).selected.filter(r=>r.sport==='MLB');
assert.equal(picks.length,4);assert.equal(new Set(picks.map(r=>r.game_id)).size,4);assert.equal(picks[0].book,'better');
f=fixture();const picked=collect(f,now).selected[0],html=rowHTML({...picked,player:'<img src=x onerror=alert(1)>'});
assert(html.includes(iso(now-60e3)));assert(!html.includes('<img'));assert(html.includes('+110'));assert(html.includes('Book A'));
// Track actual execution price, preserve exact market identity and handle pushes.
let ticket=ticketData(picked,-120,25);
assert.equal(ticket.odds,-120);assert.equal(ticket.stake_dollars,25);assert.equal(ticket.market_type,'receptions');
assert.equal(ticket.side,'over');assert.equal(ticket.line,2.5);assert.equal(ticket.model_prob,.6);
assert(Math.abs(ticket.edge_bps-(.6-120/220)*10000)<1e-8);
const mlb=collect(fixture(),now).selected.find(r=>r.sport==='MLB');
ticket=ticketData({...mlb,market:'h2h',side:'Home',player:'',line:null,model_probability:.54,model_push_probability:0},150,12.5);
assert.equal(ticket.line,null);assert.equal(ticket.side,'Home');assert.equal(ticket.market_type,'h2h');assert.equal(ticket.player,null);
const nhl=collect(fixture(),now).selected.find(r=>r.sport==='NHL');
ticket=ticketData({...nhl,push_probability:.1},100,10);
assert.equal(ticket.market_type,'points');assert(Math.abs(ticket.model_prob-2/3)<1e-8);
assert.equal(ticketData(mlb,110,10).model_prob,null,'unknown push probability must not become zero');
assert.throws(()=>ticketData(picked,99,25));assert.throws(()=>ticketData(picked,110,0));assert.throws(()=>ticketData(picked,110,2.001));
assert(rowHTML(picked).includes('Track bet'));assert(rowHTML(picked,0,true).includes('disabled'));
console.log('PASS: morning shortlist model gates, exact quotes, ET days, freshness, failures, exposure and review identity.');
module.exports={fixture};

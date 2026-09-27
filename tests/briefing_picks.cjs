const assert=require('node:assert/strict');
const {collect,rowHTML,day,ticketData,reviewKey,reviewBetKey,summaryHTML,comparison,researchStatus}=require('../docs/assets/briefing-picks.js');
const now=Date.parse('2026-09-27T12:00:00Z'), iso=t=>new Date(t).toISOString();
function fixture(t=now) {
  const base={game:'Away @ Home',commence_time:iso(t+3600e3),player:'Example player',side:'Over',line:2.5,price:110,book:'a',book_label:'Book A',market:'player_points',market_label:'Points',quoted_at:iso(t-60e3)};
  return {
    NFL:{schema_version:1,status:'ready',generated_at:iso(t),rows:[{...base,game_id:'nfl1',bookmaker:'a',market_std:'receptions',name:'Over',point:2.5,last_update:base.quoted_at,model_prob:.6,push_prob:0,mu:3.4,consensus_prob:.53,consensus_line:2.5,book_count:3,model_status:'Calibration fitted · historical',edge_bps:100}]},
    MLB:{status:'ready',last_success_at:iso(t),model_checked_at:iso(t),rows:[{...base,event_id:'mlb1',is_model_pick:true,model_probability:.6,model_push_probability:0,other_book_probability:.52,other_books:2,model_mean:3.1,model_ev_pct:8}]},
    NHL:{status:'ready',snapshot_id:'snap',last_success_at:iso(t)},
    NHLBoard:{schema_version:1,status:'ready',source_snapshot_id:'snap',generated_at:iso(t),decision_date:day(t),candidates:[{...base,nhl_game_id:'nhl1',candidate_rank:1,independent_probability:.6,final_probability:.6,push_probability:0,market_probability:.51,other_books:3,projected_mean:3.2,decision_at:iso(t),model_data_checked_at:iso(t),offer_id:'offer',forecast_id:'forecast',human_decision:'unreviewed'}]}
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
assert.equal(ticketData({...mlb,model_push_probability:undefined},110,10).model_prob,null,'unknown push probability must not become zero');
assert.throws(()=>ticketData(picked,99,25));assert.throws(()=>ticketData(picked,110,0));assert.throws(()=>ticketData(picked,110,2.001));
assert(rowHTML(picked).includes('Track bet'));assert(rowHTML(picked,0,true).includes('disabled'));
// Sourced analysis is tied to the exact offer/forecast. Earlier context is
// preserved with an explicit warning, never promoted to a current approval.
f=fixture();const original=collect(f,now).selected.find(r=>r.sport==='MLB');
const reviewed={...original,review_key:reviewKey(original),review_bet_key:reviewBetKey(original),offer_id:'o',forecast_id:'f',
  qualitative_review:{status:'concern',offer_id:'o',forecast_id:'f',reviewed_at:iso(now),countercase:'A changed role may invalidate the estimate.',open_checks:['Verify the lineup.'],
    evidence:[{source_id:'s1',direction:'concern',interpretation:'Check the projected role.',represented_in:'model_features'}]}};
f.Reviews={schema_version:1,sports:{MLB:{decision_date:day(now),review_status:'completed',candidates:[reviewed],sources:[{source_id:'s1',url:'https://www.espn.com/mlb/story/test',title:'Synthetic report',published_at:iso(now-3600e3)}]}}};
let researched=collect(f,now).selected.find(r=>r.sport==='MLB');
assert.equal(researched.review_matches_current,true);assert.match(researched.review,/Sourced concern/);
assert.match(rowHTML(researched),/Case against:/);assert.match(rowHTML(researched),/espn.com/);
f.MLB.rows[0].price=120;researched=collect(f,now).selected.find(r=>r.sport==='MLB');
assert.equal(researched.review_matches_current,false);assert.match(researched.review,/needs recheck/);
assert.match(rowHTML(researched),/earlier context, not a review of the current offer/);
f.MLB.rows[0].line=3.5;assert.equal(collect(f,now).selected.find(r=>r.sport==='MLB').qualitative_review,undefined);
f.MLB.rows[0].line=2.5;f.Reviews.sports.MLB.decision_date='2026-09-26';
assert.equal(collect(f,now).selected.find(r=>r.sport==='MLB').qualitative_review,undefined);
f.Reviews.sports.MLB.decision_date=day(now);f.Reviews.sports.MLB.candidates[0].qualitative_review.reviewed_at=iso(now+1000);
assert.equal(collect(f,now).selected.find(r=>r.sport==='MLB').qualitative_review,undefined);
f.Reviews.sports.MLB.candidates[0].qualitative_review.reviewed_at=iso(now);
f.Reviews.sports.MLB.sources[0].url='javascript:alert(1)';f.Reviews.sports.MLB.candidates[0].qualitative_review.countercase='<img src=x onerror=alert(1)>';
const safe=rowHTML(collect(f,now).selected.find(r=>r.sport==='MLB'));
assert(!safe.includes('javascript:'));assert(!safe.includes('<img'));
assert.match(summaryHTML([]),/No current candidates/);
assert.match(summaryHTML(collect(fixture(),now).selected),/awaiting our analysis/);
f.MLB.rows[0].price=110;f.Reviews.sports.MLB.sources[0].url='https://www.espn.com/mlb/story/test';
let summarized=summaryHTML(collect(f,now).selected);
assert.match(summarized,/Our review found a concern to resolve/);assert.match(summarized,/Still to check:/);
assert.match(summarized,/Verify the lineup/);assert.match(summarized,/href="#pick-review-1"/);
assert(!summarized.includes('<img'));assert.match(summarized,/1 of 3 current candidates/);
f.MLB.rows[0].price=120;summarized=summaryHTML(collect(f,now).selected);
assert.match(summarized,/price or forecast has changed/);assert.match(summarized,/assessed \+110/);
f.Reviews.sports.MLB.candidates[0].qualitative_review.offer_id='wrong';
assert.match(summaryHTML(collect(f,now).selected),/awaiting our analysis/);
// Keep it a few paragraphs, cover sports, and do not invent a new ranking.
const sample={...reviewed,reviewed_candidate:reviewed,review_sources:[],review:'Sourced concern'};
const many=[{...sample,sport:'NFL',game:'NFL first'},{...sample,sport:'NFL',game:'NFL second'},
  {...sample,sport:'MLB',game:'MLB first'}, {...sample,sport:'MLB',game:'MLB second'}];
summarized=summaryHTML(many);
assert.equal((summarized.match(/class="pick-summary-paragraph"/g)||[]).length,3);
assert(summarized.indexOf('NFL first')<summarized.indexOf('MLB first'));
assert(summarized.indexOf('MLB first')<summarized.indexOf('NFL second'));
assert(!summarized.includes('MLB second'));
// Comparable model/market percentages, exact-line labels and missing-data behavior.
const nflView={...picked,mu:42.17,consensus_line:33.5,consensus_prob:.52,book_count:5,line:32.5};
assert.equal(comparison(nflView).model,.6);
assert.equal(comparison(nflView).market,.52);
assert.match(rowHTML(nflView),/42.2/);
assert.match(rowHTML(nflView),/Median line: 33.5/);
assert.match(rowHTML(nflView),/Win chance\* at 32.5/);
assert.match(rowHTML(nflView),/includes listed book/);
assert.match(rowHTML(nflView),/47.6% break-even/);
assert.match(rowHTML(nflView),/Historical outcome calibration/);
for(const sport of ['MLB','NHL']) {
  const row=sport==='MLB'?{...mlb,model_probability:.54,model_push_probability:.1}: {...nhl,final_probability:.54,push_probability:.1};
  assert(Math.abs(comparison(row).model-.6)<1e-8);
  assert.match(rowHTML(row),/60.0%/);assert.match(rowHTML(row),/Push: 10.0%/);
  assert.match(rowHTML(row),/other paired books/);
  assert.equal(comparison({...row,[sport==='MLB'?'model_push_probability':'push_probability']:undefined}).model,null);
  assert.equal(comparison({...row,[sport==='MLB'?'model_push_probability':'push_probability']:.8}).model,null);
}
assert.equal(comparison({...nflView,consensus_prob:null}).market,null);
assert.equal(comparison({...nflView,book_count:0}).market,null);
assert.equal(comparison({...mlb,other_book_probability:null,consensus_probability:.9}).market,null);
assert(!rowHTML({...mlb,model_mean_label:'<img src=x>'}).includes('<img'));
assert(!rowHTML({...mlb,market:'h2h',model_mean:987.6}).includes('987.6'));
assert(!rowHTML({...nhl,projected_mean:null,projected_home_reg_goals:99.9}).includes('99.9'));
const noEvidence={...nflView,reviewed_candidate:reviewed,review_sources:[],qualitative_review:{...reviewed.qualitative_review,
 status:'needs_information',prompt_version:'mlb-nfl-context-1',evidence:[]}};
const detail=rowHTML(noEvidence),brief=summaryHTML([noEvidence]);
assert.match(detail,/Additional supporting context unverified/);
assert.match(detail,/Method correction/);assert.match(detail,/historical results/);
assert.match(detail,/Fourth &amp; Value analysis/);assert(!detail.includes('Astra'));
assert.match(brief,/reporting reviewed has not yet established/);
assert(!brief.includes('Astra'));assert(!brief.includes('market-calibrated'));
assert.equal(researchStatus('NFL',{sports:{NFL:{decision_date:day(now),review_status:'already_attempted_this_session'}}},[noEvidence],now),'NFL: 1/1 candidates reviewed');
assert.equal(researchStatus('NFL',{},[],now),'NFL: no current candidates');
console.log('PASS: morning shortlist model gates, exact quotes, ET days, freshness, failures, exposure and review identity.');
module.exports={fixture};

const assert=require('node:assert/strict');
const {collect,shortlist,rowHTML,day,ticketData,reviewKey,reviewBetKey,summaryHTML,comparison,researchStatus}=require('../docs/assets/briefing-picks.js');
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
for(const [sport,limit] of [['NFL',90*60e3],['MLB',90*60e3],['NHL',30*60e3]]) {
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
f.NHLBoard.candidates[0].qualitative_review.offer_id='offer';assert.match(collect(f,now).selected.at(-1).review,/Sourced support/);
// ET day, including next-UTC-day evening games and DST boundaries.
f=fixture(Date.parse('2026-09-27T02:00:00Z'));assert.equal(collect(f,Date.parse('2026-09-27T02:00:00Z')).selected.length,3);
assert.equal(day('2026-11-01T05:30:00Z'),day('2026-11-01T06:30:00Z'));
f=fixture();f.NFL.rows[0].commence_time='2026-09-28T12:00:00Z';assert.equal(collect(f,now).selected.length,2);
assert.equal(collect(fixture(),now+3600e3).selected.length,0,'started games expire while open');
// No candidate/game quota; identical offers use the best available price.
f=fixture();const r=f.MLB.rows[0];f.MLB.rows=Array.from({length:7},(_,i)=>({...r,event_id:'m'+i,model_ev_pct:8+i}));
f.MLB.rows.push({...f.MLB.rows[6],book:'better',price:120});
const picks=collect(f,now).selected.filter(r=>r.sport==='MLB');
assert.equal(picks.length,7);assert.equal(new Set(picks.map(r=>r.game_id)).size,7);assert.equal(picks[0].book,'better');
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
assert.match(summarized,/0 of 3 current candidates reviewed; 1 with earlier analysis requiring a recheck/);
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
assert.match(detail,/No relevant reporting was verified/);
assert.match(detail,/Method correction/);assert.match(detail,/historical results/);
assert.match(detail,/Fourth &amp; Value analysis/);assert(!detail.includes('Astra'));
assert.match(brief,/full betting assessment is pending/);
assert(!brief.includes('Astra'));assert(!brief.includes('market-calibrated'));
assert.equal(researchStatus('NFL',{sports:{NFL:{decision_date:day(now),review_status:'already_attempted_this_session'}}},[noEvidence],now),'NFL: 1/1 candidates reviewed');
assert.equal(researchStatus('NFL',{},[],now),'NFL: no current candidates');
console.log('PASS: morning shortlist model gates, exact quotes, ET days, freshness, failures, exposure and review identity.');
module.exports={fixture};

// Reporting status and betting assessment are distinct; model-only consideration is possible.
for(const verdict of ['consider','wait','pass']) {
 const assessed={...noEvidence,qualitative_review:{...noEvidence.qualitative_review,prompt_version:'mlb-nfl-context-3',assessment:{
  verdict,reason:'Specific model and price assessment.',model_case:'Opportunity forecast is uncertain.',price_case:'Exact-line quote assessed.',
  context_case:'No additional reporting verified.',blocking_checks:verdict==='wait'?['Confirm projected opportunity.']:[]}}};
 assert.match(rowHTML(assessed),/Our assessment:/);assert.match(rowHTML(assessed),/Model case:/);assert.match(rowHTML(assessed),/Price case:/);
 assert.match(summaryHTML([assessed]),/Specific model and price assessment/);
 assert(!summaryHTML([assessed]).includes('full betting assessment is pending'));
 assert(!rowHTML({...assessed,qualitative_review:{...assessed.qualitative_review,assessment:{...assessed.qualitative_review.assessment,reason:'<img src=x>'}}}).includes('<img'));
}

// Additional markets in one game survive, and review verdicts order the whole pool.
f=fixture();f.MLB.rows.push({...f.MLB.rows[0],market:'pitcher_strikeouts',market_label:'Strikeouts',line:5.5});
assert.equal(collect(f,now).selected.filter(r=>r.sport==='MLB').length,2);
assert.equal(collect(f,now).selected.find(r=>r.sport==='MLB').related_candidates,1);
// A current MLB consider assessment stays above the first twenty NFL rows.
f=fixture();f.NFL.rows=Array.from({length:25},(_,i)=>({...f.NFL.rows[0],game_id:'nfl-'+i}));
const crossSportReview={...reviewed,qualitative_review:{...reviewed.qualitative_review,offer_id:'o',assessment:{
 verdict:'consider',reason:'Current model and price assessed.',model_case:'Experimental model.',
 price_case:'Exact offer reviewed.',context_case:'No material blocker identified.',blocking_checks:[]}}};
f.Reviews={schema_version:1,sports:{MLB:{decision_date:day(now),review_status:'completed',candidates:[crossSportReview],sources:[]}}};
assert.equal(collect(f,now).selected[0].sport,'MLB');
f.MLB.rows[0].price=120;
assert.equal(collect(f,now).selected[0].sport,'NFL');
// Quarantine historical partial workloads and per-offer calibration extrapolation.
f=fixture();let nf=f.NFL.rows[0];nf.player='Synthetic QB';nf.market_std='pass_yds';nf.model_prob=.98;
const gkey=JSON.stringify([nf.game_id,nf.player,nf.market_std]);
const okey=JSON.stringify([nf.bookmaker,nf.name,nf.point,nf.price,nf.last_update]);
f.NFLContext={generated_at:f.NFL.generated_at,groups:{[gkey]:{offers:{[okey]:{raw_probability:.95}},calibration:{fitted_raw_range:[.1,.9]},projection:{current_sample:[{attempts:5}],career_baselines:{attempts:{mean:31}}}}}};
let out=collect(f,now);assert(!out.selected.some(r=>r.sport==='NFL'));assert.equal(out.excluded[0].exclusion_reasons.length,2);
f.NFLContext.groups[gkey].offers[okey].raw_probability=.7;
f.NFLContext.groups[gkey].projection.current_sample=[{attempts:35}];
assert(collect(f,now).selected.some(r=>r.sport==='NFL'));
// A research lead must have a real, fresh offer and retains no fake model EV.
f=fixture();const research={...mlb,sport:'MLB',game_id:'research-game',model_probability:null,model_ev_pct:null,
 discovery_origin:'independent_research',model_withheld:'No eligible model',review:'Awaiting assessment',score:0};
f.Discovery={schema_version:1,decision_date:day(now),generated_at:iso(now),candidates:[research]};
let candidate=collect(f,now).selected.find(r=>r.game_id==='research-game');assert(candidate);
assert.equal(ticketData(candidate,110,10).model_prob,null);assert.match(rowHTML(candidate),/Not established/);
assert.match(rowHTML(candidate),/Origin: independent research/);
f.Discovery.candidates[0].quoted_at=iso(now-91*60e3);assert(!collect(f,now).selected.some(r=>r.game_id==='research-game'));
// Integer quote identity is numeric even when Python serializes a 5.0 threshold.
f=fixture();nf=f.NFL.rows[0];nf.point=5;nf.push_prob=.1;
const intkey=JSON.stringify([nf.game_id,nf.player,nf.market_std]);
f.NFLContext={generated_at:f.NFL.generated_at,groups:{[intkey]:{offers:{'["a","Over",5.0,110,"2026-09-27T11:59:00.000Z"]':{raw_probability:.6}},calibration:{fitted_raw_range:[.1,.9]}}}};
assert.equal(collect(f,now).selected[0].forecast_health.raw_probability,.6);
f.NFL.rows[0].push_prob=null;assert(!collect(f,now).selected.some(r=>r.sport==='NFL'));

// A manageable card never fills slots with pending, stale, wait or pass reviews.
const readyRow=r=>({...r,reviewed_candidate:r,review_matches_current:true,qualitative_review:{
 offer_id:r.offer_id,forecast_id:r.forecast_id,status:'needs_information',reviewed_at:iso(now),
 countercase:'Experimental.',open_checks:[],evidence:[],assessment:{verdict:'consider',reason:'Case reviewed.',
 model_case:'Model examined.',price_case:'Exact quote examined.',context_case:'Context examined.',blocking_checks:[]}}});
const ready=collect(fixture(),now).selected.filter(r=>r.sport==='MLB').map(readyRow)[0];
assert.equal(shortlist(collect(fixture(),now).selected,now).length,0);
const large=Array.from({length:25},(_,i)=>({...ready,player:'Player '+i}));
// One sport and one market: the card stops at 2 per sport and market.
assert.equal(shortlist(large,now).length,2);
assert.equal(shortlist(large,now)[0].card_related_candidates,1);
// Variety: at most 4 per sport, and every sport with an eligible bet is represented.
const markets=['batter_hits','batter_total_bases','batter_home_runs','batter_rbis','pitcher_strikeouts','pitcher_outs'];
const mlbMany=markets.flatMap(m=>[0,1].map(i=>({...ready,market:m,player:m+i,line:1.5})));
const nhlReady=collect(fixture(),now).selected.filter(r=>r.sport==='NHL').map(readyRow)[0];
const nhlLast={...nhlReady,blend:{...nhlReady.blend,final:.5}};   // the weakest bet still gets the NHL slot
const varied=shortlist([...mlbMany,nhlLast],now);
assert.equal(varied.filter(r=>r.sport==='MLB').length,4);
assert.equal(varied.filter(r=>r.sport==='NHL').length,1);
assert.equal(varied.length,5,'caps never fill slots with ineligible or capped bets');
assert.equal(large[0].card_related_candidates,undefined);
for(const verdict of ['wait','pass'])assert.equal(shortlist([{...ready,qualitative_review:{...ready.qualitative_review,assessment:{...ready.qualitative_review.assessment,verdict}}}],now).length,0);
assert.equal(shortlist([{...ready,review_matches_current:false}],now).length,0);
assert.equal(shortlist([{...ready,qualitative_review:{...ready.qualitative_review,reviewed_at:iso(now-3*3600e3-1)}}],now).length,0);
const equivalent=[{...ready,market:'batter_hits',line:.5},{...ready,market:'batter_total_bases',line:.5},{...ready,market:'batter_hits',line:1.5}];
assert.equal(shortlist(equivalent,now).length,1);
assert.equal(equivalent.length,3,'research records remain intact');
assert.equal(shortlist([ready],now).length,1,'no daily minimum');
assert.equal(shortlist([{...ready,human_decision:'pass'}],now).length,0);
assert.equal(shortlist([{...ready,qualitative_review:null,human_decision:'select'}],now).length,1);
assert.equal(shortlist([...large,{...ready,player:'Analyst choice',market:'pitcher_outs',qualitative_review:null,human_decision:'select'}],now)[0].player,'Analyst choice');
console.log('PASS: bounded reviewed card, independent research pool, equivalent bets, review age and no forced picks.');

// Main-card ranks share units across sports; feed order/source scores cannot dominate.
const pool=collect(fixture(),now).selected.map(readyRow);
// Card order uses the blended probability in one unit across sports; feed scores never decide it.
const withFinal=(r,final)=>({...r,blend:{...r.blend,final}});
const nflCard=withFinal({...pool.find(r=>r.sport==='NFL'),forecast_health:{tier:1,raw_probability:.7},score:99999},.58);
const mlbCard=withFinal({...pool.find(r=>r.sport==='MLB'),forecast_health:{tier:1},score:-999},.64);
const nhlCard=withFinal({...pool.find(r=>r.sport==='NHL'),rank_score:9,score:-99999},.52);
assert.deepEqual(shortlist([nflCard,mlbCard,nhlCard],now).map(r=>r.sport),['MLB','NFL','NHL']);
assert.deepEqual(shortlist([nhlCard,nflCard,mlbCard],now).map(r=>r.sport),['MLB','NFL','NHL']);
assert(Math.abs(shortlist([nflCard,mlbCard,nhlCard],now)[1].card_rank_score-
 (.58*Math.log1p(.0025*1.1)+.42*Math.log1p(-.0025)))<1e-12);
const fairNormal=withFinal({...mlbCard,player:'Normal',price:100},.6);
const fairLong=withFinal({...mlbCard,player:'Longshot',price:300,market:'pitcher_outs'},.3);
assert.equal(shortlist([fairLong,fairNormal],now)[0].player,'Normal','equal EV does not promote longshot payout');
const refunded=withFinal({...mlbCard,model_push_probability:.1},.6);
assert(Math.abs(shortlist([refunded],now)[0].card_rank_score-(.54*Math.log1p(.0025*1.1)+.36*Math.log1p(-.0025)))<1e-12);
const absent={...mlbCard,player:'Unknown',market:'pitcher_outs',model_withheld:'No model',model_probability:null,blend:undefined};
assert.equal(shortlist([absent,mlbCard],now)[0].player,mlbCard.player);
assert.equal(shortlist([absent],now)[0].card_rank_score,null);
assert.equal(shortlist([{...absent,human_decision:'select'},mlbCard],now)[0].player,'Unknown');
console.log('PASS: cross-sport card order, comparable units, conservative NFL probability, pushes and missing models.');

// Market blend (Phase 3): 25% model in log-odds, exact-line market; a pick needs 1% EV
// at its price, and 3% or more is labeled high confidence.
{
  const {blend,BLEND}=require('../docs/assets/briefing-picks.js');
  assert.deepEqual([BLEND.weight,BLEND.minEV,BLEND.highEV,BLEND.minBooks],[.25,.01,.03,2]);
  // The brief's example: an SOG Under at -140, model 71%, market 52% -> about 57%, no bet.
  const sog={sport:'NHL',price:-140,final_probability:.71,push_probability:0,market_probability:.52,other_books:4};
  const b=blend(sog);
  assert(Math.abs(b.final-.5705)<5e-4,b.final);assert(b.ev<0);assert.equal(b.status,'below_threshold');
  // Thin market: fewer than two books at this exact line, counting the offered book.
  // MLB/NHL count other books, so one other book plus the offer makes two.
  assert.equal(blend({...sog,other_books:0}).status,'thin_market');
  assert.equal(blend({...sog,other_books:1}).status,'below_threshold');
  assert.equal(blend({...sog,other_books:1}).line_books,2);
  assert.equal(blend({sport:'NFL',price:110,model_prob:.6,push_prob:0,consensus_prob:.53,book_count:1}).status,'thin_market','NFL book_count already includes the offer');
  assert.equal(blend({sport:'NFL',price:110,model_prob:.6,push_prob:0,consensus_prob:.53,book_count:2}).status,'qualifies');
  assert.equal(blend({...sog,market_probability:null}).status,'thin_market');
  assert.equal(blend({...sog,final_probability:null}).status,'no_model');
  // EV refunds pushes.
  const pushed=blend({...sog,price:150,final_probability:.6,push_probability:.1,market_probability:.5});
  const fin=1/(1+Math.exp(-(.25*Math.log((.6/.9)/(1-.6/.9)))));
  assert(Math.abs(pushed.final-fin)<1e-12);assert(Math.abs(pushed.ev-.9*(fin*2.5-1))<1e-12);
  // In collect: a qualifying bet carries its blend; a thin or 50% NFL estimate is withheld.
  let g=fixture();let out=collect(g,now);
  assert(out.selected.every(r=>r.blend.status==='qualifies'&&r.blend.ev>=.03&&r.blend.confidence==='high'&&Math.abs(r.score-100*r.blend.ev)<1e-9));
  assert.match(rowHTML(out.selected.find(r=>r.sport==='MLB')),/Experimental · High confidence · /);
  g=fixture();g.NFL.rows[0].book_count=1;out=collect(g,now);
  assert(!out.selected.some(r=>r.sport==='NFL'));assert.match(out.excluded[0].exclusion_reasons[0],/Fewer than two books/);
  g=fixture();g.NFL.rows[0].model_prob=.5;out=collect(g,now);
  assert(!out.selected.some(r=>r.sport==='NFL'));assert.match(out.excluded[0].exclusion_reasons[0],/exactly 50%/);
  // Between 1% and 3%: a moderate-confidence pick, ranked after high-confidence picks.
  g=fixture();Object.assign(g.MLB.rows[0],{price:-110,model_probability:.56,other_book_probability:.52});
  out=collect(g,now);
  const moderate=out.selected.find(r=>r.sport==='MLB');
  assert.equal(moderate.blend.confidence,'moderate');assert(moderate.blend.ev>=.01&&moderate.blend.ev<.03);
  assert.equal(out.leans.length,0);
  assert.match(out.coverage.find(c=>c.sport==='MLB').message,/1 review candidate \(0 high, 1 moderate confidence\)/);
  assert.match(rowHTML(moderate),/Experimental · Moderate confidence · /);
  assert.match(rowHTML(moderate),/· Moderate confidence<\/span>/);
  assert.match(rowHTML(readyRow(moderate)),/Weighted 25% model and 75% market, .* so it is a moderate-confidence pick/);
  // On the card, a high-confidence pick precedes a moderate one even with a larger log-growth score.
  const hi=readyRow({...moderate,player:'High',market:'pitcher_outs',blend:{...moderate.blend,confidence:'high',final:.53}});
  const mo=readyRow({...moderate,player:'Moderate',blend:{...moderate.blend,final:.6}});
  assert.deepEqual(shortlist([mo,hi],now).map(r=>r.player),['High','Moderate']);
  // Below the 1% floor but positive: excluded from picks, offered as that sport's lean.
  g=fixture();Object.assign(g.MLB.rows[0],{price:-112,model_probability:.56,other_book_probability:.52});
  out=collect(g,now);
  assert(!out.selected.some(r=>r.sport==='MLB'));
  assert.deepEqual(out.leans.map(r=>r.sport),['MLB']);assert(out.leans[0].blend.ev>0&&out.leans[0].blend.ev<.01);
  assert.match(out.coverage.find(c=>c.sport==='MLB').message,/1 below the 1% market-blend floor/);
  // A worse price for an outcome already picked is not a lean.
  g=fixture();g.MLB.rows.push({...g.MLB.rows[0],book:'worse',price:-112,model_probability:.56,other_book_probability:.52});
  Object.assign(g.MLB.rows[0],{model_probability:.56,other_book_probability:.52});
  out=collect(g,now);
  assert.equal(out.selected.filter(r=>r.sport==='MLB').length,1);assert.equal(out.leans.length,0);
  // Negative after the blend: no lean.
  g=fixture();Object.assign(g.MLB.rows[0],{price:-130,model_probability:.56,other_book_probability:.52});
  assert.equal(collect(g,now).leans.length,0);
  // Tickets record what the pick was decided on, at the price actually taken.
  const pick=collect(fixture(),now).selected.find(r=>r.sport==='MLB');
  const t=ticketData(pick,105,10);
  assert.equal(t.market_prob,.52);assert.equal(t.blend_weight,.25);assert(Math.abs(t.final_prob-pick.blend.final)<1e-12);
  assert(Math.abs(t.expected_value-(pick.blend.final*2.05-1))<1e-12);assert.equal(t.decision_at,pick.forecast_at);
  assert.match(rowHTML(pick),/Blended \d+\.\d% · \+\d+\.\d% expected value/);
  const research=ticketData({...pick,blend:undefined},110,10);
  assert.equal(research.final_prob,undefined);assert.equal(research.decision_at,undefined,'research tickets keep the existing ledger contract');
}
console.log('PASS: market blend, thin markets, 50% calibrations, leans, coverage and ticket fields.');
// Editions keep the exact published assessment after prices expire or games start.
{
  const {editionRows}=require('../docs/assets/briefing-picks.js');
  const row={...sample,commence_time:iso(now+3600e3),quoted_at:iso(now-60e3),human_decision:'unreviewed',review_matches_current:true,
    qualitative_review:{...sample.qualitative_review,reviewed_at:iso(now),assessment:{verdict:'consider',reason:'Case',model_case:'Model',price_case:'Price',context_case:'Context',blocking_checks:[]}}};
  const card={schema_version:1,kind:'morning',decision_date:day(now),published_at:iso(now),rows:[row]};
  assert.equal(editionRows(card,now).length,1);
  assert.equal(editionRows(card,now+24*3600e3).length,1);
  assert.equal(editionRows(card,now-1).length,0,'future editions unavailable');
  assert.equal(editionRows({...card,rows:[null,{...row,commence_time:'bad'}]},now).length,0);
  assert.equal(editionRows({...card,rows:[{...row,review_matches_current:false}]},now).length,0);
  assert.equal(editionRows({...card,rows:[{...row,qualitative_review:{...row.qualitative_review,reviewed_at:iso(now-4*3600e3)}}]},now).length,0);
  assert.equal(editionRows({...card,rows:[{...row,human_decision:'pass'}]},now).length,0);
  assert.equal(editionRows({...card,rows:[{...row,qualitative_review:{...row.qualitative_review,assessment:{...row.qualitative_review.assessment,verdict:'wait'}}}]},now).length,0);
}
console.log('PASS: immutable edition, review-at-publication validity, future/invalid dates and started games.');
// The home page features the first card pick whose game has not started.
{
  const {featuredPick,featuredHTML}=require('../docs/assets/briefing-picks.js');
  const review={...sample.qualitative_review,reviewed_at:iso(now),countercase:'Carries could rise.',
    assessment:{verdict:'consider',reason:'Projection below the line.',model_case:'Model',price_case:'Price',context_case:'Context',blocking_checks:[]}};
  const row=o=>({...sample,commence_time:iso(now+3600e3),quoted_at:iso(now-60e3),human_decision:'unreviewed',review_matches_current:true,qualitative_review:review,...o});
  const first=row({player:'First <b>Player</b>'}),second=row({player:'Second Player',commence_time:iso(now+7200e3)});
  const card={schema_version:1,kind:'morning',decision_date:day(now),published_at:iso(now),status:'published',rows:[first,second]};
  assert.equal(featuredPick(card,now).player,'First <b>Player</b>');
  assert.equal(featuredPick(card,now+3600e3+1).player,'Second Player','a started game passes the slot to the next pick');
  const html=featuredHTML(card,now);
  assert.match(html,/First &lt;b&gt;Player&lt;\/b&gt;/);assert.doesNotMatch(html,/<b>Player/);
  assert.match(html,/Break-even/);assert.match(html,/Consider:<\/strong> Projection below the line\./);
  assert.match(html,/The case against:<\/strong> Carries could rise\./);assert.match(html,/\/research\/daily-process\.html/);
  assert.match(featuredHTML({...card,rows:[],status:'no_reviewed_candidates'},now),/No pick cleared today’s review/);
  assert.match(featuredHTML({...card,rows:[],status:'research_incomplete'},now),/Morning research did not finish/);
  const later=now+7200e3+1;
  assert.equal(featuredPick(card,later).player,'First <b>Player</b>','once every game has started the day’s first pick stays up');
  assert.match(featuredHTML(card,later),/Game started · historical assessment\./);assert.doesNotMatch(featuredHTML(card,later),/Confirm the current line/);
  assert.doesNotMatch(featuredHTML(card,now),/Game started/);
  assert.match(featuredHTML({...card,decision_date:'2000-01-01'},now),/not published yet/);
  assert.match(featuredHTML(null,now),/not published yet/);
}
console.log('PASS: featured pick follows the published card, keeps the day’s pick once games start and says why when empty.');

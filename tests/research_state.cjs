/* Research status separates eligibility, verification, direction, material questions and completion. */
const assert=require('node:assert/strict');
const {collect,shortlist,researchState,rowHTML,researchStatus,lateFor,day}=require('../docs/assets/briefing-picks.js');
const {fixture}=require('./briefing_picks.cjs');
const now=Date.parse('2026-09-27T12:00:00Z'),iso=t=>new Date(t).toISOString();

function reviewedFeeds(evidence,verdict='consider',changes={}) {
  const f=fixture(),selected=collect(f,now).selected,mlb=selected.find(r=>r.sport==='MLB');
  const source={source_id:'s1',url:'https://www.mlb.com/news/synthetic',title:'Lineup report',published_at:iso(now-3600e3),
    retrieved_at:iso(now-600e3),candidate_ids:['c1']};
  const review={candidate_id:'c1',status:evidence.some(e=>e.direction==='supports')?'research_support':evidence.some(e=>e.direction==='concern')?'concern':'needs_information',
    assessment:{verdict,reason:'Reason.',model_case:'Model.',price_case:'Price.',context_case:'Context.',blocking_checks:verdict==='wait'?['Confirm the starter.']:[]},
    countercase:'Countercase.',open_checks:['Reconfirm price.'],evidence,reviewed_at:iso(now-600e3),offer_id:'o1',forecast_id:'f1',prompt_version:'sports-research-7',...changes};
  const {reviewKey,reviewBetKey}=require('../docs/assets/briefing-picks.js');
  f.Reviews={schema_version:1,sports:{MLB:{decision_date:day(now),sources:[source],candidates:[{...mlb,candidate_id:'c1',offer_id:'o1',forecast_id:'f1',
    review_key:reviewKey(mlb),review_bet_key:reviewBetKey(mlb),qualitative_review:review}]}}};
  return f;
}
const item=(direction,extra={})=>({source_id:'s1',excerpt:'Synthetic Player bats second tonight',interpretation:'Expected role.',kind:'other',direction,
  represented_in:'neither',assumption:'lineup_slot',materiality:'consequential',verification:'official',effect:'scenario',applies_to:'this_game',...extra});
const stateOf=f=>{const r=collect(f,now).selected.find(r=>r.sport==='MLB');return {r,s:researchState(r,now)};};

// Empty evidence + consider: eligible, but never presented as corroborated.
let {r,s}=stateOf(reviewedFeeds([]));
assert.equal(s.gate,'model_case_only');assert.equal(s.evidence,'none_verified');assert.equal(s.direction,'none');
assert.equal(r.review,'Consider · model case only');
assert(shortlist(collect(reviewedFeeds([]),now).selected,now).some(x=>x.sport==='MLB'),'model-only consider can qualify');
assert.match(rowHTML({...r,card_snapshot_at:iso(now)}),/not independent corroboration/);
// Verified support.
({r,s}=stateOf(reviewedFeeds([item('supports')])));
assert.equal(s.gate,'verified_context');assert.equal(s.direction,'supports');assert.match(s.label,/verified context · supporting/);
// A verified, consequential adverse fact produces a pass even with a consider verdict.
let f=reviewedFeeds([item('concern')]);({r,s}=stateOf(f));
assert.equal(s.gate,'adverse_fact');assert.equal(r.review,'Pass · verified adverse fact');
assert(!shortlist(collect(f,now).selected,now).some(x=>x.sport==='MLB'),'adverse fact excluded from the card');
// Opinion cannot establish the fact; a minor concern does not block.
assert.equal(stateOf(reviewedFeeds([item('concern',{verification:'opinion'})])).s.gate,'verified_context');
assert.equal(stateOf(reviewedFeeds([item('concern',{materiality:'minor'})])).s.gate,'verified_context');
// Unresolved or conflicting consequential facts wait.
assert.equal(stateOf(reviewedFeeds([item('context',{effect:'unresolved'})])).s.gate,'material_question');
assert.equal(stateOf(reviewedFeeds([item('context',{verification:'conflicting'})])).s.gate,'material_question');
assert.equal(stateOf(reviewedFeeds([item('supports'),item('concern',{source_id:'s1',excerpt:'Synthetic Player may sit tonight'})])).s.gate,'material_question');
f=reviewedFeeds([item('context',{effect:'unresolved'})]);
assert(!shortlist(collect(f,now).selected,now).some(x=>x.sport==='MLB'));
// General-context facts never gate this event; legacy unclassified evidence keeps the verdict.
assert.equal(stateOf(reviewedFeeds([item('concern',{applies_to:'general_context'})])).s.gate,'verified_context');
const legacy=item('concern');for(const k of ['assumption','materiality','verification','effect','applies_to'])delete legacy[k];
assert.equal(stateOf(reviewedFeeds([legacy],'consider',{prompt_version:'sports-research-6'})).s.gate,'verified_context');
// Wait and pass verdicts.
assert.equal(stateOf(reviewedFeeds([],'wait')).s.gate,'material_question');
assert.equal(stateOf(reviewedFeeds([],'pass')).s.gate,'pass');
// Stale (over three hours) is not eligible.
f=reviewedFeeds([],'consider',{reviewed_at:iso(now-4*3600e3)});
assert.equal(stateOf(f).s.gate,'stale');
assert(!shortlist(collect(f,now).selected,now).some(x=>x.sport==='MLB'));
// A failed attempt is shown as a failure, not an empty successful review.
f=reviewedFeeds([]);const c=f.Reviews.sports.MLB.candidates[0];delete c.qualitative_review;
c.research_failure={category:'numeric_confidence',stage:'validation'};
({r,s}=stateOf(f));
assert.equal(s.gate,'failed');assert.equal(r.review,'Research failed · analysis rejected by validation');
assert.match(researchStatus('MLB',f.Reviews,collect(f,now).selected,now),/1\/1 candidate research attempts failed/);
assert(!shortlist(collect(f,now).selected,now).some(x=>x.sport==='MLB'));
// Later reassessments attach by identity and never alter the edition row.
const row={...collect(fixture(),now).selected.find(r=>r.sport==='MLB')};
const index={schema_version:1,decision_date:day(now),reassessments:[{identity:{sport:'MLB',game_id:row.game_id,player:row.player,market:row.market,side:row.side,line:row.line,book:row.book},
  version:2,reassessed_at:iso(now),label:'Waiting on a material fact',current_price:120}]};
assert.equal(lateFor(row,index).version,2);
assert.equal(lateFor({...row,line:3.5},index),null);
const html=rowHTML({...row,card_snapshot_at:iso(now-3600e3),late_reassessment:lateFor(row,index)});
assert.match(html,/Later reassessment v2/);assert.match(html,/morning assessment above is unchanged/);
console.log('PASS: research state separates verification, direction, material facts, failures and later reassessments.');
// NFL rows kept in Top Picks by owner policy are eligible but labelled as not validated.
{
  const {fixture}=require('./briefing_picks.cjs');
  const f=fixture();f.NFL.rows[0].model_status='Calibration fitted for a model version before the forecast-cutoff fixes; not validated for the current model';
  const nflRow=collect(f,now).selected.find(r=>r.sport==='NFL');
  assert(nflRow,'NFL stays eligible under the owner policy');
  assert.match(rowHTML(nflRow),/Calibration not validated for this model/);
  f.NFL.rows[0].model_status='Incompatible calibration (fitted for x); not validated for y';
  assert(!collect(f,now).selected.some(r=>r.sport==='NFL'),'strict policy excludes NFL');
  console.log('PASS: NFL owner policy keeps eligibility with an explicit not-validated label.');
}

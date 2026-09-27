// Exercise the actual shared save helper without auth emails or database writes.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const script=fs.readFileSync('docs/tracking/bet-tracking.js','utf8');
function setup(){
  const state={user:{id:'owner'},rows:[],error:null,existing:null,filters:[]};
  const client={auth:{getSession:async()=>({data:{session:state.user?{user:state.user}:null}})},from:()=>({
    insert:async row=>{state.rows.push(JSON.parse(JSON.stringify(row)));return {error:state.error};},
    select:()=>{const query={eq:(k,v)=>{state.filters.push([k,v]);return query;},maybeSingle:async()=>({data:state.existing,error:null})};return query;}
  })};
  const window={supabase:{createClient:()=>client}},ctx={window,console,alert:()=>{},confirm:()=>false};
  vm.runInNewContext(script,ctx);return {save:window.saveTrackedBet,summary:window.betTrackerSummary,state};
}
(async()=>{
 const {save,summary,state}=setup(),bet={id:'b7c9ba55-1234-4234-8234-123456789abc',user_id:'intruder',league:'MLB',game_date:'2026-09-26',
   team_home:'Home',team_away:'Away',market_type:'h2h',side:'Home',line:null,book:'book',odds:146,stake_dollars:25,model_prob:0,edge_bps:0};
 state.user=null;assert.equal((await save(bet)).needsSignIn,true);assert.equal(state.rows.length,0);
 state.user={id:'owner'};assert.equal((await save({...bet,odds:50})).ok,false);assert.equal(state.rows.length,0);
 assert.equal((await save(bet)).ok,true);assert.equal(state.rows[0].user_id,'owner');assert.equal(state.rows[0].line,null);
 assert.equal(state.rows[0].model_prob,0);assert.equal(state.rows[0].edge_bps,0);assert.equal(state.rows[0].status,'pending');
 state.existing=state.rows[0];state.error={code:'23505'};assert.equal((await save(bet)).ok,true,'retry recognizes same saved ticket');
 assert(state.filters.some(([k,v])=>k==='user_id'&&v==='owner'));
 assert.equal((await save({...bet,stake_dollars:30})).ok,false,'duplicate ID must not overwrite an earlier stake');
 state.existing=null;assert.equal((await save(bet)).ok,false,'no owner-visible saved row means no success');
 state.error={code:'XX000'};assert.equal((await save(bet)).ok,false);
 state.error=null;assert.equal((await save({...bet,line:0})).ok,true);assert.equal(state.rows.at(-1).line,0);
 const stats=summary([{status:'pending',stake_dollars:100},{status:'won',stake_dollars:10,payout:20},{status:'lost',stake_dollars:5,payout:0}]);
 assert.equal(stats.totalStaked,115);assert.equal(stats.profitLoss,5);assert(Math.abs(stats.roi-100/3)<1e-9);
 assert.equal(summary([{status:'pending',stake_dollars:100}]).profitLoss,0);
 console.log('PASS: tracker auth, account ownership, nullable moneylines, validation, failed saves and duplicate protection.');
})().catch(e=>{console.error(e);process.exitCode=1;});

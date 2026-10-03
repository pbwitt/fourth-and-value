// Exercise the actual shared save helper without auth emails or database writes.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const script=fs.readFileSync('docs/tracking/bet-tracking.js','utf8');
function setup(){
  const state={user:{id:'owner'},rows:[],error:null,existing:null,filters:[]};
  const client={auth:{getSession:async()=>({data:{session:state.user?{user:state.user}:null}})},from:()=>({
    insert:async row=>{state.rows.push(JSON.parse(JSON.stringify(row)));
      if(state.noTeamColumn&&'player_team' in row)return {error:{code:'PGRST204',message:"Could not find the 'player_team' column of 'bets' in the schema cache"}};
      return {error:state.error};},
    select:()=>{const query={eq:(k,v)=>{state.filters.push([k,v]);return query;},maybeSingle:async()=>({data:state.existing,error:null})};return query;}
  })};
  const window={supabase:{createClient:()=>client}},ctx={window,console,alert:()=>{},confirm:()=>false};
  vm.runInNewContext(script,ctx);return {save:window.saveTrackedBet,summary:window.betTrackerSummary,dayTotals:window.betDayTotals,state};
}
(async()=>{
 const {save,summary,dayTotals,state}=setup(),bet={id:'b7c9ba55-1234-4234-8234-123456789abc',user_id:'intruder',league:'MLB',game_date:'2026-09-26',
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
 // The player's team is saved with player bets, and a database without the column still saves the bet.
 const {id:_,...unsaved}=bet,prop={...unsaved,player:'Bryce Harper',market_type:'batter_hits',side:'over',line:.5,player_team:'PHI'};
 assert.equal((await save(prop)).ok,true);assert.equal(state.rows.at(-1).player_team,'PHI');
 assert.equal((await save({...bet,player_team:'PHI'})).ok,true);assert.equal('player_team' in state.rows.at(-1),false,'game markets have no player team');
 state.noTeamColumn=true;const before=state.rows.length;
 assert.equal((await save(prop)).ok,true,'saved without the team before the migration');
 assert.equal(state.rows.length,before+2);assert.equal('player_team' in state.rows.at(-1),false);
 state.noTeamColumn=false;
 const stats=summary([{status:'pending',stake_dollars:100},{status:'won',stake_dollars:10,payout:20},{status:'lost',stake_dollars:5,payout:0}]);
 assert.equal(stats.totalStaked,115);assert.equal(stats.profitLoss,5);assert(Math.abs(stats.roi-100/3)<1e-9);
 assert.equal(summary([{status:'pending',stake_dollars:100}]).profitLoss,0);
 // Daily totals: total staked, graded payout on settled bets, and pending upside.
 const day=dayTotals([{status:'won',stake_dollars:10,payout:19.09,odds:-110},{status:'lost',stake_dollars:20,payout:0,odds:120},
   {status:'push',stake_dollars:5,payout:5,odds:-105},{status:'pending',stake_dollars:25,odds:150},{status:'pending',stake_dollars:10,odds:null}]);
 assert.deepEqual([day.bets,day.staked,day.won,day.lost,day.push,day.pending,day.pendingStaked],[5,70,1,1,1,2,35]);
 assert.equal(day.payout,24.09);assert.equal(Math.round(day.profitLoss*100)/100,-10.91);assert.equal(day.pendingPotential,62.5);
 console.log('PASS: tracker auth, account ownership, nullable moneylines, validation, failed saves, duplicate protection and player teams.');
})().catch(e=>{console.error(e);process.exitCode=1;});

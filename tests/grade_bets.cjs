// Automatic bet grading: settles only final games after the correction window,
// pays out like the old graders, and leaves anything uncertain pending.
const assert=require('node:assert/strict');
const {gradeBet,gradeAll,backfillTeams,payout}=require('../scripts/grade_bets.cjs');

const start='2026-09-29T23:00:00Z',t0=Date.parse(start),H=3600e3;
const team=(abbrev,common,place,score)=>({abbrev,commonName:{default:common},name:{default:common},placeName:{default:place},score});
const nhlGame=(state,o={})=>({id:2026020002,startTimeUTC:start,gameState:state,gameScheduleState:'OK',period:3,
  periodDescriptor:{number:3,periodType:'REG'},clock:{timeRemaining:'00:00',secondsRemaining:0},gameOutcome:{lastPeriodType:'REG'},
  awayTeam:team('MTL','Canadiens','Montréal',2),homeTeam:team('TOR','Maple Leafs','Toronto',4),...o});
const nhlBox=state=>({...nhlGame(state),playerByGameStats:{
  homeTeam:{forwards:[{name:{default:'A. Matthews'},goals:1,assists:1,points:2,sog:3,hits:0,blockedShots:0,pim:0,powerPlayGoals:0}],defense:[],goalies:[]},
  awayTeam:{forwards:[{name:{default:'N. Suzuki'},goals:1,assists:0,points:1,sog:2,hits:0,blockedShots:0,pim:0,powerPlayGoals:0}],defense:[],goalies:[]}}});

function source(state='OFF'){
  const calls=[];
  return {calls,
    proxy:async body=>{calls.push(body);return body.game!=null?nhlBox(state):{games:[nhlGame(state)]};},
    fetchJSON:async url=>{calls.push(url);throw Error('down');}};
}
const bet=o=>({id:'x',league:'NHL',game_date:'2026-09-29',team_home:'Toronto Maple Leafs',team_away:'Montreal Canadiens',
  player:'Auston Matthews',market_type:'sog',side:'over',line:2.5,odds:-115,stake_dollars:25,...o});

(async()=>{
  // Payouts include the stake, matching the existing graders and stats cards.
  assert.equal(payout(25,-115,'won'),46.74);assert.equal(payout(25,150,'won'),62.5);
  assert.equal(payout(25,-115,'push'),25);assert.equal(payout(25,-115,'lost'),0);

  const later=t0+5*H;
  let [r]=await gradeAll({bets:[bet()],...source(),now:later});
  assert.deepEqual(r.update,{status:'won',actual_result:3,payout:46.74,graded_timestamp:new Date(later).toISOString(),player_team:'TOR'});

  // The player's team comes from the box score side; a saved team and game markets are left alone.
  [r]=await gradeAll({bets:[bet({player:'Nick Suzuki',side:'under'})],...source(),now:later});assert.equal(r.update.player_team,'MTL');
  [r]=await gradeAll({bets:[bet({player_team:'TOR'})],...source(),now:later});assert.equal('player_team' in r.update,false);
  [r]=await gradeAll({bets:[bet({player:null,market_type:'h2h',side:'TOR',line:null})],...source(),now:later});assert.equal('player_team' in r.update,false);

  [r]=await gradeAll({bets:[bet({line:3})],...source(),now:later});assert.equal(r.update.status,'push');assert.equal(r.update.payout,25);
  [r]=await gradeAll({bets:[bet({side:'under'})],...source(),now:later});assert.equal(r.update.status,'lost');assert.equal(r.update.payout,0);
  [r]=await gradeAll({bets:[bet({market_type:'player_points',side:'Over',line:1.5,odds:120})],...source(),now:later});
  assert.deepEqual([r.update.status,r.update.payout],['won',55]);

  // Game markets: NHL team_total is the game total; moneyline/spread record the margin.
  [r]=await gradeAll({bets:[bet({player:null,market_type:'team_total',side:'over',line:5.5})],...source(),now:later});
  assert.deepEqual([r.update.status,r.update.actual_result],['won',6]);
  [r]=await gradeAll({bets:[bet({player:null,market_type:'h2h',side:'Montreal Canadiens',line:null})],...source(),now:later});
  assert.deepEqual([r.update.status,r.update.actual_result],['lost',-2]);
  [r]=await gradeAll({bets:[bet({player:null,market_type:'spreads',side:'TOR',line:-1.5})],...source(),now:later});
  assert.equal(r.update.status,'won');

  // Never settle early or on uncertainty.
  [r]=await gradeAll({bets:[bet()],...source(),now:t0+3*H});assert.equal(r.skip,'waiting for corrections');
  [r]=await gradeAll({bets:[bet()],...source('LIVE'),now:later});assert.equal(r.skip,'not final');
  [r]=await gradeAll({bets:[bet({player:'Mitch Marner'})],...source(),now:later});assert.equal(r.skip,'player not found');
  [r]=await gradeAll({bets:[bet({market_type:'faceoffs_won'})],...source(),now:later});assert.equal(r.skip,'unsupported market');
  [r]=await gradeAll({bets:[bet({team_home:'Boston Bruins',team_away:'New York Rangers'})],...source(),now:later});assert.equal(r.skip,'game not found');
  [r]=await gradeAll({bets:[bet({odds:50})],...source(),now:later});assert.equal(r.skip,'invalid odds');
  [r]=await gradeAll({bets:[bet({game_date:'2026-10-05'})],...source(),now:later});assert.equal(r.skip,'not played yet');
  [r]=await gradeAll({bets:[bet({game_date:'2026-09-01'})],...source(),now:later});assert.equal(r.skip,'older than lookback');
  [r]=await gradeAll({bets:[bet({league:'CFB'})],...source(),now:later});assert.equal(r.skip,'league not supported');
  const ppd=source();ppd.proxy=async()=>({games:[nhlGame('FUT',{gameScheduleState:'PPD'})]});
  [r]=await gradeAll({bets:[bet()],...ppd,now:later});assert.equal(r.skip,'postponed');
  assert.equal(gradeBet(bet(),null,null,later).skip,'game not found');

  // A failed feed leaves bets pending; shared requests are made once per game and date.
  [r]=await gradeAll({bets:[bet({league:'MLB',team_home:'NYY',team_away:'BOS'})],...source(),now:later});assert.equal(r.skip,'feed unavailable');
  const s=source();const many=await gradeAll({bets:[bet({id:1}),bet({id:2,market_type:'points'}),bet({id:3,player:null,market_type:'h2h',side:'TOR'})],...s,now:later});
  assert(many.every(x=>x.update));assert.equal(s.calls.length,2,'one scoreboard and one box score');

  // Backfill: settled bets of any age get the team from the final box score; uncertainty adds nothing.
  const b=source();
  const filled=await backfillTeams({bets:[bet({id:1,game_date:'2025-01-02'}),bet({id:2,player:'Nick Suzuki',game_date:'2025-01-02'}),
    bet({id:3,player:'Mitch Marner',game_date:'2025-01-02'}),bet({id:4,league:'CFB'}),bet({id:5,player:null})],...b});
  assert.deepEqual(filled.map(x=>x.team||x.skip),['TOR','MTL','player not found','league not supported','no player or date']);
  assert.equal(b.calls.length,2,'one scoreboard and one box score');
  [r]=await backfillTeams({bets:[bet()],...source('LIVE')});assert.equal(r.skip,'not final');
  [r]=await backfillTeams({bets:[bet({team_home:'Boston Bruins',team_away:'New York Rangers'})],...source()});assert.equal(r.skip,'game not found');
  [r]=await backfillTeams({bets:[bet({league:'MLB',team_home:'NYY',team_away:'BOS'})],...source()});assert.equal(r.skip,'feed unavailable');

  console.log('PASS: automatic grading settles final games once, pays like the ledger, and leaves uncertainty pending; player teams come from the box score.');
})().catch(e=>{console.error(e);process.exitCode=1;});

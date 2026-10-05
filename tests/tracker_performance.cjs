// Bet Tracker performance panel: win rate against break-even, ROI against winning bettors,
// running profit and results by sport, from the bets the filters show.
const assert=require('node:assert/strict');
const {summarize,html,PRO_LOW,PRO_HIGH}=require('../docs/tracking/performance.js');
const bet=(game_date,league,status,odds,payout,extra={})=>({game_date,league,status,odds,stake_dollars:5,payout,created_at:game_date+'T12:00:00Z',...extra});
const bets=[
  bet('2026-10-02','NHL','won',-140,8.57,{player:'Skater <b>One</b>',market_type:'sog',side:'under',line:2.5}),
  bet('2026-10-01','NFL','won',125,11.25,{created_at:'2026-10-01T09:00:00Z'}),
  bet('2026-10-01','NFL','lost',175,0,{created_at:'2026-10-01T10:00:00Z'}),
  bet('2026-10-03','MLB','push',-110,5),
  bet('2026-10-03','MLB','lost',-120,0),
  bet('2026-10-04','NFL','pending',155,null),
  bet('2026-10-04','NHL','void',-110,null),
];
const s=summarize(bets);
assert.equal(s.settled,5);assert.equal(s.won,2);assert.equal(s.lost,2);assert.equal(s.push,1);assert.equal(s.pending,1,'void is neither settled nor pending');
assert.equal(s.winRate,.5,'pushes are not decided bets');
assert.equal(s.staked,25);assert.ok(Math.abs(s.pl-(3.57+6.25-5-5))<1e-9);assert.ok(Math.abs(s.roi-s.pl/25)<1e-12);
const implied=o=>o>0?100/(o+100):-o/(-o+100);
assert.ok(Math.abs(s.breakEven-[ -140,125,175,-120].map(implied).reduce((a,b)=>a+b)/4)<1e-12,'break-even from the prices of decided bets');
assert.deepEqual(s.path.map(p=>p.bet.league+p.bet.status),['NFLwon','NFLlost','NHLwon','MLBpush','MLBlost'],'order placed: date, then time logged');
assert.ok(Math.abs(s.path[s.path.length-1].total-s.pl)<1e-9);
assert.deepEqual(s.sports.map(t=>t.sport),['NHL','NFL','MLB'],'largest profit first');
assert.equal(s.sports.find(t=>t.sport==='NFL').pending,1);
assert.equal(s.sports.find(t=>t.sport==='MLB').winRate,0);
assert.deepEqual([PRO_LOW,PRO_HIGH],[.02,.05]);

const page=html(s);
assert.match(page,/Bets won/);assert.match(page,/Break-even win rate at the prices you took/);
assert.match(page,/50%<\/b><span>Bets won/);assert.match(page,/ROI against winning bettors/);assert.match(page,/\+2% to \+5% ROI/);
assert.match(page,/<th scope="row">NFL<\/th><td>50% <small>\+1 pending<\/small><\/td>/,'win percentage, not a record');
assert.doesNotMatch(page,/\b\d+[–-]\d+\b(?!%)/,'no win-loss records');
assert.doesNotMatch(page,/luck|standard error|±/i,'no luck range');
assert.doesNotMatch(page,/NaN|undefined|null/);
assert.match(html(summarize([bets[0]])),/appear once two bets in this view have settled/);
assert.match(html(summarize([])),/appear once two bets/);
console.log('PASS: tracker performance: win rate vs break-even, ROI vs winning bettors, running profit order, pushes, pending, by sport, no records or luck range.');

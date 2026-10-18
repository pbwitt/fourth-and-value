// Bet Tracker performance panel: win rate against break-even, ROI against winning bettors,
// running profit and results by sport and odds range, from the bets the filters show.
const assert=require('node:assert/strict');
const {summarize,html,context,scaleScene,pathScene,toSVG,oddsRange,ODDS_RANGES,PRO_LOW,PRO_HIGH}=require('../docs/tracking/performance.js');
globalThis.FVManualBet=require('../docs/tracking/manual-bet.js'); // the page loads it for market names
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
// Break-even is a guide, not a profit line: one −200 and one +200 bet average 50%, and going
// 1–1 makes money only if the underdog wins.
const pair=(fav,dog)=>summarize([bet('2026-10-01','NFL',fav,-200,fav==='won'?7.5:0),bet('2026-10-01','NFL',dog,200,dog==='won'?15:0)]);
for(const [fav,dog,pl] of [['won','lost',-2.5],['lost','won',5]]){
  const t=pair(fav,dog);assert.equal(t.breakEven,.5);assert.equal(t.winRate,.5);assert.ok(Math.abs(t.pl-pl)<1e-9,`${fav} favorite: ${t.pl}`);
}
assert.deepEqual(s.path.map(p=>p.bet.league+p.bet.status),['NFLwon','NFLlost','NHLwon','MLBpush','MLBlost'],'order placed: date, then time logged');
assert.ok(Math.abs(s.path[s.path.length-1].total-s.pl)<1e-9);
assert.deepEqual(s.sports.map(t=>t.sport),['NHL','NFL','MLB'],'largest profit first');
assert.equal(s.sports.find(t=>t.sport==='NFL').pending,1);
assert.equal(s.sports.find(t=>t.sport==='MLB').winRate,0);
assert.deepEqual([PRO_LOW,PRO_HIGH],[.02,.05]);

const page=html(s);
assert.doesNotMatch(page,/Bets won|ROI on \$|perf-facts|break-even was/i,'win rate, ROI and break-even live in the tiles above, not repeated here');
assert.match(page,/ROI against winning bettors/);assert.match(page,/\+2% to \+5% ROI/);
assert.deepEqual([...page.matchAll(/data-perf-share="(\w+)"/g)].map(m=>m[1]),['summary','roi','path','sports','odds'],'the summary and each part can be shared');
assert.match(page,/<th scope="row">NFL<\/th><td>50% <small>\+1 pending<\/small><\/td>/,'win percentage, not a record');
assert.doesNotMatch(page.replace(/<svg[\s\S]*?<\/svg>/g,''),/\b\d+[–-]\d+\b(?!%)/,'no win-loss records (icon path data aside)');
assert.doesNotMatch(page,/luck|standard error|±/i,'no luck range');
assert.doesNotMatch(page,/NaN|undefined|null/);
assert.match(html(summarize([bets[0]])),/appear once two bets in this view have settled/);
assert.match(html(summarize([])),/appear once two bets/);
// Odds ranges: the filter and the By odds table share one definition. Even money counts with
// plus money; a missing price, or one between −100 and +100, has no range.
assert.deepEqual(ODDS_RANGES.map(r=>r.label),['−150 or shorter','−101 to −149','Even to +199','+200 or longer']);
for(const [odds,key] of [[-400,'fav'],[-150,'fav'],[-149.5,'short'],[-149,'short'],[-101,'short'],[-100,'even'],[100,'even'],[199,'even'],
  [200,'long'],[2800,'long'],['-135','short'],['310','long'],[null,null],[undefined,null],['',null],[0,null],[-99,null],[50,null],['abc',null]])
  assert.equal(oddsRange(odds),key,`odds ${odds}`);
assert.deepEqual(s.odds.map(t=>[t.range,t.settled,t.pending]),[['short',3,0],['even',2,1]],'shortest prices first; void and empty ranges left out');
const short=s.odds[0];
assert.equal(short.winRate,.5);assert.ok(Math.abs(short.breakEven-(implied(-140)+implied(-120))/2)<1e-12,'pushes are left out of break-even');
assert.equal(short.closed,0);assert.equal(short.beatClose,null);
assert.match(page,/<table class="perf-table perf-odds" aria-label="Results by odds">/);
assert.match(page,/<th scope="row">−101\u00a0to −149 <small>3\u00a0settled<\/small><\/th><td>50% <small>needs 56\.4%<\/small><\/td><td class="r negative">−\$1\.43 <small>−9\.5% ROI<\/small><\/td><td class="r">—<\/td><\/tr>/);
assert.match(page,/<th scope="row">Even\u00a0to \+199 <small>2\u00a0settled, 1\u00a0pending<\/small><\/th>/);
assert.match(page,/fewer than 30 settled bets/);
// Beat the close: a price better than the fair close, or the line closing past the number when
// it moved. Pending bets count (closing prices come before the game); void bets and bets with
// no closing data do not. Supabase may send numeric columns as text.
const closing=summarize([
  bet('2026-10-05','NFL','won',125,11.25,{clv_ev:'0.04'}),
  bet('2026-10-05','NFL','lost',175,0,{clv_ev:-0.02,beat_line:null}),
  bet('2026-10-06','NFL','pending',155,null,{clv_ev:null,beat_line:true}),
  bet('2026-10-06','NFL','lost',150,0,{clv_ev:null,beat_line:null}),
  bet('2026-10-06','NFL','void',140,null,{clv_ev:.1}),
  bet('2026-10-06','NHL','lost',-200,0,{clv_ev:0}),
]);
const even=closing.odds.find(t=>t.range==='even');
assert.equal(even.closed,3);assert.ok(Math.abs(even.beatClose-2/3)<1e-12);
assert.equal(closing.odds.find(t=>t.range==='fav').beatClose,0,'matching the close is not beating it');
assert.equal(closing.closed,4);assert.equal(closing.beatClose,.5);
assert.match(html(closing),/<td class="r">66\.7% <small>of 3<\/small><\/td><\/tr>/);
assert.doesNotMatch(html(summarize(bets.filter(b=>b.odds===155))),/By odds/,'no table before two bets settle');
assert.doesNotMatch(html(closing),/NaN|undefined|null/);

// Share images say what they cover: the sport when only one is shown, the dates, the count.
assert.equal(context(s),'Oct 1 – Oct 3 · 5 settled bets');
assert.equal(context(summarize(bets.filter(b=>b.league==='NFL'))),'NFL · Oct 1 · 2 settled bets');

// One layout feeds the page SVG and the share image; each hover area sits just before its dot.
const path=toSVG(pathScene(s,600));
assert.equal((path.match(/<circle class="perf-dot (won|lost|push)"/g)||[]).length,5);
assert.equal((path.match(/<rect class="perf-hit" tabindex="0"[^>]*\/><circle class="perf-dot/g)||[]).length,5,'hover areas precede their dots');
assert.match(path,/data-readout="Oct 2 · NHL · Skater &lt;b&gt;One&lt;\/b&gt; Shots on goal Under 2\.5 · Won \+\$3\.57/,'readouts are escaped');
assert.match(path,/<polyline class="perf-line" points="/);
const scale=toSVG(scaleScene(s,600));
assert.match(scale,/class="perf-pro"/);assert.match(scale,/class="perf-even"/);assert.match(scale,/You −0\.7%<\/text>/);
assert.match(toSVG(scaleScene(s,360)),/>Winning bettors<\/text>/,'short label on phones');
assert.ok(pathScene(s,400).shapes.every(([kind,,...g])=>kind==='poly'?g[0].flat().every(Number.isFinite):g.filter(v=>typeof v==='number').every(Number.isFinite)),'finite geometry for the canvas');
console.log('PASS: tracker performance: ROI vs winning bettors, running profit order, pushes, pending, by sport, by odds range and beating the close, share buttons and image context, no repeated tiles, records or luck range.');

// Bet Tracker performance panel: win rate against break-even, ROI against winning bettors,
// running profit and results by sport, from the bets the filters show.
const assert=require('node:assert/strict');
const {summarize,html,context,scaleScene,pathScene,toSVG,PRO_LOW,PRO_HIGH}=require('../docs/tracking/performance.js');
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
assert.doesNotMatch(page,/Bets won|ROI on \$|perf-facts|break-even was/i,'win rate, ROI and break-even live in the tiles above, not repeated here');
assert.match(page,/ROI against winning bettors/);assert.match(page,/\+2% to \+5% ROI/);
assert.deepEqual([...page.matchAll(/data-perf-share="(\w+)"/g)].map(m=>m[1]),['summary','roi','path','sports'],'the summary and each part can be shared');
assert.match(page,/<th scope="row">NFL<\/th><td>50% <small>\+1 pending<\/small><\/td>/,'win percentage, not a record');
assert.doesNotMatch(page.replace(/<svg[\s\S]*?<\/svg>/g,''),/\b\d+[–-]\d+\b(?!%)/,'no win-loss records (icon path data aside)');
assert.doesNotMatch(page,/luck|standard error|±/i,'no luck range');
assert.doesNotMatch(page,/NaN|undefined|null/);
assert.match(html(summarize([bets[0]])),/appear once two bets in this view have settled/);
assert.match(html(summarize([])),/appear once two bets/);
// Share images say what they cover: the sport when only one is shown, the dates, the count.
assert.equal(context(s),'Oct 1 – Oct 3 · 5 settled bets');
assert.equal(context(summarize(bets.filter(b=>b.league==='NFL'))),'NFL · Oct 1 · 2 settled bets');

// One layout feeds the page SVG and the share image; each hover area sits just before its dot.
const path=toSVG(pathScene(s,600));
assert.equal((path.match(/<circle class="perf-dot (won|lost|push)"/g)||[]).length,5);
assert.equal((path.match(/<rect class="perf-hit" tabindex="0"[^>]*\/><circle class="perf-dot/g)||[]).length,5,'hover areas precede their dots');
assert.match(path,/data-readout="Oct 2 · NHL · Skater &lt;b&gt;One&lt;\/b&gt; sog under 2\.5 · Won \+\$3\.57/,'readouts are escaped');
assert.match(path,/<polyline class="perf-line" points="/);
const scale=toSVG(scaleScene(s,600));
assert.match(scale,/class="perf-pro"/);assert.match(scale,/class="perf-even"/);assert.match(scale,/You −0\.7%<\/text>/);
assert.match(toSVG(scaleScene(s,360)),/>Winning bettors<\/text>/,'short label on phones');
assert.ok(pathScene(s,400).shapes.every(([kind,,...g])=>kind==='poly'?g[0].flat().every(Number.isFinite):g.filter(v=>typeof v==='number').every(Number.isFinite)),'finite geometry for the canvas');
console.log('PASS: tracker performance: ROI vs winning bettors, running profit order, pushes, pending, by sport, share buttons and image context, no repeated tiles, records or luck range.');

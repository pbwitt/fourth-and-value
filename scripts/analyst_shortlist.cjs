/* Use the exact same selection policy as the public morning table. */
const fs=require('node:fs');
const {collect,shortlist,reviewKey,reviewBetKey,comparison,leadKey,priceCheck,editionRows,rowKey}=require('../docs/assets/briefing-picks.js');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const asof=Date.parse(input.asof);
const result=collect(input.feeds,asof);
// The afternoon edition adds new picks only. It leaves out every market on the morning
// card (either side, any line) and each side today's review passed on, so neither is
// reviewed or published again, and marks sides today's review left waiting on a fact,
// which are rechecked first. Each morning pick is also checked at the afternoon's prices,
// keyed like the page keys card rows.
const afternoon=input.afternoon;
if(afternoon){
  const side=r=>leadKey(r)+'|'+String(r.side).toLowerCase();
  const morning=afternoon.morning_edition?editionRows(afternoon.morning_edition,Date.parse(afternoon.morning_edition.published_at)+1):[];
  const markets=new Set(morning.map(leadKey)),verdicts=new Map();
  for(const r of afternoon.reviewed||[])verdicts.set(side(r),[...(verdicts.get(side(r))||[]),r.verdict]);
  result.selected=result.selected.filter(r=>!markets.has(leadKey(r))&&!(verdicts.get(side(r))||[]).includes('pass'))
    .map(r=>(verdicts.get(side(r))||[]).includes('wait')?{...r,afternoon_recheck:true}:r);
  result.price_check=morning.map(r=>({r,c:priceCheck(r,input.feeds,asof)})).filter(x=>x.c).map(({r,c})=>({key:rowKey(r),
    pick:{sport:r.sport,game:r.game,player:r.player||'',side:r.side,line:r.line,market:r.market,market_label:r.market_label,price:r.price,book_label:r.book_label||r.book},...c}));
}
result.card=shortlist(result.selected,asof);
result.selected=result.selected.filter(r=>['NFL','MLB','NHL'].includes(r.sport)).map(r=>({...r,review_key:reviewKey(r),review_bet_key:reviewBetKey(r),
  review_context:{...comparison(r),probability_basis:'conditional_on_nonpush',
    projected_quantity:r.sport==='NFL'?r.mu:r.sport==='NHL'?r.projected_mean:r.model_mean,
    // Lines sit near the median; NFL's Normal curve has no separate one.
    projected_median:(r.sport==='NHL'?r.projected_median:r.sport==='MLB'?r.model_median:null)??undefined,
    quantity_label:r.sport==='NFL'?r.market_label:r.model_mean_label,
    market_median_line:r.sport==='NFL'?r.consensus_line:null,
    validation:'Experimental; no established executable betting edge',
    calibration:r.sport==='NFL'?r.model_status:undefined,
    calibration_sample_size:null, // This feed does not expose a market-specific fitted sample.
    missing_model_detail:r.sport==='NFL'?'Market-specific calibration sample size and sensitivity unavailable in this feed':undefined}}));
process.stdout.write(JSON.stringify(result));

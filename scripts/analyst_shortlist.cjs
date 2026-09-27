/* Use the exact same selection policy as the public morning table. */
const fs=require('node:fs');
const {collect,reviewKey,reviewBetKey,comparison}=require('../docs/assets/briefing-picks.js');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const result=collect(input.feeds,Date.parse(input.asof));
result.selected=result.selected.filter(r=>['NFL','MLB','NHL'].includes(r.sport)).map(r=>({...r,review_key:reviewKey(r),review_bet_key:reviewBetKey(r),
  review_context:{...comparison(r),probability_basis:'conditional_on_nonpush',
    projected_quantity:r.sport==='NFL'?r.mu:r.sport==='NHL'?r.projected_mean:r.model_mean,
    quantity_label:r.sport==='NFL'?r.market_label:r.model_mean_label,
    market_median_line:r.sport==='NFL'?r.consensus_line:null,
    validation:'Experimental; no established executable betting edge',
    calibration:r.sport==='NFL'?r.model_status:undefined,
    calibration_sample_size:null, // This feed does not expose a market-specific fitted sample.
    missing_model_detail:r.sport==='NFL'?'Market-specific calibration sample size and sensitivity unavailable in this feed':undefined}}));
process.stdout.write(JSON.stringify(result));

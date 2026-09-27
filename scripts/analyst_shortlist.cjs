/* Use the exact same selection policy as the public morning table. */
const fs=require('node:fs');
const {collect,reviewKey,reviewBetKey}=require('../docs/assets/briefing-picks.js');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const result=collect(input.feeds,Date.parse(input.asof));
result.selected=result.selected.filter(r=>['NFL','MLB'].includes(r.sport)).map(r=>({...r,review_key:reviewKey(r),review_bet_key:reviewBetKey(r)}));
process.stdout.write(JSON.stringify(result));

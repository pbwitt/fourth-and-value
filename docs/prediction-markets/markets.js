(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const money = value => finite(value) ? '$' + Number(value).toFixed(2) : 'Unavailable';
  const cents = value => finite(value) ? (Number(value)*100).toFixed(2) + '¢' : 'Unavailable';
  const time = value => Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleString('en-US', {timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',second:'2-digit'})+' ET' : 'Unknown';
  const odds = value => finite(value) ? (Number(value)>0?'+':'')+Math.round(Number(value)) : 'Unavailable';
  const safeURL = value => {try {const u=new URL(value);return u.protocol==='https:'&&['kalshi.com','assets.kalshi.com'].includes(u.hostname)?u.href:null;}catch{return null;}};
  let snapshot=null, failed=false;
  const recent = (value, seconds) => {const age=Date.now()-Date.parse(value);return Number.isFinite(age)&&age>=0&&age<=seconds*1000;};
  function render() {
    if (!snapshot) return;
    const opened=new Set(Array.from(document.querySelectorAll('.contract details[open]')).map(e=>e.dataset.ticker));
    const count=$('contract-count').value, rows=snapshot.rows;
    const isCurrent=!failed&&snapshot.status!=='error'&&recent(snapshot.generated_at,snapshot.quote_max_seconds);
    $('snapshot-status').textContent=(failed?'Refresh failed. ':snapshot.status==='partial'?'Partial coverage. ':snapshot.status==='error'?'Feed unavailable. ':'')+
      (isCurrent?'Saved snapshot':'Historical snapshot')+' · '+time(snapshot.generated_at)+' · prices are not live';
    $('coverage').textContent=(snapshot.coverage||[]).map(c=>`${c.series}: ${c.sampled} contracts observed from ${c.discovered} returned${c.discovery_complete?'':' (discovery incomplete)'}`).join(' · ')+(snapshot.bounded?' · Bounded sample; not exhaustive coverage.':'');
    if (!rows.length) {
      $('contracts').innerHTML='<p class="empty">'+(snapshot.status==='error'?'The market feed is unavailable. No current prices are shown.':'No contract snapshots are available yet.')+'</p>';
      return;
    }
    $('contracts').innerHTML=rows.map(row=>{
      const current=isCurrent&&recent(row.observed_at,snapshot.quote_max_seconds)&&Date.parse(row.close_time)>Date.now();
      const feeCurrent=row.fee&&recent(row.fee.observed_at,snapshot.quote_max_seconds)&&(!row.fee.valid_until||Date.parse(row.fee.valid_until)>Date.now());
      const estimates=row.estimates?.[count]||{};
      const sides=['yes','no'].map(side=>{
        const q=estimates[side];
        if(!q)return `<tr><td>${side.toUpperCase()}</td><td colspan="5">Estimate unavailable</td></tr>`;
        const filled=Number(q.filled_contracts), partial=Number(q.unfilled_contracts)>0;
        return `<tr><td><strong>${side.toUpperCase()}</strong><small>${side==='yes'?'Contract resolves Yes':'Contract resolves No'}</small></td><td>${esc(q.filled_contracts)} / ${esc(q.requested_contracts)}${partial?'<small>Remainder unfilled</small>':''}</td><td>${filled?cents(q.average_price_dollars):'No offers'}</td><td>${money(q.fee_estimate_dollars)}</td><td><strong>${money(q.total_estimate_dollars)}</strong>${filled?'<small>'+cents(q.cost_per_contract_dollars)+' per contract</small>':''}</td><td>${filled?odds(q.payout_equivalent_american):'—'}</td></tr>`;
      }).join('');
      const url=safeURL(row.url), terms=safeURL(row.contract_terms_url);
      const comparisons=(row.sportsbook_comparisons||[]).map(b=>`<li>${esc(b.side.toUpperCase())}: ${esc(b.book)} ${odds(b.price)} · ${cents(b.cost_per_dollar_payout)} per $1 payout · quoted ${time(b.quoted_at)}${recent(b.quoted_at,snapshot.quote_max_seconds)&&current?'':' · Historical quote'}</li>`).join('');
      return `<article class="contract"><p class="eyebrow">${esc(row.sport)} · Kalshi</p><h3>${esc(row.title)}</h3><p><span class="badge ${current?'':'stale'}">${current?'Observed recently':'Historical quote'}</span> <span class="meta">Observed ${time(row.observed_at)}</span></p>
        <p class="meta">Market closes ${time(row.close_time)} · This is not the game start time.</p>
        <div class="table-wrap" tabindex="0" role="region" aria-label="Prices for ${esc(row.title)}"><table><caption class="meta">${current?'Snapshot purchase estimate':'Historical purchase estimate'} for up to ${esc(count)} contracts</caption><thead><tr><th scope="col">Position</th><th scope="col">Contracts</th><th scope="col">Average price</th><th scope="col">Estimated fee</th><th scope="col">Total cost</th><th scope="col">$1 payout odds</th></tr></thead><tbody>${sides}</tbody></table></div>
        ${row.fee?(feeCurrent?'':'<p class="meta">Fee estimate is historical; recheck before trading.</p>'):'<p class="meta">Fees unavailable. Total cost and equivalent odds are withheld.</p>'}
        <details data-ticker="${esc(row.ticker)}" ${opened.has(row.ticker)?'open':''}><summary>Settlement rules and sportsbook comparison</summary><p class="rules">${esc(row.rules_primary)}</p><p class="rules">${esc(row.rules_secondary)}</p>
        ${comparisons?'<ul>'+comparisons+'</ul>':'<p>No verified equivalent sportsbook offer is available. Contract prices are not substituted for a model forecast.</p>'}
        <p class="meta ticker">${esc(row.ticker)}</p><p>${url?'<a href="'+esc(url)+'" target="_blank" rel="noopener noreferrer">View on Kalshi</a>':''}${terms?' · <a href="'+esc(terms)+'" target="_blank" rel="noopener noreferrer">Contract terms</a>':''}</p></details></article>`;
    }).join('');
  }
  async function load() {
    try {
      const response=await fetch('snapshot.json',{cache:'no-store'});
      if(!response.ok)throw Error('Snapshot unavailable');
      const data=await response.json();
      if(data.schema_version!==1||!Array.isArray(data.rows)||!Number.isFinite(Date.parse(data.generated_at))||!Number.isFinite(data.quote_max_seconds)||data.quote_max_seconds<=0)throw Error('Invalid snapshot');
      snapshot=data;failed=false;render();
    } catch {
      failed=true;
      if(snapshot)render();
      else {$('snapshot-status').textContent='Market snapshot unavailable. No current prices are shown.';$('contracts').innerHTML='<p class="empty">We could not load the saved prices. Please try again later.</p>';}
    }
  }
  $('contract-count').addEventListener('change',render);
  load();setInterval(render,15000);setInterval(load,300000);
})();

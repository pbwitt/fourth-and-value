/* NHL analyst preparation. All research text is escaped; no client-side API credentials. */
(async () => {
  'use strict';
  const root=document.querySelector('[data-nhl-page]');
  const $=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=n=>Number.isFinite(n)?`${(100*n).toFixed(1)}%`:'Unavailable';
  const odds=n=>Number.isFinite(n)?(n>0?'+':'')+Math.ceil(n):'Unavailable';
  const time=v=>v?new Date(v).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})+' ET':'Unavailable';
  const day=v=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(v);
  const safeUrl=v=>{try {const u=new URL(v);return u.protocol==='https:'&&!u.username&&!u.password?u.href:null;}catch{return null;}};
  const statuses={completed:'Our analysis is complete. Interpretations need human verification.',no_candidates:'No candidates qualified; no analysis was requested.',
    no_usable_reporting:'Dated, relevant reporting was unavailable. Research requires a human check.',api_key_unavailable:'Our analysis is unavailable. The quantitative shortlist is still available for manual research.',
    disabled:'Automated research is disabled.',not_requested:'Automated research has not been requested.',already_attempted_today:'A morning research request was already attempted. These refreshed candidates need a new human review.',
    afternoon_quantitative_update:'Afternoon quantitative update. No additional context review was requested.',budget_exhausted:'The weekly research budget is exhausted. Review these candidates manually.',
    expired_during_research:'Quotes expired during source collection. Refresh and reprice before deciding.',review_unavailable:'Our analysis could not be verified. Review these candidates manually.',feed_unavailable:'The market or model feed is unavailable.'};
  let board,data;
  try {
    const responses=await Promise.all([fetch('./data/candidates.json',{cache:'no-store'}),fetch(root.dataset.feed,{cache:'no-store'})]);
    if(responses.some(r=>!r.ok))throw Error();
    [board,data]=await Promise.all(responses.map(r=>r.json()));
    if(board.schema_version!==1||!Array.isArray(board.candidates))throw Error();
  } catch {
    $('feed-status').textContent='The NHL shortlist could not be loaded. No current candidates are available.';
    $('candidate-summary').textContent='Try again after the next successful refresh.';return;
  }
  const now=Date.now();
  const sourceAge=now-Date.parse(data.last_success_at);
  const failed=!['ready','waiting_for_markets'].includes(data.status)||!!data.model_error||!Number.isFinite(sourceAge)||sourceAge<0||sourceAge>24*3600e3;
  const mismatch=board.source_snapshot_id!==data.snapshot_id;
  const today=day(new Date())===board.decision_date;
  const current=!failed&&!mismatch&&today&&board.status!=='unavailable';
  const rows=current?board.candidates.filter(r=>Date.parse(r.commence_time)>now):[];
  $('feed-status').innerHTML=`<strong>${current?'Saved NHL analyst shortlist':'No current NHL shortlist'}</strong><p>Prepared ${esc(time(board.generated_at))}. ${esc(board.session)} session. ${mismatch?'The underlying market snapshot changed; the shortlist needs a refresh.':failed?'The market or model feed is unavailable.':!today?'This shortlist belongs to a previous day.':'Confirm price and context before deciding.'}</p>`;
  $('history-status').textContent='Maximum four candidates, one per game. Scenario limits are assumptions, not confidence intervals.';
  $('model-status').textContent='Experimental model forecasts and prospective analyst research. No validated recommendations.';
  $('research-status').textContent=statuses[board.review_status]||'Research unavailable.';
  const sources=new Map((board.sources||[]).map(s=>[s.source_id,s]));
  function fresh(r){
    const t=Date.now(),quote=t-Date.parse(r.quoted_at),model=t-Date.parse(r.model_data_checked_at),forecast=t-Date.parse(r.decision_at);
    return current&&day(new Date())===board.decision_date&&t<Date.parse(r.commence_time)&&quote>=0&&quote<=1800e3&&forecast>=0&&forecast<=1800e3&&model>=0&&model<36*3600e3;
  }
  function research(r){
    const q=r.qualitative_review;
    if(!q||q.offer_id!==r.offer_id||q.forecast_id!==r.forecast_id)return '<h3>2. Context review</h3><p>Research unavailable for this exact forecast. Verify goalie, lineup and role assumptions yourself.</p>';
    const labels={research_support:'Sourced support · human review required',concern:'Sourced concern',needs_information:'Needs information'};
    const a=q.assessment,verdict={consider:'Consider · human review needed',wait:'Needs review',pass:'Pass · case not supported'};
    const judgment=a&&verdict[a.verdict]?`<p><strong>Our assessment: ${esc(verdict[a.verdict])}.</strong> ${esc(a.reason)}</p><p><strong>Model case:</strong> ${esc(a.model_case)}</p><p><strong>Price case:</strong> ${esc(a.price_case)}</p><p><strong>Relevant context:</strong> ${esc(a.context_case)}</p>${a.blocking_checks?.length?'<p><strong>What needs checking:</strong> '+a.blocking_checks.map(esc).join('; ')+'</p>':''}`:'<p>This earlier review checked reporting only. A full model-and-price assessment is pending.</p>';
    return `<h3>2. Our assessment</h3>${judgment}<p class="meta">Reporting: ${esc(labels[q.status]||'Unverified research')}. A reporting gap alone does not invalidate the numerical case.</p>${q.evidence.map(e=>{
      const s=sources.get(e.source_id),url=safeUrl(s?.url);if(!s||!url)return '<p>Evidence source unavailable; do not rely on this note.</p>';
      return `<div class="panel"><p><strong>${esc(e.direction)} · ${esc(e.kind)}</strong></p><blockquote>${esc(e.excerpt)}</blockquote><p>Our analysis: ${esc(e.interpretation)}</p><p><a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a> · published ${esc(time(s.published_at))}; retrieved ${esc(time(s.retrieved_at))}.</p><p class="meta">Possibly reflected in: ${esc(e.represented_in.replaceAll('_',' '))}.</p></div>`;
    }).join('')}<p><strong>Countercase:</strong> ${esc(q.countercase)}</p><p><strong>Still to verify:</strong> ${q.open_checks.map(esc).join('; ')}.</p><p class="meta">Fourth &amp; Value analysis · ${esc(time(q.reviewed_at))} · AI-assisted research. Source excerpts matched automatically; interpretation has not been verified by a human. Original probabilities unchanged.</p>`;
  }
  function card(r){
    const valid=fresh(r),id=r.candidate_id;
    return `<article class="panel prop-card" data-candidate="${esc(id)}"><p class="eyebrow">Experimental candidate ${r.candidate_rank}</p><p class="meta">${esc(r.game)} · ${esc(time(r.commence_time))}</p><h2>${esc(r.player||r.market_label)}</h2><p class="betline">${esc(r.side)} ${r.line===null?'':esc(r.line)} · ${esc(r.market_label)}</p><p><strong>${esc(r.book_label)} ${odds(r.price)}</strong> · Quote ${esc(time(r.quoted_at))}</p><p class="notice">${valid?'Recent saved quote. Verify availability at the book.':'Price check required. This saved quote or forecast has expired; selection is disabled until a refresh.'}</p>
    <h3>1. Model and price</h3><dl><dt>Independent win probability</dt><dd>${pct(r.independent_probability)}</dd><dt>Other-book market (non-push)</dt><dd>${pct(r.market_probability)} (${r.other_books??0} books)</dd><dt>Final win probability</dt><dd>${pct(r.final_probability)}</dd><dt>Push probability</dt><dd>${pct(r.push_probability)}</dd><dt>Fair price</dt><dd>${odds(r.fair_odds)}</dd><dt>Estimated EV per unit</dt><dd>${pct(r.estimated_ev)}</dd><dt>Minimum acceptable price</dt><dd>${odds(r.minimum_acceptable_odds)}</dd><dt>Model / market difference (non-push)</dt><dd>${Number.isFinite(r.independent_market_difference)?(r.independent_market_difference*100).toFixed(1)+' pp':'Unavailable'}</dd></dl>
    <p>Signal: ${esc((r.signal_type||'independent_model').replaceAll('_',' '))}. Market consensus is a comparison; it is not substituted for the hockey model.</p><p>${(r.key_drivers||[]).map(esc).join('. ')}.</p><details><summary>Assumptions and sensitivity</summary><p>${esc(r.goalie_assumption)} ${esc(r.lineup_assumption)}</p><p>${(r.uncertainties||[]).map(esc).join('; ')}.</p><p>${esc(r.sensitivity?.assumption)}. Win range: ${pct(r.sensitivity?.win_min)}–${pct(r.sensitivity?.win_max)}.</p><p>Recheck if: ${(r.invalidation_conditions||[]).map(esc).join('; ')}.</p><p class="meta">${esc(r.model_version)} · ${esc(r.validation_status)} · Inputs checked ${esc(time(r.model_data_checked_at))}. ${esc(r.settlement_scope||'Verify sportsbook rules.')}</p></details>
    ${research(r)}<h3>3. Your decision</h3><p>Recorded status: ${esc(r.human_decision||'unreviewed')}. Preparation only; no wager is placed.</p>
    <details><summary>Prepare an analyst review</summary><form data-review="${esc(id)}"><label>Decision<select name="decision"><option value="watch">Watch / wait</option><option value="pass">Pass</option><option value="select" ${valid?'':'disabled'}>Select for shadow tracking</option></select></label><label>Analyst name<input name="analyst" required minlength="3" maxlength="120"></label><label>Reason<textarea name="reason" required minlength="3" maxlength="1500"></textarea></label><label>Already reflected in model or market?<textarea name="double_counting_check" required minlength="3" maxlength="1500" placeholder="Describe what is new, already reflected, or unknown."></textarea></label><label>Context source URL (required to select)<input name="source_url" type="url" placeholder="https://…"></label><label>Source publication time, with time zone<input name="source_published_at" placeholder="2026-10-01T09:15:00-04:00"></label><label><input name="price_confirmed" type="checkbox"> I checked this exact line and price.</label><label><input name="context_checked" type="checkbox"> I checked participation, goalie and role assumptions as relevant.</label><button type="submit">Download review note</button><p class="meta">The download saves a local preparation note. It is not yet recorded on the site. An operator imports it into the decision ledger for prospective tracking.</p><p class="review-feedback" role="status"></p></form></details></article>`;
  }
  function render(){
    const q=$('candidate-search').value.toLowerCase(),filter=$('candidate-status').value;
    const selected=rows.filter(r=>(r.game+' '+r.player).toLowerCase().includes(q)&&(!filter||(r.qualitative_review?.status||'unreviewed')===filter));
    $('candidate-summary').textContent=`${selected.length} research candidates · ${board.eligible_count||0} offers passed the quantitative screen before exposure and shortlist limits.`;
    $('candidates').innerHTML=selected.map(card).join('')||'<p class="empty">No current candidates match this screen. An empty shortlist is a valid result; Market Watch may still contain price comparisons.</p>';
    document.querySelectorAll('form[data-review]').forEach(form=>form.addEventListener('submit',event=>{
      event.preventDefault();const r=rows.find(r=>r.candidate_id===form.dataset.review),values=new FormData(form);
      const result={board_id:board.board_id,candidate_id:r.candidate_id,offer_id:r.offer_id,forecast_id:r.forecast_id,recorded_at:new Date().toISOString()};
      for(const key of ['decision','analyst','reason','double_counting_check','source_url','source_published_at'])result[key]=String(values.get(key)||'').trim();
      for(const key of ['price_confirmed','context_checked'])result[key]=values.has(key);
      const feedback=form.querySelector('.review-feedback');
      if(result.decision==='select'&&(!fresh(r)||!result.price_confirmed||!result.context_checked||!safeUrl(result.source_url)||!Number.isFinite(Date.parse(result.source_published_at))||Date.parse(result.source_published_at)>Date.now()||!/(Z|[+-]\d\d:\d\d)$/.test(result.source_published_at))){feedback.textContent='Selection requires a current candidate, both checks, and a dated HTTPS source. Refresh expired quotes first.';return;}
      const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`nhl-review-${r.candidate_id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);feedback.textContent='Review downloaded locally. It still needs to be imported for tracking; no selection has been published.';
    }));
  }
  $('candidate-search').addEventListener('input',render);$('candidate-status').addEventListener('change',render);render();
  // Recheck quote expiry without destroying an in-progress review form.
  setInterval(()=>{document.querySelectorAll('[data-candidate]').forEach(el=>{const row=rows.find(r=>r.candidate_id===el.dataset.candidate);if(!fresh(row)){el.querySelector('.notice').textContent='Price check required. This quote or forecast has expired; refresh before selecting.';el.querySelector('option[value="select"]').disabled=true;}});},30000);
})();

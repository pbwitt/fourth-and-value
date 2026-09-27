/* Shared source-time and injury-snapshot presentation; never infers health. */
(function(root){
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const time=v=>v&&Number.isFinite(Date.parse(v))?new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}).format(new Date(v)):'unknown';
  function sourceTime(s){return s.source_kind==='live_injury_table'?`Publication time unknown · observed ${time(s.retrieved_at)}`:`Published ${time(s.published_at)}${s.updated_at?` · updated ${time(s.updated_at)}`:''}`;}
  function render(row,sources){
    const q=row.qualitative_review, id=q?.candidate_id||row.candidate_id, context=row.injury_context||row.reviewed_candidate?.injury_context;
    const tables=(sources||[]).filter(s=>s.source_kind==='live_injury_table'&&s.candidate_ids?.includes(id)&&Array.isArray(s.injury_rows)&&Number.isFinite(Date.parse(s.retrieved_at))&&Date.parse(s.retrieved_at)<=Date.parse(q?.evidence_asof||q?.reviewed_at));
    const latest=tables.sort((a,b)=>Date.parse(b.retrieved_at)-Date.parse(a.retrieved_at))[0];
    if(!latest){
      if(['unavailable','partial','available'].includes(context?.status))return '<p class="meta">Injury table: '+(context.status==='unavailable'?'unavailable for this review.':'no matched listings available for this review.')+' Health and participation remain unverified.</p>';
      return '';
    }
    let url;try{url=new URL(latest.url);if(url.username||url.password||url.origin!=='https://www.cbssports.com'||!['/mlb/injuries/','/nhl/injuries/'].includes(url.pathname))return '';}catch{return '';}
    const rows=latest.injury_rows.map(r=>`<tr><td>${esc(r.player)} (${esc(r.position)})<br><span class="meta">${esc(r.team)}</span></td><td>${esc(r.injury)}</td><td>${esc(r.status)}</td><td>${esc(r.reported_update)}</td></tr>`).join('');
    return `<details class="pick-injuries"><summary>Injury table at review time · ${esc(time(latest.retrieved_at))}</summary><p class="meta"><a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer">CBS Sports injury listings</a>. Publication time unknown; row updates are shown as reported. A listed return date is an estimate. Players absent from this table are not confirmed healthy or active. The assessment may use a subset of these rows.</p>${latest.missing_teams?.length?`<p>Team coverage unavailable: ${latest.missing_teams.map(esc).join(', ')}.</p>`:''}<div style="max-width:100%;overflow-x:auto"><table style="min-width:520px;width:100%"><thead><tr><th>Player / team</th><th>Injury</th><th>Listed status</th><th>Reported update</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
  }
  const api={render,sourceTime};if(typeof module==='object'&&module.exports)module.exports=api;else root.FVInjuryContext=api;
})(typeof window==='object'?window:globalThis);

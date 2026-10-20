/* NBA boards: exact-line comparisons; never present stale quotes as live. */
(async () => {
  'use strict';
  const root=document.querySelector('[data-nba-page]'), page=root.dataset.nbaPage;
  const $=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=n=>Number.isFinite(n)?`${(100*n).toFixed(1)}%`:'Unavailable';
  const odds=n=>Number.isFinite(n)?(n>0?`+${n}`:String(n)):'Unavailable';
  const time=s=>s?new Date(s).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})+' ET':'Not yet checked';
  const status=$('feed-status');
  let data;
  try { const response=await fetch(root.dataset.feed,{cache:'no-store'});if(!response.ok)throw Error('Unavailable');data=await response.json(); }
  catch {status.textContent='NBA data could not be loaded. Please try again later.';return;}
  const now=Date.now(), success=Date.parse(data.last_success_at);
  const stale=!Number.isFinite(success)||now-success>24*3600e3||success>now+300e3;
  const failed=data.status==='feed_error';
  const events=(data.events||[]).filter(e=>Date.parse(e.commence_time)>now);
  // Started games keep their last pregame quote for lookup on Player Props and Game Lines only.
  const started=r=>!(Date.parse(r.commence_time)>now);
  const saved=(!stale&&!failed?(data.rows||[]):[]).filter(r=>now-Date.parse(r.quoted_at)<=24*3600e3&&Date.parse(r.quoted_at)<=now+300e3);
  const rows=saved.filter(r=>!started(r));
  let message=failed?'The latest NBA refresh failed. Saved odds are hidden until the feed recovers.':stale?'The NBA snapshot needs a refresh. Stale odds are hidden.':rows.length?'Saved NBA quotes are available. Confirm the current price with your sportsbook.':'Waiting for NBA markets. No recent quotes are available in this snapshot.';
  status.innerHTML=`<strong>${esc(message)}</strong><p>Last successful check: ${esc(time(data.last_success_at))}. ${rows.length} quotes · ${events.length} upcoming provider events.</p>`;
  if(page==='methods')return;
  if(page==='overview'){
    $('schedule').innerHTML=events.slice(0,12).map(e=>`<article class="panel"><h3>${esc(e.away_team)} at ${esc(e.home_team)}</h3><p>${esc(time(e.commence_time))}</p></article>`).join('')||'<p>No upcoming events in the provider snapshot yet.</p>';
    return;
  }
  const lookup=page==='props'||page==='lines';
  const relevant=(lookup?saved:rows).filter(r=>page==='props'?r.market.startsWith('player_'):page==='lines'?!r.market.startsWith('player_'):r.other_books>=3&&r.consensus_ev>0);
  const params=new URLSearchParams(location.search);
  if($('history'))$('history').checked=params.get('history')==='1';
  ['market','book','game'].forEach(id=>{
    const key=id==='game'?'event_id':id, labels=id==='market'?'market_label':id==='book'?'book_label':'game';
    const entries=new Map(relevant.map(r=>[r[key],r[labels]]));
    for(const [value,label] of [...entries].sort((a,b)=>a[1].localeCompare(b[1]))){const option=document.createElement('option');option.value=value;option.textContent=label;$(id).append(option);}
    $(id).value=params.get(id)||'';
  });
  $('search').value=params.get('q')||'';
  let limit=30;
  function render(){
    const q=$('search').value.toLowerCase();
    const showStarted=!!$('history')?.checked;
    let selected=relevant.filter(r=>(showStarted||!started(r))&&(r.player+' '+r.game).toLowerCase().includes(q)&&(!$('market').value||r.market===$('market').value)&&(!$('book').value||r.book===$('book').value)&&(!$('game').value||r.event_id===$('game').value));
    if($('best').checked){const cheapest=new Map();for(const r of selected){const key=JSON.stringify([r.event_id,r.player,r.market,r.line,r.side]);cheapest.set(key,Math.min(cheapest.get(key)??1,r.book_probability));}selected=selected.filter(r=>r.book_probability===cheapest.get(JSON.stringify([r.event_id,r.player,r.market,r.line,r.side])));}
    selected.sort((a,b)=>page==='watch'?b.consensus_ev-a.consensus_ev:a.commence_time.localeCompare(b.commence_time)||a.player.localeCompare(b.player));
    $('result-count').textContent=`${selected.length} matching offers`;
    $('results').innerHTML=selected.slice(0,limit).map(r=>{
      const baseline=r.baseline_mean!==null&&r.baseline_mean!==undefined?`<p class="meta">Historical hit rate: ${pct(r.baseline_probability)} · Smoothed; excludes pushes. ${esc(r.baseline_status??r.model_status)}</p>`:'';
      // NBA v1 runs in shadow mode: shown for research, labeled experimental, never ranked as a pick.
      const push=Number.isFinite(r.model_push_probability)&&r.model_push_probability>=0.0005?` · push ${pct(r.model_push_probability)}`:'';
      // model_ev_pct stays in the feed for grading; ungraded EV is not shown to readers.
      const projection=Number.isFinite(r.model_mean)?` · projection: average ${r.model_mean.toFixed(1)}${Number.isFinite(r.model_median)?`, median ${esc(r.model_median)}`:''}${Number.isFinite(r.model_minutes)?` in ${r.model_minutes.toFixed(0)} min`:''}`:'';
      const injury=r.injury_status?` · ESPN injury report: ${esc(r.injury_status.status)}`:'';
      const model=Number.isFinite(r.model_probability)?`<p class="meta"><strong>NBA v1 (experimental):</strong> ${pct(r.model_probability)} to win${push}${projection}${injury}. Shadow mode; not yet graded against prices.</p>`
        :r.model_unsupported_reason?`<p class="meta">NBA v1: no forecast (${esc(r.model_unsupported_reason)}).</p>`:'';
      const watch=page==='watch'?`<p>Other-book fair probability: ${pct(r.other_book_probability)} (${r.other_books} books). Price gap: ${(100*(r.other_book_probability-r.book_probability)).toFixed(1)} percentage points.</p>`:'';
      return `<article class="panel prop-card"><p class="meta">${esc(r.game)} · ${esc(time(r.commence_time))}</p>${started(r)?'<p class="meta"><strong>Game started · historical quote</strong></p>':''}<h2>${r.player?(window.FVPlayerContext?.name?.(r,'NBA')??esc(r.player)):esc(r.market_label)}</h2><p class="betline">${esc(r.side)} ${r.line===null?'':esc(r.line)} · ${esc(r.market_label)}</p><p><strong>${esc(r.book_label)} ${esc(odds(r.price))}</strong></p><dl><dt>Book probability</dt><dd>${pct(r.book_probability)}</dd><dt>Paired fair probability</dt><dd>${pct(r.fair_probability)}</dd><dt>Same-line consensus</dt><dd>${pct(r.consensus_probability)}</dd><dt>Paired books</dt><dd>${r.paired_books}</dd></dl>${watch}${model}${baseline}<p class="meta">Quote: ${esc(time(r.quoted_at))}</p></article>`;
    }).join('')||`<p class="empty">No matching NBA offers. Markets may not be posted yet, or your filters may exclude the available quotes.${$('history')&&!showStarted&&relevant.some(started)?' Include started games to see their last pregame quotes.':''}</p>`;
    window.FVOfferTracker?.attach($('results'),selected.slice(0,limit).map(r=>({...r,sport:'NBA'})));
    $('more').hidden=selected.length<=limit;
    const p=new URLSearchParams();for(const id of ['market','book','game'])if($(id).value)p.set(id,$(id).value);if(q)p.set('q',$('search').value);if(showStarted)p.set('history','1');history.replaceState(null,'',location.pathname+(p.size?'?'+p.toString():''));
  }
  for(const id of ['search','market','book','game','best','history'])$(id)?.addEventListener(id==='search'?'input':'change',()=>{limit=30;render();});
  $('reset').onclick=()=>{['search','market','book','game'].forEach(id=>$(id).value='');$('best').checked=true;if($('history'))$('history').checked=false;limit=30;render();};
  $('more').onclick=()=>{limit+=30;render();};
  render();
  // Move started games behind the toggle and drop expired quotes on an open tab as well as at load.
  function refresh(){if(window.FVOfferTracker?.isOpen())setTimeout(refresh,30000);else location.reload();}
  setTimeout(refresh,300000);
})();

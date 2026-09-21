/* MLB boards: exact-line comparisons; never present stale quotes as live. */
(async () => {
  'use strict';
  const root=document.querySelector('[data-mlb-page]'), page=root.dataset.mlbPage;
  const $=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=n=>Number.isFinite(n)?`${(100*n).toFixed(1)}%`:'Unavailable';
  const odds=n=>n>0?`+${n}`:String(n);
  const time=s=>s?new Date(s).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})+' ET':'Not yet checked';
  const status=$('feed-status');
  let data;
  try { const response=await fetch(root.dataset.feed,{cache:'no-store'});if(!response.ok)throw Error('Unavailable');data=await response.json(); }
  catch {status.textContent='MLB data could not be loaded. Please try again later.';return;}
  const now=Date.now(), success=Date.parse(data.last_success_at);
  const stale=!Number.isFinite(success)||now-success>12*3600e3||success>now+300e3;
  const failed=data.status==='feed_error';
  const events=(data.events||[]).filter(e=>Date.parse(e.commence_time)>now);
  const rows=(!stale&&!failed?(data.rows||[]):[]).filter(r=>Date.parse(r.commence_time)>now&&now-Date.parse(r.quoted_at)<=12*3600e3&&Date.parse(r.quoted_at)<=now+300e3);
  let message=failed?'The latest MLB refresh failed. Saved odds are hidden until the feed recovers.':stale?'The MLB snapshot needs a refresh. Stale odds are hidden.':rows.length?'Saved MLB quotes are available. Confirm the current price with your sportsbook.':'Waiting for MLB markets. No recent quotes are available in this snapshot.';
  status.innerHTML=`<strong>${esc(message)}</strong><p>Last successful check: ${esc(time(data.last_success_at))}. ${rows.length} quotes · ${events.length} scheduled games.</p>`;
  $('history-status').textContent=data.history_error?'Season statistics are unavailable while the MLB feed recovers.':data.history_checked_at?'Regular-season statistics through '+data.history_through_date+'; checked '+time(data.history_checked_at)+'.':'Season statistics are awaiting an update.';
  if(data.props_events_skipped)$('history-status').textContent+=' Props coverage cap: '+data.props_events_skipped+' additional games await a later refresh.';
  if(page==='methods')return;
  const pitcher=p=>p?.fullName||'TBD';
  const phaseMatch=(g,value)=>!value||(value==='postseason'?g.game_type!=='R':g.game_type===value);
  if(page==='overview'){
    const draw=()=>{$('schedule').innerHTML=events.filter(e=>phaseMatch(e,$('schedule-phase').value)).slice(0,30).map(e=>`<article class="panel"><p class="meta">${esc(e.phase)}${e.if_necessary?' · If necessary':''}${e.doubleheader?' · Doubleheader game '+e.game_number:''}</p><h3>${esc(e.away_team)} at ${esc(e.home_team)}</h3><p>${esc(time(e.commence_time))}</p><p>Probable pitchers: ${esc(pitcher(e.away_pitcher))} vs ${esc(pitcher(e.home_pitcher))}</p><p class="meta">${esc(e.venue)} · Batting lineups not verified</p></article>`).join('')||'<p>No confirmed upcoming games in this phase yet. Postseason matchups and times appear as MLB confirms them.</p>';};
    $('schedule-phase').onchange=draw;draw();setTimeout(()=>location.reload(),300000);return;
  }
  const relevant=rows.filter(r=>page==='props'?r.market_family==='props':page==='lines'?r.market_family==='lines':r.other_books>=3&&r.consensus_ev>0);
  const params=new URLSearchParams(location.search);
  ['market','book','game'].forEach(id=>{
    const key=id==='game'?'event_id':id, labels=id==='market'?'market_label':id==='book'?'book_label':'game';
    const entries=new Map(relevant.map(r=>[r[key],r[labels]]));
    for(const [value,label] of [...entries].sort((a,b)=>a[1].localeCompare(b[1]))){const option=document.createElement('option');option.value=value;option.textContent=label;$(id).append(option);}
    $(id).value=params.get(id)||'';
  });
  $('search').value=params.get('q')||'';
  $('phase').value=params.get('phase')||'';
  let limit=30;
  function render(){
    const q=$('search').value.toLowerCase();
    let selected=relevant.filter(r=>phaseMatch(r,$('phase').value)&&(r.player+' '+r.game).toLowerCase().includes(q)&&(!$('market').value||r.market===$('market').value)&&(!$('book').value||r.book===$('book').value)&&(!$('game').value||r.event_id===$('game').value));
    if($('best').checked){const cheapest=new Map();for(const r of selected){const key=JSON.stringify([r.event_id,r.player,r.market,r.line,r.side]);cheapest.set(key,Math.min(cheapest.get(key)??1,r.book_probability));}selected=selected.filter(r=>r.book_probability===cheapest.get(JSON.stringify([r.event_id,r.player,r.market,r.line,r.side])));}
    selected.sort((a,b)=>page==='watch'?b.consensus_ev-a.consensus_ev:a.commence_time.localeCompare(b.commence_time)||a.player.localeCompare(b.player));
    $('result-count').textContent=`${selected.length} matching offers`;
    $('results').innerHTML=selected.slice(0,limit).map(r=>{
      const age=now-Date.parse(data.history_checked_at),c=r.stat_context;
      const statsFresh=!data.history_error&&age>=-300e3&&age<36*3600e3;
      const baseline=statsFresh&&c?`<details><summary>Regular-season context · ${esc(data.season)}</summary><p>${c.group==='pitching'?`${esc(c.innings)} IP · ${c.starts} starts · ${c.strikeouts} K · K/9 ${c.k_per_nine.toFixed(2)} · ERA ${c.era.toFixed(2)}`:`${c.pa} PA · AVG ${esc(c.avg)} · OPS ${esc(c.ops)} · ${c.hits} hits · ${c.home_runs} HR · ${c.total_bases} total bases · ${c.rbis} RBI`}</p><p>${esc(r.starter_status||'Batting lineup not verified')}. Descriptive statistics, not a game projection.</p></details>`:'';
      const watch=page==='watch'?`<p>Other-book fair probability: ${pct(r.other_book_probability)} (${r.other_books} books). Price gap: ${(100*(r.other_book_probability-r.book_probability)).toFixed(1)} percentage points.</p>`:'';
      return `<article class="panel prop-card"><p class="meta">${esc(r.phase)}${r.if_necessary?' · If necessary':''} · ${esc(r.game)} · ${esc(time(r.commence_time))}</p><h2>${esc(r.player||r.market_label)}</h2><p class="betline">${esc(r.side)} ${r.line===null?'':esc(r.line)} · ${esc(r.market_label)}</p><p><strong>${esc(r.book_label)} ${esc(odds(r.price))}</strong></p><dl><dt>Book probability</dt><dd>${pct(r.book_probability)}</dd><dt>Paired fair probability</dt><dd>${pct(r.fair_probability)}</dd><dt>Same-line consensus</dt><dd>${pct(r.consensus_probability)}</dd><dt>Paired books</dt><dd>${r.paired_books}</dd></dl>${watch}<p class="meta">Probable pitchers: ${esc(pitcher(r.away_pitcher))} vs ${esc(pitcher(r.home_pitcher))}. Batting lineups not verified.</p>${baseline}<p class="meta">Quote: ${esc(time(r.quoted_at))}</p></article>`;
    }).join('')||'<p class="empty">No matching MLB offers. Markets may not be posted yet, or your filters may exclude the available quotes.</p>';
    $('more').hidden=selected.length<=limit;
    const p=new URLSearchParams();for(const id of ['market','book','game','phase'])if($(id).value)p.set(id,$(id).value);if(q)p.set('q',$('search').value);history.replaceState(null,'',location.pathname+(p.size?'?'+p.toString():''));
  }
  for(const id of ['search','market','book','game','phase','best'])$(id).addEventListener(id==='search'?'input':'change',()=>{limit=30;render();});
  $('reset').onclick=()=>{['search','market','book','game','phase'].forEach(id=>$(id).value='');$('best').checked=true;limit=30;render();};
  $('more').onclick=()=>{limit+=30;render();};
  render();
  // Remove kicked-off and expired quotes on an open tab as well as at load.
  setTimeout(()=>location.reload(),300000);
})();

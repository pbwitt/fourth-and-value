/* Read existing model boards; no market-only fallback and no model rerun. */
(function (global) {
  'use strict';
  const HOUR=3600e3, MINUTE=60e3;
  const finite=n=>typeof n==='number'&&Number.isFinite(n);
  const day=t=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(t));
  const stamp=s=>typeof s==='string'&&/(Z|[+-]\d\d:\d\d)$/.test(s)?Date.parse(s):NaN;
  const recent=(s,now,limit)=>Number.isFinite(stamp(s))&&now-stamp(s)>=0&&now-stamp(s)<=limit;
  const today=(r,now)=>stamp(r.commence_time)>now&&day(stamp(r.commence_time))===day(now);
  const records=value=>Array.isArray(value)?value.filter(r=>r&&typeof r==='object'):[];
  const rows=d=>records(d?.rows);
  const priced=r=>finite(r.price)&&Math.abs(r.price)>=100&&r.game&&r.book&&r.side&&r.market_label;
  const odds=n=>(n>0?'+':'')+n;
  const time=s=>new Date(s).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',second:'2-digit'})+' ET';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const decimal=p=>p>0?1+p/100:1-100/p;

  function nhlReview(r) {
    if(r.human_decision==='select')return 'Analyst selected for shadow tracking';
    if(r.human_decision==='watch')return 'Analyst: watch / wait';
    const q=r.qualitative_review;
    if(!q||q.offer_id!==r.offer_id||q.forecast_id!==r.forecast_id)return 'Context review needed';
    return ({research_support:'Sourced support · analyst review needed',concern:'Sourced concern · review before deciding',needs_information:'More information needed'})[q.status]||'Context review needed';
  }

  function collect(feeds,now=Date.now()) {
    const selected=[],coverage=[];
    for(const sport of ['NFL','MLB','NHL']) {
      const data=feeds[sport],board=feeds.NHLBoard;
      let available=false,candidates=[];
      if(sport==='NFL') {
        available=data?.schema_version===1&&data.status==='ready'&&recent(data.generated_at,now,48*HOUR);
        if(available)candidates=rows(data).filter(r=>today(r,now)&&recent(r.last_update,now,48*HOUR)&&
          r.model_status?.startsWith('Calibration fitted')&&finite(r.model_prob)&&r.model_prob>0&&r.model_prob<1&&finite(r.edge_bps)&&r.edge_bps>0)
          .map(r=>({...r,book:r.bookmaker,side:r.name,line:r.point,quoted_at:r.last_update,
            game_id:r.game_id||r.game+'|'+r.commence_time,score:r.edge_bps,
            review:'Qualitative review needed',url:'/props/top.html?'+new URLSearchParams({q:r.player||r.game,market:r.market_std,game:r.game})}));
      } else if(sport==='MLB') {
        available=data?.status==='ready'&&!data.model_error&&recent(data.last_success_at,now,12*HOUR)&&recent(data.model_checked_at,now,90*MINUTE);
        if(available)candidates=rows(data).filter(r=>today(r,now)&&r.is_model_pick===true&&recent(r.quoted_at,now,90*MINUTE)&&
          finite(r.model_probability)&&r.model_probability>0&&r.model_probability<1&&finite(r.model_ev_pct)&&r.model_ev_pct>0)
          .map(r=>({...r,game_id:r.mlb_game_id||r.event_id,score:r.model_ev_pct,review:'Qualitative review needed',
            url:'/mlb/picks.html?'+new URLSearchParams({game:r.event_id,market:r.market,q:r.player||''})}));
      } else {
        available=['ready','waiting_for_markets'].includes(data?.status)&&!data.model_error&&recent(data.last_success_at,now,30*MINUTE)&&
          board?.schema_version===1&&['ready','no_candidates'].includes(board.status)&&
          !!data.snapshot_id&&board.source_snapshot_id===data.snapshot_id&&board.decision_date===day(now)&&recent(board.generated_at,now,30*MINUTE);
        if(available)candidates=records(board.candidates).filter(r=>today(r,now)&&r.human_decision!=='pass'&&
          recent(r.quoted_at,now,30*MINUTE)&&recent(r.decision_at,now,30*MINUTE)&&recent(r.model_data_checked_at,now,36*HOUR)&&now-stamp(r.model_data_checked_at)<36*HOUR&&
          finite(r.independent_probability)&&r.independent_probability>0&&r.independent_probability<1&&
          finite(r.final_probability)&&r.final_probability>0&&r.final_probability<1&&finite(r.candidate_rank)&&r.candidate_rank>0)
          .map(r=>({...r,game_id:r.nhl_game_id,score:-r.candidate_rank,review:nhlReview(r),url:'/nhl/picks.html'}));
      }
      // The source boards own model eligibility. This view only limits exposure
      // and today's review size; it does not compare scores between sports.
      candidates=candidates.filter(r=>priced(r)&&r.game_id&&(r.line===null||finite(r.line)));
      candidates.sort((a,b)=>b.score-a.score||decimal(b.price)-decimal(a.price)||stamp(b.quoted_at)-stamp(a.quoted_at)||String(a.book).localeCompare(String(b.book)));
      const seen=new Set();let count=0;
      for(const r of candidates) {
        if(count>=4||seen.has(r.game_id))continue;
        seen.add(r.game_id);count++;
        selected.push({...r,sport});
      }
      coverage.push({sport,count,available,message:!available?'Current model list unavailable or expired':count?`${count} review candidate${count===1?'':'s'}`:'No qualifying games remaining today'});
    }
    return {selected,coverage};
  }

  function rowHTML(r) {
    const line=r.line===null?'':r.market==='spreads'&&r.line>0?'+'+r.line:String(r.line);
    const bet=[r.player,r.side,line,r.market_label].filter(Boolean).join(' · ');
    return `<tr><td><a href="${esc(r.url)}"><strong>${esc(bet)}</strong></a><br><span class="meta">${esc(r.sport)} · ${esc(r.game)}<br>Starts ${esc(time(r.commence_time))}<br>Experimental · ${esc(r.review)}</span></td><td>${esc(odds(r.price))}</td><td><time datetime="${esc(r.quoted_at)}">${esc(time(r.quoted_at))}</time></td><td>${esc(r.book_label||r.book)}</td></tr>`;
  }

  async function mount() {
    const root=document.getElementById('daily-picks');if(!root)return;
    const urls={NFL:'/props/top-picks.json',MLB:'/mlb/data/latest.json',NHL:'/nhl/data/latest.json',NHLBoard:'/nhl/data/candidates.json'};
    let feeds={},checked=null,loading=false;
    function render() {
      const now=Date.now(),result=collect(feeds,now);
      document.getElementById('daily-picks-rows').innerHTML=result.selected.map(rowHTML).join('')||'<tr><td colspan="4">No current bets qualify for today’s review list. See the feed status below; an empty list is a valid result.</td></tr>';
      document.getElementById('picks-status').textContent=`${result.selected.length} candidates for ${new Date(now).toLocaleDateString('en-US',{timeZone:'America/New_York',month:'long',day:'numeric'})} · ${checked?'Source boards checked '+time(checked):'Checking source boards'}.`;
      document.getElementById('picks-coverage').textContent=result.coverage.map(c=>`${c.sport}: ${c.message}`).join(' · ');
    }
    async function load() {
      if(loading)return;loading=true;
      try {
        const entries=await Promise.all(Object.entries(urls).map(async([sport,url])=>{
          try {const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error();return [sport,await response.json()];}
          catch {return [sport,null];}
        }));
        feeds=Object.fromEntries(entries);checked=new Date().toISOString();render();
      } finally {loading=false;}
    }
    await load();setInterval(render,30000);setInterval(load,300000);
    document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});
  }
  if(typeof module==='object'&&module.exports)module.exports={collect,rowHTML,day};
  else mount();
})(typeof window==='undefined'?globalThis:window);

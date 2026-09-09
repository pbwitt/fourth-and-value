/* Progressive enhancement of the server-rendered NFL snapshot. */
(() => {
  'use strict';
  const {fields, dictionary, rows: packed, topOnly, root, snapshotUpcoming, snapshotVerified, lastKickoff} = JSON.parse(document.getElementById('props-data').textContent);
  const DATA = packed.map(row=>Object.fromEntries(fields.map((field,i)=>[field,dictionary[field]?dictionary[field][row[i]]:row[i]])));
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = (v, digits=1) => v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(digits);
  const pct = v => v == null ? '—' : `${num(v*100)}%`;
  const odds = v => v == null ? '—' : `${v>0?'+':''}${num(v,0)}`;
  const unique = values => [...new Set(values.filter(Boolean))].sort((a,b)=>a.localeCompare(b));
  const books = unique(DATA.map(r=>r.bookmaker));
  const params = new URLSearchParams(location.search);
  const state = {q:params.get('q')||'', market:params.get('market')||'', game:params.get('game')||'',
    sort:params.get('sort')||(topOnly?'edge':'kickoff'), best:params.get('best')!=='0',
    positive:topOnly||params.get('positive')==='1', history:!topOnly&&params.get('history')==='1',
    books:new Set(params.has('books') ? params.get('books').split(',').filter(b=>books.includes(b)) : books), page:1};
  let current = [];
  const quoteFresh = r => {
    const age=Date.now()-Date.parse(r.last_update);
    return Number.isFinite(age)&&age>=0&&age<=48*3600000;
  };
  const upcoming = r => Date.parse(r.commence_time)>Date.now();
  const qualifies = r => upcoming(r)&&quoteFresh(r)&&r.model_status.startsWith('Calibration fitted')&&r.edge_bps>0;
  const options = (id, entries) => entries.forEach(([value,label])=>$(id).add(new Option(label,value)));
  options('market', unique(DATA.map(r=>r.market_std)).map(m=>[m,DATA.find(r=>r.market_std===m).market_label]));
  options('game',unique(DATA.map(r=>r.game)).map(g=>[g,g]));
  books.forEach((book,i)=>{
    const label=document.createElement('label'), check=document.createElement('input');
    check.type='checkbox';check.value=book;check.checked=state.books.has(book);check.id=`book-${i}`;
    label.append(check,document.createTextNode(DATA.find(r=>r.bookmaker===book).book_label||book));
    $('books').append(label);
    check.addEventListener('change',()=>{check.checked?state.books.add(book):state.books.delete(book);refresh();});
  });
  const sync = () => {
    ['q','market','game','sort'].forEach(id=>$(id).value=state[id]);
    ['best','positive','history'].forEach(id=>$(id).checked=state[id]);
    $('books').querySelectorAll('input').forEach(el=>el.checked=state.books.has(el.value));
    $('positive').disabled=topOnly; $('history').disabled=topOnly;
    $('book-count').textContent=`(${state.books.size} of ${books.length})`;
  };
  function filteredURL() {
    const url=new URL(location.href);url.search='';
    ['q','market','game','sort'].forEach(key=>{if(state[key])url.searchParams.set(key,state[key]);});
    if(!state.best)url.searchParams.set('best','0');
    if(state.positive)url.searchParams.set('positive','1');
    if(state.history)url.searchParams.set('history','1');
    if(state.books.size!==books.length)url.searchParams.set('books',[...state.books].join(','));
    return url;
  }
  function card(r,index) {
    const isPast=!upcoming(r), fresh=quoteFresh(r);
    const mean=r.market_std==='anytime_td'?'—':num(r.mu);
    const edge=r.edge_bps==null?'—':`${num(r.edge_bps/100)} pp`;
    const warning=isPast?'Game started · historical quote':!fresh?'Saved quote · confirm current price':'Quote checked '+new Date(r.last_update).toLocaleString();
    return `<article class="panel prop-card"><p class="meta">${esc(r.kick_et)} · ${esc(r.game)}</p>
      <h2>${esc(r.player)}</h2><div>${esc(r.market_label)}</div>
      <p class="betline">${esc(r.name)} ${r.point==null?'':num(r.point).replace(/\.0$/,'')} · ${odds(r.price)}</p>
      <strong>${esc(r.book_label)}</strong><p class="meta">${esc(warning)}</p>
      <span class="tag">${esc(r.model_status)}</span>
      <dl><dt>Model probability</dt><dd>${pct(r.model_prob)}</dd><dt>Book probability</dt><dd>${pct(r.mkt_prob)}</dd>
      <dt>Model edge</dt><dd class="${r.edge_bps>0?'positive':r.edge_bps<0?'negative':''}">${edge}</dd>
      <dt>Expected profit / $100</dt><dd>${r.ev_per_100==null?'—':'$'+num(r.ev_per_100,2)}</dd></dl>
      <details><summary>Line comparison &amp; assumptions</summary><dl>
      <dt>Model mean</dt><dd>${mean}</dd><dt>Median book line</dt><dd>${num(r.consensus_line)}</dd>
      <dt>Paired fair probability</dt><dd>${pct(r.prob_devig)}</dd><dt>Consensus at this line</dt><dd>${pct(r.consensus_prob)}</dd>
      <dt>Books with paired quotes</dt><dd>${num(r.book_count,0)}</dd><dt>Estimated push probability</dt><dd>${pct(r.push_prob)}</dd></dl>
      <p class="meta">Missing values mean no supported estimate. Line agreement alone does not establish positive expected value.</p></details>
      <div class="actions"><button type="button" data-copy="${index}">Copy bet</button><button type="button" data-track="${index}" ${isPast?'disabled':''}>Track bet</button></div></article>`;
  }
  function render() {
    let rows=DATA.filter(r=>state.books.has(r.bookmaker)&&(!state.market||r.market_std===state.market)&&
      (!state.game||r.game===state.game)&&(!state.q||`${r.player} ${r.game} ${r.book_label}`.toLowerCase().includes(state.q.toLowerCase()))&&
      (state.history||upcoming(r))&&(!state.positive||r.edge_bps>0)&&(!topOnly||qualifies(r)));
    // Apply book filters BEFORE choosing the best price; preserve alternate lines as distinct bets.
    if(state.best){
      const best=new Map();
      rows.forEach(r=>{const key=JSON.stringify([r.game_id||r.commence_time,r.player,r.market_std,r.name,r.point]);
        if(!best.has(key)||r.mkt_prob<best.get(key).mkt_prob)best.set(key,r);});
      rows=[...best.values()];
    }
    rows.sort((a,b)=>state.sort==='player'?a.player.localeCompare(b.player):
      state.sort==='edge'?(b.edge_bps??-Infinity)-(a.edge_bps??-Infinity):
      state.sort==='ev'?(b.ev_per_100??-Infinity)-(a.ev_per_100??-Infinity):Date.parse(a.commence_time)-Date.parse(b.commence_time)||a.player.localeCompare(b.player));
    const pageSize=24,pages=Math.max(1,Math.ceil(rows.length/pageSize));state.page=Math.min(state.page,pages);
    current=rows.slice((state.page-1)*pageSize,state.page*pageSize);
    $('count').textContent=`${rows.length.toLocaleString()} ${state.best?'distinct lines':'offers'} match your filters`;
    $('results').innerHTML=current.map(card).join('')||`<div class="empty"><h2>No ${topOnly?'qualifying picks':'matching props'}</h2><p>${topOnly?'Top Picks requires upcoming games, recent quote timestamps, player evidence, a fitted calibration curve and a positive edge.':state.books.size===0?'No sportsbooks selected. Select a book or reset filters.':'Try another player, reset your filters, or include started games to inspect this snapshot.'}</p><a href="${root}/props/">Compare all props</a></div>`;
    $('pager').hidden=pages===1; $('previous').disabled=state.page===1; $('next').disabled=state.page===pages;
    $('page-info').textContent=`Page ${state.page} of ${pages}`;
    const future=DATA.filter(upcoming);
    const hasUpcoming=topOnly?snapshotUpcoming>0&&Date.parse(lastKickoff)>Date.now():future.length>0;
    $('freshness').textContent=!hasUpcoming?'No upcoming NFL games in this snapshot.':(topOnly?snapshotVerified:future.every(quoteFresh))?
      'Sportsbook quote times are within the last 48 hours. Confirm the current line before using an estimate.':
      'Quote freshness is unverified or older than 48 hours. These are saved prices; check your sportsbook.';
    sync();
  }
  function refresh(){state.page=1;render();history.replaceState(null,'',filteredURL());}
  ['q','market','game','sort'].forEach(id=>$(id).addEventListener(id==='q'?'input':'change',e=>{state[id]=e.target.value;refresh();}));
  ['best','positive','history'].forEach(id=>$(id).addEventListener('change',e=>{state[id]=e.target.checked;refresh();}));
  $('all-books').onclick=()=>{state.books=new Set(books);refresh();};
  $('no-books').onclick=()=>{state.books.clear();refresh();};
  $('reset').onclick=()=>{Object.assign(state,{q:'',market:'',game:'',sort:topOnly?'edge':'kickoff',best:true,positive:topOnly,history:false,books:new Set(books)});refresh();};
  ['previous','next'].forEach(id=>$(id).onclick=()=>{state.page+=id==='next'?1:-1;render();$('count').scrollIntoView({block:'start'});});
  async function copy(text){try{await navigator.clipboard.writeText(text);$('feedback').textContent='Copied.';}catch{$('feedback').textContent='Copy unavailable. Select the address from your browser to share filters.';}}
  $('share').onclick=()=>copy(filteredURL().href);
  let trackingReady;
  const script=src=>new Promise((resolve,reject)=>{const el=document.createElement('script');el.src=src;el.onload=resolve;el.onerror=reject;document.head.append(el);});
  $('results').addEventListener('click',async e=>{
    const btn=e.target.closest('button[data-copy],button[data-track]');if(!btn)return;
    const r=current[Number(btn.dataset.copy??btn.dataset.track)];if(!r)return;
    if(btn.dataset.copy!=null){await copy(`${r.player} ${r.market_label}: ${r.name} ${r.point??''} at ${odds(r.price)} (${r.book_label}). ${r.game}. Saved quote; confirm price. ${location.origin}${new URL(root+'/',location.href).pathname}props/`);return;}
    if(!upcoming(r)){render();return;}
    btn.disabled=true;
    try{
      trackingReady??=script('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2').then(()=>script(`${root}/tracking/bet-tracking.js`));
      await trackingReady;
      if(!await window.getCurrentUser()){location.href=`${root}/tracking/`;return;}
      const entered=prompt('Stake amount ($):');if(entered===null)return;
      const stake=Number(entered);if(!Number.isFinite(stake)||stake<=0){$('feedback').textContent='Enter a valid positive stake.';return;}
      const date=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(r.commence_time));
      if(await window.autoTrackBet({league:'NFL',game_date:date,team_home:r.home_team,team_away:r.away_team,player:r.player,
        market_type:r.market_std,side:r.name,line:r.point,book:r.bookmaker,odds:r.price,stake_dollars:stake,
        model_prob:r.model_prob,edge_bps:r.edge_bps}))btn.textContent='Tracked';
    }catch{trackingReady=null;$('feedback').textContent='Tracking is unavailable. Try the Bet Tracker page.';}
    finally{btn.disabled=false;}
  });
  $('filters').hidden=false;render();setInterval(render,60000);
})();

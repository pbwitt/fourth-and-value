/* Arbitrage finder (docs/arbitrage/). Shows the scheduled scans each league's refresh publishes
   (docs/arbitrage/data/<sport>.json, written by scripts/arbitrage/scan.cjs), the paper ledger,
   and fresh scans an editor runs with Scan now. Fresh scans go through the editor-only
   live-odds Edge Function and are graded in the browser by the same engine
   (docs/assets/arbitrage.js) with the same published rules, so a lock means the same thing in
   both. A fresh scan replaces only what it covered: game lines for a league, or one game.

   Stakes and after-tax estimates follow the reader's settings. Track buttons open Bet
   Tracker's shared dialog (docs/assets/offer-tracker.js) for each bet separately. */
(()=>{
 const A=window.FVArbitrage,T=window.FVOfferTracker,db=window.supabaseClient,$=id=>document.getElementById(id);
 const SPORTS=['NFL','NBA','NHL','MLB'],GAME=new Set(['h2h','spreads','totals']),STALE=15*60e3,RECORD_DAYS=30;
 // Bet Tracker's NFL ledger names for the props Scan now can return.
 const NFL_LEDGER={player_pass_yds:'pass_yds',player_rush_yds:'rush_yds',player_reception_yds:'recv_yds',player_receptions:'receptions'};
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const money=n=>`${n<0?'−':''}$${Math.abs(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}`;
 const pct=n=>`${n>0?'+':n<0?'−':''}${Math.abs(n).toFixed(2)}%`;
 const odds=p=>p>0?`+${p}`:`−${Math.abs(p)}`;
 const signed=n=>n>0?`+${n}`:n<0?`−${Math.abs(n)}`:'0';
 const et=(t,opts)=>new Date(t).toLocaleString('en-US',{timeZone:'America/New_York',timeZoneName:'short',...opts});
 const when=t=>et(t,{weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'});
 const clock=t=>et(t,{hour:'numeric',minute:'2-digit'});
 const ago=ms=>ms<90e3?'just now':ms<90*60e3?`${Math.round(ms/60e3)} min ago`:ms<36*3600e3?`${Math.round(ms/3600e3)} h ago`:`${Math.round(ms/86400e3)} days ago`;
 const store={get(k){try{return localStorage.getItem('fv-arb-'+k);}catch{return null;}},set(k,v){try{localStorage.setItem('fv-arb-'+k,v);}catch{}}};

 const state={rules:null,scheduled:{},ledgers:{},fresh:[],games:{},busy:false};

 async function json(url){const r=await fetch(url,{cache:'no-cache'});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}

 // ---- Settings -------------------------------------------------------------------
 const settings=()=>({sport:$('sport').value,stake:Math.max(1,Number($('stake').value)||100),rate:Number($('rate').value),
  itemize:$('itemize').checked,licensed:$('licensed').checked,taxpositive:$('taxpositive').checked});
 function restore(){
  for(const id of ['sport','stake','rate']){
   const v=store.get(id),el=$(id);
   if(v!=null&&(el.tagName!=='SELECT'||[...el.options].some(o=>o.value===v)))el.value=v;
  }
  for(const id of ['itemize','licensed','taxpositive']){const v=store.get(id);if(v!=null)$(id).checked=v==='1';}
 }
 function remember(){
  for(const id of ['sport','stake','rate'])store.set(id,$(id).value);
  for(const id of ['itemize','licensed','taxpositive'])store.set(id,$(id).checked?'1':'0');
 }
 const tax=(pair,s)=>s.rate>0?A.afterTax(pair,s.stake,{rate:s.rate,itemize:s.itemize}):A.stakes(pair,s.stake).profit;

 // ---- Pairs on show: scheduled scans, minus whatever a fresh scan has replaced ----------
 const covers=(scan,pair)=>scan.sport===pair.sport&&(scan.event_id?pair.event_id===scan.event_id:pair.legs.every(l=>GAME.has(l.market)));
 function pairs(kind){
  const out=[];
  for(const sport of SPORTS){
   const scheduled=state.scheduled[sport];
   for(const p of scheduled?.[kind]||[])
    if(!state.fresh.some(f=>covers(f,p)))out.push({...p,source:{label:'scheduled',at:scheduled.scanned_at}});
  }
  state.fresh.forEach((f,i)=>{
   for(const p of f.result[kind])
    if(!state.fresh.slice(i+1).some(later=>covers(later,p)))out.push({...p,source:{label:'fresh',at:f.fetched_at}});
  });
  const s=settings(),now=Date.now();
  return out.filter(p=>(!s.sport||p.sport===s.sport)&&Date.parse(p.commence_time)>now&&(!s.licensed||p.licensed)
   &&(kind!=='locked'||!s.taxpositive||tax(p,s)>0))
   .sort((a,b)=>kind==='near'?a.implied_total-b.implied_total:b.return_pct-a.return_pct);
 }

 // ---- Rendering --------------------------------------------------------------------
 function betLabel(l,pair){
  const what=l.market==='h2h'?`${l.side} moneyline`:l.market==='spreads'?`${l.side} ${signed(l.line)}`:`${l.side} ${l.line}`;
  return pair.cross_market&&l.market!=='h2h'?`${what} (${l.market_label})`:what;
 }
 function kindTag(l){
  const kind=state.rules?.books?.[l.book]?.kind;
  return kind==='licensed'?'':kind==='offshore'?`<span class="tag warn" title="${esc(state.rules.kinds.offshore)}">offshore: ${esc(l.book_label)}</span>`
   :kind==='sweepstakes'?`<span class="tag warn" title="${esc(state.rules.kinds.sweepstakes)}">sweepstakes: ${esc(l.book_label)}</span>`
   :`<span class="tag warn">unclassified book: ${esc(l.book_label)}</span>`;
 }
 function reasonText(r){
  if(r.code==='rules_unknown')return `${r.book}’s rules for ${r.market} are not on file.`;
  if(r.code==='rules_differ')return 'The two books settle this market differently.';
  if(r.code==='quotes_apart')return `The quotes were taken ${r.seconds<120?r.seconds+' seconds':Math.round(r.seconds/60)+' minutes'} apart.`;
  return '';
 }
 function middleText(p){
  const m=p.middle;if(!m)return '';
  const span=m.from===m.to?`${m.from}`:`${m.from} to ${m.to}`;
  if(!m.team)return `If the result is ${span}, both bets win`;
  const lo=Math.min(Math.abs(m.from),Math.abs(m.to)),hi=Math.max(Math.abs(m.from),Math.abs(m.to)),by=lo===hi?`${lo}`:`${lo} to ${hi}`;
  return `If ${m.team} ${m.from>0?'wins':'loses'} by ${by}, both bets win`;
 }
 function ticket(p,l){
  const market=p.sport==='NFL'&&NFL_LEDGER[l.market]?NFL_LEDGER[l.market]:l.market;
  return {sport:p.sport,event_id:p.event_id,commence_time:p.commence_time,game:p.game,home_team:p.home_team,away_team:p.away_team,
   player:l.player||'',market,market_label:l.market_label,side:l.side,line:l.line,book:l.book,book_label:l.book_label,price:l.price,
   quoted_at:l.quoted_at,settlement_profile:l.settlement_profile,
   settlement_scope:'One side of an arbitrage pair. Confirm both books grade this market the same way before betting; record the other side separately.'};
 }
 function card(p,s,now){
  const st=A.stakes(p,s.stake),after=s.rate>0?A.afterTax(p,s.stake,{rate:s.rate,itemize:s.itemize}):null;
  const times=p.legs.map(l=>Date.parse(l.quoted_at)),newest=Math.max(...times),oldest=Math.min(...times);
  const tags=[p.status==='locked'?'<span class="tag ok">locked</span>':p.status==='check'?'<span class="tag warn">check first</span>':'<span class="tag">near miss</span>',
   p.tier==='assumed'?'<span class="tag" title="Standard rules assumed for at least one book; not yet checked against its published rules.">rules assumed</span>':'',
   p.kind==='middle'?'<span class="tag">middle</span>':'',p.cross_market?'<span class="tag">moneyline vs spread</span>':'',
   p.push_refund?'<span class="tag" title="A result exactly on the line refunds both bets.">can refund both</span>':'',
   ...p.legs.map(kindTag)].filter(Boolean).join(' ');
  const rows=p.legs.map((l,i)=>`<tr><td>${esc(betLabel(l,p))}</td><td>${esc(l.book_label)}</td><td class="num">${esc(odds(l.price))}</td>`
   +`<td class="num">${esc(money(st.legs[i].stake))}</td><td class="track-cell" data-pair="${esc(p.id)}" data-leg="${i}"></td></tr>`).join('');
  const middle=middleText(p);
  const aftertax=after==null?'':` After federal tax at ${Math.round(s.rate*100)}%${s.itemize?', itemizing':', standard deduction'}: <strong class="${after>0?'positive':'negative'}">${esc(money(after))}</strong>.`;
  const old=now-newest>STALE?`<p class="stale">Prices from ${esc(clock(newest))} (${esc(ago(now-newest))}). They have likely moved: rescan or check both books before betting.</p>`:'';
  const reasons=p.reasons?.length?`<p class="reasons">${p.reasons.map(r=>esc(reasonText(r))).join(' ')}</p>`:'';
  return `<article class="panel pair ${p.status}"><p class="meta">${esc(p.sport)} · ${esc(p.game)} · ${esc(when(p.commence_time))}</p>`
   +`<h3>${esc(p.subject)}</h3><p class="tags">${tags}</p>`
   +`<div class="table-wrap"><table><thead><tr><th>Bet</th><th>Book</th><th>Price</th><th>Stake</th><th><span class="visually-hidden">Track</span></th></tr></thead><tbody>${rows}</tbody></table></div>`
   +`<p class="result">${p.implied_total<1?`Pays <strong>${esc(money(st.payout))}</strong> whichever side wins: <strong class="positive">${esc(money(st.profit))}</strong> (${esc(pct(p.return_pct))}).`
     :`Holding both costs ${esc(money(-st.profit))} (${esc(pct(p.return_pct))}).`}`
   +`${middle?` ${esc(middle)}: ${esc(money(st.best))}.`:''}${p.implied_total<1?aftertax:''}</p>`
   +reasons+old
   +`<p class="meta">Quoted ${esc(clock(oldest))}${newest-oldest>=60e3?`–${esc(clock(newest))}`:''} · ${p.source.label==='fresh'?'fresh scan':'scheduled scan'} ${esc(clock(p.source.at))}</p></article>`;
 }
 function nearTable(list){
  if(!list.length)return '<p class="empty">No two-way pairs within 1% of a lock in these scans.</p>';
  return '<div class="table-wrap"><table><thead><tr><th>Game</th><th>Bet</th><th>Best price each side</th><th>Implied total</th><th>Return</th></tr></thead><tbody>'
   +list.slice(0,20).map(p=>`<tr><td>${esc(p.sport)} · ${esc(p.game)}<div class="meta">${esc(when(p.commence_time))}</div></td><td>${esc(p.subject)}</td>`
    +`<td>${p.legs.map(l=>`${esc(betLabel(l,p))} ${esc(odds(l.price))} at ${esc(l.book_label)}`).join('<br>')}</td>`
    +`<td class="num">${(100*p.implied_total).toFixed(2)}%</td><td class="num ${p.return_pct>=0?'positive':'negative'}">${esc(pct(p.return_pct))}</td></tr>`).join('')
   +'</tbody></table></div>';
 }
 function attachTracking(container,list){
  if(!T)return;
  container.querySelectorAll('.track-cell').forEach(cell=>{
   const p=list.find(x=>x.id===cell.dataset.pair),l=p?.legs[Number(cell.dataset.leg)];
   if(l)cell.append(T.button(ticket(p,l)));
  });
  T.refresh();
 }
 function scanList(){
  const s=settings(),items=[];
  for(const sport of SPORTS){
   if(s.sport&&s.sport!==sport)continue;
   const d=state.scheduled[sport];
   if(!d){items.push(`<li><strong>${sport}</strong>: no scheduled scan published yet.</li>`);continue;}
   const feed=d.feed_status==='ready'?'':` <span class="warn">The ${sport} feed’s last refresh failed; these are its last saved prices.</span>`;
   items.push(`<li><strong>${sport}</strong>: prices from ${esc(when(d.snapshot_at||d.scanned_at))}, scanned ${esc(ago(Date.now()-Date.parse(d.scanned_at)))}`
    +` · ${d.stats.quotes.toLocaleString()} quotes from ${d.stats.books.length} books.${feed}</li>`);
  }
  for(const f of state.fresh)if(!s.sport||s.sport===f.sport)
   items.push(`<li><strong>${f.sport}</strong>: fresh ${f.event_id?'scan of '+esc(f.label):'game-lines scan'} at ${esc(clock(f.fetched_at))}`
    +`${f.regions==='us,us2'?' with more books':''} · ${f.result.stats.quotes.toLocaleString()} quotes.</li>`);
  return items.join('')||'<li>No scans for this league yet.</li>';
 }
 function record(){
  const s=settings(),cutoff=Date.now()-RECORD_DAYS*86400e3,rows=[];
  let started=null;
  for(const sport of SPORTS){
   if(s.sport&&s.sport!==sport)continue;
   const entries=(state.ledgers[sport]?.entries||[]).filter(e=>Date.parse(e.first_seen)>=cutoff&&(!s.licensed||e.licensed));
   for(const e of state.ledgers[sport]?.entries||[])started=!started||e.first_seen<started?e.first_seen:started;
   const locked=entries.filter(e=>e.status==='locked');
   // Worst-case after tax for a two-sided pair: one bet wins, the other loses.
   const asPair=e=>({legs:e.first_legs,outcomes:[['win','loss'],['loss','win']]});
   const pre=locked.reduce((t,e)=>t+A.stakes(asPair(e),s.stake).profit,0),post=locked.reduce((t,e)=>t+tax(asPair(e),s),0);
   rows.push(`<tr><td>${sport}</td><td class="num">${entries.length}</td><td class="num">${locked.length}</td>`
    +`<td class="num">${entries.filter(e=>e.licensed).length}</td><td class="num">${entries.filter(e=>e.snapshots>=2).length}</td>`
    +`<td class="num">${esc(money(pre))}</td><td class="num ${post>0?'positive':post<0?'negative':''}">${s.rate>0?esc(money(post)):'—'}</td></tr>`);
  }
  return `<div class="table-wrap"><table><thead><tr><th>League</th><th>Pairs seen</th><th>Locked</th><th>Licensed books</th><th>Seen twice or more</th>`
   +`<th>Locked, before tax</th><th>Locked, after tax</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`
   +`<p class="meta">Dollar columns: every Locked pair once at your total stake, at first-seen prices.${started?` Ledger kept since ${esc(when(started))}.`:' The ledger fills as refreshes run.'}</p>`;
 }
 function render(){
  const s=settings(),now=Date.now();
  const locked=pairs('locked'),check=pairs('check'),near=pairs('near');
  $('count-locked').textContent=locked.length;$('count-check').textContent=check.length;$('count-near').textContent=near.length;
  const positive=locked.filter(p=>tax(p,s)>0).length;
  $('count-locked-note').textContent=s.rate>0?`${positive} of ${locked.length} positive after tax at your settings`:'Rules on file, quotes together, pays either way';
  $('scans').innerHTML=scanList();
  $('locked').innerHTML=locked.length?locked.map(p=>card(p,s,now)).join(''):'<p class="empty">No locked pairs in these scans. That is the usual state of the market; the closest are listed below.</p>';
  $('check').innerHTML=check.length?check.map(p=>card(p,s,now)).join(''):'<p class="empty">Nothing to check in these scans.</p>';
  $('near').innerHTML=nearTable(near);
  $('record').innerHTML=record();
  attachTracking($('locked'),locked);attachTracking($('check'),check);
 }

 async function load(){
  const [rules,...rest]=await Promise.allSettled([json('/arbitrage/data/rules.json'),
   ...SPORTS.map(s=>json(`/arbitrage/data/${s.toLowerCase()}.json`)),...SPORTS.map(s=>json(`/arbitrage/data/ledger/${s.toLowerCase()}.json`))]);
  state.rules=rules.status==='fulfilled'?rules.value:null;
  SPORTS.forEach((s,i)=>{
   if(rest[i].status==='fulfilled')state.scheduled[s]=rest[i].value;
   if(rest[i+SPORTS.length].status==='fulfilled')state.ledgers[s]=rest[i+SPORTS.length].value;
  });
  render();
 }

 // ---- Scan now (editors) --------------------------------------------------------------
 const say=t=>{$('scan-message').textContent=t;};
 const credits=(d)=>`${d.reused?'Reused prices from the last minute, so no credits were spent.':d.cost==null?'The cost was not reported.':`This scan cost ${d.cost} credit${d.cost===1?'':'s'}.`}`
  +`${d.remaining!=null?` ${Number(d.remaining).toLocaleString()} credits left.`:''}`;
 async function invoke(body){
  const {data,error}=await db.functions.invoke('live-odds',{body});
  if(error){let detail='';try{detail=(await error.context?.json())?.error||'';}catch{}
   if(/Unsupported sport|Invalid request/.test(detail))detail='Scan now needs the updated live-odds function. Redeploy it in Supabase.';
   throw Error(detail||'The scan failed. Try again in a moment.');}
  return data;
 }
 function gameOptions(){
  const sport=$('scan-sport').value,now=Date.now(),seen=new Map();
  for(const g of state.games[sport]||[])if(Date.parse(g.commence_time)>now)seen.set(g.id,g);
  const list=[...seen.values()].sort((a,b)=>Date.parse(a.commence_time)-Date.parse(b.commence_time));
  $('scan-game').innerHTML=list.map(g=>`<option value="${esc(g.id)}">${esc(`${g.away_team} @ ${g.home_team} · ${when(g.commence_time)}`)}</option>`).join('')
   ||'<option value="">No upcoming games listed; scan game lines first</option>';
  $('scan-event').disabled=state.busy||!list.length;
 }
 async function listGames(){
  const sport=$('scan-sport').value;
  try{const d=await invoke({sport});if(d?.sport===sport)state.games[sport]=[...(state.games[sport]||[]),...(d.games||[])];}
  catch(e){say(e.message);}
  gameOptions();
 }
 function costs(){
  const x=$('scan-more').checked?2:1;
  $('scan-lines-cost').textContent=`Moneylines, spreads and totals for every game in the next 8 days: ${3*x} credits.`;
  $('scan-event-cost').textContent=`Game lines and four player props for one game: up to ${7*x} credits.`;
 }
 async function scan(event){
  if(state.busy||!state.rules)return;
  const sport=$('scan-sport').value,more=$('scan-more').checked,id=event?$('scan-game').value:null;
  if(event&&!id)return;
  state.busy=true;$('scan-lines').disabled=true;$('scan-event').disabled=true;say('Fetching current prices…');
  try{
   const d=await invoke(event?{sport,event:id,more}:{sport,scan:'lines',more});
   const events=event?(d?.event?[d.event]:null):d?.events;
   if(!Array.isArray(events))throw Error('Scan now needs the updated live-odds function. Redeploy it in Supabase.');
   const at=Date.parse(d.fetched_at)||Date.now();
   const result=A.scan(A.fromOddsApi(events,sport),{rules:state.rules,sport,now:at});
   const label=event?events[0]?`${events[0].away_team} @ ${events[0].home_team}`:'one game':'';
   state.fresh=state.fresh.filter(f=>!(f.sport===sport&&(event?f.event_id===id:!f.event_id)));
   state.fresh.push({sport,event_id:id,label,fetched_at:new Date(at).toISOString(),regions:more?'us,us2':'us',result});
   if(!event)state.games[sport]=[...(state.games[sport]||[]),...events.map(e=>({id:e.id,commence_time:e.commence_time,home_team:e.home_team,away_team:e.away_team}))];
   $('sport').value=$('sport').value&&$('sport').value!==sport?'':$('sport').value;
   render();gameOptions();
   say(`${event?label:sport+' game lines'}: ${result.locked.length} locked, ${result.check.length} to check, from ${result.stats.quotes.toLocaleString()} quotes at ${result.stats.books.length} books, fetched ${clock(at)}. ${credits(d)}`);
  }catch(e){say(e.message);}
  finally{state.busy=false;$('scan-lines').disabled=false;gameOptions();}
 }
 let ticketNo=0;
 async function access(){
  if(!db){say('Scan now is unavailable: the sign-in service did not load.');return;}
  const mine=++ticketNo,{data}=await db.auth.getUser();
  if(mine!==ticketNo)return;
  const user=data?.user||null,editor=user?.app_metadata?.fv_editor===true;
  $('scan-login').hidden=!!user&&editor;$('scan-desk').hidden=!editor;
  if(!user){say('Sign in with an editor account to rescan current prices. Each scan spends paid Odds API credits.');return;}
  if(!editor){$('scan-login').hidden=true;say('This account does not have editor access. Scans spend paid credits, so they are limited to editors.');return;}
  $('scan-panel').open=true;say('Choose a league and scan.');
  if(!state.games[$('scan-sport').value])await listGames();
 }

 // ---- Wiring ----------------------------------------------------------------------
 restore();
 for(const id of ['sport','stake','rate','itemize','licensed','taxpositive'])$(id).addEventListener(id==='stake'?'input':'change',()=>{remember();render();});
 $('scan-lines').onclick=()=>scan(false);$('scan-event').onclick=()=>scan(true);
 $('scan-more').onchange=costs;
 $('scan-sport').onchange=()=>{gameOptions();if(!state.games[$('scan-sport').value])listGames();};
 $('scan-login').onsubmit=async e=>{
  e.preventDefault();const b=$('scan-login').querySelector('button');b.disabled=true;
  try{const {ok,error}=await window.signInWithEmail($('scan-email').value.trim());
   say(ok?'Check your email for the sign-in link. It brings you back to this page.':window.signInErrorMessage(error));}
  catch(error){say(window.signInErrorMessage?window.signInErrorMessage(error):'Sign-in failed.');}
  finally{b.disabled=false;}
 };
 $('scan-signout').onclick=async()=>{await window.signOut();state.games={};access().catch(()=>say('Unable to check sign-in.'));};
 if(db)db.auth.onAuthStateChange(()=>setTimeout(()=>access().catch(e=>say(e.message||'Unable to check sign-in.')),0));
 access().catch(e=>say(e.message||'Unable to check sign-in.'));
 load().catch(()=>{$('locked').innerHTML='<p class="empty">The latest scans could not be loaded. Refresh to try again.</p>';});
})();

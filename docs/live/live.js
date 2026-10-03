/* Private Live Odds page. An editor picks one NHL game and presses Run now: one
   call to the live-odds Edge Function (at most 7 Odds API credits; presses within
   60 seconds reuse the same prices for free) plus the free NHL scoreboard through
   live-stats. The comparison itself lives in docs/live/live-odds.js. Track buttons
   open Bet Tracker's shared dialog (docs/assets/offer-tracker.js) for the exact
   book, line and price shown; its pregame-only buttons are not used here.
   Authentication and the Edge Function protect the credits; hiding this page does not. */
(()=>{
 const $=id=>document.getElementById(id),db=window.supabaseClient,O=window.FVLiveOdds,feeds=window.FVLiveFeeds,T=window.FVOfferTracker;
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const say=text=>{$('message').textContent=text;};
 const clock=t=>new Date(t).toLocaleTimeString([],{hour:'numeric',minute:'2-digit',second:'2-digit'});
 const age=ms=>ms==null||!Number.isFinite(ms)?'—':ms<90e3?`${Math.max(0,Math.round(ms/1000))}s`:`${Math.round(ms/60e3)} min`;
 const pct=n=>n==null?'—':`${n>0?'+':n<0?'−':''}${Math.abs(n).toFixed(1)}%`;
 const credits=n=>n==null?'':`${Number(n).toLocaleString()} Odds API credits left.`;
 let games=[],last=null,busy=false,ticket=0;

 async function invoke(name,body){
  const {data,error}=await db.functions.invoke(name,{body});
  if(error){
   let detail='';
   try{detail=(await error.context?.json())?.error||'';}catch{}
   throw new Error(detail||'The request failed. Try again in a moment.');
  }
  return data;
 }

 function gameLabel(g,now=Date.now()){
  const t=Date.parse(g.commence_time);
  const when=new Date(t).toLocaleString([],{weekday:'short',hour:'numeric',minute:'2-digit'});
  return `${g.away_team} @ ${g.home_team} · ${t<=now?'started '+when:when}`;
 }

 async function loadGames(){
  say('Loading games…');
  const data=await invoke('live-odds',{});
  const now=Date.now();
  games=(data.games||[]).slice().sort((a,b)=>Date.parse(a.commence_time)-Date.parse(b.commence_time));
  $('game').innerHTML=games.map(g=>`<option value="${esc(g.id)}">${esc(gameLabel(g,now))}</option>`).join('');
  // Default to a game in progress, else the next one to start.
  const live=games.filter(g=>Date.parse(g.commence_time)<=now&&now-Date.parse(g.commence_time)<4*3600e3).pop();
  const pick=live||games.find(g=>Date.parse(g.commence_time)>now)||games[0];
  if(pick)$('game').value=pick.id;
  $('run').disabled=!games.length;
  say(games.length?`${games.length} NHL game${games.length===1?'':'s'} from earlier today through tomorrow. ${credits(data.remaining)}`
   :'No NHL games are listed from earlier today through tomorrow.');
 }

 async function scoreboard(g){
  const plan=feeds.scoreboardPlan('NHL',O.easternDate(g.commence_time));
  return O.findGame(plan.build(await Promise.all(plan.requests.map(r=>invoke('live-stats',r.proxy)))),g);
 }

 async function run(){
  const g=games.find(x=>x.id===$('game').value);
  if(!g||busy)return;
  busy=true;$('run').disabled=true;say('Fetching prices…');
  try{
   const [odds,score]=await Promise.all([invoke('live-odds',{event:g.id}),scoreboard(g).catch(()=>null)]);
   // A different game chosen while this one loaded: keep the new selection clean.
   if($('game').value!==g.id){say('Game changed. Press Run now for the selected game.');return;}
   const fetched=Date.parse(odds.fetched_at);
   last={game:g,odds,score,fetched,board:O.board(odds.event,fetched)};
   render();
   say(odds.reused?`Showing prices fetched at ${clock(fetched)}. Reused, so no credits were spent; new prices are available 60 seconds after that fetch.`
    :`Prices fetched at ${clock(fetched)}. ${odds.cost==null?'The cost was not reported.':`This run cost ${odds.cost} credit${odds.cost===1?'':'s'}.`} ${credits(odds.remaining)}`);
  }catch(e){say(e.message);}
  finally{busy=false;$('run').disabled=!games.length;}
 }

 function scoreLine(g,s){
  if(!s)return `<span>${esc(g.away_team)} @ ${esc(g.home_team)}</span><span class="detail">Live score unavailable</span>`;
  const team=t=>esc(t.abbrev||t.short||t.name);
  if(s.state==='pre')return `<span>${team(s.away)} @ ${team(s.home)}</span><span class="detail">Starts ${esc(new Date(s.start||g.commence_time).toLocaleTimeString([],{hour:'numeric',minute:'2-digit'}))}</span>`;
  return `<span>${team(s.away)} ${esc(s.away.score??'')} – ${esc(s.home.score??'')} ${team(s.home)}</span><span class="detail">${esc(s.detail)}</span>`;
 }

 function betCell(l){
  const name=l.player?`<strong>${esc(l.player)}</strong><span class="sub">${esc(l.market_label)} · ${esc(l.label)}`
   :`<strong>${esc(l.market_label)}</strong><span class="sub">${esc(l.label)}`;
  return `${name}${l.push_possible?' · can push':''}</span>`;
 }

 function priceCell(l,q){
  return `${esc(O.signed(q.price))}<span class="sub">${esc(q.book_label)}${q.stale?' · <span class="old">old</span>':''}</span>${track(l,q)}`;
 }

 const track=(l,q,text='Track')=>`<button type="button" class="track" data-line="${esc(l.id)}" data-book="${esc(q.book)}"`
  +` aria-label="Track bet: ${esc(l.player?l.player+' ':'')}${esc(l.market_label)} ${esc(l.label)} at ${esc(q.book_label)}">${text}</button>`;

 function booksCell(l){
  const items=l.quotes.map(q=>`<li>${esc(q.book_label)} ${esc(O.signed(q.price))}${q.fair_probability!=null?` · fair ${esc(O.signed(O.american(q.fair_probability)))}`:''}`
   +`${q.advantage!=null?` · ${esc(pct(q.advantage))}`:''} · ${esc(age(last.fetched-q.at))}${q.stale?' · <span class="old">old</span>':''} ${track(l,q)}</li>`).join('');
  return `<details><summary>${l.quotes.length}</summary><ul>${items}</ul></details>`;
 }

 function render(){
  const {game,score,board:b}=last;
  $('result').hidden=false;
  $('scoreline').innerHTML=scoreLine(game,score);
  $('asof').textContent=b.books.length
   ?`${b.lines.length} lines from ${b.books.length} book${b.books.length===1?'':'s'} (${b.books.join(', ')}). Newest quote ${clock(Date.parse(b.newest))}, ${age(last.fetched-Date.parse(b.newest))} before the fetch.`
   :'No book is offering prices on this game right now. Books may have closed its markets.';
  $('play-notice').hidden=!(score?.state==='live'&&!/\bInt\b/.test(score.detail||''));
  const flagged=b.lines.filter(l=>l.flagged);
  $('flags').innerHTML=flagged.length?flagged.map(l=>`<li><strong>${esc(l.player?`${l.player} · ${l.market_label}`:l.market_label)} · ${esc(l.label)}</strong>`
   +` — ${esc(O.signed(l.best.price))} at ${esc(l.best.book_label)}. Other books’ fair price ${esc(O.signed(O.american(l.best.other_probability)))}`
   +` (${l.best.other_books} books), so this price is <span class="pos">${esc(pct(l.best.advantage))}</span> against them. Quote ${esc(age(last.fetched-l.best.at))} old${l.push_possible?'; can push':''}.`
   +` ${track(l,l.best,'Track this bet')}</li>`).join('')
   :`<li class="none">${b.books.length<O.MIN_OTHER_BOOKS+1?'Fewer than four books are pricing this game, so there is nothing to compare yet.'
    :'No book beats the others by 2% or more right now.'}</li>`;
  const markets=[...new Set(b.lines.map(l=>l.market))];
  const chosen=$('market').value;
  $('market').innerHTML='<option value="">All markets</option>'+markets.map(m=>`<option value="${esc(m)}">${esc(O.MARKETS[m])}</option>`).join('');
  $('market').value=markets.includes(chosen)?chosen:'';
  table();
 }

 function table(){
  if(!last)return;
  const market=$('market').value,player=$('player').value.trim().toLowerCase();
  const rows=last.board.lines.filter(l=>(!market||l.market===market)&&(!player||l.player.toLowerCase().includes(player)));
  $('lines').innerHTML=rows.length?rows.map(l=>`<tr${l.flagged?' class="flagged"':''}><td>${betCell(l)}</td><td class="num">${priceCell(l,l.best)}</td>`
   +`<td class="num${l.flagged?' pos':''}">${esc(pct(l.best.advantage))}${l.best.advantage==null?`<span class="sub">${l.best.stale?'quote too old':'needs 3+ other books'}</span>`:''}</td>`
   +`<td class="num">${l.fair_odds==null?'—':esc(O.signed(l.fair_odds))}<span class="sub">${l.fair_books} book${l.fair_books===1?'':'s'}</span></td>`
   +`<td>${booksCell(l)}</td><td class="num">${esc(age(last.fetched-l.best.at))}</td></tr>`).join('')
   :'<tr><td colspan="6" class="meta">No lines match these filters.</td></tr>';
 }

 async function access(){
  const mine=++ticket;
  const {data}=await db.auth.getUser();
  if(mine!==ticket)return;
  const user=data?.user||null,editor=user?.app_metadata?.fv_editor===true;
  $('login').hidden=editor;$('desk').hidden=!editor;
  if(!user){say('Sign in with your editor account to use live odds.');return;}
  if(!editor){say('This account does not have editor access. Live odds spend paid credits, so they are limited to editors.');return;}
  if(!games.length)await loadGames();
 }

 $('login-form').onsubmit=async e=>{
  e.preventDefault();
  const button=$('login-form').querySelector('button');button.disabled=true;
  try{const {ok,error}=await window.signInWithEmail($('email').value.trim());
   say(ok?'Check your email for the sign-in link. It brings you back to this page.':window.signInErrorMessage(error));}
  catch(error){say(window.signInErrorMessage(error));}
  finally{button.disabled=false;}
 };
 // Track: the shared dialog records the bet you actually placed (price and stake) in Bet Tracker.
 function openTicket(event){
  const button=event.target.closest('button.track');
  if(!button||!last)return;
  const line=last.board.lines.find(l=>l.id===button.dataset.line),q=line?.quotes.find(x=>x.book===button.dataset.book);
  if(q)T.open(O.ticket(last.game,q));
 }
 $('flags').addEventListener('click',openTicket);$('lines').addEventListener('click',openTicket);
 $('run').onclick=run;
 $('game').onchange=()=>{last=null;$('result').hidden=true;};
 $('market').onchange=table;$('player').oninput=table;
 $('signout').onclick=async()=>{await window.signOut();games=[];last=null;$('result').hidden=true;access().catch(()=>say('Unable to check sign-in.'));};
 db.auth.onAuthStateChange(()=>setTimeout(()=>access().catch(e=>say(e.message||'Unable to check sign-in.')),0));
 access().catch(e=>say(e.message||'Unable to connect. Refresh to try again.'));
})();

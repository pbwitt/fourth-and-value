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
  const key=r=>JSON.stringify([r.sport,r.game_id,r.player,r.market_std||r.market,r.side,r.line,r.book,r.quoted_at]);
  const betLabel=r=>[r.player,r.side,r.line===null?'':r.market==='spreads'&&r.line>0?'+'+r.line:String(r.line),r.market_label].filter(Boolean).join(' · ');
  const reviewBetKey=r=>JSON.stringify([r.sport,String(r.game_id),r.player||'',r.market_std||r.market,r.side,r.line,r.book,r.commence_time]);
  const reviewKey=r=>JSON.stringify([reviewBetKey(r),r.price,r.quoted_at,r.forecast_at,
    r.model_prob??r.model_probability,r.push_prob??r.model_push_probability,
    r.consensus_prob??r.other_book_probability,r.model_version??r.model_status]);
  const reviewLabel=q=>({research_support:'Sourced support · analyst review needed',concern:'Sourced concern · review before deciding',needs_information:'More information needed'})[q?.status]||'Qualitative review needed';
  const completeReview=q=>q&&['research_support','concern','needs_information'].includes(q.status)&&
    typeof q.countercase==='string'&&Array.isArray(q.open_checks)&&q.open_checks.every(x=>typeof x==='string')&&
    Array.isArray(q.evidence)&&q.evidence.every(e=>e&&['source_id','direction','interpretation','represented_in'].every(k=>typeof e[k]==='string'));
  const safeSourceURL=url=>{try{const u=new URL(url);return u.protocol==='https:'&&!u.username&&!u.password&&['espn.com','cbssports.com','nhl.com','nfl.com','mlb.com'].some(h=>u.hostname===h||u.hostname.endsWith('.'+h));}catch{return false;}};
  const hasReview=r=>completeReview(r.qualitative_review)&&(r.sport==='NHL'?
    r.qualitative_review.offer_id===r.offer_id&&r.qualitative_review.forecast_id===r.forecast_id:
    !!r.reviewed_candidate);
  const citedEvidence=r=>r.qualitative_review.evidence.map(e=>({e,s:r.review_sources?.find(s=>s.source_id===e.source_id)}))
    .filter(({s})=>s&&safeSourceURL(s.url));
  const probability=n=>finite(n)&&n>=0&&n<=1;
  const pct=n=>probability(n)?(100*n).toFixed(1)+'%':'Unavailable';
  const number=n=>n.toFixed(1);
  const oldNFLReview=r=>r.sport==='NFL'&&hasReview(r)&&r.qualitative_review.prompt_version==='mlb-nfl-context-1';
  // All comparison percentages use the same outcome and exclude refunded pushes.
  // Keep missing push mass unknown; never substitute a book probability.
  function comparison(r) {
    const push=r.sport==='NFL'?r.push_prob:r.sport==='MLB'?r.model_push_probability:r.push_probability;
    const win=r.sport==='NFL'?null:r.sport==='MLB'?r.model_probability:r.final_probability;
    const model=r.sport==='NFL'?(probability(r.model_prob)?r.model_prob:null):
      probability(win)&&probability(push)&&push<1&&win+push<=1?win/(1-push):null;
    const market=r.sport==='NFL'?r.consensus_prob:r.sport==='MLB'?r.other_book_probability:r.market_probability;
    const books=r.sport==='NFL'?r.book_count:r.other_books;
    return {model,market:probability(market)&&finite(books)&&books>0?market:null,
      books,push:probability(push)?push:null,breakEven:1/decimal(r.price)};
  }
  function modelHTML(r) {
    const c=comparison(r);
    const mean=r.sport==='NFL'?r.mu:r.sport==='MLB'?r.model_mean:r.projected_mean;
    let projection='';
    if(finite(mean)&&r.market!=='h2h') {
      const label=r.sport==='MLB'?r.model_mean_label||'Projected '+r.market_label.toLowerCase():'Projected '+r.market_label.toLowerCase();
      projection=`<span class="estimate-detail">${esc(label)}: ${esc(number(mean))}</span>`;
    }
    const basis=r.sport==='NFL'?'Historical outcome calibration':r.sport==='NHL'&&r.final_probability!==r.independent_probability?'Final model estimate':'Independent model';
    return `<strong>${pct(c.model)}</strong><span class="estimate-detail">Win chance*</span>${projection}<span class="meta estimate-detail">${basis} · experimental${c.push>0?'<br>Push: '+pct(c.push):''}</span>`;
  }
  function marketHTML(r) {
    const c=comparison(r),line=finite(r.line)?' at '+r.line:' for this outcome';
    const median=r.sport==='NFL'&&finite(r.consensus_line)?`<span class="estimate-detail">Median line: ${esc(r.consensus_line)}</span>`:'';
    const basis=c.market===null?'Paired prices unavailable':`${c.books} ${r.sport==='NFL'?'paired':'other paired'} book${c.books===1?' only':'s'}${r.sport==='NFL'?' · includes listed book':''}`;
    return `<strong>${pct(c.market)}</strong><span class="estimate-detail">Win chance*${esc(line)}</span>${median}<span class="meta estimate-detail">${esc(basis)}</span>`;
  }
  function screenReason(r) {
    const c=comparison(r);
    if(c.model===null)return 'This offer passed its sport’s numerical screen; a comparable probability is unavailable.';
    const market=c.market===null?'':`, versus ${pct(c.market)} from ${c.books===1?(r.sport==='NFL'?'one paired book':'one other book'):(r.sport==='NFL'?'the market':'other books')}`;
    return `The experimental model estimates a ${pct(c.model)} win chance${market}. The offered ${odds(r.price)} needs ${pct(c.breakEven)} to break even, excluding pushes.`;
  }
  function researchStatus(sport,feed,selected,now) {
    const rows=selected.filter(r=>r.sport===sport),reviewed=rows.filter(hasReview);
    if(reviewed.length)return `${sport}: ${reviewed.length}/${rows.length} candidates reviewed${reviewed.some(r=>r.review_matches_current===false)?' · changed offers need recheck':''}`;
    if(!rows.length)return `${sport}: no current candidates`;
    const board=feed?.sports?.[sport];
    const status=board?.decision_date===day(now)?board.review_status:null;
    const label={no_usable_reporting:'relevant reporting unavailable',review_unavailable:'analysis unavailable',api_key_unavailable:'analysis unavailable',budget_exhausted:'analysis unavailable',outside_review_window:'review pending'}[status]||'review pending';
    return `${sport}: ${label}`;
  }

  function attachReview(r,feed,now) {
    const board=feed?.sports?.[r.sport];
    if(feed?.schema_version!==1||board?.decision_date!==day(now))return r;
    const previous=records(board.candidates).find(p=>p.review_bet_key===reviewBetKey(r));
    const q=previous?.qualitative_review;
    if(!completeReview(q)||q.offer_id!==previous.offer_id||q.forecast_id!==previous.forecast_id||
      !recent(q.reviewed_at,now,12*HOUR)||stamp(q.reviewed_at)>=stamp(r.commence_time))return r;
    const exact=previous.review_key===reviewKey(r);
    return {...r,qualitative_review:q,review_sources:records(board.sources),reviewed_candidate:previous,
      review_matches_current:exact,review:exact?reviewLabel(q):'Price or forecast changed · research needs recheck'};
  }

  function ticketData(r,price,stake) {
    price=Number(price);stake=Number(stake);
    if(!Number.isInteger(price)||Math.abs(price)<100)throw Error('Enter valid American odds, such as -110 or +150.');
    if(!Number.isFinite(stake)||stake<=0||Math.abs(stake*100-Math.round(stake*100))>1e-6)throw Error('Enter a positive stake in dollars and cents.');
    const teams=r.game.split(' @ '),home=r.home_team||teams[1],away=r.away_team||teams[0];
    if(!home||!away||!r.book)throw Error('This bet is missing its teams or sportsbook. Open Bet Tracker to enter it manually.');
    // The existing ledger's model_prob is conditional on non-push settlement.
    // NFL already provides that quantity; MLB/NHL provide an unconditional win.
    let probability=r.sport==='NFL'?r.model_prob:null;
    if(r.sport!=='NFL') {
      const win=r.sport==='MLB'?r.model_probability:r.final_probability;
      const push=r.sport==='MLB'?r.model_push_probability:r.push_probability;
      if(finite(win)&&finite(push)&&push>=0&&push<1&&win>=0&&win+push<=1)probability=win/(1-push);
    }
    if(!finite(probability)||probability<0||probability>1)probability=null;
    const nhlMarkets={player_goals:'goals',player_assists:'assists',player_points:'points',player_shots_on_goal:'sog',totals:'team_total'};
    return {league:r.sport,game_date:day(r.commence_time),team_home:home,team_away:away,
      player:r.player||null,market_type:r.sport==='NFL'?r.market_std:r.sport==='NHL'?(nhlMarkets[r.market]||r.market):r.market,
      side:['over','under'].includes(r.side.toLowerCase())?r.side.toLowerCase():r.side,
      line:r.line,book:r.book,odds:price,stake_dollars:stake,model_prob:probability,
      edge_bps:probability===null?null:(probability-1/decimal(price))*10000};
  }

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
          .map(r=>({...r,book:r.bookmaker,side:r.name,line:r.point,quoted_at:r.last_update,forecast_at:data.generated_at,
            game_id:r.game_id||r.game+'|'+r.commence_time,score:r.edge_bps,
            review:'Qualitative review needed',url:'/props/top.html?'+new URLSearchParams({q:r.player||r.game,market:r.market_std,game:r.game})}));
      } else if(sport==='MLB') {
        available=data?.status==='ready'&&!data.model_error&&recent(data.last_success_at,now,12*HOUR)&&recent(data.model_checked_at,now,90*MINUTE);
        if(available)candidates=rows(data).filter(r=>today(r,now)&&r.is_model_pick===true&&recent(r.quoted_at,now,90*MINUTE)&&
          finite(r.model_probability)&&r.model_probability>0&&r.model_probability<1&&finite(r.model_ev_pct)&&r.model_ev_pct>0)
          .map(r=>({...r,game_id:r.mlb_game_id||r.event_id,forecast_at:data.model_checked_at,model_version:data.model_version,score:r.model_ev_pct,review:'Qualitative review needed',
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
        selected.push(sport==='NHL'?{...r,sport,review_sources:records(board.sources)}:attachReview({...r,sport},feeds.Reviews,now));
      }
      coverage.push({sport,count,available,message:!available?'Current model list unavailable or expired':count?`${count} review candidate${count===1?'':'s'}`:'No qualifying games remaining today'});
    }
    return {selected,coverage};
  }

  function researchHTML(r) {
    const q=r.qualitative_review;
    if(!hasReview(r))return '';
    const items=citedEvidence(r).map(({e,s})=>{
      return `<li><strong>${esc(e.direction)}:</strong> ${esc(e.interpretation)} <a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a> <span class="meta">Published ${esc(time(s.published_at))} · may already be reflected in ${esc(e.represented_in.replaceAll('_',' '))}.</span></li>`;
    }).join('');
    const prev=r.reviewed_candidate;
    const original=prev?`<p class="meta">Reviewed ${esc(odds(prev.price))} at ${esc(time(prev.quoted_at))}. ${r.review_matches_current?'Matches this offer and forecast.':'Current price or forecast differs; this is earlier context, not a review of the current offer.'}</p>`:'';
    const correction=oldNFLReview(r)?'<p class="notice">Method correction: this earlier note was given an incorrect description of the NFL model. Its probabilities are calibrated to historical results, not current market prices. Any claim below that it is “market-calibrated” is incorrect. The forecast itself is unchanged.</p>':'';
    return `<details class="pick-research"><summary>Fourth &amp; Value analysis · ${esc(time(q.reviewed_at))}</summary><p><strong>Why it surfaced:</strong> ${esc(screenReason(r))}</p>${original}${correction}${items?`<ul>${items}</ul>`:'<p><strong>Additional supporting context unverified.</strong> The numerical screen flagged this offer, but the reporting reviewed did not verify additional context supporting the bet. It remains a candidate for further review.</p>'}<p><strong>Case against:</strong> ${esc(q.countercase)}</p><p><strong>Check before deciding:</strong></p><ul>${q.open_checks.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><p class="meta">AI-assisted research; human verification still required. This review does not change the model probability or establish a betting edge.</p></details>`;
  }

  function summaryHTML(selected) {
    if(!selected.length)return '<p>No current candidates are available to summarize. The feed status below shows whether games have started, prices have expired, or a model list is unavailable.</p>';
    const reviewed=selected.filter(hasReview);
    if(!reviewed.length)return `<p>${selected.length} current candidate${selected.length===1?' is':'s are'} awaiting our analysis. These offers passed the numerical screen; the reporting review is still pending.</p>`;
    // Cover the leading reviewed candidate in each sport, then fill up to three
    // paragraphs in existing model order. No new score or cross-sport ranking.
    const featured=[];
    for(const sport of ['NFL','MLB','NHL']) {
      const row=reviewed.find(r=>r.sport===sport);if(row)featured.push(row);
    }
    for(const row of reviewed)if(featured.length<3&&!featured.includes(row))featured.push(row);
    const paragraphs=featured.map(r=>{
      const q=r.qualitative_review,prior=r.reviewed_candidate||r;
      const status={research_support:'Our review found relevant supporting reporting; analyst verification is still needed.',concern:'Our review found a concern to resolve.',needs_information:'The reporting reviewed has not yet established a supporting case for this bet.'}[q.status];
      const evidence=citedEvidence(r).find(({e})=>['supports','concern'].includes(e.direction));
      const source=evidence?` ${esc(evidence.e.interpretation)} <a href="${esc(evidence.s.url)}" target="_blank" rel="noopener noreferrer">Source</a>.`:'';
      const changed=r.review_matches_current===false?' <strong>The price or forecast has changed since this review; reassess the current offer.</strong>':'';
      const original=changed?` This review assessed ${esc(odds(prior.price))} quoted ${esc(time(prior.quoted_at))}.`:'';
      const check=q.open_checks[0]?` <strong>Still to check:</strong> ${esc(q.open_checks[0])}`:'';
      // Keep long countercases, source timestamps and offer metadata in the full
      // review/table. No new AI call, shortened quotation or invented narrative.
      const countercase=q.status==='concern'&&!oldNFLReview(r)?` ${esc(q.countercase)}`:'';
      return `<p class="pick-summary-paragraph"><strong class="summary-bet">${esc(betLabel(r).replaceAll(' · ',' '))}</strong><span class="meta summary-game">${esc(r.sport)} · ${esc(r.game)}</span>${esc(screenReason(r))} ${status}${changed}${original}${source}${countercase}${check} <a class="read-pick-review" href="#pick-review-${selected.indexOf(r)}">Read our full analysis</a>.</p>`;
    }).join('');
    return paragraphs+`<p class="meta">${reviewed.length} of ${selected.length} current candidates reviewed; ${featured.length} summarized here. Full findings and remaining checks appear under each bet. A completed review is not bet approval.</p>`;
  }

  function rowHTML(r,index=0,saved=false) {
    const research=researchHTML(r);
    return `<tr><td><a href="${esc(r.url)}"><strong>${esc(betLabel(r))}</strong></a><br><span class="meta">${esc(r.sport)} · ${esc(r.game)}<br>Starts ${esc(time(r.commence_time))}<br>Experimental · ${esc(r.review)}</span></td><td class="pick-estimate">${modelHTML(r)}</td><td class="pick-estimate">${marketHTML(r)}</td><td>${esc(odds(r.price))}<span class="meta estimate-detail">${pct(comparison(r).breakEven)} break-even*</span></td><td><time datetime="${esc(r.quoted_at)}">${esc(time(r.quoted_at))}</time></td><td>${esc(r.book_label||r.book)}<br><button type="button" class="track-pick secondary" data-track-pick="${index}" ${saved?'disabled':''} aria-label="${esc((saved?'Tracked: ':'Track bet: ')+betLabel(r))}">${saved?'Tracked':'Track bet'}</button></td></tr>${research?`<tr class="pick-research-row" id="pick-review-${index}"><td colspan="6">${research}</td></tr>`:''}`;
  }

  async function mount() {
    const root=document.getElementById('daily-picks');if(!root)return;
    const urls={NFL:'/props/top-picks.json',MLB:'/mlb/data/latest.json',NHL:'/nhl/data/latest.json',NHLBoard:'/nhl/data/candidates.json',Reviews:'/briefing/reviews.json'};
    let feeds={},checked=null,loading=false,current=[],draft=null,trackingReady=null,saving=false;
    const tickets=new Map(),dialog=document.getElementById('pick-tracker'),form=document.getElementById('track-bet-form');
    const $=id=>document.getElementById(id);
    function render() {
      const now=Date.now(),result=collect(feeds,now);
      current=result.selected;
      const openReviews=new Set(Array.from(root.querySelectorAll('.pick-research[open]')).map(el=>el.closest('tr').dataset.reviewKey));
      document.getElementById('daily-picks-rows').innerHTML=current.map((r,i)=>rowHTML(r,i,tickets.get(key(r))?.saved)).join('')||'<tr><td colspan="6">No current bets qualify for today’s review list. See the feed status below; an empty list is a valid result.</td></tr>';
      current.forEach((r,i)=>{const row=$('pick-review-'+i);if(row){row.dataset.reviewKey=key(r);row.querySelector('details').open=openReviews.has(key(r));}});
      const summary=$('picks-analysis-text');if(summary)summary.innerHTML=summaryHTML(current);
      document.getElementById('picks-status').textContent=`${result.selected.length} candidates for ${new Date(now).toLocaleDateString('en-US',{timeZone:'America/New_York',month:'long',day:'numeric'})} · ${checked?'Source boards checked '+time(checked):'Checking source boards'}.`;
      document.getElementById('picks-coverage').textContent=result.coverage.map(c=>`${c.sport}: ${c.message}`).join(' · ');
      const research=document.getElementById('picks-research-status');
      if(research)research.textContent=['NFL','MLB','NHL'].map(s=>researchStatus(s,feeds.Reviews,current,now)).join(' · ');
      if(draft&&dialog.open&&!saving&&!draft.saved&&!collect(feeds,now).selected.some(r=>key(r)===key(draft.row))) {
        $('track-quote').textContent=`Saved quote: ${odds(draft.row.price)} at ${time(draft.row.quoted_at)}. This offer has expired or changed. Enter the price of the bet you actually placed.`;
      }
    }
    root.addEventListener('click',event=>{
      const link=event.target.closest('.read-pick-review');if(!link)return;
      const row=document.getElementById(link.hash.slice(1)),detail=row?.querySelector('details');
      if(detail){detail.open=true;detail.querySelector('summary').focus();}
    });
    const script=src=>new Promise((resolve,reject)=>{const el=document.createElement('script');el.src=src;const fail=()=>{clearTimeout(timer);el.remove();reject(Error('Bet Tracker could not load. Please try again.'));};const timer=setTimeout(fail,15000);el.onload=()=>{clearTimeout(timer);resolve();};el.onerror=fail;document.head.append(el);});
    async function tracker() {
      if(!trackingReady)trackingReady=(async()=>{
        if(!window.supabase?.createClient)await script('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2');
        if(!window.saveTrackedBet)await script('/tracking/bet-tracking.js?v=3');
      })().catch(error=>{trackingReady=null;throw error;});
      await trackingReady;
    }
    $('daily-picks-rows').addEventListener('click',event=>{
      const button=event.target.closest('[data-track-pick]');if(!button)return;
      const row=current[Number(button.dataset.trackPick)];if(!row)return;
      const itemKey=key(row);
      if(!collect(feeds,Date.now()).selected.some(r=>key(r)===itemKey)){render();return;}
      draft=tickets.get(itemKey)||{id:crypto.randomUUID(),row:{...row},saved:false};tickets.set(itemKey,draft);
      form.reset();$('track-save').disabled=!!draft.saved;$('track-feedback').textContent='';$('track-signin').hidden=true;
      $('track-bet-description').textContent=`${row.sport} · ${row.game} · ${betLabel(row)} · ${row.book_label||row.book}`;
      $('track-quote').textContent=`Saved quote: ${odds(row.price)} at ${time(row.quoted_at)}. Confirm the actual price below.`;
      $('track-review').textContent=`Review status: ${row.review}.`;
      $('track-grading').textContent=row.sport==='MLB'||(row.sport==='NHL'&&['h2h','spreads'].includes(row.market))?'This market can be logged, but automatic result grading is not connected yet. The bet will be saved as pending.':'';
      $('track-odds').value=row.price;dialog.showModal();$('track-stake').focus();
    });
    $('track-cancel').addEventListener('click',()=>{if(!saving)dialog.close();});
    dialog.addEventListener('cancel',event=>{if(saving)event.preventDefault();});
    form.addEventListener('submit',async event=>{
      event.preventDefault();if(saving||!draft||draft.saved||!form.reportValidity())return;
      let ticket;
      try {ticket={...ticketData(draft.row,$('track-odds').value,$('track-stake').value),id:draft.id};}
      catch(error){$('track-feedback').textContent=error.message;return;}
      saving=true;$('track-feedback').textContent='Saving…';$('track-signin').hidden=true;
      for(const id of ['track-save','track-cancel','track-odds','track-stake','track-confirm'])$(id).disabled=true;
      try {
        await tracker();const result=await window.saveTrackedBet(ticket);
        if(result.ok){draft.saved=true;$('track-feedback').textContent='Saved to your Bet Tracker. Qualitative review status is unchanged.';render();}
        else {$('track-feedback').textContent=result.error;$('track-signin').hidden=!result.needsSignIn;}
      } catch(error){$('track-feedback').textContent='The save could not be confirmed. Check Bet Tracker before retrying; this form retains the same ticket reference.';}
      finally {
        saving=false;
        for(const id of ['track-cancel','track-odds','track-stake','track-confirm'])$(id).disabled=false;
        $('track-save').disabled=!!draft.saved;
      }
    });
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
  if(typeof module==='object'&&module.exports)module.exports={collect,rowHTML,day,ticketData,reviewKey,reviewBetKey,summaryHTML,comparison,researchStatus};
  else mount();
})(typeof window==='undefined'?globalThis:window);

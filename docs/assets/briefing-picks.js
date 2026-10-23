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
  const assessment=q=>q?.assessment&&['consider','wait','pass'].includes(q.assessment.verdict)&&
    ['reason','model_case','price_case','context_case'].every(k=>typeof q.assessment[k]==='string')&&
    Array.isArray(q.assessment.blocking_checks)&&q.assessment.blocking_checks.every(x=>typeof x==='string')?q.assessment:null;
  const displayReview=r=>String(r.review||'').replace(/ · (?:human|analyst) review (?:needed|required)/gi,'');
  const verdictLabel=a=>({consider:'Consider',wait:'Needs review',pass:'Pass · case not supported'})[a.verdict];
  const reviewLabel=q=>assessment(q)?verdictLabel(assessment(q)):({research_support:'Sourced support',concern:'Sourced concern · review before deciding',needs_information:'Reporting reviewed · full assessment pending'})[q?.status]||'Qualitative review needed';
  const completeReview=q=>q&&['research_support','concern','needs_information'].includes(q.status)&&
    typeof q.countercase==='string'&&Array.isArray(q.open_checks)&&q.open_checks.every(x=>typeof x==='string')&&
    Array.isArray(q.evidence)&&q.evidence.every(e=>e&&['source_id','direction','interpretation','represented_in'].every(k=>typeof e[k]==='string'));
  const safeSourceURL=url=>{try{const u=new URL(url);return u.protocol==='https:'&&!u.username&&!u.password&&["espn.com","cbssports.com","actionnetwork.com","covers.com","vsin.com","nhl.com","nfl.com","mlb.com","azcardinals.com","atlantafalcons.com","baltimoreravens.com","buffalobills.com","panthers.com","chicagobears.com","bengals.com","clevelandbrowns.com","dallascowboys.com","denverbroncos.com","detroitlions.com","packers.com","houstontexans.com","colts.com","jaguars.com","chiefs.com","raiders.com","chargers.com","therams.com","miamidolphins.com","vikings.com","patriots.com","neworleanssaints.com","giants.com","newyorkjets.com","philadelphiaeagles.com","steelers.com","49ers.com","seahawks.com","buccaneers.com","tennesseetitans.com","commanders.com"].some(h=>u.hostname===h||u.hostname.endsWith('.'+h));}catch{return false;}};
  const hasReview=r=>completeReview(r.qualitative_review)&&(r.sport==='NHL'?
    r.qualitative_review.offer_id===r.offer_id&&r.qualitative_review.forecast_id===r.forecast_id:
    !!r.reviewed_candidate);
  const citedEvidence=r=>r.qualitative_review.evidence.map(e=>({e,s:r.review_sources?.find(s=>s.source_id===e.source_id)}))
    .filter(({s})=>s&&safeSourceURL(s.url));
  const probability=n=>finite(n)&&n>=0&&n<=1;
  const pct=n=>probability(n)?(100*n).toFixed(1)+'%':'Unavailable';
  const number=n=>finite(n)?n.toFixed(1):'Unavailable';
  const oldNFLReview=r=>r.sport==='NFL'&&hasReview(r)&&r.qualitative_review.prompt_version==='mlb-nfl-context-1';
  // All comparison percentages use the same outcome and exclude refunded pushes.
  // Keep missing push mass unknown; never substitute a book probability.
  function comparison(r) {
    // NBA rows (board sort only; NBA is not on the card) use MLB's field names.
    const mlbFields=r.sport==='MLB'||r.sport==='NBA';
    const push=r.sport==='NFL'?r.push_prob:mlbFields?r.model_push_probability:r.push_probability;
    const win=r.sport==='NFL'?null:mlbFields?r.model_probability:r.final_probability;
    const model=r.sport==='NFL'?(probability(r.model_prob)?r.model_prob:null):
      probability(win)&&probability(push)&&push<1&&win+push<=1?win/(1-push):null;
    const market=r.sport==='NFL'?r.consensus_prob:mlbFields?r.other_book_probability:r.market_probability;
    const books=r.sport==='NFL'?r.book_count:r.other_books;
    return {model,market:probability(market)&&finite(books)&&books>0?market:null,
      books,push:probability(push)?push:null,breakEven:1/decimal(r.price)};
  }
  // Market blend (docs/model-improvement-plan.md, Phase 3), in log-odds against the
  // exact-line no-vig consensus. Model weights by sport (Step 2c, 2026-10-14): fitted
  // on graded lines, NHL and NFL props scored best near 0 (90% range up to about 0.2)
  // and MLB game lines trailed the market, so they get 0.10; MLB props beat the market
  // over six games and keep 0.25 until more data. A pick needs 1% expected value at
  // its own price: 3% or more is high confidence, 1-3% moderate. The tier describes
  // the size of the blended EV, not a chance of winning. At least two books must post
  // that exact line, counting the offered book (a lone alternate line has no market to
  // blend with). The model must give the side at least the market's chance: a better
  // price alone is a Market Watch signal, not a pick. Expected value refunds pushes.
  // NBA gets the NHL rule its shadow grade scores (scripts/nba/grade_shadow.py). It
  // serves only the NBA boards' Best value sort; NBA rows never enter collect().
  const BLEND={version:'market-blend-3',weights:{NFL:.10,NHL:.10,MLB:.25,MLB_game:.10,NBA:.10},minEV:.01,highEV:.03,maxEV:.30,minBooks:2};
  const GAME_MARKETS=['h2h','spreads','totals'];
  const blendWeight=r=>r.sport==='MLB'&&GAME_MARKETS.includes(r.market)?BLEND.weights.MLB_game:BLEND.weights[r.sport]??BLEND.weights.NFL;
  const logit=p=>Math.log(p/(1-p)),sigmoid=x=>1/(1+Math.exp(-x));
  const clip=p=>Math.min(1-1e-6,Math.max(1e-6,p));
  function blend(r,modelOverride) {
    const c=comparison(r),model=probability(modelOverride)?modelOverride:c.model,weight=blendWeight(r);
    if(model===null)return {status:'no_model'};
    // NFL's book_count includes the offered book; MLB/NHL count only the other books.
    const lineBooks=finite(c.books)?c.books+(r.sport==='NFL'?0:1):null;
    if(c.market===null||lineBooks===null||lineBooks<BLEND.minBooks)return {status:'thin_market',model,market:c.market,books:c.books};
    if(model<c.market)return {status:'model_below_market',version:BLEND.version,weight,model,market:c.market,books:c.books,line_books:lineBooks};
    const final=sigmoid(weight*logit(clip(model))+(1-weight)*logit(clip(c.market)));
    const ev=(1-(c.push??0))*(final*decimal(r.price)-1);
    return {status:ev>=BLEND.minEV?'qualifies':'below_threshold',confidence:ev>=BLEND.highEV?'high':ev>=BLEND.minEV?'moderate':null,
      version:BLEND.version,weight,model,market:c.market,books:c.books,line_books:lineBooks,final,ev,threshold:BLEND.minEV};
  }
  const signedPct=n=>(n>=0?'+':'')+(100*n).toFixed(1)+'%';
  const barPct=n=>(100*n).toFixed(0)+'%';
  // Editions published before the tiers carry no label; their blended picks all cleared 3%.
  const confidenceOf=b=>b?.confidence||(finite(b?.ev)&&b.ev>=BLEND.highEV?'high':null);
  const CONFIDENCE={high:'High confidence',moderate:'Moderate confidence'};
  function confidenceReason(b) {
    const c=confidenceOf(b);
    if(c==='high')return ` That clears the ${barPct(BLEND.highEV)} bar, so it is a high-confidence pick.`;
    if(c==='moderate')return ` That clears the ${barPct(BLEND.minEV)} floor for a pick but not the ${barPct(BLEND.highEV)} high-confidence bar, so it is a moderate-confidence pick: there is less cushion if the model is too optimistic.`;
    return '';
  }
  // Player Props and Game Lines boards sort "Best value" by this same blend: picks by
  // blended EV, then the research tier above 30%, then offers under the 1% floor. Offers
  // it cannot score (no model, one book, model below market) keep game order after them.
  // It is an ordering only: none of the card's health checks, reviews or caps apply.
  const boardTier=b=>b?.status==='qualifies'?(b.ev>BLEND.maxEV?1:0):b?.status==='below_threshold'?2:3;
  const boardOrder=(a,b)=>boardTier(a)-boardTier(b)||(boardTier(a)<3?b.ev-a.ev:0);
  function boardBlendHTML(b) {
    if(b?.status==='model_below_market')return `<p class="meta">Our blend: model ${pct(b.model)} is below the market's ${pct(b.market)} · not a pick</p>`;
    if(b?.status==='thin_market')return '<p class="meta">Our blend: fewer than two books post this line · not scored</p>';
    if(!finite(b?.final))return '';
    const note=b.status==='below_threshold'?`below the ${barPct(BLEND.minEV)} floor`:
      b.ev>BLEND.maxEV?`above ${barPct(BLEND.maxEV)}, research only`:CONFIDENCE[confidenceOf(b)];
    return `<p class="meta"><strong>Our blend:</strong> ${pct(b.final)} · ${signedPct(b.ev)} expected value · ${esc(note)}</p>`;
  }
  function blendHTML(r) {
    const b=r.blend,c=confidenceOf(b);
    if(!b||!finite(b.final))return '';
    return `<span class="meta estimate-detail">Blended ${pct(b.final)} · ${signedPct(b.ev)} expected value${c?' · '+CONFIDENCE[c]:''}</span>`;
  }
  function modelHTML(r) {
    const c=comparison(r);
    if(r.model_withheld||c.model===null)return `<strong>Not established</strong><span class="estimate-detail">${r.model_withheld?'Forecast withheld: '+esc(r.model_withheld):'Independent research candidate; no eligible probability model'}</span><span class="meta">Prospective research only</span>`;
    const mean=r.sport==='NFL'?r.mu:r.sport==='MLB'?r.model_mean:r.projected_mean;
    // Lines sit near the median; a skewed count's average can clear a line the model expects to miss.
    const median=r.sport==='NHL'?r.projected_median:r.sport==='NFL'?null:r.model_median;
    let projection='';
    if(finite(mean)&&r.market!=='h2h') {
      const label=r.sport==='MLB'?r.model_mean_label||'Projected '+r.market_label.toLowerCase():'Projected '+r.market_label.toLowerCase();
      const middle=finite(median)?` · median ${esc(Number.isInteger(median)?median:number(median))}`:'';
      projection=`<span class="estimate-detail">${esc(label)}: average ${esc(number(mean))}${middle}</span>`;
    }
    const basis=r.sport==='NFL'?'Historical outcome calibration':r.sport==='NHL'&&r.final_probability!==r.independent_probability?'Final model estimate':'Independent model';
    return `<strong>${pct(c.model)}</strong><span class="estimate-detail">Win chance*</span>${projection}<span class="meta estimate-detail">${basis} · experimental${c.push>0?'<br>Push: '+pct(c.push):''}</span>`;
  }
  function marketHTML(r) {
    const c=comparison(r),line=finite(r.line)?' at '+r.line:' for this outcome';
    const median=r.sport==='NFL'&&finite(r.consensus_line)?`<span class="estimate-detail">Median line: ${esc(r.consensus_line)}</span>`:'';
    const basis=c.market===null?'Paired prices unavailable':`${c.books} ${r.sport==='NFL'?'paired':'other paired'} book${c.books===1?' only':'s'}${r.sport==='NFL'?' · includes listed book':''}`;
    return `<strong>${pct(c.market)}</strong><span class="estimate-detail">Win chance*${esc(line)}</span>${median}<span class="meta estimate-detail">${esc(basis)}</span>${pinnacleHTML(r.pinnacle)}`;
  }
  function screenReason(r) {
    const c=comparison(r);
    if(c.model===null||r.model_withheld)return 'Independent research surfaced this offer. A reliable model win probability and numerical edge are not established.';
    const market=c.market===null?'':`, versus ${pct(c.market)} from ${c.books===1?(r.sport==='NFL'?'one paired book':'one other book'):(r.sport==='NFL'?'the market':'other books')}`;
    const w=finite(r.blend?.weight)?r.blend.weight:blendWeight(r);
    const b=r.blend&&finite(r.blend.final)?` Weighted ${Math.round(100*w)}% model and ${Math.round(100*(1-w))}% market, that is ${pct(r.blend.final)}, or ${signedPct(r.blend.ev)} expected value at this price.${confidenceReason(r.blend)}`:'';
    return `The experimental model estimates a ${pct(c.model)} win chance${market}. The offered ${odds(r.price)} needs ${pct(c.breakEven)} to break even, excluding pushes.${b}`;
  }
  function researchStatus(sport,feed,selected,now) {
    const rows=selected.filter(r=>r.sport===sport),reviewed=rows.filter(hasReview);
    if(reviewed.length)return `${sport}: ${reviewed.length}/${rows.length} candidates reviewed${reviewed.some(r=>r.review_matches_current===false)?' · changed offers need recheck':''}`;
    if(!rows.length)return `${sport}: no current candidates`;
    const board=feed?.sports?.[sport];
    const status=board?.decision_date===day(now)?board.review_status:null;
    const label={no_usable_reporting:'relevant reporting unavailable',review_unavailable:'analysis unavailable',api_key_unavailable:'analysis unavailable',budget_exhausted:'daily research budget reached · review queued',outside_review_window:'review pending'}[status]||'review pending';
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
    if(r.model_withheld||!finite(probability)||probability<0||probability>1)probability=null;
    const nhlMarkets={player_goals:'goals',player_assists:'assists',player_points:'points',player_shots_on_goal:'sog'};
    return {league:r.sport,game_date:day(r.commence_time),team_home:home,team_away:away,
      player:r.player||null,market_type:r.sport==='NFL'?(r.market_std||r.market):r.sport==='NHL'?(nhlMarkets[r.market]||r.market):r.market,
      side:['over','under'].includes(r.side.toLowerCase())?r.side.toLowerCase():r.side,
      line:r.line,book:r.book,odds:price,stake_dollars:stake,model_prob:probability,
      edge_bps:probability===null?null:(probability-1/decimal(price))*10000,
      ...blendTicket(r,price)};
  }

  // What the pick was decided on, at the price actually entered.
  function blendTicket(r,price) {
    const b=r.blend,decided=r.card_snapshot_at||r.forecast_at||r.quoted_at||null;
    if(!b||!finite(b.final)||!probability(b.market))return {};
    const push=comparison(r).push??0;
    return {market_prob:b.market,final_prob:b.final,blend_weight:b.weight,
      expected_value:(1-push)*(b.final*decimal(price)-1),decision_at:decided};
  }

  function nhlReview(r) {
    if(r.human_decision==='select')return 'Analyst selected for shadow tracking';
    if(r.human_decision==='watch')return 'Analyst: watch / wait';
    const q=r.qualitative_review;
    if(!q||q.offer_id!==r.offer_id||q.forecast_id!==r.forecast_id)return 'Context review needed';
    return reviewLabel(q);
  }

  // Diagnostics are bound to the same forecast and exact offer, never a player's
  // first row or another threshold's raw probability.
  function forecastHealth(r, context) {
    if(r.sport!=='NFL')return {eligible:true,tier:1,reasons:[]};
    const group=context?.generated_at===r.forecast_at?context.groups?.[JSON.stringify([String(r.game_id),r.player,r.market_std])]:null;
    const offerEntry=Object.entries(group?.offers||{}).find(([key])=>{try{const [book,side,line,price,at]=JSON.parse(key);return book===r.book&&side===r.side&&line===r.line&&price===r.price&&at===r.quoted_at;}catch{return false;}});
    const raw=offerEntry?.[1]?.raw_probability;
    const range=group?.calibration?.fitted_raw_range,reasons=[];
    if(finite(raw)&&Array.isArray(range)&&range.length===2&&(raw<range[0]||raw>range[1]))reasons.push('Calibration input outside fitted range');
    const projection=group?.projection;
    if(projection?.current_sample?.some(g=>finite(g.attempts)&&g.attempts<10)&&projection?.career_baselines?.attempts?.mean>=20)
      reasons.push('Possible partial appearance in passing workload history');
    if(!finite(raw)&&r.model_prob>=.9)reasons.push('Extreme estimate lacks matching calibration diagnostics');
    return {eligible:!reasons.length,tier:finite(raw)?1:2,reasons,raw_probability:finite(raw)?raw:null};
  }
  const outcomeKey=r=>JSON.stringify([r.sport,String(r.game_id),r.player||'',r.market_std||r.market,String(r.side).toLowerCase(),r.line,r.settlement_profile||'']);
  // Discovery joins a lead to every quote with its game, market, player and side,
  // whatever the line. A lead is one bet: its market's main line, the line the most
  // books post (ties: price nearest even), and only where at least two books post it,
  // the blend's two-book rule. Lone alternate lines are dropped, and so are quotes with
  // unverified settlement rules when a verified book posts that line. Leads on both
  // sides of one market cancel: neither enters the pool or the review queue.
  const leadKey=r=>JSON.stringify([r.sport,String(r.game_id),r.player||'',r.market_std||r.market]);
  function researchLeads(rows) {
    const books={},sides={},main={};let opposed=0,unposted=0;
    const lineKey=r=>JSON.stringify([leadKey(r),r.line]);
    for(const r of rows){(books[lineKey(r)]??=new Set()).add(r.book);(sides[leadKey(r)]??=new Set()).add(String(r.side).toLowerCase());}
    // NFL's book_count includes the offered book; MLB/NHL count only the other books.
    const posted=r=>{const c=comparison(r).books;return Math.max(books[lineKey(r)].size,finite(c)?c+(r.sport==='NFL'?0:1):0);};
    const even=r=>Math.abs(1/decimal(r.price)-.5);
    for(const r of rows) {
      const k=leadKey(r),n=posted(r);
      if(sides[k].size>1||n<BLEND.minBooks)continue;
      const m=main[k];
      if(!m||n>m.n||n===m.n&&(even(r)<m.even||even(r)===m.even&&r.line<m.line))main[k]={n,even:even(r),line:r.line};
    }
    for(const k of Object.keys(sides))if(sides[k].size>1)opposed++;else if(!main[k])unposted++;
    const atMain=rows.filter(r=>main[leadKey(r)]?.line===r.line);
    const verified=new Set(atMain.filter(r=>r.settlement_verified===true).map(leadKey));
    return {kept:atMain.filter(r=>r.settlement_verified!==false||!verified.has(leadKey(r))),opposed,unposted};
  }
  const SHORTLIST_LIMIT=10;
  // Variety without forcing picks: each sport's best eligible bet is taken first,
  // then the rest in order, at most 4 per sport, 2 per sport and market, 2 per game
  // and 1 per player in a game.
  const CARD_CAPS={sport:4,market:2,game:2};
  function cardValue(r) {
    if(r.model_withheld||r.discovery_origin==='independent_research')return null;
    // NHL already supplies worst-scenario log growth at this fixed fraction.
    // Use the same units for NFL/MLB; sport-specific ranks are not comparable.
    const push=comparison(r).push??(r.sport==='NFL'?r.push_prob:r.model_push_probability);
    let win;
    if(probability(r.blend?.final)&&probability(push)&&push<1)win=r.blend.final*(1-push);   // blended, unconditional
    else {
      if(r.sport==='NHL')return finite(r.rank_score)?r.rank_score:null;
      win=r.sport==='NFL'?r.model_prob:r.model_probability;
      if(!probability(win)||!probability(push)||push>=1)return null;
      if(r.sport==='NFL') {
        if(probability(r.forecast_health?.raw_probability))win=Math.min(win,r.forecast_health.raw_probability);
        win*=1-push;
      }
    }
    const loss=1-win-push;
    if(loss<0||!finite(r.price)||Math.abs(r.price)<100)return null;
    return win*Math.log1p(.0025*(decimal(r.price)-1))+loss*Math.log1p(-.0025);
  }
  const cardTier=r=>cardValue(r)===null?4:r.forecast_health?.tier||1;
  const confidenceRank=r=>confidenceOf(r.blend)==='moderate'?1:0;
  // No side: the card never carries both sides of one market.
  function ideaKey(r) {
    return JSON.stringify([r.sport,String(r.game_id),r.player||'',r.market_std||r.market,r.settlement_profile||'']);
  }
  function shortlist(selected,now=Date.now()) {
    const seen=new Set(),taken=new Set(),perSport={},perMarket={},perGame={};
    const ordered=[...selected].sort((a,b)=>Number(b.human_decision==='select')-Number(a.human_decision==='select')||
      cardTier(a)-cardTier(b)||confidenceRank(a)-confidenceRank(b)||(cardValue(b)??-Infinity)-(cardValue(a)??-Infinity)||
      stamp(a.commence_time)-stamp(b.commence_time)||outcomeKey(a).localeCompare(outcomeKey(b)));
    const eligible=ordered.filter(r=>{
      const a=hasReview(r)&&r.review_matches_current!==false&&recent(r.qualitative_review.reviewed_at,now,3*HOUR)?assessment(r.qualitative_review):null;
      return r.human_decision!=='pass'&&(r.human_decision==='select'||a?.verdict==='consider'&&!a.blocking_checks.length);
    });
    const marketOf=r=>r.sport+'|'+(r.market_std||r.market),gameOf=r=>r.sport+'|'+r.game_id;
    function take(r) {
      if(taken.size>=SHORTLIST_LIMIT||taken.has(r))return false;
      if((perSport[r.sport]||0)>=CARD_CAPS.sport||(perMarket[marketOf(r)]||0)>=CARD_CAPS.market||
        (perGame[gameOf(r)]||0)>=CARD_CAPS.game)return false;
      const ideas=[ideaKey(r)];
      // One player's props in one game share the same minutes and role, and some are
      // the same event (an assist is a point; one hit is one total base at 0.5), so the
      // card takes the best one. Display consolidation never overwrites a forecast.
      if(r.player)ideas.push(JSON.stringify([r.sport,String(r.game_id),r.player]));
      if(ideas.some(v=>seen.has(v)))return false;
      ideas.forEach(v=>seen.add(v));taken.add(r);
      perSport[r.sport]=(perSport[r.sport]||0)+1;perMarket[marketOf(r)]=(perMarket[marketOf(r)]||0)+1;
      perGame[gameOf(r)]=(perGame[gameOf(r)]||0)+1;
      return true;
    }
    for(const sport of [...new Set(eligible.map(r=>r.sport))])eligible.filter(r=>r.sport===sport).some(take);
    for(const r of eligible)take(r);
    const card=eligible.filter(r=>taken.has(r));
    return card.map(r=>({...r,card_rank_score:cardValue(r),card_related_candidates:card.filter(q=>q.sport===r.sport&&q.game_id===r.game_id).length-1}));
  }
  function rankTier(r) {
    const a=hasReview(r)&&r.review_matches_current!==false?assessment(r.qualitative_review):null;
    return a?.verdict==='pass'?9:a?.verdict==='consider'?0:a?.verdict==='wait'?3:r.discovery_origin==='independent_research'?2:1;
  }

  // An edition is a historical assessment. Later quotes and kickoffs must not
  // silently rewrite it or erase its original analysis.
  function editionRows(edition,now=Date.now()) {
    const at=stamp(edition?.published_at);
    if(edition?.schema_version!==1||!['morning','afternoon','test'].includes(edition.kind)||!Number.isFinite(at)||at>now||
      edition.decision_date!==day(at)||!Array.isArray(edition.rows)||edition.rows.length>SHORTLIST_LIMIT)return [];
    return edition.rows.filter(r=>r&&priced(r)&&Number.isFinite(stamp(r.commence_time))&&day(stamp(r.commence_time))===edition.decision_date&&
      stamp(r.quoted_at)<=at&&stamp(r.commence_time)>at&&hasReview(r)&&
      recent(r.qualitative_review.reviewed_at,at,3*HOUR)&&r.human_decision!=='pass'&&r.review_matches_current!==false&&
      (r.human_decision==='select'||assessment(r.qualitative_review)?.verdict==='consider'&&!assessment(r.qualitative_review).blocking_checks.length))
      .map(r=>({...r,card_snapshot_at:edition.published_at,card_edition:edition.kind}));
  }
  // Today's Picks and the home page show only the current Eastern day's edition.
  // Before it publishes, the previous day's card stays in its dated archive rather
  // than standing in for today's.
  function currentEditionRows(edition,now=Date.now()) {
    return edition?.decision_date===day(now)?editionRows(edition,now):[];
  }

  // Each sport's published rows in the selector's own shape, at every book and line.
  function feedRows(sport,feeds) {
    if(sport==='NFL')return rows(feeds.NFL).map(r=>({...r,sport,book:r.bookmaker,side:r.name,line:r.point,quoted_at:r.last_update,
      game_id:r.game_id||r.game+'|'+r.commence_time}));
    if(sport==='MLB')return rows(feeds.MLB).map(r=>({...r,sport,game_id:r.mlb_game_id||r.event_id}));
    return rows(feeds.NHL).map(r=>({...r,sport,game_id:r.nhl_game_id}));
  }
  // A published pick at later prices: the same outcome and line at the best current book,
  // under the same market-blend rule that put it on the card. It never changes the
  // edition; it says whether the pick still qualifies at the price available now. Null
  // when no prices for its game have arrived since it was quoted, or the game started.
  function priceCheck(pick,feeds,now=Date.now()) {
    if(!pick||pick.discovery_origin==='independent_research'||!(stamp(pick.commence_time)>now))return null;
    const later=feedRows(pick.sport,feeds).filter(r=>String(r.game_id)===String(pick.game_id)&&priced(r)&&
      stamp(r.quoted_at)>stamp(pick.quoted_at)&&stamp(r.quoted_at)<=now&&(r.line===null||finite(r.line)));
    if(!later.length)return null;
    const checked_at=new Date(Math.max(...later.map(r=>stamp(r.quoted_at)))).toISOString();
    const same=later.filter(r=>outcomeKey(r)===outcomeKey(pick)).sort((a,b)=>decimal(b.price)-decimal(a.price));
    if(!same.length) {
      const side=String(pick.side).toLowerCase();
      const lines=[...new Set(later.filter(r=>leadKey(r)===leadKey(pick)&&String(r.side).toLowerCase()===side).map(r=>r.line))].sort((a,b)=>a-b);
      return {status:lines.length?'line_moved':'not_offered',checked_at,lines};
    }
    const best=same[0],raw=pick.sport==='NFL'?forecastHealth(best,feeds.NFLContext).raw_probability:null;
    const b=blend(best,pick.sport==='NFL'&&probability(raw)?Math.min(best.model_prob,raw):undefined);
    const push=comparison(best).push??0;
    // The worst price that still clears the floor at the current blend, rounded toward the bettor.
    const need=probability(b.final)&&b.final>0&&push<1?(BLEND.minEV/(1-push)+1)/b.final:null;
    const min_price=need===null?null:need>=2?Math.max(100,Math.ceil(+((need-1)*100).toFixed(6))):-Math.floor(+(100/(need-1)).toFixed(6));
    return {status:b.status,checked_at,book:best.book,book_label:best.book_label||best.book,price:best.price,quoted_at:best.quoted_at,line:best.line,
      model:b.model??null,market:b.market??null,final:b.final??null,ev:b.ev??null,min_price};
  }
  function priceCheckHTML(c) {
    if(!c)return '';
    const at=esc(time(c.checked_at)),offer=finite(c.price)?`${esc(odds(c.price))} at ${esc(c.book_label)}`:'',lines=c.lines||[];
    const text={
      qualifies:`Still a pick at ${offer}: ${signedPct(c.ev)} expected value after blending with the market.`,
      below_threshold:`No longer a pick at current prices: ${offer} gives ${signedPct(c.ev)} after blending; ${barPct(BLEND.minEV)} is required${finite(c.min_price)?`, so it needs ${esc(odds(c.min_price))} or better`:''}.`,
      model_below_market:`No longer a pick: the market now gives this side ${pct(c.market)}, above the model’s ${pct(c.model)}. Best price ${offer}.`,
      thin_market:`Not rechecked: fewer than two books now post this line. Best price ${offer}.`,
      no_model:`Not rechecked: no current model estimate. Best price ${offer}.`,
      line_moved:`Line moved: this line is no longer offered; current line${lines.length===1?'':'s'} ${esc(lines.join(', '))}.`,
      not_offered:'No longer offered at the books we follow.'}[c.status]||'';
    return text?`<span class="meta estimate-detail price-check price-check-${c.status==='qualifies'?'ok':'changed'}"><strong>Price check ${at}:</strong> ${text}</span>`:'';
  }

  // Pinnacle's no-vig chance at the same line, from the morning snapshot (/briefing/pinnacle.json).
  // A sharp reference beside morning picks only: it never selects, ranks or reprices a pick, and a
  // line counts as not posted only when Pinnacle was actually asked for that game.
  const pinnacleKey=(sport,event,market,player,side,line)=>JSON.stringify([sport,String(event??''),market,
    String(player??'').toLowerCase(),String(side??'').toLowerCase(),finite(line)?Number(line):null]);
  function pinnacleIndex(feed,decisionDate) {
    if(feed?.schema_version!==1||feed.status!=='recorded'||!decisionDate||feed.decision_date!==decisionDate||!Number.isFinite(stamp(feed.generated_at)))return null;
    const index=new Map();
    for(const p of records(feed.lines))if(probability(p.probability))
      for(const m of [p.market,p.market_std].filter(Boolean))index.set(pinnacleKey(p.sport,p.event_id,m,p.player,p.side,p.line),p);
    return {index,checked_at:feed.generated_at,games:new Set(feed.checked_games||[]),events:new Set(feed.checked_events||[])};
  }
  function pinnacleFor(r,ref) {
    if(!ref)return null;
    const event=r.event_id||r.game_id,markets=[r.market,r.market_std].filter(Boolean);
    for(const m of markets){const hit=ref.index.get(pinnacleKey(r.sport,event,m,r.player,r.side,r.line));if(hit)return {status:'posted',probability:hit.probability,checked_at:ref.checked_at};}
    const asked=markets.some(m=>GAME_MARKETS.includes(m))?ref.games.has(r.sport):ref.events.has(String(event??''));
    return asked?{status:'not_posted',checked_at:ref.checked_at}:null;
  }
  function pinnacleHTML(p) {
    if(!p)return '';
    const at=esc(new Date(p.checked_at).toLocaleString('en-US',{timeZone:'America/New_York',hour:'numeric',minute:'2-digit'})+' ET');
    return p.status==='posted'
      ?`<span class="meta estimate-detail pinnacle-ref"><strong>Pinnacle ${pct(p.probability)}</strong> no-vig at ${at} · sharp reference, not used to choose picks</span>`
      :`<span class="meta estimate-detail pinnacle-ref">Pinnacle had not posted this line at ${at}.</span>`;
  }

  function collect(feeds,now=Date.now()) {
    const selected=[],coverage=[],excluded=[],leans=[];
    for(const sport of ['NFL','MLB','NHL']) {
      const data=feeds[sport],board=feeds.NHLBoard;
      let available=false,candidates=[];
      if(sport==='NFL') {
        available=data?.schema_version===1&&data.status==='ready'&&recent(data.generated_at,now,48*HOUR);
        if(available)candidates=rows(data).filter(r=>today(r,now)&&recent(r.last_update,now,90*MINUTE)&&
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
      // Independent discovery must resolve to current quotes in a separate feed.
      // It can add research candidates, but never invent a model probability.
      const discovered=feeds.Discovery;
      if(discovered?.schema_version===1&&discovered.decision_date===day(now)&&recent(discovered.generated_at,now,12*HOUR)) {
        const expiry=(sport==='NHL'?30:90)*MINUTE;
        candidates.push(...records(discovered.candidates).filter(r=>r.sport===sport&&today(r,now)&&recent(r.quoted_at,now,expiry)));
      }
      candidates=candidates.filter(r=>priced(r)&&r.game_id&&(r.line===null||finite(r.line))).map(r=>({...r,sport}));
      const leads=researchLeads(candidates.filter(r=>r.discovery_origin==='independent_research'));
      candidates=[...candidates.filter(r=>r.discovery_origin!=='independent_research'),...leads.kept];
      const discoveryByOutcome=new Map(candidates.filter(r=>r.discovery_origin==='independent_research').map(r=>[outcomeKey(r),r.discovery]));
      for(const r of candidates)if(!r.discovery_origin&&discoveryByOutcome.has(outcomeKey(r))){r.discovery_origin='model_and_independent_research';r.discovery=discoveryByOutcome.get(outcomeKey(r));}
      const healthy=[],belowBlend=[];
      for(const r of candidates) {
        r.forecast_health=forecastHealth(r,feeds.NFLContext);
        if(!r.forecast_health.eligible&&r.discovery_origin!=='independent_research') {
          excluded.push({...r,exclusion_reasons:r.forecast_health.reasons});continue;
        }
        // Every model candidate must clear the market blend. Independent research
        // has no model probability to blend and keeps its separate review route.
        if(r.discovery_origin!=='independent_research') {
          if(sport==='NFL'&&(!probability(r.push_prob)||r.push_prob>=1)){excluded.push({...r,exclusion_reasons:['Push probability unavailable or invalid']});continue;}
          // A calibrated 50% is the curve's flat centre: no information at this line.
          if(sport==='NFL'&&Math.abs(r.model_prob-.5)<1e-9){excluded.push({...r,exclusion_reasons:['Calibrated estimate is exactly 50%, so the model has no information at this line']});continue;}
          // NFL keeps its sensitivity heuristic: the lower of raw and calibrated.
          const raw=r.forecast_health.raw_probability;
          const b=r.blend=blend(r,sport==='NFL'&&probability(raw)?Math.min(r.model_prob,raw):undefined);
          if(b.status==='thin_market'){excluded.push({...r,exclusion_reasons:['Fewer than two books, counting the offered one, post this exact line, so there is no market price to blend with']});continue;}
          if(b.status==='model_below_market'){excluded.push({...r,exclusion_reasons:[`The model gives this side ${pct(b.model)}, below the market’s ${pct(b.market)}; a better price alone belongs in Market Watch`]});continue;}
          if(b.status!=='qualifies'){
            if(b.status==='below_threshold'&&b.ev>0)belowBlend.push(r);
            excluded.push({...r,blend_shortfall:true,exclusion_reasons:[b.status==='below_threshold'?`${signedPct(b.ev)} expected value after blending with the market; ${barPct(BLEND.minEV)} required`:'No model probability']});continue;
          }
          if(b.ev>BLEND.maxEV){r.forecast_health.tier=3;r.forecast_health.reasons.push('Unusually large return estimate requires research');r.review='Unusually large estimate · research needed';}
          if(sport==='NFL')r.screening_ev=b.ev;
          r.score=100*b.ev;
        }
        healthy.push(attachReview(sport==='NHL'?{...r,review_sources:records(board?.sources)}:r,feeds.Reviews,now));
      }
      // Best available book for the identical outcome/line. Different bets in
      // one game remain eligible; exposure is descriptive, never a picks quota.
      healthy.sort((a,b)=>decimal(b.price)-decimal(a.price)||stamp(b.quoted_at)-stamp(a.quoted_at)||String(a.book).localeCompare(String(b.book)));
      const seen=new Set(),offers=[];
      for(const r of healthy) {
        const identity=outcomeKey(r);if(seen.has(identity))continue;seen.add(identity);offers.push(r);
      }
      offers.sort((a,b)=>rankTier(a)-rankTier(b)||(a.forecast_health.tier-b.forecast_health.tier)||
        (b.score||0)-(a.score||0)||stamp(a.commence_time)-stamp(b.commence_time)||outcomeKey(a).localeCompare(outcomeKey(b)));
      // One offer per market: never both sides (a research lead against a model pick, a
      // model middle) nor several lines or settlement terms of one bet. The highest-ranked
      // offer stays.
      const markets=new Set(),unique=offers.filter(r=>!markets.has(leadKey(r))&&markets.add(leadKey(r)));
      const repeated=offers.length-unique.length;
      // Leans: the sport's best play that is positive after the blend but under the
      // 1% floor, for a market not already listed at any line, side or book. Shown for
      // interest, never bet, tracked or put on the card.
      const lean=belowBlend.filter(r=>!markets.has(leadKey(r)))
        .sort((a,b)=>b.blend.ev-a.blend.ev||decimal(b.price)-decimal(a.price))[0];
      if(lean)leans.push({...lean,sport,lean:true});
      for(const r of unique) {
        r.exposure_group=sport+':'+r.game_id;
        r.related_candidates=unique.filter(q=>q.game_id===r.game_id).length-1;
        selected.push(r);
      }
      const count=unique.length,mine=excluded.filter(r=>r.sport===sport);
      const high=unique.filter(r=>confidenceOf(r.blend)==='high').length,moderate=unique.filter(r=>confidenceOf(r.blend)==='moderate').length;
      const short=mine.filter(r=>r.blend_shortfall).length,held=mine.length-short;
      coverage.push({sport,count,available,held,observed_markets:[...new Set(rows(data).map(r=>r.market_std||r.market))],
        message:(!available?'Current model list unavailable or expired. ':'')+(count?`${count} review candidate${count===1?'':'s'}`:'No qualifying offers currently available')+
          (moderate?` (${high} high, ${moderate} moderate confidence)`:'')+
          (short?` · ${short} below the ${barPct(BLEND.minEV)} market-blend floor`:'')+
          (held?` · ${held} withheld (thin market, model below market, 50% calibration or unreliable forecast)`:'')+
          (leads.opposed?` · ${leads.opposed} research market${leads.opposed===1?'':'s'} withheld (leads on both sides)`:'')+
          (leads.unposted?` · ${leads.unposted} research lead${leads.unposted===1?'':'s'} withheld (no line two books post)`:'')+
          (repeated?` · ${repeated} withheld as another side or line of a listed market`:'')+(sport==='NHL'?' · unposted markets can enter on the next refresh':'')});
    }
    // A reviewed MLB/NHL opportunity must not disappear below the initial
    // twenty unreviewed NFL rows. Compare assessment status across sports;
    // retain each sport's existing numerical order within that status.
    selected.sort((a,b)=>rankTier(a)-rankTier(b));
    return {selected,coverage,excluded,leans};
  }

  function diagnosticHTML(r) {
    const d=r.reviewed_candidate?.model_diagnostics||r.model_diagnostics;
    if(!d)return '';
    const p=d.projection,c=d.calibration,s=d.raw_distribution_stress;
    const inputs=p?`<p><strong>Passing inputs:</strong> ${esc(number(p.attempts))} attempts × ${pct(p.completion_rate)} completion rate × ${esc(number(p.yards_per_completion))} yards per completion. Their product gives expected passing yards before matchup and venue adjustments.</p><p><strong>Recent sample:</strong> ${(p.current_sample||[]).map(g=>`${esc(g.season)} week ${esc(g.week)}: ${esc(g.attempts)} attempts, ${esc(g.completions)} completions, ${esc(g.passing_yards)} yards`).join('; ')||'No current-season sample'}. ${pct(p.recent_mean_weight)} recent weight for attempts and ${pct(p.yards_per_completion_recent_weight)} for yards per completion; partial appearances are not separately adjusted.</p>`:'';
    const calibration=c&&finite(c.raw_probability)?`<p><strong>Probability calculation:</strong> ${pct(c.raw_probability)} before historical calibration. ${c.outside_fitted_range?'This input is outside the calibration sample’s fitted range and receives an endpoint value.':'Historical calibration maps this estimate to the displayed probability.'} The calibration artifact does not report the number of observations in this tail.</p>`:'';
    const ladder=d.offered_book_nearby_quotes?.length?`<p><strong>Same-book line choices:</strong> ${d.offered_book_nearby_quotes.map(q=>`${esc(q.name)} ${esc(q.point)} at ${esc(odds(q.price))}`).join('; ')}. ${d.offered_book_central_quote?`Its most evenly priced line is ${esc(d.offered_book_central_quote.point)} at ${esc(odds(d.offered_book_central_quote.price))}.`:''}</p>`:'';
    const sensitivity=s?`<p><strong>What would change the price case:</strong> In a hypothetical Normal distribution using the existing spread of outcomes, moving the mean to ${esc(number(s.market_centered_mean))} gives ${pct(s.probability)} for this side. ${finite(s.mean_at_break_even)?'In this hypothetical distribution, break-even occurs at a mean of '+esc(number(s.mean_at_break_even))+'. ':''}This is a sensitivity check, not a new forecast; a market median line is not necessarily a mean.</p>`:'';
    return `<details class="pick-diagnostics"><summary>See the model inputs and line comparison</summary>${inputs}${calibration}${ladder}<p>${esc(d.other_books_at_exact_line)} other paired books at this exact line. Different thresholds are not interchangeable prices.</p>${sensitivity}</details>`;
  }

  function researchHTML(r) {
    const q=r.qualitative_review;
    if(!hasReview(r))return '';
    const items=citedEvidence(r).map(({e,s})=>{
      return `<li><strong>${esc(e.direction)}:</strong> ${esc(e.interpretation)} <a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a> <span class="meta">${esc(global.FVInjuryContext?.sourceTime(s)||`Published ${time(s.published_at)}`)} · may already be reflected in ${esc(e.represented_in.replaceAll('_',' '))}.</span></li>`;
    }).join('');
    const prev=r.reviewed_candidate;
    const original=prev?`<p class="meta">Reviewed ${esc(odds(prev.price))} at ${esc(time(prev.quoted_at))}. ${r.review_matches_current?'Matches this offer and forecast.':'Current price or forecast differs; this is earlier context, not a review of the current offer.'}</p>`:'';
    const correction=oldNFLReview(r)?'<p class="notice">Method correction: this earlier note was given an incorrect description of the NFL model. Its probabilities are calibrated to historical results, not current market prices. Any claim below that it is “market-calibrated” is incorrect. The forecast itself is unchanged.</p>':'';
    const a=assessment(q);
    const judgment=a?`<p><strong>Our assessment: ${esc(verdictLabel(a))}.</strong> ${esc(a.reason)}</p><p><strong>Model case:</strong> ${esc(a.model_case)}</p><p><strong>Price case:</strong> ${esc(a.price_case)}</p><p><strong>Relevant context:</strong> ${esc(a.context_case)}</p>${a.blocking_checks.length?'<p><strong>What needs checking:</strong></p><ul>'+a.blocking_checks.map(s=>`<li>${esc(s)}</li>`).join('')+'</ul>':''}`:'<p class="meta">This earlier review checked reporting only. A full model-and-price assessment is pending.</p>';
    return `<details class="pick-research"><summary>Fourth &amp; Value analysis · ${esc(time(q.reviewed_at))}</summary><p><strong>Why it surfaced:</strong> ${esc(screenReason(r))}</p>${original}${correction}${judgment}${global.FVInjuryContext?.render(r,r.review_sources)||''}${diagnosticHTML(r)}${items?`<ul>${items}</ul>`:'<p class="meta">No relevant reporting was verified for this review. That does not, by itself, invalidate the model-and-price case.</p>'}<p><strong>Case against:</strong> ${esc(q.countercase)}</p><p><strong>Final checks:</strong></p><ul>${q.open_checks.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><p class="meta">Review the analysis and confirm the current line, price and conditions before deciding. Original model probabilities remain unchanged.</p></details>`;
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
      const a=assessment(q);
      const status=a?`<strong>Our assessment: ${esc(verdictLabel(a))}.</strong> ${esc(a.reason)}`:{research_support:'Our review found relevant supporting reporting; analyst verification is still needed.',concern:'Our review found a concern to resolve.',needs_information:'The earlier review checked reporting only; a full betting assessment is pending.'}[q.status];
      const evidence=citedEvidence(r).find(({e})=>['supports','concern'].includes(e.direction));
      const source=evidence?` ${esc(evidence.e.interpretation)} <a href="${esc(evidence.s.url)}" target="_blank" rel="noopener noreferrer">Source</a>.`:'';
      const changed=r.review_matches_current===false?' <strong>The price or forecast has changed since this review; reassess the current offer.</strong>':'';
      const original=changed?` This review assessed ${esc(odds(prior.price))} quoted ${esc(time(prior.quoted_at))}.`:'';
      const next=a?a.blocking_checks[0]:q.open_checks[0];
      const check=next?` <strong>Still to check:</strong> ${esc(next)}`:'';
      // Keep long countercases, source timestamps and offer metadata in the full
      // review/table. No new AI call, shortened quotation or invented narrative.
      const countercase=!a&&q.status==='concern'&&!oldNFLReview(r)?` ${esc(q.countercase)}`:'';
      return `<p class="pick-summary-paragraph"><strong class="summary-bet">${esc(betLabel(r).replaceAll(' · ',' '))}</strong><span class="meta summary-game">${esc(r.sport)} · ${esc(r.game)}</span>${esc(screenReason(r))} ${status}${changed}${original}${source}${countercase}${check} <a class="read-pick-review" href="#pick-review-${selected.indexOf(r)}">Read our full analysis</a>.</p>`;
    }).join('');
    const currentReviews=reviewed.filter(r=>r.review_matches_current!==false).length;
    const earlier=reviewed.length-currentReviews;
    return paragraphs+`<p class="meta">${currentReviews} of ${selected.length} ${selected.some(r=>r.card_snapshot_at)?'published':'current'} candidates reviewed${earlier?`; ${earlier} with earlier analysis requiring a recheck`:''}; ${featured.length} summarized here. Full findings and remaining checks appear under each bet. Review every bet and confirm the current line and price before deciding.</p>`;
  }

  function rowHTML(r,index=0,saved=false) {
    const context=global.FVPlayerContext?.render(r,r.sport,{saved:!!r.card_snapshot_at})||'';
    const research=context+researchHTML(r);
    const a=hasReview(r)&&r.review_matches_current!==false?assessment(r.qualitative_review):null;
    const reason=a?.verdict==='wait'&&r.human_decision!=='select'?`<span class="estimate-detail"><strong>Why:</strong> ${esc(a.reason)}</span>`:'';
    const line=r.line===null?`${r.side} · ${r.market_label}`:`${r.side} ${r.market==='spreads'&&r.line>0?'+':''}${r.line}`;
    const offer=`<strong class="book-offer-line">${esc(line)}</strong><span class="estimate-detail">${esc(odds(r.price))}</span>`;
    const related=r.card_related_candidates??r.related_candidates;
    const exposure=related?'<br>Shared game: '+related+' other '+(r.card_related_candidates!==undefined?'shortlisted bet(s)':'research candidate(s)'):'';
    const dated=r.card_snapshot_at?`<br><strong>${stamp(r.commence_time)<=Date.now()?'Game started · historical assessment':'Published assessment · confirm current conditions'}</strong><br>Analysis and prices preserved from ${esc(time(r.card_snapshot_at))}`:'';
    return `<tr class="pick-offer-row"><td class="pick-bet"><a href="${esc(r.url)}"><strong>${esc(betLabel(r))}</strong></a><br><span class="meta">${esc(r.sport)} · ${esc(r.game)}<br>Starts ${esc(time(r.commence_time))}<br>Experimental · ${confidenceOf(r.blend)?esc(CONFIDENCE[confidenceOf(r.blend)])+' · ':''}${esc(displayReview(r))}${reason}${exposure}${dated}${r.discovery_origin?'<br>Origin: '+(r.discovery_origin==='independent_research'?'independent research':'model and independent research'):''}</span></td><td class="pick-estimate pick-model" data-label="Model prediction">${modelHTML(r)}</td><td class="pick-estimate pick-market" data-label="Market consensus">${marketHTML(r)}</td><td class="pick-offer" data-label="Book line / price">${offer}<span class="meta estimate-detail">${pct(comparison(r).breakEven)} break-even*</span>${blendHTML(r)}${priceCheckHTML(r.price_check)}</td><td class="pick-quote-time" data-label="Price time (ET)"><time datetime="${esc(r.quoted_at)}">${esc(time(r.quoted_at))}</time></td><td class="pick-book" data-label="Book">${esc(r.book_label||r.book)}<br><button type="button" class="track-pick secondary" data-track-pick="${index}" ${saved?'disabled':''} aria-label="${esc((saved?'Tracked: ':'Track bet: ')+betLabel(r))}">${saved?'Tracked':'Track bet'}</button></td></tr>${research?`<tr class="pick-research-row" id="pick-review-${index}"><td colspan="6">${research}</td></tr>`:''}`;
  }

  async function mount() {
    const root=document.getElementById('daily-picks');if(!root)return;
    const urls={NFL:'/props/top-picks.json',MLB:'/mlb/data/latest.json',NHL:'/nhl/data/latest.json',NHLBoard:'/nhl/data/candidates.json',NFLContext:'/props/model-context.json',Discovery:'/briefing/discovery.json',Reviews:'/briefing/reviews.json',Card:'/briefing/morning-card.json',Afternoon:'/briefing/afternoon-card.json',Pinnacle:'/briefing/pinnacle.json'};
    let feeds={},checked=null,loading=false,current=[],visibleLimit=20,draft=null,trackingReady=null,saving=false;
    const tickets=new Map(),dialog=document.getElementById('pick-tracker'),form=document.getElementById('track-bet-form');
    const $=id=>document.getElementById(id);
    function render() {
      const now=Date.now(),result=collect(feeds,now);
      const edition=feeds.Card,today=edition?.decision_date===day(now),afternoon=feeds.Afternoon,laterToday=afternoon?.decision_date===day(now);
      // The afternoon edition's price check, recorded once after the 4:30 p.m. refresh, stays
      // with each morning pick; the morning edition itself is never rewritten.
      const checks=new Map(laterToday&&afternoon.morning_edition_id===edition?.edition_id?records(afternoon.price_check).map(c=>[c.key,c]):[]);
      // Pinnacle's morning reference attaches to morning picks only; afternoon picks use later prices.
      const pin=pinnacleIndex(feeds.Pinnacle,today?edition.decision_date:null);
      const card=currentEditionRows(edition,now).map(r=>({...r,...(checks.has(key(r))?{price_check:checks.get(key(r))}:{}),...(pin?{pinnacle:pinnacleFor(r,pin)}:{})})),later=currentEditionRows(afternoon,now);
      const published=[...card,...later],cardKeys=new Set(published.map(key)),
        cardSides=new Map(published.map(r=>[leadKey(r),String(r.side).toLowerCase()])),
        pool=result.selected.filter(r=>!cardKeys.has(key(r))&&(cardSides.get(leadKey(r))??String(r.side).toLowerCase())===String(r.side).toLowerCase());
      current=[...card,...later,...pool];
      const openReviews=new Set(Array.from(root.querySelectorAll('.pick-research[open]')).map(el=>el.closest('tr').dataset.reviewKey));
      const incomplete=today&&edition.status==='research_incomplete';
      const emptyMessage=edition&&!today?'Today’s card is not published yet. The morning edition publishes after research finishes; the previous day’s card is not carried over.':incomplete?'Morning research did not finish. This is not a completed no-pick day. See the research status below.':edition?.status==='no_reviewed_candidates'?'Research completed; no reviewed offers qualified for this edition.':'No published morning picks are available. The research pool is separate from the morning edition.';
      document.getElementById('daily-picks-rows').innerHTML=card.map((r,i)=>rowHTML(r,i,tickets.get(key(r))?.saved)).join('')||`<tr><td colspan="6">${emptyMessage}</td></tr>`;
      const laterRows=$('afternoon-picks-rows'),laterStatus=$('afternoon-status');
      if(laterRows)laterRows.innerHTML=later.map((r,i)=>rowHTML(r,i+card.length,tickets.get(key(r))?.saved)).join('')||
        `<tr><td colspan="6">${!laterToday?'Afternoon picks publish after the 4:30 p.m. Eastern refresh.':afternoon.status==='research_incomplete'?'Afternoon research did not finish; this is not a no-pick afternoon.':
          afternoon.research?.notes?.length?'No new picks this afternoon: the day’s research allowance was used up before new candidates could be reviewed.':'No new picks qualified this afternoon.'}</td></tr>`;
      if(laterStatus)laterStatus.textContent=laterToday?`${later.length} afternoon pick${later.length===1?'':'s'} · Published ${time(afternoon.published_at)} · New picks only: markets already on the morning card are left out. Original prices and analysis; not continuously reassessed.`:
        'Afternoon picks use the 4:30 p.m. prices, after lineups and goalies are more settled. They add new picks only and never repeat a morning pick.';
      const poolRows=$('research-picks-rows');if(poolRows)poolRows.innerHTML=pool.slice(0,visibleLimit).map((r,i)=>rowHTML(r,i+card.length+later.length,tickets.get(key(r))?.saved)).join('')||'<tr><td colspan="6">No additional research candidates.</td></tr>';
      const poolLabel=$('research-pool-label');if(poolLabel)poolLabel.textContent=`Research pool · ${pool.length} additional offers · not the daily card`;
      const more=$('picks-show-more');if(more){more.hidden=pool.length<=visibleLimit;more.textContent=`Show all ${pool.length} research offers (${Math.min(visibleLimit,pool.length)} shown)`;}
      current.forEach((r,i)=>{const row=$('pick-review-'+i);if(row){row.dataset.reviewKey=key(r);row.querySelector('details').open=openReviews.has(key(r));}});
      const editionAt=stamp(edition?.published_at),validEdition=Number.isFinite(editionAt)&&editionAt<=now&&edition?.schema_version===1&&['morning','test'].includes(edition.kind)&&edition.decision_date===day(editionAt)&&Array.isArray(edition.rows)&&edition.rows.length<=SHORTLIST_LIMIT;
      const tierCounts=rows=>{const h=rows.filter(r=>confidenceOf(r.blend)==='high').length,m=rows.filter(r=>confidenceOf(r.blend)==='moderate').length;return m?` (${h} high, ${m} moderate confidence)`:'';};
      const health=incomplete?' · Research incomplete: '+(edition.research?.issues||['Some assessments could not be completed']).join('; '):edition?.status==='no_reviewed_candidates'?' · Research completed; no qualifying picks':'';
      document.getElementById('picks-status').textContent=validEdition&&today?`${card.length} reviewed picks${tierCounts(card)} · ${edition.kind==='test'?'Test':'Morning'} edition for ${edition.decision_date} · Published ${time(edition.published_at)}${health}. Original prices and analysis; not continuously reassessed.`:'0 reviewed picks · Waiting for today’s morning edition. Data refreshes are scheduled for 7:05 a.m. Eastern, with recovery starts at 7:35, 8:05 and 8:35; the card publishes after research finishes.';
      const exposure=$('picks-exposure');if(exposure){const games=new Set(card.map(r=>r.exposure_group||r.sport+':'+r.game_id));exposure.textContent=card.length>1&&games.size===1?'All shortlisted offers are from one game and share exposure. They are not independent signals.':'';}
      document.getElementById('picks-coverage').textContent=result.coverage.map(c=>`${c.sport}: ${c.message}`).join(' · ');
      const leanList=$('picks-leans'),leanSection=$('picks-leans-section');
      if(leanList&&leanSection) {
        leanList.innerHTML=result.leans.map(r=>`<li><strong>${esc(r.sport)}</strong> · ${esc(betLabel(r))} · ${esc(odds(r.price))} at ${esc(r.book_label||r.book)} · ${esc(r.game)} <span class="meta">Blended ${pct(r.blend.final)} · ${signedPct(r.blend.ev)} expected value</span></li>`).join('');
        leanSection.hidden=!result.leans.length;
      }
      // Research spending is not shown publicly; the owner gets a private report.
      const budgetStatus=document.getElementById('picks-budget-status');
      if(budgetStatus)budgetStatus.textContent='Research runs in the morning within a fixed daily allowance; the afternoon edition reviews new candidates only with what the morning left.';
      const research=document.getElementById('picks-research-status');
      if(research){const d=feeds.Discovery;const extra=d?.decision_date===day(now)?` · Independent discovery: ${d.status.replaceAll('_',' ')}; ${d.submitted_games?.length||0}/${d.slate_games||0} games submitted (not exhaustive research)`:'';research.textContent=['NFL','MLB','NHL'].map(s=>researchStatus(s,feeds.Reviews,result.selected,now)).join(' · ')+extra;}
      if(draft&&dialog.open&&!saving&&!draft.saved&&!collect(feeds,now).selected.some(r=>key(r)===key(draft.row))) {
        $('track-quote').textContent=`Saved quote: ${odds(draft.row.price)} at ${time(draft.row.quoted_at)}. This offer has expired or changed. Enter the price of the bet you actually placed.`;
      }
    }
    $('picks-show-more')?.addEventListener('click',()=>{visibleLimit=Infinity;render();});
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
    root.addEventListener('click',event=>{
      const button=event.target.closest('[data-track-pick]');if(!button)return;
      const row=current[Number(button.dataset.trackPick)];if(!row)return;
      const itemKey=key(row);
      if(![...currentEditionRows(feeds.Card),...currentEditionRows(feeds.Afternoon),...collect(feeds,Date.now()).selected].some(r=>key(r)===itemKey)){render();return;}
      draft=tickets.get(itemKey)||{id:crypto.randomUUID(),row:{...row},saved:false};tickets.set(itemKey,draft);
      form.reset();$('track-save').disabled=!!draft.saved;$('track-feedback').textContent='';$('track-signin').hidden=true;
      $('track-bet-description').textContent=`${row.sport} · ${row.game} · ${betLabel(row)} · ${row.book_label||row.book}`;
      $('track-quote').textContent=`${row.card_snapshot_at?'Historical edition quote':'Saved quote'}: ${odds(row.price)} at ${time(row.quoted_at)}. Enter the price of the bet you actually placed.`;
      $('track-review').textContent=`Review status: ${displayReview(row)}.`;
      $('track-grading').textContent='Bet Tracker settles this bet automatically from the final box score, usually a few hours after the game ends.';
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
        const next=Object.fromEntries(entries);if(!next.Card&&feeds.Card)next.Card=feeds.Card;if(!next.Afternoon&&feeds.Afternoon)next.Afternoon=feeds.Afternoon;
        feeds=next;checked=new Date().toISOString();render();
      } finally {loading=false;}
    }
    // The owner email's "I placed this" link (scripts/morning_email.py) opens Track bet for
    // that pick of the same edition, morning or afternoon, with the suggested stake filled
    // in, once. A link from another edition opens nothing; the cards below are today's.
    function openFromLink() {
      const params=new URLSearchParams(location.search),edition=params.get('edition');
      if(!edition)return;
      history.replaceState(null,'',location.pathname+location.hash);
      const pick=Number(params.get('pick')),stake=Number(params.get('stake'));
      const morning=currentEditionRows(feeds.Card),later=feeds.Afternoon?.edition_id===edition;
      if(!Number.isInteger(pick)||!(later?currentEditionRows(feeds.Afternoon):feeds.Card?.edition_id===edition?morning:[])[pick])return;
      root.querySelector(later?`#afternoon-picks-rows [data-track-pick="${morning.length+pick}"]`:`#daily-picks-rows [data-track-pick="${pick}"]`)?.click();
      if(dialog.open&&Number.isFinite(stake)&&stake>0)$('track-stake').value=stake.toFixed(2);
    }
    await load();openFromLink();setInterval(render,30000);setInterval(load,300000);
    document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});
  }
  // Home page: the first pick on today's published card whose game has not started,
  // else the card's first pick (shown as started), so the day's pick stays up all day.
  // It reads the same edition through currentEditionRows, so it can never disagree
  // with Today's Picks; it never promotes a research-pool offer.
  function featuredPick(edition,now=Date.now()) {
    const rows=currentEditionRows(edition,now);
    return rows.find(r=>stamp(r.commence_time)>now)||rows[0]||null;
  }
  const shortTime=s=>new Date(s).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})+' ET';
  function pickTitle(r) {
    const line=r.line===null||r.line===undefined?'':r.market==='spreads'&&r.line>0?'+'+r.line:String(r.line);
    const side=String(r.side||'');
    return [r.player,side.charAt(0).toUpperCase()+side.slice(1),line,String(r.market_label||'').toLowerCase()].filter(Boolean).join(' ');
  }
  function featuredHTML(edition,now=Date.now()) {
    const links='<p class="pick-links"><a href="/briefing/#daily-picks">Today’s Picks →</a><a href="/research/daily-process.html">How picks are chosen →</a></p>';
    const r=featuredPick(edition,now);
    if(!r) {
      const current=edition?.decision_date===day(now);
      const [head,body]=!current?['Today’s card is not published yet.','The morning edition publishes after the sports refresh. Until then nothing is featured.']:
        edition.status==='research_incomplete'?['Morning research did not finish.','This is not a no-pick day. Today’s Picks shows the research status.']:
        ['No pick cleared today’s review.','The model flagged ideas, but none passed both the price check and the research review. We would rather show nothing than force one.'];
      return `<p class="eyebrow">Featured pick</p><h2>${esc(head)}</h2><p class="muted">${esc(body)}</p>${links}`;
    }
    const c=comparison(r),conf=confidenceOf(r.blend),a=assessment(r.qualitative_review),started=stamp(r.commence_time)<=now;
    const tile=(label,value)=>`<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
    return `<div class="pick-head"><p class="eyebrow">Featured pick · ${esc(r.sport)}</p>${conf?`<span class="pick-chip">${esc(CONFIDENCE[conf])}</span>`:''}</div>`+
      `<h2>${esc(pickTitle(r))}</h2>`+
      `<p class="meta">${esc(r.game)} · <strong>${esc(r.book_label||r.book)} ${esc(odds(r.price))}</strong> · quoted ${esc(shortTime(r.quoted_at))} · starts ${esc(shortTime(r.commence_time))}</p>`+
      `<div class="pick-tiles">${tile('Model',c.model===null?'Not established':pct(c.model))}${tile('Market',pct(c.market))}${tile('Break-even',pct(c.breakEven))}</div>`+
      (a?`<p><strong>${esc(verdictLabel(a))}:</strong> ${esc(a.reason)}</p>`:'')+
      `<p class="muted"><strong>The case against:</strong> ${esc(r.qualitative_review.countercase)}</p>`+
      `<p class="meta">${started?'<strong>Game started · historical assessment.</strong> ':''}Original price and analysis from the ${esc(shortTime(r.card_snapshot_at))} edition.${started?'':' Confirm the current line before deciding.'}</p>${links}`;
  }
  if(typeof module==='object'&&module.exports)module.exports={collect,shortlist,editionRows,currentEditionRows,priceCheck,pinnacleIndex,pinnacleFor,pinnacleHTML,leadKey,rowKey:key,ideaKey,SHORTLIST_LIMIT,CARD_CAPS,BLEND,blend,forecastHealth,outcomeKey,rowHTML,day,ticketData,reviewKey,reviewBetKey,summaryHTML,comparison,researchStatus,featuredPick,featuredHTML,boardOrder,boardBlendHTML};
  else {global.FVPicks={featuredPick,featuredHTML,blend,boardOrder,boardBlendHTML};mount();}
})(typeof window==='undefined'?globalThis:window);

/* Track bet controls for every sport's offer boards; actual wagers use the existing per-user Bet Tracker. */
(function (global) {
  'use strict';
  // Ledger market names per league. They match the grader (docs/tracking/live-stats.js);
  // `manual` markets are saved but have no box-score stat, so they stay pending until graded by hand.
  const game={h2h:'h2h',spreads:'spreads',totals:'totals'};
  const same=names=>Object.fromEntries(names.map(name=>[name,name]));
  const leagues={
    NHL:{markets:{player_goals:'goals',player_assists:'assists',player_points:'points',player_shots_on_goal:'sog',...game,totals:'team_total'}},
    MLB:{markets:{...game,...same(['batter_hits','batter_total_bases','batter_rbis','batter_home_runs','batter_runs_scored','batter_walks',
      'batter_singles','batter_doubles','batter_stolen_bases','batter_strikeouts','batter_hits_runs_rbis','pitcher_strikeouts','pitcher_outs',
      'pitcher_hits_allowed','pitcher_earned_runs','pitcher_walks'])}},
    NBA:{markets:{...game,...same(['player_points','player_rebounds','player_assists','player_threes','player_steals','player_blocks','player_turnovers',
      'player_points_rebounds_assists','player_points_rebounds','player_points_assists','player_rebounds_assists','player_blocks_steals'])}},
    NFL:{markets:{...game,...same(['anytime_td','pass_yds','pass_tds','pass_completions','pass_attempts','interceptions','rush_yds','rush_attempts',
      'receptions','recv_yds','rush_reception_yds','pass_rush_yds','rush_tds','rec_tds','kicking_points','field_goals','tackles','sacks',
      'first_td','last_td','reception_longest','rush_longest'])},manual:['first_td','last_td','reception_longest','rush_longest']},
  };
  const day=value=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(value));
  const probability=value=>Number.isFinite(value)&&value>=0&&value<=1;
  const age=(value,now)=>now-Date.parse(value);
  const identity=r=>JSON.stringify([r.sport,r.event_id,r.commence_time,r.game,r.player,r.market,r.side,r.line,r.book,r.price,r.quoted_at,r.settlement_profile]);
  // The ledger's model_prob is conditional on a non-push. Missing or expired
  // model inputs must never become a market probability.
  function modelProbability(r,now) {
    const conditional=(win,push)=>probability(win)&&probability(push)&&push<1&&win+push<=1?win/(1-push):null;
    if(r.sport==='NHL'){
      const checked=age(r.model_data_checked_at,now);
      return !r.model_withheld&&checked>=0&&checked<36*3600e3&&probability(r.independent_probability)?conditional(r.final_probability,r.push_probability):null;
    }
    if(r.sport==='MLB'){
      // Same window the MLB boards use before showing a forecast as current.
      const checked=age(r.model_checked_at,now);
      return checked>=-300e3&&checked<=90*60e3?conditional(r.model_probability,r.model_push_probability):null;
    }
    // NFL publishes the non-push probability directly, but only a fitted calibration is an estimate.
    if(r.sport==='NFL')return String(r.model_status||'').startsWith('Calibration fitted')&&probability(r.model_prob)?r.model_prob:null;
    return null; // NBA has no published model yet.
  }
  function ticketData(row,price,stake,now=Date.now()) {
    price=Number(price);stake=Number(stake);
    if(!Number.isInteger(price)||Math.abs(price)<100)throw Error('Enter valid American odds, such as -110 or +150.');
    if(!Number.isFinite(stake)||stake<=0||Math.abs(stake*100-Math.round(stake*100))>1e-6)throw Error('Enter a positive stake in dollars and cents.');
    const league=leagues[row.sport];
    if(!league)throw Error('Open Bet Tracker to record this bet manually.');
    const teams=String(row.game||'').split(' @ '),home=row.home_team||teams[1],away=row.away_team||teams[0];
    if(!home||!away||!row.book||!Number.isFinite(Date.parse(row.commence_time)))throw Error('This offer is missing its game or sportsbook. Open Bet Tracker to enter it manually.');
    if(!Object.hasOwn(league.markets,row.market))throw Error('Open Bet Tracker to record this market manually.');
    const side=String(row.side||''),lineless=row.market==='h2h'||row.line==null&&['yes','no'].includes(side.toLowerCase());
    if(!side||(!lineless&&!Number.isFinite(row.line)))throw Error('This offer is missing its selection or line. Open Bet Tracker to enter it manually.');
    const model=modelProbability(row,now);
    return {league:row.sport,game_date:day(row.commence_time),team_home:home,team_away:away,player:row.player||null,
      market_type:league.markets[row.market],side:['over','under'].includes(side.toLowerCase())?side.toLowerCase():side,
      line:lineless?null:row.line,book:row.book,odds:price,stake_dollars:stake,model_prob:model,
      edge_bps:model===null?null:(model-1/(price>0?1+price/100:1-100/price))*10000};
  }
  const manualGrade=row=>!!leagues[row.sport]?.manual?.includes(row.market);
  if(typeof module==='object'&&module.exports){module.exports={ticketData,identity,manualGrade,leagues};return;}
  const tickets=new Map();
  let dialog,form,draft,saving=false,trackingReady;
  const $=id=>document.getElementById('fv-track-'+id);
  const time=value=>Number.isFinite(Date.parse(value))?new Date(value).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})+' ET':'Unavailable';
  const odds=value=>(value>0?'+':'')+value;
  const script=src=>new Promise((resolve,reject)=>{
    const el=document.createElement('script');el.src=src;
    const fail=()=>{clearTimeout(timer);el.remove();reject(Error('Bet Tracker could not load.'));};
    const timer=setTimeout(fail,15000);el.onload=()=>{clearTimeout(timer);resolve();};el.onerror=fail;document.head.append(el);
  });
  async function tracker(){
    if(!trackingReady)trackingReady=(async()=>{
      if(!global.supabase?.createClient)await script('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2');
      if(!global.saveTrackedBet)await script('/tracking/bet-tracking.js?v=3');
    })().catch(error=>{trackingReady=null;throw error;});
    await trackingReady;
  }
  function updateButtons(){
    document.querySelectorAll('[data-fv-ticket]').forEach(button=>{
      const saved=!!tickets.get(button.dataset.fvTicket)?.saved,started=Date.parse(button.dataset.fvStart)<=Date.now();
      button.disabled=saved||started;button.textContent=saved?'Tracked':started?'Game started':button.dataset.fvLabel||'Track bet';
    });
  }
  function mount(){
    if(dialog)return;
    dialog=document.createElement('dialog');dialog.id='fv-bet-tracker';dialog.setAttribute('aria-labelledby','fv-track-title');
    dialog.innerHTML=`<h2 id="fv-track-title">Track this bet</h2>
      <p id="fv-track-description"></p><p class="meta" id="fv-track-quote"></p><p class="meta" id="fv-track-rules"></p>
      <form id="fv-track-form"><label for="fv-track-odds">Price you received (American)</label><input id="fv-track-odds" type="number" step="1" required>
      <label for="fv-track-stake">Stake ($)</label><input id="fv-track-stake" type="number" min="0.01" step="0.01" inputmode="decimal" required>
      <label class="fv-track-confirm"><input id="fv-track-confirm" type="checkbox" required> I confirm this bet, line, book, price and stake.</label>
      <p class="meta">Record a bet you placed. Saving does not place a wager or change the analyst review.</p>
      <p id="fv-track-feedback" role="status" aria-live="polite"></p>
      <p id="fv-track-signin" hidden><a href="/tracking/" target="_blank" rel="noopener">Sign in to Bet Tracker</a>, then return to this form and save.</p>
      <div class="actions"><button id="fv-track-save" type="submit">Save to Bet Tracker</button><button id="fv-track-close" type="button">Close</button></div>
      <p><a href="/tracking/">Open Bet Tracker →</a></p></form>`;
    document.body.append(dialog);form=$('form');
    $('close').addEventListener('click',()=>{if(!saving)dialog.close();});
    dialog.addEventListener('cancel',event=>{if(saving)event.preventDefault();});
    form.addEventListener('submit',async event=>{
      event.preventDefault();if(saving||!draft||draft.saved||!form.reportValidity())return;
      let ticket;
      try{ticket={...ticketData(draft.row,$('odds').value,$('stake').value),id:draft.id};}
      catch(error){$('feedback').textContent=error.message;return;}
      saving=true;$('feedback').textContent='Saving…';$('signin').hidden=true;
      for(const id of ['save','close','odds','stake','confirm'])$(id).disabled=true;
      try{
        await tracker();const result=await global.saveTrackedBet(ticket);
        if(result.ok){draft.saved=true;$('feedback').textContent='Saved to your Bet Tracker. Analyst review status is unchanged.';updateButtons();}
        else{$('feedback').textContent=result.error;$('signin').hidden=!result.needsSignIn;}
      }catch{$('feedback').textContent='The save could not be confirmed. Check Bet Tracker before retrying; this form retains the same ticket reference.';}
      finally{saving=false;for(const id of ['close','odds','stake','confirm'])$(id).disabled=false;$('save').disabled=!!draft.saved;}
    });
  }
  function open(row){
    mount();if(saving)return;
    const key=identity(row);
    draft=tickets.get(key)||{id:crypto.randomUUID(),row:{...row},saved:false};tickets.set(key,draft);
    form.reset();$('save').disabled=!!draft.saved;$('feedback').textContent='';$('signin').hidden=true;
    $('description').textContent=[row.sport,row.game,row.player,row.side,row.line==null?'':row.line,row.market_label,row.book_label||row.book].filter(v=>v!==''&&v!=null).join(' · ');
    $('quote').textContent=`Saved quote: ${odds(row.price)} at ${time(row.quoted_at)}. Confirm the exact line and sportsbook shown above and enter the price of the bet you actually placed. This saved quote may have changed.`;
    $('rules').textContent=(row.settlement_scope||'Verify sportsbook settlement rules.')+(manualGrade(row)?' This market is not graded automatically; settle it in Bet Tracker after the game.':'');
    $('odds').value=row.price;dialog.showModal();$('stake').focus();
  }
  // A button for one offer. Callers place it; open/saved/started state is shared page-wide.
  function button(row,label){
    const el=document.createElement('button');el.type='button';el.dataset.fvTicket=identity(row);el.dataset.fvStart=row.commence_time;
    if(label){el.dataset.fvLabel=label;el.setAttribute('aria-label','Track bet: '+label+' · '+(row.book_label||row.book));}
    el.textContent=label||'Track bet';
    el.addEventListener('click',()=>open(row));return el;
  }
  // Rows must be tagged with `sport` and listed in the order their .prop-card elements render.
  function attach(container,rows){
    Array.from(container.querySelectorAll('.prop-card')).forEach((card,index)=>{
      const row=rows[index];if(!row)return;
      const actions=document.createElement('div');actions.className='actions fv-track-actions';
      actions.append(button(row));card.append(actions);
    });
    updateButtons();
  }
  // The ledger helpers are shared with Bet Tracker's manual entry form (docs/tracking/manual-bet.js).
  global.FVOfferTracker={attach,button,open,refresh:updateButtons,isOpen:()=>!!dialog?.open,ticketData,manualGrade,leagues};
})(typeof window==='undefined'?globalThis:window);

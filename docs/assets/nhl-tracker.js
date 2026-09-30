/* NHL offer controls; actual wagers use the existing per-user Bet Tracker. */
(function (global) {
  'use strict';
  const markets={player_goals:'goals',player_assists:'assists',player_points:'points',player_shots_on_goal:'sog',totals:'team_total',h2h:'h2h',spreads:'spreads'};
  const day=value=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(value));
  const probability=value=>Number.isFinite(value)&&value>=0&&value<=1;
  const identity=r=>JSON.stringify([r.event_id,r.commence_time,r.game,r.player,r.market,r.side,r.line,r.book,r.price,r.quoted_at,r.settlement_profile]);
  function ticketData(row,price,stake,now=Date.now()) {
    price=Number(price);stake=Number(stake);
    if(!Number.isInteger(price)||Math.abs(price)<100)throw Error('Enter valid American odds, such as -110 or +150.');
    if(!Number.isFinite(stake)||stake<=0||Math.abs(stake*100-Math.round(stake*100))>1e-6)throw Error('Enter a positive stake in dollars and cents.');
    const teams=String(row.game||'').split(' @ '),home=row.home_team||teams[1],away=row.away_team||teams[0];
    if(!home||!away||!row.book||!Number.isFinite(Date.parse(row.commence_time)))throw Error('This offer is missing its game or sportsbook. Open Bet Tracker to enter it manually.');
    if(!Object.hasOwn(markets,row.market))throw Error('Open Bet Tracker to record this market manually.');
    const side=String(row.side||'');
    if(!side||(row.market!=='h2h'&&!Number.isFinite(row.line)))throw Error('This offer is missing its selection or line. Open Bet Tracker to enter it manually.');
    // Match the existing ledger: model_prob is conditional on a non-push.
    // Missing or expired model inputs must never become a market probability.
    const age=now-Date.parse(row.model_data_checked_at),win=row.final_probability,push=row.push_probability;
    const model=!row.model_withheld&&age>=0&&age<36*3600e3&&probability(row.independent_probability)&&probability(win)&&probability(push)&&push<1&&win+push<=1?win/(1-push):null;
    return {league:'NHL',game_date:day(row.commence_time),team_home:home,team_away:away,player:row.player||null,
      market_type:markets[row.market],side:['over','under'].includes(side.toLowerCase())?side.toLowerCase():side,
      line:row.market==='h2h'?null:row.line,book:row.book,odds:price,stake_dollars:stake,model_prob:model,
      edge_bps:model===null?null:(model-1/(price>0?1+price/100:1-100/price))*10000};
  }
  if(typeof module==='object'&&module.exports){module.exports={ticketData,identity};return;}
  const tickets=new Map();
  let dialog,form,draft,saving=false,trackingReady;
  const $=id=>document.getElementById('nhl-track-'+id);
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
    document.querySelectorAll('[data-nhl-ticket]').forEach(button=>{
      const saved=!!tickets.get(button.dataset.nhlTicket)?.saved;
      button.disabled=saved;button.textContent=saved?'Tracked':'Track bet';
    });
  }
  function mount(){
    if(dialog)return;
    dialog=document.createElement('dialog');dialog.id='nhl-bet-tracker';dialog.setAttribute('aria-labelledby','nhl-track-title');
    dialog.innerHTML=`<h2 id="nhl-track-title">Track this bet</h2>
      <p id="nhl-track-description"></p><p class="meta" id="nhl-track-quote"></p><p class="meta" id="nhl-track-rules"></p>
      <form id="nhl-track-form"><label for="nhl-track-odds">Price you received (American)</label><input id="nhl-track-odds" type="number" step="1" required>
      <label for="nhl-track-stake">Stake ($)</label><input id="nhl-track-stake" type="number" min="0.01" step="0.01" inputmode="decimal" required>
      <label class="nhl-track-confirm"><input id="nhl-track-confirm" type="checkbox" required> I confirm this bet, line, book, price and stake.</label>
      <p class="meta">Record a bet you placed. Saving does not place a wager or change the analyst review.</p>
      <p id="nhl-track-feedback" role="status" aria-live="polite"></p>
      <p id="nhl-track-signin" hidden><a href="/tracking/" target="_blank" rel="noopener">Sign in to Bet Tracker</a>, then return to this form and save.</p>
      <div class="actions"><button id="nhl-track-save" type="submit">Save to Bet Tracker</button><button id="nhl-track-close" type="button">Close</button></div>
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
    $('description').textContent=['NHL',row.game,row.player,row.side,row.line==null?'':row.line,row.market_label,row.book_label||row.book].filter(v=>v!==''&&v!=null).join(' · ');
    $('quote').textContent=`Saved quote: ${odds(row.price)} at ${time(row.quoted_at)}. Confirm the exact line and sportsbook shown above and enter the price of the bet you actually placed. This saved quote may have changed.`;
    $('rules').textContent=row.settlement_scope||'Verify sportsbook settlement rules.';
    $('odds').value=row.price;dialog.showModal();$('stake').focus();
  }
  function attach(container,rows){
    Array.from(container.querySelectorAll('.prop-card')).forEach((card,index)=>{
      const row=rows[index];if(!row)return;
      const actions=document.createElement('div');actions.className='actions nhl-track-actions';
      const button=document.createElement('button');button.type='button';button.dataset.nhlTicket=identity(row);button.textContent='Track bet';
      button.addEventListener('click',()=>open(row));actions.append(button);card.append(actions);
    });
    updateButtons();
  }
  global.FVNHLTracker={attach,isOpen:()=>!!dialog?.open};
})(typeof window==='undefined'?globalThis:window);

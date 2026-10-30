/* Share buttons for player prop cards. Each draws the card's exact offer (player, side, line, book,
   saved price and quote time, in Eastern Time) as a bet-slip image on this device, in the style of
   Bet Tracker's Share. Nothing is uploaded unless the reader sends the image. */
(function (global) {
  'use strict';
  const FONT='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  const ACCENT='#4FC3F7';
  const valid=value=>Number.isFinite(Date.parse(value));
  const eastern=(value,options)=>new Date(value).toLocaleString('en-US',{timeZone:'America/New_York',...options});
  const clock=value=>eastern(value,{hour:'numeric',minute:'2-digit',timeZoneName:'short'});
  const day=value=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(value));
  const odds=value=>{const n=Number(value);return Number.isFinite(n)?(n>0?'+':'')+n:'';};
  const capital=value=>{const s=String(value??'');return s.charAt(0).toUpperCase()+s.slice(1);};
  const started=row=>!(Date.parse(row.commence_time)>Date.now());
  const slug=value=>String(value??'').normalize('NFD').replace(/[̀-ͯ]/g,'').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');

  // The offered line exactly as quoted: "Over 25.5 Points"; lineless yes/no props read "Anytime TD: Yes".
  function pick(row){
    const side=capital(row.side),line=row.line==null||row.line===''?'':String(row.line);
    if(!line&&/^(yes|no)$/i.test(side))return `${row.market_label}: ${side}`;
    return [side,line,row.market_label].filter(Boolean).join(' ');
  }

  // Everything the image shows; the alt text reads the same values.
  function slip(row){
    return {
      sport:row.sport||'',game:row.game||'',player:row.player||row.market_label||'',pick:pick(row),
      odds:odds(row.price),book:row.book_label||row.book||'',started:started(row),
      start:valid(row.commence_time)?`${eastern(row.commence_time,{weekday:'short',month:'short',day:'numeric',year:'numeric'})} · ${clock(row.commence_time)}`:'',
      quoted:valid(row.quoted_at)?`Quoted ${eastern(row.quoted_at,{month:'short',day:'numeric'})}, ${clock(row.quoted_at)}`:'Quote time unavailable',
    };
  }
  function describe(row){
    const s=slip(row);
    return `${[s.sport,'player prop'].filter(Boolean).join(' ')}: ${s.player}, ${s.pick} at ${s.odds} (${s.book}). `+
      `${[s.game,s.start].filter(Boolean).join(', ')}.${s.started?' Game started.':''} ${s.quoted}; confirm the current price.`;
  }
  function fileName(row){
    return ['fourth-and-value',slug(row.sport),'prop',slug(row.player||row.market_label),valid(row.commence_time)?day(row.commence_time):'undated'].filter(Boolean).join('-')+'.png';
  }

  // Canvas helpers (roundRect drawn by hand for older Safari).
  function roundRectPath(ctx,x,y,w,h,r){
    ctx.beginPath();
    ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
    ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
  }
  function fitText(ctx,str,maxW){
    str=String(str??'');
    if(ctx.measureText(str).width<=maxW)return str;
    while(str.length&&ctx.measureText(str+'…').width>maxW)str=str.slice(0,-1);
    return str+'…';
  }

  function draw(row){
    const s=slip(row),W=600,PAD=28,HEADER_H=128,CARD_H=128,FOOTER_H=56,SCALE=2,H=HEADER_H+CARD_H+FOOTER_H;
    const canvas=document.createElement('canvas');
    canvas.width=W*SCALE;canvas.height=H*SCALE;
    const ctx=canvas.getContext('2d');
    ctx.scale(SCALE,SCALE);ctx.textBaseline='alphabetic';
    ctx.fillStyle='#0f0f0f';ctx.fillRect(0,0,W,H);
    ctx.fillStyle=ACCENT;ctx.fillRect(0,0,W,5);

    // Header
    ctx.font=`800 13px ${FONT}`;ctx.fillStyle=ACCENT;ctx.textAlign='left';
    ctx.fillText('FOURTH & VALUE',PAD,40);
    ctx.font=`600 13px ${FONT}`;ctx.fillStyle='#777';ctx.textAlign='right';
    ctx.fillText([s.sport,'PLAYER PROPS'].filter(Boolean).join(' '),W-PAD,40);
    ctx.textAlign='left';
    ctx.font=`800 28px ${FONT}`;ctx.fillStyle='#fff';
    ctx.fillText('Player prop',PAD,80);
    ctx.font=`500 15px ${FONT}`;ctx.fillStyle='#999';
    ctx.fillText(fitText(ctx,s.start,W-PAD*2),PAD,104);

    // Offer card
    const y=HEADER_H,inner=PAD+18,right=W-PAD-18;
    roundRectPath(ctx,PAD,y,W-PAD*2,CARD_H,14);
    ctx.fillStyle='#1a1a1a';ctx.fill();
    ctx.strokeStyle='#2a2a2a';ctx.lineWidth=1;ctx.stroke();

    // A started game's card shows its last pregame quote, never a live price.
    const [pillBg,pillFg,label]=s.started?['#3d3413','#FFC107','GAME STARTED']:['#3a3a3a','#cccccc','PREGAME'];
    ctx.font=`700 11px ${FONT}`;
    const pillW=ctx.measureText(label).width+20;
    roundRectPath(ctx,right-pillW,y+14,pillW,22,11);
    ctx.fillStyle=pillBg;ctx.fill();
    ctx.fillStyle=pillFg;ctx.textAlign='center';
    ctx.fillText(label,right-pillW/2,y+29);

    ctx.textAlign='left';
    ctx.font=`600 12px ${FONT}`;ctx.fillStyle='#999';
    ctx.fillText(fitText(ctx,[s.sport,s.game].filter(Boolean).join('  ·  ').toUpperCase(),right-pillW-inner-12),inner,y+30);

    // Price + book (right column)
    ctx.textAlign='right';
    ctx.font=`800 20px ${FONT}`;ctx.fillStyle='#fff';
    ctx.fillText(s.odds,right,y+62);
    const oddsW=Math.max(ctx.measureText(s.odds).width,70);
    ctx.font=`500 12px ${FONT}`;ctx.fillStyle='#999';
    ctx.fillText(fitText(ctx,s.book,110),right,y+82);

    // Player + pick (left column)
    ctx.textAlign='left';
    const leftW=right-inner-Math.max(oddsW,110)-16;
    ctx.font=`800 20px ${FONT}`;ctx.fillStyle='#fff';
    ctx.fillText(fitText(ctx,s.player,leftW),inner,y+62);
    ctx.font=`600 15px ${FONT}`;ctx.fillStyle=ACCENT;
    ctx.fillText(fitText(ctx,s.pick,leftW),inner,y+83);

    // Quote time: a saved price, so the reader confirms it before betting.
    ctx.fillStyle='#262626';ctx.fillRect(inner,y+97,right-inner,1);
    ctx.font=`500 13px ${FONT}`;ctx.fillStyle='#999';ctx.textAlign='right';
    const confirm='Confirm current price';
    ctx.fillText(confirm,right,y+117);
    const confirmW=ctx.measureText(confirm).width;
    ctx.textAlign='left';ctx.fillStyle='#bbb';
    ctx.fillText(fitText(ctx,s.quoted,right-inner-confirmW-16),inner,y+117);

    ctx.font=`500 12px ${FONT}`;ctx.fillStyle='#666';ctx.textAlign='center';
    ctx.fillText('fourthandvalue.com  ·  Analysis, not a guarantee. Bet responsibly.',W/2,H-24);
    return canvas;
  }

  let dialog,file=null,url=null;
  const $=id=>document.getElementById('fv-share-'+id);
  function mount(){
    if(dialog)return;
    dialog=document.createElement('dialog');dialog.id='fv-share';dialog.setAttribute('aria-labelledby','fv-share-title');
    dialog.innerHTML=`<div class="fv-share-inner"><h2 id="fv-share-title">Share this prop</h2><img id="fv-share-image" alt="">
      <div class="fv-share-buttons"><button type="button" id="fv-share-now">Share</button><a class="button" id="fv-share-save" download>Save</a><button type="button" id="fv-share-close">Close</button></div>
      <p class="meta">Drawn on this device. You can also press and hold the image to copy or save it.</p></div>`;
    document.body.append(dialog);
    $('close').addEventListener('click',()=>dialog.close());
    // A click outside the panel lands on the dialog itself (its backdrop).
    dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
    $('now').addEventListener('click',async()=>{
      if(!file)return;
      try{await navigator.share({files:[file]});}
      catch(error){if(error?.name!=='AbortError')alert('Sharing isn’t available here. Use Save, or press and hold the image.');}
    });
  }
  function open(row){
    mount();
    draw(row).toBlob(blob=>{
      if(!blob)return alert('Could not create the image on this device.');
      if(url)URL.revokeObjectURL(url);
      url=URL.createObjectURL(blob);file=new File([blob],fileName(row),{type:'image/png'});
      $('image').src=url;$('image').alt=describe(row);
      $('save').href=url;$('save').download=file.name;
      $('now').hidden=!(navigator.canShare&&navigator.canShare({files:[file]}));
      if(!dialog.open)dialog.showModal();
    },'image/png');
  }
  function button(row){
    const el=document.createElement('button');el.type='button';el.className='fv-share-button';el.textContent='Share';
    el.setAttribute('aria-label',`Share image: ${row.player||row.market_label} ${pick(row)} at ${row.book_label||row.book}`);
    el.addEventListener('click',()=>open(row));return el;
  }
  // Rows use Bet Tracker's offer shape, tagged with `sport` and listed in the order their .prop-card elements render.
  // The button joins the card's action row (Track bet's, when present).
  function attach(container,rows){
    Array.from(container.querySelectorAll('.prop-card')).forEach((card,index)=>{
      const row=rows[index];if(!row)return;
      let actions=Array.from(card.children).filter(el=>el.classList.contains('actions')).pop();
      if(!actions){actions=document.createElement('div');actions.className='actions fv-track-actions';card.append(actions);}
      actions.append(button(row));
    });
  }
  global.FVOfferShare={attach,button,open,draw,describe,fileName,isOpen:()=>!!dialog?.open};
})(window);

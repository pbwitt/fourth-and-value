/* Bet Tracker performance: ROI against winning bettors, the running profit bet by bet and
   results by sport, built from the bets the filters show. The summary and each part can be
   shared as a branded image drawn on this device, like the tracker's bet slips; nothing is
   uploaded. Display only: nothing here changes a bet. */
(function(global){
  'use strict';
  const SETTLED=['won','lost','push'];
  // Long-run ROI that bettors who beat the market usually report, over thousands of bets.
  const PRO_LOW=.02,PRO_HIGH=.05;
  const ROI_NOTE='Bettors who beat the market long term usually report about +2% to +5% ROI over thousands of bets. At standard −110 prices, winning 55% is +5%; winning 60% would be +14.5%.';
  const finite=Number.isFinite;
  const num=v=>{const n=Number(v);return finite(n)?n:null;};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const money=v=>(v>0?'+':v<0?'−':'')+'$'+Math.abs(v).toFixed(2);
  const pct=(v,d=1)=>finite(v)?(v>0?'+':v<0?'−':'')+Math.abs(v*100).toFixed(d)+'%':'—';
  const rate=v=>finite(v)?(v*100).toFixed(1).replace(/\.0$/,'')+'%':'—';
  const implied=o=>o>0?100/(o+100):-o/(-o+100);
  const day=d=>/^\d{4}-\d{2}-\d{2}$/.test(d||'')?new Date(d+'T12:00:00Z').toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'}):'';

  // Order placed: game date, then when the bet was logged.
  const placed=(a,b)=>String(a.game_date||'').localeCompare(String(b.game_date||''))
    ||String(a.created_at||a.timestamp||'').localeCompare(String(b.created_at||b.timestamp||''));
  const profit=b=>(num(b.payout)??0)-(num(b.stake_dollars)??0);

  function tally(bets){
    const settled=bets.filter(b=>SETTLED.includes(b.status)&&num(b.stake_dollars)>0);
    const decided=settled.filter(b=>b.status!=='push'),won=decided.filter(b=>b.status==='won').length;
    const staked=settled.reduce((a,b)=>a+num(b.stake_dollars),0),pl=settled.reduce((a,b)=>a+profit(b),0);
    const priced=decided.filter(b=>Math.abs(num(b.odds)??0)>=100);
    return {settled:settled.length,won,lost:decided.length-won,push:settled.length-decided.length,
      pending:bets.filter(b=>!SETTLED.includes(b.status)&&b.status!=='void').length,staked,pl,
      roi:staked>0?pl/staked:null,winRate:decided.length?won/decided.length:null,
      // The win rate at which these prices break even; plus-money bets lower it.
      breakEven:priced.length?priced.reduce((a,b)=>a+implied(num(b.odds)),0)/priced.length:null};
  }

  function summarize(bets){
    const list=Array.isArray(bets)?bets:[];
    const settled=list.filter(b=>SETTLED.includes(b.status)&&num(b.stake_dollars)>0).sort(placed);
    let run=0;
    const path=settled.map(b=>({bet:b,profit:profit(b),total:(run+=profit(b))}));
    const sports=[...new Set(list.map(b=>b.league||'Other'))]
      .map(sport=>({sport,...tally(list.filter(b=>(b.league||'Other')===sport))}))
      .filter(t=>t.settled||t.pending).sort((a,b)=>b.pl-a.pl||a.sport.localeCompare(b.sport));
    return {...tally(list),path,sports};
  }

  // What a share image covers: the sport when the filters show one, the dates, the count.
  function context(s){
    const sports=s.sports.filter(t=>t.settled).map(t=>t.sport);
    const dates=s.path.map(p=>day(p.bet.game_date)).filter(Boolean);
    const range=dates.length?(dates[0]===dates[dates.length-1]?dates[0]:`${dates[0]} – ${dates[dates.length-1]}`):'';
    return [sports.length===1?sports[0]:'',range,`${s.settled} settled bet${s.settled===1?'':'s'}`].filter(Boolean).join(' · ');
  }

  // Each chart is laid out once as shapes, then drawn as SVG on the page or onto a share image.
  // A shape is [kind, class, ...geometry]; a hover area ('hit') comes just before its dot.
  function scaleScene(s,W){
    const H=112,pad=6,mid=46,axis=96,roi=s.roi;
    const lo=Math.min(-.2,Math.floor(roi*10-1)/10),hi=Math.max(.2,Math.ceil(roi*10+1)/10);
    // As many ticks as fit about 46 px apart, always on multiples of the step so 0% is one.
    const fits=Math.max(2,Math.floor((W-2*pad)/46)),step=[.05,.1,.2,.25,.5,1].find(v=>(hi-lo)/v+1<=fits+1e-9)||1;
    const x=v=>pad+(W-2*pad)*((v-lo)/(hi-lo));
    const shapes=[['line','perf-axis',pad,axis,W-pad,axis]];
    for(let t=Math.ceil(lo/step-1e-9)*step;t<=hi+1e-9;t+=step){
      const v=Math.round(t*100)/100,at=x(v);
      shapes.push(['text','',at,H-2,at-pad<18?'start':W-pad-at<18?'end':'middle',`${v>0?'+':v<0?'−':''}${Math.abs(Math.round(v*100))}%`]);
    }
    shapes.push(['rect','perf-pro',x(PRO_LOW),mid-24,Math.max(3,x(PRO_HIGH)-x(PRO_LOW)),48,2],
      ['line','perf-even',x(0),mid-24,x(0),axis],
      ['circle',`perf-you ${roi>=0?'up':'down'}`,x(roi),mid,6],
      ['text','perf-lab',x((PRO_LOW+PRO_HIGH)/2),mid-30,'middle',W<520?'Winning bettors':'Winning bettors, long run'],
      ['text','perf-lab',x(roi),mid+42,'middle',`You ${pct(roi)}`],
      ['text','',x(0)-6,mid-10,'end','Break-even']);
    return {W,H,shapes};
  }

  function pathScene(s,W){
    const pts=s.path,H=Math.round(Math.min(240,Math.max(170,W*.36))),pad={l:46,r:60,t:12,b:24};
    const lo=Math.min(0,...pts.map(p=>p.total)),hi=Math.max(0,...pts.map(p=>p.total));
    const span=hi-lo||1,step=[1,2,5,10,20,25,50,100,250,500,1000].find(v=>span/v<=6)||Math.ceil(span/6);
    const ymin=Math.floor(lo/step)*step,ymax=Math.max(step,Math.ceil(hi/step)*step);
    const x=i=>pad.l+(W-pad.l-pad.r)*(pts.length>1?i/(pts.length-1):.5),y=v=>pad.t+(H-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
    const shapes=[];
    for(let v=ymin;v<=ymax+1e-9;v+=step){
      shapes.push(['line',v===0?'perf-zero':'perf-gridline',pad.l,y(v),W-pad.r,y(v)],
        ['text','',pad.l-8,y(v)+4,'end',v===0?'$0':money(v).replace(/\.00$/,'')]);
    }
    // A date under the first bet of each day, skipped where it would touch the one before.
    let last='',lastRight=-1e9;
    pts.forEach((p,i)=>{
      const d=p.bet.game_date||'';if(d===last)return;last=d;
      const at=x(i),label=day(d),w=label.length*6.6,anchor=i===0?'start':'middle';
      if((anchor==='start'?at:at-w/2)<lastRight+10)return;
      lastRight=anchor==='start'?at+w:at+w/2;
      shapes.push(['text','',at,H-6,anchor,label]);
    });
    shapes.push(['poly','perf-line',pts.map((p,i)=>[x(i),y(p.total)])]);
    const band=(W-pad.l-pad.r)/Math.max(1,pts.length-1);
    pts.forEach((p,i)=>{
      const b=p.bet,what=[b.player,b.market_type,b.side,b.line].filter(v=>v!==null&&v!==undefined&&v!=='').join(' ');
      const text=`${day(b.game_date)} · ${b.league||''} · ${what} · ${b.status==='won'?'Won':b.status==='push'?'Push':'Lost'} ${money(p.profit)} · total ${money(p.total)}`;
      shapes.push(['hit','perf-hit',x(i)-band/2,pad.t,band,H-pad.t-pad.b,text],
        ['circle',`perf-dot ${b.status}`,x(i),y(p.total),pts.length>60?2.5:W<480?3.5:4.5]);
    });
    const end=pts[pts.length-1];
    shapes.push(['text','perf-end',x(pts.length-1)+10,y(end.total)+4,'start',money(end.total)]);
    return {W,H,shapes};
  }

  const f=n=>(+n).toFixed(1);
  function toSVG({W,H,shapes}){
    let out=`<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">`;
    for(const [kind,cls,...g] of shapes){
      const c=cls?` class="${cls}"`:'';
      if(kind==='line')out+=`<line${c} x1="${f(g[0])}" y1="${f(g[1])}" x2="${f(g[2])}" y2="${f(g[3])}"/>`;
      else if(kind==='rect')out+=`<rect${c} x="${f(g[0])}" y="${f(g[1])}" width="${f(g[2])}" height="${f(g[3])}" rx="${g[4]}"/>`;
      else if(kind==='circle')out+=`<circle${c} cx="${f(g[0])}" cy="${f(g[1])}" r="${g[2]}"/>`;
      else if(kind==='text')out+=`<text${c} x="${f(g[0])}" y="${f(g[1])}" text-anchor="${g[2]}">${esc(g[3])}</text>`;
      else if(kind==='poly')out+=`<polyline${c} points="${g[0].map(([px,py])=>f(px)+','+f(py)).join(' ')}"/>`;
      else if(kind==='hit')out+=`<rect${c} tabindex="0" x="${f(g[0])}" y="${f(g[1])}" width="${f(g[2])}" height="${f(g[3])}" data-readout="${esc(g[4])}"/>`;
    }
    return out+'</svg>';
  }

  // ---- Share images: the tracker's bet-slip look, 600 px wide drawn at 2x for phones ----
  const FONT='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  const C={bg:'#0f0f0f',card:'#1a1a1a',edge:'#2a2a2a',accent:'#4FC3F7',ink:'#fff',muted:'#999',green:'#66BB6A',red:'#ff6b6b'};
  const IMG={W:600,PAD:28,SCALE:2,CHART:400,GAP:12};
  const tone=v=>v>0?C.green:v<0?C.red:C.ink;
  // The page's chart styles (index.html .perf-*), for drawing the same shapes on a canvas.
  const PAINT={
    '':{text:C.muted,font:'400 12px'},'perf-lab':{text:C.ink,font:'600 12px'},'perf-end':{text:C.ink,font:'600 12px'},
    'perf-axis':{stroke:C.edge},'perf-gridline':{stroke:C.edge},'perf-zero':{stroke:'#777',dash:[3,4]},'perf-even':{stroke:'#777',dash:[3,4]},
    'perf-pro':{fill:'rgba(79,195,247,.3)'},'perf-you up':{fill:C.green},'perf-you down':{fill:C.red},
    'perf-line':{stroke:C.accent,width:2},'perf-dot won':{fill:C.green,ring:true},'perf-dot lost':{fill:C.red,ring:true},'perf-dot push':{fill:C.muted,ring:true},
  };
  const SHARES={
    summary:{title:'My betting performance',file:'summary'},
    roi:{title:'ROI against winning bettors',file:'roi'},
    path:{title:'Running profit, bet by bet',file:'running-profit'},
    sports:{title:'Results by sport',file:'by-sport'},
  };

  // Rounded rectangles drawn by hand for older Safari.
  function roundRect(ctx,x,y,w,h,r){
    ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
    ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
  }
  function fit(ctx,text,maxW){
    let t=String(text??'');if(ctx.measureText(t).width<=maxW)return t;
    while(t.length&&ctx.measureText(t+'…').width>maxW)t=t.slice(0,-1);
    return t+'…';
  }
  function wrap(ctx,text,maxW){
    const lines=[];let line='';
    for(const word of String(text).split(/\s+/).filter(Boolean)){
      const next=line?line+' '+word:word;
      if(ctx.measureText(next).width>maxW&&line){lines.push(line);line=word;}else line=next;
    }
    return line?[...lines,line]:lines;
  }
  function drawScene(ctx,{W,shapes},x0,y0,width){
    const k=width/W;
    ctx.save();ctx.translate(x0,y0);ctx.scale(k,k);
    for(const [kind,cls,...g] of shapes){
      const p=PAINT[cls]||{};
      ctx.setLineDash(p.dash||[]);ctx.lineWidth=p.width||1;ctx.strokeStyle=p.stroke||C.edge;
      if(kind==='line'){ctx.beginPath();ctx.moveTo(g[0],g[1]);ctx.lineTo(g[2],g[3]);ctx.stroke();}
      else if(kind==='rect'){roundRect(ctx,g[0],g[1],g[2],g[3],g[4]);ctx.fillStyle=p.fill;ctx.fill();}
      else if(kind==='circle'){
        ctx.beginPath();ctx.arc(g[0],g[1],g[2],0,2*Math.PI);ctx.fillStyle=p.fill;ctx.fill();
        if(p.ring){ctx.lineWidth=2;ctx.strokeStyle=C.card;ctx.stroke();}
      }
      else if(kind==='poly'){
        ctx.beginPath();g[0].forEach(([px,py],i)=>i?ctx.lineTo(px,py):ctx.moveTo(px,py));
        ctx.lineJoin='round';ctx.lineCap='round';ctx.stroke();
      }
      else if(kind==='text'){
        ctx.font=`${p.font||'400 12px'} ${FONT}`;ctx.fillStyle=p.text||C.muted;
        ctx.textAlign={start:'left',middle:'center',end:'right'}[g[2]];ctx.fillText(g[3],g[0],g[1]);
      }
    }
    ctx.restore();
  }

  // Parts of the image, each {h, draw(ctx, x, y, w)}; cards hold the charts and the table.
  function tilesPart(s,w){
    const tw=(w-2*IMG.GAP)/3,h=96;
    const tiles=[['WIN RATE',rate(s.winRate),C.ink,finite(s.breakEven)?`Break-even ${rate(s.breakEven)}`:''],
      ['ROI',pct(s.roi),tone(s.roi),'Winning pros +2% to +5%'],
      ['PROFIT',money(s.pl),tone(s.pl),`On $${s.staked.toFixed(2)} risked`]];
    return {h,draw(ctx,x,y){
      tiles.forEach(([label,value,color,sub],i)=>{
        const tx=x+i*(tw+IMG.GAP);
        roundRect(ctx,tx,y,tw,h,14);ctx.fillStyle=C.card;ctx.fill();ctx.strokeStyle=C.edge;ctx.lineWidth=1;ctx.setLineDash([]);ctx.stroke();
        ctx.textAlign='left';ctx.font=`700 11px ${FONT}`;ctx.fillStyle=C.muted;ctx.fillText(label,tx+14,y+26);
        ctx.font=`800 26px ${FONT}`;ctx.fillStyle=color;ctx.fillText(fit(ctx,value,tw-28),tx+14,y+60);
        ctx.font=`500 12px ${FONT}`;ctx.fillStyle=C.muted;ctx.fillText(fit(ctx,sub,tw-28),tx+14,y+82);
      });
    }};
  }
  function card(title,body){
    const top=title?44:18,h=top+body.h+18;
    return {h,draw(ctx,x,y,w){
      roundRect(ctx,x,y,w,h,14);ctx.fillStyle=C.card;ctx.fill();ctx.strokeStyle=C.edge;ctx.lineWidth=1;ctx.setLineDash([]);ctx.stroke();
      if(title){ctx.textAlign='left';ctx.font=`600 15px ${FONT}`;ctx.fillStyle=C.ink;ctx.fillText(title,x+18,y+30);}
      body.draw(ctx,x+18,y+top,w-36);
    }};
  }
  function roiBody(s,w,ctx,note){
    const scene=scaleScene(s,IMG.CHART),ch=scene.H*w/IMG.CHART;
    ctx.font=`400 13px ${FONT}`;const lines=note?wrap(ctx,ROI_NOTE,w):[];
    return {h:ch+(lines.length?14+lines.length*19:0),draw(ctx,x,y,cw){
      drawScene(ctx,scene,x,y,cw);
      ctx.textAlign='left';ctx.font=`400 13px ${FONT}`;ctx.fillStyle=C.muted;
      lines.forEach((l,i)=>ctx.fillText(l,x,y+ch+28+i*19));
    }};
  }
  function pathBody(s,w){
    const scene=pathScene(s,IMG.CHART),ch=scene.H*w/IMG.CHART;
    const low=s.path.reduce((a,p)=>p.total<a.total?p:a,s.path[0]);
    const line=`Low ${money(low.total)} on ${day(low.bet.game_date)} · ${money(s.pl)} now`;
    const key=[['Win',C.green],['Loss',C.red],...(s.push?[['Push',C.muted]]:[])];
    return {h:ch+30,draw(ctx,x,y,cw){
      drawScene(ctx,scene,x,y,cw);
      let kx=x;const ky=y+ch+22;
      ctx.font=`500 13px ${FONT}`;ctx.textAlign='left';
      key.forEach(([label,color])=>{
        ctx.beginPath();ctx.arc(kx+5,ky-4,5,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();
        ctx.fillStyle=C.muted;ctx.fillText(label,kx+15,ky);kx+=15+ctx.measureText(label).width+16;
      });
      ctx.textAlign='right';ctx.fillStyle=C.ink;ctx.fillText(fit(ctx,line,x+cw-kx),x+cw,ky);
    }};
  }
  function sportsBody(s){
    const rows=s.sports,rowH=40,head=26;
    return {h:head+rows.length*rowH,draw(ctx,x,y,cw){
      const col={won:cw*.36,profit:cw*.58,roi:cw*.74},barL=cw*.79,barW=cw-barL,mid=barL+barW/2;
      const span=Math.max(...rows.map(t=>Math.abs(t.pl)),1e-9);
      ctx.font=`700 11px ${FONT}`;ctx.fillStyle=C.muted;
      ctx.textAlign='left';ctx.fillText('SPORT',x,y+14);
      ctx.textAlign='right';ctx.fillText('WON',x+col.won,y+14);ctx.fillText('PROFIT',x+col.profit,y+14);ctx.fillText('ROI',x+col.roi,y+14);
      rows.forEach((t,i)=>{
        const ry=y+head+i*rowH,base=ry+26;
        ctx.fillStyle=C.edge;ctx.fillRect(x,ry,cw,1);
        ctx.textAlign='left';ctx.font=`700 15px ${FONT}`;ctx.fillStyle=C.ink;ctx.fillText(t.sport,x,base);
        ctx.textAlign='right';ctx.font=`500 15px ${FONT}`;ctx.fillText(rate(t.winRate),x+col.won,base);
        ctx.font=`700 15px ${FONT}`;ctx.fillStyle=t.settled?tone(t.pl):C.muted;ctx.fillText(t.settled?money(t.pl):'—',x+col.profit,base);
        ctx.font=`500 15px ${FONT}`;ctx.fillStyle='#ccc';ctx.fillText(pct(t.roi),x+col.roi,base);
        ctx.fillStyle='#555';ctx.fillRect(x+mid,ry+12,1,20);
        if(t.settled&&t.pl){
          const bw=Math.max(3,Math.abs(t.pl)/span*(barW/2));
          roundRect(ctx,t.pl>0?x+mid:x+mid-bw,ry+18,bw,8,4);ctx.fillStyle=tone(t.pl);ctx.fill();
        }
      });
    }};
  }

  function shareImage(el,kind){
    const s=state.get(el),spec=SHARES[kind];
    if(!s||s.settled<2||!spec)throw new Error('Nothing to share yet');
    const {W,PAD,SCALE,GAP}=IMG,w=W-2*PAD,inner=w-36;
    const measure=document.createElement('canvas').getContext('2d');
    const all=kind==='summary',parts=[];
    if(all)parts.push(tilesPart(s,w));
    if(all||kind==='roi')parts.push(card(all?SHARES.roi.title:'',roiBody(s,inner,measure,!all)));
    if(all||kind==='path')parts.push(card(all?SHARES.path.title:'',pathBody(s,inner)));
    if(all||kind==='sports')parts.push(card(all?SHARES.sports.title:'',sportsBody(s)));
    const top=128,foot=56,H=top+parts.reduce((a,p)=>a+p.h+GAP,0)-GAP+foot;

    const canvas=document.createElement('canvas');
    canvas.width=W*SCALE;canvas.height=Math.round(H*SCALE);
    const ctx=canvas.getContext('2d');
    ctx.scale(SCALE,SCALE);ctx.textBaseline='alphabetic';
    ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);
    ctx.fillStyle=C.accent;ctx.fillRect(0,0,W,5);
    ctx.font=`800 13px ${FONT}`;ctx.fillStyle=C.accent;ctx.textAlign='left';ctx.fillText('FOURTH & VALUE',PAD,40);
    ctx.font=`600 13px ${FONT}`;ctx.fillStyle='#777';ctx.textAlign='right';ctx.fillText('BET TRACKER',W-PAD,40);
    ctx.textAlign='left';ctx.font=`800 28px ${FONT}`;ctx.fillStyle=C.ink;ctx.fillText(spec.title,PAD,80);
    ctx.font=`500 15px ${FONT}`;ctx.fillStyle=C.muted;ctx.fillText(fit(ctx,context(s),w),PAD,104);
    let y=top;
    for(const part of parts){part.draw(ctx,PAD,y,w);y+=part.h+GAP;}
    ctx.font=`500 12px ${FONT}`;ctx.fillStyle='#666';ctx.textAlign='center';
    ctx.fillText('fourthandvalue.com  ·  Analysis, not a guarantee. Bet responsibly.',W/2,H-24);

    const last=s.path[s.path.length-1].bet.game_date;
    const alt={summary:`Bet Tracker performance, ${context(s)}: ${rate(s.winRate)} won, ROI ${pct(s.roi)}, profit ${money(s.pl)}.`,
      roi:`ROI ${pct(s.roi)}, against break-even and winning bettors at about +2% to +5%.`,
      path:`Running profit after ${s.settled} settled bets, now ${money(s.pl)}.`,
      sports:`Results by sport: ${s.sports.map(t=>`${t.sport} ${rate(t.winRate)} won, ${t.settled?money(t.pl):'no settled bets'}`).join('; ')}.`}[kind];
    return {canvas,alt,name:`fourth-and-value-performance-${spec.file}${/^\d{4}-\d{2}-\d{2}$/.test(last||'')?'-'+last:''}.png`};
  }

  function sportsHTML(s){
    const span=Math.max(...s.sports.map(t=>Math.abs(t.pl)),1e-9);
    return s.sports.map(t=>{
      const w=(Math.abs(t.pl)/span*50).toFixed(1);
      const bar=t.pl>=0?`<i class="up" style="left:50%;width:${w}%"></i>`:`<i class="down" style="right:50%;width:${w}%"></i>`;
      return `<tr><th scope="row">${esc(t.sport)}</th><td>${rate(t.winRate)}${t.pending?` <small>+${t.pending} pending</small>`:''}</td>`
        +`<td class="r ${t.pl>0?'positive':t.pl<0?'negative':''}">${t.settled?money(t.pl):'—'}</td><td class="r">${pct(t.roi)}</td>`
        +`<td class="perf-barcell"><div class="perf-bar" aria-hidden="true">${t.settled?bar:''}</div></td></tr>`;
    }).join('');
  }

  const state=new WeakMap();
  function draw(el){
    const s=state.get(el);if(!s)return;
    const width=node=>Math.max(280,node?.clientWidth||600);
    const scale=el.querySelector('[data-perf=scale]'),path=el.querySelector('[data-perf=path]');
    if(scale)scale.innerHTML=toSVG(scaleScene(s,width(scale)));
    if(!path)return;
    path.innerHTML=toSVG(pathScene(s,width(path)));
    const out=el.querySelector('[data-perf=readout]'),low=s.path.reduce((a,p)=>p.total<a.total?p:a,s.path[0]);
    const rest=`Low point ${money(low.total)} on ${day(low.bet.game_date)}; ${money(s.path[s.path.length-1].total)} now. Point at a dot for the bet.`;
    out.textContent=rest;
    path.querySelectorAll('.perf-hit').forEach(h=>{
      const show=()=>{out.textContent=h.dataset.readout;},hide=()=>{out.textContent=rest;};
      h.addEventListener('pointerenter',show);h.addEventListener('focus',show);
      h.addEventListener('pointerleave',hide);h.addEventListener('blur',hide);
    });
  }

  const ICON='<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M18 16a3 3 0 0 0-2.4 1.2l-6.7-3.4a3 3 0 0 0 0-1.6l6.7-3.4A3 3 0 1 0 15 7l-6.7 3.4a3 3 0 1 0 0 3.2L15 17a3 3 0 1 0 3-1Z"/></svg>';
  const shareButton=(kind,label)=>`<button type="button" class="perf-share" data-perf-share="${kind}" aria-label="${label}">${ICON}Share</button>`;

  // Win rate and ROI are the tiles above; the panel adds what they cannot show.
  function html(s){
    if(s.settled<2)return `<h3>Performance</h3><p class="perf-note">ROI against winning bettors, your running profit and results by sport appear once two bets in this view have settled.</p>`;
    return `<div class="perf-head"><div><h3>Performance</h3><p class="perf-note">${s.settled} settled bet${s.settled===1?'':'s'} in this view${s.pending?`, ${s.pending} pending`:''}.</p></div>`
        +shareButton('summary','Share an image of this performance')+`</div>`
      +`<div class="perf-grid">`
        +`<figure class="perf-card" aria-label="${SHARES.roi.title}"><figcaption class="perf-cap">${SHARES.roi.title}${shareButton('roi','Share this chart')}</figcaption>`
          +`<div class="perf-svg" data-perf="scale" role="img" aria-label="Your ROI ${esc(pct(s.roi))}, against break-even at 0% and winning bettors at about +2% to +5% over thousands of bets."></div>`
          +`<p class="perf-note">${ROI_NOTE}</p></figure>`
        +`<figure class="perf-card" aria-label="${SHARES.path.title}"><figcaption class="perf-cap">${SHARES.path.title}${shareButton('path','Share this chart')}</figcaption>`
          +`<div class="perf-svg" data-perf="path" role="img" aria-label="Running profit after each of ${s.settled} settled bets, now ${esc(money(s.pl))}."></div>`
          +`<div class="perf-key" aria-hidden="true"><span><i class="won"></i>Win</span><span><i class="lost"></i>Loss</span>${s.push?'<span><i class="push"></i>Push</span>':''}</div>`
          +`<p class="perf-readout" data-perf="readout" aria-live="polite"></p></figure>`
      +`</div>`
      +`<div class="perf-card"><div class="perf-cap">By sport${shareButton('sports','Share results by sport')}</div>`
        +`<table class="perf-sports" aria-label="Results by sport"><thead><tr><th scope="col">Sport</th><th scope="col">Won</th><th scope="col" class="r">Profit</th><th scope="col" class="r">ROI</th><th scope="col"><span class="sr-only">Profit bar</span></th></tr></thead><tbody>${sportsHTML(s)}</tbody></table></div>`;
  }

  function render(el,bets){
    if(!el)return null;
    const s=summarize(bets);
    state.set(el,s);
    el.hidden=!(Array.isArray(bets)&&bets.length);
    el.innerHTML=html(s);
    if(s.settled>=2){
      draw(el);
      if(!el.dataset.perfObserved&&typeof ResizeObserver==='function'){
        el.dataset.perfObserved='1';
        let width=el.clientWidth;
        new ResizeObserver(()=>{if(el.clientWidth!==width){width=el.clientWidth;draw(el);}}).observe(el);
      }
    }
    return s;
  }

  const api={summarize,render,html,context,scaleScene,pathScene,toSVG,shareImage,PRO_LOW,PRO_HIGH};
  global.FVPerformance=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);

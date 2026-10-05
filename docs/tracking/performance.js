/* Bet Tracker performance: win rate against break-even, ROI against winning bettors, the
   running profit bet by bet, and results by sport. Built from the bets the filters show;
   display only, nothing here changes a bet. */
(function(global){
  'use strict';
  const SETTLED=['won','lost','push'];
  // Long-run ROI that bettors who beat the market usually report, over thousands of bets.
  const PRO_LOW=.02,PRO_HIGH=.05;
  const finite=Number.isFinite;
  const num=v=>{const n=Number(v);return finite(n)?n:null;};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const money=v=>(v>0?'+':v<0?'−':'')+'$'+Math.abs(v).toFixed(2);
  const pct=(v,d=1)=>finite(v)?(v>0?'+':v<0?'−':'')+Math.abs(v*100).toFixed(d)+'%':'—';
  const share=v=>finite(v)?(v*100).toFixed(1).replace(/\.0$/,'')+'%':'—';
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

  // ROI on one scale with break-even and the winning-bettor band.
  function scaleSVG(s,W){
    const H=112,pad=6,mid=46,axis=96,roi=s.roi;
    const lo=Math.min(-.2,Math.floor(roi*10-1)/10),hi=Math.max(.2,Math.ceil(roi*10+1)/10),step=W<520?.1:.05;
    const x=v=>pad+(W-2*pad)*((v-lo)/(hi-lo));
    let svg=`<svg viewBox="0 0 ${W} ${H}" aria-hidden="true"><line class="perf-axis" x1="${pad}" x2="${W-pad}" y1="${axis}" y2="${axis}"/>`;
    for(let t=lo;t<=hi+1e-9;t+=step){
      const v=Math.round(t*100)/100;
      svg+=`<text x="${x(v).toFixed(1)}" y="${H-2}" text-anchor="${Math.abs(v-lo)<1e-9?'start':Math.abs(v-hi)<1e-9?'end':'middle'}">${v>0?'+':v<0?'−':''}${Math.abs(Math.round(v*100))}%</text>`;
    }
    svg+=`<rect class="perf-pro" x="${x(PRO_LOW).toFixed(1)}" y="${mid-24}" width="${Math.max(3,x(PRO_HIGH)-x(PRO_LOW)).toFixed(1)}" height="48" rx="2"/>`
      +`<line class="perf-even" x1="${x(0).toFixed(1)}" x2="${x(0).toFixed(1)}" y1="${mid-24}" y2="${axis}"/>`
      +`<circle class="perf-you ${roi>=0?'up':'down'}" cx="${x(roi).toFixed(1)}" cy="${mid}" r="6"/>`
      +`<text class="perf-lab" x="${x((PRO_LOW+PRO_HIGH)/2).toFixed(1)}" y="${mid-30}" text-anchor="middle">${W<520?'Winning bettors':'Winning bettors, long run'}</text>`
      +`<text class="perf-lab" x="${x(roi).toFixed(1)}" y="${mid+42}" text-anchor="middle">You ${pct(roi)}</text>`
      +`<text x="${(x(0)-6).toFixed(1)}" y="${mid-10}" text-anchor="end">Break-even</text></svg>`;
    return svg;
  }

  // Running profit after each settled bet, with a dot per bet.
  function pathSVG(s,W){
    const pts=s.path,H=Math.round(Math.min(240,Math.max(170,W*.36))),pad={l:46,r:60,t:12,b:24};
    const lo=Math.min(0,...pts.map(p=>p.total)),hi=Math.max(0,...pts.map(p=>p.total));
    const span=hi-lo||1,step=[1,2,5,10,20,25,50,100,250,500,1000].find(v=>span/v<=6)||Math.ceil(span/6);
    const ymin=Math.floor(lo/step)*step,ymax=Math.max(step,Math.ceil(hi/step)*step);
    const x=i=>pad.l+(W-pad.l-pad.r)*(pts.length>1?i/(pts.length-1):.5),y=v=>pad.t+(H-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
    let svg=`<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">`;
    for(let v=ymin;v<=ymax+1e-9;v+=step){
      svg+=`<line class="${v===0?'perf-zero':'perf-gridline'}" x1="${pad.l}" x2="${W-pad.r}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/>`
        +`<text x="${pad.l-8}" y="${(y(v)+4).toFixed(1)}" text-anchor="end">${v===0?'$0':money(v).replace(/\.00$/,'')}</text>`;
    }
    let last='',lastX=-1e9;
    pts.forEach((p,i)=>{
      const d=p.bet.game_date||'';if(d===last)return;last=d;const at=x(i);
      if(at-lastX<52)return;lastX=at;
      svg+=`<text x="${at.toFixed(1)}" y="${H-6}" text-anchor="${i===0?'start':'middle'}">${esc(day(d))}</text>`;
    });
    svg+=`<polyline class="perf-line" points="${pts.map((p,i)=>`${x(i).toFixed(1)},${y(p.total).toFixed(1)}`).join(' ')}"/>`;
    const band=(W-pad.l-pad.r)/Math.max(1,pts.length-1);
    pts.forEach((p,i)=>{
      const b=p.bet,what=[b.player,b.market_type,b.side,b.line].filter(v=>v!==null&&v!==undefined&&v!=='').join(' ');
      const text=`${day(b.game_date)} · ${b.league||''} · ${what} · ${b.status==='won'?'Won':b.status==='push'?'Push':'Lost'} ${money(p.profit)} · total ${money(p.total)}`;
      svg+=`<rect class="perf-hit" tabindex="0" x="${(x(i)-band/2).toFixed(1)}" y="${pad.t}" width="${band.toFixed(1)}" height="${H-pad.t-pad.b}" data-readout="${esc(text)}"/>`
        +`<circle class="perf-dot ${b.status}" cx="${x(i).toFixed(1)}" cy="${y(p.total).toFixed(1)}" r="${pts.length>60?2.5:W<480?3.5:4.5}"/>`;
    });
    const end=pts[pts.length-1];
    return svg+`<text class="perf-end" x="${(x(pts.length-1)+10).toFixed(1)}" y="${(y(end.total)+4).toFixed(1)}">${money(end.total)}</text></svg>`;
  }

  function sportsHTML(s){
    const span=Math.max(...s.sports.map(t=>Math.abs(t.pl)),1e-9);
    return s.sports.map(t=>{
      const w=(Math.abs(t.pl)/span*50).toFixed(1);
      const bar=t.pl>=0?`<i class="up" style="left:50%;width:${w}%"></i>`:`<i class="down" style="right:50%;width:${w}%"></i>`;
      return `<tr><th scope="row">${esc(t.sport)}</th><td>${share(t.winRate)}${t.pending?` <small>+${t.pending} pending</small>`:''}</td>`
        +`<td class="r ${t.pl>0?'positive':t.pl<0?'negative':''}">${t.settled?money(t.pl):'—'}</td><td class="r">${pct(t.roi)}</td>`
        +`<td class="perf-barcell"><div class="perf-bar" aria-hidden="true">${t.settled?bar:''}</div></td></tr>`;
    }).join('');
  }

  const state=new WeakMap();
  function draw(el){
    const s=state.get(el);if(!s)return;
    const width=node=>Math.max(280,node?.clientWidth||600);
    const scale=el.querySelector('[data-perf=scale]'),path=el.querySelector('[data-perf=path]');
    if(scale)scale.innerHTML=scaleSVG(s,width(scale));
    if(!path)return;
    path.innerHTML=pathSVG(s,width(path));
    const out=el.querySelector('[data-perf=readout]'),low=s.path.reduce((a,p)=>p.total<a.total?p:a,s.path[0]);
    const rest=`Low point ${money(low.total)} on ${day(low.bet.game_date)}; ${money(s.path[s.path.length-1].total)} now. Point at a dot for the bet.`;
    out.textContent=rest;
    path.querySelectorAll('.perf-hit').forEach(h=>{
      const show=()=>{out.textContent=h.dataset.readout;},hide=()=>{out.textContent=rest;};
      h.addEventListener('pointerenter',show);h.addEventListener('focus',show);
      h.addEventListener('pointerleave',hide);h.addEventListener('blur',hide);
    });
  }

  function html(s){
    if(s.settled<2)return `<h3>Performance</h3><p class="perf-note">Win rate against break-even, ROI against winning bettors and your running profit appear once two bets in this view have settled.</p>`;
    const vs=finite(s.winRate)&&finite(s.breakEven)?(s.winRate>=s.breakEven?'above':'below'):'';
    return `<div class="perf-head"><h3>Performance</h3><p class="perf-note">${s.settled} settled bet${s.settled===1?'':'s'} in this view${s.pending?`, ${s.pending} pending`:''}.</p></div>`
      +`<div class="perf-facts">`
        +`<div><b>${share(s.winRate)}</b><span>Bets won</span></div>`
        +`<div><b>${share(s.breakEven)}</b><span>Break-even win rate at the prices you took</span></div>`
        +`<div><b class="${s.roi>0?'positive':s.roi<0?'negative':''}">${pct(s.roi)}</b><span>ROI on $${s.staked.toFixed(2)} risked</span></div>`
      +`</div>`
      +`<div class="perf-grid">`
        +`<figure class="perf-card"><figcaption>ROI against winning bettors</figcaption><div class="perf-svg" data-perf="scale" role="img" aria-label="Your ROI ${esc(pct(s.roi))}, against break-even at 0% and winning bettors at about +2% to +5% over thousands of bets."></div>`
          +`<p class="perf-note">You won ${share(s.winRate)} of decided bets; at your prices, break-even was ${share(s.breakEven)}${vs?`, so you finished ${vs} it`:''}. `
          +`Bettors who beat the market long term usually report about +2% to +5% ROI over thousands of bets. At standard −110 prices, winning 55% is +5%; winning 60% would be +14.5%.</p></figure>`
        +`<figure class="perf-card"><figcaption>Running profit, bet by bet</figcaption><div class="perf-svg" data-perf="path" role="img" aria-label="Running profit after each of ${s.settled} settled bets, now ${esc(money(s.pl))}."></div>`
          +`<div class="perf-key" aria-hidden="true"><span><i class="won"></i>Win</span><span><i class="lost"></i>Loss</span>${s.push?'<span><i class="push"></i>Push</span>':''}</div>`
          +`<p class="perf-readout" data-perf="readout" aria-live="polite"></p></figure>`
      +`</div>`
      +`<div class="perf-card"><table class="perf-sports"><caption>By sport</caption><thead><tr><th scope="col">Sport</th><th scope="col">Won</th><th scope="col" class="r">Profit</th><th scope="col" class="r">ROI</th><th scope="col"><span class="sr-only">Profit bar</span></th></tr></thead><tbody>${sportsHTML(s)}</tbody></table></div>`;
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

  const api={summarize,render,html,PRO_LOW,PRO_HIGH};
  global.FVPerformance=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);

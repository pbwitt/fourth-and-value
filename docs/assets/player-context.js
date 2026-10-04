/* Shared, descriptive player context. Never participates in pick selection. */
(function(global){
  'use strict';
  const finite=Number.isFinite;
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=(v,d=2)=>finite(v)?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const date=v=>/^\d{4}-\d{2}-\d{2}$/.test(v||'')?new Date(v+'T12:00:00Z').toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}):'';
  // Innings in thirds: decimal 5.67 and box-score 5.2 both mean 5⅔ (17 outs).
  const thirds=v=>{if(!finite(v))return '—';const outs=Math.round(v*3);return Math.floor(outs/3)+['','⅓','⅔'][outs%3];};
  const boxInnings=v=>{const m=/^(\d+)\.([012])$/.exec(String(v??''));return m?m[1]+['','⅓','⅔'][+m[2]]:String(v??'');};
  const measure=(v,unit,d=2)=>unit==='IP'?thirds(v):number(v,d);
  function tile(label,value,unit='',detail='',featured=false){
    return `<div class="pc-stat${featured?' pc-stat-primary':''}"><dt>${esc(label)}</dt><dd>${esc(value)}${unit?` <span>${esc(unit)}</span>`:''}${detail?`<span class="pc-detail">${esc(detail)}</span>`:''}</dd></div>`;
  }
  function inputsHTML(inputs){
    return `<dl class="pc-inputs">${inputs.filter(i=>i&&finite(i.value)).map(i=>`<div><dt>${esc(i.label)} <span class="pc-kind">${i.used?'Model input':'Context'}</span></dt><dd>${measure(i.value,i.unit,i.unit==='days'?0:2)}${i.unit?' '+esc(i.unit):''}${i.detail?`<span class="pc-detail">${esc(i.detail)}</span>`:''}</dd></div>`).join('')}</dl>`;
  }
  function fallback(r,sport){
    if(sport==='NFL'){
      let p=r.model_diagnostics?.projection;
      if(!p&&typeof r.projection_diagnostics==='string')try{p=JSON.parse(r.projection_diagnostics);}catch{}
      const market=r.market_std||r.market,stat={pass_yds:'passing_yards',pass_attempts:'attempts',pass_completions:'completions'}[market];
      if(!p||!stat)return null;
      const sample=Array.isArray(p.current_sample)?p.current_sample:[];
      const mean=key=>{const values=sample.map(g=>g[key]).filter(finite);return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;};
      const inputs=[{label:'Expected attempts',value:p.attempts,unit:'att',used:true}];
      if(market!=='pass_attempts')inputs.push({label:'Completion rate',value:finite(p.completion_rate)?p.completion_rate*100:null,unit:'%',used:true});
      if(market==='pass_yds')inputs.push({label:'Yards / completion',value:p.yards_per_completion,unit:'yd',used:true});
      inputs.push({label:'Recent weight: attempts',value:finite(p.recent_mean_weight)?p.recent_mean_weight*100:null,unit:'%',detail:'Share given to the recent sample in the career blend'});
      if(market==='pass_yds')inputs.push({label:'Recent weight: yards / completion',value:finite(p.yards_per_completion_recent_weight)?p.yards_per_completion_recent_weight*100:null,unit:'%'});
      const games=sample.slice().reverse().map(g=>({week:finite(g.week)?'Wk '+g.week:'—',att:g.attempts,cmp:g.completions,yds:g.passing_yards}));
      return {schema_version:1,source:'NFL saved forecast trace',sample_games:sample.length,sample_label:'recent appearances',stat_label:{pass_yds:'yd',pass_attempts:'att',pass_completions:'cmp'}[market],
        workload_label:'Attempts / game',workload_unit:'att',recent:sample.length?[{games:sample.length,mean:mean(stat),workload:mean('attempts')}]:[],inputs,
        games,game_columns:[['week','Week'],['att','Att'],['cmp','Cmp'],['yds','Yds']],game_focus:{pass_yds:'yds',pass_attempts:'att',pass_completions:'cmp'}[market],
        note:'Passing components are shown before matchup and venue adjustments. Recent weight describes the share given to current form versus career history. Partial appearances are not separately adjusted.'};
    }
    if(sport==='NBA'&&finite(r.baseline_mean))return {schema_version:1,source:'NBA historical baseline',through:r.baseline_last_game,
      sample_games:r.baseline_games,sample_label:'appearances',stat_label:r.market_label,workload_label:'Minutes / game',workload_unit:'min',
      recent:[{games:r.baseline_games,mean:r.baseline_mean,workload:null}],inputs:[],
      note:'Historical observed average, not a current-game projection. Minutes were not recorded in this snapshot; current role and injuries are not adjusted.'};
    if(sport==='NHL'&&r.model_inputs&&!Array.isArray(r.model_inputs)){
      const f=r.model_inputs,j=['player_shots_on_goal','player_goals','player_assists','player_points'].indexOf(r.market);
      if(j<0||!finite(f.projected_toi))return null;
      return {schema_version:1,source:'Saved NHL model inputs',through:f.last_game,sample_games:f.history_games,
        sample_label:'appearances',stat_label:['SOG','goals','assists','points'][j],recent:[],inputs:[
          {label:'Projected ice time',value:f.projected_toi,unit:'min',detail:'Prior-adjusted workload estimate'},
          {label:'Weighted production / 60',value:f.projected_toi>0?f.opportunity_means?.[j]/f.projected_toi*60:null,detail:'Prior-adjusted production rate'}],
        note:'This saved forecast includes workload and production estimates. Recent-game averages were not recorded in this snapshot. Current role and participation remain unconfirmed.'};
    }
    if(sport==='MLB'&&r.model_inputs&&r.player){
      const legacy={
        'Starter outs/start, last five':['Recent innings / start',1/3,'IP','Last 5 starts; prior-adjusted'],
        'Starter strikeouts per batter faced':['Pitcher strikeout rate',100,'%','Up to 15 starts; prior-adjusted K / batters faced'],
        'Opponent strikeouts per PA':['Opponent strikeout rate',100,'%','Up to 40 games; prior-adjusted K / PA'],
        'Expected PA input':['Expected plate appearances',1,'PA','Opportunity estimate'],
      };
      return {schema_version:1,source:'Saved MLB model inputs',through:r.model_input_through,through_label:'History through',stat_label:({pitcher_strikeouts:'K',pitcher_outs:'outs',batter_hits:'hits',batter_total_bases:'TB',batter_home_runs:'HR',batter_rbis:'RBI'})[r.market]||r.market_label,recent:[],
        inputs:Object.entries(r.model_inputs).filter(([,v])=>finite(v)).map(([key,value])=>{
          const [label,multiplier,unit,detail]=legacy[key]||[key,1,'',''];return {label,value:value*multiplier,unit,detail};
        }),
        note:'Selected context recorded with this forecast. Prior-adjusted rates are not raw recent averages; the snapshot does not identify which fields the selected model used.'};
    }
    return null;
  }
  function render(r,sport=r.sport,options={}){
    if(!r.player||r.model_withheld)return '';
    const c=r.player_context?.schema_version===1?r.player_context:fallback(r,sport);
    if(!c)return '';
    const recent=(c.recent||[]).filter(w=>w&&finite(w.games)&&w.games>0);
    const inputs=Array.isArray(c.inputs)?c.inputs:[];
    const mean=sport==='MLB'?r.model_mean:sport==='NFL'?r.mu:r.projected_mean;
    const first=recent[0],workload=inputs.find(i=>/Projected ice time|Recent innings \/ start|Expected plate appearances/.test(i.label));
    let stats=finite(mean)?tile('Model projection',number(mean),c.stat_label,'Expected count · experimental',true):'';
    if(first&&finite(first.mean))stats+=tile(`Last ${first.games} ${c.sample_label||'games'}`,number(first.mean),c.stat_label,'Observed average');
    if(workload&&finite(workload.value))stats+=tile(workload.label,measure(workload.value,workload.unit),workload.unit,'Prior-adjusted estimate');
    else if(first&&finite(first.workload))stats+=tile(c.workload_label,measure(first.workload,c.workload_unit),c.workload_unit,'Observed average');
    const recentHTML=recent.length?`<div class="pc-recent"><h4>Observed recent form</h4><table><thead><tr><th scope="col">Sample</th><th scope="col">${esc(c.stat_label)} / ${c.sample_label==='starts'?'start':'game'}</th><th scope="col">${esc(c.workload_label)}</th></tr></thead><tbody>${recent.map(w=>`<tr><th scope="row">Last ${w.games}</th><td>${number(w.mean)}</td><td>${measure(w.workload,c.workload_unit)} <span>${esc(c.workload_unit)}</span></td></tr>`).join('')}</tbody></table></div>`:'';
    const sample=finite(c.sample_games)?`${c.sample_games} ${c.sample_label||'games'} in history`:'Saved forecast';
    return `<section class="player-context" aria-label="Player form and model inputs"><div class="pc-heading"><h3>Player context</h3><span>${esc(options.saved?'Saved forecast':sample)}</span></div>${stats?`<dl class="pc-stats">${stats}</dl>`:''}<details class="pc-more"><summary>Recent form &amp; model inputs</summary>${recentHTML}${inputs.length?`<h4>Inside the forecast</h4>${inputsHTML(inputs)}<p class="pc-detail">“Model input” identifies a field used by the selected forecast model. “Context” provides additional background.</p>`:''}<p class="pc-note">${esc(c.note)}</p></details><p class="pc-source">${esc(c.source)}${date(c.through)?' · '+esc(c.through_label||'Last appearance')+' '+esc(date(c.through)):''}${options.saved?' · Preserved with this forecast':''}</p></section>`;
  }
  function season(c,season){
    if(!c)return '';
    const rows=c.group==='pitching'?[
      ['Innings',boxInnings(c.innings),'IP'],['Starts',c.starts,''],['Strikeouts',c.strikeouts,'K'],['K / 9',number(c.k_per_nine),''],['ERA',number(c.era),'']]:[
      ['Plate appearances',c.pa,'PA'],['Average',c.avg,'AVG'],['OPS',c.ops,''],['Hits',c.hits,''],['Home runs',c.home_runs,'HR'],['Total bases',c.total_bases,''],['Runs batted in',c.rbis,'RBI']];
    return `<details class="player-context pc-season"><summary>${esc(season)} regular-season totals</summary><dl class="pc-stats">${rows.filter(([,v])=>v!==null&&v!==undefined).map(([label,value,unit])=>tile(label,String(value),unit)).join('')}</dl><p class="pc-note">Descriptive season totals. Postseason results are excluded; these are not game projections.</p></details>`;
  }
  // ---- Name pop-up: the same context as a quick snapshot on the player's name. ----
  // Hover previews it on a mouse or trackpad; click, tap or Enter pins it; Escape, the close
  // button or a click elsewhere closes it. Small screens get a bottom sheet. Descriptive only.
  const ids=new WeakMap(),entries=new Map();
  let seq=0,pop=null,owner=null,pinned=false,hoverTimer=0,leaveTimer=0,installed=false,nhlPages=null;
  const contextFor=(r,sport)=>r.player_context?.schema_version===1?r.player_context:fallback(r,sport);
  const short=v=>/^\d{4}-\d{2}-\d{2}$/.test(v||'')?new Date(v+'T12:00:00Z').toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'}):esc(v??'—');
  const clock=v=>{const s=Math.round(v*60);return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');};
  // One decimal (two below 1, as in 0.35 goals); innings in thirds; NHL ice time as minutes:seconds.
  function fmt(v,unit,sport){
    if(!finite(v))return '—';
    if(unit==='IP')return thirds(v);
    if(unit==='min'&&sport==='NHL')return clock(v);
    return v.toLocaleString('en-US',{maximumFractionDigits:unit==='days'||Number.isInteger(v)?0:Math.abs(v)<1?2:1});
  }
  const withUnit=(v,unit,sport)=>fmt(v,unit,sport)+(!unit||unit==='min'&&sport==='NHL'?'':unit==='%'?'%':' '+esc(unit));
  function name(r,sport=r?.sport,options={}){
    const text=esc(r?.player??'');
    if(!r||!r.player||r.model_withheld)return text;
    const c=contextFor(r,sport);
    if(!c&&!options.season?.c)return text;
    let id=ids.get(r);
    if(!id){id='pc'+(++seq);ids.set(r,id);}
    entries.set(id,{r,sport,options,c});
    install();
    return `<button type="button" class="pc-name" data-pc="${id}" aria-haspopup="dialog" aria-expanded="false">${text}<span class="pc-cue" aria-hidden="true"></span></button>`;
  }
  function seasonLine(s){
    if(!s?.c)return '';
    const c=s.c,items=c.group==='pitching'?[[c.starts,'starts'],[c.innings!=null?boxInnings(c.innings):null,'IP'],[c.strikeouts,'K'],[finite(c.era)?c.era.toFixed(2):null,'ERA']]
      :[[c.pa,'PA'],[c.avg,'AVG'],[c.ops,'OPS'],[c.home_runs,'HR'],[c.rbis,'RBI']];
    const text=items.filter(([v])=>v!==null&&v!==undefined&&v!=='').map(([v,u])=>`${esc(v)} ${u}`).join(' · ');
    return text?`<h4>${esc(s.season)} regular season</h4><p class="pc-pop-season">${text}</p>`:'';
  }
  function gameLog(c,sport,recent){
    const cols=Array.isArray(c.game_columns)?c.game_columns.filter(x=>Array.isArray(x)&&x.length===2):[];
    const games=Array.isArray(c.games)?c.games.filter(g=>g&&typeof g==='object'):[];
    const sample=c.sample_label==='starts'?'start':'game';
    // Averages over each window, in one line under the log.
    const averages=recent.length>1&&recent.some(w=>finite(w.mean))?`<p class="pc-pop-avg">${esc(c.stat_label)} per ${sample}: ${recent.filter(w=>finite(w.mean)).map(w=>`last ${w.games} <strong>${fmt(w.mean,'',sport)}</strong>`).join(' · ')}</p>`:'';
    if(cols.length>1&&games.length){
      const css=k=>k===c.game_focus?' class="pc-focus"':k==='opp'?' class="pc-text"':'';
      const cell=(g,[k],i)=>i?`<td${css(k)}>${esc(g[k]??'—')}</td>`:`<th scope="row">${short(g[k])}</th>`;
      return `<h4>Recent games</h4><div class="pc-log"><table><thead><tr>${cols.map(([k,l])=>`<th scope="col"${css(k)}>${esc(l)}</th>`).join('')}</tr></thead><tbody>${
        games.map(g=>`<tr>${cols.map((col,i)=>cell(g,col,i)).join('')}</tr>`).join('')}</tbody></table></div>${averages}`;
    }
    if(recent.length<2)return '';
    return `<h4>Recent averages</h4><div class="pc-log"><table><thead><tr><th scope="col">Sample</th><th scope="col">${esc(c.stat_label)} / ${sample}</th><th scope="col">${esc(c.workload_label||'')}</th></tr></thead><tbody>${
      recent.map(w=>`<tr><th scope="row">Last ${w.games}</th><td>${fmt(w.mean,'',sport)}</td><td>${withUnit(w.workload,c.workload_unit,sport)}</td></tr>`).join('')}</tbody></table></div>`;
  }
  const statName=c=>({yd:'Pass yds',att:'Attempts',cmp:'Completions'})[c.stat_label]||String(c.stat_label||'').replace(/^[a-z]/,x=>x.toUpperCase());
  const workName=(c,sport)=>({IP:'IP',min:sport==='NHL'?'TOI':'Minutes',PA:'PA',att:'Attempts'})[c.workload_unit]||c.workload_label||'Workload';
  function snapshot({r,sport,options,c}){
    const recent=(c?.recent||[]).filter(w=>w&&finite(w.games)&&w.games>0);
    // Model inputs first. Only a field the selected model is known not to use is marked as
    // context; older saved snapshots do not record usage, so they carry no tag.
    const inputs=(c?.inputs||[]).filter(i=>i&&finite(i.value)).sort((a,b)=>!!b.used-!!a.used).slice(0,6);
    const mean=sport==='MLB'?r.model_mean:sport==='NFL'?r.mu:r.projected_mean,first=recent[0];
    const per=c?.sample_label==='starts'?'start':'game';
    const projection=c&&finite(mean)?`<dl class="pc-stats pc-proj">${tile('Model projection · experimental',fmt(mean,'',sport),c.stat_label,'',true)}</dl>`:'';
    let tiles='';
    if(first&&finite(first.mean))tiles+=tile(`${statName(c)} / ${per}`,fmt(first.mean,'',sport));
    if(first&&finite(first.workload))tiles+=tile(`${workName(c,sport)} / ${per}`,fmt(first.workload,c.workload_unit,sport));
    if(first&&finite(first.pitches))tiles+=tile(`Pitches / ${per}`,fmt(Math.round(first.pitches),'',sport));
    const form=tiles?`<h4>Last ${first.games} ${per}s</h4><dl class="pc-stats">${tiles}</dl>`:'';
    const model=inputs.length?`<h4>What goes into the forecast</h4><dl class="pc-pop-inputs">${inputs.map(i=>`<div><dt>${esc(i.label)}</dt><dd>${withUnit(i.value,i.unit,sport)}${i.used===false?' <span class="pc-kind">Context only</span>':''}</dd></div>`).join('')}</dl>`:'';
    const source=c?`${esc(c.source)}${date(c.through)?' · through '+esc(date(c.through)):''}${options.saved?' · saved with this forecast':''}`:'';
    return `<div class="pc-pop-head"><div><strong>${esc(r.player)}</strong>${r.game?`<span>${esc(r.game)}</span>`:''}</div><button type="button" class="pc-close" aria-label="Close player snapshot">×</button></div>`
      +projection+form+(c?gameLog(c,sport,recent):'')+model+seasonLine(options.season)
      +(c?.note?`<details class="pc-pop-note"><summary>About these numbers</summary><p>${esc(c.note)}</p></details>`:'')
      +(source?`<p class="pc-source">${source}</p>`:'')+'<p class="pc-pop-link" hidden></p>';
  }
  function place(){
    if(!pop||pop.hidden||!owner)return;
    if(!owner.isConnected)return close(false);
    const sheet=!!global.matchMedia?.('(max-width: 600px)').matches;
    pop.classList.toggle('pc-sheet',sheet);
    pop.style.maxHeight='';
    if(sheet){pop.style.left=pop.style.top='';return;}
    const b=owner.getBoundingClientRect(),gap=8;
    // An anchored snapshot follows its name; it closes once the name leaves the screen.
    if(b.bottom<0||b.top>innerHeight)return close(false);
    // Below the name if it fits, else above; otherwise the roomier side, scrolling inside
    // rather than covering the name. The site's sticky navigation bar is the top edge.
    const nav=document.querySelector('.fv-nav')?.getBoundingClientRect(),edge=Math.max(gap,nav&&nav.bottom>0?nav.bottom+gap:gap);
    const below=innerHeight-b.bottom-2*gap,above=b.top-gap-edge,w=pop.offsetWidth;
    let h=pop.offsetHeight,top;
    if(h<=below)top=b.bottom+gap;
    else if(h<=above)top=b.top-gap-h;
    else{
      const down=below>=above;
      pop.style.maxHeight=Math.max(120,down?below:above)+'px';h=pop.offsetHeight;
      top=down?b.bottom+gap:b.top-gap-h;
    }
    pop.style.top=Math.max(edge,Math.min(top,innerHeight-h-gap))+'px';
    pop.style.left=Math.max(gap,Math.min(b.left,innerWidth-w-gap))+'px';
  }
  async function playerPage(entry,id){
    // NHL players have their own page (/nhl/players/); link it when the index knows them.
    if(entry.sport!=='NHL'||typeof fetch!=='function')return;
    try{
      nhlPages=nhlPages||fetch('/nhl/players/players.json').then(r=>r.ok?r.json():{}).then(d=>d?.players||{}).catch(()=>({}));
      const players=await nhlPages,key=s=>String(s||'').normalize('NFKD').replace(/[^\w]/g,'').toLowerCase();
      const hits=Object.entries(players).filter(([,p])=>entry.r.player_id?p.player_id===entry.r.player_id:key(p.name)===key(entry.r.player));
      const link=pop?.querySelector('.pc-pop-link');
      if(hits.length===1&&link&&!pop.hidden&&owner?.dataset.pc===id){
        link.innerHTML=`<a href="/nhl/players/${encodeURIComponent(hits[0][0])}/">Full player page →</a>`;link.hidden=false;place();
      }
    }catch{}
  }
  function open(t,pin){
    const entry=entries.get(t.dataset.pc);
    if(!entry)return;
    if(!pop){
      pop=document.createElement('div');pop.className='pc-pop player-context';pop.setAttribute('role','dialog');pop.tabIndex=-1;pop.hidden=true;
      pop.addEventListener('click',e=>{if(e.target.closest('.pc-close'))close(true);else pinned=true;});
      document.body.append(pop);
    }
    clearTimeout(hoverTimer);clearTimeout(leaveTimer);
    if(owner&&owner!==t)owner.setAttribute('aria-expanded','false');
    owner=t;pinned=pin;
    pop.innerHTML=snapshot(entry);pop.setAttribute('aria-label',entry.r.player+' player snapshot');
    pop.hidden=false;pop.scrollTop=0;t.setAttribute('aria-expanded','true');place();
    if(pin)pop.focus({preventScroll:true});
    playerPage(entry,t.dataset.pc);
  }
  function close(focus){
    clearTimeout(hoverTimer);clearTimeout(leaveTimer);
    if(!pop||pop.hidden)return;
    pop.hidden=true;
    const was=owner,wasPinned=pinned;owner=null;pinned=false;
    if(was){was.setAttribute('aria-expanded','false');if(focus&&wasPinned&&was.isConnected)was.focus();}
  }
  function install(){
    if(installed||typeof document==='undefined')return;
    installed=true;
    const hover=()=>!!global.matchMedia?.('(hover: hover) and (pointer: fine)').matches;
    document.addEventListener('click',e=>{
      const t=e.target.closest?.('.pc-name');
      if(t){e.preventDefault();if(owner===t&&pinned)close(true);else open(t,true);return;}
      if(pop&&!pop.hidden&&!pop.contains(e.target))close(false);
    });
    document.addEventListener('pointerover',e=>{
      if(!hover())return;
      const t=e.target.closest?.('.pc-name');
      if(t){clearTimeout(leaveTimer);if(owner!==t&&!pinned){clearTimeout(hoverTimer);hoverTimer=setTimeout(()=>open(t,false),200);}}
      else if(pop?.contains(e.target))clearTimeout(leaveTimer);
    });
    document.addEventListener('pointerout',e=>{
      if(!hover()||pinned)return;
      const from=e.target.closest?.('.pc-name')||(pop?.contains(e.target)?pop:null);
      if(!from)return;
      clearTimeout(hoverTimer);
      const to=e.relatedTarget;
      if(to&&(pop?.contains(to)||owner?.contains(to)))return;
      leaveTimer=setTimeout(()=>close(false),220);
    });
    // Keyboard: Escape closes and returns focus; tabbing away closes a pinned snapshot.
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&pop&&!pop.hidden){e.preventDefault();close(true);}});
    document.addEventListener('focusin',e=>{if(pop&&!pop.hidden&&pinned&&!pop.contains(e.target)&&e.target!==owner)close(false);});
    global.addEventListener('resize',place);
    global.addEventListener('scroll',place,{passive:true,capture:true});
  }
  // snapshot: the pop-up body as HTML, for tests and server-free previews.
  const api={render,season,name,snapshot:(r,sport=r.sport,options={})=>snapshot({r,sport,options,c:contextFor(r,sport)})};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.FVPlayerContext=api;
})(typeof window!=='undefined'?window:globalThis);

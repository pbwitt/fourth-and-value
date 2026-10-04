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
      // How the projection is built: the model's own components and adjustment stages.
      const st=p.mean_stages||{},unit={pass_yds:'yd',pass_attempts:'att',pass_completions:'cmp'}[market],steps=[];
      steps.push({label:'Expected attempts',value:p.attempts,unit:''});
      if(market!=='pass_attempts')steps.push({label:'Completion rate',value:finite(p.completion_rate)?p.completion_rate*100:null,unit:'%',op:'×'});
      if(market==='pass_yds')steps.push({label:'Yards per completion',value:p.yards_per_completion,unit:'',op:'×'});
      if(finite(st.before_adjustments)&&market!=='pass_attempts')steps.push({label:'Before matchup',value:st.before_adjustments,unit,op:'='});
      const ratio=(a,b)=>finite(a)&&finite(b)&&b>0?a/b:null,defense=ratio(st.after_defense,st.before_adjustments),venue=ratio(st.after_venue,st.after_defense),other=ratio(st.final,st.after_venue);
      if(finite(defense))steps.push({label:'Opposing pass defense',value:defense,unit:'×',op:'×'});
      if(finite(venue)&&Math.abs(venue-1)>1e-6)steps.push({label:venue>1?'Home field':'Road game',value:venue,unit:'×',op:'×'});
      if(finite(other)&&Math.abs(other-1)>1e-6)steps.push({label:'Injury or manual adjustment',value:other,unit:'×',op:'×'});
      if(finite(st.final))steps.push({label:'Projection',value:st.final,unit,op:'='});
      const explain={build:{steps,note:'Attempts, completion rate and yards per completion are blended from this season and his career, then scaled for the opponent and venue.'},
        trend:{label:unit,note:`This season’s games count ${finite(p.recent_mean_weight)?Math.round(p.recent_mean_weight*100)+'%':'part'} in the attempts estimate; career history and the position average fill the rest.`,
          rows:sample.map(g=>[finite(g.week)?'Wk '+g.week:'',g[stat],null,null,g.attempts])},
        blend:[{label:'Attempts',own:p.recent_mean_weight,own_label:'this season',rest_label:'career',detail:`${sample.length} game${sample.length===1?'':'s'} this season`},
          ...(market==='pass_yds'&&finite(p.yards_per_completion_recent_weight)?[{label:'Yards per completion',own:p.yards_per_completion_recent_weight,own_label:'this season',rest_label:'career'}]:[])],
        opponent:finite(defense)?{label:'Opposing pass defense',items:[{label:'Defense adjustment',value:defense,unit:'×',league:1,used:true,rank:defense>1?'Allows more passing yards than average':defense<1?'Allows fewer passing yards than average':'About average'}]}:null,
        missing:['Weather and wind','Game script (trailing teams pass more)','Teammate injuries and target changes']};
      return {schema_version:1,source:'NFL saved forecast trace',...explain,sample_games:sample.length,sample_label:'recent appearances',stat_label:{pass_yds:'yd',pass_attempts:'att',pass_completions:'cmp'}[market],
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
    const d=unit==='×'?2:unit==='days'||Number.isInteger(v)?0:Math.abs(v)<1?2:1;
    return v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  }
  const withUnit=(v,unit,sport)=>fmt(v,unit,sport)+(!unit||unit==='min'&&sport==='NHL'?'':unit==='%'||unit==='×'?unit:' '+esc(unit));
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
  // ---- Small charts: inline SVG. Every value is also in text (caption, readout, tables, chart label). ----
  const W=320;
  const plain=v=>/^\d{4}-\d{2}-\d{2}$/.test(v||'')?new Date(v+'T12:00:00Z').toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'}):String(v??'');
  const share=v=>finite(v)?Math.round(v*100)+'%':'—';
  const odds=v=>finite(v)?(v*100).toFixed(1).replace(/\.0$/,'')+'%':'—';
  const lineText=v=>finite(v)?String(+v.toFixed(2)):'';
  const winner=(k,line,side)=>!finite(line)?null:k===line?'push':side==='Under'?k<line:k>line;
  // A rounded data end (4px) and a square baseline, per the house chart spec.
  function bar(x,top,w,h,cls,opacity=1){
    if(!(h>0))return '';
    const r=Math.min(4,w/2,h),b=top+h;
    return `<path class="${cls}" d="M${x.toFixed(1)},${b.toFixed(1)}V${(top+r).toFixed(1)}Q${x.toFixed(1)},${top.toFixed(1)} ${(x+r).toFixed(1)},${top.toFixed(1)}H${(x+w-r).toFixed(1)}Q${(x+w).toFixed(1)},${top.toFixed(1)} ${(x+w).toFixed(1)},${(top+r).toFixed(1)}V${b.toFixed(1)}Z"${opacity<1?` fill-opacity="${opacity.toFixed(2)}"`:''}/>`;
  }
  function figure(cls,label,svg,below,caption,note){
    return `<figure class="pc-chart ${cls}"><div class="pc-plot" role="img" aria-label="${esc(label)}">${svg}</div>${below||''}<figcaption class="pc-readout" aria-live="polite" data-default="${esc(caption)}">${esc(caption)}</figcaption>${note?`<p class="pc-chart-note">${esc(note)}</p>`:''}</figure>`;
  }
  function workText(v,c,sport){
    if(!finite(v))return '';
    if(sport==='NHL')return clock(v)+' TOI';
    if(sport==='MLB')return Math.round(v)+(c.sample_label==='starts'?' pitches':' PA');
    if(sport==='NFL')return Math.round(v)+' att';
    return Math.round(v)+' min';
  }
  // Last ten games against tonight's line. Faded bars count less in the model.
  function trendChart(c,r,sport){
    const t=c?.trend,rows=(Array.isArray(t?.rows)?t.rows:[]).filter(x=>Array.isArray(x)&&finite(x[1]));
    if(rows.length<2)return '';
    const line=finite(r.line)?r.line:null,side=r.side==='Under'?'Under':'Over',H=84,n=rows.length,band=W/n,bw=Math.min(22,band-4);
    const max=Math.max(...rows.map(x=>x[1]),line??0,1)*1.15,y=v=>H-v/max*H;
    let marks='',targets='',wins=0,decided=0;
    const words=[];
    rows.forEach(([d,v,w,opp,work],i)=>{
      const outcome=winner(v,line,side),x=i*band+(band-bw)/2,h=v>0?Math.max(2,H-y(v)):0;
      if(outcome===true||outcome===false){decided++;if(outcome)wins++;}
      marks+=bar(x,H-h,bw,h,outcome==='push'?'pc-push':outcome?'pc-hit':'pc-miss',finite(w)?.3+.7*Math.min(1,Math.max(0,w)):1);
      const text=[plain(d),opp,`${fmt(v,'',sport)} ${t.label}`,workText(work,c,sport),finite(w)?`counts ${share(w)}`:''].filter(Boolean).join(' · ');
      words.push(text);
      targets+=`<rect class="pc-target" x="${(i*band).toFixed(1)}" y="0" width="${band.toFixed(1)}" height="${H}" data-readout="${esc(text)}"/>`;
    });
    const rule=line!==null?`<line class="pc-rule" x1="0" x2="${W}" y1="${y(line).toFixed(1)}" y2="${y(line).toFixed(1)}"/>`:'';
    const tag=line!==null?`<span class="pc-rule-label" style="top:${(y(line)/H*100).toFixed(1)}%">${esc(lineText(line))}</span>`:'';
    const svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${marks}${rule}${targets}</svg>${tag}`;
    const axis=`<div class="pc-axis"><span>${esc(plain(rows[0][0]))}</span><span>${esc(plain(rows[n-1][0]))}</span></div>`;
    const caption=line!==null?`${wins} of ${decided} cleared ${side} ${lineText(line)}${rows.some(x=>finite(x[2]))?' · faded games count less':''}`:`Last ${n} games`;
    return `<h4>Last ${n} ${c.sample_label==='starts'?'starts':'games'} vs. the line</h4>`+figure('pc-trend',`Last ${n}: ${words.join('; ')}`,svg,axis,caption,t.note);
  }
  // The model's chance of each outcome, with the bet's winning outcomes highlighted.
  function distChart(c,r,sport){
    const d=c?.distribution;
    if(Array.isArray(d?.empirical))return pastChart(d.empirical.filter(finite),c,r);
    const p=(Array.isArray(d?.p)?d.p:[]).map(Number);
    if(p.length<2||!p.every(finite))return '';
    const line=finite(r.line)?r.line:null,side=r.side==='Under'?'Under':'Over',H=72,n=p.length,band=W/n,bw=Math.min(22,band-4);
    const max=Math.max(...p)*1.1||1,start=finite(d.start)?d.start:0;
    let marks='',targets='',labels='',win=0,push=0;
    const words=[];
    p.forEach((v,i)=>{
      const k=start+i,outcome=winner(k,line,side),h=v>0?Math.max(1.5,v/max*H):0;
      if(outcome==='push')push+=v;else if(outcome)win+=v;
      const name=(i===0&&d.low?'≤':'')+k+(i===n-1&&d.high?'+':'');
      marks+=bar(i*band+(band-bw)/2,H-h,bw,h,outcome==='push'?'pc-push':outcome?'pc-hit':'pc-miss');
      const text=`${name} ${c.stat_label}: ${odds(v)}`;
      words.push(text);
      targets+=`<rect class="pc-target" x="${(i*band).toFixed(1)}" y="0" width="${band.toFixed(1)}" height="${H}" data-readout="${esc(text)}"/>`;
      labels+=`<span>${n>14&&i%2?'':esc(name)}</span>`;
    });
    if(line!==null){
      const row=sport==='MLB'?[r.model_probability,r.model_push_probability]:sport==='NHL'?[r.independent_probability,r.push_probability]:[];
      if(finite(row[0])){win=row[0];push=finite(row[1])?row[1]:push;}
    }
    const at=line!==null?((line-start+.5)/n*W):null;
    const rule=at!==null&&at>0&&at<W?`<line class="pc-rule" x1="${at.toFixed(1)}" x2="${at.toFixed(1)}" y1="0" y2="${H}"/>`:'';
    const svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${marks}${rule}${targets}</svg>`;
    const other=side==='Over'?'Under':'Over';
    const caption=line!==null?`${side} ${lineText(line)}: ${odds(win)}${push>.0005?` · Push: ${odds(push)}`:''} · ${other}: ${odds(Math.max(0,1-win-push))}`:'Chance of each outcome';
    return `<h4>Range of outcomes</h4>`+figure('pc-dist',`Model chances: ${words.join('; ')}`,svg,`<div class="pc-ticks" style="grid-template-columns:repeat(${n},1fr)">${labels}</div>`,caption,
      'The model gives a chance for every count, not one guess. Highlighted bars win this bet.');
  }
  // No model yet (NBA): how the recent games fell around the line.
  function pastChart(values,c,r){
    if(values.length<5)return '';
    const line=finite(r.line)?r.line:null,side=r.side==='Under'?'Under':'Over';
    const lo=Math.floor(Math.min(...values)),hi=Math.ceil(Math.max(...values)),size=Math.max(1,Math.ceil((hi-lo+1)/10)),n=Math.floor((hi-lo)/size)+1;
    const counts=Array(n).fill(0);values.forEach(v=>counts[Math.min(n-1,Math.floor((v-lo)/size))]++);
    const H=64,band=W/n,bw=Math.min(22,band-4),max=Math.max(...counts);
    let marks='',targets='',labels='';
    counts.forEach((k,i)=>{
      const from=lo+i*size,to=from+size-1,mid=(from+to)/2,outcome=winner(mid,line,side),h=k?Math.max(2,k/max*H):0;
      marks+=bar(i*band+(band-bw)/2,H-h,bw,h,outcome===true?'pc-hit':'pc-miss');
      const text=`${size>1?`${from}–${to}`:from}: ${k} game${k===1?'':'s'}`;
      targets+=`<rect class="pc-target" x="${(i*band).toFixed(1)}" y="0" width="${band.toFixed(1)}" height="${H}" data-readout="${esc(text)}"/>`;
      labels+=`<span>${n>8&&i%2?'':esc(size>1?from:from)}</span>`;
    });
    const over=values.filter(v=>line!==null&&v>line).length,under=values.filter(v=>line!==null&&v<line).length;
    const svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${marks}${targets}</svg>`;
    return `<h4>Last ${values.length} games</h4>`+figure('pc-dist',`Past results: ${counts.join(', ')}`,svg,`<div class="pc-ticks" style="grid-template-columns:repeat(${n},1fr)">${labels}</div>`,
      line!==null?`${over} over ${lineText(line)} · ${under} under`:`${values.length} games`,'Past results, not a forecast.');
  }
  // Our number next to the market's and the break-even for this price, on one scale.
  function marketStrip(r,sport){
    const cond=(w,p)=>finite(w)?(finite(p)&&p<1?w/(1-p):w):null;
    const v={NHL:[r.conditional_probability??cond(r.independent_probability,r.push_probability),r.market_probability,r.book_probability],
      MLB:[r.model_conditional_probability,r.consensus_probability??r.fair_probability,r.book_probability],
      NFL:[cond(r.model_prob,r.push_prob),r.consensus_prob??r.prob_devig,r.mkt_prob],
      NBA:[r.baseline_probability,r.consensus_probability??r.fair_probability,r.book_probability]}[sport];
    if(!v||!finite(v[0])||!finite(v[2]))return '';
    const shown=v.filter(finite),span=Math.max(.24,Math.max(...shown)-Math.min(...shown)+.12),mid=(Math.max(...shown)+Math.min(...shown))/2;
    const lo=Math.max(0,Math.min(1-span,mid-span/2)),hi=Math.min(1,lo+span),x=p=>((p-lo)/(hi-lo)*(W-16)+8).toFixed(1);
    const model=sport==='NBA'?'Past hit rate':'Model';
    const svg=`<svg viewBox="0 0 ${W} 22" aria-hidden="true"><line class="pc-track" x1="8" x2="${W-8}" y1="11" y2="11"/>`
      +`<line class="pc-even" x1="${x(v[2])}" x2="${x(v[2])}" y1="3" y2="19"/>`
      +(finite(v[1])?`<circle class="pc-market" cx="${x(v[1])}" cy="11" r="5"/>`:'')
      +`<circle class="pc-model" cx="${x(v[0])}" cy="11" r="5"/></svg>`;
    const gap=(v[0]-v[2])*100;
    const keys=`<p class="pc-keys"><span class="pc-key-model">${esc(model)} ${odds(v[0])}</span>${finite(v[1])?`<span class="pc-key-market">Market ${odds(v[1])}</span>`:''}<span class="pc-key-even">Break-even ${odds(v[2])}</span></p>`;
    const caption=`${model} vs. break-even: ${gap>=0?'+':'−'}${Math.abs(gap).toFixed(1)} points${finite(r.line)&&Number.isInteger(r.line)?' · chances exclude pushes':''}`;
    return `<figure class="pc-chart pc-strip"><div class="pc-plot" role="img" aria-label="${esc(`${model} ${odds(v[0])}, market ${odds(v[1])}, break-even ${odds(v[2])}`)}">${svg}</div>${keys}<figcaption class="pc-readout">${esc(caption)}</figcaption></figure>`;
  }
  function buildHTML(c,sport){
    const steps=(c?.build?.steps||[]).filter(s=>s&&finite(s.value));
    if(!steps.length)return '';
    return `<h4>How the number is built</h4><ol class="pc-build">${steps.map(s=>`<li class="${s.op==='='||s.op==='→'?'pc-total':''}"><span class="pc-op" aria-hidden="true">${esc(s.op||'')}</span><span class="pc-step">${esc(s.label)}</span><strong>${withUnit(s.value,s.unit,sport)}</strong></li>`).join('')}</ol>${c.build.note?`<p class="pc-chart-note">${esc(c.build.note)}</p>`:''}`;
  }
  function blendHTML(c){
    const items=(Array.isArray(c?.blend)?c.blend:[]).filter(b=>b&&finite(b.own));
    if(!items.length)return '';
    return `<h4>How much his own recent games count</h4>${items.map(b=>{const own=b.own_label||'his games',rest=b.rest_label||'average';return `<div class="pc-blend"><div class="pc-blend-label"><span>${esc(b.label)}</span><span>${share(b.own)} ${esc(own)} · ${share(1-b.own)} ${esc(rest)}</span></div><div class="pc-meter" role="img" aria-label="${esc(`${b.label}: ${share(b.own)} ${own}, ${share(1-b.own)} ${rest}`)}"><span style="width:${(b.own*100).toFixed(1)}%"></span></div>${b.detail?`<p class="pc-chart-note">${esc(b.detail)}</p>`:''}</div>`;}).join('')}<p class="pc-chart-note">Small samples lean on the longer history, so one hot or cold week moves the forecast less.</p>`;
  }
  function opponentHTML(c,sport){
    const o=c?.opponent,items=(Array.isArray(o?.items)?o.items:[]).filter(i=>i&&finite(i.value));
    if(!items.length)return '';
    return `<h4>${esc(o.label||'Opponent')}${o.team?` · ${esc(o.team)}`:''}</h4><dl class="pc-pop-inputs pc-opp">${items.map(i=>`<div><dt>${esc(i.label)}</dt><dd>${withUnit(i.value,i.unit,sport)}${i.rank?`<span class="pc-detail">${esc(i.rank)}</span>`:''}${finite(i.league)?`<span class="pc-detail">League ${withUnit(i.league,i.unit,sport)}</span>`:''}<span class="pc-kind">${i.used?'Used by the model':'Not in this model'}</span></dd></div>`).join('')}</dl>`;
  }
  function missingHTML(c){
    const items=(Array.isArray(c?.missing)?c.missing:[]).filter(x=>typeof x==='string'&&x);
    return items.length?`<h4>What the model doesn’t know</h4><ul class="pc-missing">${items.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:'';
  }
  // ---- Track record: how past forecasts for this market turned out (published backtests). ----
  const tracks={};
  const trackSource=sport=>({MLB:'/mlb/data/validation.json',NHL:'/nhl/data/track-record.json'})[sport];
  function trackFor(sport){
    const url=trackSource(sport);
    if(!url||typeof fetch!=='function')return Promise.resolve(null);
    return tracks[sport]=tracks[sport]||fetch(url).then(x=>x.ok?x.json():null).catch(()=>null);
  }
  // One shape for both sources: MLB's validation report and the NHL track-record file.
  function trackTest(report,r,sport){
    if(!report)return null;
    if(sport==='MLB'){
      const post=r.game_type&&r.game_type!=='R',test=(post?report.postseason:report.regular)?.[r.market]||report.regular?.[r.market];
      return test&&{bins:test.calibration_bins,forecasts:test.forecasts||test.samples,skill:test.brier_skill,ece:test.ece,
        window:post?`${report.postseason_year} postseason`:`${date(report.test_start)}–${date(report.test_end)}`,note:''};
    }
    // NHL: only the running model version's evidence describes this forecast.
    const m=report.markets?.[r.market];
    if(!m||report.model_version!==r.model_version)return null;
    return {bins:m.calibration_bins,forecasts:m.forecasts,ece:m.ece,window:`${date(report.test_start)}–${date(report.test_end)}`,
      note:`Checked as “${m.side}” for every game. ${report.note||''}`};
  }
  function trackHTML(report,r,sport){
    const test=trackTest(report,r,sport);
    const bins=(test?.bins||[]).filter(b=>finite(b.predicted)&&finite(b.observed)&&b.n>0);
    if(!bins.length)return '<p class="pc-chart-note">No published track record for this market yet.</p>';
    const H=150,x=p=>(8+p*(W-16)).toFixed(1),y=p=>(H-8-p*(H-16)).toFixed(1);
    let dots='',targets='';
    bins.forEach(b=>{
      const text=`Said ${share(b.predicted)} → happened ${share(b.observed)} (${b.n.toLocaleString('en-US')} chances)`;
      dots+=`<circle class="pc-dot" cx="${x(b.predicted)}" cy="${y(b.observed)}" r="4"/>`;
      targets+=`<circle class="pc-target" cx="${x(b.predicted)}" cy="${y(b.observed)}" r="12" data-readout="${esc(text)}"/>`;
    });
    const svg=`<svg viewBox="0 0 ${W} ${H}" aria-hidden="true"><line class="pc-track" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>${dots}${targets}</svg><span class="pc-y-label">Happened ↑</span>`;
    const skill=finite(test.skill)?` Accuracy ${test.skill>=0?'+':''}${Math.round(test.skill*100)}% vs. a baseline from past results (Brier skill).`:'';
    const gap=finite(test.ece)?` Average gap between said and happened: ${(test.ece*100).toFixed(1)} points.`:'';
    return figure('pc-record','Predicted vs. observed: '+bins.map(b=>`${share(b.predicted)} to ${share(b.observed)}`).join('; '),svg,
      '<div class="pc-axis"><span>Model said 0%</span><span>100%</span></div>',
      'On the diagonal = it happened as often as the model said.',
      `Tested on ${(test.forecasts||0).toLocaleString('en-US')} forecasts it had not seen (${test.window}).${skill}${gap}${test.note?' '+test.note:''}`);
  }
  let lastTab='form';
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
    const note=c?.note?`<details class="pc-pop-note"><summary>About these numbers</summary><p>${esc(c.note)}</p></details>`:'';
    const formPanel=(c?trendChart(c,r,sport):'')+form+(c?gameLog(c,sport,recent):'')+seasonLine(options.season);
    const modelPanel=c?buildHTML(c,sport)+distChart(c,r,sport)+opponentHTML(c,sport)+blendHTML(c)+model+missingHTML(c)+note:'';
    const record=sport!=='NBA'&&trackSource(sport);
    const tabs=[['form','Form',formPanel],['model','How it works',modelPanel],...(record?[['record','Track record','<div class="pc-record"><p class="pc-chart-note">Loading the track record…</p></div>']]:[])]
      .filter(([,,html])=>html);
    const pick=tabs.some(([k])=>k===lastTab)?lastTab:tabs[0]?.[0];
    const body=tabs.length>1?`<div class="pc-tabs" role="tablist" aria-label="Player snapshot sections">${tabs.map(([k,l])=>`<button type="button" role="tab" id="pc-tab-${k}" data-tab="${k}" aria-controls="pc-panel-${k}" aria-selected="${k===pick}" tabindex="${k===pick?0:-1}">${l}</button>`).join('')}</div>`
      +tabs.map(([k,,html])=>`<div class="pc-panel" role="tabpanel" id="pc-panel-${k}" aria-labelledby="pc-tab-${k}"${k===pick?'':' hidden'}>${html}</div>`).join('')
      :tabs.map(([,,html])=>html).join('');
    return `<div class="pc-pop-head"><div><strong>${esc(r.player)}</strong>${r.game?`<span>${esc(r.game)}</span>`:''}</div><button type="button" class="pc-close" aria-label="Close player snapshot">×</button></div>`
      +projection+marketStrip(r,sport)+body
      +(source?`<p class="pc-source">${source}</p>`:'')+'<p class="pc-pop-link" hidden></p>';
  }
  function selectTab(key){
    if(!pop)return;
    lastTab=key;
    pop.querySelectorAll('[role=tab]').forEach(t=>{const on=t.dataset.tab===key;t.setAttribute('aria-selected',on);t.tabIndex=on?0:-1;});
    pop.querySelectorAll('[role=tabpanel]').forEach(p=>{p.hidden=p.id!=='pc-panel-'+key;});
    place();
  }
  async function fillRecord(entry,id){
    const box=pop?.querySelector('.pc-record');
    if(!box)return;
    const report=await trackFor(entry.sport);
    if(pop&&!pop.hidden&&owner?.dataset.pc===id&&box.isConnected){box.innerHTML=trackHTML(report,entry.r,entry.sport);place();}
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
      pop.addEventListener('click',e=>{
        const tab=e.target.closest('[role=tab]');
        if(e.target.closest('.pc-close'))return close(true);
        pinned=true;
        if(tab)selectTab(tab.dataset.tab);
        const mark=e.target.closest?.('[data-readout]');
        if(mark)readout(mark);
      });
      pop.addEventListener('keydown',e=>{
        const tab=e.target.closest('[role=tab]');
        if(!tab||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
        const all=[...pop.querySelectorAll('[role=tab]')],i=all.indexOf(tab);
        const next=all[e.key==='Home'?0:e.key==='End'?all.length-1:(i+(e.key==='ArrowRight'?1:-1)+all.length)%all.length];
        e.preventDefault();selectTab(next.dataset.tab);next.focus();
      });
      // Chart readouts: hover or tap a bar or dot; leaving the chart restores its summary.
      pop.addEventListener('pointerover',e=>{const mark=e.target.closest?.('[data-readout]');if(mark)readout(mark);});
      pop.addEventListener('pointerout',e=>{
        const chart=e.target.closest?.('.pc-chart');
        if(chart&&!chart.contains(e.relatedTarget))reset(chart);
      });
      document.body.append(pop);
    }
    clearTimeout(hoverTimer);clearTimeout(leaveTimer);
    if(owner&&owner!==t)owner.setAttribute('aria-expanded','false');
    owner=t;pinned=pin;
    pop.innerHTML=snapshot(entry);pop.setAttribute('aria-label',entry.r.player+' player snapshot');
    pop.hidden=false;pop.scrollTop=0;t.setAttribute('aria-expanded','true');place();
    if(pin)pop.focus({preventScroll:true});
    playerPage(entry,t.dataset.pc);
    fillRecord(entry,t.dataset.pc);
  }
  function readout(mark){
    const chart=mark.closest('.pc-chart'),out=chart?.querySelector('.pc-readout');
    if(!out)return;
    chart.querySelectorAll('.pc-on').forEach(x=>x.classList.remove('pc-on'));
    mark.classList.add('pc-on');out.textContent=mark.dataset.readout;
  }
  function reset(chart){
    const out=chart.querySelector('.pc-readout');
    chart.querySelectorAll('.pc-on').forEach(x=>x.classList.remove('pc-on'));
    if(out?.dataset.default)out.textContent=out.dataset.default;
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

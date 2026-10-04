/* Shared, descriptive player context. Never participates in pick selection. */
(function(global){
  'use strict';
  const finite=Number.isFinite;
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=(v,d=2)=>finite(v)?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
  const date=v=>/^\d{4}-\d{2}-\d{2}$/.test(v||'')?new Date(v+'T12:00:00Z').toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}):'';
  function tile(label,value,unit='',detail='',featured=false){
    return `<div class="pc-stat${featured?' pc-stat-primary':''}"><dt>${esc(label)}</dt><dd>${esc(value)}${unit?` <span>${esc(unit)}</span>`:''}${detail?`<span class="pc-detail">${esc(detail)}</span>`:''}</dd></div>`;
  }
  function inputsHTML(inputs){
    return `<dl class="pc-inputs">${inputs.filter(i=>i&&finite(i.value)).map(i=>`<div><dt>${esc(i.label)} <span class="pc-kind">${i.used?'Model input':'Context'}</span></dt><dd>${number(i.value,i.unit==='days'?0:2)}${i.unit?' '+esc(i.unit):''}${i.detail?`<span class="pc-detail">${esc(i.detail)}</span>`:''}</dd></div>`).join('')}</dl>`;
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
      return {schema_version:1,source:'NFL saved forecast trace',sample_games:sample.length,sample_label:'recent appearances',stat_label:{pass_yds:'yd',pass_attempts:'att',pass_completions:'cmp'}[market],
        workload_label:'Attempts / game',workload_unit:'att',recent:sample.length?[{games:sample.length,mean:mean(stat),workload:mean('attempts')}]:[],inputs,
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
    if(workload&&finite(workload.value))stats+=tile(workload.label,number(workload.value),workload.unit,'Prior-adjusted estimate');
    else if(first&&finite(first.workload))stats+=tile(c.workload_label,number(first.workload),c.workload_unit,'Observed average');
    const recentHTML=recent.length?`<div class="pc-recent"><h4>Observed recent form</h4><table><thead><tr><th scope="col">Sample</th><th scope="col">${esc(c.stat_label)} / ${c.sample_label==='starts'?'start':'game'}</th><th scope="col">${esc(c.workload_label)}</th></tr></thead><tbody>${recent.map(w=>`<tr><th scope="row">Last ${w.games}</th><td>${number(w.mean)}</td><td>${number(w.workload)} <span>${esc(c.workload_unit)}</span></td></tr>`).join('')}</tbody></table></div>`:'';
    const sample=finite(c.sample_games)?`${c.sample_games} ${c.sample_label||'games'} in history`:'Saved forecast';
    return `<section class="player-context" aria-label="Player form and model inputs"><div class="pc-heading"><h3>Player context</h3><span>${esc(options.saved?'Saved forecast':sample)}</span></div>${stats?`<dl class="pc-stats">${stats}</dl>`:''}<details class="pc-more"><summary>Recent form &amp; model inputs</summary>${recentHTML}${inputs.length?`<h4>Inside the forecast</h4>${inputsHTML(inputs)}<p class="pc-detail">“Model input” identifies a field used by the selected forecast model. “Context” provides additional background.</p>`:''}<p class="pc-note">${esc(c.note)}</p></details><p class="pc-source">${esc(c.source)}${date(c.through)?' · '+esc(c.through_label||'Last appearance')+' '+esc(date(c.through)):''}${options.saved?' · Preserved with this forecast':''}</p></section>`;
  }
  function season(c,season){
    if(!c)return '';
    const rows=c.group==='pitching'?[
      ['Innings',c.innings,'IP'],['Starts',c.starts,''],['Strikeouts',c.strikeouts,'K'],['K / 9',number(c.k_per_nine),''],['ERA',number(c.era),'']]:[
      ['Plate appearances',c.pa,'PA'],['Average',c.avg,'AVG'],['OPS',c.ops,''],['Hits',c.hits,''],['Home runs',c.home_runs,'HR'],['Total bases',c.total_bases,''],['Runs batted in',c.rbis,'RBI']];
    return `<details class="player-context pc-season"><summary>${esc(season)} regular-season totals</summary><dl class="pc-stats">${rows.filter(([,v])=>v!==null&&v!==undefined).map(([label,value,unit])=>tile(label,String(value),unit)).join('')}</dl><p class="pc-note">Descriptive season totals. Postseason results are excluded; these are not game projections.</p></details>`;
  }
  const api={render,season};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  global.FVPlayerContext=api;
})(typeof window!=='undefined'?window:globalThis);

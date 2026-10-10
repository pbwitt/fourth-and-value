/* Descriptive player stats for saved bets, using the same public sources as
   the sport boards. No ledger data is written or sent to a stats provider. */
(function(global){
  'use strict';
  const node=typeof module==='object'&&module.exports;
  const live=node?require('./live-stats.js'):global.FVLiveStats;
  const context=node?require('../assets/player-context.js'):global.FVPlayerContext;
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const person=value=>String(value??'').normalize('NFKD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');
  function marketKey(league,market){
    const spec=live.marketSpec(league,market);
    return spec?.stats?JSON.stringify([!!spec.pitcher,[...spec.stats].sort()]):String(market||'').toLowerCase();
  }
  // NFL already publishes one saved context per event/player/market, without
  // the thousands of repeated book quotes on the full props board.
  function rowsFrom(data,league){
    if(league!=='NFL')return Array.isArray(data?.rows)?data.rows:[];
    return Object.entries(data?.groups||{}).flatMap(([key,model_diagnostics])=>{
      try{const [,player,market]=JSON.parse(key);return [{player,market,model_diagnostics}];}catch{return [];}
    });
  }
  function observed(c){
    if(!c||!(c.games?.length||c.recent?.some(w=>Number.isFinite(w.mean))||c.trend?.rows?.length))return null;
    // Never transfer another event's forecast, matchup, probabilities or model
    // evidence to a saved ticket. These are the latest published observed stats.
    return {schema_version:1,source:c.source,through:c.through,sample_games:c.sample_games,sample_label:c.sample_label,
      stat_label:c.stat_label,workload_label:c.workload_label,workload_unit:c.workload_unit,
      recent:c.recent||[],games:c.games||[],game_columns:c.game_columns||[],game_focus:c.game_focus,
      trend:c.trend?{label:c.trend.label,note:'Recent completed appearances.',rows:c.trend.rows.map(([date,value,,opponent,workload])=>[date,value,null,opponent,workload])}:null};
  }
  function findContext(bet,data){
    const league=String(bet.league||'').toUpperCase(),key=marketKey(league,bet.market_type);
    const candidates=rowsFrom(data,league).filter(r=>person(r.player)===person(bet.player)&&marketKey(league,r.market_std||r.market)===key)
      .map(r=>observed(context.context(r,league))).filter(Boolean);
    candidates.sort((a,b)=>String(b.through||'').localeCompare(String(a.through||'')));
    return candidates[0]||null;
  }
  function nhlContext(market,games){
    const stat=live.marketSpec('NHL',market)?.stats?.[0];
    const field={sog:'shots',goals:'goals',assists:'assists',points:'points'}[stat];
    if(!field)return null;
    const valid=games.filter(g=>/^\d{4}-\d{2}-\d{2}$/.test(g.date)&&Number.isFinite(g[field])).sort((a,b)=>b.date.localeCompare(a.date));
    if(!valid.length)return null;
    const mean=(rows,key)=>{const nums=rows.map(r=>r[key]).filter(Number.isFinite);return nums.length?nums.reduce((a,b)=>a+b,0)/nums.length:null;};
    const label=field==='shots'?'SOG':field;
    return {schema_version:1,source:'NHL completed-game logs',through:valid[0].date,sample_games:valid.length,sample_label:'appearances',
      stat_label:label,workload_label:'Ice time',workload_unit:'min',
      recent:[...new Set([Math.min(5,valid.length),valid.length])].map(n=>({games:n,mean:mean(valid.slice(0,n),field),workload:mean(valid.slice(0,n),'minutes')})),
      games:valid.slice(0,5),game_columns:[['date','Date'],['opp','Opp'],['toi','TOI'],['shots','SOG'],['goals','G'],['assists','A'],['points','P']],game_focus:field,
      trend:{label,note:'Recent completed appearances.',rows:valid.slice().reverse().map(g=>[g.date,g[field],null,g.opp,g.minutes])}};
  }
  const api={person,marketKey,rowsFrom,observed,findContext,nhlContext};
  if(node){module.exports=api;return;}

  const requests=new Map(),names=new Map();
  function read(url,json=true){
    if(!requests.has(url))requests.set(url,(async()=>{
      const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
      try{const response=await fetch(url,{signal:controller.signal});if(!response.ok)throw Error('Stats unavailable');return await (json?response.json():response.text());}
      finally{clearTimeout(timer);}
    })().catch(error=>{requests.delete(url);throw error;}));
    return requests.get(url);
  }
  async function load(bet){
    const league=String(bet.league||'').toUpperCase();
    if(league==='NHL'){
      // The compact player pages retain recent logs even without current props.
      // This also avoids downloading the full, repeated NHL quote board.
      const index=await read('/nhl/players/players.json');
      const matches=Object.entries(index.players||{}).filter(([,p])=>person(p.name)===person(bet.player));
      if(matches.length!==1)return null;
      const html=await read('/nhl/players/'+encodeURIComponent(matches[0][0])+'/',false);
      const doc=new DOMParser().parseFromString(html,'text/html');
      const games=[...doc.querySelectorAll('#recent tbody tr')].map(tr=>{
        const cells=[...tr.cells].map(td=>td.textContent.trim());
        const number=v=>v!==''&&v!=null&&Number.isFinite(Number(v))?Number(v):null;
        const time=/^(\d+):(\d{2})$/.exec(cells[2]||'');
        return {date:cells[0],opp:cells[1],toi:cells[2],minutes:time?Number(time[1])+Number(time[2])/60:null,
          shots:number(cells[3]),goals:number(cells[4]),assists:number(cells[5]),points:number(cells[6])};
      });
      return nhlContext(bet.market_type,games);
    }
    const url={NFL:'/props/model-context.json',MLB:'/mlb/data/latest.json',NBA:'/nba/data/latest.json'}[league];
    return url?findContext(bet,await read(url)):null;
  }
  function name(bet){
    if(!bet.player)return 'Game market';
    const league=String(bet.league||'').toUpperCase();
    if(!['NHL','MLB','NFL','NBA'].includes(league))return esc(bet.player);
    const signature=JSON.stringify([bet.league,bet.player,bet.market_type,bet.side,bet.line,bet.game_date,bet.team_away,bet.team_home]);
    let cached=names.get(bet.id);
    if(!cached||cached.signature!==signature){
      const copy={...bet},spec=live.marketSpec(league,bet.market_type);
      const row={player:bet.player,sport:league,market:bet.market_type,market_label:spec?.label||bet.market_type,
        game:[bet.game_date,[bet.team_away,bet.team_home].filter(Boolean).join(' @ ')].filter(Boolean).join(' · '),
        side:bet.side,line:bet.line==null||bet.line===''?null:Number(bet.line)};
      const options={statsOnly:true,share:false,notice:'Latest published player stats. These can include games played after this bet.',load:()=>load(copy)};
      cached={signature,row,options};names.set(bet.id,cached);
    }
    return context.name(cached.row,league,cached.options);
  }
  global.FVTrackerPlayerStats={name};
})(typeof window==='undefined'?globalThis:window);

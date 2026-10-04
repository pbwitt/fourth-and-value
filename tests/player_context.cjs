const assert=require('node:assert/strict');
const {render,season}=require('../docs/assets/player-context.js');
const row={player:'Test Player',market:'pitcher_strikeouts',model_mean:5.7,player_context:{schema_version:1,
  source:'MLB completed-game box scores',through:'2026-09-30',sample_games:12,sample_label:'starts',stat_label:'K',
  workload_label:'Innings / start',workload_unit:'IP',recent:[{games:5,mean:0,workload:5.5},{games:10,mean:null,workload:null}],
  inputs:[{label:'Pitcher K rate',value:25,unit:'%',used:true},{label:'Recent innings / start',value:5.4,unit:'IP',used:false}],note:'IP averages are decimal innings.'}};
const html=render(row,'MLB');
assert.match(html,/5\.70/);assert.match(html,/0\.00/);assert.match(html,/5\.50/);assert.match(html,/25\.00 %/);
assert.match(html,/Model input/);assert.match(html,/>Context</);assert.match(html,/Sep 30, 2026/);
assert.doesNotMatch(html,/NaN|undefined|null/);
assert.equal(render({...row,model_withheld:'No forecast'},'MLB'),'');
assert.equal(render({...row,player:''},'MLB'),'');
assert.doesNotMatch(render({...row,model_mean:null},'MLB'),/Model projection/);
assert.match(render(row,'MLB',{saved:true}),/Preserved with this forecast/);
const malicious=render({...row,player_context:{...row.player_context,source:'<img src=x onerror=alert(1)>',stat_label:'<script>bad</script>'}},'MLB');
assert.doesNotMatch(malicious,/<img|<script>/);assert.match(malicious,/&lt;img/);
const old=render({player:'Player',market:'player_shots_on_goal',model_inputs:{projected_toi:18,history_games:30,last_game:'2026-09-30',opportunity_means:[3,1,1,2]},projected_mean:3},'NHL');
assert.match(old,/18\.00/);assert.match(old,/10\.00/);assert.match(old,/not recorded/);assert.doesNotMatch(old,/Observed recent form/);
assert.equal(render({player:'Player',market:'player_shots_on_goal'},'NHL'),'');
const nfl=render({player:'Player',market_std:'pass_yds',mu:245,projection_diagnostics:JSON.stringify({attempts:32,completion_rate:.65,yards_per_completion:11,recent_mean_weight:.2,yards_per_completion_recent_weight:.15,current_sample:[{attempts:30,passing_yards:220}]})},'NFL');
assert.match(nfl,/20\.00 %/);assert.match(nfl,/65\.00 %/);assert.match(nfl,/220\.00/);
assert.match(season({group:'pitching',innings:'5.2',starts:0,strikeouts:0,k_per_nine:null,era:null},2026),/5\.2/);
assert.doesNotMatch(season({group:'pitching',innings:'5.2',starts:0,strikeouts:0,k_per_nine:null,era:null},2026),/NaN|undefined|null/);
console.log('PASS: player context preserves missing values, zeros, units, input labels, archived context and escaped content.');

const assert=require('node:assert/strict');
const {render,season,name,snapshot}=require('../docs/assets/player-context.js');
const row={player:'Test Player',market:'pitcher_strikeouts',model_mean:5.7,player_context:{schema_version:1,
  source:'MLB completed-game box scores',through:'2026-09-30',sample_games:12,sample_label:'starts',stat_label:'K',
  workload_label:'Innings / start',workload_unit:'IP',recent:[{games:5,mean:0,workload:5.5},{games:10,mean:null,workload:null}],
  inputs:[{label:'Pitcher K rate',value:25,unit:'%',used:true},{label:'Recent innings / start',value:5.4,unit:'IP',used:false}],note:'IP averages are decimal innings.'}};
const html=render(row,'MLB');
assert.match(html,/5\.70/);assert.match(html,/0\.00/);assert.match(html,/5⅔ <span>IP/,'innings in thirds');assert.match(html,/25\.00 %/);
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
assert.match(season({group:'pitching',innings:'5.2',starts:0,strikeouts:0,k_per_nine:null,era:null},2026),/5⅔/,'box-score 5.2 is 5⅔');
assert.doesNotMatch(season({group:'pitching',innings:'5.2',starts:0,strikeouts:0,k_per_nine:null,era:null},2026),/NaN|undefined|null/);
// Name pop-up: the trigger only appears with something to show, and every value is escaped.
assert.equal(name({player:'A <b>B</b>',market:'batter_hits'},'MLB'),'A &lt;b&gt;B&lt;/b&gt;','no context, plain escaped text');
assert.equal(name({...row,model_withheld:'x'},'MLB'),'Test Player');
const trigger=name({...row,player:'O\'Neil <i>'},'MLB');
assert.match(trigger,/^<button type="button" class="pc-name" data-pc="pc\d+" aria-haspopup="dialog" aria-expanded="false"/);
assert.match(trigger,/O&#39;Neil &lt;i&gt;/);
assert.match(name({player:'Batter',market:'batter_hits'},'MLB',{season:{c:{group:'hitting',pa:10},season:2026}}),/pc-name/,'season totals alone open a snapshot');
const games=[{date:'2026-09-28',opp:'@ BOS',ip:'5⅔',pitches:98,k:8,bb:1,er:2},{date:'2026-09-22',opp:'<img src=x>',ip:'6',pitches:null,k:0,bb:0,er:0}];
const pop=snapshot({...row,game:'NYY @ BOS',player_context:{...row.player_context,recent:[{games:5,mean:5.8,workload:5.67,pitches:91.6},{games:10,mean:6.1,workload:5.4,pitches:90}],
  games,game_columns:[['date','Date'],['opp','Opp'],['ip','IP'],['pitches','Pitches'],['k','K'],['bb','BB'],['er','ER']],game_focus:'k'}},'MLB',
  {season:{c:{group:'pitching',innings:'160.2',starts:28,strikeouts:172,era:3.42},season:2026}});
assert.match(pop,/Recent games/);assert.match(pop,/<th scope="row">Sep 28<\/th>/);assert.match(pop,/<td class="pc-focus">8<\/td>/);
assert.match(pop,/<h4>Last 5 starts<\/h4>/);assert.match(pop,/Pitches \/ start<\/dt><dd>92/);assert.match(pop,/IP \/ start<\/dt><dd>5⅔</);assert.match(pop,/K \/ start<\/dt><dd>5\.8</);
assert.match(pop,/last 5 <strong>5\.8<\/strong> · last 10 <strong>6\.1<\/strong>/);
assert.match(pop,/28 starts · 160⅔ IP · 172 K · 3\.42 ERA/);assert.match(pop,/Pitcher K rate<\/dt><dd>25%<\/dd>/,'model inputs first, untagged');
assert.match(pop,/5⅓ IP <span class="pc-kind">Context only/);assert.match(pop,/<td class="pc-text">@ BOS/);
assert.match(pop,/&lt;img src=x&gt;/);assert.doesNotMatch(pop,/<img|NaN|undefined|null/);assert.match(pop,/<td>—<\/td>/,'missing pitches stay missing');
const hockey=snapshot({player:'Skater',market:'player_shots_on_goal',projected_mean:3.14,player_context:{schema_version:1,source:'NHL completed-game logs',through:'2026-04-16',
  sample_label:'appearances',stat_label:'SOG',workload_label:'Ice time',workload_unit:'min',recent:[{games:5,mean:3.4,workload:21.2}],
  inputs:[{label:'Projected ice time',value:20.4,unit:'min',used:true}],games:[{date:'2026-04-16',opp:'vs CHI',toi:'21:05',shots:4,goals:1,assists:0,points:1}],
  game_columns:[['date','Date'],['opp','Opp'],['toi','TOI'],['shots','SOG'],['goals','G'],['assists','A'],['points','P']],game_focus:'shots'}},'NHL');
assert.match(hockey,/TOI \/ game<\/dt><dd>21:12/);assert.match(hockey,/Model projection · experimental<\/dt><dd>3\.1 <span>SOG/);assert.match(hockey,/20:24<\/dd>/);assert.match(hockey,/3\.1 <span>SOG/);assert.match(hockey,/Apr 16, 2026/);
const qb=snapshot({player:'QB',market_std:'pass_yds',mu:245,projection_diagnostics:JSON.stringify({attempts:32,completion_rate:.65,yards_per_completion:11,
  current_sample:[{season:2026,week:3,attempts:30,completions:20,passing_yards:220},{season:2026,week:4,attempts:36,completions:25,passing_yards:301}]})},'NFL');
assert.match(qb,/<th scope="row">Wk 4<\/th><td>36<\/td><td>25<\/td><td class="pc-focus">301<\/td>/,'newest week first');
assert.match(qb,/65%/);
console.log('PASS: player context preserves missing values, zeros, units, input labels, archived context and escaped content; name pop-up snapshots.');

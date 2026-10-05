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
const legacy=snapshot({player:'Skater',market:'player_goals',projected_mean:.124,model_inputs:{projected_toi:15.2,history_games:30,last_game:'2026-04-16',opportunity_means:[2,.15,.2,.35]}},'NHL');
assert.match(legacy,/0\.12 <span>goals/,'two decimals below one');assert.doesNotMatch(legacy,/Context only/,'unknown usage is not called context');assert.match(legacy,/15:12/);
// How the model works: tabs, charts, build-up, odds, opponent, blend and limits.
const explained={...row,player:'Ace',line:5.5,side:'Over',model_probability:.54,model_push_probability:0,model_conditional_probability:.54,consensus_probability:.5,book_probability:.476,
  player_context:{...row.player_context,
    trend:{label:'K',note:'Every start counts equally.',rows:[['2026-09-01',4,null,'<b>BOS</b>',90],['2026-09-07',7,null,'@ TOR',98],['2026-09-13',6,.5,'vs NYY',95]]},
    build:{steps:[{label:'Batters faced per start',value:23.4,unit:''},{label:'Strikeout rate',value:25.1,unit:'%',op:'×'},{label:'Opponent adjustment',value:1.02,unit:'×',op:'×'},{label:'Simple estimate',value:5.99,unit:'K',op:'='}],note:'Simple.'},
    distribution:{start:2,p:[.1,.2,.2,.2,.15,.1,.05],low:true,high:true},
    blend:[{label:'Strikeout rate',own:.78,detail:'352 batters faced'}],
    opponent:{team:'BOS',label:'Opposing lineup',items:[{label:'Strikeout rate',value:23.9,unit:'%',league:22.4,rank:'7th highest of 30',used:true}]},
    missing:['Weather and umpire']}};
const full=snapshot(explained,'MLB');
assert.match(full,/role="tablist"/);assert.match(full,/data-tab="form"[^>]*aria-selected="true"/);assert.match(full,/id="pc-panel-model"[^>]*hidden/);
assert.match(full,/2 of 3 cleared Over 5\.5 · faded games count less/);
assert.match(full,/data-readout="Sep 1 · &lt;b&gt;BOS&lt;\/b&gt; · 4 K · 90 pitches"/,'readouts are escaped');assert.doesNotMatch(full,/<b>BOS/);
assert.match(full,/Opponent adjustment<\/span><strong>1\.02×/);assert.match(full,/Simple estimate<\/span><strong>6\.0 K/);
assert.match(full,/Over 5\.5: 54% · Under: 46%/,'odds come from the row, not the trimmed bars');
assert.match(full,/≤2 K: 10%/);assert.match(full,/8\+ K: 5%/);
assert.match(full,/Model 54%.*Market 50%.*Break-even 47\.6%/);assert.match(full,/Model vs\. break-even: \+6\.4 points/);
assert.match(full,/78% his games · 22% average/);assert.match(full,/Used by the model/);assert.match(full,/7th highest of 30/);
assert.match(full,/What the model doesn’t know.*Weather and umpire/);
assert.match(full,/data-tab="record"/,'MLB has a published track record');
assert.doesNotMatch(full,/NaN|undefined|null/);
const qbFull=snapshot({player:'QB',market_std:'pass_yds',mu:233.3,line:239.5,side:'Over',model_prob:.45,push_prob:0,prob_devig:.5,mkt_prob:.52,projection_diagnostics:JSON.stringify({attempts:32,completion_rate:.65,yards_per_completion:11,recent_mean_weight:.43,yards_per_completion_recent_weight:.3,
  current_sample:[{season:2026,week:1,attempts:30,completions:20,passing_yards:220},{season:2026,week:2,attempts:36,completions:25,passing_yards:301}],
  mean_stages:{before_adjustments:228.8,after_defense:248.2,after_venue:233.3,final:233.3}})},'NFL');
assert.match(qbFull,/Opposing pass defense<\/span><strong>1\.08×/);assert.match(qbFull,/Road game<\/span><strong>0\.94×/);assert.match(qbFull,/Projection<\/span><strong>233\.3 pass yds/);
assert.match(qbFull,/43% this season · 57% career/);assert.match(qbFull,/Adjustment to the projection<\/dt><dd>1\.08×<span class="pc-detail">Easier than average/);
assert.doesNotMatch(qbFull,/Range of outcomes/,'older traces without a spread draw no curve');assert.doesNotMatch(qbFull,/NaN|undefined|null/);
// NFL rushing and receiving: the trace's parts, matchup, venue and the forecast's bell curve.
const back={version:'nfl-projection-trace-1',family:'rush',carries:16.2,yards_per_carry:4.4,recent_mean_weight:.5,sigma:30,home:true,
  current_sample:[{season:2026,week:1,opponent_team:'MIA',carries:15,rushing_yards:61},{season:2026,week:2,opponent_team:'NYJ',carries:18,rushing_yards:92},
    {season:2026,week:3,opponent_team:'NE',carries:12,rushing_yards:40},{season:2026,week:4,opponent_team:'KC',carries:20,rushing_yards:111}],
  mean_stages:{before_adjustments:71.28,after_defense:68.1,after_venue:72.2,final:72.2},
  opponent:{team:'WAS',kind:'rush',rating:1.15,rank:6,of:32,allowed:98.4,league:112.3,games:4,season:2026}};
const rush=snapshot({player:'Back',market_std:'rush_yds',name:'under',point:70.5,mu:72.2,model_prob:.47,push_prob:0,prob_devig:.5,mkt_prob:.52,projection_diagnostics:JSON.stringify(back)},'NFL');
assert.match(rush,/2 of 4 cleared Under 70\.5/,'the row’s side and point drive the trend');
assert.match(rush,/<th scope="row">Wk 4<\/th><td class="pc-text">KC<\/td><td>20<\/td><td class="pc-focus">111<\/td>/,'newest week first, with the opponent');
assert.match(rush,/Rush yds \/ game<\/dt><dd>76</);assert.match(rush,/Carries \/ game<\/dt><dd>16\.3</);
assert.match(rush,/Expected carries<\/span><strong>16\.2<.*Yards per carry<\/span><strong>4\.4<.*Before matchup<\/span><strong>71\.3 rush yds<.*Opposing run defense<\/span><strong>0\.96×<.*Home game<\/span><strong>1\.06×<.*Projection<\/span><strong>72\.2 rush yds/s);
assert.match(rush,/Opposing run defense · WAS/);assert.match(rush,/Rushing yards allowed \/ game<\/dt><dd>98\.4<span class="pc-detail">6th toughest of 32 · 2026 season, 4 games/);
assert.match(rush,/League 112\.3/);assert.match(rush,/Tougher than average/);assert.match(rush,/50% this season · 50% career/);
assert.match(rush,/Under 70\.5: 47\.7% on this curve · 47% after calibration/,'curve chance from the Normal, published chance beside it');
assert.match(rush,/56–70 rush yds: 18\.9%/);assert.match(rush,/≤10 rush yds: 2%/,'negative totals stay in the open first bin');assert.match(rush,/161\+ rush yds: 0\.2%/);
assert.match(rush,/spread of ±30 rush yds/);assert.match(rush,/Offensive line injuries/);assert.doesNotMatch(rush,/NaN|undefined|null|≤−/);
const total=[...rush.matchAll(/data-readout="[^"]* rush yds: ([\d.]+)%"/g)].reduce((a,m)=>a+Number(m[1]),0);
assert(Math.abs(total-100)<.5,`the bins hold the whole curve (${total})`);
const catcher={...back,family:'receive',targets:7.1,catch_rate:.66,yards_per_reception:11.8,sigma:2.1,home:false,opponent:{team:'BUF',kind:'pass',rating:.8,rank:25,of:32},
  current_sample:[{season:2026,week:4,opponent_team:'MIA',targets:6,receptions:4,receiving_yards:52}],mean_stages:{before_adjustments:4.69,after_defense:4.69,after_venue:4.69,final:4.69}};
const catches=snapshot({player:'WR',market_std:'receptions',name:'under',point:5,mu:4.69,model_prob:.52,push_prob:.17,prob_devig:.5,mkt_prob:.55,projection_diagnostics:JSON.stringify(catcher)},'NFL');
assert.match(catches,/Expected targets<\/span><strong>7\.1<.*Catch rate<\/span><strong>66%<.*Before matchup<\/span><strong>4\.7 receptions/s);
assert.doesNotMatch(catches,/Opposing pass defense<\/span>/,'a neutral factor is not a step');
assert.match(catches,/Pass defense rating<\/dt><dd>0\.80<span class="pc-detail">25th toughest of 32</,'older traces fall back to the rating');
assert.match(catches,/0 receptions: 2\.3%/,'zero holds the curve below zero');
const caught=[...catches.matchAll(/data-readout="[^"]* receptions: ([\d.]+)%"/g)].reduce((a,m)=>a+Number(m[1]),0);
assert(Math.abs(caught-100)<.5,`count bins hold the whole curve (${caught})`);assert.match(catches,/11\+ receptions: 0\.3%/);
assert.match(catches,/<path class="pc-push"[^>]*\/>(?:<path class="pc-miss"[^>]*\/>){6}<line class="pc-rule" x1="(\d+\.\d)"/,'whole-number line: a push bin, then the losing bins');
assert.match(catches,/Under 5: 57% on this curve · 52% after calibration · Push: 18\.6%/,'matches market_math: Φ(4.5) ÷ (1 − push)');
assert.match(catches,/Model 52%.*Break-even 55%/,'NFL chances are already conditional on no push');
const far=snapshot({player:'WR',market_std:'recv_yds',name:'over',point:160.5,mu:55.1,model_prob:.1,push_prob:0,mkt_prob:.2,
  projection_diagnostics:JSON.stringify({...catcher,sigma:28,mean_stages:{before_adjustments:55.3,after_defense:55.3,after_venue:55.1,final:55.1}})},'NFL');
assert.doesNotMatch(snapshot({player:'WR',market_std:'recv_yds',name:'over',point:60.5,mu:55,model_prob:.4,mkt_prob:.5,projection_diagnostics:JSON.stringify({...catcher,sigma:1e6})},'NFL'),/Range of outcomes/,'an absurd spread draws no curve');
assert.match(far,/146–160 rec yds: 0\.1%/);assert.match(far,/161\+ rec yds: &lt;0\.1%/);assert.match(far,/Over 160\.5: &lt;0\.1% on this curve · 10% after calibration/,'a far line still gets its own edge');
const nba=snapshot({player:'Guard',market:'player_points',line:21.5,side:'Over',baseline_probability:.556,fair_probability:.5,book_probability:.52,player_context:{schema_version:1,source:'NBA',stat_label:'Points',
  distribution:{empirical:[18,19,20,22,23,24,25,26,27,30]},trend:{label:'PTS',note:'',rows:[['2026-03-01',22,null,'vs BOS',33],['2026-03-02',19,null,'@ BOS',30]]}}},'NBA');
assert.match(nba,/Past hit rate 55\.6%/);assert.match(nba,/7 over 21\.5 · 3 under/);assert.doesNotMatch(nba,/data-tab="record"/,'no model, no track record');
// Positions sit beside every name, with or without a snapshot, and in the snapshot header.
assert.equal(name({player:'Plain',player_position:'WR',market_std:'first_td'},'NFL'),'Plain <span class="pc-pos" title="Position">WR</span>');
assert.equal(name({player:'Held',player_position:'D',model_withheld:'x'},'NHL'),'Held <span class="pc-pos" title="Position">D</span>');
assert.match(name({...row,player_position:'SP'},'MLB'),/<\/button> <span class="pc-pos" title="Position">SP<\/span>$/);
assert.equal(name({player:'Odd',player_position:'<b>'},'NBA'),'Odd <span class="pc-pos" title="Position">&lt;b&gt;</span>','escaped');
assert.equal(name({player:'Blank',player_position:'  '},'NBA'),'Blank','blank positions are not shown');
assert.match(snapshot({...row,player_position:'SP',game:'NYY @ BOS'},'MLB'),/<strong>Test Player<\/strong><span>SP · NYY @ BOS<\/span>/);
assert.match(snapshot({...row,player_position:'SP'},'MLB'),/<strong>Test Player<\/strong><span>SP<\/span>/);
console.log('PASS: player context preserves missing values, zeros, units, input labels, archived context and escaped content; name pop-up snapshots.');

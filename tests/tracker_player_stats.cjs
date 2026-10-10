const assert=require('node:assert/strict');
const stats=require('../docs/tracking/player-stats.js');
const shared=require('../docs/assets/player-context.js');
const c={schema_version:1,source:'MLB completed-game box scores',through:'2026-10-07',stat_label:'K',sample_label:'starts',
  workload_label:'Innings',workload_unit:'IP',recent:[{games:5,mean:6,workload:5.5}],games:[{date:'2026-10-07',k:7}],
  game_columns:[['date','Date'],['k','K']],game_focus:'k',
  trend:{label:'K',note:'Weighted model inputs',rows:[['2026-10-07',7,.8,'@ NYY',5.5]]},
  inputs:[{label:'Expected innings',value:6}],build:{steps:[{label:'Projection',value:8}]},
  distribution:{start:0,p:[.1,.9]},versus:{team:'NYY',values:[7]},opponent:{team:'NYY'},blend:[],missing:[]};
const bet={league:'MLB',player:'J.T. Pitcher',market_type:'pitcher_strikeouts',side:'under',line:6,game_date:'2026-06-01'};
const source={player:'J. T. Pitcher',market:'pitcher_strikeouts',line:3.5,side:'Over',player_context:c,
  model_mean:8,model_probability:.8,model_withheld:true};
const result=stats.findContext(bet,{rows:[source]});
assert.equal(result.recent[0].mean,6);assert.equal(result.through,'2026-10-07');
for(const key of ['inputs','build','distribution','versus','opponent','blend','missing','model_mean','model_probability'])assert(!Object.hasOwn(result,key),key);
assert.equal(result.trend.rows[0][2],null,'model weights are not observed stats');
assert.deepEqual(bet,{league:'MLB',player:'J.T. Pitcher',market_type:'pitcher_strikeouts',side:'under',line:6,game_date:'2026-06-01'});
assert.equal(stats.findContext({...bet,player:'Other Pitcher'},{rows:[source]}),null);
assert.equal(stats.findContext({...bet,market_type:'batter_strikeouts'},{rows:[source]}),null,'batter and pitcher strikeouts must not mix');
assert.equal(stats.findContext({...bet,player:'J.T. Pitcher Jr.'},{rows:[source]}),null,'suffixes distinguish players');
assert.equal(stats.findContext(bet,{rows:[{...source,player_context:null}]}),null);
assert.equal(stats.findContext(bet,{rows:[{...source,player_context:{...c,through:'2026-10-05'}},source]}).through,'2026-10-07');
const html=shared.snapshot({...bet,market:bet.market_type,model_mean:99,model_probability:.9,book_probability:.5,player_context:result},'MLB',
  {statsOnly:true,share:false,notice:'Latest published player stats.'});
assert.match(html,/Recent games/);assert.match(html,/Latest published player stats/);
assert.doesNotMatch(html,/Model projection|How it works|Track record|Share player snapshot|Weighted model inputs|NaN|undefined/);
assert.equal(stats.marketKey('NHL','sog'),stats.marketKey('NHL','player_shots_on_goal'));
assert.equal(stats.marketKey('NBA','pts'),stats.marketKey('NBA','player_points'));
const nfl={groups:{'["game","Example Back","rush_yds"]':{projection:{family:'rush',carries:15,yards_per_carry:4,
  current_sample:[{season:2026,week:1,carries:15,rushing_yards:60}],mean_stages:{final:60}}},'bad JSON':{}}};
assert.equal(stats.rowsFrom(nfl,'NFL').length,1);
assert.equal(stats.findContext({league:'NFL',player:'Example Back',market_type:'rush_yds'},nfl).recent[0].mean,60);
const games=[{date:'2026-10-08',opp:'@ BOS',toi:'20:30',minutes:20.5,shots:4,goals:1,assists:0,points:1},
  {date:'2026-10-07',opp:'vs NYR',toi:'19:00',minutes:19,shots:2,goals:0,assists:0,points:0},
  {date:'2026-10-06',shots:null,goals:0},{date:'bad',shots:100}];
const nhl=stats.nhlContext('sog',games);
assert.equal(nhl.through,'2026-10-08');assert.equal(nhl.recent[0].mean,3);assert.equal(nhl.recent[0].workload,19.75);
assert.equal(nhl.games.length,2);assert.equal(nhl.trend.rows[0][0],'2026-10-07');assert.equal(nhl.game_focus,'shots');
assert.equal(stats.nhlContext('assists',games).recent[0].mean,0,'zero stats are real results');
assert.equal(stats.nhlContext('saves',games),null);assert.equal(stats.nhlContext('sog',[]),null);
console.log('PASS: tracker matches player and market, retains the saved pick, and displays only observed stats with honest dates.');

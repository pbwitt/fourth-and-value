// Bet Tracker performance panel on the real page with sample bets: win rate against
// break-even, ROI against winning bettors, running profit, by sport, filters, two widths.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..','docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
const bet=(id,game_date,league,status,odds,payout,extra={})=>({id,game_date,league,status,odds,stake_dollars:5,payout,team_away:'Away',team_home:'Home',
  market_type:'sog',side:'under',line:2.5,player:'Sample Player',book:'draftkings',created_at:game_date+'T12:00:00Z',timestamp:game_date+'T12:00:00Z',...extra});
const rows=[
  bet('1','2026-09-29','NHL','lost',-138,0),bet('2','2026-09-29','NHL','won',-140,8.57),bet('3','2026-09-30','MLB','lost',105,0),
  bet('4','2026-10-01','NFL','won',125,11.25,{market_type:'receptions',side:'over',line:1.5,player:'Receiver <b>X</b>'}),
  bet('5','2026-10-01','NFL','won',148,12.4),bet('6','2026-10-02','NHL','won',-105,9.76),bet('7','2026-10-03','MLB','won',-116,9.31),
  bet('8','2026-10-04','NFL','lost',175,0),bet('9','2026-10-04','NFL','pending',155,null)];

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  try{
    for(const width of [390,1280]){
      const p=await browser.newPage({viewport:{width,height:1100}}),errors=[];
      p.on('pageerror',error=>errors.push(error.message));
      await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
      await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
      await p.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'text/javascript',body:''}));
      await p.addInitScript(rows=>{
        try{localStorage.clear();}catch(e){}
        window.db={rows};const user={id:'test-owner',email:'reader@example.com'};
        const query={order:async()=>({data:db.rows,error:null})};
        window.supabase={createClient:()=>({
          auth:{getSession:async()=>({data:{session:{user}}}),onAuthStateChange:()=>{}},
          functions:{invoke:async()=>({data:{games:[]},error:null})},
          from:()=>({select:()=>query,insert:async()=>({error:null})})})};
      },rows);
      await p.goto(base+'/tracking/');
      await p.waitForSelector('#performance',{state:'visible'});
      const panel=p.locator('#performance');
      const text=(await panel.textContent()).replace(/\s+/g,' ');
      // 5 wins of 8 decided; ROI = (3.57+6.25+7.4+4.76+4.31-15)/40.
      assert.match(text,/8 settled bets in this view, 1 pending/);
      assert.match(text,/62\.5%\s?Bets won/);assert.match(text,/Break-even win rate at the prices you took/);
      assert.match(text,/\+28\.2%\s?ROI on \$40\.00 risked/);
      assert.equal(await panel.locator('[data-perf=path] .perf-dot').count(),8);
      assert.match(await panel.locator('[data-perf=scale]').textContent(),/You \+28\.2%/);
      assert.deepEqual((await panel.locator('.perf-sports tbody th').allTextContents()),['NFL','NHL','MLB']);
      assert.doesNotMatch(text,/luck|±|NaN|undefined/i);
      const readout=panel.locator('[data-perf=readout]');
      await panel.locator('.perf-hit').nth(3).hover();
      assert.match(await readout.textContent(),/Oct 1 · NFL · Receiver <b>X<\/b> receptions over 1\.5 · Won \+\$6\.25/,'escaped text, not markup');
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`overflow at ${width}`);
      await p.screenshot({path:`/tmp/fv-tracker-performance-${width}.png`,fullPage:true});
      // The panel follows the filters.
      await p.locator('#f-sport').selectOption('NFL');
      await p.waitForFunction(()=>/3 settled bets in this view/.test(document.getElementById('performance').textContent));
      assert.equal(await panel.locator('.perf-sports tbody tr').count(),1);
      assert.deepEqual(errors,[]);
      await p.close();
    }
    console.log('PASS: tracker performance panel renders from the shown bets, follows filters, escapes text and fits phone and desktop widths.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

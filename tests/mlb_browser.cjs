// NODE_PATH=/path/to/node_modules node tests/mlb_browser.cjs
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  try{
    const errors=[];
    for(const width of [390,768,1280,1440,1920]){
      const p=await browser.newPage({viewport:{width,height:1000}});p.on('pageerror',e=>errors.push(e.message));
      for(const route of ['/mlb/','/mlb/props/','/mlb/totals/','/mlb/top.html','/mlb/methods.html','/nba/','/']){
        await p.goto(base+route);if(route.startsWith('/mlb'))await p.waitForFunction(()=>!document.getElementById('feed-status').textContent.includes('Enable JavaScript'));
        assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`Overflow ${width} ${route}`);
        assert.equal(await p.locator('.mlb-sport').count(),1);
        if(!await p.locator('.fv-burger').isVisible()){
          const logo=await p.locator('.fv-logo').boundingBox(),links=await p.locator('.fv-links').boundingBox();
          assert(logo.x+logo.width<=links.x,`Nav logo overlaps links at ${width}`);
        }
      }
      if(await p.locator('.fv-burger').isVisible()){
        assert((await p.locator('.fv-burger span').first().boundingBox()).height>=2,'Mobile menu icon must remain visible');
        await p.locator('.fv-burger').click();
      }
      await p.locator('.mlb-sport button').click();await p.getByRole('link',{name:'MLB Overview',exact:true}).click();assert(p.url().endsWith('/mlb/'));
      if(width===390||width===1440)await p.screenshot({path:`/tmp/fv-mlb-${width}.png`,fullPage:true});
      await p.close();
    }
    // Exercise a populated props board even when the real slate has no props.
    const p=await browser.newPage({viewport:{width:390,height:900}});
    const now=new Date(),future=new Date(+now+3600e3).toISOString();
    const row={event_id:'g',game:'Away @ Home',commence_time:future,quoted_at:now.toISOString(),player:'Example Player',market:'pitcher_strikeouts',market_label:'Pitcher strikeouts',market_family:'props',game_type:'F',phase:'Wild Card',line:20.5,side:'Over',price:-110,book:'a',book_label:'Book A',book_probability:.5238,fair_probability:.5,consensus_probability:.5,paired_books:2,baseline_mean:null};
    let fixture={status:'ready',last_success_at:now.toISOString(),events:[],rows:[row,{...row,book:'b',book_label:'Book B',price:100,book_probability:.5}]};
    await p.route('**/mlb/data/latest.json',r=>r.fulfill({json:fixture}));
    await p.goto(base+'/mlb/props/');await p.waitForSelector('.prop-card');assert.equal(await p.locator('.prop-card').count(),1);
    assert((await p.locator('.prop-card').textContent()).includes('Book B'));
    await p.locator('#book').selectOption('a');assert((await p.locator('.prop-card').textContent()).includes('Book A'));
    await p.locator('#search').fill('missing');assert.equal(await p.locator('.prop-card').count(),0);
    await p.locator('#reset').click();assert.equal(await p.locator('.prop-card').count(),1);
    await p.locator('#phase').selectOption('R');assert.equal(await p.locator('.prop-card').count(),0);
    await p.locator('#phase').selectOption('postseason');assert.equal(await p.locator('.prop-card').count(),1);
    assert(p.url().includes('phase=postseason'));
    fixture={...fixture,status:'feed_error'};await p.reload();await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('failed'));assert.equal(await p.locator('.prop-card').count(),0);
    fixture={...fixture,status:'ready',last_success_at:new Date(+now-25*3600e3).toISOString()};await p.reload();await p.waitForFunction(()=>document.getElementById('feed-status').textContent.includes('needs a refresh'));assert.equal(await p.locator('.prop-card').count(),0);
    assert.deepEqual(errors,[]);console.log('PASS: MLB nav/layout at five widths, postseason/props filters, best-book selection, error/stale suppression.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});

const {chromium}=require(process.env.FV_PLAYWRIGHT||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.FV_CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});const page=await browser.newPage({viewport:{width:390,height:844}});
 const root=require('node:path').join(__dirname,'../docs');
 await page.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'application/javascript',body:''}));
 await page.route('https://fourthandvalue.com/**',route=>{
  const path=new URL(route.request().url()).pathname;
  if(path==='/tracking/bet-tracking.js')return route.fulfill({contentType:'application/javascript',body:`window.mode='editor';window.rows=[];window.supabaseClient={auth:{getUser:async()=>({data:{user:window.mode==='editor'?{app_metadata:{fv_editor:true}}:null}}),onAuthStateChange:fn=>{window.authChanged=fn;},signOut:async()=>{}},from:()=>({select(){return this},eq(){return this},order(){return this},limit:async()=>window.mode==='schema'?{error:{code:'PGRST205'}}:{data:window.rows}})};`});
  try{return route.fulfill({contentType:path.endsWith('.js')?'application/javascript':path.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(root+path)});}catch{return route.fulfill({status:404,body:'Not found'});}
 });
 await page.goto('https://fourthandvalue.com/editorial/diagnostics.html');
 await page.waitForFunction(()=>document.getElementById('message').textContent.includes('No report'));
 assert.equal(await page.locator('#report').isVisible(),false);
 await page.evaluate(()=>{window.rows=[{id:'1',observed_at:new Date().toISOString(),report:{phase:'finish',observed_at:new Date().toISOString(),status:'saved_not_verified',saved:1,expected:2,selection_count:1,run:{started_at:new Date().toISOString(),event:'schedule',stages:{}},data:[{sport:'NFL',prices:'stale',games:0,model:'unknown'}],articles:[{sport:'NFL',status:'waiting_for_data',title:'<img src=x onerror="window.hacked=1">',review:'not_recorded'}],recovery:{eligible:true,next_opportunity:new Date().toISOString()},unselected:[{sport:'NFL',reason:'Insufficient reporting'}]}}];});
 await page.locator('#refresh').click();await page.waitForFunction(()=>!document.getElementById('report').hidden);
 assert.equal(await page.locator('#articles img').count(),0);
 assert.ok((await page.locator('#articles').textContent()).includes('not recorded'));
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
 if(process.env.FV_SCREENSHOT)await page.screenshot({path:process.env.FV_SCREENSHOT,fullPage:true});
 await page.evaluate(()=>{window.mode='schema'});await page.locator('#refresh').click();await page.waitForFunction(()=>document.getElementById('message').textContent.includes('One-time'));
 assert.equal(await page.locator('#report').isVisible(),false);
 await page.evaluate(()=>{window.mode='signedout';window.authChanged();});await page.waitForFunction(()=>!document.getElementById('login').hidden);
 assert.equal(await page.locator('#dashboard').isVisible(),false);
 await browser.close();console.log('Dashboard passed: private access, no reports, setup error, safe rendering, mobile layout.');
})().catch(e=>{console.error(e);process.exit(1)});

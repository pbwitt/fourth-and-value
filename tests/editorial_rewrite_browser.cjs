const {chromium}=require(process.env.FV_PLAYWRIGHT||'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.FV_CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 for(const width of [390,1440]){
  const page=await browser.newPage({viewport:{width,height:900}});
  await page.route('https://**/*',route=>{
   const u=new URL(route.request().url());
   if(u.origin!=='https://fourthandvalue.com')return route.fulfill({body:'',contentType:'application/javascript'});
   if(u.pathname==='/tracking/bet-tracking.js')return route.fulfill({contentType:'application/javascript',body:`
    window.updates=[];window.dispatches=[];window.revision=0;window.failDispatch=false;
    window.row={id:'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',user_id:'owner',requires_review:false,status:'review',kind:'analysis',sport:'NFL',idea:'Analyze the Falcons matchup.\\n\\nChanges for the next draft:\\nOld feedback',title:'Existing draft title',body:'Existing draft body.',byline:'Fourth & Value',sources:'https://www.nfl.com/news/example',updated_at:'v0'};
    window.supabaseClient={auth:{getUser:async()=>({data:{user:{id:'owner',app_metadata:{fv_editor:true}}}}),onAuthStateChange:()=>{}},functions:{invoke:async(name,input)=>{window.dispatches.push(input.body);window.row={...window.row,write_now_requested_at:new Date().toISOString(),updated_at:'v'+(++window.revision)};if(window.failDispatch)return {error:{message:'Dispatch failed'}};return {data:{message:'Requested'}};}},from:()=>({change:null,select(){if(this.change){window.updates.push(this.change);window.row={...window.row,...this.change,updated_at:'v'+(++window.revision)};return Promise.resolve({data:[window.row]});}return this},update(x){this.change=x;return this},eq(){return this},order(){return this},limit:async()=>({data:[window.row]})})};`});
   try{return route.fulfill({body:fs.readFileSync(path.join(__dirname,'../docs',u.pathname)),contentType:u.pathname.endsWith('.js')?'application/javascript':u.pathname.endsWith('.css')?'text/css':'text/html'});}catch{return route.fulfill({status:404,body:'Not found'});}
  });
  await page.goto('https://fourthandvalue.com/editorial/inbox.html');await page.locator('.queue-item').click();
  assert.equal(await page.locator('#rewrite').isVisible(),true);
  await page.locator('#rewrite').click();
  assert.equal(await page.locator('#confirm-rewrite').isEnabled(),false);
  await page.locator('#rewrite-feedback').fill('Make the matchup comparison clearer.');
  await page.evaluate(()=>document.getElementById('publish-now').checked=true);
  await page.locator('#confirm-rewrite').click();
  await page.waitForFunction(()=>document.getElementById('action-message').textContent.includes('Rewrite requested'));
  const state=await page.evaluate(()=>({updates,dispatches,row}));
  assert.equal(state.dispatches.length,1);assert.equal(state.dispatches[0].publish_own,false);
  assert.equal(state.updates.length,2);assert.equal(state.updates[1].status,'submitted');
  assert.ok(state.updates[0].idea.includes('Make the matchup comparison clearer.'));
  assert.ok(!state.updates[0].idea.includes('Old feedback'));
  assert.equal(state.row.body,'Existing draft body.');
  assert.equal(await page.locator('#publish-now-option').isVisible(),false);
  assert.equal(await page.locator('#approve').isVisible(),false);
  assert.equal(await page.locator('#write-now').textContent(),'Rewrite queued');
  assert.equal(await page.locator('#write-now').isEnabled(),false);
  assert.equal(await page.locator('#save').isEnabled(),false);
  await page.evaluate(()=>document.getElementById('write-now').click());
  assert.equal(await page.evaluate(()=>dispatches.length),1);
  // Pending survives a page refresh/selection, and expiry exposes manual recovery.
  await page.locator('.queue-item').click();
  assert.ok((await page.locator('#action-message').textContent()).includes('Rewrite queued'));
  await page.evaluate(()=>{row.write_now_requested_at=new Date(Date.now()-16*60000).toISOString();row.updated_at='v'+(++revision);});
  await page.locator('#refresh').click();
  await page.waitForFunction(()=>document.getElementById('write-now').textContent==='Retry rewrite');
  assert.equal(await page.locator('#write-now').isEnabled(),true);
  // Completion replaces the request banner and exposes approval only on review.
  await page.evaluate(()=>{row.status='review';row.updated_at='v'+(++revision);row.body='Replacement draft body.';});
  await page.locator('#refresh').click();await page.locator('.queue-item').click();
  assert.equal(await page.locator('#body').inputValue(),'Replacement draft body.');
  assert.ok((await page.locator('#action-message').textContent()).includes('ready for review'));
  assert.equal(await page.locator('#approve').isVisible(),true);
  // A failed dispatch is immediately recoverable and preserves the current draft.
  await page.evaluate(()=>{window.failDispatch=true;row.research_error='Earlier attempt failed';});
  await page.locator('#rewrite').click();await page.locator('#rewrite-feedback').fill('Try another angle.');
  await page.locator('#confirm-rewrite').click();
  await page.waitForFunction(()=>document.getElementById('action-message').textContent.includes('could not start'));
  assert.equal(await page.locator('#write-now').textContent(),'Retry rewrite');
  assert.equal(await page.locator('#write-now').isEnabled(),true);
  assert.equal(await page.locator('#body').inputValue(),'Replacement draft body.');
  assert.equal(await page.evaluate(()=>row.research_error),null);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.close();
 }
 await browser.close();console.log('Rewrite UI passed on mobile and desktop: feedback, preserved current draft, one private dispatch, cleared old instructions, and no approval before completion.');
})().catch(e=>{console.error(e);process.exit(1)});

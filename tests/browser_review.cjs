// Run against the local docs server. No auth, tracking writes, or external API calls.
const playwright=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browserType=playwright[process.env.BROWSER || 'chromium'];
const assert=require('node:assert/strict');
(async()=>{
 const browser=await browserType.launch({executablePath:process.env.CHROME_PATH || undefined,headless:true});
 const failures=[];
 for(const width of [390,768,1024,1440]){
  const page=await browser.newPage({viewport:{width,height:900}});
  await page.route('https://**/*',route=>route.abort());
  page.on('pageerror',e=>failures.push(`${width}: ${e.message}`));
  for(const path of ['/','/nfl/','/props/','/props/top.html','/nfl/totals/','/methods.html','/tracking/','/blog/','/research/']){
   await page.goto('http://127.0.0.1:8010'+path);await page.waitForTimeout(250);
   if(path==='/tracking/')continue; // Its external auth SDK is deliberately blocked in this test.
   const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
   if(overflow)failures.push(`${width} ${path}: horizontal overflow`);
   if(['/','/nfl/','/props/'].includes(path))await page.screenshot({path:`/private/tmp/fv-${width}-${path==='/'?'home':path.split('/')[1]}.png`,fullPage:false});
  }
  await page.goto('http://127.0.0.1:8010/props/');
  await page.waitForSelector('#filters:not([hidden])');
  assert((await page.locator('.prop-card').count())<=24,'pagination bounds the rendered cards');
  await page.locator('#books').evaluate(el=>el.parentElement.open=true);
  await page.locator('#no-books').click();assert.equal(await page.locator('.prop-card').count(),0);
  await page.locator('#reset').click();
  await page.locator('#q').fill('no-such-player-92834');assert.equal(await page.locator('.prop-card').count(),0);
  assert(new URL(page.url()).searchParams.get('q')==='no-such-player-92834');
  await page.reload();assert.equal(await page.locator('#q').inputValue(),'no-such-player-92834');
  await page.locator('#reset').click();
  if(width<=1050)await page.locator('.fv-burger').click();
  const nfl=page.locator('.nfl-sport .fv-sport-toggle');await nfl.focus();await page.keyboard.press('Enter');
  assert.equal(await nfl.getAttribute('aria-expanded'),'true');await page.keyboard.press('Escape');
  assert.equal(await nfl.getAttribute('aria-expanded'),'false');
  await page.close();
 }
 await browser.close();
 // Expected network-blocked auth errors are reported separately.
 const real=failures.filter(f=>!f.includes("Cannot read properties of undefined (reading 'createClient')"));
 console.log(JSON.stringify({failures:real,blockedExternalAuthErrors:failures.length-real.length},null,2));
 assert.equal(real.length,0);
})().catch(e=>{console.error(e);process.exit(1);});

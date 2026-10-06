// Requires Playwright 1.55.1 and Chromium; use CHROME_PATH for a local Chrome install.
// Run after build_matchup_paper.py. Serves local docs, checks rendering, and prints the PDF.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../../docs');
const slug='player-matchups-and-home-field';
const server=http.createServer((req,res)=>{
 let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
 const file=path.resolve(root,'.'+name);
 if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
 fs.readFile(file,(error,data)=>{
  if(error){res.writeHead(404);return res.end();}
  res.setHeader('Content-Type',({'.html':'text/html','.css':'text/css','.js':'text/javascript','.json':'application/json','.svg':'image/svg+xml','.pdf':'application/pdf'})[path.extname(file)]||'application/octet-stream');res.end(data);
 });
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const base=`http://127.0.0.1:${server.address().port}`;
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const results={routes:[],pdf:`docs/research/${slug}.pdf`};
 try {
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  for(const width of [390,768,1440]){
   await page.setViewportSize({width,height:1000});
   for(const route of ['/research/',`/research/${slug}.html`]){
    const response=await page.goto(base+route);assert.equal(response.status(),200);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`Overflow: ${route} at ${width}`);
    results.routes.push({route,width,status:response.status()});
   }
   assert.equal(await page.locator('article table').count(),7);
   const bad=await page.locator('a[href^="#"]').evaluateAll(links=>links.filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash));
   assert.deepEqual(bad,[],'Broken contents/citation anchors');
   const text=await page.locator('main').innerText();assert(!text.includes('@@'));assert(text.includes('Against')||text.includes('1.025'));assert(text.includes('FV-2026-03'));
   if(width===390||width===1440){
    await page.screenshot({path:`/tmp/fv-matchup-paper-cover-${width}.png`});
    await page.screenshot({path:`/tmp/fv-matchup-paper-${width}.png`,fullPage:true});
   }
  }
  const data=await page.request.get(base+`/research/${slug}.json`);assert(data.ok());
  assert.equal((await data.json()).paper,'FV-2026-03');
  await page.emulateMedia({media:'print'});
  assert.equal(await page.locator('.fv-nav').isVisible(),false,'Site navigation must not enter the paper PDF');
  await page.setViewportSize({width:794,height:1123});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'Print horizontal overflow');
  await page.pdf({path:path.join(root,`research/${slug}.pdf`),format:'A4',preferCSSPageSize:true,
   printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',
   footerTemplate:'<div style="width:100%;text-align:center;font:8px Georgia;color:#666">Fourth &amp; Value · FV-2026-03 &nbsp; | &nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
  await page.screenshot({path:'/tmp/fv-matchup-paper-print.png',fullPage:true});
  assert.deepEqual(errors,[]);
  results.bytes=fs.statSync(path.join(root,`research/${slug}.pdf`)).size;
  fs.writeFileSync('/tmp/fv-matchup-paper-browser.json',JSON.stringify(results,null,2));
  console.log(JSON.stringify(results,null,2));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1});

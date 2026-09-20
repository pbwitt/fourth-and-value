// Render the deterministic canvas + narrated timeline with local Chrome.
// Requires Playwright. Usage: NODE_PATH=/path/to/node_modules node scripts/render_basics_video.cjs
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '../docs');
const output = path.join(root, 'videos/why-we-devig');
const server = http.createServer((req, res) => {
  const file = path.resolve(root, '.' + decodeURIComponent(req.url.split('?')[0]));
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    const ext = path.extname(file);
    res.setHeader('Content-Type', ({'.html':'text/html','.js':'text/javascript','.json':'application/json','.wav':'audio/wav'})[ext] || 'application/octet-stream');
    res.end(data);
  });
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({headless:true,
      executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      args:['--autoplay-policy=no-user-gesture-required','--disable-background-timer-throttling']});
    const page = await browser.newPage({viewport:{width:1080,height:1920},deviceScaleFactor:1});
    const errors=[];
    page.on('pageerror', err=>errors.push(err.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/videos/why-we-devig/render.html?silent=1`);
    await page.waitForFunction(()=>window.videoReady);
    await page.locator('canvas').screenshot({path:path.join(output,'poster.png')});
    const data = await page.evaluate(()=>window.renderVideo());
    if(errors.length) throw new Error(errors.join('\n'));
    fs.writeFileSync(path.join(output,'why-we-devig.mp4'),Buffer.from(data,'base64'));
    // Decode the exported file in Chrome as a container/playback sanity check.
    const verification = await page.evaluate(async()=>{
      const video=document.createElement('video'); video.preload='auto';
      video.src=window.exportedVideoURL;
      await new Promise((resolve,reject)=>{video.onloadedmetadata=resolve;video.onerror=()=>reject(new Error('Export cannot be decoded'));});
      return {width:video.videoWidth,height:video.videoHeight,duration:video.duration,bytes:window.exportedVideoBlob.size};
    });
    fs.writeFileSync(path.join(output,'render-info.json'),JSON.stringify(verification,null,2)+'\n');
    console.log('Export verified:',verification);
  } finally {if(browser) await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;server.close();});

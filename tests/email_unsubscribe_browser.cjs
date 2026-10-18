// Morning Card unsubscribe page: opening the link changes nothing, one click unsubscribes
// or resubscribes that token's row only, bad links and outages say so, phone and desktop.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..','docs');
const server=http.createServer((req,res)=>{
  let name=decodeURIComponent(req.url.split('?')[0]);if(name.endsWith('/'))name+='index.html';
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
  fs.readFile(file,(error,data)=>{if(error){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css'})[path.extname(file)]||'application/octet-stream');res.end(data);});
});
const TOKEN='3f2b8c1e-9d4a-4e6b-8f1a-2c5d7e9b0a14';

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  try{
    for(const width of [360,1280]){
      const p=await browser.newPage({viewport:{width,height:900}}),errors=[];
      p.on('pageerror',error=>errors.push(error.message));
      await p.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
      await p.route('**/nav.js*',route=>route.fulfill({contentType:'text/javascript',body:''}));
      await p.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({contentType:'text/javascript',body:''}));
      // A fake database with one row that survives reloads; the page reaches it only
      // through the two functions. Calls are counted per page load.
      await p.addInitScript(token=>{
        const saved=JSON.parse(sessionStorage.getItem('fake-db')||'null');
        window.db={rows:saved||{[token]:{list:'morning_card',subscribed:true}},calls:[],fail:false};
        const save=()=>sessionStorage.setItem('fake-db',JSON.stringify(db.rows));
        window.supabase={createClient:()=>({auth:{getSession:async()=>({data:{session:null}}),onAuthStateChange:()=>{}},
          rpc:async(name,args)=>{
            db.calls.push([name,args]);
            if(db.fail)return {data:null,error:{message:'offline'}};
            const row=db.rows[args.p_token];if(!row)return {data:[],error:null};
            if(name==='email_subscription_set'){row.subscribed=args.p_subscribed;save();}
            return {data:[{...row}],error:null};
          }})};
      },TOKEN);
      const status=()=>p.locator('#email-status').textContent();
      const visible=async()=>(await Promise.all(['email-unsubscribe','email-resubscribe','email-retry'].map(async id=>
        await p.locator('#'+id).isVisible()?id:null))).filter(Boolean);

      await p.goto(`${base}/email/unsubscribe.html?token=${TOKEN}`);
      await p.waitForFunction(()=>!/Checking/.test(document.getElementById('email-status').textContent));
      assert.equal(await status(),'You’re subscribed to the Morning Card email.');
      assert.deepEqual(await visible(),['email-unsubscribe']);
      assert.deepEqual(await p.evaluate(()=>db.calls.map(c=>c[0])),['email_subscription_get'],'opening the link changes nothing');
      assert.equal(await p.locator('h1').count(),1);
      assert.equal(await p.locator('meta[name=robots]').getAttribute('content'),'noindex');

      await p.locator('#email-unsubscribe').click();
      await p.waitForFunction(()=>/unsubscribed\./.test(document.getElementById('email-status').textContent));
      assert.equal(await status(),'You’re unsubscribed. You won’t receive the Morning Card email.');
      assert.deepEqual(await p.evaluate(t=>[db.rows[t].subscribed,db.calls.at(-1)],TOKEN),
        [false,['email_subscription_set',{p_token:TOKEN,p_subscribed:false}]]);
      assert.deepEqual(await visible(),['email-resubscribe']);
      assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`overflow at ${width}`);
      await p.screenshot({path:`/tmp/fv-email-unsubscribed-${width}.png`,fullPage:true});

      await p.locator('#email-resubscribe').click();
      await p.waitForFunction(()=>/subscribed again/.test(document.getElementById('email-status').textContent));
      assert.equal(await p.evaluate(t=>db.rows[t].subscribed,TOKEN),true);

      // Reloading an unsubscribed link shows that state and offers to resubscribe.
      await p.evaluate(t=>{db.rows[t].subscribed=false;sessionStorage.setItem('fake-db',JSON.stringify(db.rows));},TOKEN);
      await p.reload();
      await p.waitForFunction(()=>!/Checking/.test(document.getElementById('email-status').textContent));
      assert.equal(await status(),'You’re unsubscribed from the Morning Card email.');
      assert.deepEqual(await visible(),['email-resubscribe']);

      // An outage changes nothing and offers a retry that repeats the same step.
      await p.evaluate(()=>{db.fail=true;});
      await p.locator('#email-resubscribe').click();
      await p.waitForFunction(()=>/couldn’t reach/.test(document.getElementById('email-status').textContent));
      assert.deepEqual(await visible(),['email-retry']);
      await p.evaluate(()=>{db.fail=false;});
      await p.locator('#email-retry').click();
      await p.waitForFunction(()=>/subscribed again/.test(document.getElementById('email-status').textContent));

      // Unknown and malformed tokens: no buttons; a malformed one never reaches the database.
      await p.goto(`${base}/email/unsubscribe.html?token=11111111-2222-4333-8444-555555555555`);
      await p.waitForFunction(()=>/isn’t valid/.test(document.getElementById('email-status').textContent));
      assert.deepEqual(await visible(),[]);
      for(const bad of ['?token=not-a-token','']){
        await p.goto(`${base}/email/unsubscribe.html${bad}`);
        await p.waitForFunction(()=>/isn’t valid/.test(document.getElementById('email-status').textContent));
        assert.deepEqual(await p.evaluate(()=>db.calls),[]);
      }
      assert.deepEqual(errors,[]);
      await p.close();
    }
    console.log('PASS: unsubscribe page: opening changes nothing, one-click unsubscribe and resubscribe, stored state, outage retry, bad links, phone and desktop.');
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

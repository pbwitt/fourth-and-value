const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const code=fs.readFileSync(path.join(__dirname,'../supabase/functions/editorial-write-now/index.ts'),'utf8');
const id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
async function scenario({editor=true,reader=false,origin='https://fourthandvalue.com',token=true,conflict=false,dispatchOK=true,publish=false}={}){
 let handler;const calls=[];
 const fetch=async(url,options={})=>{calls.push({url,options});let data;
  if(url.endsWith('/auth/v1/user'))data={id:'owner',app_metadata:{fv_editor:editor}};
  else if(url.includes('/rest/')&&options.method!=='PATCH')data=[{id,user_id:reader?'reader':'owner',requires_review:reader,status:'submitted',kind:'analysis',sport:'NFL',updated_at:'2026-09-24T12:00:00Z'}];
  else if(url.includes('/rest/'))data=conflict?[]:[{id}];
  else if(url.includes('api.github.com'))return new Response('',{status:dispatchOK?200:500});
  return Response.json(data);
 };
 vm.runInNewContext(code,{Request,Response,fetch,Date,JSON,Deno:{env:{get:k=>({SUPABASE_URL:'https://db.example',SUPABASE_ANON_KEY:'public',GITHUB_EDITORIAL_TOKEN:token?'private-token':null}[k])},serve:h=>handler=h}});
 const response=await handler(new Request('https://function.example',{method:'POST',headers:{origin,authorization:'Bearer user-session'},body:JSON.stringify({idea_id:id,publish_own:publish})}));
 return {response,data:await response.json(),calls,dispatch:calls.find(c=>c.url.includes('api.github.com'))};
}
(async()=>{
 let r=await scenario({editor:false});assert.equal(r.response.status,403);assert.equal(r.dispatch,undefined);
 r=await scenario({origin:'https://other.example'});assert.equal(r.response.status,403);assert.equal(r.calls.length,0);
 r=await scenario({token:false});assert.equal(r.response.status,503);assert.equal(r.dispatch,undefined);
 r=await scenario({reader:true,publish:true});assert.equal(r.response.status,202);assert.equal(JSON.parse(r.dispatch.options.body).inputs.publish_own,false);
 assert.ok(!JSON.stringify(r.data).includes('private-token'));
 r=await scenario({publish:true});assert.equal(JSON.parse(r.dispatch.options.body).inputs.publish_own,true);
 r=await scenario();assert.equal(JSON.parse(r.dispatch.options.body).inputs.publish_own,false);
 r=await scenario({conflict:true});assert.equal(r.response.status,409);assert.equal(r.dispatch,undefined);
 r=await scenario({dispatchOK:false});assert.equal(r.response.status,502);
 console.log('Write-now dispatcher passed authorization, origin, private-token, draft-default, reader-approval, duplicate-click and failure checks.');
})().catch(e=>{console.error(e);process.exitCode=1;});

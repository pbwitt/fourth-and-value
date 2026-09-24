// Deploy as editorial-write-now. Keep GITHUB_EDITORIAL_TOKEN in Edge Function secrets.
// No service-role token is needed: database calls retain the editor's own JWT.
const origin = 'https://fourthandvalue.com';
const cors = {'Access-Control-Allow-Origin':origin,'Access-Control-Allow-Headers':'authorization, x-client-info, apikey, content-type','Access-Control-Allow-Methods':'POST, OPTIONS','Vary':'Origin'};
function reply(status,message){return new Response(JSON.stringify(message),{status,headers:{...cors,'Content-Type':'application/json'}});}
async function handle(req){
 if(req.headers.get('origin')!==origin)return reply(403,{error:'Unsupported origin'});
 if(req.method==='OPTIONS')return new Response(null,{status:204,headers:cors});
 if(req.method!=='POST')return reply(405,{error:'POST required'});
 const authorization=req.headers.get('authorization')||'';
 if(!authorization.startsWith('Bearer '))return reply(401,{error:'Sign in first'});
 const base=Deno.env.get('SUPABASE_URL'),key=Deno.env.get('SUPABASE_ANON_KEY'),token=Deno.env.get('GITHUB_EDITORIAL_TOKEN');
 if(!base||!key||!token)return reply(503,{error:'Write now needs its server connection configured.'});
 try{
  const headers={apikey:key,Authorization:authorization,'Content-Type':'application/json'};
  const auth=await fetch(base+'/auth/v1/user',{headers});
  if(!auth.ok)return reply(401,{error:'Sign in again'});
  const user=await auth.json();
  if(user.app_metadata?.fv_editor!==true)return reply(403,{error:'Editor access required'});
  const raw=await req.text();if(raw.length>1000)return reply(400,{error:'Request too large'});
  let input;try{input=JSON.parse(raw);}catch{return reply(400,{error:'Invalid request'});}
  if(!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(input.idea_id||''))return reply(400,{error:'Invalid idea identifier'});
  const endpoint=base+'/rest/v1/editorial_ideas?id=eq.'+input.idea_id;
  const loaded=await fetch(endpoint+'&select=*',{headers});
  if(!loaded.ok)return reply(502,{error:'Cannot load that idea'});
  const row=(await loaded.json())[0];
  if(!row||row.status!=='submitted'||row.kind!=='analysis'||!['NFL','MLB','NBA','NHL'].includes(row.sport))return reply(409,{error:'Choose a submitted analysis idea under a specific sport.'});
  if(row.write_now_requested_at&&Date.now()-Date.parse(row.write_now_requested_at)<60000)return reply(409,{error:'That request was just sent. Give it a minute before retrying.'});
  const publish=!!input.publish_own&&row.user_id===user.id&&!row.requires_review;
  // Compare-and-swap stops two clicks queuing separate requests concurrently.
  const saved=await fetch(endpoint+'&status=eq.submitted&updated_at=eq.'+encodeURIComponent(row.updated_at),{
   method:'PATCH',headers:{...headers,Prefer:'return=representation'},body:JSON.stringify({
    research_requested_at:new Date().toISOString(),write_now_requested_at:new Date().toISOString(),write_now_publish:publish})});
  if(!saved.ok)return reply(503,{error:'Write now database setup is missing or the idea changed. Refresh the desk.'});
  if(!(await saved.json()).length)return reply(409,{error:'Idea changed. Refresh before requesting again.'});
  const dispatched=await fetch('https://api.github.com/repos/pbwitt/fourth-and-value/actions/workflows/editorial-daily.yml/dispatches',{
   method:'POST',headers:{Authorization:'Bearer '+token,Accept:'application/vnd.github+json','Content-Type':'application/json','X-GitHub-Api-Version':'2026-03-10'},
   body:JSON.stringify({ref:'main',inputs:{idea_id:row.id,publish_own:publish,refresh_briefing:true,check_api:false}})});
  if(!dispatched.ok)return reply(502,{error:'The writer could not be started. The idea is saved; retry Write now after a minute.'});
  return reply(202,{message:publish?'Research requested. Your article can publish after checks pass.':'Research requested. The finished draft will wait for your approval.'});
 }catch{return reply(502,{error:'Connection failed. Your idea is saved; refresh and check its status before retrying.'});}
}
Deno.serve(handle);

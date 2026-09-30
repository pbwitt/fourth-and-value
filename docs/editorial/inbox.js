/* Authentication and database policies protect the data; hiding this page does not. */
(()=>{
 const $=id=>document.getElementById(id),db=window.supabaseClient;
 let user=null,current=null,dirty=false,busy=false,signingIn=false;
 const dispatchFailures=new Set();
 const fields=['idea','kind','sport','publish_on','title','byline','body','sources','featured'];
 const message=text=>{$('message').textContent=text;$('action-message').textContent=text;$('action-message').hidden=false;};
 const friendly=error=>error?.code==='PGRST205'?'The editorial database needs its one-time setup. Your text is still in the form; copy it before leaving.':error?.message||'The request failed. Your text is still in the form.';
 function pending(row=current){return row?.status==='submitted'&&!row.research_error&&!dispatchFailures.has(row.id)&&Date.now()-Date.parse(row.write_now_requested_at)<15*60000;}
 function statusLabel(row){
  if(row.status==='published')return 'PUBLISHED';
  if(row.status==='publishing')return 'PUBLISHING';
  if(row.status==='approved')return 'PUBLICATION REQUESTED';
  if(row.status==='researching')return 'WRITING';
  if(row.research_error)return row.body?.trim()?'DRAFT · LAST WRITING ATTEMPT FAILED':'NEEDS ATTENTION · NO DRAFT';
  if(row.status==='submitted')return pending(row)?'WRITING REQUESTED':row.write_now_requested_at?'REQUEST NEEDS CHECKING':row.research_requested_at?'QUEUED FOR RESEARCH':'IDEA ONLY';
  return row.status==='review'?'DRAFT · AWAITING YOUR PUBLISH ACTION':row.status.toUpperCase();
 }
 function missingDraft(row){
  const missing=[];
  if(!row?.title?.trim())missing.push('a title');
  if(!row?.byline?.trim())missing.push('a byline');
  if((row?.body?.trim().length||0)<100)missing.push('the complete article (at least 100 characters)');
  if(row?.kind==='analysis'&&!row?.sources?.trim())missing.push('source links for analysis');
  return missing;
 }
 function publicationHelp(){
  if(current?.status==='published')return 'Published. Further edits must be saved and explicitly published again.';
  if(current?.status==='publishing')return 'Your publication request is being delivered. Do not submit a second copy.';
  if(current?.status==='approved'&&!dirty)return 'You requested publication of this saved version. Delivery runs hourly and honors the earliest publication date. No further action is needed.';
  if(dirty)return 'Save your changes first. Saving keeps this draft private and does not publish it.';
  if(pending()||current?.status==='researching')return 'Writing is still processing. The finished draft will wait here for your review.';
  const missing=missingDraft(current);
  if(missing.length)return 'Not ready to publish. Add '+missing.join(', ')+', then save the draft. Story instructions alone are not an article.';
  if(current?.status!=='review')return 'Open the article editor and save the draft for review before publishing.';
  return 'Ready for your decision. Preview the saved article, then choose Publish saved article. Nothing goes live until you confirm.';
 }
 function controls(){
  const queued=pending(),locked=queued||['publishing','researching'].includes(current?.status);
  $('approve').disabled=busy||dirty||!current||current.status!=='review'||!!missingDraft(current).length;
  $('approve').hidden=false;
  $('withdraw').hidden=current?.status!=='approved';
  $('withdraw').disabled=busy||dirty;
  $('archive').disabled=busy||locked||!current;
  $('save').disabled=busy||locked;
  $('preview').hidden=!$('body').value.trim();
  $('preview').disabled=busy||dirty;
  $('research').hidden=!(current?.requires_review&&current.status==='submitted'&&['analysis','opinion'].includes(current.kind)&&!current.research_requested_at);
  $('research').disabled=busy||dirty||queued;
  $('write-now').hidden=!(current?.status==='submitted'&&['analysis','opinion'].includes(current.kind)&&current.sport!=='Sports');
  $('write-now').disabled=busy||dirty||queued;
  $('write-now').textContent=busy?'Please wait…':queued?(current?.body?.trim()?'Rewrite queued':'Draft requested'):current?.body?.trim()?'Retry rewrite':'Generate draft';
  $('write-help').hidden=!dirty;
  $('save').textContent=$('body').value.trim()?'Save draft':'Save idea';
  $('rewrite').hidden=!current?.body?.trim()||!['review','archived'].includes(current?.status)||!['analysis','opinion'].includes(current?.kind)||current?.sport==='Sports';
  $('rewrite').disabled=busy||dirty;
  $('confirm-rewrite').disabled=busy||dirty||$('rewrite').hidden||!$('rewrite-feedback').value.trim();
  $('edit-draft').disabled=busy||locked;
  $('publication-help').textContent=publicationHelp();
  $('reader-rule').hidden=!current?.requires_review;
  $('story-state').textContent=dirty?'Unsaved changes':current?statusLabel(current):'New idea — no article yet';
  $('draft-step').textContent=$('body').value.trim()?'2. Review the article':'2. Generate and review the draft';
  for(const field of fields)$(field).disabled=locked||busy;
 }
 function values(){const o={};for(const f of fields)o[f]=f==='featured'?$(f).checked:$(f).value;o.publish_on=o.publish_on||null;return o;}
 function load(row){if(dirty&&!confirm('Discard unsaved changes?'))return;const changed=current?.id!==row?.id;current=row;if(changed){$('rewrite-feedback').value='';$('rewrite-panel').hidden=true;}for(const f of fields){if(f==='featured')$(f).checked=!!row?.[f];else $(f).value=row?.[f]??(f==='kind'?'analysis':f==='sport'?'NFL':'');}dirty=false;controls();$('form-title').textContent=row?'Review story':'New idea';$('draft-fields').open=!!row?.body;$('preview-panel').hidden=true;if(row?.research_error&&!['approved','publishing','published'].includes(row.status))message(row.research_error);else if(row?.status==='review')message('Your draft is ready for review. Saving and homepage placement do not publish it.');else if(row?.status==='researching')message(row.body?.trim()?'Rewrite in progress. Your current draft stays available until the replacement passes checks.':'Research is in progress.');else if(pending(row))message(row.body?.trim()?'Rewrite queued. Data refresh and writing can take several minutes; this page checks for updates automatically.':'Draft requested. This page checks for updates automatically.');else message(publicationHelp());}
 async function queue(){const {data,error}=await db.from('editorial_ideas').select('*').order('updated_at',{ascending:false}).limit(100);if(error)throw error;$('queue').replaceChildren();for(const row of data){const b=document.createElement('button');b.type='button';b.className='queue-item';b.textContent=`${row.requires_review?'READER IDEA':row.user_id===user?.id?'YOUR IDEA':'EDITOR IDEA'} · ${row.sport} · ${statusLabel(row)} · ${row.kind} · ${row.title||row.idea.slice(0,100)}`;b.onclick=()=>load(row);$('queue').append(b);}const latest=current&&data.find(r=>r.id===current.id);if(latest&&!dirty&&latest.updated_at!==current.updated_at)load(latest);controls();if(!data.length)$('queue').textContent='Your first idea can start below.';const waiting=data.filter(r=>r.requires_review&&r.status==='submitted'&&!r.research_requested_at).length;const review=data.filter(r=>r.status==='review').length;const own=data.filter(r=>r.user_id===user?.id&&r.status==='submitted').length;const attention=data.filter(r=>r.research_error&&!['approved','publishing','published'].includes(r.status)).length;const researching=data.filter(r=>r.status==='researching'||(r.status==='submitted'&&r.research_requested_at&&!r.research_error)).length;$('inbox-notice').textContent=`${own} of your ideas queued · ${waiting} reader suggestions to consider · ${researching} awaiting or undergoing research · ${review} drafts awaiting approval${attention?' · '+attention+' need attention':''}`;return data;}
 async function access(){const {data,error}=await db.auth.getUser();user=data?.user;$('login').hidden=!!user;$('desk').hidden=true;if(!user){message('Sign in to your private editorial workspace.');return;}if(user.app_metadata?.fv_editor!==true){message('This account does not have editorial-owner access yet.');$('login').hidden=false;return;}try{await queue();$('desk').hidden=false;message('Reader submissions stay private until you explicitly publish the saved article. Select an idea or draft below.');}catch(e){message(friendly(e));}}
 $('login-form').onsubmit=async e=>{e.preventDefault();if(signingIn)return;signingIn=true;const button=$('login-form').querySelector('button');button.disabled=true;try{const {error}=await db.auth.signInWithOtp({email:$('email').value.trim(),options:{emailRedirectTo:location.origin+'/editorial/inbox.html',shouldCreateUser:true}});message(error?window.signInErrorMessage(error):'Check your email for the sign-in link. Once signed in, you can keep reviewing without another sign-in email.');}catch(error){message(window.signInErrorMessage(error));}finally{signingIn=false;button.disabled=false;}};
 $('idea-form').addEventListener('input',e=>{if(!fields.includes(e.target.id))return;dirty=true;controls();});
 $('edit-draft').onclick=()=>{$('draft-fields').open=true;$('body').focus();$('body').scrollIntoView({behavior:'smooth',block:'center'});};
 $('idea-form').onsubmit=async e=>{e.preventDefault();if(busy)return;busy=true;controls();try{const v=values();for(const link of v.sources.split('\n').map(x=>x.trim()).filter(Boolean)){const u=new URL(link);if(u.protocol!=='https:'||u.username||u.password)throw Error('Sources must be https links, one per line.');}let q;if(current)q=db.from('editorial_ideas').update(v).eq('id',current.id).eq('updated_at',current.updated_at);else q=db.from('editorial_ideas').insert({...v,user_id:user.id});const {data,error}=await q.select();if(error)throw error;if(!data?.length)throw Error('This draft changed on another device. Refresh the inbox before saving again.');current=data[0];dirty=false;controls();await queue();message('Saved. '+(['approved','publishing','published'].includes(current.status)?publicationHelp():current.status==='review'?'Your draft is private. Preview it, then choose Publish saved article when ready.':current?.sport==='Sports'?'Choose NFL, MLB, NBA or NHL to generate a draft, or write the article yourself.':current?.requires_review?'Choose Generate draft to research and write this idea. The finished draft will wait for your review.':'Choose Generate draft to write now, or wait for the daily drafting queue. The draft will need your Publish action.'));}catch(e){message(friendly(e));}finally{busy=false;controls();}};
 $('write-now').onclick=async()=>{if(!current||busy||dirty||pending()||current.status!=='submitted'||!['analysis','opinion'].includes(current.kind))return;busy=true;controls();message('Requesting research for this idea…');try{const {data,error}=await db.functions.invoke('editorial-write-now',{body:{idea_id:current.id,publish_own:false}});if(error){dispatchFailures.add(current.id);let detail;try{detail=await error.context?.json();}catch{}throw Error(detail?.error||detail?.message||'Draft writing needs its server connection configured. Your idea is saved.');}dispatchFailures.delete(current.id);await queue();message((data?.message||'Draft requested.')+' This is an extra writing attempt within the weekly spending cap. Data refresh and writing can take several minutes.');}catch(e){message(friendly(e));}finally{busy=false;controls();}};
 $('rewrite').onclick=()=>{if(current?.research_error&&!$('rewrite-feedback').value.trim())$('rewrite-feedback').value=current.idea.split('\n\nChanges for the next draft:\n').slice(1).join('\n\nChanges for the next draft:\n');$('rewrite-panel').hidden=false;controls();$('rewrite-feedback').focus();};
 $('rewrite-feedback').addEventListener('input',controls);
 $('confirm-rewrite').onclick=async()=>{
  if(!current||busy||dirty||$('rewrite').hidden)return;
  const feedback=$('rewrite-feedback').value.trim();if(!feedback)return;
  busy=true;controls();message('Saving your rewrite instructions…');
  try{
   const marker='\n\nChanges for the next draft:\n';
   const topic=current.idea.split(marker)[0]+marker+feedback;
   if(topic.length>12000)throw Error('Shorten the story idea or rewrite instructions before sending.');
   let result=await db.from('editorial_ideas').update({idea:topic,status:'review',research_error:null}).eq('id',current.id).eq('updated_at',current.updated_at).eq('status',current.status).select();
   if(result.error)throw result.error;
   if(!result.data?.length)throw Error('The draft changed. Refresh and review it before requesting a rewrite.');
   current=result.data[0];
   result=await db.from('editorial_ideas').update({status:'submitted'}).eq('id',current.id).eq('updated_at',current.updated_at).eq('status','review').select();
   if(result.error)throw result.error;
   if(!result.data?.length)throw Error('The draft changed before the rewrite could be requested. Refresh the inbox.');
   current=result.data[0];
   const {error}=await db.functions.invoke('editorial-write-now',{body:{idea_id:current.id,publish_own:false}});
   if(error){dispatchFailures.add(current.id);let detail;try{detail=await error.context?.json();}catch{}throw Error(detail?.error||detail?.message||'The rewrite could not start. Your draft and instructions are saved. Use Retry rewrite to try again.');}
   dispatchFailures.delete(current.id);$('rewrite-feedback').value='';$('rewrite-panel').hidden=true;
   await queue();load(current);message('Rewrite requested. The new draft will replace this one after checks pass and wait for your approval. This is another writing attempt within the weekly spending cap.');
  }catch(e){message(friendly(e));}finally{busy=false;controls();}
 };
 $('research').onclick=async()=>{if(!current||busy||dirty)return;busy=true;controls();try{const {data,error}=await db.from('editorial_ideas').update({research_requested_at:new Date().toISOString()}).eq('id',current.id).eq('updated_at',current.updated_at).select();if(error)throw error;if(!data?.length)throw Error('Suggestion changed. Refresh the inbox.');current=data[0];await queue();message('Accepted for research within the daily article budget. The finished draft will still require your approval.');}catch(e){message(friendly(e));}finally{busy=false;controls();}};
 async function transition(status){
  if(!current||busy||dirty)return;
  if(status==='approved'){
   if(current.status!=='review'||missingDraft(current).length){message(publicationHelp());return;}
   const placement=current.kind==='opinion'?'Opinion section':current.featured?'homepage feature and analysis archive':'analysis archive';
   if(!confirm('Publish “'+current.title+'” by '+current.byline+'?\n\nThis authorizes publication of the exact saved '+current.kind+' article in the '+placement+'. Delivery runs hourly'+(current.publish_on?', no earlier than '+current.publish_on:'')+'.'))return;
  }
  busy=true;controls();
  try{
   const {data,error}=await db.from('editorial_ideas').update({status}).eq('id',current.id).eq('updated_at',current.updated_at).eq('status',current.status).select();
   if(error)throw error;if(!data?.length)throw Error('Draft changed. Refresh and review the latest version.');
   current=data[0];await queue();
   message(status==='approved'?'Publication requested for this saved article. The publisher checks hourly and honors your earliest publication date. Do not submit a second copy.':status==='review'?'Publication request withdrawn. This draft is private and needs your Publish action again.':'Archived.');
  }catch(e){message(friendly(e));}finally{busy=false;controls();}
 }
 $('approve').onclick=()=>transition('approved');$('withdraw').onclick=()=>{if(current?.status==='approved')return transition('review');};$('archive').onclick=()=>transition('archived');$('new').onclick=()=>load(null);
 $('refresh').onclick=async()=>{try{await queue();if(current?.research_error&&!['approved','publishing','published'].includes(current.status))message(current.research_error);else message(current?publicationHelp():'Inbox refreshed. Select an item to load its latest saved version.');}catch(e){message(friendly(e));}};
 $('signout').onclick=async()=>{if(dirty&&!confirm('Sign out and discard unsaved changes?'))return;await db.auth.signOut();location.reload();};
 $('preview').onclick=()=>{const v=values();$('preview-kind').textContent=v.kind==='opinion'?'Opinion':'Analysis';$('preview-title').textContent=v.title||'Untitled';$('preview-byline').textContent=v.byline?'By '+v.byline:'Add a byline';$('preview-body').textContent=v.body;$('preview-panel').hidden=false;$('preview-panel').scrollIntoView({behavior:'smooth'});};
 window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
 setInterval(()=>{if(user?.app_metadata?.fv_editor===true&&!busy&&!document.hidden)queue().catch(()=>{});},60000);
 db.auth.onAuthStateChange(()=>setTimeout(access,0));access().catch(()=>message('Unable to connect. Refresh to try again.'));
})();

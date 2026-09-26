/* Private report access is enforced by RLS, not by this page being unlisted. */
(()=>{
 const db=window.supabaseClient,$=id=>document.getElementById(id);
 let editor=false,generation=0,rows=[],pinned=false;
 const today=()=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
 const when=value=>{if(!value)return 'Not recorded';const d=new Date(value);return Number.isNaN(d.valueOf())?'Not recorded':new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(d)+' ET';};
 const labels={success:'Passed',failure:'Failed',skipped:'Not run',in_progress:'Running',queued:'Queued',not_checked:'Not checked',not_recorded:'Not recorded',ready:'Ready',current:'Current',stale:'Stale',unknown:'Not recorded',no_markets:'No usable markets',unavailable:'Unavailable',published:'Saved for publication',waiting_for_data:'Waiting for evidence',review:'Private draft awaiting approval',started:'Writing started',passed:'Passed',failed:'Failed'};
 const text=value=>labels[value]||String(value||'Not recorded').replaceAll('_',' ');
 const el=(tag,content,cls)=>{const node=document.createElement(tag);if(content!==undefined)node.textContent=content;if(cls)node.className=cls;return node;};
 const pill=value=>el('span',text(value),'pill '+(['success','ready','current','passed'].includes(value)?'good':['failure','failed','unavailable'].includes(value)?'bad':['stale','waiting_for_data'].includes(value)?'warn':''));
 const append=(id,node)=>$(id).append(node);
 function safeLink(value,workflow=false){try{const u=new URL(value,location.origin);return workflow?(u.origin==='https://github.com'&&u.pathname.startsWith('/pbwitt/fourth-and-value/actions/runs/')):u.origin===location.origin&&u.pathname.startsWith('/editorial/articles/');}catch{return false;}}
 function stage(title,status,detail){const li=el('li');li.append(el('h3',title),pill(status),el('p',detail));append('pipeline',li);}
 function render(row){
  const r=row.report||{},st=r.run?.stages||{};$('report').hidden=false;
  $('headline').textContent=`${r.saved??0} of ${r.expected??2} articles saved · ${r.status==='delivered'?'live delivery verified':r.status==='overdue'?'morning edition overdue':r.status==='saved_not_verified'?'live delivery not verified':'edition pending'}`;
  $('observed').textContent=`Observed ${when(r.observed_at)} · ${text(r.phase)} snapshot`;
  const age=(Date.now()-new Date(r.observed_at).valueOf())/3600000;
  $('freshness').textContent=$('day').value!==today()?'Historical report. Checks describe that moment, not the site now.':age>1.5?'This report is over 90 minutes old. No newer observation has arrived; do not treat it as current health.':'Latest selected observation. A green check applies only to the recorded time.';
  const retry=r.recovery||{};$('recovery').textContent=retry.eligible?`Next scheduled recovery opportunity: ${when(retry.next_opportunity)}. Timing is best effort.`:`Automatic writing is not currently eligible: ${text(retry.reason)}.`;
  const link=$('run-link');link.hidden=!safeLink(r.run?.url,true);if(!link.hidden)link.href=r.run.url;
  $('pipeline').replaceChildren();
  stage(r.phase==='watchdog'?'Watchdog started':'Run started',r.run?.started_at?'success':'not_recorded',r.run?.started_at?`${text(r.run.event)} · ${when(r.run.started_at)}`:'No confirmed start time.');
  stage('Prices refreshed',st.prices?.status||'not_checked','See the per-sport checks below. Reused snapshots are not new refreshes.');
  stage('Stories selected',r.selection_count>=r.expected?'success':'waiting_for_data',`${r.selection_count||0} slots selected; ${(r.unselected||[]).length} sports/candidates lacked usable evidence.`);
  stage('Writing',st.writing?.status||'not_checked',`Last writer check: ${when(r.writer_checked_at)}`);
  stage('Factual review',(r.articles||[]).some(a=>a.review==='failed')?'failure':(r.articles||[]).length&&(r.articles||[]).every(a=>a.review==='passed')?'success':'not_checked','Each article lists its review result. A completed writer alone is not a passing review.');
  stage('Public delivery',r.live_check||'not_checked',`Repository publication: ${text(st.publication?.status)}. Site build request: ${text(st.deployment?.status)}.`);
  $('data').replaceChildren();for(const d of r.data||[]){const tr=el('tr');tr.append(el('th',d.sport));const price=el('td');price.append(pill(d.prices));price.title=d.detail||'';tr.append(price,el('td',String(d.games??0)));const model=el('td');model.append(pill(d.model));model.title=when(d.model_checked_at);tr.append(model,el('td',when(d.checked_at)));append('data',tr);}
  $('articles').replaceChildren();for(const a of r.articles||[]){const card=el('article',undefined,'article-card');card.append(el('h3',a.title||`${a.sport} article`),pill(a.status),el('p',`Review: ${text(a.review)}. Source fallback: ${a.source_fallback===true?'used':a.source_fallback===false?'not needed':'not recorded'}.`));if(a.reason)card.append(el('p',a.reason));if(safeLink(a.url)){const link=el('a','Open article ↗');link.href=a.url;card.append(link);}append('articles',card);}if(!(r.articles||[]).length)append('articles',el('p','No writing attempt recorded for this edition.'));
  $('sources').replaceChildren();for(const s of r.sources||[]){append('sources',el('p',`${s.sport}: ${s.ready===true?'evidence ready':s.ready===false?'insufficient evidence':'not recorded'} · ${s.candidates??'unknown'} candidate links · ${(s.hosts||[]).join(', ')||'no readable publishers recorded'} · ${s.fallback_used===true?'official-source fallback used':s.fallback_attempted===true?'fallback attempted; insufficient evidence':'fallback not recorded or not needed'}`));}if(!(r.sources||[]).length)append('sources',el('p','Source diagnostics were not recorded for these older attempts.'));
  $('fallback-policy').textContent=r.fallback_policy||'';$('unselected').replaceChildren();for(const s of r.unselected||[])append('unselected',el('li',`${s.sport}: ${s.reason}`));if(!(r.unselected||[]).length)append('unselected',el('li','No exclusions recorded.'));
 }
 async function refresh(){
  if(!editor)return;const ticket=++generation;$('message').textContent='Loading private reports…';
  const {data,error}=await db.from('editorial_pipeline_reports').select('id,observed_at,report').eq('edition_day',$('day').value).order('observed_at',{ascending:false}).limit(100);
  if(!editor||ticket!==generation)return;
  if(error){$('report').hidden=true;$('snapshot').replaceChildren();$('message').textContent=error.code==='PGRST205'||error.code==='42P01'?'One-time database setup is needed. Run editorial_diagnostics.sql in the Supabase SQL editor.':'Reports could not be loaded. Check your access or try Refresh.';return;}
  rows=data||[];const selected=$('snapshot').value;$('snapshot').replaceChildren();for(const row of rows){const option=el('option',`${when(row.observed_at)} · ${text(row.report?.phase)}`);option.value=row.id;$('snapshot').append(option);}
  $('message').textContent=rows.length?`${rows.length} recorded observations for this date.`:'No report received for this date. The pipeline may not have started, or report storage may be unavailable.';
  if(!rows.length){$('report').hidden=true;$('freshness').textContent='No observation is not a successful check.';return;}
  if(pinned&&rows.some(r=>r.id===selected))$('snapshot').value=selected;render(rows.find(r=>r.id===$('snapshot').value)||rows[0]);
 }
 async function access(){const ticket=++generation;editor=false;$('dashboard').hidden=true;$('report').hidden=true;const {data}=await db.auth.getUser();if(ticket!==generation)return;editor=data?.user?.app_metadata?.fv_editor===true;$('login').hidden=editor;if(!editor){rows=[];$('articles').replaceChildren();$('message').textContent='Sign in with your editor account to view private reports.';return;}$('dashboard').hidden=false;await refresh();}
 $('day').value=today();$('day').max=today();$('day').onchange=()=>{pinned=false;$('snapshot').replaceChildren();refresh().catch(()=>{$('message').textContent='Unable to load reports.';});};$('snapshot').onchange=()=>{pinned=true;const row=rows.find(r=>r.id===$('snapshot').value);if(row)render(row);};$('refresh').onclick=()=>refresh().catch(()=>{$('message').textContent='Unable to refresh reports.';});$('signout').onclick=async()=>{editor=false;++generation;rows=[];$('dashboard').hidden=true;$('report').hidden=true;await db.auth.signOut();location.reload();};
 db.auth.onAuthStateChange(()=>setTimeout(()=>access().catch(()=>{$('message').textContent='Sign-in check failed.';}),0));setInterval(()=>{if(editor&&!document.hidden)refresh().catch(()=>{});},60000);access().catch(()=>{$('message').textContent='Unable to check sign-in.';});
})();

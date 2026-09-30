/* Offline controller checks: no browser, live accounts, paid calls or publication. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'docs/editorial/inbox.html'), 'utf8');
const code = fs.readFileSync(path.join(root, 'docs/editorial/inbox.js'), 'utf8');
class Element {
  constructor(id) { Object.assign(this, {id, value:'', textContent:'', hidden:false, disabled:false, checked:false, children:[], handlers:{}}); }
  addEventListener(name, fn) { this.handlers[name] = fn; }
  replaceChildren() { this.children = []; }
  append(child) { this.children.push(child); }
  focus() {}
  scrollIntoView() {}
}
async function desk(overrides = {}) {
  const elements = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element(m[1])]));
  const state = {row: {id:'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', user_id:'reader', requires_review:true,
    status:'submitted', kind:'opinion', sport:'NFL', idea:'Discuss the battle in the trenches.',
    title:'', byline:'', body:'', sources:'', featured:true, publish_on:'2026-09-30',
    updated_at:'v0', research_error:'Research did not produce a draft.', ...overrides},
    writes:[], dispatches:[], confirmations:[], confirm:true, revision:0};
  const db = {
    auth:{getUser:async()=>({data:{user:{id:'owner',app_metadata:{fv_editor:true}}}}),onAuthStateChange:()=>{}},
    functions:{invoke:async(name, request)=>{state.dispatches.push(request.body);return {data:{message:'Draft requested.'}};}},
    from:()=>({change:null, conditions:[],
      update(change){this.change=change;return this;},
      eq(key,value){this.conditions.push([key,value]);return this;},
      select(){if(!this.change)return this;
        if(this.conditions.some(([key,value])=>state.row[key]!==value))return Promise.resolve({data:[]});
        state.writes.push({...this.change});
        const changed=['title','body','byline','sources','kind','featured','publish_on','idea','sport'].some(k=>k in this.change&&this.change[k]!==state.row[k]);
        state.row={...state.row,...this.change,updated_at:'v'+(++state.revision)};
        if(changed){state.row.status=state.row.body.trim()?'review':'submitted';state.row.approved_by=null;}
        if(this.change.status==='approved')state.row.approved_by='owner';
        return Promise.resolve({data:[{...state.row}]});},
      order(){return this;}, limit:async()=>({data:[{...state.row}]})})
  };
  vm.runInNewContext(code, {document:{getElementById:id=>{assert.ok(elements[id],`Missing HTML element ${id}`);return elements[id];},createElement:()=>new Element(''),hidden:false},
    window:{supabaseClient:db,addEventListener:()=>{}},location:{origin:'https://fourthandvalue.com'},URL,Date,
    confirm:prompt=>{state.confirmations.push(prompt);return state.confirm;},setInterval:()=>{},setTimeout});
  await new Promise(resolve=>setImmediate(resolve));
  elements.queue.children[0].onclick();
  const edit=(id,value)=>{if(id==='featured')elements[id].checked=value;else elements[id].value=value;elements['idea-form'].handlers.input({target:elements[id]});};
  const save=()=>elements['idea-form'].onsubmit({preventDefault(){}});
  return {elements,state,edit,save};
}
(async()=>{
  const {elements:e,state:s,edit,save}=await desk();
  assert.equal(e['draft-fields'].open,true,'Opinion editor must be visible');
  assert.equal(e.approve.hidden,false,'Publish action must be discoverable even with an incomplete idea');
  assert.equal(e.approve.disabled,true);
  assert.match(e['publication-help'].textContent,/complete article/);
  assert.match(e['story-state'].textContent,/NO DRAFT/);
  assert.equal(e['reader-rule'].hidden,false);
  assert.equal(e['publish-now-option'],undefined,'No automatic publication control in the desk');
  await e.approve.onclick();assert.equal(s.writes.length,0);
  edit('title','Football is decided in the trenches');edit('byline','Fourth & Value contributor');
  edit('body','A complete contributor opinion article, with a clear thesis and supporting argument. '.repeat(5));
  assert.equal(e.approve.disabled,true,'Unsaved changes cannot be published');
  await save();
  assert.equal(s.row.status,'review');assert.equal(s.dispatches.length,0);
  assert.equal(s.writes.some(w=>w.status==='approved'),false,'Saving is not publication');
  assert.equal(e.approve.disabled,false);
  s.confirm=false;await e.approve.onclick();
  assert.equal(s.row.status,'review','Cancel leaves the article private');
  s.confirm=true;await e.approve.onclick();
  assert.equal(s.row.status,'approved');assert.equal(s.row.approved_by,'owner');
  assert.match(s.confirmations.at(-1),/Football is decided in the trenches/);
  assert.match(s.confirmations.at(-1),/Opinion section/);
  assert.match(e['publication-help'].textContent,/requested publication/);
  assert.equal(e.approve.disabled,true,'Do not offer another publish request');
  assert.equal(e.withdraw.hidden,false);
  await e.withdraw.onclick();assert.equal(s.row.status,'review');
  await e.approve.onclick();
  edit('featured',false);await save();
  assert.equal(s.row.status,'review','Placement edits also require a fresh manual publish action');
  s.row.updated_at='changed-elsewhere';await e.approve.onclick();
  assert.match(e['action-message'].textContent,/Draft changed/);
  assert.equal(s.row.status,'review','Stale editor state cannot publish');

  const reader=await desk({kind:'analysis',research_error:null});
  await reader.elements['write-now'].onclick();
  assert.equal(reader.state.dispatches.length,1);
  assert.equal(reader.state.dispatches[0].publish_own,false);
  assert.equal(reader.state.writes.some(w=>w.status==='approved'),false);

  const incomplete=await desk({status:'review',title:'Title',byline:'Author',body:'Too short',research_error:null});
  assert.equal(incomplete.elements.approve.disabled,true);
  await incomplete.elements.approve.onclick();assert.equal(incomplete.state.writes.length,0);
  const owner=await desk({user_id:'owner',requires_review:false,kind:'analysis',research_error:null});
  assert.equal(owner.elements['publish-now-option'],undefined);
  await owner.elements['write-now'].onclick();
  assert.equal(owner.state.dispatches[0].publish_own,false,'Even owner requests must stop at a private draft');
  console.log('Manual publication checks passed: idea vs draft, opinion editor, save without publication, explicit confirmation, withdrawal, stale edits, and reader auto-publish rejection.');
})().catch(error=>{console.error(error);process.exitCode=1;});

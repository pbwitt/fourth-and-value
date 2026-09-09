// Exercise the shipped UI script against generated snapshots without a browser binary.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.value='';this.checked=false;this.innerHTML='';}
 append(...nodes){this.children.push(...nodes);}
 add(option){this.children.push(option);}
 addEventListener(type,fn){this.events[type]=fn;}
 querySelectorAll(selector){return this.children.flatMap(c=>c instanceof Element?[(c.tag==='input'?c:null),...c.querySelectorAll(selector)].filter(Boolean):[]);}
 scrollIntoView(){}
}
function setup(file){
 const html=fs.readFileSync(file,'utf8');
 const payload=JSON.parse(html.match(/<script type="application\/json" id="props-data">(.*?)<\/script>/s)[1]);
 const snapshotNow=Date.parse(payload.lastKickoff)-7*86400000;
 class SnapshotDate extends Date {constructor(...args){super(...(args.length?args:[snapshotNow]));}static now(){return snapshotNow;}}
 const ids=['props-data','q','market','game','sort','best','positive','history','books','book-count','count','results','pager','previous','next','page-info','freshness','all-books','no-books','reset','share','feedback','filters'];
 const els=Object.fromEntries(ids.map(id=>[id,new Element()]));els['props-data'].textContent=JSON.stringify(payload);
 const location={href:'https://fourthandvalue.com/props/',search:'',origin:'https://fourthandvalue.com'};
 const context={document:{getElementById:id=>els[id],createElement:tag=>new Element(tag),createTextNode:t=>t,head:new Element()},
  location,URL,URLSearchParams,Option:class{constructor(label,value){this.label=label;this.value=value;}},
  history:{replaceState:(_,__,url)=>{location.href=String(url);location.search=new URL(url).search;}},
  navigator:{clipboard:{writeText:async()=>{}}},setInterval:()=>{},window:{},console,Date:SnapshotDate,Intl};
 vm.runInNewContext(fs.readFileSync('docs/assets/props.js','utf8'),context);
 return {els,location,payload};
}
let {els,location}=setup('docs/props/index.html');
const count=()=>Number(els.count.textContent.split(' ')[0].replaceAll(',',''));
assert(count()>24);assert.equal((els.results.innerHTML.match(/<article /g)||[]).length,24);
els['no-books'].onclick();assert.equal(count(),0);assert(els.results.innerHTML.includes('No sportsbooks selected'));
els.reset.onclick();assert(count()>24);
els.q.events.input({target:{value:'no-such-player-92834'}});assert.equal(count(),0);assert(new URL(location.href).searchParams.has('q'));
els.reset.onclick();
const choices=els.books.querySelectorAll('input');els['no-books'].onclick();
const selected=choices[0];selected.checked=true;selected.events.change();assert(count()>0);
assert.equal(els['book-count'].textContent,`(1 of ${choices.length})`);
els.next.onclick();assert.equal(els['page-info'].textContent.split(' ')[1],'2');
els.reset.onclick();assert.equal(els['page-info'].textContent.split(' ')[1],'1');
({els}=setup('docs/props/top.html'));
assert(els.results.innerHTML.includes('No qualifying picks'));
assert(els.results.innerHTML.includes('recent quote timestamps'));
assert(!els.freshness.textContent.includes('No upcoming NFL games'),'empty shortlist must not claim no upcoming games');
assert(els.history.disabled);assert(els.positive.disabled);
console.log('PASS: pagination, zero-book filter, search, URL state, selected-book comparison, reset and empty shortlist freshness.');

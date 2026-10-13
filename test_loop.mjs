// UI state-machine regression tests, with a deterministic fake model service.
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=readFileSync(new URL('./stepper.html',import.meta.url),'utf8').split('<script>')[1].split('</script>')[0];
function setup(mode,stop=false){
 const elements=new Map();const element=()=>({value:'',textContent:'',disabled:false,style:{},append(){},replaceChildren(){}});
 const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
 const calls=[];let ctx;
 const prediction={completion_probabilities:[{id:2,top_logprobs:[{id:2,token:'x',logprob:0}]}]};
 ctx=vm.createContext({console,Date,JSON,Math,Number,Error,document:{getElementById:get,querySelectorAll:()=>[],createElement:element},fetch:async(url,opts)=>{
   const b=JSON.parse(opts.body);calls.push(b.action);
   let data={};
   if(b.action==='start')data={tokens:[1]};
   if(b.action==='inspect')data={h:[1],baseline:prediction};
   if(b.action==='explain')data={text:'original',reconstruction:[.9],error:[.1],metrics:{}};
   if(b.action==='propose'){data={proposal:'edited',instruction:b.instruction};if(stop)get('stop').onclick();}
   if(b.action==='compare')data={original:prediction,roundtrip:prediction,edited_raw:prediction,edited_corrected:prediction,error:[.1],metrics:{}};
   if(b.action==='decode')data={content:'x'.repeat(b.tokens.length)};
   return {ok:true,json:async()=>data};
 }});
 vm.runInContext(source,ctx);get('mode').value=mode;get('steps').value='2';get('instruction').value='test edit';get('prompt').value='test';
 return {get,calls,state:()=>JSON.parse(vm.runInContext('JSON.stringify(s)',ctx))};
}
let a=setup('review');await a.get('start').onclick();await a.get('loop').onclick();
assert.equal(a.calls.includes('compare'),false);assert.equal(a.state().tokens.length,1);assert.equal(a.get('approve').disabled,false);
await a.get('approve').onclick();assert.equal(a.state().history.length,1);assert.equal(a.state().history[0].approval.mode,'manual');assert.equal(a.state().tokens.length,2);assert.equal(a.state().pending,true);
a.get('stop').onclick();assert.equal(a.state().pending,false);
a=setup('auto');await a.get('start').onclick();await a.get('loop').onclick();assert.equal(a.state().history.length,2);assert.equal(a.calls.filter(x=>x==='compare').length,2);assert.equal(a.state().history[0].approval.mode,'automatic');
a=setup('auto',true);await a.get('start').onclick();await a.get('loop').onclick();assert.equal(a.calls.includes('compare'),false);assert.equal(a.state().history.length,0);
console.log('PASS: manual gating, edited history, two-step auto limit, stop before injection');

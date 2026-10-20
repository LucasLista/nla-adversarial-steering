// Exercise all exported case/branch/step content through the language controls.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const path=new URL('../research/2026-10-19/deccp/',import.meta.url);
const data=JSON.parse(fs.readFileSync(new URL('review.json',path),'utf8'));
const translations=JSON.parse(fs.readFileSync(new URL('review.en.json',path),'utf8'));
const template=fs.readFileSync(new URL('../ui/deccp_review.html',import.meta.url),'utf8');
class Element{
  constructor(tag='div'){this.tag=tag;this.children=[];this._text='';this._value='';this.selectedIndex=0;this.style={};this.checked=false;this.attrs={}}
  append(...items){this.children.push(...items)}
  replaceChildren(...items){this._text='';this.children=items;this.selectedIndex=0}
  setAttribute(k,v){this.attrs[k]=v}
  scrollIntoView(){}
  get textContent(){return this._text+this.children.map(x=>x.textContent).join('')}
  set textContent(v){this._text=String(v);this.children=[]}
  get value(){return this.tag==='select'?(this.children[this.selectedIndex]?.value??''):this._value}
  set value(v){if(this.tag==='select')this.selectedIndex=this.children.findIndex(x=>x.value===v);else this._value=String(v)}
}
const ids={};
for(const id of ['data','translations','language','translation-note','message-heading','snapshot','cases','branch','step','case-id','question','instruction','responses','prev','next','changed','inspection','step-note','nla','edited','prefix','message','token-text'])ids[id]=new Element(['language','branch','step'].includes(id)?'select':'div');
for(const mode of ['original','english','both']){const e=new Element('option');e.value=mode;ids.language.append(e)}
ids.data.textContent=JSON.stringify(data);ids.translations.textContent=JSON.stringify(translations);
const context={document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag)}};
vm.createContext(context);vm.runInContext(template.split('<script>')[1].split('</script>')[0],context);
const en=x=>translations.texts[x]??x;
const hasChinese=x=>/[\u3400-\u9fff]/u.test(x);
const saved=JSON.stringify(data);
let inspected=0;
for(let ci=0;ci<data.cases.length;ci++){
  const c=data.cases[ci];ids.cases.children[ci].onclick();
  for(const mode of ['original','english','both']){
    ids.language.value=mode;ids.language.onchange();
    if(mode==='original')assert.equal(ids.question.textContent,c.prompt);
    if(mode==='english'){
      assert.equal(ids.question.textContent,en(c.prompt));assert.equal(ids.instruction.textContent,en(c.instruction));
      Object.values(c.branches).forEach((b,i)=>assert.equal(ids.responses.children[i].children[2].textContent,en(b.response)));
    }
  }
  for(const [variant,branch] of Object.entries(c.branches)){
    ids.branch.value=variant;ids.changed.checked=false;ids.branch.onchange();
    for(let i=0;i<branch.steps.length;i++){
      const s=branch.steps[i];ids.step.selectedIndex=i;ids.step.onchange();
      ids.language.value='english';ids.language.onchange();
      assert.equal(ids.step.selectedIndex,i,'Language switch reset the selected step');
      assert.equal(ids.branch.value,variant);
      assert.equal(ids['token-text'].textContent,branch.response,'Token spans must reproduce the original response');
      assert.equal(ids['token-text'].children[i].attrs['aria-pressed'],'true');
      assert.equal(ids.nla.textContent,en(s.nla));
      if(s.edited!==null)assert.equal(ids.edited.textContent,en(s.edited));
      if(s.generated_prefix)assert.equal(ids.prefix.textContent,en(s.generated_prefix));
      for(const k of ['nla','edited','prefix','message'])assert(!hasChinese(ids[k].textContent),'Untranslated '+k);
      ids.language.value='original';ids.language.onchange();
      assert.equal(ids.nla.textContent,s.nla);
      if(s.edited!==null)assert.equal(ids.edited.textContent,s.edited);
      if(s.generated_prefix)assert.equal(ids.prefix.textContent,s.generated_prefix);
      if(s.editor_request)assert.equal(ids.message.textContent,s.editor_message,'Exact editor message changed');
      inspected++;
    }
    ids.changed.checked=true;ids.changed.onchange();
    const n=branch.steps.filter(s=>s.edited!==null&&s.edited!==s.nla).length;
    ids.language.value='both';ids.language.onchange();
    assert(ids.changed.checked);assert.equal(ids.step.children.length,n);
    if(n){const s=branch.steps.find(s=>s.edited!==null&&s.edited!==s.nla);assert(ids.nla.textContent.includes(s.nla));assert(ids.nla.textContent.includes(en(s.nla)))}
  }
}
ids.changed.checked=true;ids.changed.onchange();
ids['token-text'].children[3].onclick();
assert.equal(ids.changed.checked,false);
assert.equal(ids.step.selectedIndex,3);
assert.equal(ids['token-text'].children[3].attrs['aria-pressed'],'true');
assert.equal(JSON.stringify(data),saved);
console.log(`Passed all 5 cases, 20 branches, ${inspected} steps, exact original messages, English coverage, bilingual text and preserved selection/filter.`);

// Read-only regression: original training lesson, now optional Part 2B.
// node src/check_part3_continuity.mjs [part2b.html]
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {forward} from './toy_ref.mjs';

const require=createRequire(import.meta.url);
let pw;
for(const candidate of [process.env.PLAYWRIGHT_PATH,'/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)){
  try{pw=require(candidate);break;}catch{}
}
if(!pw)throw Error('Set PLAYWRIGHT_PATH to an existing installation; this check installs nothing.');
const base=JSON.parse(readFileSync(new URL('toy.json',import.meta.url),'utf8'));
const saved=JSON.parse(readFileSync(new URL('toy3.json',import.meta.url),'utf8'));
for(const key of Object.keys(base))assert.deepEqual(saved[key],base[key],'Part III initial '+key);
const parameterNames=['W_Q','W_K','W_V','W_O','W_hidden','b_hidden','W_vocab','b_vocab'];
const training=saved.training,tokens=training.sentence,target=base.vocab.indexOf(training.target);
let numbers=0,maxError=0;
function near(actual,expected,label){
  if(Array.isArray(expected)){assert.equal(actual.length,expected.length,label+' shape');expected.forEach((x,i)=>near(actual[i],x,label+'['+i+']'));return;}
  assert(Number.isFinite(actual)&&Number.isFinite(expected),label+' finite');
  const error=Math.abs(actual-expected);numbers++;maxError=Math.max(maxError,error);
  assert(error<1e-12,label+': '+actual+' != '+expected);
}
function update(values,grad,eta){return Array.isArray(values)?values.map((x,i)=>update(x,grad[i],eta)):values-eta*grad;}
function updatedModel(parameters,grad,eta){
  const model=structuredClone(base);
  for(const key of parameterNames){near(parameters[key],update(base[key],grad[key],eta),'SGD '+key);model[key]=parameters[key];}
  for(const [word,row] of Object.entries(parameters.tok_emb_used)){
    near(row,update(base.tok_emb[word],grad.tok_emb_used[word],eta),'SGD token '+word);model.tok_emb[word]=row;
  }
  parameters.pos_emb_used.forEach((row,i)=>{near(row,update(base.pos_emb[i],grad.pos_emb_used[i],eta),'SGD position '+i);model.pos_emb[i]=row;});
  return model;
}
const initial=forward(base,tokens),single=training.single;
near(initial.probs.at(-1)[target],single.p_target_before,'starting probability');
near(-Math.log(initial.probs.at(-1)[target]),single.loss_before,'starting loss');
for(const step of Object.values(single.steps)){
  const result=forward(updatedModel(step.parameters,single.gradients,step.eta),tokens);
  near(result.probs.at(-1)[target],step.p_target,'updated probability '+step.eta);
  near(-Math.log(result.probs.at(-1)[target]),step.loss,'updated loss '+step.eta);
  near(result.Enew.at(-1),step.e_prime_10,'updated context '+step.eta);
  assert(step.p_target>0&&step.p_target<1&&step.loss>0,'stored precision must not imply certainty');
}
const parallel=training.parallel;
const after=forward(updatedModel(parallel.parameters_after,parallel.gradients,parallel.eta),tokens);
let beforeMean=0,afterMean=0;
parallel.per_position.forEach((row,i)=>{
  const targetId=base.vocab.indexOf(row.target),p0=initial.probs[i][targetId],p1=after.probs[i][targetId];
  near(p0,row.p_target_before,'parallel p before');near(p1,row.p_target_after,'parallel p after');
  near(-Math.log(p0),row.loss_before,'parallel loss before');near(-Math.log(p1),row.loss_after,'parallel loss after');
  beforeMean-=Math.log(p0)/tokens.length;afterMean-=Math.log(p1)/tokens.length;
});
near(beforeMean,parallel.mean_loss_before,'parallel mean before');near(afterMean,parallel.mean_loss_after,'parallel mean after');

const browser=await pw.chromium.launch(),errors=[];
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'part2b.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  assert.deepEqual(await page.evaluate(()=>window.__TOY__),saved,'assembled model must be current');
  assert.equal(await page.locator('.frame:not(.context-lesson)').count(),50,'original classroom frames before presentation title');
  assert.equal(await page.locator('.frame.context-lesson').count(),28,'context-cost extension moved from Part II');
  for(const id of ['s02-learning-break','s07-block-break','s13-generation-break'])assert.equal(await page.locator('#'+id+'.frame').count(),1,id);
  assert.equal(await page.locator('#s01 .frame').count(),1,'one opening recap');
  assert.equal(await page.locator('#s12 .frame').count(),1,'one full-model objective reminder');
  assert.equal(await page.locator('#s16 .frame:not(.context-lesson)').count(),1,'one limits frame after the cost extension');
  for(const id of ['s04-forward','s04-backward']){
    const text=await page.locator('#'+id).textContent();
    for(const label of ['8 hidden','20 logits','W₁, b₁','W₂, b₂','loss L'])assert(text.includes(label),id+' includes '+label);
  }
  assert.equal(await page.locator('#s08-net .node.hid').count(),8,'eight block FFN hidden units');
  assert.equal(await page.locator('#s08-net .node.out:not(.ell)').count(),4,'four block FFN outputs');
  assert.equal(await page.locator('#s11-lm .node.hid').count(),8,'eight toy predictor hidden units');
  assert((await page.locator('#s11 .frame').first().textContent()).includes('linear final readout'),'architecture distinction visible');
  assert((await page.locator('#s11-output tbody tr').count())===20,'complete toy MLP vocabulary worksheet');
  assert.equal(await page.locator('#s03-output-shapes tbody tr').count(),5,'both MLP matrices and biases plus W_O');
  assert.equal(await page.locator('#s09-inputs input').count(),4,'four-coordinate LayerNorm example');
  const colourChecks=await page.evaluate(()=>{
    function expected(role){const span=document.createElement('span');span.style.color='var(--c-'+role+')';document.body.appendChild(span);const c=getComputedStyle(span).color;span.remove();return c;}
    return [['#s01 p .obj-d','d'],['#s02-callout .obj-a','a'],['#s05-rule .obj-q','q'],['#s05-rule .obj-k','k'],['#s05-rule .obj-a','a'],['#s12 p .obj-ep','ep']].map(([selector,role])=>({selector,actual:getComputedStyle(document.querySelector(selector)).color,expected:expected(role)}));
  });
  for(const check of colourChecks)assert.equal(check.actual,check.expected,check.selector+' colour');
  assert(await page.evaluate(()=>{
    const colour=getComputedStyle(document.querySelector('#s05-rule .obj-a')).color;
    return [...document.querySelectorAll('#s04-backward path[marker-start]')].every(el=>getComputedStyle(el).stroke===colour)&&[...document.querySelectorAll('#s04-backward .s04-gradient')].every(el=>getComputedStyle(el).fill===colour);
  }),'backward arrows and labels match the gradient equations');
  // Each stored learning-rate preset renders its own full-precision computation.
  for(let k=0;k<3;k++){
    await page.locator('#s05-slider input').evaluate((el,k)=>{el.value=k;el.dispatchEvent(new Event('input',{bubbles:true}));},k);
    const text=await page.locator('#s05-table').textContent(),eta=[.05,.1,.3][k],step=single.steps[String(eta)];
    assert(text.includes(step.p_target.toFixed(4)), 'rendered target probability '+eta);
    assert(text.includes(step.loss.toFixed(4)), 'rendered loss '+eta);
  }
  // Verify live hidden rows, the causal mask, LN and all illustrative head rows.
  const live=await page.evaluate(()=>({F:AT.forward(AT.train.sentence()),ln:AT.ln([1,3,5,7]),heads:AT.heads.examples,ffn:AT.ffn.W2.length}));
  near(live.F.HeadPre,initial.HeadPre,'live hidden preactivation');near(live.F.HeadHidden,initial.HeadHidden,'live hidden activation');
  near(live.ln.output,[-3,-1,1,3].map(x=>x/Math.sqrt(5+1e-5)),'LayerNorm');
  for(let i=0;i<tokens.length;i++){near(live.F.A[i].reduce((a,b)=>a+b,0),1,'attention row sum');for(let j=i+1;j<tokens.length;j++)assert.equal(live.F.A[i][j],0,'causal mask');}
  for(const example of Object.values(live.heads))for(const row of example.rows){near(row.weights.reduce((a,b)=>a+b,0),1,'illustrative head sum');row.weights.forEach((x,j)=>{if(j>example.queryIndex)assert.equal(x,0);});}
  for(const [exampleIndex,name] of ['priya','river'].entries())for(let headIndex=0;headIndex<3;headIndex++)for(const mask of [true,false]){
    await page.locator('#s07-example button').nth(exampleIndex).click();
    await page.locator('#s07-head-buttons button').nth(headIndex).click();
    await page.locator('#s07-toggle button').evaluate((button,on)=>button.set(on),mask);
    const cells=await page.locator('#s07-table tbody td').allTextContents();
    const example=live.heads[name];
    assert.deepEqual(cells.map(x=>x.trim()),example.rows[headIndex].weights.map((w,j)=>mask&&j>example.queryIndex?'×':w.toFixed(2)),'head/mask control '+name+headIndex+mask);
  }
  for(const values of [[2,2,2,2],[-5,0,4,8]]){
    for(let i=0;i<4;i++)await page.locator('#s09-inputs input').nth(i).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},values[i]);
    const cells=(await page.locator('#s09-ln tbody tr').last().locator('td').allTextContents()).map(x=>Number(x.replaceAll('−','-')));
    const mean=values.reduce((a,b)=>a+b,0)/4,variance=values.reduce((a,b)=>a+(b-mean)**2,0)/4;
    near(cells,values.map(x=>Number(((x-mean)/Math.sqrt(variance+1e-5)).toFixed(2))),'live LN sliders');
  }
  for(const count of [1,4,10]){
    for(const selector of ['#s13-cache-slider input','#s13-cost-slider input'])await page.locator(selector).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},count);
    assert((await page.locator('#s13-cache').textContent()).includes(count+' rows cached'),'cache length');
    assert((await page.locator('#s13-cost').textContent()).includes((count*count)+' full score cells'),'score-cell count');
    assert((await page.locator('#s13-cost').textContent()).includes(count*(base.d_k+base.d_v)+' cached coordinates'),'cache coordinates');
  }
  await page.evaluate(()=>AT.present.enter());
  assert.equal(await page.locator('.frame:not(.context-lesson)').count(),51,'core classroom frames including generated title');
  for(const [section,frame] of [['s01',1],['s02',1],['s04',2],['s04',3],['s05',1],['s07',1],['s08',2],['s09',1],['s11',1],['s13',1],['s19',1]]){
    await page.evaluate(([section,frame])=>AT.present.go(section,frame,99),[section,frame]);await page.waitForTimeout(100);
    assert.equal((await page.evaluate(()=>AT.present.fitReport())).overflow,false,section+' fits');
  }
  // Every extra explanation remains available in the article, not as another slide.
  assert.equal(await page.locator('#s11-lm').isVisible(),false,'optional toy worksheet hidden in presentation');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  // Leaving the hidden fixed-size stage can defer Chromium's inherited math
  // font update. Check the settled reading typography, not its first resize tick.
  await page.waitForFunction(()=>{
    const line=document.querySelector('.s05-scalar-line');
    return [...line.children].every(term=>term.getBoundingClientRect().width<=line.clientWidth+1)&&document.documentElement.scrollWidth<=innerWidth+1&&[...line.querySelectorAll(':scope>span>.katex')].every(el=>Math.abs(parseFloat(getComputedStyle(el).fontSize)-1.21*parseFloat(getComputedStyle(el.parentElement).fontSize))<.01);
  },{},{timeout:5000});
  const phoneWidth=await page.evaluate(()=>({width:document.documentElement.scrollWidth,offenders:[...document.querySelectorAll('body *')].filter(el=>{
    if(el.closest('.katex-mathml')||el.getBoundingClientRect().right<=innerWidth+1||getComputedStyle(el).display==='none')return false;
    for(let parent=el.parentElement;parent&&parent!==document.body;parent=parent.parentElement)if(['auto','scroll','hidden','clip'].includes(getComputedStyle(parent).overflowX)&&parent.getBoundingClientRect().right<=innerWidth+1)return false;
    return true;
  }).slice(0,20).map(el=>({id:el.id,section:el.closest('.sec')?.id,tag:el.tagName,cls:String(el.className),right:el.getBoundingClientRect().right,text:el.textContent.slice(0,60)}))}));
  assert(phoneWidth.width<=391,'phone page width: '+JSON.stringify(phoneWidth));
  assert.equal(await page.locator('#s11-lm').isVisible(),true,'optional toy worksheet available in reading mode');
  assert(await page.locator('.s05-scalar-line').evaluate(el=>[...el.children].every(term=>term.getBoundingClientRect().width<=el.clientWidth+1)),'phone scalar equation wraps between complete terms');
  assert.deepEqual(await page.evaluate(()=>window.__TOY__),saved,'interactions must not mutate the model');
  assert.deepEqual(errors,[],'page errors');
  console.log(`PASS: exact Part II initialization; ${numbers} independent numeric checks, max error ${maxError.toExponential(3)}; 51-frame flow, MLP/FFN diagrams, colours, controls, causal mask, and phone layout.`);
}finally{await browser.close();}

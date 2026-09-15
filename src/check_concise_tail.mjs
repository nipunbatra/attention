// Concise classroom route; detailed arithmetic remains usable in reading mode.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {forward} from './toy_ref.mjs';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const model=JSON.parse(fs.readFileSync(new URL('toy.json',import.meta.url),'utf8'));
const shots=fs.mkdtempSync('/private/tmp/concise-tail-');
const expected={
  s11:['s11-frame1','s11-frame-separate-maps','s11-frame-query-setup','s11-frame-query-calc','s11-frame3'],
  s12:['s12-frame-scaling','s12-frame-spread','s12-frame-softmax','s12-frame-bank'],
  s13:['s13-frame1','s13-frame2','s13-frame-mask-row','s13-frame3','s13-frame4'],
  s14:['s14-routing','s14-probabilities','s14-layer-boundary'],
  s15:['s15-frame1'],
  s16:['s16-flow-frame','s16-frame3-routing'],
  s17:['s17-weights'],
  s18:['s18-frame1'],
  s19:['s19-frame-generation','s19-frame-training','s19-frame4']
};
const browser=await chromium.launch();
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify(AT.model));
  await page.evaluate(()=>AT.present.enter());
  async function go(id,build=99){
    await page.evaluate(({id,build})=>{
      const el=document.getElementById(id),sec=el.closest('.sec');
      AT.present.go(sec.id,[...sec.querySelectorAll('.frame')].indexOf(el)+1,build);
    },{id,build});
    await page.waitForTimeout(200);
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id);
  }
  async function fit(label){
    const report=await page.evaluate(()=>AT.present.fitReport());
    assert(!report.overflow,label+': '+JSON.stringify(report));
  }
  // Training and generation share a concrete prefix, with the answer clearly separate.
  const example=await page.locator('#s13-frame1 tbody td').allTextContents();
  assert.deepEqual(example,[model.sentences.river.slice(0,5).join(' '),model.sentences.river[5],model.sentences.river[6]+' …']);
  assert.match(await page.locator('#s13-frame1').getAttribute('data-title'),/Training and generation/);
  assert.match(await page.locator('#s13-training').innerText(),/full sentence is available/);
  assert.match(await page.locator('#s13-training').innerText(),/hides river and all later words/);
  assert.match(await page.locator('#s13-generation').innerText(),/future words are not available/);
  assert.match(await page.locator('#s13-prefix-rule').innerText(),/including itself/);
  assert.match(await page.locator('#s13-prefix-rule').innerText(),/scoring a full test sentence/);
  for(const build of [0,1,2,3,2,1,0,3]){
    await go('s13-frame1',build);await fit('training/generation build '+build);
    assert(await page.locator('#s13-frame1 table').isVisible(),'The concrete example stays visible.');
    for(const [n,id]of ['s13-training','s13-generation','s13-prefix-rule'].entries()){
      assert.equal(await page.locator('#'+id).evaluate(e=>getComputedStyle(e).visibility),build>n?'visible':'hidden');
    }
    await page.screenshot({path:path.join(shots,'causal-context-'+build+'.png')});
  }
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s13-frame2');
  await page.evaluate(()=>AT.present.prev());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s13-frame1');
  for(const [sec,ids]of Object.entries(expected)){
    assert.deepEqual(await page.locator('#'+sec+' .frame').evaluateAll(els=>els.map(e=>e.id)),ids);
    for(const id of ids){
      await go(id);await fit(id);
      await page.screenshot({path:path.join(shots,id+'.png')});
    }
  }
  assert.equal(Object.values(expected).flat().length,25);
  assert.equal(await page.locator('.frame-auto').count(),0,'No companion section accidentally becomes a slide.');
  for(const id of ['s11-frame-key-calc','s11-frame-value-calc','s13-frame-targets','s13-frame-mask-triangle','s14-head-arithmetic','s16-frame2','s18-question8','s19-operational']){
    assert(await page.locator('#'+id).evaluate(el=>el.classList.contains('companion')));
    assert(!(await page.locator('#'+id).isVisible()));
  }
  // Local worksheet controls do not intercept presentation Next.
  await go('s15-frame1',0);
  assert.equal(await page.evaluate(()=>AT.present.state().stepper),null);
  await page.locator('#s15-stepper .btn-next').click();
  assert.equal(await page.evaluate(()=>document.querySelector('#s15-stepper .stepper').stepperApi.index()),1);
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s16-flow-frame');
  await page.evaluate(()=>AT.present.prev());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s15-frame1');
  for(let i=0;i<18;i++){
    await page.evaluate(i=>document.querySelector('#s15-stepper .stepper').stepperApi.go(i),i);
    await page.waitForTimeout(100);await fit('optional stage '+i);
    if([2,8,16,17].includes(i))await page.screenshot({path:path.join(shots,'optional-'+i+'.png')});
  }
  // Preserve every progressive stage of the full flowchart and its final receiver.
  await go('s16-flow-frame',0);
  const stageCount=await page.evaluate(()=>document.querySelector('#s16-flow-stepper .stepper').stepperApi.steps.length);
  assert.equal(stageCount,await page.evaluate(()=>ATTENTION_PREVIEW.stages.length));
  assert(stageCount>5);
  for(let i=0;i<stageCount;i++){
    await page.evaluate(i=>document.querySelector('#s16-flow-stepper .stepper').stepperApi.go(i),i);
    await page.waitForTimeout(100);await fit('flowchart stage '+i);
  }
  assert.match(await page.locator('#s16-flow-stepper .flow-classroom').innerText(),/receiver 10/);
  await page.screenshot({path:path.join(shots,'flowchart-complete.png')});
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s16-frame3-routing');
  // Masked and unmasked tables/messages retain the exact original computation.
  const Fmasked=forward(model,model.sentences.river),Funmasked=forward(model,model.sentences.river,{mask:false});
  const maskRow=await page.locator('#s13-mask-row tbody tr').evaluateAll(es=>es.map(e=>[...e.querySelectorAll('td')].map(c=>c.textContent)));
  const display=(r,d)=>r.map(x=>Number.isFinite(x)?x.toFixed(d):'−∞');
  assert.deepEqual(maskRow,[display(Funmasked.S[4],2),model.sentences.river.map((_,j)=>j<=4?'0':'−∞'),display(Fmasked.S[4],2),display(Fmasked.A[4],3)]);
  assert(Math.abs(Fmasked.A[4].slice(0,5).reduce((s,x)=>s+x,0)-1)<1e-12);
  assert(Fmasked.A[4].slice(5).every(x=>x===0));
  assert.match(await page.locator('#s13-river-mask-arithmetic').textContent(),new RegExp(Funmasked.S[4][5].toFixed(2).replace('.','\\.')));
  assert.match(await page.locator('#s13-frame2').innerText(),/match scores/);
  assert.match(await page.locator('#s13-frame2').innerText(),/causal mask/);
  assert.match(await page.locator('#s13-frame2').innerText(),/attention weights/);
  assert.match(await page.locator('#s13-frame2').innerText(),/Softmax each row/);
  for(const id of ['s13-frame2','s13-frame-mask-row']){
    for(const build of [0,1,2,1,0,2]){
      await go(id,build);await fit(id+' build '+build);
      for(const n of [1,2])assert((await page.locator('#'+id+' [data-build="'+n+'"]').evaluateAll(es=>es.map(e=>getComputedStyle(e).visibility))).every(v=>v===(build>=n?'visible':'hidden')));
      await page.screenshot({path:path.join(shots,id+'-'+build+'.png')});
    }
    await page.evaluate(()=>AT.present.next());
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id==='s13-frame2'?'s13-frame-mask-row':'s13-frame3');
  }
  for(const on of [true,false,true]){
    await go('s13-frame3',0);
    await page.evaluate(on=>document.querySelector('#s13-ctl button').set(on),on);
    const F=forward(model,model.sentences.river,{mask:on});
    const cells=await page.locator('#s13-heat .cell[data-r]').evaluateAll(els=>els.map(e=>({r:+e.dataset.r,c:+e.dataset.c,masked:e.classList.contains('is-masked'),text:e.textContent})));
    assert.equal(cells.length,100);
    for(const cell of cells){
      assert.equal(cell.masked,on&&cell.c>cell.r);
      if(!cell.masked)assert.equal(Number(cell.text),Number(F.A[cell.r][cell.c].toFixed(2)));
    }
    await fit('causal mask '+on);
    await go('s13-frame4',0);
    await page.locator('#s13-send').click();
    await page.waitForTimeout(100);
    const sent=(await page.locator('#s13-target .cell').allTextContents()).map(Number);
    assert.deepEqual(sent,F.Mmsg[4].map(x=>Number(x.toFixed(2))));
    await fit('received message '+on);
  }
  // The comparison's control now lives on the table slide, not on a hidden slide.
  await go('s17-weights',0);let previous=null;
  for(const context of ['river','cheque','river']){
    await page.locator('#s17-ctx-'+(context==='river'?'a':'b')).click();
    const rows=await page.locator('#s17-table tbody tr').evaluateAll(els=>els.map(e=>[...e.querySelectorAll('td')].map(c=>c.textContent)));
    const F=forward(model,model.sentences[context]);
    assert.deepEqual(rows[3].slice(0,10).map(Number),F.A[9].map(x=>Number(x.toFixed(2))));
    if(previous)assert.deepEqual(rows.slice(0,3),previous.slice(0,3));
    previous=rows;await fit('comparison '+context);
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),original);
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  for(const id of ['s11-frame-key-calc','s12-frame-bank','s13-frame1','s13-frame2','s13-frame-mask-row','s14-head-arithmetic','s15-stepper','s16-frame3-routing','s17-weights','s18-question8']){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.locator('#'+id).isVisible(),'Reading retains '+id);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Phone width at '+id);
    await page.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert(await page.locator('#s13-training').isVisible());
  assert(await page.locator('#s13-generation').isVisible());
  assert(await page.locator('#s13-prefix-rule').isVisible());
  const tableBox=await page.locator('#s13-frame1 table').boundingBox();
  assert(tableBox.x>=0&&tableBox.x+tableBox.width<=391,'The concrete prefix table fits a phone.');
  assert.deepEqual(errors,[]);
  console.log('PASS: 25-frame tail; companion retention; manual walkthrough navigation and all 18 stages; full '+stageCount+'-stage flowchart; labelled causal-mask definitions, row arithmetic, reveals and tables/messages; synchronized four-rule comparison; unchanged model; phone reading. Screenshots: '+shots);
}finally{await browser.close();}

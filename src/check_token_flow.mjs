// Supplied vectors -> attention -> residual -> prediction -> projection provenance.
// Uses the existing browser runtime and the independent numerical reference.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {forward} from './toy_ref.mjs';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const model=JSON.parse(fs.readFileSync(new URL('toy.json',import.meta.url),'utf8'));
const ref=forward(model,model.sentences.river.slice(0,7));
const rounded=(row,n=2)=>row.map(x=>Number(x.toFixed(n)));
const shots=fs.mkdtempSync('/private/tmp/token-flow-');
const browser=await chromium.launch();
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify(AT.model));
  const rows=selector=>page.locator(selector+' tbody tr').evaluateAll(els=>els.map(e=>[...e.querySelectorAll('td')].map(c=>Number(c.textContent.replaceAll('−','-').trim()))));
  assert.deepEqual(await rows('#s07-query-given'),[rounded(ref.Q[6])]);
  assert.deepEqual(await rows('#s07-keys-given'),ref.K.map(r=>rounded(r)));
  assert.deepEqual(await rows('#s07-vtab'),ref.V.map(r=>rounded(r)));
  for(const sec of ['s07','s08','s09','s10']){
    const math=(await page.locator('#'+sec+' annotation').allTextContents()).join(' ');
    assert(!/W_[QKV]/.test(math),sec+' must use the supplied rows without projection equations.');
    assert(!/W_[QKV]/.test((await page.locator('#'+sec+' pre').allTextContents()).join(' ')));
  }
  const sequence=await page.locator('.frame').evaluateAll(els=>els.map(e=>e.id));
  const index=id=>{const i=sequence.indexOf(id);assert(i>=0,id+' exists');return i;};
  assert(index('s07-frame1')<index('s07-frame-keys'));
  assert(index('s07-frame-keys')<index('s07-frame-values'));
  assert(index('s09-frame-chain')<index('s09-frame-prediction-input'));
  assert(index('s09-frame-prediction-output')<index('s11-frame1'),'Complete the prediction flow before deriving Q/K/V.');
  assert(index('s11-frame1')<index('s11-frame-query-calc'));
  for(let j=0;j<7;j++){
    await page.locator('#s07-rchips button').nth(j).click();
    assert.deepEqual(await rows('#s07-record-key'),[rounded(ref.K[j])]);
    assert.deepEqual(await rows('#s07-record-value'),[rounded(ref.V[j])]);
    const math=(await page.locator('#s07-record-route annotation').allTextContents()).join(' ');
    assert(math.includes('\\vq{q_7}')&&math.includes('\\va{\\alpha_{7,'+(j+1)+'}}\\,\\vv{v_{'+(j+1)+'}}'),'Receiver stays 7; selected source key/value stay paired.');
  }
  await page.evaluate(()=>AT.present.enter());
  async function go(id){
    await page.evaluate(id=>{
      const el=document.getElementById(id),sec=el.closest('.sec');
      AT.present.go(sec.id,[...sec.querySelectorAll('.frame')].indexOf(el)+1,0);
    },id);
    await page.waitForTimeout(150);
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id);
  }
  for(const context of ['river','cheque','river']){
    const F=forward(model,model.sentences[context]);
    await go('s09-frame-prediction-input');
    await page.locator('#s09-frame-prediction-input [data-context="'+context+'"]').click();
    assert.deepEqual(await rows('#s09-prediction-input'),[rounded(F.E[9],3),rounded(F.Delta[9],3)]);
    const footer=await page.locator('#s09-prediction-input tfoot td').allTextContents();
    assert.deepEqual(footer.map(x=>Number(x.replaceAll('−','-'))),rounded(F.Enew[9],3));
    await go('s09-frame-prediction-output');
    const top=F.probs[9].map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p).slice(0,4);
    assert.deepEqual(await rows('#s09-prediction-output'),[rounded(top.map(x=>x.p),3)]);
    const labels=await page.locator('#s09-prediction-output thead th').allTextContents();
    assert.deepEqual(labels.slice(-4),top.map(x=>model.vocab[x.i]));
    assert.equal(await page.locator('#s09 [data-context="'+context+'"][aria-pressed=true]').count(),2,'Both preview controls share the context.');
    await page.screenshot({path:path.join(shots,'prediction-'+context+'.png')});
  }
  for(const [role,field]of [['query','Q'],['key','K'],['value','V']]){
    await go('s11-frame-'+role+'-calc');
    for(const j of [6,5,1,0,6]){
      await page.locator('#s11-frame-'+role+'-calc [data-token="'+j+'"]').click();
      for(const [r,f]of [['query','Q'],['key','K'],['value','V']]){
        const results=await page.locator('#s11-'+r+'-calc .calc-res').allTextContents();
        assert.deepEqual(results.map(x=>Number(x.replaceAll('−','-').replace(/[=≈]/g,'').trim())),rounded(ref[f][j]),'Recover the supplied '+f+' row with the unchanged shared matrix.');
      }
      assert.equal(await page.locator('#s11 .s11-calc-controls [data-token="'+j+'"][aria-pressed=true]').count(),3);
    }
    await page.screenshot({path:path.join(shots,role+'-derivation.png')});
  }
  await go('s07-frame-pair');
  await page.locator('#s07-rchips button').nth(5).click();
  const changedFrames=['s07-frame1','s07-frame-keys','s07-frame-values','s07-frame-pair','s09-frame-prediction-input','s09-frame-prediction-output','s11-frame1','s11-frame-query-calc','s11-frame-key-calc','s11-frame-value-calc'];
  for(const id of changedFrames){
    await go(id);await page.screenshot({path:path.join(shots,id+'.png')});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),original,'All controls preserve the canonical model.');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  for(const id of changedFrames){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone page overflow at '+id);
    await page.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: supplied Q/K/V match reference; seven paired source records; no premature projections; both position-10 predictions; all shared-matrix derivations; model/state retention; phone layout. Screenshots: '+shots);
}finally{await browser.close();}

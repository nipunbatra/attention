// Example-first explanation of sqrt(d_k), with independent numerical checks.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {softmax} from './toy_ref.mjs';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const shots=fs.mkdtempSync('/private/tmp/scaling-examples-');
const ids=['s12-frame-scaling','s12-frame-spread','s12-frame-softmax','s12-frame-bank'];
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} differs from ${b}`);
function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
const browser=await chromium.launch();
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const model=await page.evaluate(()=>JSON.stringify(AT.model));
  const rows=selector=>page.locator(selector+' tbody tr').evaluateAll(es=>es.map(e=>[...e.querySelectorAll('td')].map(c=>c.textContent.trim())));
  const numeric=async selector=>(await rows(selector)).map(r=>r.map(x=>Number(x.replaceAll('−','-'))));
  const q=[-1,1,1,1,-1,-1,-1,1],k=[-1,-1,1,-1,1,1,-1,-1];
  const products=q.map((x,l)=>x*k[l]);
  assert.deepEqual(await numeric('#s12-sign-vectors'),[q,k,products]);
  assert.deepEqual(await numeric('#s12-prefix-scores'),[[2,4,8].map(d=>products.slice(0,d).reduce((a,b)=>a+b,0))]);
  const experiment=await page.locator('#s12-spread-table').evaluate(e=>JSON.parse(e.dataset.experiment));
  assert.equal(experiment.n,3000);assert.equal(experiment.seedBase,2400);
  const expected=[4,16,64,256].map(d=>{
    const r=rng(2400+d),scores=Array.from({length:3000},()=>{
      let score=0;for(let l=0;l<d;l++){const a=r()<.5?-1:1,b=r()<.5?-1:1;score+=a*b;}return score;
    });
    const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
    const sd=Math.sqrt(scores.reduce((s,x)=>s+(x-mean)**2,0)/scores.length);
    return {d,mean,sd,divisor:Math.sqrt(d),scaledSD:sd/Math.sqrt(d)};
  });
  expected.forEach((r,i)=>{
    for(const key of Object.keys(r))near(experiment.rows[i][key],r[key]);
    assert(r.scaledSD>.95&&r.scaledSD<1.05,'Sample spread after scaling stays near one.');
  });
  assert.deepEqual(await numeric('#s12-spread-table'),expected.map(r=>[+r.sd.toFixed(2),r.divisor,+r.scaledSD.toFixed(2)]));
  const raw=[-8,8],scaled=raw.map(x=>x/8),prob=[softmax(raw),softmax(scaled)];
  const sat=await page.locator('#s12-saturation-table').evaluate(e=>JSON.parse(e.dataset.example));
  assert.equal(sat.d,64);assert.deepEqual(sat.scores,[raw,scaled]);
  sat.weights.forEach((r,i)=>r.forEach((x,j)=>near(x,prob[i][j])));
  const satRows=await rows('#s12-saturation-table');
  satRows.forEach((r,i)=>{
    assert.equal(+r[1],i?2:16);
    assert.deepEqual(r.slice(2),prob[i].map(x=>(100*x).toFixed(i?2:5)+'%'));
  });
  const stable=softmax(raw.map(x=>x-Math.max(...raw)));
  stable.forEach((x,j)=>near(x,prob[0][j]));
  assert(Number.isFinite(Math.exp(8)),'This example saturates without exponential overflow.');
  assert(prob[0][1]*(1-prob[0][1])<2e-7);
  assert(prob[1][1]*(1-prob[1][1])/8>.01,'Including the divisor, the local derivative is still much larger here.');
  await page.evaluate(()=>AT.present.enter());
  assert.deepEqual(await page.locator('#s12 .frame').evaluateAll(es=>es.map(e=>e.id)),ids);
  for(let i=0;i<ids.length;i++){
    for(const build of i===3?[0]:[0,1,0,1]){
      await page.evaluate(({i,build})=>AT.present.go('s12',i+1,build),{i,build});await page.waitForTimeout(200);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,ids[i]+' fits');
      const reveals=await page.locator('#'+ids[i]+' [data-build="1"]').evaluateAll(es=>es.map(e=>getComputedStyle(e).visibility));
      assert(reveals.every(v=>v===(build?'visible':'hidden')),'Reveal and reverse all pieces together.');
      await page.screenshot({path:path.join(shots,ids[i]+'-'+build+'.png')});
    }
    if(i<ids.length-1){await page.evaluate(()=>AT.present.next());assert.equal(await page.locator('.frame.is-live').getAttribute('id'),ids[i+1]);}
  }
  await page.evaluate(()=>AT.present.go('s12',2,1));
  const bars=await page.locator('#s12-spread-table td[data-sd]').evaluateAll(es=>es.map(e=>({value:+e.dataset.sd,width:e.querySelector('.s12-sd-bar').getBoundingClientRect().width,track:e.querySelector('.s12-sd-track').getBoundingClientRect().width})));
  bars.forEach(b=>assert(Math.abs(b.width/b.track-b.value/17)<.001,'Bar length uses the common 0–17 scale.'));
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  for(const id of ids){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Tables scroll locally on phones.');
    await page.locator('#'+id).screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),model,'Explanatory experiments do not alter the model.');
  assert.deepEqual(await page.locator('#s12 .katex-error').allTextContents(),[]);assert.deepEqual(errors,[]);
  console.log('PASS: random-vector products, independent 3,000-pair spread calculations, common-scale bars, exact softmax/gradient contrast, stable-shift distinction, reveals/navigation and phone layout. Screenshots: '+shots);
}finally{await browser.close();}

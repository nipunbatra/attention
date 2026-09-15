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
const ids=['s12-frame-scaling','s12-frame-draws','s12-frame-spread','s12-frame-variance','s12-frame-softmax','s12-frame-bank'];
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
  assert.deepEqual(await page.locator('#s12 .frame').evaluateAll(es=>es.map(e=>e.id)),['s12-topic-break',...ids]);
  await page.evaluate(()=>AT.present.go('s12',1,0));
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s12-topic-break');
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),ids[0]);
  async function go(id,build=0){
    await page.evaluate(({id,build})=>{
      const el=document.getElementById(id),section=el.closest('.sec');
      AT.present.go(section.id,[...section.querySelectorAll('.frame')].indexOf(el)+1,build);
    },{id,build});
    await page.waitForTimeout(200);
  }
  await go('s12-frame-draws');
  const readDraws=()=>page.locator('#s12-frame-draws').evaluate(e=>JSON.parse(e.dataset.simulation));
  const initial=await readDraws();
  assert.equal(initial.n,1);
  assert.match(await page.locator('#s12-live-histogram').innerText(),/SD —/);
  for(const [button,n]of [['one',2],['hundred',102],['all',3000]]){
    await page.locator('#s12-draw-'+button).click();
    const actual=await readDraws();assert.equal(actual.n,n);
    for(const [i,d]of [4,16].entries()){
      const random=rng(2400+d),scores=[];let latest;
      for(let a=0;a<n;a++){
        const q=[],k=[];for(let l=0;l<d;l++){q.push(random()<.5?-1:1);k.push(random()<.5?-1:1);}
        latest={q,k,score:q.reduce((sum,x,l)=>sum+x*k[l],0)};scores.push(latest.score);
      }
      if(d===4)assert.deepEqual(actual.latest,latest,'Visible q/k pair produces the latest score.');
      const counts=Array.from({length:17},(_,bin)=>scores.filter(s=>s===bin*2-16).length);
      assert.deepEqual(actual.rows[i].counts,counts);assert.equal(counts.reduce((a,b)=>a+b),n);
      const mean=scores.reduce((a,b)=>a+b,0)/n;
      near(actual.rows[i].mean,mean);
      near(actual.rows[i].sd,Math.sqrt(scores.reduce((s,x)=>s+(x-mean)**2,0)/n));
      if(n===3000)near(actual.rows[i].sd,experiment.rows[i].sd);
    }
    const bars=await page.locator('#s12-live-histogram rect').evaluateAll(es=>es.map(e=>({d:+e.dataset.width,score:+e.dataset.score,count:+e.dataset.count,height:+e.getAttribute('height'),x:+e.getAttribute('x')})));
    const max=Math.max(...actual.rows.flatMap(r=>r.counts));
    bars.forEach(bar=>{
      assert.equal(bar.count,actual.rows.find(r=>r.d===bar.d).counts[(bar.score+16)/2]);
      near(bar.height,64*bar.count/max);near(bar.x,205+(bar.score+16)*25);
    });
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Live experiment fits at '+n+' draws.');
    await page.screenshot({path:path.join(shots,'simulation-'+n+'.png')});
  }
  for(const id of ['one','hundred','all'])assert(await page.locator('#s12-draw-'+id).isDisabled());
  await page.locator('#s12-draw-reset').click();assert.deepEqual(await readDraws(),initial,'Reset replays the same draws.');
  await page.screenshot({path:path.join(shots,'simulation-reset.png')});
  await page.locator('#s12-draw-one').click();await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s12-frame-spread');
  await go('s12-frame-draws');assert.equal((await readDraws()).n,2,'Navigation preserves the simulation.');
  // Exact enumeration supplies an independent check of the displayed variance argument.
  for(const d of [4,16]){
    const scores=Array.from({length:2**d},(_,bits)=>Array.from({length:d},(_,l)=>(bits>>l)&1?1:-1).reduce((a,b)=>a+b));
    near(scores.reduce((a,b)=>a+b,0)/scores.length,0);
    near(scores.reduce((s,x)=>s+x*x,0)/scores.length,d);
    near(scores.reduce((s,x)=>s+(x/Math.sqrt(d))**2,0)/scores.length,1);
  }
  assert.match(await page.locator('#s12-frame-variance').innerText(),/squared distance.*variance is 1/s);
  assert.match(await page.locator('#s12-frame-variance').innerText(),/independent products, the variances add/);
  assert.match(await page.locator('#s12-frame-variance').innerText(),/Learned vectors need not obey/);
  for(let i=0;i<ids.length;i++){
    for(const build of ['s12-frame-draws','s12-frame-bank'].includes(ids[i])?[0]:[0,1,0,1]){
      await go(ids[i],build);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,ids[i]+' fits');
      const reveals=await page.locator('#'+ids[i]+' [data-build="1"]').evaluateAll(es=>es.map(e=>getComputedStyle(e).visibility));
      assert(reveals.every(v=>v===(build?'visible':'hidden')),'Reveal and reverse all pieces together.');
      if(ids[i]==='s12-frame-variance'){
        const leaves=await page.locator('#s12-frame-variance [data-build="1"] .katex-html *').evaluateAll(es=>es.filter(e=>!e.children.length&&e.textContent.trim()).map(e=>getComputedStyle(e).visibility));
        assert(leaves.every(v=>v===(build?'visible':'hidden')),'Square roots and subscripts follow forward and reverse reveals.');
      }
      await page.screenshot({path:path.join(shots,ids[i]+'-'+build+'.png')});
    }
    if(i<ids.length-1){await page.evaluate(()=>AT.present.next());assert.equal(await page.locator('.frame.is-live').getAttribute('id'),ids[i+1]);}
  }
  await go('s12-frame-spread',1);
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
  console.log('PASS: live draw/batch/reset simulation, exact histograms and summary agreement, enumerated variance derivation, independent 3,000-pair calculations, common-scale bars, softmax/gradient contrast, reveals/navigation and phone layout. Screenshots: '+shots);
}finally{await browser.close();}

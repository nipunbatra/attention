// node src/check_pooling_flow.mjs [attention.html]
// Concrete rows and a visible result must precede the pooling notation.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const dir of fs.readdirSync(cache))candidates.push(path.join(cache,dir,'node_modules/playwright'));
let pw,PNG;
for(const candidate of candidates){try{
  pw=require(candidate);
  const localRequire=createRequire(require.resolve(candidate));
  const core=path.dirname(localRequire.resolve('playwright-core/package.json'));
  PNG=localRequire(path.join(core,'lib/utilsBundle.js')).PNG;
  break;
}catch{pw=undefined;}}
assert(pw,'Use an existing Playwright installation; do not install another dependency.');
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-pooling-flow-'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
const close=(actual,expected)=>{
  assert.equal(actual.length,expected.length);
  actual.forEach((x,i)=>assert(Math.abs(x-expected[i])<1e-12,`${x} != ${expected[i]}`));
};
const weighted=(rows,weights)=>rows[0].map((_,c)=>rows.reduce((s,row,j)=>s+row[c]*weights[j],0));
const go=async(section,frame,build)=>{
  await page.evaluate(([s,f,b])=>AT.present.go(s,f,b),[section,frame,build]);
  await page.waitForTimeout(100);
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,`${section}/${frame}/${build} must fit.`);
};
try{
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs}));
  const E=await page.evaluate(()=>AT.embed(AT.sentences.river.slice(0,7)));
  assert.equal(E.length,7);assert(E.every(row=>row.length===4));
  const order=await page.locator('#s03 .frame').evaluateAll(els=>els.map(e=>e.id));
  assert.equal(order.length,7);
  assert.deepEqual(order.slice(-2),['s03-frame-window-cost','s03-frame-pooling-bridge'],'Show the cost before proposing pooling.');
  assert.equal(await page.locator('#s03-frame-changing-clues').count(),0);
  assert.equal(await page.locator('#s03-frame3').getAttribute('class'),'companion','Keep the redundant slot recap in the article only.');
  assert(!(await page.locator('#s03').textContent()).includes('later toy query'),'Do not use query results before teaching queries.');
  assert(!(await page.locator('#s03 script').allTextContents()).join('').includes('AT.forward('));
  await page.evaluate(()=>AT.present.enter());
  for(const build of [0,1]){
    await go('s03',7,build);
    assert.equal(await page.locator('.pool-bridge-row[data-build="1"]').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
    const bridge=await page.evaluate(()=>({
      intro:document.querySelector('.pool-bridge-intro').textContent,
      chips:[...document.querySelectorAll('#s03-pool-bridge-chips .chip-t')].map(e=>e.textContent),
      active:[...document.querySelectorAll('#s03-pool-bridge-chips .is-active')].map(e=>e.dataset.i),
      blanks:document.querySelectorAll('#s03-pool-bridge-chips .is-slot').length,
      blocks:[...document.querySelectorAll('.pool-bridge-block')].map(e=>[Number(e.dataset.position),e.children.length]),
      tops:[...document.querySelectorAll('.pool-bridge-block span')].map(e=>Math.round(e.getBoundingClientRect().top)),
      summary:document.querySelector('#s03-pool-bridge-summary').children.length,
      shape:document.querySelector('#s03-pool-bridge-concat-shape').textContent
    }));
    assert(bridge.intro.includes('Before prediction')&&bridge.intro.includes('not a word to guess'));
    assert.deepEqual(bridge.chips,['The','fisherman','sat','beside','the','river','bank']);
    assert.deepEqual(bridge.active,['6']);assert.equal(bridge.blanks,0);
    assert.deepEqual(bridge.blocks,Array.from({length:7},(_,i)=>[i+1,4]));
    assert.equal(new Set(bridge.tops).size,1,'Concatenation is a single row, not a stacked matrix.');
    assert.equal(bridge.summary,4);assert(bridge.shape.startsWith('1 × 28'));
    await page.screenshot({path:path.join(shots,`bridge-${build}.png`)});
  }
  await go('s04',1,0);
  const slider=page.locator('#s04-islider input');
  assert.equal(await slider.inputValue(),'7');
  for(let i=1;i<=7;i++){
    await slider.fill(String(i));
    const weights=Array.from({length:7},(_,j)=>j<i?1/i:0),expected=weighted(E,weights);
    const mean=await page.evaluate(()=>{
      const v=document.querySelector('#s04-mean-live .vec');
      return {values:JSON.parse(v.dataset.values),position:Number(v.dataset.position),
        cells:[...v.querySelectorAll('.cell')].map(e=>Number(e.textContent.replace('−','-'))),
        axes:[...v.querySelectorAll('.vec-ax')].map(e=>e.textContent),label:v.querySelector('annotation').textContent,
        scope:document.querySelector('#s04-mean-scope').textContent,shape:document.querySelector('#s04-mean-shape').textContent,
        active:[...document.querySelectorAll('#s04-chips .is-active')].map(e=>Number(e.dataset.i)),
        muted:[...document.querySelectorAll('#s04-chips .is-muted')].map(e=>Number(e.dataset.i)),
        footer:document.querySelector('#s04-mean-tab tfoot th annotation').textContent,
        labelColour:getComputedStyle(v.querySelector('.vec-label')).color,
        mathColour:getComputedStyle(document.querySelector('.pool-equation .katex-html .pool-summary')).color};
    });
    close(mean.values,expected);close(mean.cells,expected.map(x=>Number(x.toFixed(2))));
    assert.equal(mean.position,i);assert.equal(mean.label,`m_{${i}}`);assert.equal(mean.footer,`m_{${i}}`);
    assert.deepEqual(mean.axes,['water','finance','person','glue']);
    assert.deepEqual(mean.active,[i-1]);assert.deepEqual(mean.muted,Array.from({length:7-i},(_,j)=>i+j));
    assert(mean.scope.includes(`i = ${i}`)&&mean.shape.includes('one 1 × 4'));
    assert.equal(mean.labelColour,mean.mathColour,'Match the result and equation symbol colours.');
    for(const build of [0,1]){
      await go('s04',1,build);
      assert.equal(await page.locator('.pool-equation').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
      for(const selector of ['.op-symbol','.frac-line'])assert.equal(await page.locator('.pool-equation '+selector).evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden','Every equation glyph follows the reveal, including after it has been shown once.');
      assert.equal(await slider.inputValue(),String(i));
    }
    if([3,7].includes(i)){
      const shot=await page.screenshot({path:path.join(shots,`mean-${i}.png`)});
      const box=await page.locator('.pool-equation .katex-html .op-symbol').boundingBox();
      const png=PNG.sync.read(shot);let ink=0;
      for(let y=Math.ceil(box.y);y<Math.floor(box.y+box.height);y++)for(let x=Math.ceil(box.x);x<Math.floor(box.x+box.width);x++){
        const p=(y*png.width+x)*4;
        if(png.data[p]<100&&png.data[p+1]<100&&png.data[p+2]<100)ink++;
      }
      assert(ink>150,'The summation must actually paint after repeated slider and build changes.');
    }
    await go('s04',2,1);
    await go('s04',1,0);
    assert.equal(await slider.inputValue(),String(i),'Retain the mean position across slide navigation.');
  }
  await go('s04',2,1);
  await page.screenshot({path:path.join(shots,'mean-table.png')});
  await go('s04',1,0);
  await page.evaluate(()=>AT.present.next());
  await page.waitForTimeout(100);
  assert.equal(await page.locator('.pool-equation').evaluate(e=>getComputedStyle(e).visibility),'visible','The normal Next action reveals the equation.');
  await go('s04',3,1);
  const formula=await page.locator('#s04 .frame').nth(2).locator('annotation').allTextContents();
  assert(formula.some(t=>t.includes('{m_i}')&&t.includes('j \\le i')));
  const presets={equal:[1,1,1,1,1,1,1],river:[.3,1.2,.3,.8,.3,4,.6],fisherman:[.3,4,.3,.6,.3,1,.6],self:[0,0,0,0,0,0,1]};
  await go('s04',4,0);
  async function checkWeighted(raw){
    const total=raw.reduce((a,b)=>a+b,0),weights=raw.map(x=>total?x/total:1/7);
    const expected=weighted(E,weights);
    const result=await page.locator('#s04-live-pool .vec').evaluate(v=>({
      values:JSON.parse(v.dataset.values),weights:JSON.parse(v.dataset.weights),
      cells:[...v.querySelectorAll('.cell')].map(e=>Number(e.textContent.replace('−','-'))),label:v.querySelector('annotation').textContent
    }));
    close(result.values,expected);close(result.weights,weights);close(result.cells,expected.map(x=>Number(x.toFixed(2))));
    assert.equal(result.label,'m_7');
  }
  for(const [preset,raw]of Object.entries(presets)){
    await page.locator(`#s04-presets [data-preset="${preset}"]`).click();
    await checkWeighted(raw);
    await go('s04',5,1);await go('s04',4,0);await checkWeighted(raw);
    if(['river','self'].includes(preset))await page.screenshot({path:path.join(shots,`weighted-${preset}.png`)});
  }
  for(let j=0;j<7;j++)await page.locator(`#s04-w${j}`).fill('0');
  await checkWeighted(Array(7).fill(0));
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs})),original,'Pooling controls never update model parameters.');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await page.locator('#s04-presets [data-preset="equal"]').click();
  for(const [id,name]of [['s03-frame-pooling-bridge','phone-bridge'],['s04-frame-mean','phone-mean']]){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone document overflow.');
    await page.screenshot({path:path.join(shots,name+'.png')});
  }
  const tops=await page.locator('.pool-bridge-block span').evaluateAll(els=>els.map(e=>Math.round(e.getBoundingClientRect().top)));
  assert.equal(new Set(tops).size,1,'The phone concatenation strip scrolls locally instead of wrapping into a matrix.');
  const phoneVector=await page.locator('#s04-mean-live .vec').boundingBox();
  assert(phoneVector.x>=0&&phoneVector.x+phoneVector.width<=391,'The complete four-coordinate mean fits on a phone.');
  assert.deepEqual(errors,[]);
  console.log(`PASS: pooling sequence, 7 mean positions, 4 weighted presets, zero-weight fallback, retained state, colour/shape semantics and phone layout. Screenshots: ${shots}`);
}finally{await browser.close();}

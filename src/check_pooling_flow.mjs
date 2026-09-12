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
const goPool=async(id,build=0)=>{
  const frame=await page.locator('#'+id).evaluate(e=>[...e.closest('.sec').querySelectorAll('.frame')].indexOf(e)+1);
  assert(frame>0,'Resolve the teaching frame by its stable id.');
  await go('s04',frame,build);
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
  assert.deepEqual(await page.locator('#s04 .frame').evaluateAll(els=>els.slice(0,3).map(e=>e.id)),['s04-frame-prefix','s04-frame-mean','s04-frame-mean-table'],'Explain available rows before averaging them.');
  await goPool('s04-frame-prefix');
  const positionSlider=page.locator('#s04-prefix-slider input');
  const slider=page.locator('#s04-islider input');
  for(let i=1;i<=7;i++){
    await positionSlider.fill(String(i));
    const view=await page.evaluate(()=>{
      const lane=id=>{
        const root=document.getElementById(id);
        return {active:[...root.querySelectorAll('.is-active')].map(e=>Number(e.dataset.i)),
          muted:[...root.querySelectorAll('.is-muted')].map(e=>Number(e.dataset.i)),
          current:[...root.querySelectorAll('.is-current')].map(e=>Number(e.dataset.i)),
          words:[...root.querySelectorAll('.is-active .chip-t')].map(e=>e.textContent)};
      };
      return {window:lane('s04-window-chips'),prefix:lane('s04-prefix-chips'),
        data:{...document.getElementById('s04-frame-prefix').dataset},
        windowScope:document.getElementById('s04-window-scope').textContent,
        prefixScope:document.getElementById('s04-prefix-scope').textContent,
        takeaway:document.getElementById('s04-prefix-takeaway').textContent,
        labels:[...document.querySelectorAll('#s04 .position-control .slider-label')].map(e=>e.textContent)};
    });
    const prefix=Array.from({length:i},(_,j)=>j),window=prefix.slice(-3),all=Array.from({length:7},(_,j)=>j);
    assert.deepEqual(view.prefix.active,prefix);assert.deepEqual(view.window.active,window);
    assert.deepEqual(view.prefix.muted,all.filter(j=>!prefix.includes(j)));assert.deepEqual(view.window.muted,all.filter(j=>!window.includes(j)));
    for(const lane of [view.prefix,view.window]){assert.deepEqual(lane.current,[i-1]);assert(lane.active.every(j=>j<i),'Neither rule uses a later token.');}
    assert.equal(view.data.position,String(i));assert.equal(view.data.windowWidth,'3','Moving the current position keeps window width fixed.');
    assert.deepEqual(view.labels,['Current token position i','Current token position i']);
    assert.equal(await slider.inputValue(),String(i),'The mean control follows the comparison position.');
    assert(view.prefixScope.includes('all '+i+' token'),'The full prefix grows with the current position.');
    if(i<7)assert(view.takeaway.includes('cannot contribute yet'));
    if(i===7){
      assert.deepEqual(view.window.words,['the','river','bank']);
      assert.deepEqual(view.prefix.words,['The','fisherman','sat','beside','the','river','bank']);
      assert(view.windowScope.includes('5–7')&&view.prefixScope.includes('1–7'));
      assert(view.takeaway.includes('keeps fisherman available'),'Explain why removing the old cutoff matters.');
    }
    await goPool('s04-frame-prefix');
    if([2,4,7].includes(i))await page.screenshot({path:path.join(shots,`prefix-${i}.png`)});
    await goPool('s04-frame-mean');await goPool('s04-frame-prefix');
    assert.equal(await positionSlider.inputValue(),String(i),'Keep the position across comparison/arithmetic navigation.');
  }
  await positionSlider.focus();await page.keyboard.press('ArrowLeft');
  assert.equal(await positionSlider.inputValue(),'6');assert.equal(await slider.inputValue(),'6');
  await page.keyboard.press('End');
  await goPool('s04-frame-mean');
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
    assert.deepEqual(mean.active,Array.from({length:i},(_,j)=>j));assert.deepEqual(mean.muted,Array.from({length:7-i},(_,j)=>i+j));
    assert.equal(await positionSlider.inputValue(),String(i),'The arithmetic control also updates the comparison.');
    assert(mean.scope.includes(`i = ${i}`)&&mean.shape.includes('one 1 × 4'));
    assert.equal(mean.labelColour,mean.mathColour,'Match the result and equation symbol colours.');
    for(const build of [0,1]){
      await goPool('s04-frame-mean',build);
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
    await goPool('s04-frame-mean-table',1);
    await goPool('s04-frame-mean');
    assert.equal(await slider.inputValue(),String(i),'Retain the mean position across slide navigation.');
  }
  await goPool('s04-frame-mean-table',1);
  await page.screenshot({path:path.join(shots,'mean-table.png')});
  await goPool('s04-frame-mean');
  await page.evaluate(()=>AT.present.next());
  await page.waitForTimeout(100);
  assert.equal(await page.locator('.pool-equation').evaluate(e=>getComputedStyle(e).visibility),'visible','The normal Next action reveals the equation.');
  const poolOrder=await page.locator('#s04 .frame').evaluateAll(els=>els.map(e=>e.id));
  assert.deepEqual(poolOrder.slice(3,6),['s04-frame-weight-rule','s04-frame-weight-sum','s04-frame-choose'],'Explain one scalar contribution, then the sum, then let students choose all weights.');
  await goPool('s04-frame-weight-rule',1);
  const example=await page.locator('#s04-frame-weight-rule').evaluate(root=>{
    const table=root.querySelector('.alpha-table');
    return {text:root.querySelector('.alpha-example').textContent,
      data:{...table.dataset},
      rows:[...table.querySelectorAll('tbody tr')].map(tr=>({values:JSON.parse(tr.dataset.values),cells:[...tr.querySelectorAll('td')].map(e=>Number(e.textContent.replace('−','-')))})),
      axes:[...table.querySelectorAll('thead th')].slice(1).map(e=>e.textContent),
      source:[...root.querySelectorAll('.chip.is-active')].map(e=>Number(e.dataset.i)),
      receiver:[...root.querySelectorAll('.chip.is-receiver')].map(e=>Number(e.dataset.i)),
      colours:{weight:getComputedStyle(root.querySelector('.alpha-symbol .m-a')).color,
        weightText:getComputedStyle(root.querySelector('.alpha-indices .pool-weight')).color,
        source:getComputedStyle(table.querySelector('.alpha-source .m-e')).color,
        sourceCell:getComputedStyle(table.querySelector('.alpha-source td')).color,
        sourceText:getComputedStyle(root.querySelector('.alpha-indices .pool-input')).color}};
  });
  assert.equal(example.data.weight,'0.5');assert.equal(example.data.source,'6');assert.equal(example.data.receiver,'7');
  assert.deepEqual(example.axes,['water','finance','person','glue']);
  assert.deepEqual(example.source,[5]);assert.deepEqual(example.receiver,[6]);
  close(example.rows[0].values,E[5]);close(example.rows[1].values,E[5].map(x=>0.5*x));
  for(const row of example.rows)close(row.cells,row.values.map(x=>Number(x.toFixed(2))));
  assert(example.text.includes('First index 7: receiver bank')&&example.text.includes('Second index 6: source river'));
  assert(example.text.includes('chosen share')&&example.text.includes('other six weights must add up to'));
  assert.equal(example.colours.weight,example.colours.weightText);
  assert.equal(example.colours.source,example.colours.sourceCell);assert.equal(example.colours.source,example.colours.sourceText);
  for(const id of ['s04-frame-weight-rule','s04-frame-weight-sum']){
    for(const build of [0,1,0,1]){
      await goPool(id,build);
      const visibility=await page.locator('#'+id+' .alpha-reveal').evaluate(root=>({root:getComputedStyle(root).visibility,
        glyphs:[...root.querySelectorAll('.katex-html .m-a,.katex-html .m-e,.katex-html .op-symbol,.katex-html .pool-summary')].map(e=>getComputedStyle(e).visibility)}));
      assert.equal(visibility.root,build?'visible':'hidden');
      assert(visibility.glyphs.length>0&&visibility.glyphs.every(v=>v===(build?'visible':'hidden')),`Every coloured term follows the reveal, including after revisiting: ${id}/${build} ${JSON.stringify(visibility)}`);
      await page.screenshot({path:path.join(shots,`${id}-${build}.png`)});
    }
  }
  const formula=await page.locator('#s04-frame-weight-sum annotation').allTextContents();
  assert(formula.some(t=>t.includes('{m_7}')&&t.includes('\\alpha_{7,6}')&&t.includes('\\ve{e_6}')));
  assert(formula.some(t=>t.includes('{m_i}')&&t.includes('\\sum_{j=1}^{i}')));
  assert(formula.some(t=>t.includes('\\alpha_{ij}\\geq0'))&&formula.some(t=>t.includes('\\sum_{j=1}^{i}')&&t.includes('=1')),'State nonnegative weights normalized over sources for a fixed receiver.');
  const legend=await page.locator('#s04-frame-weight-sum').evaluate(root=>{
    const colour=selector=>getComputedStyle(root.querySelector(selector)).color;
    return {summary:[colour('.alpha-reveal .katex-html .pool-summary'),colour('.alpha-legend .pool-summary')],
      weight:[colour('.alpha-reveal .katex-html .m-a'),colour('.alpha-legend .pool-weight')],
      input:[colour('.alpha-reveal .katex-html .m-e'),colour('.alpha-legend .pool-input')],
      widths:[...root.querySelectorAll('.alpha-legend p')].map(e=>({content:e.scrollWidth,box:e.clientWidth}))};
  });
  for(const role of ['summary','weight','input'])assert.equal(...legend[role],`Match the ${role} symbol to its prose.`);
  assert(new Set(['summary','weight','input'].map(role=>legend[role][0])).size===3,'Summary, scalar weights and input rows have distinct colours.');
  assert(legend.widths.every(w=>w.content<=w.box+1),'Colour definitions must not overlap neighbouring columns.');
  const sumGlyph=await page.locator('#s04-frame-weight-sum .alpha-reveal .katex-display .op-symbol').first().boundingBox();
  const sumPNG=PNG.sync.read(await page.screenshot());let sumInk=0;
  for(let y=Math.ceil(sumGlyph.y);y<Math.floor(sumGlyph.y+sumGlyph.height);y++)for(let x=Math.ceil(sumGlyph.x);x<Math.floor(sumGlyph.x+sumGlyph.width);x++){
    const p=(y*sumPNG.width+x)*4;if(sumPNG.data[p]<100&&sumPNG.data[p+1]<100&&sumPNG.data[p+2]<100)sumInk++;
  }
  assert(sumInk>150,'The weighted summation actually paints after its reveal.');
  const presets={equal:[1,1,1,1,1,1,1],river:[.3,1.2,.3,.8,.3,4,.6],fisherman:[.3,4,.3,.6,.3,1,.6],self:[0,0,0,0,0,0,1]};
  await goPool('s04-frame-choose');
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
    await goPool('s04-frame-weighted-table',1);await goPool('s04-frame-choose');await checkWeighted(raw);
    if(['river','self'].includes(preset))await page.screenshot({path:path.join(shots,`weighted-${preset}.png`)});
  }
  for(let j=0;j<7;j++)await page.locator(`#s04-w${j}`).fill('0');
  await checkWeighted(Array(7).fill(0));
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs})),original,'Pooling controls never update model parameters.');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await page.locator('#s04-presets [data-preset="equal"]').click();
  for(const [id,name]of [['s03-frame-pooling-bridge','phone-bridge'],['s04-frame-prefix','phone-prefix'],['s04-frame-mean','phone-mean'],['s04-frame-weight-rule','phone-alpha-example'],['s04-frame-weight-sum','phone-alpha-sum']]){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone document overflow.');
    await page.screenshot({path:path.join(shots,name+'.png')});
  }
  const tops=await page.locator('.pool-bridge-block span').evaluateAll(els=>els.map(e=>Math.round(e.getBoundingClientRect().top)));
  assert.equal(new Set(tops).size,1,'The phone concatenation strip scrolls locally instead of wrapping into a matrix.');
  const phoneVector=await page.locator('#s04-mean-live .vec').boundingBox();
  assert(phoneVector.x>=0&&phoneVector.x+phoneVector.width<=391,'The complete four-coordinate mean fits on a phone.');
  const phoneExample=await page.locator('#s04-alpha-example-table').boundingBox();
  assert(phoneExample.x>=0&&phoneExample.x+phoneExample.width<=391,'All four coordinates of the worked alpha term fit on a phone.');
  for(const control of [positionSlider,slider])assert((await control.boundingBox()).width>160,'Both current-position sliders have usable phone tracks.');
  assert.deepEqual(errors,[]);
  console.log(`PASS: all 7 window/prefix positions and means, concrete alpha indices and scalar products, matched colours and painted reveals, 4 weighted presets, zero-weight fallback, retained state and phone layout. Screenshots: ${shots}`);
}finally{await browser.close();}

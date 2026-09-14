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
const model=JSON.parse(fs.readFileSync(new URL('toy.json',import.meta.url),'utf8'));
// Independent affine/softmax calculation: never use the browser's AT.head as the oracle.
const referenceHead=c=>{
  const logits=model.b_vocab.map((bias,j)=>bias+c.reduce((sum,x,k)=>sum+x*model.W_vocab[k][j],0));
  const max=Math.max(...logits),exps=logits.map(x=>Math.exp(x-max)),total=exps.reduce((a,b)=>a+b,0);
  return {logits,probs:exps.map(x=>x/total),winners:logits.flatMap((x,j)=>Math.abs(x-max)<1e-12?[j]:[])};
};
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
  assert.deepEqual(order,['s03-frame2','s03-frame-concatenate','s03-frame1','s03-frame-boundary'],'End the window section at its limitation, before the new summary topic.');
  for(const id of ['s03-frame-window-head','s03-frame-window-cost'])assert.equal(await page.locator('#'+id).getAttribute('class'),'companion','Keep the alternative linear-head derivation in the article only.');
  assert.equal(await page.locator('#s03-frame-changing-clues').count(),0);
  assert.equal(await page.locator('#s03-frame3').getAttribute('class'),'companion','Keep the redundant slot recap in the article only.');
  assert(!(await page.locator('#s03').textContent()).includes('later toy query'),'Do not use query results before teaching queries.');
  assert(!(await page.locator('#s03 script').allTextContents()).join('').includes('AT.forward('));
  await page.evaluate(()=>AT.present.enter());
  await go('s03',4,1);
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s04-frame-summary-break','Normal Next reaches the summary divider before the bridge.');
  assert.equal(await page.locator('#s04>.sec-head').evaluate(e=>getComputedStyle(e).display),'none','Show one title on the divider.');
  assert((await page.locator('.summary-break h3').evaluate(e=>parseFloat(getComputedStyle(e).fontSize)))>=64);
  assert.deepEqual(await page.locator('.summary-approaches li strong').allTextContents(),['Averaging.','Hand-chosen weights.','Attention.']);
  assert.equal(await page.locator('.summary-approaches').evaluate(e=>getComputedStyle(e).listStyleType),'decimal');
  const summaryLabels=await page.evaluate(()=>[
    document.querySelector('#s04').dataset.title,document.querySelector('#s04 .sec-head h2').textContent,
    window.__PART__.chain.find(x=>x.section==='s04').label,window.__PART__.sections.find(x=>x.id==='s04').title]);
  assert(summaryLabels.every(t=>t==='Summarizing the prefix'),'The new section name matches its navigation.');
  for(const [id,title]of [['s04-frame-mean','1. Averaging: equal shares'],['s04-frame-weight-rule','2. Hand-chosen weights'],['s05-frame-attention-break','3. Attention']]){
    assert.equal(await page.locator('#'+id).getAttribute('data-title'),title,'Carry the roadmap number into each approach introduction.');
  }
  await goPool('s04-frame-summary-break');
  await page.screenshot({path:path.join(shots,'summary-break.png')});
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s04-frame-pooling-bridge');
  assert.notEqual(await page.locator('#s04>.sec-head').evaluate(e=>getComputedStyle(e).display),'none','Restore the usual frame title after the divider.');
  for(const build of [0,1]){
    await goPool('s04-frame-pooling-bridge',build);
    assert.equal(await page.locator('.pool-bridge-row[data-build="1"]').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
    const bridge=await page.evaluate(()=>({
      intro:document.querySelector('.pool-bridge-intro').textContent,
      chips:[...document.querySelectorAll('#s04-pool-bridge-chips .chip-t')].map(e=>e.textContent),
      active:[...document.querySelectorAll('#s04-pool-bridge-chips .is-active')].map(e=>e.dataset.i),
      blanks:document.querySelectorAll('#s04-pool-bridge-chips .is-slot').length,
      blocks:[...document.querySelectorAll('.pool-bridge-block')].map(e=>[Number(e.dataset.position),e.children.length]),
      tops:[...document.querySelectorAll('.pool-bridge-block span')].map(e=>Math.round(e.getBoundingClientRect().top)),
      summary:document.querySelector('#s04-pool-bridge-summary').children.length,
      shape:document.querySelector('#s04-pool-bridge-concat-shape').textContent
    }));
    assert(bridge.intro.includes('Before prediction')&&bridge.intro.includes('not a word to guess'));
    assert.deepEqual(bridge.chips,['The','fisherman','sat','beside','the','river','bank']);
    assert.deepEqual(bridge.active,['6']);assert.equal(bridge.blanks,0);
    assert.deepEqual(bridge.blocks,Array.from({length:7},(_,i)=>[i+1,4]));
    assert.equal(new Set(bridge.tops).size,1,'Concatenation is a single row, not a stacked matrix.');
    assert.equal(bridge.summary,4);assert(bridge.shape.startsWith('1 × 28'));
    await page.screenshot({path:path.join(shots,`bridge-${build}.png`)});
  }
  assert.deepEqual(await page.locator('#s04 .frame').evaluateAll(els=>els.slice(0,5).map(e=>e.id)),['s04-frame-summary-break','s04-frame-pooling-bridge','s04-frame-prefix','s04-frame-mean','s04-frame-mean-table'],'Introduce the topic and the fixed-width goal, then explain available rows before averaging them.');
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
  assert.deepEqual(poolOrder.slice(5,9),['s04-frame-weight-motivation','s04-frame-weight-rule','s04-frame-weight-sum','s04-frame-choose'],'Motivate unequal contributions before the scalar example, the sum and the interactive weights.');
  await goPool('s04-frame-mean-table',1);
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s04-frame-weight-motivation','The example-led motivation follows the mean worksheet naturally.');
  const motivation=await page.locator('#s04-frame-weight-motivation').evaluate(root=>({
    words:[...root.querySelectorAll('.clue-sentence [data-position]')].map(e=>e.textContent),
    positions:[...root.querySelectorAll('.clue-sentence [data-position]')].map(e=>Number(e.dataset.position)),
    clues:[...root.querySelectorAll('.clue-sentence mark')].map(e=>({word:e.textContent,weight:getComputedStyle(e.querySelector('strong')).fontWeight,colour:getComputedStyle(e).color,background:getComputedStyle(e).backgroundColor})),
    receiver:root.querySelector('.clue-receiver').textContent,
    receiverColour:getComputedStyle(root.querySelector('.clue-receiver')).color,
    text:root.querySelector('.weight-motivation').textContent,
    math:root.querySelectorAll('.katex').length,
    inputColour:getComputedStyle(root.querySelector('.clue-reading .pool-input')).color,
    summaryColour:getComputedStyle(root.querySelector('.weight-motivation .pool-summary')).color
  }));
  assert.deepEqual(motivation.words,await page.evaluate(()=>AT.sentences.river.slice(0,7)));
  assert.deepEqual(motivation.positions,[1,2,3,4,5,6,7]);
  assert.deepEqual(motivation.clues.map(c=>c.word),['fisherman','river']);
  for(const clue of motivation.clues){assert(Number(clue.weight)>=700);assert.equal(clue.colour,motivation.inputColour);assert.notEqual(clue.background,'rgba(0, 0, 0, 0)');}
  assert.equal(motivation.receiver,'bank');assert.equal(motivation.receiverColour,motivation.summaryColour);
  assert(motivation.text.includes('riverside or a financial institution')&&motivation.text.includes('Should they contribute equally'));
  assert(motivation.text.includes('unequal shares')&&motivation.text.includes('choose the shares ourselves'));
  assert.equal(motivation.math,0,'Motivate the idea with words before showing a formula.');
  assert(!/alpha|α|query|0\.5/i.test(motivation.text),'Do not introduce the later notation or numerical share here.');
  for(const build of [0,1,2,0,2]){
    await goPool('s04-frame-weight-motivation',build);
    for(const [selector,step]of [['.clue-reading',1],['.clue-proposal',2]])assert.equal(await page.locator(selector).evaluate(e=>getComputedStyle(e).visibility),build>=step?'visible':'hidden');
    assert(await page.locator('.clue-sentence').isVisible());
    await page.screenshot({path:path.join(shots,`weight-motivation-${build}.png`)});
  }
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s04-frame-weight-rule','Only after proposing unequal shares do we introduce the alpha example.');
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
  assert.deepEqual(await page.locator('#s04 .frame').evaluateAll(els=>els.slice(-4).map(e=>e.id)),['s04-frame-choose','s04-frame-weighted-table','s04-frame-preset-predictions','s04-frame-weight-question'],'Build the context, inspect its contributions, compare cases, then ask how to compute weights.');
  for(const build of [0,1,0,1]){
    await goPool('s04-frame-choose',build);
    assert.equal(await page.locator('#s04-prediction').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
    assert.equal(await page.locator('#s04-live-pool').evaluate(e=>getComputedStyle(e).visibility),'visible','Show the context before its prediction.');
  }
  const readResult=el=>({values:JSON.parse(el.dataset.values),weights:JSON.parse(el.dataset.weights),
    logits:JSON.parse(el.dataset.logits),probs:JSON.parse(el.dataset.probs),winners:JSON.parse(el.dataset.winners),position:el.dataset.position});
  const checkHead=(actual,expected)=>{
    close(actual.logits,expected.logits);close(actual.probs,expected.probs);
    assert.equal(actual.logits.length,20,'Softmax includes the whole vocabulary, not just the displayed winners.');
    assert(Math.abs(actual.probs.reduce((a,b)=>a+b,0)-1)<1e-12);
    assert.deepEqual(actual.winners,expected.winners);assert.equal(actual.position,'8');
  };
  async function checkWeighted(raw){
    const total=raw.reduce((a,b)=>a+b,0),weights=raw.map(x=>total?x/total:1/7);
    const expected=weighted(E,weights);
    const result=await page.locator('#s04-live-pool .vec').evaluate(v=>({
      values:JSON.parse(v.dataset.values),weights:JSON.parse(v.dataset.weights),
      cells:[...v.querySelectorAll('.cell')].map(e=>Number(e.textContent.replace('−','-'))),label:v.querySelector('annotation').textContent
    }));
    close(result.values,expected);close(result.weights,weights);close(result.cells,expected.map(x=>Number(x.toFixed(2))));
    assert.equal(result.label,'m_7');
    const head=referenceHead(expected),actual=await page.locator('#s04-prediction').evaluate(readResult);
    close(actual.values,expected);close(actual.weights,weights);checkHead(actual,head);
    const winner=head.winners[0],tied=head.winners.length>1;
    assert.equal(await page.locator('#s04-prediction-word').textContent(),tied?head.winners.length+' words tied for first':'Top word: '+model.vocab[winner]);
    assert((await page.locator('#s04-prediction-score').textContent()).includes(head.logits[winner].toFixed(3)));
    assert((await page.locator('#s04-prediction-prob').textContent()).includes(head.probs[winner].toFixed(3)));
    const products=await page.locator('#s04-w-tab .dt-fig').evaluate(e=>JSON.parse(e.dataset.contributions));
    const expectedProducts=E.map((row,j)=>row.map(x=>x*weights[j]));
    close(products.flat(),expectedProducts.flat());
    const originals=await page.locator('#s04-w-tab .dt-fig').evaluate(e=>JSON.parse(e.dataset.inputs));
    close(originals.flat(),E.flat());
    const originalCells=await page.locator('#s04-w-tab tbody td[data-c]').allTextContents();
    close(originalCells.map(s=>Number(s.replace('−','-'))),E.flat().map(x=>Number(x.toFixed(2))));
    const shares=await page.locator('#s04-w-tab tbody td.dt-lead').allTextContents();
    close(shares.map(Number),weights.map(x=>Number(x.toFixed(3))));
    const displayed=await page.locator('#s04-w-tab tbody td[data-k]').allTextContents();
    close(displayed.map(s=>Number(s.replace('−','-'))),expectedProducts.flat().map(x=>Number(x.toFixed(2))));
    const footer=await page.locator('#s04-w-tab tfoot td:not(.dt-lead)').allTextContents();
    close(footer.map(s=>Number(s.replace('−','-'))),expected.map(x=>Number(x.toFixed(2))));
    const selected=Object.entries(presets).filter(([,values])=>values.every((x,j)=>Math.abs(x-raw[j])<1e-12)).map(([key])=>key);
    for(const id of ['s04-presets','s04-contribution-presets']){
      assert.deepEqual(await page.locator('#'+id+' [aria-pressed="true"]').evaluateAll(els=>els.map(e=>e.dataset.preset)),selected);
    }
  }
  for(const [preset,raw]of Object.entries(presets)){
    await goPool('s04-frame-choose',1);
    await page.locator(`#s04-presets [data-preset="${preset}"]`).click();
    await checkWeighted(raw);
    await page.screenshot({path:path.join(shots,`weighted-${preset}.png`)});
    await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s04-frame-weighted-table');
    await page.locator('#s04-contribution-presets [data-preset="equal"]').click();
    await page.locator(`#s04-contribution-presets [data-preset="${preset}"]`).click();
    for(const build of [0,1,2,0,2]){
      await goPool('s04-frame-weighted-table',build);await checkWeighted(raw);
      const originalVisibility=await page.locator('#s04-w-tab tbody td[data-c], #s04-w-tab tbody .dt-lead').evaluateAll(els=>els.map(e=>getComputedStyle(e).visibility));
      assert(originalVisibility.every(v=>v==='visible'),'Original rows and their shares stay visible in all three stages.');
      const productVisibility=await page.locator('#s04-w-tab [data-build="1"]').evaluateAll(els=>els.map(e=>getComputedStyle(e).visibility));
      assert(productVisibility.every(v=>v===(build>=1?'visible':'hidden')),'Reveal weighted rows after originals.');
      assert.equal(await page.locator('#s04-w-tab tfoot tr').evaluate(e=>getComputedStyle(e).visibility),build>=2?'visible':'hidden','Add the weighted columns only on the third stage.');
      const layout=await page.locator('#s04-w-tab tbody tr').evaluateAll(rows=>rows.map(row=>{
        const left=row.querySelector('[data-c="0"]').getBoundingClientRect();
        const right=row.querySelector('[data-k="0"]').getBoundingClientRect();
        return {left:left.x,right:right.x,dy:Math.abs(left.y-right.y)};
      }));
      assert(layout.every(r=>r.right>r.left&&r.dy<1),'Keep corresponding original and weighted coordinates on the same source row.');
      await page.screenshot({path:path.join(shots,`contributions-${preset}-${build}.png`)});
    }
    await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s04-frame-preset-predictions','The completed sum leads to the existing prediction comparison.');
    await goPool('s04-frame-choose',1);await checkWeighted(raw);
  }
  await page.locator('#s04-w0').focus();await page.keyboard.press('ArrowRight');
  await checkWeighted([.1,0,0,0,0,0,1]);
  for(let j=0;j<7;j++)await page.locator(`#s04-w${j}`).fill('0');
  await checkWeighted(Array(7).fill(0));
  await page.locator('#s04-w4').fill('1');
  const onlyThe=[0,0,0,0,1,0,0];await checkWeighted(onlyThe);
  assert((await page.locator('#s04-prediction-word').textContent()).includes('2 words tied'),'Do not arbitrarily call the first tied word the winner.');
  const caseRows=page.locator('#s04-preset-predictions tbody tr');
  assert.equal(await caseRows.count(),4);
  for(const [preset,raw]of Object.entries(presets)){
    const weights=raw.map(w=>w/raw.reduce((a,b)=>a+b,0)),c=weighted(E,weights),h=referenceHead(c);
    const caseRow=page.locator('#s04-preset-predictions tr[data-preset="'+preset+'"]');
    const actual=await caseRow.evaluate(readResult);
    close(actual.values,c);close(actual.weights,weights);checkHead(actual,h);
    assert.equal(await caseRow.locator('.word-cell').textContent(),preset==='self'?'teller':'water');
    assert.equal(await caseRow.locator('.pool-score').textContent(),h.logits[h.winners[0]].toFixed(3));
    assert.equal(await caseRow.locator('.pool-prob').textContent(),h.probs[h.winners[0]].toFixed(3));
    close((await caseRow.locator('.summary-cell').allTextContents()).map(Number),c.map(x=>Number(x.toFixed(2))));
  }
  for(const build of [0,1,0,1]){
    await goPool('s04-frame-preset-predictions',build);
    assert((await caseRows.locator('.summary-cell').evaluateAll(els=>els.map(e=>getComputedStyle(e).visibility))).every(v=>v==='visible'));
    assert((await page.locator('#s04-preset-predictions [data-build="1"]').evaluateAll(els=>els.map(e=>getComputedStyle(e).visibility))).every(v=>v===(build?'visible':'hidden')));
    await page.screenshot({path:path.join(shots,`preset-predictions-${build}.png`)});
  }
  for(const id of ['s04-frame-choose','s04-frame-preset-predictions']){
    const note=await page.locator('#'+id+' > .prediction-note').last().textContent();
    assert(note.includes('untrained')&&note.includes('and'),'Explain why this head is not a reliable language predictor.');
  }
  await goPool('s04-frame-choose',1);await checkWeighted(onlyThe);
  const headColours=await page.locator('#s04-frame-preset-predictions').evaluate(root=>{
    const c=s=>getComputedStyle(root.querySelector(s)).color;
    return ['summary','param','score','prob'].map(role=>[c('.katex-html .pool-'+role),c('.prediction-note .pool-'+role)]);
  });
  headColours.forEach(pair=>assert.equal(...pair,'Match the head equation colours to their explanation.'));
  const searchOrder=await page.locator('#s05 .frame').evaluateAll(els=>els.map(e=>e.id));
  assert.deepEqual(searchOrder.slice(0,4),['s05-frame-attention-break','s05-frame-attention-idea','s05-frame-retrieval-detour','s05-frame-search'],'Name and explain attention before the retrieval analogy or query terminology.');
  const visibleCopy=async(id)=>page.locator('#'+id).evaluate(root=>{
    const copy=root.cloneNode(true);copy.querySelectorAll('script,.companion').forEach(e=>e.remove());return copy.textContent;
  });
  assert(!(await visibleCopy('s04-frame-weight-question')).includes('Attention is a way to build that function'));
  for(const id of searchOrder.slice(0,2))assert(!/\b(query|keys?|values?)\b/i.test(await visibleCopy(id)),'Introduce the mechanism and purpose before retrieval jargon.');
  const definition=await visibleCopy('s05-frame-attention-idea');
  assert(definition.includes('bank at position 7')&&definition.includes('river'));
  assert(definition.includes('Higher scores get larger shares')&&definition.includes('add up to 1')&&definition.includes('fixed-width context summary'));
  assert(definition.includes('A trained model')&&definition.includes('A new input can produce different weights'));
  const detour=await visibleCopy('s05-frame-retrieval-detour');
  assert(detour.includes('Information retrieval')&&/finding useful information in a collection/i.test(detour));
  assert(detour.includes('A short detour')&&detour.includes('determine the weights')&&detour.includes('return to bank and river'));
  assert.deepEqual(await page.locator('.retrieval-roles dt').allTextContents(),['Query','Key','Value']);
  assert.deepEqual(await page.locator('.retrieval-roles dd').allTextContents(),['What am I looking for?','A description of each source, used for matching.','The information that source returns.']);
  assert.equal(await page.locator('#s05-frame-retrieval-detour .katex-error').count(),0);
  const goSearch=async(id,build=0)=>{
    const index=searchOrder.indexOf(id);assert(index>=0);await go('s05',index+1,build);
  };
  await goPool('s04-frame-weight-question',2);
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-attention-break','Normal Next crosses the new section break.');
  assert.equal(await page.locator('#s05>.sec-head').evaluate(e=>getComputedStyle(e).display),'none','The section break has a single prominent title.');
  assert((await page.locator('.attention-break h3').evaluate(e=>parseFloat(getComputedStyle(e).fontSize)))>=64);
  await page.screenshot({path:path.join(shots,'attention-break.png')});
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-attention-idea');
  assert.notEqual(await page.locator('#s05>.sec-head').evaluate(e=>getComputedStyle(e).display),'none','The normal title returns after the section break.');
  for(const build of [0,1,2,0,2]){
    await goSearch('s05-frame-attention-idea',build);
    const steps=await page.locator('.attention-steps li').evaluateAll(els=>els.map(e=>getComputedStyle(e).visibility));
    assert.deepEqual(steps,['visible',build>=1?'visible':'hidden',build>=2?'visible':'hidden']);
    assert.equal(await page.locator('.attention-learning').evaluate(e=>getComputedStyle(e).visibility),build>=2?'visible':'hidden');
    await page.screenshot({path:path.join(shots,`attention-idea-${build}.png`)});
  }
  const introColours=await page.evaluate(()=>{
    const colour=selector=>getComputedStyle(document.querySelector(selector)).color;
    return ['input','weight','summary'].map(role=>[colour('#s05 .attention-'+role),colour('#s04 .pool-'+role)]);
  });
  introColours.forEach(pair=>assert.equal(...pair,'Keep the pooling colour meanings in the attention introduction.'));
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-retrieval-detour');
  await goSearch('s05-frame-retrieval-detour');
  assert.equal(await page.locator('#s05>.sec-head').evaluate(e=>getComputedStyle(e).display),'none','The detour is a visible section break, not a duplicate heading.');
  assert((await page.locator('.retrieval-intro h3').evaluate(e=>parseFloat(getComputedStyle(e).fontSize)))>=64);
  await page.screenshot({path:path.join(shots,'retrieval-detour.png')});
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-search');
  await page.locator('#s05-qbtns button').nth(1).click();
  await goSearch('s05-frame-results');
  assert.equal(await page.locator('#s05-vgrid .is-top .vt').textContent(),'Regularisation','The overfitting request still ranks the matching source first.');
  await goSearch('s05-frame-return');
  assert((await page.locator('#s05-return-title').textContent()).includes('Regularisation'),'Continuation frames retain the chosen query and source/value pairing.');
  await goSearch('s05-frame-attention-break');
  await goSearch('s05-frame-retrieval-detour');
  await goSearch('s05-frame-results');
  assert.equal(await page.locator('#s05-vgrid .is-top .vt').textContent(),'Regularisation','Visiting the new introduction does not masquerade as entering the search demo.');
  await goSearch('s05-frame-search');
  assert((await page.locator('#s05-frame-search .s05-qtext').textContent()).includes('gradient information backwards'),'Re-entering the original search frame keeps its original query-reset behaviour.');
  const sectionLabels=await page.evaluate(()=>[
    document.querySelector('#s05').dataset.title,document.querySelector('#s05 .sec-head h2').textContent,
    window.__PART__.chain.find(x=>x.section==='s05').label,window.__PART__.sections.find(x=>x.id==='s05').title]);
  assert(sectionLabels.every(t=>t==='Attention and the search analogy'),'Navigation and article heading match the broader section contents.');
  const recordsOrder=await page.locator('#s07 .frame').evaluateAll(els=>els.map(e=>e.id));
  assert.deepEqual(recordsOrder.slice(0,2),['s07-frame-return-to-sentence','s07-frame1'],'Return to the concrete sentence before projection formulas.');
  const temperatureFrame=await page.locator('#s06-frame-temperature').evaluate(e=>[...e.closest('.sec').querySelectorAll('.frame')].indexOf(e)+1);
  await go('s06',temperatureFrame,1);
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('#s07 .frame.is-live').getAttribute('id'),'s07-frame-return-to-sentence','The retrieval example ends with an explicit return transition.');
  const returnCopy=await visibleCopy('s07-frame-return-to-sentence');
  assert(returnCopy.includes('bank, position 7')&&returnCopy.includes('river, position 6')&&returnCopy.includes('tokens, not videos'));
  assert(returnCopy.includes('every available')&&returnCopy.includes('normalize')&&returnCopy.includes("river's share for bank"));
  assert.equal((await page.locator('.return-prefix').textContent()).trim(),'The fisherman sat beside the river bank');
  assert.equal(await page.locator('#s07-frame-return-to-sentence annotation').evaluateAll(els=>els.some(e=>/W_[QKV]/.test(e.textContent))),false,'Defer projection equations to the next frame.');
  for(const build of [0,1,0,1]){
    await go('s07',1,build);
    assert.equal(await page.locator('#s07>.sec-head').evaluate(e=>getComputedStyle(e).display),'none');
    assert.equal(await page.locator('.return-roles').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
    await page.waitForTimeout(250); // Let the reveal opacity settle before inspecting subscript pixels.
    await page.screenshot({path:path.join(shots,`return-to-sentence-${build}.png`)});
  }
  const retrievalColours=await page.evaluate(()=>{
    const c=s=>getComputedStyle(document.querySelector(s)).color;
    return ['q','k','v'].map(role=>[c('.retrieval-roles .sym-'+role),c('.return-roles .sym-'+role),c('.return-roles .katex-html .m-'+role),c('#s05-frame-three-jobs .obj-'+role+' .big')]);
  });
  retrievalColours.forEach(colours=>colours.slice(1).forEach(c=>assert.equal(c,colours[0],'Match each retrieval role across text, math and the sentence example.')));
  assert.equal(await page.locator('.retrieval-route .m-a').first().evaluate(e=>getComputedStyle(e).color),await page.locator('.return-result .m-a').first().evaluate(e=>getComputedStyle(e).color));
  await page.evaluate(()=>AT.present.next());await page.waitForTimeout(100);
  assert.equal(await page.locator('#s07 .frame.is-live').getAttribute('id'),'s07-frame1');
  assert.notEqual(await page.locator('#s07>.sec-head').evaluate(e=>getComputedStyle(e).display),'none');
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs})),original,'Pooling controls never update model parameters.');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.locator('.summary-break h3').evaluate(e=>getComputedStyle(e).display),'none','In the article, use the section heading instead of repeating the same divider title.');
  assert.notEqual(await page.locator('#s04>.sec-head').evaluate(e=>getComputedStyle(e).display),'none');
  await page.locator('#s04-presets [data-preset="equal"]').click();
  for(const [id,name]of [['s04-frame-summary-break','phone-summary-break'],['s04-frame-pooling-bridge','phone-bridge'],['s04-frame-prefix','phone-prefix'],['s04-frame-mean','phone-mean'],['s04-frame-weight-motivation','phone-weight-motivation'],['s04-frame-weight-rule','phone-alpha-example'],['s04-frame-weight-sum','phone-alpha-sum'],['s04-frame-choose','phone-weighted-prediction'],['s04-frame-weighted-table','phone-contributions'],['s04-frame-preset-predictions','phone-preset-predictions'],['s05-frame-attention-break','phone-attention-break'],['s05-frame-attention-idea','phone-attention-idea'],['s05-frame-retrieval-detour','phone-retrieval-detour'],['s07-frame-return-to-sentence','phone-return-to-sentence']]){
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
  const phoneLive=await page.locator('#s04-live-pool .vec').boundingBox();
  assert(phoneLive.x>=0&&phoneLive.x+phoneLive.width<=391,'The complete live summary fits on a phone.');
  const phoneComparison=await page.locator('.preset-table-wrap').evaluate(e=>({width:e.clientWidth,scroll:e.scrollWidth}));
  assert(phoneComparison.scroll>phoneComparison.width,'The four-case table scrolls locally on narrow screens.');
  const phoneContributions=await page.locator('#s04-w-tab .dt-scroll').evaluate(e=>({width:e.clientWidth,scroll:e.scrollWidth}));
  assert(phoneContributions.scroll>phoneContributions.width,'The original/product comparison scrolls locally on narrow screens.');
  assert.equal(await page.locator('#s04-w-tab tbody td[data-c]').count(),28);
  assert.equal(await page.locator('#s04-w-tab tbody td[data-k]').count(),28);
  for(const control of [positionSlider,slider])assert((await control.boundingBox()).width>160,'Both current-position sliders have usable phone tracks.');
  assert.deepEqual(errors,[]);
  console.log(`PASS: pooling flow, 7 prefix positions, alpha arithmetic, original/product/sum reveals for 4 synchronized presets, independent 20-word head predictions, four-case comparison, ties/custom/zero-weight states, colour/reveal/state retention, attention/retrieval detour/return-to-sentence flow and phone layout. Screenshots: ${shots}`);
}finally{await browser.close();}

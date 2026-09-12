// node src/check_window_network.mjs [attention.html]
// The window control selects real token rows; its network is an architecture
// sketch, not a family of trained models or a source of predicted probabilities.
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
  // Reuse Playwright's own PNG reader for the visible-notation regression.
  PNG=localRequire(path.join(core,'lib/utilsBundle.js')).PNG;
  break;
}catch{pw=undefined;}}
assert(pw,'Use an existing Playwright installation; do not install another dependency.');
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-window-network-'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
try{
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const initial=await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs}));
  await page.evaluate(()=>{AT.present.enter();AT.present.go('s03',2,0);});
  const slider=page.locator('#s03-slider input');
  for(let w=1;w<=10;w++){
    await slider.fill(String(w));
    const result=await page.evaluate(()=>{
      const root=document.getElementById('s03-window-network'),E=AT.embed(AT.sentences.river);
      return {
        active:[...document.querySelectorAll('#s03-chips .is-active')].map(e=>Number(e.dataset.i)+1),
        muted:[...document.querySelectorAll('#s03-chips .is-muted')].map(e=>Number(e.dataset.i)+1),
        boundary:document.querySelector('#s03-chips .bound')?.nextElementSibling?.dataset.i,
        diagrams:[...root.querySelectorAll('svg')].map(svg=>({
          data:{...svg.dataset},label:svg.getAttribute('aria-label'),
          groups:[...svg.querySelectorAll('.token-group')].map(g=>({
            position:Number(g.dataset.tokenPosition),token:g.dataset.token,
            values:[...g.querySelectorAll('circle')].map(c=>Number(c.dataset.value)),
            expected:E[Number(g.dataset.tokenPosition)-1],
            coordinates:[...g.querySelectorAll('circle')].map(c=>Number(c.dataset.coordinate))
          })),
          edges:[...svg.querySelectorAll('.connection')].map(e=>[Number(e.dataset.sourcePosition),Number(e.dataset.coordinate),e.dataset.outputWord]),
          outputs:[...svg.querySelectorAll('.output-node')].map(e=>e.dataset.outputWord)
        })),
        dimensions:document.getElementById('s03-window-dimensions').textContent,
        concatenated:document.querySelector('#s03-ctab tfoot').textContent,
        animated:getComputedStyle(root.querySelector('.token-group')).animationName
      };
    });
    const inside=Array.from({length:w},(_,i)=>11-w+i),outside=Array.from({length:10-w},(_,i)=>i+1);
    assert.deepEqual(result.active,inside);
    assert.deepEqual(result.muted,outside);
    assert.equal(result.boundary,w<10?String(10-w):undefined);
    assert.equal(result.animated,'none','Respect reduced-motion preferences.');
    assert(result.dimensions.includes(`${w} tokens × 4 numbers = ${w*4} inputs`));
    assert(result.dimensions.includes(`W: ${w*4} × 20 = ${w*80} weights`));
    assert(result.concatenated.includes(`${w} × 4 = ${w*4} numbers`));
    for(const diagram of result.diagrams){
      const expectedShown=w<=3?inside:[inside[0],inside[1],10];
      assert.deepEqual(diagram.groups.map(g=>g.position),expectedShown);
      for(const group of diagram.groups){assert.deepEqual(group.values,group.expected);assert.deepEqual(group.coordinates,[1,2,3,4]);}
      assert.deepEqual(diagram.outputs,['the','water','teller','money']);
      assert.deepEqual(diagram.edges,expectedShown.flatMap(i=>[1,2,3,4].flatMap(j=>diagram.outputs.map(word=>[i,j,word]))));
      for(const [key,value] of Object.entries({window:w,inputCount:w*4,outputCount:20,omittedInputs:Math.max(w-3,0)*4,weightRows:w*4,weightCols:20,weightCount:w*80,biasCount:20}))assert.equal(Number(diagram.data[key]),value,`${key} at w=${w}`);
      assert(diagram.label.includes('No predictions are computed.'));
    }
    for(const build of [0,1]){
      await page.evaluate(build=>AT.present.go('s03',2,build),build);
      await page.waitForTimeout(50);
      assert.equal(await slider.inputValue(),String(w),'Do not reset the chosen window on a build.');
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,`Window ${w}, build ${build} must fit.`);
    }
    const bounds=await page.evaluate(()=>{
      const svg=document.querySelector('.window-net-wide'),v=svg.viewBox.baseVal;
      return [...svg.querySelectorAll('text')].filter(el=>{const b=el.getBBox();return b.x<0||b.y<0||b.x+b.width>v.width+1||b.y+b.height>v.height+1;}).map(e=>e.textContent);
    });
    assert.deepEqual(bounds,[],'Keep all SVG labels within the diagram.');
    if([1,3,5,10].includes(w))await page.screenshot({path:path.join(shots,`window-${w}.png`)});
    await page.evaluate(()=>AT.present.go('s03',3,0));
    assert.equal(await page.locator('.concat-after').evaluate(e=>getComputedStyle(e).visibility),'hidden','Show the stacked rows before revealing concatenation.');
    const concat=await page.evaluate(()=>{
      const flat=document.getElementById('s03-concat-flat'),w=Number(flat.dataset.window),tokens=AT.sentences.river;
      const table=document.querySelector('#s03-ctab table');
      return {
        data:{...flat.dataset},expected:AT.embed(tokens).slice(-w).flat(),
        before:document.getElementById('s03-concat-before-shape').textContent,
        after:document.getElementById('s03-concat-after-shape').textContent,
        scope:document.getElementById('s03-concat-scope').textContent,
        definition:document.querySelector('.concat-definition').textContent,
        footer:table.querySelector('tfoot th').textContent,
        headers:[...table.querySelectorAll('thead th')].map(e=>e.textContent),
        shownRows:[...table.querySelectorAll('tbody tr:not(.s03-window-hidden):not(.s03-window-ellipsis)')].map(e=>Number(e.querySelector('th').textContent.match(/^\d+/)[0])),
        pieces:[...flat.querySelectorAll('.concat-piece')].map(e=>Number(e.dataset.tokenPosition)),
        entries:[...flat.querySelectorAll('.concat-value')].map(e=>({value:Number(e.dataset.value),expected:AT.embed(tokens)[Number(e.dataset.position)-1][Number(e.dataset.coordinate)-1]})),
        equation:document.querySelector('#s03-ceq annotation').textContent,
        colours:['concat-name','concat-position'].map(cls=>{
          const root=document.querySelector('.concat-definition');
          return getComputedStyle(root.querySelector('.katex-html .'+cls)).color===getComputedStyle(root.querySelector('p .'+cls)).color;
        })
      };
    });
    assert.equal(concat.data.rows,'1');assert.equal(Number(concat.data.columns),w*4);
    assert.deepEqual(JSON.parse(concat.data.values),concat.expected,'Concatenation preserves every coordinate and token order.');
    assert.equal(concat.before,`Before: ${w} × 4 (token rows × coordinates)`);
    assert.equal(concat.after,`After concatenation: 1 × ${w*4} (one row)`);
    assert(concat.scope.includes('predicting position 11'));
    assert(concat.definition.includes('the concatenated context row')&&concat.definition.includes('the last input position')&&concat.definition.includes('stays 10'));
    assert.equal(concat.footer,'before concatenation','Do not label the stacked table as c_10.');
    assert.deepEqual(concat.headers,['token','water','finance','person','glue']);
    assert.deepEqual(concat.shownRows,concat.pieces,'Use matching token blocks before and after.');
    assert(concat.entries.every(e=>e.value===e.expected));
    assert.equal(concat.entries.length+Number(concat.data.omittedCoordinates),w*4);
    assert(concat.equation.includes(`1\\times ${w*4}`),'Use an explicit row-vector shape.');
    assert(concat.colours.every(Boolean),'Match the c and subscript definitions to the equation colours.');
    for(const build of [0,1]){
      await page.evaluate(build=>AT.present.go('s03',3,build),build);
      await page.waitForTimeout(200);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,`Concatenation w=${w}, build=${build} must fit.`);
      if(build===1){
        assert.equal(await page.locator('.concat-after').evaluate(e=>getComputedStyle(e).visibility),'visible');
        const tops=await page.locator('.concat-value').evaluateAll(els=>els.map(e=>Math.round(e.getBoundingClientRect().top)));
        assert.equal(new Set(tops).size,1,'All joined coordinates must visibly lie on one row.');
      }
      if([3,10].includes(w)){
        const screenshot=await page.screenshot({path:path.join(shots,`concat-${w}-${build}.png`)});
        const box=await page.locator('.concat-definition').boundingBox();
        const png=PNG.sync.read(screenshot);
        let blue=0,orange=0;
        for(let y=Math.ceil(box.y);y<Math.floor(box.y+box.height);y++)for(let x=Math.ceil(box.x);x<Math.floor(box.x+box.width);x++){
          const i=(y*png.width+x)*4,[r,g,b]=png.data.subarray(i,i+3);
          if(b>r*1.3&&b>g*1.2)blue++;
          if(r>g*1.5&&g>b*1.5)orange++;
        }
        assert(blue>50&&orange>20,`The c/10 explanation must actually paint at w=${w}, build=${build}, not merely exist in the DOM.`);
      }
    }
    await page.evaluate(()=>AT.present.go('s03',2,1));
  }
  await slider.fill('3');await slider.focus();await page.keyboard.press('ArrowRight');
  assert.equal(await slider.inputValue(),'4');
  await page.evaluate(()=>AT.present.go('s03',3,0));
  await page.evaluate(()=>AT.present.go('s03',2,1));
  assert.equal(await slider.inputValue(),'4','Keep the same window when navigating to the concatenation table and back.');
  await page.evaluate(()=>AT.present.go('s03',1,0));
  for(const w of [1,3,5,100]){
    await page.locator('#s03-net-slider input').fill(String(w));
    const label=await page.locator('#s03-net svg').getAttribute('aria-label');
    assert(label.includes(`${w*4} scalar input nodes`),'The preceding MLP must also count scalar coordinates, not tokens.');
    assert((await page.locator('#s03-net .caps').textContent()).includes(`${w*4} input numbers`));
    assert((await page.locator('#s03-net-read').innerText()).includes(`${w*4} input numbers`));
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow);
  }
  await page.evaluate(()=>AT.present.go('s03',1,2));
  await page.screenshot({path:path.join(shots,'mlp-100.png')});
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await slider.fill('5');
  // The shared SVG sketches reposition their HTML math labels on resize.
  await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth+1,{},{timeout:3000});
  const phone=await page.evaluate(()=>{
    const svg=document.querySelector('.window-net-compact'),v=svg.viewBox.baseVal;
    const input=document.querySelector('#s03-slider input');
    return {
      compact:getComputedStyle(svg).display,wide:getComputedStyle(document.querySelector('.window-net-wide')).display,
      slider:input.getBoundingClientRect().width,
      overflow:document.documentElement.scrollWidth>innerWidth+1,
      labels:[...svg.querySelectorAll('text')].filter(e=>{const b=e.getBBox();return b.x<0||b.y<0||b.x+b.width>v.width+1||b.y+b.height>v.height+1;}).map(e=>e.textContent)
    };
  });
  assert.equal(phone.compact,'block');assert.equal(phone.wide,'none');
  assert(phone.slider>=180,'The phone control needs a usable track, not a collapsed thumb.');
  assert(!phone.overflow,'No phone document overflow.');assert.deepEqual(phone.labels,[]);
  await page.locator('#s03-frame2').screenshot({path:path.join(shots,'phone-window-5.png')});
  const rowScroll=await page.locator('.concat-row-scroll').evaluate(e=>({width:e.clientWidth,content:e.scrollWidth,overflow:getComputedStyle(e).overflowX}));
  assert.equal(rowScroll.overflow,'auto');assert(rowScroll.content>rowScroll.width,'A long joined row should scroll locally on a phone, never wrap into a matrix.');
  await page.locator('#s03-frame-concatenate').screenshot({path:path.join(shots,'phone-concat-5.png')});
  const final=await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs}));
  assert.equal(final,initial,'Architecture controls must not mutate the trained toy or its predictions.');
  assert.deepEqual(errors,[]);
  console.log('PASS: ten windows, scalar nodes/edges, stacked-to-concatenated values and shapes, notation definitions, progressive reveal, fixed outputs, keyboard, retained state, reduced motion, MLP consistency, phone layout, and model immutability.');
  console.log('Screenshots: '+shots);
}finally{await browser.close();}

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
    await page.evaluate(()=>AT.present.go('s03',4,0));
    const headSlider=page.locator('#s03-head-slider input');
    assert.equal(await headSlider.inputValue(),String(w),'Both window controls share the same choice.');
    const head=await page.evaluate(()=>{
      const root=document.getElementById('s03-head-network'),svg=root.querySelector('svg');
      return {
        data:{...svg.dataset},label:svg.getAttribute('aria-label'),
        groups:[...svg.querySelectorAll('.head-token-group')].map(g=>({position:Number(g.dataset.tokenPosition),values:[...g.querySelectorAll('circle')].map(c=>Number(c.dataset.value)),expected:AT.embed(AT.sentences.river)[Number(g.dataset.tokenPosition)-1]})),
        edges:[...svg.querySelectorAll('.head-edge')].map(e=>[Number(e.dataset.sourcePosition),Number(e.dataset.coordinate),e.dataset.outputWord]),
        outputs:[...svg.querySelectorAll('.head-output')].map(e=>e.dataset.outputWord),
        equation:document.querySelector('#s03-head-equation annotation').textContent,
        count:Number(document.getElementById('s03-head-count').dataset.parameters),
        context:document.getElementById('s03-head-context').textContent,
        colours:['window-input','window-param','window-score'].map(cls=>{
          const eq=document.querySelector('#s03-head-equation .katex-html .'+cls),label=svg.querySelector('text.'+cls),prose=document.querySelector('.window-head-math p.'+cls);
          return [eq,label,prose].map(e=>getComputedStyle(e).color);
        })
      };
    });
    const shown=w<=3?inside:[inside[0],inside[1],10];
    assert.deepEqual(head.groups.map(g=>g.position),shown);
    for(const group of head.groups)assert.deepEqual(group.values,group.expected);
    assert.deepEqual(head.outputs,['the','water','teller','money']);
    assert.deepEqual(head.edges,shown.flatMap(i=>[1,2,3,4].flatMap(j=>head.outputs.map(word=>[i,j,word]))));
    for(const [key,value]of Object.entries({window:w,inputCount:w*4,outputCount:20,omittedInputs:Math.max(w-3,0)*4,omittedOutputs:16,weightRows:w*4,weightCols:20,weightCount:w*80,biasCount:20}))assert.equal(Number(head.data[key]),value);
    assert(head.label.includes('No trained predictions are computed.'));
    assert(head.context.includes('10 = last input position; predict position 11.'));
    assert(head.equation.includes(`c_{10}}_{1\\times ${w*4}}`)&&head.equation.includes(`W}_{${w*4}\\times 20}`),'The coloured equation uses this exact window and explicit row shapes.');
    assert(head.colours.every(values=>new Set(values).size===1),'Match diagram, equation, and definition colours.');
    assert.equal(head.count,w*80+20);
    for(const build of [0,1,2]){
      await page.evaluate(build=>AT.present.go('s03',4,build),build);
      await page.waitForTimeout(100);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,`Head w=${w}, build=${build} must fit.`);
      assert.equal(await page.locator('#s03-head-equation').evaluate(e=>getComputedStyle(e).visibility),build===0?'hidden':'visible');
      assert.equal(await page.locator('.window-head-math>[data-build="2"]').evaluate(e=>getComputedStyle(e).visibility),build<2?'hidden':'visible');
      assert.equal(await headSlider.inputValue(),String(w));
    }
    for(const word of head.outputs){
      await page.locator(`#s03-head-network [role="button"][data-output-word="${word}"]`).click();
      const trace=await page.evaluate(()=>({
        pressed:[...document.querySelectorAll('#s03-head-network [aria-pressed="true"]')].map(e=>e.dataset.outputWord),
        edges:[...document.querySelectorAll('#s03-head-network .head-edge.is-traced')].map(e=>e.dataset.outputWord),
        note:document.getElementById('s03-head-trace').textContent
      }));
      assert.deepEqual(trace.pressed,[word]);
      assert.deepEqual(trace.edges,Array(shown.length*4).fill(word),'One output highlights exactly one column of W, not all columns.');
      assert(trace.note.startsWith(`${word}: ${w*4} incoming weights + 1 bias.`));
    }
    const headBounds=await page.locator('#s03-head-network svg').evaluate(svg=>{
      const v=svg.viewBox.baseVal;
      return [...svg.querySelectorAll('text')].filter(e=>{const b=e.getBBox();return b.x<0||b.y<0||b.x+b.width>v.width+1||b.y+b.height>v.height+1;}).map(e=>e.textContent);
    });
    assert.deepEqual(headBounds,[],'All interactive head labels fit in the viewBox.');
    if([1,3,5,10].includes(w))await page.screenshot({path:path.join(shots,`head-${w}.png`)});
    await page.evaluate(()=>AT.present.go('s03',2,1));
  }
  await slider.fill('3');await slider.focus();await page.keyboard.press('ArrowRight');
  assert.equal(await slider.inputValue(),'4');
  await page.evaluate(()=>AT.present.go('s03',3,0));
  await page.evaluate(()=>AT.present.go('s03',2,1));
  assert.equal(await slider.inputValue(),'4','Keep the same window when navigating to the concatenation table and back.');
  await page.evaluate(()=>AT.present.go('s03',4,2));
  const headSlider=page.locator('#s03-head-slider input');
  for(let w=1;w<=10;w++){
    await headSlider.fill(String(w));
    assert.equal(await slider.inputValue(),String(w),'Head slider updates the earlier window control.');
    assert.equal(await page.locator('#s03-concat-flat').getAttribute('data-columns'),String(w*4),'Head control also updates the concatenation example.');
    assert.equal(await page.locator('#s03-head-network svg').getAttribute('data-weight-count'),String(w*80));
  }
  await headSlider.focus();await page.keyboard.press('ArrowLeft');
  assert.equal(await headSlider.inputValue(),'9');assert.equal(await slider.inputValue(),'9');
  for(const [word,key]of [['water','Enter'],['teller',' ']]){
    const output=page.locator(`#s03-head-network [role="button"][data-output-word="${word}"]`);
    await output.focus();await page.keyboard.press(key);
    assert.equal(await output.getAttribute('aria-pressed'),'true');
    assert.equal(await page.locator('#s03-head-network [aria-pressed="true"]').count(),1);
    assert.equal((await page.evaluate(()=>AT.present.fitReport())).frame,4,'Tracing with the keyboard must not advance the presentation.');
  }
  await page.evaluate(()=>AT.present.go('s03',3,1));
  await page.evaluate(()=>AT.present.go('s03',4,2));
  assert.equal(await headSlider.inputValue(),'9');
  assert.equal(await page.locator('#s03-head-network').getAttribute('data-traced-word'),'teller');
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
  // Longer prefixes illustrate token visibility, without running out-of-vocab
  // words or positions through the saved ten-position toy model.
  const shortWindow=await slider.inputValue();
  await page.evaluate(()=>AT.present.go('s03',5,0));
  const longSlider=page.locator('#s03-long-slider input');
  assert.equal(await longSlider.inputValue(),'10','Introduce the longer example with ten slots.');
  for(let w=10;w<=20;w++){
    await longSlider.fill(String(w));
    const long=await page.evaluate(()=>({
      examples:['river','cheque'].map(id=>{
        const host=document.getElementById('s03-long-'+id);
        return {
          tokens:JSON.parse(host.dataset.tokens),first:Number(host.dataset.firstPosition),prediction:Number(host.dataset.predictionPosition),
          inside:[...host.querySelectorAll('.is-inside')].map(e=>Number(e.dataset.position)),
          outside:[...host.querySelectorAll('.is-outside')].map(e=>Number(e.dataset.position)),
          clue:{word:host.querySelector('.is-clue').dataset.word,position:Number(host.querySelector('.is-clue').dataset.position),inside:host.querySelector('.is-clue').classList.contains('is-inside')},
          firstMarker:[...host.querySelectorAll('.is-first')].map(e=>Number(e.dataset.position)),
          slotCount:host.querySelectorAll('.long-slot').length
        };
      }),
      result:{...document.getElementById('s03-long-result').dataset},
      cost:{...document.getElementById('s03-long-cost').dataset},
      scope:document.getElementById('s03-long-scope').textContent,
      table:[...document.querySelectorAll('#s03-ktab tbody tr')].map(tr=>[...tr.querySelectorAll('th,td')].map(e=>Number(e.textContent.replaceAll(',','')))),
      notes:document.getElementById('s03-frame-boundary').textContent,
      costNotes:document.getElementById('s03-frame-window-cost').textContent
    }));
    assert.deepEqual(long.examples.map(e=>e.tokens.length),[20,20]);
    assert.deepEqual(long.examples[0].tokens.slice(-10),['after','a','long','quiet','afternoon','turned','around','and','watched','the']);
    assert.deepEqual(long.examples[0].tokens.slice(-10),long.examples[1].tokens.slice(-10),'Both twenty-token prefixes have exactly the same last ten words.');
    for(const example of long.examples){
      assert.deepEqual(example.inside,Array.from({length:w},(_,i)=>21-w+i));
      assert.deepEqual(example.outside,Array.from({length:20-w},(_,i)=>i+1));
      assert.equal(example.first,21-w);assert.equal(example.prediction,21);
      assert.deepEqual(example.firstMarker,[21-w]);assert.equal(example.slotCount,1);
      assert.equal(example.clue.inside,example.clue.position>=21-w);
    }
    assert.deepEqual(long.examples.map(e=>[e.clue.word,e.clue.position]),[['river',6],['cheque',5]]);
    assert.equal(long.result.identical,String(w<=11),'Sunday/yesterday already differ at w=12, before the main clues enter.');
    assert.equal(Number(long.result.cluesAvailable),Number(w>=15)+Number(w>=16));
    assert.equal(Number(long.cost.weights),w*4*20);assert.equal(Number(long.cost.biases),20);assert.equal(Number(long.cost.parameters),w*80+20);
    assert(long.scope.includes(`positions ${21-w}–20`)&&long.scope.includes('Predict 21'));
    assert.deepEqual(long.table,[[10,40,800,20,820],[20,80,1600,20,1620],[100,400,8000,20,8020]]);
    assert(long.notes.includes('Sentences illustrate visibility only')&&long.notes.includes('35-token prefix')&&long.notes.includes('positions 16–35'));
    assert(long.costNotes.includes('linear head only')&&long.costNotes.includes('12.8 million')&&long.costNotes.includes('128 million'));
    assert.equal(10*128*10000,12.8e6);assert.equal(100*128*10000,128e6);
    assert.equal(await slider.inputValue(),shortWindow,'The longer illustration must not resize the saved short-prefix example.');
    for(const frame of [5,8])for(const build of [0,1]){
      await page.evaluate(({frame,build})=>AT.present.go('s03',frame,build),{frame,build});
      await page.waitForTimeout(100);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,`Long-window frame ${frame}, w=${w}, build=${build} fits.`);
      if(frame===5)assert.equal(await page.locator('#s03-long-result').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
      if([10,20].includes(w)&&build===1)await page.screenshot({path:path.join(shots,`long-${w}-frame-${frame}.png`)});
    }
    await page.evaluate(()=>AT.present.go('s03',5,1));
    assert.equal(await longSlider.inputValue(),String(w),'Long-window control survives navigation and reveals.');
  }
  await longSlider.focus();await page.keyboard.press('ArrowLeft');
  assert.equal(await longSlider.inputValue(),'19');
  assert.equal(await page.locator('#s03-long-cost').getAttribute('data-weights'),String(19*80));
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
  await page.locator('#s03-frame-window-head').screenshot({path:path.join(shots,'phone-head-5.png')});
  await longSlider.fill('10');
  await page.locator('#s03-frame-boundary').screenshot({path:path.join(shots,'phone-long-10.png')});
  await page.locator('#s03-frame-window-cost').screenshot({path:path.join(shots,'phone-window-cost.png')});
  const phoneCostTable=await page.locator('#s03-ktab table').evaluate(table=>({width:table.getBoundingClientRect().width,available:table.parentElement.clientWidth,cellOverflow:[...table.querySelectorAll('th,td')].some(cell=>cell.scrollWidth>cell.clientWidth+1)}));
  assert(phoneCostTable.width<=phoneCostTable.available+1&&!phoneCostTable.cellOverflow,'Show every cost-table column on the phone, including biases and the head total.');
  assert((await longSlider.boundingBox()).width>=180);
  assert.equal(await headSlider.inputValue(),'5');
  assert((await headSlider.boundingBox()).width>=180);
  assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)));
  const final=await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs}));
  assert.equal(final,initial,'Architecture controls must not mutate the trained toy or its predictions.');
  assert.deepEqual(errors,[]);
  console.log('PASS: ten short windows, eleven long windows, exact suffixes and clue boundaries, head-only parameter scaling, scalar nodes/edges, concatenation, output tracing, coloured math, controls, progressive reveal, keyboard, retained state, reduced motion, phone layout, and model immutability.');
  console.log('Screenshots: '+shots);
}finally{await browser.close();}

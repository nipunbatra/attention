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
assert(model.W_O.every(row=>row.every(x=>x!==0)),'Both message coordinates contribute to every output coordinate.');
assert(ref.Delta[6].every(x=>Math.abs(x)>.01),'All four update coordinates visibly change in the bank example.');
assert(ref.Delta[6].slice(0,2).every((x,c)=>Math.abs(x-ref.Mmsg[6][c])>.1),'The first two coordinates must not look copied from the message.');
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
  const checkNumberBounds=async selector=>{
    assert(await page.locator(selector).first().isVisible(),'Measure cells while their diagram is visible.');
    const clipped=await page.locator(selector).evaluateAll(groups=>groups.flatMap(g=>[...g.querySelectorAll('.po-number,.az-number')].flatMap((t,i)=>{
      const text=t.getBBox(),cell=g.querySelectorAll('rect[data-coordinate]')[i].getBBox();
      return text.x<cell.x+1||text.x+text.width>cell.x+cell.width-1?[g.id+': '+t.textContent]:[];
    })));
    assert.deepEqual(clipped,[],'Signed numbers fit inside their cells with padding.');
  };
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
  assert(index('s11-frame-query-setup')+1===index('s11-frame-query-calc'),'Show the input and matrix immediately before expanding the arithmetic.');
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
  // The two-phase overview shares the prose's reveal schedule and colour roles.
  assert.equal(await page.locator('#s08 .frame').count(),9,'Keep the weighted-sum lesson and move the two value-intervention slides into one optional reading exercise.');
  const section8Titles=await page.locator('#s08 .frame').evaluateAll(es=>es.map(e=>e.dataset.title));
  assert.equal(section8Titles.at(-1),'Add the weighted values, coordinate by coordinate');
  assert(!section8Titles.some(t=>/Change only the values|Same weights, different information/.test(t)),'Neither repetition remains in the classroom route.');
  await page.evaluate(()=>AT.present.go('s08',9,99));
  assert(!(await page.locator('#s08-value-experiment').isVisible()),'The optional exercise stays out of presentation mode.');
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s09-frame1','The message goes directly to the output-projection diagram.');
  await page.evaluate(()=>AT.present.prev());
  assert.equal(await page.locator('.frame.is-live').getAttribute('data-title'),'Add the weighted values, coordinate by coordinate','Reverse navigation also skips both optional slides.');
  await go('s08-frame-phases');
  for(const build of [0,1,2,3,2,1,0,3]){
    await page.evaluate(build=>AT.present.go('s08',1,build),build);
    for(const [id,threshold] of [['s08-phase-a',1],['s08-phase-b',2]]){
      assert.equal(await page.locator('#'+id).evaluate(e=>getComputedStyle(e).visibility),build>=threshold?'visible':'hidden','Diagram phase tracks its prose build in both directions.');
    }
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The complete phase overview fits at build '+build);
  }
  for(const role of ['q','k','v']){
    const colours=await page.evaluate(role=>({
      diagram:getComputedStyle(document.querySelector('.phase-overview .phase-'+role+' .phase-symbol')).fill,
      prose:getComputedStyle(document.querySelector('#s08-frame-phases .prose .m-'+role)).color
    }),role);
    assert.equal(colours.diagram,colours.prose,'Match '+role+' across the words, math and diagram.');
  }
  const alphaColours=await page.evaluate(()=>({
    diagram:getComputedStyle(document.querySelector('.phase-overview tspan.phase-a')).fill,
    weight:getComputedStyle(document.querySelector('.phase-overview .phase-weight-edge')).stroke
  }));
  assert.equal(alphaColours.diagram,alphaColours.weight,'Use the weight colour on the connector and both products.');
  assert.match(await page.locator('#s08-phase-desc').textContent(),/scale the dot products.*same weights multiply values.*source index/);
  await page.waitForTimeout(300); // Let the existing reveal animation settle before inspecting math glyphs.
  await page.screenshot({path:path.join(shots,'two-phase-overview.png')});
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('data-title'),'Score bank against every key');
  // Explain the first scaled score briefly, then pay off the signpost later.
  const scaleNotes=page.locator('#s08 [data-scale-intro]');
  assert.equal(await scaleNotes.count(),2,'Reading and presentation modes each need the first-use note.');
  assert.equal((await scaleNotes.allTextContents())[0],(await scaleNotes.allTextContents())[1]);
  await go('s08-frame-scores');
  assert(await page.locator('#s08-frame-scores [data-scale-intro]').isVisible());
  assert.match(await page.locator('#s08-frame-scores [data-scale-intro]').innerText(),/coordinates per query\/key, not tokens.*before softmax.*Section 12/s);
  const scaleMath=(await page.locator('#s08-frame-scores [data-scale-intro] annotation').allTextContents()).join(' ');
  assert(scaleMath.includes('d_k='+model.d_k)&&scaleMath.includes('\\sqrt{'+model.d_k+'}'));
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'First-use scale note fits with all seven scores.');
  assert.equal(await page.locator('#s12 .frame').count(),2,'Keep the scaling explanation and real before/after example; extra experiments remain in reading mode.');
  await go('s12-frame-scaling');
  assert.match(await page.locator('#s12-frame-scaling .prose p').first().innerText(),/Earlier.*three-coordinate.*Here is why/s);
  await page.evaluate(()=>AT.present.go('s12',1,3));
  await page.waitForTimeout(300); // Nested text follows the existing reveal animation.
  assert.match(await page.locator('#s12-frame-scaling [data-build="2"]').innerText(),/mutually independent, mean 0, variance 1/);
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The deferred explanation still fits.');
  // Teach W_O as a learned dimension mapping; keep the fixed arithmetic optional.
  assert.equal(await page.locator('#s09 .frame').count(),8,'The detailed projection calculation is outside the classroom sequence.');
  await go('s09-frame1');
  for(const build of [0,1,0,1]){
    await page.evaluate(build=>AT.present.go('s09',1,build),build);await page.waitForTimeout(350);
    assert(await page.locator('#s09-projection-overview svg').isVisible(),'Show the graphical operation before the equations.');
    assert.equal(await page.locator('#s09-projection-equations').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Projection overview fits in both reveal states.');
    await page.screenshot({path:path.join(shots,'projection-overview-'+build+'.png')});
  }
  const checkVector=async(id,expected)=>{
    const actual=await page.locator('#'+id).evaluate(e=>({vector:JSON.parse(e.dataset.vector),cells:e.querySelectorAll('rect[data-coordinate]').length,text:[...e.querySelectorAll('.po-number')].map(t=>Number(t.textContent.replaceAll('−','-')))}));
    assert.equal(actual.cells,expected.length,'Visible coordinate count matches the vector width.');
    assert.deepEqual(actual.text,rounded(expected),'Diagram displays the independently calculated coordinates.');
    actual.vector.forEach((x,c)=>assert(Math.abs(x-expected[c])<1e-12,'Diagram data retain full precision.'));
  };
  for(let j=0;j<7;j++)await checkVector('s09-po-value-'+j,ref.V[j]);
  for(const [id,field]of [['message','Mmsg'],['update','Delta'],['original','E'],['result','Enew']])await checkVector('s09-po-'+id,ref[field][6]);
  const weights=await page.locator('#s09-projection-overview [data-weight]').evaluateAll(es=>es.map(e=>Number(e.dataset.weight)));
  assert.equal(weights.length,7);weights.forEach((x,j)=>assert(Math.abs(x-ref.A[6][j])<1e-12,'Each displayed weight belongs to its own source value.'));
  for(const [role,id]of [['v','message'],['d','update'],['e','original'],['ep','result']]){
    const colors=await page.evaluate(({role,id})=>({diagram:getComputedStyle(document.querySelector('#s09-po-'+id+' .po-role')).fill,math:getComputedStyle(document.querySelector('#s09-projection-equations .m-'+role)).color}),{role,id});
    assert.equal(colors.diagram,colors.math,'Match diagram and equation role colours.');
  }
  const escapedText=await page.locator('#s09-projection-overview svg text').evaluateAll(es=>es.filter(e=>{const r=e.getBBox();return r.x<0||r.y<0||r.x+r.width>1120||r.y+r.height>374;}).map(e=>e.textContent));
  assert.deepEqual(escapedText,[],'Every diagram label fits within its viewBox.');
  await checkNumberBounds('#s09-projection-overview [data-vector]');
  assert.match(await page.locator('#s09-projection-desc').textContent(),/two-coordinate message.*four embedding coordinates.*unchanged original embedding/s);
  const overviewMath=(await page.locator('#s09-projection-equations annotation').allTextContents()).join(' ');
  assert(overviewMath.includes('1\\times '+model.d_v)&&overviewMath.includes('1\\times '+model.d_model),'The revealed equations use the same concrete dimensions as the diagram.');
  assert.equal(await page.locator('#s09-projection-overview .po-residual').getAttribute('d'),'M618 317 H695','The original embedding goes straight into the residual plus.');
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s09-frame-motif','The existing residual motif follows the equations without an extra slide.');
  // The same slide can open a numerical close-up without adding classroom frames.
  const zoomButton=page.locator('#s09-attention-zoom');
  assert.equal(await zoomButton.getAttribute('aria-expanded'),'false');
  assert(await page.locator('#s09-motif svg').isVisible());
  assert(!(await page.locator('#s09-attention-detail').isVisible()));
  for(const build of [0,1]){
    await page.evaluate(build=>AT.present.setBuild(build),build);
    await zoomButton.click();await page.waitForTimeout(250);
    assert.equal(await zoomButton.getAttribute('aria-expanded'),'true');
    assert(!(await page.locator('#s09-attention-overview').isVisible()));
    assert(await page.locator('#s09-attention-detail svg').isVisible());
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The detailed view fits at either reveal build.');
    await page.screenshot({path:path.join(shots,'attention-zoom-'+build+'.png')});
    await zoomButton.click();await page.waitForTimeout(150);
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The overview still fits after returning.');
  }
  const zoomTarget=page.locator('#s09-motif [data-stage=att]');
  for(const key of ['Enter',' ']){
    await zoomTarget.focus();await page.keyboard.press(key);await page.waitForTimeout(150);
    assert.equal(await zoomButton.getAttribute('aria-expanded'),'true','The SVG attention box opens from the keyboard.');
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s09-frame-motif','Space activates the zoom rather than advancing the slide.');
    assert(await zoomButton.evaluate(e=>e===document.activeElement),'Focus moves to the visible return control.');
    await page.keyboard.press('Enter');
    assert.equal(await zoomButton.getAttribute('aria-expanded'),'false');
  }
  await zoomTarget.click();
  const checkZoomVector=async(id,expected,decimals=2)=>{
    const actual=await page.locator('#s09-az-'+id).evaluate(e=>({vector:JSON.parse(e.dataset.vector),cells:e.querySelectorAll('rect[data-coordinate]').length,text:[...e.querySelectorAll('.az-number')].map(t=>Number(t.textContent.replaceAll('−','-')))}));
    assert.equal(actual.cells,expected.length);
    assert.deepEqual(actual.text,rounded(expected,decimals),'Every close-up number matches the independent forward calculation.');
    actual.vector.forEach((x,c)=>assert(Math.abs(x-expected[c])<1e-12,'Retain full precision before display rounding.'));
  };
  for(const [id,field]of [['query','Q'],['scores','S'],['message','Mmsg'],['update','Delta'],['original','E'],['result','Enew']])await checkZoomVector(id,ref[field][6]);
  await checkZoomVector('weights',ref.A[6],3);
  for(let j=0;j<7;j++)await checkZoomVector('value-'+j,ref.V[j]);
  const zoomText=await page.locator('#s09-attention-detail svg').textContent();
  for(const phrase of ['Dot product','Softmax','Mix values','Project','3 numbers per key','2 value numbers','4 embedding numbers','2 × 4 mapping','Original e₇ (unchanged)','Updated embedding'])assert(zoomText.includes(phrase),'The close-up explains '+phrase+'.');
  assert.equal(await page.locator('#s09-attention-detail .az-residual').getAttribute('d'),'M629 420 H744','The original row bypasses the projection and meets the update at addition.');
  for(const [role,id]of [['v','message'],['d','update'],['e','original'],['ep','result'],['a','weights'],['q','query']]){
    const colour=await page.evaluate(({role,id})=>({actual:getComputedStyle(document.querySelector('#s09-az-'+id+' .az-role')).fill,reference:getComputedStyle(document.querySelector('#s09 .m-'+role)).color}),{role,id});
    assert.equal(colour.actual,colour.reference,'The close-up preserves the '+role+' colour role.');
  }
  assert.deepEqual(await page.locator('#s09-attention-detail svg text').evaluateAll(es=>es.filter(e=>{const r=e.getBBox();return r.x<0||r.y<0||r.x+r.width>1120||r.y+r.height>450;}).map(e=>e.textContent)),[],'Detailed diagram labels stay within the viewBox.');
  await checkNumberBounds('#s09-attention-detail [data-vector]');
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s09-frame2','The optional zoom does not add an extra navigation step.');
  await page.evaluate(()=>AT.present.prev());
  assert.equal(await zoomButton.getAttribute('aria-expanded'),'false','Returning to the slide starts with its overview.');
  assert.equal(await page.locator('#s09-frame-wo-calc').count(),0);
  await go('s09-frame3');
  assert.equal(await page.locator('#s09-frame3 table').count(),0,'Keep full matrix arithmetic in optional reading.');
  assert.match(await page.locator('#s09-frame3 .prose').innerText(),/In a trained model.*learns.*required dimensions/s);
  const projectionMath=(await page.locator('#s09-projection-shapes annotation').allTextContents()).join(' ');
  for(const shape of ['1\\times '+model.d_v,model.d_v+'\\times '+model.d_model,'1\\times '+model.d_model])assert(projectionMath.includes(shape),'Projection shape '+shape+' matches the model.');
  assert(!(await page.locator('#s09-projection-details').isVisible()),'The optional matrix never appears in presentation mode.');
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow);
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s09-frame4','Proceed directly from the mapping to residual addition.');
  await go('s09-frame-chain');
  assert.deepEqual(await rows('#s09-chain'),[rounded(ref.Mmsg[6]),rounded(ref.Delta[6]),rounded(ref.E[6])],'The chain uses the new dense projection and the unchanged message/input.');
  assert((await page.locator('#s09-chain td').evaluateAll(es=>es.map(e=>getComputedStyle(e).opacity))).every(x=>x==='1'),'Show all numerical rows clearly on the classroom result slide.');
  const mixMath=(await page.locator('#s09-projection-mix-example annotation').allTextContents()).join(' ');
  for(const value of [model.W_O[0][0].toFixed(1),model.W_O[1][0].toFixed(1),ref.Mmsg[6][0].toFixed(3),ref.Mmsg[6][1].toFixed(3),ref.Delta[6][0].toFixed(2)])assert(mixMath.replaceAll('−','-').includes(value),'The shown scalar expansion includes '+value+'.');
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The mixed projection explanation fits with the numerical chain.');
  await page.screenshot({path:path.join(shots,'dense-projection-chain.png')});
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
  await go('s11-frame-query-setup');
  for(const build of [0,1,2,1,0,2]){
    await page.evaluate(build=>AT.present.setBuild(build),build);await page.waitForTimeout(150);
    assert(await page.locator('#s11-query-input').isVisible(),'The input row is visible from the first build.');
    assert.equal(await page.locator('#s11-query-matrix-block').evaluate(e=>getComputedStyle(e).visibility),build>=1?'visible':'hidden');
    assert.equal(await page.locator('#s11 .s11-query-product').evaluate(e=>getComputedStyle(e).visibility),build>=2?'visible':'hidden');
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Query setup build '+build+' fits.');
    await page.screenshot({path:path.join(shots,'query-setup-'+build+'.png')});
  }
  assert.deepEqual(await rows('#s11-query-matrix'),model.W_Q,'Show the canonical 4×3 query matrix in its row-vector orientation.');
  assert.deepEqual(await page.locator('#s11-query-input .vec-ax').allTextContents(),model.axes.short.e);
  assert.deepEqual(await page.locator('#s11-query-matrix tbody th').allTextContents(),model.axes.short.e,'Input axis order matches the weight-matrix rows.');
  assert.deepEqual((await page.locator('#s11-query-matrix thead th').allTextContents()).slice(1),model.axes.short.qk,'Weight-matrix columns match query axes.');
  const setupMath=(await page.locator('#s11-query-product-equation annotation').allTextContents()).join(' ');
  for(const shape of ['1\\times 4','4\\times 3','1\\times 3'])assert(setupMath.includes(shape),'Show shape '+shape+'.');
  assert(setupMath.indexOf('e_{7}')<setupMath.indexOf('W_Q')&&setupMath.indexOf('W_Q')<setupMath.indexOf('q_{7}'),'Multiply the input row on the left by W_Q on the right.');
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s11-frame-query-calc','The next slide expands the product.');
  await page.evaluate(()=>AT.present.prev());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s11-frame-query-setup');
  for(const [role,field]of [['query','Q'],['key','K'],['value','V']]){
    if(role==='query') await go('s11-frame-query-calc');
    else {
      await page.evaluate(()=>AT.present.exit());
      await page.locator('#s11-frame-'+role+'-calc').scrollIntoViewIfNeeded();
      assert(await page.locator('#s11-frame-'+role+'-calc').evaluate(el=>el.classList.contains('companion')),'Additional projection arithmetic stays available in reading mode.');
    }
    for(const j of [6,5,1,0,6]){
      await page.locator('#s11-frame-'+role+'-calc [data-token="'+j+'"]').click();
      for(const [r,f]of [['query','Q'],['key','K'],['value','V']]){
        const results=await page.locator('#s11-'+r+'-calc .calc-res').allTextContents();
        assert.deepEqual(results.map(x=>Number(x.replaceAll('−','-').replace(/[=≈]/g,'').trim())),rounded(ref[f][j]),'Recover the supplied '+f+' row with the unchanged shared matrix.');
      }
      assert.equal(await page.locator('#s11 .s11-calc-controls [data-token="'+j+'"][aria-pressed=true]').count(),3);
      for(const [id,f]of [['input','E'],['product-result','Q']]){
        const cells=(await page.locator('#s11-query-'+id+' .cell').allTextContents()).map(x=>Number(x.replaceAll('−','-')));
        assert.deepEqual(cells,rounded(ref[f][j]),'The setup retains the token selected in the arithmetic.');
      }
      assert.deepEqual(await rows('#s11-query-matrix'),model.W_Q,'Changing the token never changes W_Q.');
    }
    await page.screenshot({path:path.join(shots,role+'-derivation.png')});
  }
  await page.evaluate(()=>AT.present.enter());
  await go('s07-frame-pair');
  await page.locator('#s07-rchips button').nth(5).click();
  const changedFrames=['s07-frame1','s07-frame-keys','s07-frame-values','s07-frame-pair','s08-frame-phases','s08-frame-scores','s12-frame-scaling','s09-frame1','s09-frame3','s09-frame4','s09-frame-prediction-input','s09-frame-prediction-output','s11-frame1','s11-frame-query-setup','s11-frame-query-calc'];
  for(const id of changedFrames){
    await go(id);await page.screenshot({path:path.join(shots,id+'.png')});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),original,'All controls preserve the canonical model.');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await page.locator('#s09-frame-motif').scrollIntoViewIfNeeded();
  await zoomButton.click();
  assert(await page.locator('#s09-attention-detail svg').isVisible(),'The close-up also works in phone reading mode.');
  const detailScroll=page.locator('#s09-attention-detail .attention-detail-scroll');
  assert(await detailScroll.evaluate(e=>e.scrollWidth>e.clientWidth),'The detailed diagram scrolls locally on phones.');
  await page.screenshot({path:path.join(shots,'phone-attention-zoom-start.png')});
  await detailScroll.evaluate(e=>e.scrollLeft=e.scrollWidth);
  await page.screenshot({path:path.join(shots,'phone-attention-zoom-end.png')});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Zoom does not introduce phone page overflow.');
  await zoomButton.click();
  assert(await page.locator('#s09-attention-overview').isVisible());
  await page.emulateMedia({media:'print'});
  assert(await page.locator('#s09-attention-detail').isVisible()&&await page.locator('#s09-attention-overview').isVisible(),'Print includes both diagram versions.');
  await page.emulateMedia({media:'screen'});
  assert.equal(await page.locator('#s09-projection-equations').evaluate(e=>getComputedStyle(e).visibility),'visible','Reading mode includes the equations without requiring a reveal.');
  await page.locator('#s09-projection-overview').scrollIntoViewIfNeeded();
  const projectionScroll=page.locator('#s09-frame1 .projection-overview-scroll');
  assert(await projectionScroll.evaluate(e=>e.scrollWidth>e.clientWidth),'Keep diagram labels readable with local phone scrolling.');
  await page.screenshot({path:path.join(shots,'phone-projection-mix.png')});
  await projectionScroll.evaluate(e=>e.scrollLeft=e.scrollWidth);
  await page.screenshot({path:path.join(shots,'phone-projection-add.png')});
  const equations=await page.locator('#s09-projection-equations>div').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().toJSON()));
  assert(equations[1].top>=equations[0].bottom&&equations[2].top>=equations[1].bottom,'Phone equations stack in the same order as the diagram.');
  const woGlyphs=await page.locator('#s09-projection-shapes .mord.mathnormal').evaluateAll(els=>els.filter(e=>e.textContent==='W'||e.textContent==='O').map(e=>({text:e.textContent,rect:e.getBoundingClientRect().toJSON()})));
  const wGlyph=woGlyphs.find(e=>e.text==='W').rect,oGlyph=woGlyphs.find(e=>e.text==='O').rect;
  assert(oGlyph.left>=wGlyph.right-1&&oGlyph.top<wGlyph.bottom,'W_O stays together above its shape label on phones.');
  assert.equal(await page.locator('#s09-projection-details').getAttribute('open'),null,'The full projection arithmetic starts collapsed in reading mode.');
  await page.locator('#s09-projection-details summary').click();
  assert(await page.locator('#s09-wo').isVisible());
  assert.deepEqual(await rows('#s09-wo'),model.W_O,'The optional source matrix is unchanged.');
  assert(await page.locator('#s09-wocalc').isVisible());
  assert.match(await page.locator('#s09-projection-details p').first().innerText(),/hand-chosen.*Every column mixes both message coordinates/s);
  assert.match(await page.locator('#s09-projection-details p').nth(1).innerText(),/linear map.*orthogonal.*affine.*no output bias/s);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Optional arithmetic scrolls locally on phones.');
  await page.locator('#s09-projection-details summary').click();
  assert(await page.locator('#s08 .companion [data-scale-intro]').isVisible(),'The first-use note also appears in reading mode.');
  for(const id of changedFrames){
    if(id==='s08-frame-scores')continue; // Reading mode uses the combined phase-A table.
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone page overflow at '+id);
    await page.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert.equal(await page.locator('#s08-phase-a').evaluate(e=>getComputedStyle(e).visibility),'visible');
  assert.equal(await page.locator('#s08-phase-b').evaluate(e=>getComputedStyle(e).visibility),'visible');
  const diagramScroll=await page.locator('.phase-overview-scroll').evaluate(e=>({client:e.clientWidth,scroll:e.scrollWidth}));
  assert(diagramScroll.scroll>diagramScroll.client,'The full two-phase diagram scrolls locally on phones.');
  await page.locator('.phase-overview-scroll').evaluate(e=>e.scrollLeft=e.scrollWidth);
  await page.locator('.phase-overview').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(shots,'phone-phase-message.png')});
  assert.deepEqual(errors,[]);
  console.log('PASS: supplied Q/K/V match reference; seven paired source records; staged two-phase diagram and colour matching; first-use scaling signpost and later explanation; learned output-mapping slide and optional fixed arithmetic; no premature projections; both position-10 predictions; all shared-matrix derivations; model/state retention; phone layout. Screenshots: '+shots);
}finally{await browser.close();}

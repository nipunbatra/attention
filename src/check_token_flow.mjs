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
  // The two-phase overview shares the prose's reveal schedule and colour roles.
  assert.equal(await page.locator('#s08 .frame').count(),11,'Add the diagram to the existing slide, without another frame.');
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
  assert.equal(await page.locator('#s12 .frame').count(),5,'Keep the existing full scaling lesson.');
  await go('s12-frame-scaling');
  assert.match(await page.locator('#s12-frame-scaling .prose p').first().innerText(),/Earlier.*three-coordinate.*Here is why/s);
  await page.evaluate(()=>AT.present.go('s12',1,3));
  await page.waitForTimeout(300); // Nested text follows the existing reveal animation.
  assert.match(await page.locator('#s12-frame-scaling [data-build="2"]').innerText(),/mutually independent, mean 0, variance 1/);
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The deferred explanation still fits.');
  // Teach W_O as a learned dimension mapping; keep the fixed arithmetic optional.
  assert.equal(await page.locator('#s09 .frame').count(),8,'The detailed projection calculation is outside the classroom sequence.');
  assert.equal(await page.locator('#s09-frame-wo-calc').count(),0);
  await go('s09-frame3');
  assert.equal(await page.locator('#s09-frame3 table').count(),0,'No zero-heavy matrix on the mapping slide.');
  assert.match(await page.locator('#s09-frame3 .prose').innerText(),/In a trained model.*learns.*required dimensions/s);
  const projectionMath=(await page.locator('#s09-projection-shapes annotation').allTextContents()).join(' ');
  for(const shape of ['1\\times '+model.d_v,model.d_v+'\\times '+model.d_model,'1\\times '+model.d_model])assert(projectionMath.includes(shape),'Projection shape '+shape+' matches the model.');
  assert(!(await page.locator('#s09-projection-details').isVisible()),'The optional matrix never appears in presentation mode.');
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow);
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s09-frame4','Proceed directly from the mapping to residual addition.');
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
  const changedFrames=['s07-frame1','s07-frame-keys','s07-frame-values','s07-frame-pair','s08-frame-phases','s08-frame-scores','s12-frame-scaling','s09-frame3','s09-frame4','s09-frame-prediction-input','s09-frame-prediction-output','s11-frame1','s11-frame-query-calc','s11-frame-key-calc','s11-frame-value-calc'];
  for(const id of changedFrames){
    await go(id);await page.screenshot({path:path.join(shots,id+'.png')});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),original,'All controls preserve the canonical model.');
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  const woGlyphs=await page.locator('#s09-projection-shapes .mord.mathnormal').evaluateAll(els=>els.filter(e=>e.textContent==='W'||e.textContent==='O').map(e=>({text:e.textContent,rect:e.getBoundingClientRect().toJSON()})));
  const wGlyph=woGlyphs.find(e=>e.text==='W').rect,oGlyph=woGlyphs.find(e=>e.text==='O').rect;
  assert(oGlyph.left>=wGlyph.right-1&&oGlyph.top<wGlyph.bottom,'W_O stays together above its shape label on phones.');
  assert.equal(await page.locator('#s09-projection-details').getAttribute('open'),null,'The zero-heavy arithmetic starts collapsed in reading mode.');
  await page.locator('#s09-projection-details summary').click();
  assert(await page.locator('#s09-wo').isVisible());
  assert.deepEqual(await rows('#s09-wo'),model.W_O,'The optional source matrix is unchanged.');
  assert(await page.locator('#s09-wocalc').isVisible());
  assert.match(await page.locator('#s09-projection-details p').first().innerText(),/hand-chosen.*specific to this example/s);
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

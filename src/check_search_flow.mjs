// node src/check_search_flow.mjs [attention.html]
// Queries first, then four key lessons, then matching. Installs no dependencies.
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
let pw;for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
assert(pw,'Use an existing Playwright runtime.');
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-search-flow-'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const axes=['gradient flow','optimisation','architecture','generalisation'];
const queries=[[2,.6,.2,0],[0,.4,.1,2],[0,.1,2,0],[.2,2,0,.1]];
const keys=[[2.2,.6,.2,0],[.4,1,0,.2],[.1,0,2,.1],[0,0,1,0],[.2,.5,1,.6],[.1,.5,0,2]];
const titles=['Backpropagation','Gradient descent','CNNs','Transformers','Batch normalisation','Regularisation'];
const valueAxes=['chain rule','gradient step','step size','shared filters','token mixing','activation scaling','weight penalty','dropout'];
const values=[[1,.2,0,0,0,0,0,0],[.2,1,.9,0,0,0,0,0],[.2,.1,.1,1,0,0,0,0],[.2,.1,.1,0,1,0,0,0],[.1,.2,.2,0,0,1,0,0],[.1,.2,.1,0,0,0,1,.8]];
const valueFrames=['s05-frame-value-axes','s05-frame-values','s05-frame-return','s06-frame-weighted-values','s06-frame-message'];
const queryFrames=['s05-frame-search','s05-frame-query-axes','s05-frame-request','s05-frame-query-examples'];
const keyFrames=['s05-frame-key-idea','s05-frame-key-vector','s05-frame-key-examples','s05-frame-key-collection'];
const go=async(id,build=0)=>{
  const [sec,n]=await page.locator('#'+id).evaluate(e=>[e.closest('.sec').id,[...e.closest('.sec').querySelectorAll('.frame')].indexOf(e)+1]);
  await page.evaluate(([sec,n,b])=>AT.present.go(sec,n,b),[sec,n,build]);await page.waitForTimeout(500);
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' must fit.');
};
const matrix=async(id)=>page.locator('#'+id+' tbody tr').evaluateAll(rows=>rows.map(row=>[...row.querySelectorAll('td[data-c]')].map(c=>Number(c.textContent))));
const copy=async(id)=>page.locator('#'+id).evaluate(e=>{const n=e.cloneNode(true);n.querySelectorAll('script,.companion,.katex-mathml').forEach(x=>x.remove());return n.textContent;});
try{
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);await page.evaluate(()=>document.fonts.ready);
  const model=await page.evaluate(()=>JSON.stringify({model:AT.model,p:AT.forward(AT.sentences.river).probs}));
  const order=await page.locator('#s05 .frame').evaluateAll(els=>els.map(e=>e.id));
  assert.deepEqual(order.slice(3,13),[...queryFrames,...keyFrames,'s05-frame-matching','s05-frame-results']);
  for(const id of queryFrames)assert(!/\b(keys?|values?)\b/i.test(await copy(id)),id+' must teach only the request before source roles.');
  assert.equal(keyFrames.length,4);
  assert((await copy('s05-frame-query-axes')).includes('not probabilities'));
  assert((await copy('s05-frame-query-axes')).includes('by hand'));
  assert.deepEqual(await matrix('s05-query-examples'),queries);
  assert.deepEqual(await matrix('s05-key-first'),[keys[0]]);
  assert.deepEqual(await matrix('s05-vals'),values);
  assert((await copy('s05-frame-values')).includes('throughout retrieval'));
  assert((await copy('s05-frame-value-axes')).includes('Hand-chosen'));
  const guide=await matrix('s05-value-axis-guide');
  assert.deepEqual(guide.map(row=>row[1]),values[0],'Named-axis guide uses the same original Backpropagation value.');
  for(const id of ['s05-query-vector','s05-query-examples','s05-key-first','s05-key-example-vector','s05-key-collection']){
    const headers=await page.locator('#'+id+' thead th').allTextContents();
    assert.deepEqual(headers.slice(1).map(s=>s.trim().toLowerCase()),axes,id+' must use the same named axis order.');
  }
  await page.evaluate(()=>AT.present.enter());
  for(const id of [...queryFrames,...keyFrames]){
    for(const build of id==='s05-frame-key-collection'?[0]:[0,1]){
      await go(id,build);await page.screenshot({path:path.join(shots,id+'-'+build+'.png')});
      const clippedHeaders=await page.locator('#'+id+' .dt thead th').evaluateAll(els=>els.filter(e=>{
        if(e.querySelector('.katex')||getComputedStyle(e).display==='none')return false;
        const r=document.createRange();r.selectNodeContents(e);const box=e.getBoundingClientRect();
        return [...r.getClientRects()].some(t=>t.width>0&&(t.left<box.left-1||t.right>box.right+1));
      }).map(e=>e.textContent));
      assert.deepEqual(clippedHeaders,[],id+' headers must stay within their own columns.');
    }
  }
  // Real Next navigation must finish the query examples before the first key lesson.
  await go('s05-frame-query-examples',1);await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-key-idea');
  await go('s05-frame-key-collection');await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-matching');
  for(const [i,q]of queries.entries()){
    await go('s05-frame-request');await page.locator('#s05-score-qbtns button').nth(i).click();
    await page.evaluate(()=>AT.present.setBuild(1));await page.waitForTimeout(250);
    assert.deepEqual(await matrix('s05-query-vector'),[q]);
    assert.deepEqual(await matrix('s05-key-collection'),[q,...keys]);
    const strongest=await page.locator('#s05-query-vector .is-axis-focus').getAttribute('data-c');
    assert.equal(Number(strongest),q.indexOf(Math.max(...q)));
    const read=await page.locator('#s05-qread').textContent();
    for(const [c,n]of q.entries())if(n>0)assert(read.includes(axes[c]+' ('+n.toFixed(1)+')'),'Do not describe a small nonzero signal as absent.');
    await page.screenshot({path:path.join(shots,'query-'+i+'.png')});
    await go('s05-frame-key-collection');await page.locator('#s05-collection-qbtns button').nth(i).click();
    assert.deepEqual(await matrix('s05-key-collection'),[q,...keys],'Source descriptions remain fixed for every query.');
    assert.equal(await page.locator('#s05-key-collection .dt-comp').count(),0,'No scores before matching is taught.');
    const scores=keys.map(k=>k.reduce((sum,x,c)=>sum+x*q[c],0));
    const winner=scores.indexOf(Math.max(...scores));
    await go('s05-frame-matching',6);
    const shown=(await page.locator('#s05-dot tbody td.dt-comp').allTextContents()).slice(1).map(Number);
    assert.deepEqual(shown,scores.map(x=>Number(x.toFixed(1))),'All six scores agree with independent dot products.');
    await go('s05-frame-results');assert.equal(await page.locator('#s05-vgrid .is-top .vt').textContent(),titles[winner]);
    await go('s05-frame-three-jobs',1);
    assert.equal(await page.locator('#s05-example-title').textContent(),titles[winner]);
    assert.equal(await page.locator('#s05-example-explanation').textContent(),await page.locator('#s05-return-summary').textContent());
    await go('s05-frame-return',1);
    assert.deepEqual(await matrix('s05-return-value'),[values[winner]],'Winning value stays eight-wide and source-paired.');
  }
  // Same eight source coordinates in the next section, before and after weighting.
  for(const id of ['s05-vals','s05-return-value','s06-mix','s06-card-mix']){
    const headers=await page.locator('#'+id+' thead th:not(.dt-comp):not(.dt-lead)').allTextContents();
    assert.deepEqual(headers.slice(1).map(s=>s.trim().toLowerCase()),valueAxes);
  }
  for(const id of valueFrames){
    for(const build of id==='s06-frame-weighted-values'?[0,1,2,3,0,3]:id==='s05-frame-values'?[0]:[0,1]){
      await go(id,build);await page.screenshot({path:path.join(shots,id+'-'+build+'.png')});
      if(id==='s06-frame-weighted-values'){
        const view=await page.locator('#s06-mix .s06-mix-value').evaluateAll(cells=>cells.map(cell=>{
          const raw=cell.querySelector('.s06-raw'),weighted=cell.querySelector('.s06-weight');
          return {raw:getComputedStyle(raw).display!=='none',weighted:getComputedStyle(weighted).display!=='none'&&getComputedStyle(weighted).visibility==='visible'};
        }));
        assert(view.every(v=>v.raw===(build<2)&&v.weighted===(build>=2)),'Original and weighted entries have distinct reveal stages.');
      }
    }
  }
  const setScores=async(scores)=>page.locator('#s06-sliders input').evaluateAll((els,vals)=>els.forEach((e,j)=>{e.value=vals[j];e.dispatchEvent(new Event('input',{bubbles:true}));}),scores);
  const setSoft=async(soft)=>{await go('s06-frame-weights');const button=page.locator('#s06-mode button');if((await button.getAttribute('aria-pressed')==='true')!==soft)await button.click();};
  const close=(actual,expected,label)=>{assert.equal(actual.length,expected.length,label);actual.forEach((x,i)=>assert(Math.abs(x-expected[i])<=.000501,label+' coordinate '+i));};
  const baseScores=keys.map(k=>Number(k.reduce((s,x,c)=>s+x*queries[0][c],0).toFixed(1)));
  for(const test of [
    {name:'hard',scores:baseScores,soft:false,tau:1},
    {name:'soft',scores:baseScores,soft:true,tau:1},
    {name:'tied-soft',scores:baseScores.map((x,j)=>j===1?baseScores[0]:x),soft:true,tau:1},
    {name:'sharp',scores:baseScores,soft:true,tau:.1},
    {name:'broad',scores:baseScores,soft:true,tau:4},
    {name:'dropout-source',scores:baseScores.map((x,j)=>j===5?8:x),soft:false,tau:1}
  ]){
    await setSoft(test.soft);await setScores(test.scores);
    await page.locator('#s06-temp input').evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));},test.tau);
    const max=Math.max(...test.scores),exp=test.scores.map(s=>Math.exp((s-max)/test.tau)),total=exp.reduce((a,b)=>a+b,0);
    const alpha=test.soft?exp.map(e=>e/total):test.scores.map((s,j)=>j===test.scores.indexOf(max)?1:0);
    const sum=valueAxes.map((_,c)=>values.reduce((s,row,j)=>s+alpha[j]*row[c],0));
    await go('s06-frame-weighted-values',3);
    const rows=await page.locator('#s06-mix tbody tr').evaluateAll(els=>els.map(e=>({raw:[...e.querySelectorAll('.s06-raw')].map(c=>+c.textContent),weighted:[...e.querySelectorAll('.s06-weight')].map(c=>+c.textContent),alpha:+e.querySelector('.dt-lead').textContent})));
    assert.deepEqual(rows.map(r=>r.raw),values,'Stored values never change.');
    close(rows.map(r=>r.alpha),alpha,test.name+' alpha');
    rows.forEach((r,j)=>close(r.weighted,values[j].map(v=>alpha[j]*v),test.name+' product '+j));
    const footer=await page.locator('#s06-mix tfoot td:not(.dt-lead)').allTextContents();
    close(footer.map(Number),sum,test.name+' sum');
    assert.deepEqual(await matrix('s05-vals'),values);
    await page.screenshot({path:path.join(shots,'mixture-'+test.name+'.png')});
    await go('s06-frame-message',1);close((await matrix('s06-card-mix'))[0],sum,test.name+' returned row');
    assert.equal(await page.evaluate(()=>AT.present.fitReport().overflow),false);
  }
  const qBefore=await matrix('s05-query-vector');
  await go('s05-frame-key-examples',1);
  for(const j of [2,5,1]){
    await page.locator('#s05-key-example-buttons [data-source="'+j+'"]').click();
    for(const build of [0,1,0,1]){
      await page.evaluate(b=>AT.present.setBuild(b),build);await page.waitForTimeout(250);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow);
      assert.equal(await page.locator('#s05-key-example-vector').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
      assert.deepEqual(await matrix('s05-key-example-vector'),[keys[j]]);
      assert.deepEqual(await matrix('s05-query-vector'),qBefore,'Choosing a source example never edits the query.');
      assert((await page.locator('#s05-key-example-title').textContent()).includes(titles[j]));
    }
    await page.screenshot({path:path.join(shots,'key-'+j+'.png')});
  }
  await go('s05-frame-search');
  assert.deepEqual(await matrix('s05-query-vector'),[queries[0]],'New search visits keep the original query reset.');
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,p:AT.forward(AT.sentences.river).probs})),model);
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  for(const id of [...queryFrames,...keyFrames,...valueFrames]){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone document overflow.');
    await page.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  for(const id of ['s05-vals','s05-return-value','s06-mix','s06-card-mix']){
    const box=await page.locator('#'+id+' .dt-scroll').evaluate(e=>{e.scrollLeft=e.scrollWidth;return {width:e.clientWidth,scroll:e.scrollWidth,left:e.scrollLeft,right:e.getBoundingClientRect().right};});
    assert(box.scroll>box.width&&box.left>0&&box.right<=391,'All eight value columns remain reachable by local phone scrolling.');
  }
  for(const id of ['s05-query-vector','s05-query-examples','s05-key-first','s05-key-example-vector','s05-key-collection']){
    const box=await page.locator('#'+id+' .dt-scroll').evaluate(e=>({width:e.clientWidth,scroll:e.scrollWidth,x:e.getBoundingClientRect().x,right:e.getBoundingClientRect().right}));
    assert(box.scroll>box.width&&box.x>=0&&box.right<=391,'Named axes scroll locally on phones.');
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: query/key lessons and 24 scores; eight named value axes, all 48 stored values, 288 weighted products, 48 mixture coordinates, hard/soft/tied/temperature cases, independent state, reveals, desktop and phone layout. Screenshots: '+shots);
}finally{await browser.close();}

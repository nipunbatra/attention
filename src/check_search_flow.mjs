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
const queryFrames=['s05-frame-search','s05-frame-query-axes','s05-frame-request','s05-frame-query-examples'];
const keyFrames=['s05-frame-key-idea','s05-frame-key-vector','s05-frame-key-examples','s05-frame-key-collection'];
const go=async(id,build=0)=>{
  const n=await page.locator('#'+id).evaluate(e=>[...e.closest('.sec').querySelectorAll('.frame')].indexOf(e)+1);
  await page.evaluate(([n,b])=>AT.present.go('s05',n,b),[n,build]);await page.waitForTimeout(500);
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
  for(const id of ['s05-query-vector','s05-query-examples','s05-key-first','s05-key-example-vector','s05-key-collection']){
    const headers=await page.locator('#'+id+' thead th').allTextContents();
    assert.deepEqual(headers.slice(1).map(s=>s.trim().toLowerCase()),axes,id+' must use the same named axis order.');
  }
  await page.evaluate(()=>AT.present.enter());
  for(const id of [...queryFrames,...keyFrames]){
    for(const build of id==='s05-frame-key-collection'?[0]:[0,1]){
      await go(id,build);await page.screenshot({path:path.join(shots,id+'-'+build+'.png')});
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
  for(const id of [...queryFrames,...keyFrames]){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone document overflow.');
    await page.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  for(const id of ['s05-query-vector','s05-query-examples','s05-key-first','s05-key-example-vector','s05-key-collection']){
    const box=await page.locator('#'+id+' .dt-scroll').evaluate(e=>({width:e.clientWidth,scroll:e.scrollWidth,x:e.getBoundingClientRect().x,right:e.getBoundingClientRect().right}));
    assert(box.scroll>box.width&&box.x>=0&&box.right<=391,'Named axes scroll locally on phones.');
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: four query lessons, four key lessons, named axes, exact query/key vectors, 24 independent match scores, fixed-source invariant, source/query state separation, reveals, desktop and phone layout. Screenshots: '+shots);
}finally{await browser.close();}

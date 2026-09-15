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
const valueFrames=['s05-frame-value-axes','s05-frame-values','s05-frame-payload','s05-frame-return','s06-frame-weighted-values','s06-frame-message'];
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
  assert.deepEqual(order.slice(-4),['s05-frame-value-axes','s05-frame-values','s05-frame-payload','s05-frame-return'],'Unpack the returned content after defining its axes, before the compact recap.');
  assert.equal(order.length,20,'The video bridge is followed by contact and pronoun examples before numerical values.');
  assert.deepEqual(order.slice(13,16),['s05-frame-three-jobs','s05-frame-contact-example','s05-frame-pronoun-example']);
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
  for(const build of [0,1,0,1]){
    await go('s06-frame-hard-retrieval',build);
    const frame=page.locator('#s06-frame-hard-retrieval');
    for(const role of ['q','k','v']){
      const wordColor=await frame.locator('.sym-'+role).evaluate(e=>getComputedStyle(e).color);
      const mathColors=await frame.locator('.katex-html .m-'+role).evaluateAll(es=>es.map(e=>getComputedStyle(e).color));
      assert(mathColors.length>=2,'Each role appears in the prose and display equation.');
      assert(mathColors.every(color=>color===wordColor),'Hard retrieval '+role+' text and math share the same colour.');
    }
    assert((await copy('s06-frame-hard-retrieval')).includes('winning index'));
    assert(/match\s*score/.test(await copy('s06-frame-hard-retrieval')));
    assert.equal(await frame.locator('[data-build="1"]').evaluate(e=>getComputedStyle(e).visibility),build?'visible':'hidden');
    await page.screenshot({path:path.join(shots,'hard-retrieval-'+build+'.png')});
  }
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('#s06 .frame.is-live').getAttribute('id'),'s06-frame-scores');
  await go('s06-frame-scores');
  const scoreFrame=page.locator('#s06-frame-scores');
  for(const role of ['q','k','a']){
    const color=await scoreFrame.locator('.sym-'+role).evaluate(e=>getComputedStyle(e).color);
    const mathColors=await scoreFrame.locator('.katex-html .m-'+role).evaluateAll(es=>es.map(e=>getComputedStyle(e).color));
    assert(mathColors.length>=2&&mathColors.every(c=>c===color),'Score provenance uses matching role colours.');
  }
  const arithmetic=(await page.locator('#s06-score-example annotation').allTextContents()).join(' ');
  assert.deepEqual([...arithmetic.matchAll(/\\vq\{([\d.]+)\}/g)].map(m=>Number(m[1])),queries[0],'Worked row uses the original query coordinates.');
  assert.deepEqual([...arithmetic.matchAll(/\\vk\{([\d.]+)\}/g)].map(m=>Number(m[1])),keys[0],'Worked row uses the Backpropagation key coordinates.');
  const firstScore=queries[0].reduce((sum,q,c)=>sum+q*keys[0][c],0);
  assert(arithmetic.includes('=\\va{'+firstScore.toFixed(1)+'}'));
  assert.deepEqual((await page.locator('#s06-dot tbody td.dt-comp').allTextContents()).slice(1).map(Number),keys.map(k=>Number(k.reduce((sum,x,c)=>sum+x*queries[0][c],0).toFixed(1))));
  assert.equal(await page.locator('#s06-frame-change-score, #s06-sliders, #s06-reset, #s06-adjust-bars, #s06-score-edited').count(),0,'Remove the score-change slide and its controls/notices from both views.');
  assert.deepEqual(await page.locator('#s06 .frame').evaluateAll(es=>es.map(e=>e.id)),['s06-frame-hard-retrieval','s06-frame-scores','s06-frame-normalize','s06-frame-weights','s06-frame-weighted-values','s06-frame-message','s06-frame-temperature'],'The result now leads directly to temperature, with one fewer frame.');
  assert(!/later sliders|sliders below|slider has set|Try it: drag/.test(await copy('s06')),'No obsolete instructions for the removed sliders remain.');
  await page.screenshot({path:path.join(shots,'score-provenance.png')});
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
    assert.equal(await page.locator('#s05-example-transcript').getAttribute('data-source'),String(winner));
    assert.equal(await page.locator('#s05-example-thumbnail svg title').textContent(),titles[winner]+' thumbnail');
    assert((await page.locator('#s05-example-transcript').textContent()).includes(['chain rule','dropout','reuse the same weights','learning rate'][i]),'Each matched video supplies its own illustrative transcript.');
    assert((await page.locator('#s05-frame-three-jobs .transcript-label').textContent()).includes('written for this example'));
    for(const build of [0,1,2,0,2]){
      await go('s05-frame-three-jobs',build);
      assert.equal(await page.locator('#s05-example-transcript').evaluate(e=>getComputedStyle(e).visibility),build>=1?'visible':'hidden');
      assert.equal(await page.locator('#s05-frame-three-jobs .example-takeaway').evaluate(e=>getComputedStyle(e).visibility),build>=2?'visible':'hidden');
      if(build===2)await page.screenshot({path:path.join(shots,'key-value-transcript-'+i+'.png')});
    }
    await go('s05-frame-payload');await page.locator('#s05-payload-qbtns button').nth(i).click();
    assert.equal(await page.locator('#s05-payload-title').textContent(),'Matched video '+(winner+1)+': '+titles[winner]);
    const lesson=await page.locator('#s05-payload-steps li').allTextContents();
    assert.equal(lesson.length,3);assert(lesson.every(text=>text.length>40),'The returned content has concrete explanations, not just labels.');
    if(winner===0)assert(lesson[2].includes('separate step'),'Distinguish computing gradients from updating weights.');
    const features=await page.locator('#s05-payload-steps [data-axis]').evaluateAll(es=>es.map(e=>e.dataset.axis));
    const linked=await page.locator('#s05-payload-vector .is-payload-feature').evaluateAll(es=>es.map(e=>e.dataset.axis));
    assert.deepEqual(linked,features,'Colour-matched ideas point to the same named coordinates.');
    assert.deepEqual(features,[[valueAxes[0],valueAxes[1]],[valueAxes[6],valueAxes[7]],[valueAxes[3]],[valueAxes[1],valueAxes[2]]][i]);
    assert((await page.locator('#s05-payload-label .katex annotation').textContent()).includes('v_'+(winner+1)));
    assert((await page.locator('#s05-payload-return').textContent()).includes('all eight numbers'));
    for(const build of [0,1,2,0,2]){
      await go('s05-frame-payload',build);
      assert.equal(await page.locator('#s05-frame-payload .payload-numbers').evaluate(e=>getComputedStyle(e).visibility),build>=1?'visible':'hidden');
      assert.equal(await page.locator('#s05-payload-return').evaluate(e=>getComputedStyle(e).visibility),build>=2?'visible':'hidden');
      assert.deepEqual(await page.locator('#s05-payload-vector svg').evaluate(e=>JSON.parse(e.dataset.vector)),values[winner],'The explanation never changes the returned value row.');
      assert.deepEqual(await matrix('s05-vals'),values,'The six stored source values stay fixed.');
      if(i===0)await page.screenshot({path:path.join(shots,'payload-reveal-'+build+'.png')});
    }
    await page.screenshot({path:path.join(shots,'payload-query-'+i+'.png')});
    await page.evaluate(()=>AT.present.next());
    assert.equal(await page.locator('#s05 .frame.is-live').getAttribute('id'),'s05-frame-return','The explanation leads directly into the three-role recap.');
    await go('s05-frame-return',2);
    await page.locator('#s05-role-qbtns button').nth(i).click();
    for(const group of ['s05-qbtns','s05-score-qbtns','s05-collection-qbtns','s05-payload-qbtns','s05-role-qbtns']){
      assert.equal(await page.locator('#'+group+' button').nth(i).getAttribute('aria-pressed'),'true','Query controls stay synchronized.');
    }
    const roleRows=await page.locator('#s05-frame-return .role-vector').evaluateAll(svgs=>svgs.map(svg=>({
      role:svg.dataset.role,vector:JSON.parse(svg.dataset.vector),description:svg.getAttribute('aria-label'),
      axes:[...svg.querySelectorAll('.role-coordinate')].map(e=>e.dataset.axis),
      values:[...svg.querySelectorAll('.role-coordinate')].map(e=>Number(e.dataset.value)),
      bars:[...svg.querySelectorAll('.role-bar')].map(e=>Number(e.getAttribute('width'))),
      color:getComputedStyle(svg).color,
      headingColor:getComputedStyle(svg.closest('.role-card').querySelector('.big')).color,
      collisions:[...svg.querySelectorAll('.role-coordinate')].filter(row=>{
        const [label,number]=row.querySelectorAll('text');
        const text=label.getBBox(),n=number.getBBox();
        return text.x+text.width>182||n.x<267||n.x+n.width>320;
      }).map(e=>e.dataset.axis)
    })));
    for(const [r,role]of ['q','k','v'].entries()){
      const row=roleRows[r],expected=r===0?q:r===1?keys[winner]:values[winner];
      assert.equal(row.role,role);assert.deepEqual(row.vector,expected);assert.deepEqual(row.values,expected);
      assert.deepEqual(row.axes,r===2?valueAxes:axes);
      expected.forEach((v,c)=>assert(Math.abs(row.bars[c]-79*v/(r===2?1:2.5))<1e-8,'Bars encode the exact coordinate on the stated scale.'));
      assert.equal(row.color,row.headingColor,'SVGs and role labels share colours.');
      assert.deepEqual(row.collisions,[],'Named axes, bars and values have separate space.');
      assert(row.description.includes('row of '+(expected.length===4?'four':'eight')+' numbers'),'Accessible SVG description identifies the vector width.');
    }
    assert.equal(await page.locator('#s05-role-key-title').textContent(),titles[winner]);
    assert.equal(await page.locator('#s05-return-title').textContent(),titles[winner]);
    assert((await page.locator('#s05-role-key-label .katex annotation').textContent()).includes('k_'+(winner+1)));
    assert((await page.locator('#s05-role-value-label .katex annotation').textContent()).includes('v_'+(winner+1)));
    assert((await page.locator('#s05-role-outcome .katex annotation').first().textContent()).endsWith('='+scores[winner].toFixed(2)),'The winning score is independently recomputed without lossy rounding.');
    for(const build of [0,1,2,0,2]){
      await go('s05-frame-return',build);
      for(const [r,role]of ['q','k','v'].entries()){
        assert.equal(await page.locator('#s05-frame-return .role-card.obj-'+role).evaluate(e=>getComputedStyle(e).visibility),build>=r?'visible':'hidden','Reveal query, then key, then value.');
      }
      assert.deepEqual(await matrix('s05-query-vector'),[q],'Recap reveals do not reset the question.');
      if(i===0)await page.screenshot({path:path.join(shots,'roles-reveal-'+build+'.png')});
    }
    await page.screenshot({path:path.join(shots,'roles-query-'+i+'.png')});
    assert.deepEqual(await matrix('s05-return-value'),[values[winner]],'Winning value stays eight-wide and source-paired.');
  }
  // Same eight source coordinates in the next section, before and after weighting.
  for(const id of ['s05-vals','s05-return-value','s06-mix','s06-card-mix']){
    const headers=id==='s06-mix'
      ?await page.locator('#s06-mix .s06-value-axis').allTextContents()
      :(await page.locator('#'+id+' thead th:not(.dt-comp):not(.dt-lead)').allTextContents()).slice(1);
    assert.deepEqual(headers.map(s=>s.trim().toLowerCase()),valueAxes);
  }
  for(const id of valueFrames){
    for(const build of id==='s06-frame-weighted-values'?[0,1,2,3,1,0,2,3]:id==='s05-frame-values'?[0]:['s05-frame-return','s05-frame-payload'].includes(id)?[0,1,2,0,2]:[0,1]){
      await go(id,build);await page.screenshot({path:path.join(shots,id+'-'+build+'.png')});
      if(id==='s06-frame-weighted-values'){
        const view=await page.locator('#s06-mix .s06-mix-value').evaluateAll(cells=>cells.map(cell=>{
          const raw=cell.querySelector('.s06-raw'),weighted=cell.querySelector('.s06-weight');
          return {raw:getComputedStyle(raw).display!=='none',weighted:getComputedStyle(weighted).display!=='none'&&getComputedStyle(weighted).visibility==='visible'};
        }));
        assert(view.every(v=>v.raw===(build<2)&&v.weighted===(build>=2)),'Original and weighted entries have distinct reveal stages.');
        const headers=await page.locator('#s06-mix .s06-weight-label').evaluateAll(es=>es.map(e=>({
          shown:getComputedStyle(e).display!=='none'&&getComputedStyle(e).visibility==='visible',
          math:e.querySelector('annotation').textContent,
          alphaColor:getComputedStyle(e.querySelector('.m-a')).color,
          axis:e.nextElementSibling.textContent,
          axisColor:getComputedStyle(e.nextElementSibling).color,
          clipped:[e,e.nextElementSibling].some(label=>{
            const box=e.closest('th').getBoundingClientRect(),range=document.createRange();range.selectNodeContents(label);
            return [...range.getClientRects()].some(r=>r.width>0&&(r.left<box.left-1||r.right>box.right+1));
          })
        })));
        assert.equal(headers.length,8);
        assert.deepEqual(headers.map(h=>h.axis),valueAxes);
        assert(headers.every(h=>h.shown===(build>=2)&&h.math==='\\va{\\alpha_j}\\times'&&!h.clipped),'Every header gains alpha_j times at the same stage as its weighted cells, and reverses without clipping.');
        const alphaColor=await page.locator('#s06-mix th.dt-lead .m-a').evaluate(e=>getComputedStyle(e).color);
        const valueColor=await page.locator('#s05-role-value-label .m-v').evaluate(e=>getComputedStyle(e).color);
        assert(headers.every(h=>h.alphaColor===alphaColor&&h.axisColor===valueColor),'Weights stay rose and feature names stay teal.');
      }
    }
  }
  const setSoft=async(soft)=>{await go('s06-frame-weights');const button=page.locator('#s06-mode button');if((await button.getAttribute('aria-pressed')==='true')!==soft)await button.click();};
  const close=(actual,expected,label)=>{assert.equal(actual.length,expected.length,label);actual.forEach((x,i)=>assert(Math.abs(x-expected[i])<=.000501,label+' coordinate '+i));};
  const baseScores=keys.map(k=>Number(k.reduce((s,x,c)=>s+x*queries[0][c],0).toFixed(1)));
  for(const test of [
    {name:'hard',scores:baseScores,soft:false,tau:1},
    {name:'soft',scores:baseScores,soft:true,tau:1},
    {name:'sharp',scores:baseScores,soft:true,tau:.1},
    {name:'broad',scores:baseScores,soft:true,tau:4}
  ]){
    await setSoft(test.soft);
    await go('s06-frame-scores');
    assert.deepEqual((await page.locator('#s06-dot tbody td.dt-comp').allTextContents()).slice(1).map(Number),baseScores,'All retrieval modes retain the original query–key scores.');
    assert.equal((await page.locator('#s06-score-example annotation').allTextContents()).join(' '),arithmetic,'The original worked example stays fixed.');
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
    assert.equal(await page.locator('#s06-verdict').count(),0,'The result slide omits the redundant mode/threshold status.');
    assert(!/contribute at least|Largest weight:/.test(await copy('s06-frame-message')),'No arbitrary contribution cutoff appears on the result slide.');
    assert(await page.locator('#s06-mread').isVisible(),'Keep the coordinate interpretation visible.');
    assert((await page.locator('#s06-mread').textContent()).includes('what comes back is'),'Keep the returned-feature explanation in every retrieval mode.');
    assert.equal(await page.evaluate(()=>AT.present.fitReport().overflow),false);
  }
  await setSoft(false);
  await go('s06-frame-message',1);await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('#s06 .frame.is-live').getAttribute('id'),'s06-frame-temperature','Next skips the removed slide.');
  assert.equal(await page.locator('#s06-mode button').getAttribute('aria-pressed'),'true','Temperature still enters soft mode at its new frame index.');
  assert(await page.locator('#s06-temp input').isEnabled());
  await page.evaluate(()=>AT.present.prev());
  assert.equal(await page.locator('#s06 .frame.is-live').getAttribute('id'),'s06-frame-message','Previous returns directly to the result.');
  await go('s06-frame-temperature',1);await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s07-frame-return-to-sentence','The final retrieval slide still hands off to the sentence example.');
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
  assert.equal(await page.locator('#s06-mix .s06-weight-label:visible').count(),8,'Reading mode labels the weighted contributions it displays.');
  for(const id of [...queryFrames,...keyFrames,...valueFrames,'s05-frame-three-jobs','s05-frame-contact-example','s05-frame-pronoun-example','s06-frame-hard-retrieval','s06-frame-scores']){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone document overflow.');
    await page.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  for(let i=0;i<queries.length;i++){
    await page.locator('#s05-payload-qbtns button').nth(i).click();
    const payloadBoxes=await page.locator('#s05-frame-payload .payload-layout>div').evaluateAll(es=>es.map(e=>{const b=e.getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,bottom:b.bottom};}));
    assert(payloadBoxes.every(b=>b.left>=0&&b.right<=391)&&payloadBoxes[1].top>payloadBoxes[0].bottom,'The phone shows the explanation before its full value chart.');
    for(const part of ['payload-explanation','payload-numbers']){
      const block=page.locator('#s05-frame-payload .'+part);
      await block.scrollIntoViewIfNeeded();await page.waitForTimeout(250);
      await block.screenshot({path:path.join(shots,'phone-payload-'+i+'-'+part+'.png')});
    }
    await page.locator('#s05-role-qbtns button').nth(i).click();
    const boxes=await page.locator('#s05-frame-return .role-card').evaluateAll(cards=>cards.map(e=>{const b=e.getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,bottom:b.bottom,visible:getComputedStyle(e).visibility};}));
    assert(boxes.every(b=>b.left>=0&&b.right<=391&&b.visible==='visible'),'All three cards remain visible and inside phone width.');
    assert(boxes[1].top>boxes[0].bottom&&boxes[2].top>boxes[1].bottom,'Phone cards follow query/key/value reading order.');
    // Capture each card in the real viewport. Expanding Chromium's viewport for
    // the entire tall grid can leave off-screen SVG text incompletely painted.
    for(const role of ['q','k','v']){
      const card=page.locator('#s05-frame-return .role-card.obj-'+role);
      await card.scrollIntoViewIfNeeded();await page.waitForTimeout(250);
      await card.screenshot({path:path.join(shots,'phone-role-'+i+'-'+role+'.png')});
    }
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
  console.log('PASS: four expanded source explanations with exact returned vectors and linked content features; query/key lessons and 24 scores; four concrete Q/K/V recaps with exact SVG vectors, paired sources, scaled bars and staged reveals; eight named value axes, all 48 stored values, 192 weighted products, 32 mixture coordinates, hard/soft/temperature cases, removed score-change slide and forward/back transitions, independent state, desktop and phone layout. Screenshots: '+shots);
}finally{await browser.close();}

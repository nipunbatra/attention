// Concrete role examples and the independent two-feature projection example.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const dir of fs.readdirSync(cache))candidates.push(path.join(cache,dir,'node_modules/playwright'));
let pw;for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
assert(pw,'Use an existing Playwright runtime.');
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-key-value-'));
const ids=['s05-frame-three-jobs','s05-frame-contact-example','s05-frame-pronoun-example','s05-frame-value-axes','s05-frame-values','s11-frame-separate-maps'];
const browser=await pw.chromium.launch();
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const model=await page.evaluate(()=>JSON.stringify({model:AT.model,p:AT.forward(AT.sentences.river).probs}));
  async function go(id,build=0){
    await page.evaluate(({id,build})=>{const e=document.getElementById(id),sec=e.closest('.sec');AT.present.go(sec.id,[...sec.querySelectorAll('.frame')].indexOf(e)+1,build);},{id,build});
    await page.waitForTimeout(350);
    const fit=await page.evaluate(()=>AT.present.fitReport());
    assert(!fit.overflow,id+': '+JSON.stringify(fit));
  }
  await page.evaluate(()=>AT.present.enter());
  for(const id of ids){
    for(const build of [0,1,2,0,2]){
      await go(id,build);
      const staged=await page.locator('#'+id+' [data-build]').evaluateAll(es=>es.filter(e=>!e.closest('.companion')).map(e=>({build:+e.dataset.build,visible:getComputedStyle(e).visibility==='visible'})));
      assert(staged.every(e=>e.visible===(e.build<=build)),id+' reveal order');
    }
    await page.screenshot({path:path.join(shots,id+'.png')});
  }
  const copy=await page.locator('#s11-frame-separate-maps').innerText();
  assert(copy.includes('their entries can differ')&&copy.includes('raw scores'));
  const inputs=[[1,0],[0,1]],wq=[0,1],wk=[1,0],dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const projected=inputs.map(e=>[dot(e,wq),dot(e,wk)]);
  assert.deepEqual(projected,[[0,1],[1,0]]);
  const displayed=await page.locator('#s11-frame-separate-maps tbody tr').evaluateAll(rows=>rows.map(row=>[...row.querySelectorAll('td')].map(cell=>JSON.parse(cell.textContent))));
  assert.deepEqual(displayed,inputs.map((e,i)=>[e,[projected[i][0]],[projected[i][1]]]));
  assert.equal(projected[1][0]*projected[0][1],1);
  assert.equal(projected[0][0]*projected[1][1],0);
  const formulas=await page.locator('#s11-frame-separate-maps annotation').allTextContents();
  assert(formulas.some(s=>s.includes('q_{\\text{she}}')&&s.includes('=1')&&s.includes('=0')));
  for(const role of ['q','k']){
    const colour=await page.locator('#s11-frame-separate-maps .sym-'+role).first().evaluate(e=>getComputedStyle(e).color);
    const math=await page.locator('#s11-frame-separate-maps .katex-html .m-'+role).evaluateAll(es=>es.map(e=>getComputedStyle(e).color));
    assert(math.length&&math.every(c=>c===colour),'Roles and equations share colours.');
  }
  for(const [id,next]of [['s05-frame-three-jobs','s05-frame-contact-example'],['s05-frame-contact-example','s05-frame-pronoun-example'],['s05-frame-pronoun-example','s05-frame-value-axes'],['s11-frame-separate-maps','s11-frame-query-calc']]){
    await go(id,99);await page.evaluate(()=>AT.present.next());
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),next);
  }
  const order=await page.locator('.frame').evaluateAll(es=>es.map(e=>e.id));
  assert(order.indexOf('s09-frame-prediction-output')<order.indexOf('s11-frame-separate-maps'),'Complete the forward flow before explaining projection matrices.');
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,p:AT.forward(AT.sentences.river).probs})),model);
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  for(const id of ids){
    const frame=page.locator('#'+id);await frame.scrollIntoViewIfNeeded();await page.waitForTimeout(300);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No phone document overflow.');
    const boxes=await frame.locator('.role-example-table,.s11-role-table,blockquote,.example-source,.dt-scroll').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right};}));
    assert(boxes.every(r=>r.left>=-1&&r.right<=391),id+' phone containment');
    await frame.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: video/transcript, contact and pronoun examples; compatible query/key widths; independent directional projection arithmetic; staged and forward navigation; unchanged bank model; desktop and phone layout. Screenshots: '+shots);
}finally{await browser.close();}

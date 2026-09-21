// Lecture 2 topic pauses: placement, navigation, one title and responsive layout.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const shots=fs.mkdtempSync('/private/tmp/topic-breaks-');
const targets=[
  ['s02-topic-break','s02-frame-problem'],
  ['s05-values-topic-break','s05-frame-three-jobs'],
  ['s06-topic-break','s06-frame-hard-retrieval'],
  ['s09-topic-break','s09-frame1'],
  ['s11-topic-break','s11-frame1'],
  ['s12-topic-break','s12-frame-scaling'],
  ['s13-topic-break','s13-frame1'],
  ['s14-topic-break','s14-routing'],
  ['s16-topic-break','s16-flow-frame'],
  ['s16-cost-break','s16-cost-symbols'],
  ['s17-position-break','s17-position-order'],
  ['s19-topic-break','s19-frame-generation'],
  ['s19-pipeline-break','s19-pipeline-mlp']
];
const browser=await chromium.launch();
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const url=pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href;
  await page.goto(url);await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify({model:AT.model,result:AT.forward(AT.sentences.river)}));
  await page.evaluate(()=>AT.present.enter());
  assert.deepEqual(await page.locator('.lecture-topic-break').evaluateAll(es=>es.map(e=>e.id)),targets.map(t=>t[0]));
  async function go(id,build=0){
    await page.evaluate(({id,build})=>{
      const e=document.getElementById(id),s=e.closest('.sec');
      AT.present.go(s.id,[...s.querySelectorAll('.frame')].indexOf(e)+1,build);
    },{id,build});
    await page.waitForTimeout(200);
  }
  for(const [id,next]of targets){
    await go(id);
    const frame=page.locator('#'+id);
    const before=await frame.evaluate(e=>{const all=[...document.querySelectorAll('.frame')];return all.indexOf(e)-1;});
    assert.equal(await frame.locator('[data-build],.stepper').count(),0,'A topic pause has no hidden teaching steps.');
    assert.equal(await frame.locator('h3').textContent(),await frame.getAttribute('data-title'));
    assert.equal(await frame.locator('p').count(),2,'One recap and one guiding question.');
    assert(await frame.locator('script').textContent(),'Presenter notes identify the pause.');
    assert.equal(await frame.evaluate(e=>getComputedStyle(e.closest('.sec').querySelector('.sec-head')).display),'none','Only the topic title appears.');
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' fits');
    await page.screenshot({path:path.join(shots,id+'.png')});
    await page.evaluate(()=>AT.present.next());await page.waitForTimeout(120);
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),next);
    assert.notEqual(await page.locator('#'+next).evaluate(e=>getComputedStyle(e.closest('.sec').querySelector('.sec-head')).display),'none','Restore the normal slide title after a pause.');
    await page.evaluate(()=>AT.present.prev());
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id,'Previous returns to the topic pause.');
    await page.evaluate(()=>AT.present.prev());
    assert.equal(await page.locator('.frame.is-live').evaluate(e=>[...document.querySelectorAll('.frame')].indexOf(e)),before);
    await page.evaluate(()=>AT.present.next());
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id,'Completing the preceding slide reaches the pause.');
  }
  // Existing useful breaks remain, and calculations are not interrupted midway.
  for(const id of ['s04-frame-summary-break','s05-frame-attention-break','s07-frame-return-to-sentence']){
    await go(id);assert(await page.locator('#'+id).isVisible());
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow);
    await page.screenshot({path:path.join(shots,id+'.png')});
  }
  assert.equal(await page.locator('#s08 .lecture-topic-break,#s10 .lecture-topic-break,#s15 .lecture-topic-break').count(),0);
  await page.goto(url+'?present#s12/1/0');await page.waitForTimeout(300);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s12-topic-break','The section link opens at its new topic break.');
  await page.evaluate(()=>AT.present.next());
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s12-frame-scaling');
  assert.equal(new URL(page.url()).hash,'#s12/2/0');
  await page.reload();await page.waitForTimeout(300);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s12-frame-scaling','A resumed example deep link remains exact.');
  await page.setViewportSize({width:1250,height:1041});await go('s12-topic-break');
  assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow);
  await page.screenshot({path:path.join(shots,'scaling-tall.png')});
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth+1);
  for(const [id]of targets){
    const frame=page.locator('#'+id);await frame.scrollIntoViewIfNeeded();
    assert.equal(await frame.locator('h3').isVisible(),['s05-values-topic-break','s16-cost-break','s17-position-break'].includes(id),'In reading mode, only mid-section breaks need an additional heading.');
    assert(await frame.locator('.topic-question').isVisible());
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    const clipped=await frame.locator('h3,p').evaluateAll(es=>es.filter(e=>e.clientWidth&&e.scrollWidth>e.clientWidth+1).map(e=>e.textContent));
    assert.deepEqual(clipped,[],id+' phone text fits');
    await frame.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,result:AT.forward(AT.sentences.river)})),original);
  assert.deepEqual(errors,[]);
  console.log('PASS: '+targets.length+' topic breaks, existing pauses, forward/back navigation, header restoration, scaling deep links, unchanged model, desktop/tall/phone layouts. Screenshots: '+shots);
}finally{await browser.close();}

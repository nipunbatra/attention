// node src/check_part2_opening.mjs [attention.html]
// Check the two Part 1 recaps and three original-paper orientation slides.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const dir of fs.readdirSync(cache))candidates.push(path.join(cache,dir,'node_modules/playwright'));
let pw;
for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
assert(pw,'Use an existing Playwright installation.');
const model1=JSON.parse(fs.readFileSync(new URL('./toy1.json',import.meta.url),'utf8'));
const file=path.resolve(process.argv[2]||'attention.html');
const url=pathToFileURL(file).href;
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'part2-opening-'));
const frames=[['s01-recap-model',2],['s01-recap-context',1],['s01-paper-intro',1],['s01-paper-importance',2],['s01-paper-architecture',1]];
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
try{
  await page.goto(url);await page.evaluate(()=>document.fonts.ready);
  await page.waitForFunction(()=>[...document.querySelectorAll('#s01 .paper-layout img')].every(im=>im.complete&&im.naturalWidth>0));
  const original=await page.evaluate(()=>JSON.stringify({model:AT.model,result:AT.forward(AT.sentences.river)}));
  assert.deepEqual(await page.locator('#s01 .frame').evaluateAll(es=>es.slice(0,6).map(e=>e.id)),[...frames.map(f=>f[0]),'s01-frame1']);
  const recap=await page.locator('#s01-recap-model').innerText();
  assert.equal(model1.W1.length/model1.E[0].length,3);
  assert(recap.includes('3 × '+model1.E[0].length+' = '+model1.W1.length));
  assert(recap.includes(model1.W1[0].length+' hidden units + ReLU'));
  assert(recap.includes(model1.vocab.length+' scores'));
  assert(recap.includes('a a b')&&recap.includes('aabid'));
  const bridge=await page.locator('#s01-recap-context').innerText();
  for(const phrase of ['Cross-entropy','backpropagation','parameters fixed','shift the window','MLP input wider','hand-chosen'])assert(bridge.includes(phrase),phrase);
  const paper=await page.locator('#s01-paper-intro').innerText();
  assert(paper.includes('2017')&&paper.includes('Attention already existed')&&paper.includes('arXiv v7'));
  const architecture=await page.locator('#s01-paper-architecture').innerText();
  for(const phrase of ['Left: encoder','Right: decoder','not this whole translation model'])assert(architecture.includes(phrase),phrase);
  const sources=await page.locator('#s01-paper-importance a').evaluateAll(es=>es.map(e=>e.href));
  assert(sources.some(u=>u==='https://aclanthology.org/N19-1423/'));
  assert(sources.some(u=>u.endsWith('language_understanding_paper.pdf')));
  const images=await page.locator('#s01 .paper-layout img').evaluateAll(es=>es.map(e=>({src:e.getAttribute('src'),w:e.naturalWidth,h:e.naturalHeight,alt:e.alt,link:e.parentElement.getAttribute('href'),caption:e.closest('figure').querySelector('figcaption').textContent})));
  assert.equal(images.length,2);
  for(const im of images){
    assert(fs.existsSync(path.resolve(path.dirname(file),im.src)));
    assert.equal(im.src,im.link,'Each original image opens at full resolution.');
    assert(im.w>=1300&&im.h>=1800,'Keep sufficient source resolution.');
    assert(im.alt.length>70&&im.caption.includes('Vaswani'),'Keep alt text and attribution.');
  }
  async function go(id,build){
    await page.evaluate(({id,build})=>{
      AT.present.enter();const e=document.getElementById(id),s=e.closest('.sec');
      AT.present.go(s.id,[...s.querySelectorAll('.frame')].indexOf(e)+1,build);
    },{id,build});
    await page.waitForTimeout(220);
  }
  for(const viewport of [{width:1280,height:720},{width:1250,height:1041}]){
    await page.setViewportSize(viewport);
    for(const [id,max]of frames){
      for(const build of [...Array(max+1).keys(),...Array(max).keys()].concat(max)){
        await go(id,build);
        assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' fits at build '+build);
        assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id);
        const visibility=await page.locator('#'+id+' [data-build]').evaluateAll(es=>es.map(e=>({level:Number(e.dataset.build),visible:getComputedStyle(e).visibility!=='hidden'&&getComputedStyle(e).display!=='none'})));
        for(const item of visibility)assert.equal(item.visible,item.level<=build,id+' reveal '+item.level);
      }
      await page.screenshot({path:path.join(shots,id+'-'+viewport.width+'.png')});
    }
  }
  for(let i=0;i<frames.length;i++){
    const [id,max]=frames[i];await go(id,max);
    await page.evaluate(()=>AT.present.next());
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),frames[i+1]?.[0]||'s01-frame1');
    await page.evaluate(()=>AT.present.prev());
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id);
  }
  await page.goto(url+'?present#s01/1/0');await page.waitForTimeout(350);
  assert.equal(await page.locator('.frame.is-live').getAttribute('id'),'s01-recap-model','The beginning URL opens with the recap.');
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth+1);
  for(const [id]of frames){
    const frame=page.locator('#'+id);await frame.scrollIntoViewIfNeeded();
    const clipped=await frame.locator('p,td,th,figcaption,img').evaluateAll(es=>es.filter(e=>e.clientWidth&&e.scrollWidth>e.clientWidth+1).map(e=>e.textContent));
    assert.deepEqual(clipped,[],id+' phone contents fit');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await frame.screenshot({path:path.join(shots,'phone-'+id+'.png')});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,result:AT.forward(AT.sentences.river)})),original);
  assert.deepEqual(errors,[]);
  console.log('PASS: two model-grounded recaps, three sourced paper slides, original local images, reveal/navigation/deep-link checks, unchanged model, desktop/tall/phone layouts. Screenshots: '+shots);
}finally{await browser.close();}

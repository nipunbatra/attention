import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const browser=await chromium.launch();
const shots=fs.mkdtempSync('/tmp/browser-lab-slides-');
try{
  const page=await browser.newPage({viewport:{width:1280,height:720}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve('attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const ids=await page.locator('.lab-comparison').evaluateAll(es=>es.map(e=>e.id));
  assert.equal(ids.length,9);
  assert.match(await page.locator('#s19-lab-scores').textContent(),/51.59.*31.34.*28.78/s);
  assert.match(await page.locator('#s19-lab-ppl').textContent(),/geometric mean/);
  assert.equal(await page.locator('#s19-lab-play .lab-live-link').getAttribute('href'),'word-lab/');
  assert.equal(await page.locator('#s19-lab-ppl .katex').count(),2);
  const indices={};await page.evaluate(()=>AT.present.enter());
  for(const viewport of [{width:1280,height:720},{width:1024,height:768}]){
    await page.setViewportSize(viewport);
    for(const id of ids){
      const index=await page.evaluate(id=>{
        const el=document.getElementById(id),sec=el.closest('.sec');
        const index=[...sec.querySelectorAll('.frame')].indexOf(el)+1;
        AT.present.go(sec.id,index,99);return index;
      },id);
      indices[id]=index;await page.waitForTimeout(60);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' fits '+viewport.width);
      if(viewport.width===1280)await page.screenshot({path:shots+'/'+id+'.png'});
    }
  }
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'reading tables scroll within their panels');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({frames:ids.length,indices,shots},null,2));
}finally{await browser.close();}

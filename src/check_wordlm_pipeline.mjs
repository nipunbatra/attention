// Four master maps, code-linked focus views, and presentation/reading layouts.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const browser=await chromium.launch();
const shots=fs.mkdtempSync('/private/tmp/wordlm-pipeline-');
try{
  const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const url=pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href;
  await page.goto(url);await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify(AT.model));
  const ids=await page.locator('.pipeline-lesson').evaluateAll(es=>es.map(e=>e.id));
  const manifest=JSON.parse(fs.readFileSync(new URL('../notebooks/wordlm/lesson-manifest.json',import.meta.url)));
  assert.equal(ids.length,manifest.length+1);
  assert.equal(await page.locator('template[id^="pipeline-"]').count(),4);
  assert.equal(await page.locator('.pipeline-lesson script[type="text/x-notes"]').count(),ids.length);
  assert.match(await page.locator('#s19-pipeline-boundaries').innerText(),/7 supervised targets/);
  assert.match(await page.locator('#s19-pipeline-mask script').textContent(),/no future columns/);
  for(const id of ['story-complete','story-excerpts']){
    const frame=page.locator('#s19-pipeline-'+id);
    assert.equal(await frame.locator('pre').count(),0,'story slides prioritize reading the data');
    assert.equal(await frame.locator('.story-credit a').count(),2,'dataset and license attribution');
  }
  assert.match(await page.locator('#s19-pipeline-story-complete').innerText(),/52 words · 60 tokens/);
  assert.match(await page.locator('#s19-pipeline-story-excerpts').innerText(),/248 words · 300 tokens/);
  assert.equal(await page.locator('#s19-pipeline-tokenization-intro.lecture-topic-break h3').innerText(),'Tokenization');
  assert.equal(await page.locator('.tokenization-lesson').count(),3);
  assert.equal(await page.locator('.tokenization-lesson pre').count(),0,'conceptual detour precedes implementation');
  assert.match(await page.locator('#s19-pipeline-tokenization-choices').innerText(),/Actual splits depend on the tokenizer/);
  assert.match(await page.locator('#s19-pipeline-tokenization-rules').innerText(),/training and generation/);
  const summary=JSON.parse(fs.readFileSync(new URL('../figures/wordlm-pipeline/benchmark-summary.json',import.meta.url)));
  for(const kind of ['mlp','attention'])assert.match(await page.locator('#s19-pipeline-benchmark').innerText(),new RegExp(summary.aggregate[kind].test_perplexity.mean.toFixed(2).replace('.','\\.')));
  await page.evaluate(()=>AT.present.enter());
  async function go(id){
    await page.evaluate(id=>{
      const el=document.getElementById(id),sec=el.closest('.sec');
      AT.present.go(sec.id,[...sec.querySelectorAll('.frame')].indexOf(el)+1,99);
    },id);
    await page.waitForTimeout(80);
  }
  for(const viewport of [{width:1280,height:720},{width:1024,height:768}]){
    await page.setViewportSize(viewport);
    for(const id of ids){
      await go(id);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' fits '+viewport.width);
      const outside=await page.locator('#'+id+' [data-stage]').evaluateAll(gs=>gs.flatMap(g=>{
        const r=g.querySelector('rect').getBBox();
        return [...g.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<r.x-1||b.x+b.width>r.x+r.width+1;}).map(t=>t.textContent);
      }));
      assert.deepEqual(outside,[],id+' SVG labels stay inside their nodes');
      const clipped=await page.locator('#'+id+' .step-figure svg').evaluateAll(es=>es.flatMap(s=>{
        const v=s.viewBox.baseVal;
        return [...s.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;}).map(t=>t.textContent);
      }));
      assert.deepEqual(clipped,[],id+' SVG text stays within the figure');
      const stage=manifest.find(s=>'s19-pipeline-'+s.id===id);
      if(stage)assert.equal(await page.evaluate(()=>AT.present.state().frame.index+1),stage.slide,'notebook link matches slide index');
      const button=page.locator('#'+id+' .pipeline-toggle');
      if(await button.count()){
        const svg=page.locator('#'+id+' .pipeline-map svg');const before=await svg.getAttribute('viewBox');
        await button.click();assert.equal(await button.getAttribute('aria-expanded'),'true');
        assert.notEqual(await svg.getAttribute('viewBox'),before);
        assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' expanded');
        await button.click();assert.equal(await svg.getAttribute('viewBox'),before);
      }
      await page.screenshot({path:path.join(shots,id+'-'+viewport.width+'.png')});
    }
  }
  assert.equal(await page.evaluate(()=>JSON.stringify(AT.model)),original,'diagrams do not mutate the hand-chosen lecture model');
  assert.equal(await page.locator('.pipeline-lesson .katex-error').count(),0);
  await page.goto(url);await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(150);
  assert(await page.locator('#s19-pipeline-tokenization-intro .tokenization-mobile').isVisible(),'intro prose wraps on phones');
  assert(!(await page.locator('#s19-pipeline-tokenization-intro svg').isVisible()),'wide intro SVG has a readable phone equivalent');
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile overflow stays inside diagrams');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,frames:ids.length,viewports:[1280,1024,390],screenshots:shots}));
}finally{await browser.close();}

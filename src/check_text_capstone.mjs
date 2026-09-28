// Brief head introduction plus the restored TinyStories teaching route.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const manifest=JSON.parse(fs.readFileSync('figures/multihead/manifest.json'));
const data=JSON.parse(fs.readFileSync('notebooks/wordlm/artifacts/heads/comparison.json'));
const browser=await chromium.launch();
const shots=fs.mkdtempSync('/private/tmp/text-capstone-');
try {
  const page=await browser.newPage({viewport:{width:1280,height:720}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve('part3.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  assert.equal(manifest.length,80);
  assert.deepEqual(await page.locator('.mh-frame').evaluateAll(es=>es.map(e=>e.id)),manifest.map(s=>s.key));
  assert.equal(await page.locator('#s01 .frame,#s02 .frame').count(),7,'Finish head overview in seven frames');
  assert.equal(await page.locator('#s02 .frame').last().getAttribute('id'),'s02-cap-map');
  assert.equal(await page.locator('.mh-frame').last().getAttribute('id'),'s06-cap-app');
  assert.equal(await page.locator('#s02-cap-map [data-map-head]').count(),2);
  assert.equal(await page.locator('#s02-cap-map [data-map-message]').count(),2);
  assert.equal(await page.locator('#s03 .frame').count(),54,'Every setup step has a full-map/code checkpoint and its preserved example');
  assert.equal(await page.locator('#s04 .frame').count(),13,'Model setup and all forward, training and generation operations');
  const setup=['data','story-complete','story-excerpts','split','sentence','tokenization-intro','tokenization-choices','tokenization-rules','tokenize','vocabulary','special','boundaries','story-indices','one-pair','pair-first','pair-second','windows-first','windows-last','pairs-tensors','counts-story','counts','context','batch','batch-ids','batches'];
  assert.deepEqual((await page.locator('#s03 .frame').evaluateAll(es=>es.map(e=>e.id))).slice(0,50),setup.flatMap(s=>['s03-map-'+s,'s03-data-'+s]));
  const maps = page.locator('svg[data-pipeline="tinystories"]');
  assert.equal(await maps.count(),40,'Full map on all 27 setup and 13 model checkpoints');
  const structure = await maps.evaluateAll(svgs=>svgs.map(svg=>({
    nodes:[...svg.querySelectorAll('[data-stage]')].map(g=>[g.dataset.stage,...['x','y','width','height'].map(a=>g.querySelector('rect').getAttribute(a))]),
    edges:[...svg.querySelectorAll('[data-edge]')].map(g=>[g.dataset.edge,g.getAttribute('d')]),
    focus:svg.dataset.focus.split(' '),
    highlighted:[...svg.querySelectorAll('[data-active="true"]')].map(g=>g.dataset.stage),
    code:svg.closest('.frame').querySelector('pre code')?.textContent,
  })));
  for(const m of structure){
    assert.deepEqual(m.nodes,structure[0].nodes,'Node positions stay fixed');
    assert.deepEqual(m.edges,structure[0].edges,'Connections stay fixed');
    assert.deepEqual([...m.focus].sort(),[...m.highlighted].sort());
    assert(m.code?.trim(),'Every map has the matching code');
  }
  assert.deepEqual(await page.locator('#s04-cap-embed svg').getAttribute('data-focus'),'qkv0 qkv1 qkv2 qkv3');
  assert((await page.locator('#s04-cap-embed pre').textContent()).includes('reshape(B, T, 4, 16).transpose(1, 2)'));
  for(const key of setup){
    const caption=await page.locator('#s03-data-'+key+' svg title').textContent();
    assert(caption.length>0,'Original figure survives: '+key);
  }
  assert((await page.locator('#s03-data-vocabulary').textContent()).includes('Beginning of sequence'));
  assert((await page.locator('#s03-data-boundaries').textContent()).includes('whole story'));
  assert((await page.locator('#s04-cap-prompt pre').textContent()).includes('"<BOS>"'));
  assert.equal(await page.locator('#s02-v-head1-matrices,#s07-v-patches,#s06-v-model-mlp').count(),0,'Long calculations and other model walkthroughs are off the lecture path');
  assert((await page.locator('#s02-cap-roles > p').textContent()).split(/\s+/).length<22,'Keep the roles recap terse');
  for(const [kind,color] of [['mlp','e'],['attention','k'],['multihead','d']]) {
    const trace=data.runs[kind].find(r=>r.seed===11).trace;
    const actual=await page.locator(`[data-curve="${kind}"] circle`).evaluateAll(es=>es.map(e=>({step:+e.dataset.step,loss:+e.dataset.loss})));
    assert.deepEqual(actual,trace.map(p=>({step:p.step,loss:p.validation_loss})));
    const copy=await page.locator('#s05-cap-scores').textContent();
    assert(copy.includes(data.aggregate[kind].test_loss.mean.toFixed(3)));
    assert(copy.includes(data.aggregate[kind].test_perplexity.mean.toFixed(2)));
  }
  const links=await page.locator('a[href]').evaluateAll(es=>es.map(e=>e.href));
  for(const link of links.filter(l=>l.startsWith('file:'))) {
    const url=new URL(link);assert(fs.existsSync(fileURLToPath(url)),'Local link exists: '+link);
  }
  assert.equal(await page.locator('#s06-cap-app a[href="word-lab/"]').count(),1);
  assert.equal(await page.locator('#s06-cap-app a[href="word-lab/#results"]').count(),1);
  const geometry=await page.locator('.mh-figure svg').evaluateAll(svgs=>svgs.flatMap(svg=>{
    const vb=svg.viewBox.baseVal;
    return [...svg.querySelectorAll('text')].flatMap(t=>{
      const b=t.getBBox(),cell=t.closest('[data-cell-left]');
      const rect=t.parentElement.localName==='g'?t.parentElement.querySelector(':scope > rect'):null;
      const left=cell?+cell.dataset.cellLeft:rect?+rect.getAttribute('x'):0;
      const right=cell?+cell.dataset.cellRight:rect?left+(+rect.getAttribute('width')):vb.width;
      return b.x<left-1||b.x+b.width>right+1||b.y<0||b.y+b.height>vb.height ? [{id:svg.closest('.frame').id,text:t.textContent,left,right,x:b.x,width:b.width}] : [];
    });
  }));
  assert.deepEqual(geometry,[],'SVG text fits its own node and viewBox');
  await page.evaluate(()=>AT.present.enter());
  for(const viewport of [{width:1280,height:720},{width:995,height:1031},{width:760,height:1041}]) {
    await page.setViewportSize(viewport);
    for(const step of manifest) {
      await page.evaluate(id=>{
        const el=document.getElementById(id),sec=el.closest('.sec');
        const index=[...sec.querySelectorAll('.frame')].indexOf(el)+1;
        AT.present.go(sec.id,index,0);
      },step.key);
      assert.equal(await page.locator('.frame.is-live').getAttribute('id'),step.key);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,step.key+' fits '+viewport.width);
      if(viewport.width===1280||['s02-cap-map','s03-map-data','s03-map-vocabulary','s03-map-windows-first','s03-map-real-windows','s03-data-story-complete','s03-data-vocabulary','s03-data-pairs-tensors','s03-cap-scale','s04-cap-embed','s04-cap-weights','s05-cap-curves','s06-cap-samples'].includes(step.key))
        await page.screenshot({path:path.join(shots,step.key+'-'+viewport.width+'.png')});
    }
  }
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.goto(pathToFileURL(path.resolve('attention.html')).href);
  assert.equal(await page.locator('.pipeline-lesson,.lab-comparison').count(),0,'TinyStories is no longer in Part II');
  assert.equal(await page.locator('#s19-pipeline-break a[href="part3.html#s03"]').count(),1);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({frames:manifest.length,shots,verified:'saved metrics, traces, links, relocation, SVG geometry and 3 presentation sizes'}));
} finally {await browser.close();}

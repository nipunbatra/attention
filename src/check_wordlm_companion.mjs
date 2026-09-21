// The public notebook guide must work without the author's private lab folder.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const dir of fs.readdirSync(cache))candidates.push(path.join(cache,dir,'node_modules/playwright'));
let pw;for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
assert(pw,'Use an existing Playwright runtime.');
const root=path.resolve('notebooks/wordlm');
const manifest=JSON.parse(fs.readFileSync(path.join(root,'lesson-manifest.json')));
const book=JSON.parse(fs.readFileSync(path.join(root,'05_training_and_inference_maps.ipynb')));
assert.equal(manifest.length,47);
assert.equal(book.cells.filter(c=>c.cell_type==='code').length,48);
assert(book.cells.filter(c=>c.cell_type==='code').every(c=>c.execution_count!==null));
assert(!book.cells.some(c=>c.outputs?.some(o=>o.output_type==='error')));
const browser=await pw.chromium.launch();
const errors=[];
const shots=fs.mkdtempSync('/private/tmp/wordlm-companion-');
try{
  const page=await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror',e=>errors.push(e.message));
  const url=pathToFileURL(path.join(root,'05_training_and_inference_maps.html')).href;
  await page.goto(url);await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.lesson-step').count(),47);
  assert.equal(await page.locator('.chapters ol a').count(),7);
  const broken=[];
  for(const href of await page.locator('a[href]').evaluateAll(es=>es.map(e=>e.href))){
    const u=new URL(href);
    if(u.protocol==='file:'&&!fs.existsSync(fileURLToPath(u)))broken.push(href);
    if(u.pathname===new URL(url).pathname&&u.hash)assert(await page.locator(u.hash).count(),'anchor exists '+u.hash);
  }
  assert.deepEqual(broken,[],'all local download/navigation targets exist');
  for(const stage of manifest){
    const locator=page.locator('#'+stage.id);
    assert.equal(await locator.locator('figure > .figure-wrap > svg').count(),1);
    assert(await locator.locator('pre code').innerText(),'step has code');
    const numeric=await locator.locator('figure svg text').allTextContents();
    const notebookCell=book.cells.find(c=>c.metadata?.lesson_stage===stage.id);
    const svg=notebookCell.outputs.find(o=>o.data?.['image/svg+xml']).data['image/svg+xml'];
    const source=Array.isArray(svg)?svg.join(''):svg;
    const file=fs.readFileSync(path.join('figures/wordlm-pipeline/steps',stage.id+'.svg'),'utf8');
    assert.equal(source.trim(),file.trim(),'executed notebook and lecture use identical figure '+stage.id);
    assert(numeric.length>0);
  }
  const toggle=page.locator('#mix .route-map');
  await toggle.locator('summary').click();
  assert(await toggle.getAttribute('open')!==null);
  await page.locator('#batch').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(shots,'desktop-batch.png')});
  for(const width of [1280,768,390]){
    await page.setViewportSize({width,height:900});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'no page overflow at '+width);
  }
  await page.locator('#mix').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(shots,'phone-mix.png')});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({steps:47,chapters:7,executedNotebook:true,identicalFigures:47,localLinks:'pass',viewports:[1280,768,390],screenshots:shots},null,2));
}finally{await browser.close();}

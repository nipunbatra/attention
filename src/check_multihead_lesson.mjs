// Part III: independent arithmetic, SVG labels, controls, and notebook parity.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright'];
for(const dir of fs.readdirSync(path.join(os.homedir(),'.npm/_npx'))) candidates.push(path.join(os.homedir(),'.npm/_npx',dir,'node_modules/playwright'));
let pw;for(const p of candidates.filter(Boolean)){try{pw=require(p);break;}catch{}}
assert(pw,'Use an existing Playwright installation.');
const root=path.resolve('');
const expected=JSON.parse(fs.readFileSync('figures/multihead/worksheet.json'));
const manifest=JSON.parse(fs.readFileSync('figures/multihead/manifest.json'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720}});
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
let numbers=0;
function near(a,b,key){
  if(Array.isArray(b)){assert.equal(a.length,b.length,key);b.forEach((v,i)=>near(a[i],v,key+'['+i+']'));return;}
  if(b===null){assert(a===null||a===-Infinity,key);return;}
  assert(Math.abs(a-b)<1e-10,key+': '+a+' vs '+b);numbers++;
}
try{
  await page.goto(pathToFileURL(path.join(root,'part3.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.mh-frame').count(),manifest.length);
  const notation=await page.evaluate(()=>AT.ui.notationCard().textContent);
  assert(notation.includes('4×2')&&notation.includes('10×2'),'Notation distinguishes per-head and packed widths');
  assert(!notation.includes('4×3'),'No stale Part II query width');
  assert(notation.includes('Additive causal mask')&&notation.includes('message matrix'),'Mask and message notation stays consistent with Part II');
  assert(await page.locator('.mh-frame .python-code .py-call').count()>4,'Static syntax highlighting');
  assert(manifest.length<=59,'Keep the worked lesson compact, including motivation and the image-classification bridge');
  const keys=manifest.map(s=>s.key);
  for(const [key,heads] of [['s01-v-coat-one',1],['s01-v-coat-many',3],['s01-v-grammar',2],['s01-v-event',2],['s01-v-event-change',2]]){
    const frame=page.locator('#'+key);
    assert.equal(await frame.locator('[data-example-head]').count(),heads,'Illustrated head count: '+key);
    assert((await frame.textContent()).includes('not a trained model'),'Separate hypothetical roles from measured weights');
  }
  for(const [before,after] of [['s01-v-coat-one','s01-v-coat-many'],['s01-v-event','s01-v-event-change']]){
    const heldHead=key=>page.locator(`#${key} [data-example-head="1"]`).evaluate(el=>el.outerHTML);
    assert.equal(await heldHead(before),await heldHead(after),'Hold the first illustrative reading fixed while adding/changing another');
  }
  assert((await page.locator('#s01-v-grammar').textContent()).includes('usually'),'Follow a receiver after both useful source words');
  const motivationOrder=['s01-v-coat-one','s01-v-coat-many','s01-v-grammar','s01-v-event','s01-v-event-change','s01-v-shared','s01-v-independent','s01-v-coupled','s01-v-one'];
  for(let i=1;i<motivationOrder.length;i++)assert(keys.indexOf(motivationOrder[i-1])<keys.indexOf(motivationOrder[i]),'Examples, fixed-value bottleneck, then computed heads');
  assert((await page.locator('#s01-v-coupled').textContent()).includes('α = 0.8'));
  assert((await page.locator('#s01-v-coupled').textContent()).includes('α = 0.2'));
  assert.match(await page.locator('#s07-v-patches').textContent(),/16 × 4/);
  assert.match(await page.locator('#s07-v-classifier').textContent(),/no future-token mask/);
  assert.equal(await page.locator('#s07-v-classifier a').getAttribute('href'),'vision1.html');
  for(const key of ['s01-v-independent','s01-v-one','s02-v-recall','s02-v-query','s02-v-query-person','s02-v-match'])assert(keys.includes(key),'Required teaching step: '+key);
  assert(keys.indexOf('s01-v-independent')<keys.indexOf('s01-v-one'),'Explain independent mixtures before showing computed head patterns');
  assert(keys.indexOf('s02-v-recall')<keys.indexOf('s02-v-query'),'Recall query/key/value roles before the projection arithmetic');
  assert(keys.indexOf('s02-v-query-person')<keys.indexOf('s02-v-match'),'Introduce both queries before key matching');
  const arithmeticOrder=['s02-v-head1-matrices','s02-v-match','s02-v-weights',
    ...['alpha','values','products','sum'].map(phase=>'s02-v-head1-'+phase),
    's02-v-head2-matrices','s02-v-head2-dots','s02-v-head2-softmax',
    ...['alpha','values','products','sum'].map(phase=>'s02-v-head2-'+phase),'s02-v-message'];
  for(let i=0;i<arithmeticOrder.length;i++){
    assert(keys.includes(arithmeticOrder[i]),'Required arithmetic step: '+arithmeticOrder[i]);
    if(i)assert(keys.indexOf(arithmeticOrder[i-1])<keys.indexOf(arithmeticOrder[i]),'Finish each head separately before combining');
  }
  for(const head of [1,2]){
    const stages=['alpha','values','products','sum'];
    const parts=['sources','weights','values','contributions','sum'];
    const held=new Map();
    for(let level=0;level<stages.length;level++){
      const figure=page.locator(`#s02-v-head${head}-${stages[level]} svg`);
      assert.equal(await figure.locator('[data-value-phase]').getAttribute('data-value-phase'),stages[level]);
      for(const [index,part] of parts.entries()){
        const col=figure.locator(`[data-values-part="${part}"]`);
        const visible=index<=level+1;
        assert.equal(await col.count(),visible?1:0,`Head ${head}: reveal ${part} only when reached`);
        if(!visible)continue;
        const contents=await col.evaluate(el=>el.outerHTML);
        if(held.has(part))assert.equal(contents,held.get(part),`Head ${head}: keep ${part} fixed between stages`);
        else held.set(part,contents);
        if(part!=='sum'){
          assert.equal(await col.locator('[data-source-row]').count(),10,'Every source stays visible');
          const gap=await col.evaluate(el=>{
            const heading=el.querySelector(':scope > text').getBBox();
            const row=el.querySelector('[data-source-row="1"] text').getBBox();
            return row.y-heading.y-heading.height;
          });
          assert(gap>=2,`Head ${head}: ${part} heading must clear the first numeric row`);
        }
      }
    }
  }
  assert(!await page.locator('.mh-frame').evaluateAll(frames=>frames.some(f=>/0[1-3] → 0[2-4]/.test(f.textContent))),'No unexplained section-number transitions');
  const plan=await page.locator('#s02-v-plan').textContent();
  assert(plan.includes('Concatenate')&&plan.includes('Project with W')&&plan.includes('[4×4]'),'Show concatenation and output projection separately');
  for(const name of ['river','cheque']){
    const actual=await page.evaluate(name=>AT.multiheadWorksheet.compute(name),name);
    const ref=expected.headsLesson.cases[name];
    for(const key of ['E','joined','delta','updated','logits','probabilities'])near(actual[key],ref[key],name+'.'+key);
    for(let h=0;h<2;h++)for(const key of ['Q','K','V','A','messages'])near(actual.heads[h][key],ref.heads[h][key],name+'.head'+h+'.'+key);
  }
  const first=await page.locator('#s04-live-prediction').textContent();
  await page.selectOption('#s04-context','cheque');
  assert.notEqual(await page.locator('#s04-live-prediction').textContent(),first);
  await page.selectOption('#s04-head','1');
  assert.match(await page.locator('#s04-live-message').textContent(),/^Head 2:/);
  const overflows=await page.evaluate(()=>[...document.querySelectorAll('.mh-figure svg')].flatMap(svg=>{
    const vb=svg.viewBox.baseVal,frame=svg.closest('.frame').id;
    return [...svg.querySelectorAll('text')].flatMap(t=>{
      if(!t.textContent.trim())return [];
      const b=t.getBBox(),cell=t.closest('[data-cell-left]');
      const rect=t.parentElement.localName==='g'?t.parentElement.querySelector(':scope > rect'):null;
      const left=cell?+cell.dataset.cellLeft:rect?+rect.getAttribute('x'):vb.x;
      const right=cell?+cell.dataset.cellRight:rect?left+(+rect.getAttribute('width')):vb.width;
      return b.x<left-1||b.x+b.width>right+1||b.y<0||b.y+b.height>vb.height?[{frame,text:t.textContent,left,right,x:b.x,width:b.width}]:[];
    });
  }));
  assert.deepEqual(overflows,[],'SVG labels must fit their cells/boxes: '+JSON.stringify(overflows));
  const nb=JSON.parse(fs.readFileSync('notebooks/wordlm/07_multihead_step_by_step.ipynb'));
  for(const step of manifest){
    const cell=nb.cells.find(c=>c.cell_type==='markdown'&&c.source.join('').includes('id="'+step.key+'"'));
    assert(cell,'Notebook heading '+step.key);
    const file='figures/multihead/'+step.key+'.svg';
    if(fs.existsSync(file))assert(cell.source.join('').includes(fs.readFileSync(file,'utf8')),'Same embedded figure '+step.key);
  }
  for(const cell of nb.cells.filter(c=>c.cell_type==='code')){
    assert(cell.execution_count!==null,'Notebook cells executed');
    assert(!cell.outputs.some(o=>o.output_type==='error'),'No saved cell errors');
  }
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No page-level mobile overflow');
  for(const filename of ['part3.html','notebooks/wordlm/07_multihead_step_by_step.html']){
    await page.goto(pathToFileURL(path.join(root,filename)).href);
    const links=await page.locator('a[href]').evaluateAll(links=>links.map(a=>a.href));
    for(const link of links.filter(l=>l.startsWith('file:'))){
      const url=new URL(link);url.hash='';url.search='';
      assert(fs.existsSync(fileURLToPath(url)),filename+' links to missing '+link);
    }
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),filename+' mobile fit');
  }
  assert.deepEqual(errors,[]);
  console.log(`PASS: ${manifest.length} frames; ${numbers} independently checked numbers; live controls; SVG label bounds; embedded notebook figures and execution; mobile fit.`);
}finally{await browser.close();}

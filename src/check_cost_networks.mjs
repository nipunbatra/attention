// Shape and layout regressions for the three comparable next-token networks.
// Run: node src/check_cost_networks.mjs [attention.html]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url),candidates=['playwright','playwright-core'];
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const dir of fs.readdirSync(cache))candidates.push(path.join(cache,dir,'node_modules/playwright'));
let pw;for(const name of candidates){try{pw=require(name);break;}catch{}}
assert(pw,'Use an existing Playwright installation.');
const browser=await pw.chromium.launch(),page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-cost-networks-')),errors=[];
page.on('pageerror',e=>errors.push(e.message));
const ids=['concat-network','average-network','attention-network','attention-projections','attention-products'].map(x=>'s16-cost-'+x);
try{
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify({model:AT.model,forward:AT.forward(AT.sentences.river)}));
  const frames=await page.locator('#s16 .frame').evaluateAll(es=>es.map(e=>e.id));
  for(const [diagram,table] of [['concat-network','concat'],['average-network','average'],['attention-products','projections']])assert.equal(frames.indexOf('s16-cost-'+diagram)+1,frames.indexOf('s16-cost-'+table));
  for(const kind of ['concat','average','attention']){
    const host=page.locator('[data-cost-network="'+kind+'"]'),svg=host.locator('svg');
    const inputs=kind==='concat'?12:4;
    for(const [layer,n]of [['input',inputs],['hidden',8],['output',4]])assert.equal(await host.locator('[data-neuron="'+layer+'"]').count(),n,kind+' '+layer);
    assert.equal(await svg.getAttribute('data-output-width'),'20');
    assert.equal(await host.locator('[data-layer="input-hidden"]').count(),inputs*8);
    assert.equal(await host.locator('[data-layer="hidden-output"]').count(),8*4);
    assert.equal(await host.locator('[data-residual-add]').count(),kind==='attention'?1:0);
    assert.equal(await host.locator('[data-residual-wire]').count(),kind==='attention'?1:0);
    const context=page.locator('[data-cost-context="'+kind+'"]');
    assert.deepEqual(await context.locator('.context-token').allTextContents(),await page.evaluate(()=>AT.sentences.river));
    assert.deepEqual(await context.locator('[data-used="true"]').evaluateAll(es=>es.map(e=>Number(e.dataset.position))),kind==='concat'?[8,9,10]:[1,2,3,4,5,6,7,8,9,10]);
  }
  const shapes=await page.locator('[data-cost-matrices] [data-matrix]').evaluateAll(es=>es.map(e=>({name:e.dataset.matrix,rows:Number(e.dataset.rows),cols:Number(e.dataset.cols),cells:e.querySelectorAll('[data-cell]').length})));
  for(const m of shapes)assert.equal(m.cells,m.rows*m.cols,m.name+' draws one square per entry');
  assert.deepEqual(shapes.map(m=>[m.rows,m.cols]),[[10,4],[4,3],[10,3],[4,3],[10,3],[4,2],[10,2],[10,3],[3,10],[10,10],[10,10],[10,10],[10,2],[10,2],[2,4],[10,4]]);
  async function go(id,build=99){
    await page.evaluate(({id,build})=>{const f=document.getElementById(id),s=f.closest('.sec');AT.present.enter();AT.present.go(s.id,[...s.querySelectorAll('.frame')].indexOf(f)+1,build);},{id,build});
    await page.waitForTimeout(100);
  }
  for(const viewport of [{width:1280,height:720},{width:1024,height:768}]){
    await page.setViewportSize(viewport);
    for(const id of ids){
      await go(id);const max=await page.evaluate(()=>AT.present.state().frame.maxBuild);
      for(const build of [0,max,0,max]){
        await go(id,build);
        assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' build '+build);
        const hiddenGroups=await page.locator('#'+id+' svg [data-build]').evaluateAll(es=>es.map(e=>({build:Number(e.dataset.build),hidden:getComputedStyle(e).visibility==='hidden'||getComputedStyle(e).opacity==='0'})));
        for(const group of hiddenGroups)assert.equal(group.hidden,group.build>build,id+' SVG reveal follows navigation');
      }
      const bad=await page.locator('#'+id+' svg').evaluate(svg=>{
        const v=svg.viewBox.baseVal,labels=[...svg.querySelectorAll('text')].map(e=>({text:e.textContent,b:e.getBBox(),size:parseFloat(getComputedStyle(e).fontSize)}));
        return labels.filter(({b,size})=>b.x<-.5||b.y<-.5||b.x+b.width>v.width+.5||b.y+b.height>v.height+.5||size<22).map(({text,b,size})=>({text,x:b.x,y:b.y,w:b.width,h:b.height,size}));
      });assert.deepEqual(bad,[],id+' all SVG labels fit and stay at least 22px on the logical stage');
      await page.screenshot({path:path.join(shots,id+'-'+viewport.width+'.png')});
    }
  }
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  for(const id of ids){
    const frame=page.locator('#'+id),overflow=await frame.evaluate(e=>e.scrollWidth>e.clientWidth+2);
    assert(!overflow,id+' contains its scrollable diagram on a phone');
    assert.equal(await frame.locator('.cost-network-viewport').evaluate(e=>getComputedStyle(e).overflowX),'auto');
  }
  await page.locator('#s16-cost-attention-network').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,'reading-phone.png')});
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,forward:AT.forward(AT.sentences.river)})),original);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({frames:ids.length,checks:'same context; neurons, connections and matrix cells; hidden layers and residual; reveal reversal; SVG labels; projector and mobile fit; model immutability',screenshots:shots},null,2));
}finally{await browser.close();}

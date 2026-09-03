// Regression for the shared SVG authoring source inside the actual classroom page.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
let pw;
for(const p of [process.env.PLAYWRIGHT_PATH,'/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)){try{pw=require(p);break;}catch{}}
if(!pw)throw Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
const b=await pw.chromium.launch();
try{
  const p=await b.newPage({viewport:{width:1280,height:720}}),errors=[];
  p.on('pageerror',e=>errors.push(e.message));
  p.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await p.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href+'?present#s16/1/0');
  for(let i=0;i<12;i++){
    const state=await p.evaluate(()=>AT.present.state());
    assert.equal(state.stepper.index,i);
    const stats=await p.evaluate(()=>{const f=document.querySelector('#s16-flow-frame'),g=f.querySelector('.flow-classroom svg');return{
      wide:document.documentElement.scrollWidth>innerWidth+1,tall:f.scrollHeight>f.clientHeight+2,
      graphWidth:g.getBoundingClientRect().width,graphHeight:g.getBoundingClientRect().height,
      text:g.textContent,dim:ATTENTION_PREVIEW_DATA.dims,model:[AT.d_model,AT.d_k,AT.d_v]};});
    assert(!stats.wide&&!stats.tall,`stage${i+1} should fit a classroom screen`);
    assert(stats.graphWidth>400&&stats.graphHeight>400,'diagram is not collapsed into a narrow grid column');
    assert.deepEqual([stats.dim.dModel,stats.dim.dKey,stats.dim.dValue],stats.model);
    assert(stats.text.includes(i===11?'receiver 10':'receiver 7'));
    if(i<11)await p.keyboard.press('ArrowRight');
  }
  const si=process.argv.indexOf('--shot');
  if(si>=0){await p.waitForTimeout(200);await p.screenshot({path:process.argv[si+1]});}
  await p.keyboard.press('Escape');
  assert.equal(await p.evaluate(()=>AT.present.isActive()),false);
  assert(await p.locator('#s16-flow-stepper .flow-full svg').isVisible());
  await p.setViewportSize({width:390,height:844});
  assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  assert.deepEqual(errors,[]);
  console.log('PASS:12 classroom diagram stages fit1280×720; live model dimensions, receiver switch, reading mode, mobile containment, no errors.');
}finally{await b.close();}

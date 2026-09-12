// Check Part II's lookup -> order -> position-cue teaching sequence.
// node src/check_position_intro.mjs [attention.html]
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
const screenshots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-position-intro-'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
try{
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const content=await page.evaluate(()=>{
    const frames=[...document.querySelectorAll('#s01 .frame')];
    const ids=['s01-frame-lookup','s01-frame-order','s01-frame-position-cue'];
    const indices=ids.map(id=>frames.findIndex(f=>f.id===id));
    const lookup=document.getElementById(ids[0]),order=document.getElementById(ids[1]),cue=document.getElementById(ids[2]);
    return {
      indices,
      lookupMath:[...lookup.querySelectorAll('annotation')].map(el=>el.textContent).join(' '),
      rows:[...order.querySelectorAll('tbody tr')].map(row=>[...row.cells].map(c=>c.textContent)),
      hasPart1Bridge:order.textContent.includes('concatenation kept the positions in separate input slots'),
      colours:['start-token','start-position','start-row'].map(role=>{
        const symbol=cue.querySelector('.katex-display .katex-html .'+role),key=cue.querySelector('.position-key .'+role);
        return symbol&&key&&getComputedStyle(symbol).color===getComputedStyle(key).color;
      }),
      toyCaveat:document.getElementById('s01-frame-position-limit').textContent.includes('ignore coordinate 5')
    };
  });
  assert.deepEqual(content.indices,[3,4,5],'Lookup, order intuition, and positional addition must be consecutive.');
  assert(!/p_i|e_i/.test(content.lookupMath),'Keep position addition out of the lookup introduction.');
  assert.deepEqual(content.rows,[['Maya','helps','Ravi'],['Ravi','helps','Maya']]);
  assert(content.hasPart1Bridge,'Connect to the previous fixed-window MLP.');
  assert(content.colours.every(Boolean),'Explain each equation term in the same colour.');
  assert(content.toyCaveat,'Keep the toy position limitation explicit.');
  for(const [i,id] of ['lookup','order','position-cue'].entries()){
    for(const build of [0,1,2]){
      await page.evaluate(({frame,build})=>{AT.present.enter();AT.present.go('s01',frame,build);},{frame:i+4,build});
      await page.waitForTimeout(90);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' build '+build+' must fit.');
    }
    // Let the build's opacity transition finish before inspecting KaTeX glyphs.
    await page.waitForTimeout(300);
    await page.screenshot({path:path.join(screenshots,id+'.png')});
  }
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await page.locator('#s01-frame-position-cue').scrollIntoViewIfNeeded();
  const phone=await page.evaluate(()=>({
    columns:getComputedStyle(document.querySelector('#s01 .position-key')).gridTemplateColumns.split(' ').length,
    fits:[...document.querySelectorAll('#s01-frame-order,#s01-frame-position-cue')].every(el=>el.scrollWidth<=el.clientWidth+1)
  }));
  assert(phone.columns===1&&phone.fits,'Phone article explanations must wrap without horizontal scrolling.');
  await page.screenshot({path:path.join(screenshots,'phone-position-cue.png')});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({content,phone,screenshots,errors},null,2));
}finally{await browser.close();}

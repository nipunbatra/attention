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
  const bankBefore=await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs}));
  const samples=[];
  for(const position of [1,5,10]){
    await page.locator('#s01-position-choice [data-position="'+position+'"]').click();
    const sample=await page.evaluate(()=>{
      const root=document.querySelector('#s01-emb-tab');
      const nums=selector=>[...root.querySelectorAll(selector)].map(row=>[...row.querySelectorAll('td')].map(cell=>Number(cell.textContent.replaceAll('−','-'))));
      return {
        headers:[...root.querySelectorAll('thead th')].map(el=>el.textContent),
        rows:nums('tbody tr'),sum:nums('tfoot tr')[0],
        label:root.querySelector('tbody tr:nth-child(2) th').textContent,
        selected:[...document.querySelectorAll('#s01-position-choice [aria-pressed="true"]')].map(el=>Number(el.dataset.position)),
        colours:[root.querySelector('tbody tr:first-child td'),root.querySelector('tbody tr:nth-child(2) td'),root.querySelector('tfoot td')].map(el=>getComputedStyle(el).color),
        equationColours:[...document.querySelectorAll('#s01-position-example span')].map(el=>getComputedStyle(el).color)
      };
    });
    const word=[.2,-.4,.6,.1],pos=position-1;
    // Independent general formula, rather than the demo's four-entry shortcut.
    const encoding=Array.from({length:4},(_,j)=>{
      const angle=pos/Math.pow(10000,2*Math.floor(j/2)/4);
      return j%2===0?Math.sin(angle):Math.cos(angle);
    });
    const close=(actual,expected)=>assert(Math.abs(actual-expected)<=.000501,actual+' vs '+expected);
    assert.deepEqual(sample.headers,['Coordinate','1','2','3','4'],'No dedicated position coordinate.');
    assert.deepEqual(sample.rows[0],word,'Moving a word keeps its vocabulary row fixed.');
    encoding.forEach((x,j)=>{close(sample.rows[1][j],x);close(sample.sum[j],word[j]+x);});
    assert.equal(sample.label,'+ Position '+position);
    assert.deepEqual(sample.selected,[position]);
    assert.deepEqual(sample.colours,sample.equationColours,'Colours link the table to the worked addition.');
    assert.equal(new Set(sample.colours).size,3,'Word, position, and sum have distinct colours.');
    samples.push({position,...sample});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs})),bankBefore,'The separate encoding demo must not change bank-model arithmetic.');
  for(const position of [1,5,10]){
    await page.evaluate(()=>{AT.present.enter();AT.present.go('s01',7,0);});
    await page.locator('#s01-position-choice [data-position="'+position+'"]').click();
    await page.waitForTimeout(300);
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Every encoding choice must fit.');
    await page.screenshot({path:path.join(screenshots,'sinusoidal-position-'+position+'.png')});
  }
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
  for(const [id,frame] of [['starting-rows',8],['bank-scope',9]]){
    await page.evaluate(frame=>AT.present.go('s01',frame,2),frame);
    await page.waitForTimeout(300);
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' must fit.');
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
  await page.locator('#s01-frame2').scrollIntoViewIfNeeded();
  const phoneTable=await page.locator('#s01-emb-tab').evaluate(el=>({
    fits:el.scrollWidth<=el.clientWidth+1,
    scrolls:[...el.querySelectorAll('*')].filter(node=>node.clientWidth>0&&node.scrollWidth>node.clientWidth+1).map(node=>node.className)
  }));
  assert(phoneTable.fits&&phoneTable.scrolls.length===0,'The article position table must fit a phone without a nested scrollbar.');
  await page.screenshot({path:path.join(screenshots,'phone-sinusoidal.png')});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({content,samples,phone,phoneTable,screenshots,errors},null,2));
}finally{await browser.close();}

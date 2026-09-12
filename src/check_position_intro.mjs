// Protect the opening flow: task -> lookup -> order -> one position example
// -> concrete sentence matrix -> context problem -> last-token baseline. Keep extra notation in
// article companions instead of reintroducing standalone recap/caveat slides.
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
  async function goFrame(id,build=0){
    await page.evaluate(({id,build})=>{
      const frame=document.getElementById(id),section=frame.closest('.sec');
      const number=[...section.querySelectorAll('.frame')].indexOf(frame)+1;
      AT.present.enter();AT.present.go(section.id,number,build);
    },{id,build});
  }
  const content=await page.evaluate(()=>{
    const frames=[...document.querySelectorAll('#s01 .frame')];
    const lookup=document.getElementById('s01-frame-lookup'),order=document.getElementById('s01-frame-order'),cue=document.getElementById('s01-position-notation');
    return {
      flow:frames.map(frame=>frame.id),
      notationInArticle:cue.classList.contains('companion')&&cue.closest('.frame').id==='s01-frame2',
      lookupMath:[...lookup.querySelectorAll('annotation')].map(el=>el.textContent).join(' '),
      rows:[...order.querySelectorAll('tbody tr')].map(row=>[...row.cells].map(c=>c.textContent)),
      hasPart1Bridge:order.textContent.includes('concatenation kept the positions in separate input slots'),
      colours:['start-token','start-position','start-row'].map(role=>{
        const symbol=cue.querySelector('.katex-display .katex-html .'+role),key=cue.querySelector('.position-key .'+role);
        return symbol&&key&&getComputedStyle(symbol).color===getComputedStyle(key).color;
      }),
      toyCaveat:document.getElementById('s01-model-scope').textContent.includes('ignore coordinate 5'),
      modelScopeInArticle:document.getElementById('s01-model-scope').classList.contains('companion'),
      baselineBridge:document.querySelector('#s02-frame1 .prose').textContent.includes('of the ten rows'),
      baselineRow:[...document.querySelectorAll('#s02-frame1 annotation')].filter(el=>!el.closest('.companion')).map(el=>el.textContent),
      baselineStartingNotation:document.querySelector('#s02-frame-head .prose').textContent.includes('The superscript')
    };
  });
  assert.deepEqual(content.flow,['s01-frame1','s01-frame-probabilities','s01-frame-lookup','s01-frame-order','s01-frame2','s01-frame-starting-row'],'Each opening frame has one new job; do not add repeated task, position-formula, or position-caveat frames.');
  assert(content.notationInArticle&&content.modelScopeInArticle,'Keep elaboration with the example in the article, not on extra recap slides.');
  assert(content.baselineBridge,'The next experiment must explicitly start from the ten-row matrix.');
  assert(content.baselineRow.every(expr=>!expr.includes('e_t')&&!expr.includes('^{(0)}')),'Keep the first baseline view on the already-defined row e_10.');
  assert(content.baselineStartingNotation,'Define the starting-row superscript where the head first uses it.');
  assert(!/p_i|e_i/.test(content.lookupMath),'Keep position addition out of the lookup introduction.');
  assert.deepEqual(content.rows,[['Maya','helps','Ravi'],['Ravi','helps','Maya']]);
  assert(content.hasPart1Bridge,'Connect to the previous fixed-window MLP.');
  assert(content.colours.every(Boolean),'Explain each equation term in the same colour.');
  assert(content.toyCaveat,'Keep the toy position limitation explicit.');
  const problem=await page.evaluate(()=>{
    const root=document.getElementById('s02-frame-problem');
    return {
      flow:[...document.querySelectorAll('#s02 .frame')].map(frame=>frame.id),
      sentences:[...root.querySelectorAll('.problem-sentence')].map(p=>p.textContent),
      expected:[AT.sentences.river,AT.sentences.cheque].map(tokens=>tokens.join(' ')+' ___'),
      clues:[...root.querySelectorAll('.problem-clue')].map(el=>el.textContent),
      endings:[...root.querySelectorAll('.problem-ending')].map(el=>el.textContent),
      question:root.querySelector(':scope > p:last-of-type').textContent,
      formulas:root.querySelectorAll('.katex').length,
      baselineTitle:document.getElementById('s02-frame1').dataset.title,
      restricted:document.querySelector('#s02-frame1 .prose').textContent.includes('A deliberately limited test'),
      nextStep:document.querySelector('#s02-frame2 > p').textContent
    };
  });
  assert.deepEqual(problem.flow,['s02-frame-problem','s02-frame1','s02-frame-probabilities','s02-frame-head','s02-frame2'],'State the problem before the baseline, observe its prediction, then explain its limitation.');
  assert.deepEqual(problem.sentences,problem.expected,'Use the exact two contexts from the numerical example.');
  assert.deepEqual(problem.clues,['river','cheque']);
  assert.deepEqual(problem.endings,['and watched the ___','and watched the ___']);
  assert(problem.question.includes('earlier clues change the next-word probabilities'));
  assert.equal(problem.formulas,0,'The section break should state the problem in plain language.');
  assert(problem.baselineTitle.startsWith('Baseline 1:')&&problem.restricted,'Do not present the restricted baseline as the solution.');
  assert(problem.nextStep.includes('several recent token rows'),'Explain the next experiment after establishing the limitation.');
  const matrix=await page.evaluate(()=>{
    const frame=document.getElementById('s01-frame-starting-row'),table=frame.querySelector('table');
    const rows=[...table.querySelectorAll('tbody tr')];
    return {
      sentence:document.getElementById('s01-matrix-sentence').textContent,
      labels:rows.map(row=>row.querySelector('th').textContent.trim()),
      values:rows.map(row=>[...row.querySelectorAll('td')].map(cell=>Number(cell.textContent.replaceAll('−','-')))),
      expected:AT.embed(AT.sentences.river),tokens:AT.sentences.river,
      headers:[...table.querySelectorAll('thead th')].map(cell=>cell.textContent),
      formulas:[...frame.querySelectorAll('annotation')].map(el=>el.textContent),
      text:frame.querySelector('.sentence-legend').textContent,
      scopeBeforeTable:!!(document.getElementById('s01-model-scope').compareDocumentPosition(frame)&Node.DOCUMENT_POSITION_FOLLOWING),
      visibleScope:frame.querySelector('h3').textContent.includes('hand-chosen bank example')&&frame.querySelector('.sentence-legend').textContent.includes('This toy ignores column 5.'),
      colours:[
        getComputedStyle(frame.querySelector('.sentence-row-number')).color===getComputedStyle(frame.querySelector('.sentence-legend .sentence-count')).color,
        getComputedStyle(table.querySelector('thead th:nth-child(2)')).color===getComputedStyle(frame.querySelector('.sentence-legend .sentence-width')).color
      ]
    };
  });
  assert(matrix.scopeBeforeTable,'Explain the return to five-coordinate bank vectors before showing them.');
  assert(matrix.visibleScope,'Retain the model limitation as a short label with its data.');
  assert.equal(matrix.sentence,'“'+matrix.tokens.join(' ')+' ___”');
  assert.deepEqual(matrix.labels,matrix.tokens.map((token,i)=>(i+1)+' '+token));
  assert.deepEqual(matrix.headers,['Token / row','1','2','3','4','5']);
  assert.equal(matrix.values.length,10,'All ten occurrences need rows, including repeated the.');
  matrix.expected.forEach((row,i)=>row.forEach((x,j)=>assert(Math.abs(matrix.values[i][j]-x)<.050001)));
  assert.deepEqual(matrix.values[5],[3,0,0,0,.6],'Row 6 is river in the actual bank model.');
  assert(matrix.formulas.includes('T=10')&&matrix.formulas.includes('d_{\\text{model}}=5'));
  assert(matrix.text.includes('Tokens in this input.')&&matrix.text.includes('Numbers in each token row.'));
  assert(matrix.text.includes('One sentence, ten token rows.')&&matrix.text.includes('The blank has no row yet.'));
  assert(matrix.colours.every(Boolean),'Match row-count and column-width explanations to the table.');
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
    await goFrame('s01-frame2');
    await page.locator('#s01-position-choice [data-position="'+position+'"]').click();
    await page.waitForTimeout(300);
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Every encoding choice must fit.');
    await page.screenshot({path:path.join(screenshots,'sinusoidal-position-'+position+'.png')});
  }
  for(const id of content.flow.concat(problem.flow)){
    for(const build of [0,1,2]){
      await goFrame(id,build);
      await page.waitForTimeout(300);
      assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' build '+build+' must fit.');
    }
    await page.screenshot({path:path.join(screenshots,id+'.png')});
  }
  const baselineProbabilities=await page.evaluate(()=>{
    const a=AT.baseline(AT.sentences.river),b=AT.baseline(AT.sentences.cheque);
    return [a.probs.at(-1),b.probs.at(-1)];
  });
  assert.deepEqual(...baselineProbabilities,'The baseline must still give identical distributions for these contexts.');
  for(const entry of ['s02-frame-problem','s02-frame1']){
    await goFrame('s02-frame-probabilities');
    await page.locator('#s02-ctx-b').click();
    assert.equal(await page.locator('#s02-ctx-b').getAttribute('aria-pressed'),'true');
    await goFrame(entry);
    assert.equal(await page.locator('#s02-ctx-a').getAttribute('aria-pressed'),'true','Re-entering the experiment resets A after the new section break.');
  }
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await page.locator('#s02-frame-problem').scrollIntoViewIfNeeded();
  assert(await page.locator('#s02-frame-problem').evaluate(el=>el.scrollWidth<=el.clientWidth+1),'The problem statement must fit the phone article.');
  await page.screenshot({path:path.join(screenshots,'phone-context-problem.png')});
  await page.locator('#s01-position-notation').scrollIntoViewIfNeeded();
  const phone=await page.evaluate(()=>({
    columns:getComputedStyle(document.querySelector('#s01 .position-key')).gridTemplateColumns.split(' ').length,
    fits:[...document.querySelectorAll('#s01-frame-order,#s01-position-notation')].every(el=>el.scrollWidth<=el.clientWidth+1)
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
  await page.locator('#s01-sentence-matrix').scrollIntoViewIfNeeded();
  const phoneMatrix=await page.locator('#s01-sentence-matrix').evaluate(el=>({
    fits:el.scrollWidth<=el.clientWidth+1,
    scrolls:[...el.querySelectorAll('*')].filter(node=>node.clientWidth>0&&node.scrollWidth>node.clientWidth+1).map(node=>node.className)
  }));
  assert(phoneMatrix.fits&&phoneMatrix.scrolls.length===0,'The full sentence matrix must fit a phone.');
  await page.screenshot({path:path.join(screenshots,'phone-sentence-matrix.png')});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({content,problem,matrix,samples,phone,phoneTable,phoneMatrix,screenshots,errors},null,2));
}finally{await browser.close();}

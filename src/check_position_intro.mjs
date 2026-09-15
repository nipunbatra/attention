// Protect the opening flow: Part 1 recap -> Transformer paper -> task -> lookup -> coordinate meaning -> order -> position
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
const model=JSON.parse(fs.readFileSync(new URL('./toy.json',import.meta.url),'utf8'));
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
      toyCaveat:document.getElementById('s01-model-scope').textContent.includes('same four-coordinate sums'),
      modelScopeInArticle:document.getElementById('s01-model-scope').classList.contains('companion'),
      baselineBridge:document.querySelector('#s02-frame1 .prose').textContent.includes('of the ten rows'),
      baselineRow:[...document.querySelectorAll('#s02-frame1 annotation')].filter(el=>!el.closest('.companion')).map(el=>el.textContent),
      baselineStartingNotation:document.querySelector('#s02-frame-head .head-key .head-input').textContent.includes('before context updates')
    };
  });
  assert.deepEqual(content.flow,['s01-recap-model','s01-recap-context','s01-paper-intro','s01-paper-importance','s01-paper-architecture','s01-frame1','s01-frame-probabilities','s01-frame-lookup','s01-frame-coordinates','s01-frame-order','s01-frame2','s01-frame-starting-row'],'Start with the requested recap and paper orientation, then define coordinate meaning before position.');
  assert(content.notationInArticle&&content.modelScopeInArticle,'Keep elaboration with the example in the article, not on extra recap slides.');
  assert(content.baselineBridge,'The next experiment must explicitly start from the ten-row matrix.');
  assert(content.baselineRow.every(expr=>!expr.includes('e_t')&&!expr.includes('^{(0)}')),'Keep the first baseline view on the already-defined row e_10.');
  assert(content.baselineStartingNotation,'Define the starting-row superscript where the head first uses it.');
  assert(!/p_i|e_i/.test(content.lookupMath),'Keep position addition out of the lookup introduction.');
  assert.deepEqual(content.rows,[['Maya','helps','Ravi'],['Ravi','helps','Maya']]);
  assert(content.hasPart1Bridge,'Connect to the previous fixed-window MLP.');
  assert(content.colours.every(Boolean),'Explain each equation term in the same colour.');
  assert(content.toyCaveat,'Use the same model for the introduction and the rest of the lesson.');
  const features=await page.evaluate(()=>{
    const frame=document.getElementById('s01-frame-coordinates'),table=frame.querySelector('table');
    return {
      headers:[...table.querySelectorAll('thead th')].map(el=>el.textContent),
      values:[...table.querySelectorAll('tbody tr')].map(row=>[...row.querySelectorAll('td')].map(el=>Number(el.textContent))),
      text:frame.textContent,
      colours:['water','finance','person','glue'].map((name,j)=>getComputedStyle(frame.querySelector('.feature-'+name)).color===getComputedStyle(table.querySelector('tbody tr td:nth-of-type('+(j+1)+')')).color),
      forbidden:[...document.querySelectorAll('thead th:not(:first-child),.vec-axis')].filter(el=>/^(pos|position)$/i.test(el.textContent.trim())).length,
      dimensions:AT.d_model,axes:AT.axes.short.e
    };
  });
  assert.deepEqual(features.headers,['Token','1: water','2: finance','3: person','4: glue']);
  assert.deepEqual(features.values,['river','bank','fisherman','the'].map(token=>model.tok_emb[token]));
  assert(features.text.includes('not probabilities')&&features.text.includes('Real learned embeddings'));
  assert(features.colours.every(Boolean),'Feature labels and their columns must use matching colours.');
  assert.equal(features.dimensions,4);
  assert.deepEqual(features.axes,['water','finance','person','glue']);
  assert.equal(features.forbidden,0,'No dedicated position column anywhere in the lesson.');
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
      nextStep:document.querySelector('#s02-frame2 .same-next').textContent
    };
  });
  assert.deepEqual(problem.flow,['s02-topic-break','s02-frame-problem','s02-frame1','s02-frame-probabilities','s02-frame-head','s02-frame-head-softmax','s02-frame2'],'Pause at the topic break, state the problem, test the baseline, then explain its nodes, scores and probabilities in order.');
  assert.deepEqual(problem.sentences,problem.expected,'Use the exact two contexts from the numerical example.');
  assert.deepEqual(problem.clues,['river','cheque']);
  assert.deepEqual(problem.endings,['and watched the ___','and watched the ___']);
  assert(problem.question.includes('earlier clues change the next-word probabilities'));
  assert.equal(problem.formulas,0,'The section break should state the problem in plain language.');
  assert(problem.baselineTitle.startsWith('Baseline 1:')&&problem.restricted,'Do not present the restricted baseline as the solution.');
  assert(problem.nextStep.includes('several recent token rows'),'Explain the next experiment after establishing the limitation.');
  const comparison=await page.evaluate(()=>{
    const frame=document.getElementById('s02-frame2'),table=frame.querySelector('table');
    return {
      title:frame.dataset.title,
      names:[...frame.querySelectorAll('.same-name')].map(el=>el.textContent),
      sentences:[...frame.querySelectorAll('.same-sentence')].map(el=>el.textContent),
      expected:[AT.sentences.river,AT.sentences.cheque].map(tokens=>tokens.join(' ')+' ___'),
      selected:[...frame.querySelectorAll('.same-last')].map(el=>el.textContent),
      clues:[...frame.querySelectorAll('.problem-clue')].map(el=>el.textContent),
      headers:[...table.querySelectorAll('thead th')].map(el=>el.textContent),
      rowNames:[...table.querySelectorAll('tbody th')].map(el=>el.textContent),
      rows:[...table.querySelectorAll('tbody tr')].map(tr=>[...tr.querySelectorAll('td')].map(td=>Number(td.textContent))),
      expectedRows:[AT.baseline(AT.sentences.river).E.at(-1),AT.baseline(AT.sentences.cheque).E.at(-1)],
      difference:[...table.querySelectorAll('tfoot td')].map(td=>Number(td.textContent)),
      footer:table.querySelector('tfoot th').textContent,
      reason:document.getElementById('s02-same-reason').textContent,
      conclusion:document.getElementById('s02-same-note').textContent,
      controls:['s02-ctx-a','s02-ctx-b'].map(id=>document.getElementById(id).textContent),
      highlighted:table.querySelectorAll('.is-hl').length
    };
  });
  assert.equal(comparison.title,'Why both sentences give the same prediction');
  assert.deepEqual(comparison.names,['River sentence','Cheque sentence']);
  assert.deepEqual(comparison.sentences,comparison.expected,'Show both full contexts beside the named rows.');
  assert.deepEqual(comparison.selected,['the','the']);
  assert.deepEqual(comparison.clues,['river','cheque']);
  assert.deepEqual(comparison.headers,['sentence','water','finance','person','glue']);
  assert.deepEqual(comparison.rowNames,comparison.names,'Use descriptive names, never unexplained A/B abbreviations.');
  assert.deepEqual(comparison.controls,comparison.names,'Use the same names on the context-switch buttons.');
  assert.deepEqual(comparison.rows,comparison.expectedRows);
  assert.deepEqual(comparison.difference,[0,0,0,0]);
  assert.equal(comparison.footer,'Difference');
  assert(comparison.reason.includes('only the final')&&comparison.reason.includes('position 10'));
  assert(comparison.conclusion.includes('Every difference is zero')&&comparison.conclusion.includes('predictions are identical'));
  assert.equal(comparison.highlighted,0,'Both input rows deserve equal emphasis.');
  const head=await page.evaluate(()=>{
    const frame=document.getElementById('s02-frame-head'),softmax=document.getElementById('s02-frame-head-softmax');
    const svg=frame.querySelector('svg'),baseline=AT.baseline(AT.sentences.river),row=baseline.E.at(-1),logits=baseline.logits.at(-1);
    const probability=document.getElementById('s02-head-water-prob'),water=AT.vocab.indexOf('water');
    const pairs=[['head-input','--c-e'],['head-hidden','--c-v'],['head-param','--c-q'],['head-score','--ink-2']];
    return {
      widths:[Number(svg.dataset.inputWidth),Number(svg.dataset.hiddenWidth),Number(svg.dataset.outputWidth)],
      hidden:[...svg.querySelectorAll('[data-hidden-node]')].map(el=>Number(el.dataset.activation)),expectedHidden:baseline.HeadHidden.at(-1),
      inputs:[...svg.querySelectorAll('[data-input-node]')].map(el=>Number(el.dataset.inputNode)),
      outputs:[...svg.querySelectorAll('[data-output-token]')].map(el=>({word:el.dataset.outputToken,logit:Number(el.dataset.logit),expected:logits[AT.vocab.indexOf(el.dataset.outputToken)]})),
      edges:[...svg.querySelectorAll('[data-weight]')].map(el=>({actual:Number(el.dataset.weight),expected:el.dataset.layer==='hidden'?AT.model.W_hidden[Number(el.dataset.input)][Number(el.dataset.hidden)]:AT.model.W_vocab[Number(el.dataset.hidden)][AT.vocab.indexOf(el.dataset.token)]})),
      inputValues:[...svg.querySelectorAll('.net-value.head-input')].map(el=>Number(el.textContent)),row,
      colours:pairs.map(([role,variable])=>{
        const key=frame.querySelector('.head-key .'+role),math=frame.querySelector('.head-equation .katex-html .'+role),node=svg.querySelector('text.'+role);
        const probe=document.createElement('span');probe.style.color='var('+variable+')';frame.appendChild(probe);
        const expected=getComputedStyle(probe).color;probe.remove();
        return [getComputedStyle(key).color,getComputedStyle(math).color,getComputedStyle(node).fill].every(value=>value===expected);
      }),
      probabilityColours:[softmax.querySelector('.head-key .head-prob'),softmax.querySelector('.head-equation .katex-html .head-prob'),probability.querySelector('.katex-html .head-prob')].map(el=>getComputedStyle(el).color),
      text:frame.textContent,
      probability:Number(probability.dataset.probability),expectedProbability:baseline.probs.at(-1)[water],
      denominator:Number(probability.dataset.denominator),expectedDenominator:logits.reduce((s,x)=>s+Math.exp(x),0),
      calculation:probability.querySelector('annotation').textContent,
      boundedLabels:[...svg.querySelectorAll('text')].every(el=>{const b=el.getBBox();return b.x>=0&&b.y>=0&&b.x+b.width<=660&&b.y+b.height<=390;}),
      worksheet:[...document.querySelectorAll('#s02-head-worked tbody tr')].map(tr=>[...tr.querySelectorAll('td')].map(td=>Number(td.textContent))),
      expectedWorksheet:baseline.HeadHidden.at(-1).map((x,i)=>[x,AT.model.W_vocab[i][water],x*AT.model.W_vocab[i][water]])
    };
  });
  assert.deepEqual(head.widths,[4,8,20]);
  assert.deepEqual(head.hidden,head.expectedHidden,'All eight hidden activations come from the actual ReLU predictor.');
  assert.deepEqual(head.inputs,[0,1,2,3],'Four coordinate nodes, not four tokens.');
  assert.deepEqual(head.inputValues,head.row);
  assert.deepEqual(head.outputs.map(o=>o.word),['the','water','teller','money']);
  assert(head.outputs.every(o=>o.logit===o.expected));
  assert.equal(head.edges.length,64,'32 input-to-hidden edges plus 32 hidden-to-displayed-output edges.');
  assert(head.edges.every(e=>e.actual===e.expected));
  assert(head.colours.every(Boolean),'Match Part I colours between nodes, equations, and definitions.');
  assert.equal(new Set(head.probabilityColours).size,1,'Probability symbols and prose must share green.');
  assert(head.text.includes('Called')&&head.text.includes('in Part I')&&head.text.includes('Dots omit 16 outputs')&&head.text.includes('ReLU'));
  assert.equal(head.probability,head.expectedProbability);
  assert.equal(head.denominator,head.expectedDenominator,'Normalize across the entire saved vocabulary.');
  assert(head.calculation.includes('10.678')&&head.calculation.includes('0.094')&&head.calculation.includes('\\approx'));
  assert(head.boundedLabels,'Every SVG label must fit within the drawing.');
  assert.deepEqual(head.worksheet,head.expectedWorksheet);
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
      visibleScope:frame.querySelector('h3').textContent.includes('hand-chosen bank example'),
      colours:[
        getComputedStyle(frame.querySelector('.sentence-row-number')).color===getComputedStyle(frame.querySelector('.sentence-legend .sentence-count')).color,
        getComputedStyle(table.querySelector('thead th:nth-child(2)')).color===getComputedStyle(frame.querySelector('.sentence-legend .sentence-width')).color
      ]
    };
  });
  assert(matrix.scopeBeforeTable,'Explain the shared word/position model before stacking its rows.');
  assert(matrix.visibleScope,'Label these numbers as hand-chosen.');
  assert.equal(matrix.sentence,'“'+matrix.tokens.join(' ')+' ___”');
  assert.deepEqual(matrix.labels,matrix.tokens.map((token,i)=>(i+1)+' '+token));
  assert.deepEqual(matrix.headers,['Token / row','1','2','3','4']);
  assert.equal(matrix.values.length,10,'All ten occurrences need rows, including repeated the.');
  matrix.expected.forEach((row,i)=>row.forEach((x,j)=>assert(Math.abs(matrix.values[i][j]-x)<.050001)));
  assert.deepEqual(matrix.values[5],[3.1,-.1,0,.1],'Row 6 is river in the actual bank model.');
  assert(matrix.formulas.includes('T=10')&&matrix.formulas.includes('d_{\\text{model}}=4'));
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
    const word=model.tok_emb.the,encoding=model.pos_emb[position-1];
    const close=(actual,expected)=>assert(Math.abs(actual-expected)<=.000501,actual+' vs '+expected);
    assert.deepEqual(sample.headers,['Coordinate','1','2','3','4'],'No dedicated position coordinate.');
    assert.deepEqual(sample.rows[0],word,'Moving a word keeps its vocabulary row fixed.');
    encoding.forEach((x,j)=>{close(sample.rows[1][j],x);close(sample.sum[j],word[j]+x);});
    matrix.expected[position-1].forEach((x,j)=>close(sample.sum[j],x));
    assert.equal(sample.label,'+ Position '+position);
    assert.deepEqual(sample.selected,[position]);
    assert.deepEqual(sample.colours,sample.equationColours,'Colours link the table to the worked addition.');
    assert.equal(new Set(sample.colours).size,3,'Word, position, and sum have distinct colours.');
    samples.push({position,...sample});
  }
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,probs:AT.forward(AT.sentences.river).probs})),bankBefore,'Inspecting the actual position table must not mutate it.');
  for(const position of [1,5,10]){
    await goFrame('s01-frame2');
    await page.locator('#s01-position-choice [data-position="'+position+'"]').click();
    await page.waitForTimeout(300);
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Every encoding choice must fit.');
    await page.screenshot({path:path.join(screenshots,'position-table-'+position+'.png')});
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
  await goFrame('s02-frame-probabilities',0);
  assert.equal(await page.locator('#s02-frame-probabilities').getAttribute('data-title'),'This baseline cannot tell the contexts apart');
  assert(await page.locator('#s02-baseline-input').isVisible(),'The reason must be visible on the probability slide, including build zero.');
  assert(await page.locator('#s02-baseline-takeaway').isVisible(),'The takeaway must be visible, not relegated to notes or article companions.');
  assert((await page.locator('#s02-baseline-input').innerText()).includes('“the” at position 10'));
  assert((await page.locator('#s02-baseline-takeaway').innerText()).includes('earlier clues'));
  const distributions=[];
  for(const context of ['a','b']){
    await page.locator('#s02-ctx-'+context).click();
    assert.equal(await page.locator('#s02-ctx-'+context).getAttribute('aria-pressed'),'true');
    distributions.push(await page.locator('#s02-bars .bv').allTextContents());
    assert.equal(await page.locator('#s02-same-tab .is-hl').count(),0,'Switching the earlier chart must not emphasize one of the equal comparison rows.');
    assert((await page.locator('#s02-bars-note').innerText()).includes('Every probability stays unchanged. The eight-way tie is specific to this toy.'));
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'The conclusion, input explanation and bars must fit together.');
    await page.screenshot({path:path.join(screenshots,'baseline-takeaway-'+context+'.png')});
  }
  assert.equal(distributions[0].length,9,'Keep all eight candidates and the aggregate bar.');
  assert.deepEqual(distributions[0],distributions[1],'Context switching must visibly leave the probability bars unchanged.');
  for(const entry of ['s02-frame-problem','s02-frame1']){
    await goFrame('s02-frame-probabilities');
    await page.locator('#s02-ctx-b').click();
    assert.equal(await page.locator('#s02-ctx-b').getAttribute('aria-pressed'),'true');
    await goFrame(entry);
    assert.equal(await page.locator('#s02-ctx-a').getAttribute('aria-pressed'),'true','Re-entering the experiment resets A after the new section break.');
  }
  await page.evaluate(()=>AT.present.exit());
  await page.setViewportSize({width:390,height:844});
  await page.locator('#s02-frame2').scrollIntoViewIfNeeded();
  assert(await page.locator('#s02-frame2').evaluate(el=>[el,...el.querySelectorAll('.dt-scroll')].every(node=>node.scrollWidth<=node.clientWidth+1)),'Both named contexts and all four coordinate columns must fit on a phone.');
  await page.locator('#s02-frame2').screenshot({path:path.join(screenshots,'phone-context-comparison.png')});
  for(const id of ['s02-frame-head','s02-frame-head-softmax']){
    await page.locator('#'+id).scrollIntoViewIfNeeded();
    assert(await page.locator('#'+id).evaluate(el=>el.scrollWidth<=el.clientWidth+1),'The head explanation must fit the phone article.');
    await page.screenshot({path:path.join(screenshots,'phone-'+id+'.png')});
  }
  assert(await page.locator('#s02-head-worked').evaluate(el=>[el,...el.querySelectorAll('.dt-scroll,.katex-display')].every(node=>node.scrollWidth<=node.clientWidth+1)),'The worked water calculation must fit without a nested horizontal scrollbar (exclude KaTeX’s intentionally clipped accessibility tree).');
  await page.locator('#s02-frame-probabilities').scrollIntoViewIfNeeded();
  assert(await page.locator('#s02-frame-probabilities').evaluate(el=>el.scrollWidth<=el.clientWidth+1),'The baseline explanation must fit the phone article.');
  await page.screenshot({path:path.join(screenshots,'phone-baseline-takeaway.png')});
  await page.locator('#s01-frame-coordinates').scrollIntoViewIfNeeded();
  assert(await page.locator('#s01-coordinate-table').evaluate(el=>el.scrollWidth<=el.clientWidth+1&&!Array.from(el.querySelectorAll('*')).some(n=>n.clientWidth>0&&n.scrollWidth>n.clientWidth+1)),'Feature definitions and table must fit on a phone.');
  await page.screenshot({path:path.join(screenshots,'phone-coordinate-meaning.png')});
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
  await page.screenshot({path:path.join(screenshots,'phone-position-table.png')});
  await page.locator('#s01-sentence-matrix').scrollIntoViewIfNeeded();
  const phoneMatrix=await page.locator('#s01-sentence-matrix').evaluate(el=>({
    fits:el.scrollWidth<=el.clientWidth+1,
    scrolls:[...el.querySelectorAll('*')].filter(node=>node.clientWidth>0&&node.scrollWidth>node.clientWidth+1).map(node=>node.className)
  }));
  assert(phoneMatrix.fits&&phoneMatrix.scrolls.length===0,'The full sentence matrix must fit a phone.');
  await page.screenshot({path:path.join(screenshots,'phone-sentence-matrix.png')});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({content,features,problem,comparison,head,matrix,samples,phone,phoneTable,phoneMatrix,screenshots,errors},null,2));
}finally{await browser.close();}

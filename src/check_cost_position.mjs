// Optional Part 2B costs and Part II positions: arithmetic, controls and responsive fit.
// Run: node src/check_cost_position.mjs [attention.html]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const dir of fs.readdirSync(cache))candidates.push(path.join(cache,dir,'node_modules/playwright'));
let pw;for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
assert(pw,'Use an existing Playwright installation.');
const costs=['break','symbols','matmul','concat-network','concat','average-network','average','attention-network','attention-projections','attention-products','projections','pairs','scores','mask','messages','predictor','calculator','training','baselines-training','prefix-sum','total','training-compare','backward','last-row','prompt','cache','generation-compare','memory'].map(x=>'s16-cost-'+x);
const positions=[
  'break','order','permute','scores','swapped','contributions','consequence',
  'addition-break','shift','move-a','move-b','moved','toy','slot-scores','experiment','updated',
  'alternatives-break','append','append-scores','append-softmax','append-scale','append-tradeoffs',
  'learned-break','learned','clock-choice','clock','repeat','waves','period','sine-rule','worked-sine',
  'absolute-range','absolute-context','absolute-shift','relative-break','relative','alibi','rotate','rope-shift','rope-identity','rope-pairs','insertion','overview'
].map(x=>'s17-position-'+x);
const readingExtras=['add','width','routing','rates','sine','rope','mean','length','choices'].map(x=>'s17-position-'+x);
const ids=positions;
const shots=fs.mkdtempSync(path.join(os.tmpdir(),'attention-cost-position-'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const close=(a,b,label)=>assert(Math.abs(a-b)<1e-11,`${label}: ${a} vs ${b}`);
try{
  await page.goto(pathToFileURL(path.resolve(process.argv[3]||'part2b.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  assert.deepEqual(await page.locator('.frame.context-lesson').evaluateAll(es=>es.map(e=>e.id)),costs);
  async function go(id,build=99){
    await page.evaluate(({id,build})=>{const f=document.getElementById(id),s=f.closest('.sec');AT.present.enter();AT.present.go(s.id,[...s.querySelectorAll('.frame')].indexOf(f)+1,build);},{id,build});
    await page.waitForTimeout(80);
    assert.equal(await page.locator('.frame.is-live').getAttribute('id'),id);
  }
  const shapes=n=>[[n,4,3],[n,4,3],[n,4,2],[n,2,4],[n,3,n],[n,n,2],[n,4,8],[n,8,20]];
  for(const n of [1,2,8,16,64,256,1024]){
    const products=shapes(n).map(([a,k,b])=>a*k*b);
    const expected={projections:products.slice(0,4).reduce((a,b)=>a+b,0),pairs:products[4]+products[5],predictor:products[6]+products[7],total:products.reduce((a,b)=>a+b,0)};
    assert.deepEqual(await page.evaluate(n=>AT.contextCost.counts(n),n),expected);
  }
  const grids=await page.locator('[data-cost-grid]').evaluateAll(es=>es.map(e=>({n:Number(e.dataset.costGrid),causal:e.hasAttribute('data-cost-causal'),last:e.hasAttribute('data-cost-last'),all:e.children.length,allowed:e.querySelectorAll('span:not(.blocked)').length})));
  for(const g of grids){assert.equal(g.all,g.n*g.n);assert.equal(g.allowed,g.causal?g.n*(g.n+1)/2:g.last?g.n:g.n*g.n);}
  await go('s16-cost-pairs',99);
  const sizes=await page.locator('#s16-cost-pairs [data-cost-grid]').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().width));
  close(sizes[1]/sizes[0],2,'Double side length, quadruple plotted area');
  await go('s16-cost-calculator');
  for(const n of [8,16,64,256,1024]){
    await page.locator('#cost-length').selectOption(String(n));
    const cells=await page.locator('#cost-counts .num').allTextContents();
    assert.deepEqual(cells.map(x=>Number(x.replaceAll(',',''))),[40*n,5*n*n,192*n,232*n+5*n*n]);
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Cost control fits at '+n);
  }
  await go('s16-cost-training');await go('s16-cost-calculator');
  assert.equal(await page.locator('#cost-length').inputValue(),'1024','Control state survives navigation.');
  await page.locator('#cost-length').selectOption('16');
  await page.screenshot({path:path.join(shots,'calculator-16.png')});
  for(const viewport of [{width:1280,height:720},{width:1024,height:768}]){
    await page.setViewportSize(viewport);
    for(const id of costs){await go(id);assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' in Part III at '+viewport.width);}
  }
  await page.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  await page.setViewportSize({width:1280,height:720});await page.evaluate(()=>document.fonts.ready);
  const original=await page.evaluate(()=>JSON.stringify({model:AT.model,result:AT.forward(AT.sentences.river)}));
  assert.deepEqual(await page.locator('.frame.context-lesson').evaluateAll(es=>es.map(e=>e.id)),ids);
  assert.equal(ids.length,43,'Include the alternatives divider and keep the worked limitations before relative positions, without duplicate recaps.');
  assert.deepEqual(await page.locator('.position-reading[id]:not(#s17-position-map-notes)').evaluateAll(es=>es.map(e=>e.id)),readingExtras,'Recaps remain available for reading.');
  assert.equal(await page.locator('#s17 [data-position-journey="overview"]').count(),1,'Use one closing map with selectable highlights.');
  for(const id of readingExtras){
    assert.equal(await page.locator('#'+id).evaluate(e=>e.classList.contains('frame')),false);
    assert(await page.locator('#'+id+' h3').count(),'Reading notes retain a descriptive heading.');
  }
  assert.equal(await page.locator('[id^="s16-cost-"]').count(),0,'cost lesson is absent from Part II');
  for(const shift of [0,5,10]){
    await go('s17-position-rope-shift');await page.locator('#rope-shift').selectOption(String(shift));
    const r=await page.evaluate(s=>AT.positionVisuals.shifted(s),shift);
    close(r.dot,Math.sqrt(3)/2,'relative rotary match after common shift');
    close(r.q[0]**2+r.q[1]**2,1,'query norm');close(r.k[0]**2+r.k[1]**2,1,'key norm');
    assert.match(await page.locator('#rope-shift-result').innerText(),/0.866/);
  }
  await page.locator('#rope-shift').selectOption('0');
  const base={Maya:[1,0],Ravi:[0,1],helps:[.2,.2],today:[.8,.2]},pos=[[0,0],[.2,-.1],[.4,-.2],[.6,-.3]];
  const sequences=[['Maya','helps','Ravi','today'],['Ravi','helps','Maya','today']];
  function reference(tokens,on){
    const rows=tokens.map((token,i)=>base[token].map((x,j)=>x+(on?pos[i][j]:0)));
    const scores=rows.map(row=>(rows[3][0]*row[0]+rows[3][1]*row[1])/Math.sqrt(2));
    const exps=scores.map(Math.exp),sum=exps.reduce((a,b)=>a+b);
    const weights=exps.map(x=>x/sum),message=rows[0].map((_,j)=>rows.reduce((s,r,i)=>s+r[j]*weights[i],0));
    return {rows,q:rows[3],scores,weights,message};
  }
  const unpositioned=sequences.map(tokens=>reference(tokens,false));
  // Independent reference: append one feature, not the additive toy's offsets.
  function appendedReference(tokens,scale){
    const rows=tokens.map((token,j)=>[...base[token],scale*(j+1)]),q=rows[3];
    const wordTerms=rows.map(row=>row[0]*.8+row[1]*.2);
    const positionTerms=rows.map((_,j)=>scale*scale*4*(j+1));
    const rawScores=rows.map(row=>row.reduce((sum,x,c)=>sum+x*q[c],0));
    const scores=rawScores.map(x=>x/Math.sqrt(3)),max=Math.max(...scores);
    const exps=scores.map(x=>Math.exp(x-max)),denominator=exps.reduce((a,b)=>a+b);
    return {rows,q,wordTerms,positionTerms,rawScores,scores,weights:exps.map(x=>x/denominator)};
  }
  const toyBefore=await page.evaluate(()=>JSON.stringify({e:AT.positionLesson.embeddings,p:AT.positionLesson.positions}));
  for(const scale of [0,.1,1,2]){
    for(const tokens of sequences){
      const expected=appendedReference(tokens,scale);
      const actual=await page.evaluate(({tokens,scale})=>AT.positionLesson.appendedExperiment(tokens,scale),{tokens,scale});
      assert.deepEqual(actual.rows,expected.rows);
      for(const key of ['q','wordTerms','positionTerms','rawScores','scores','weights'])
        actual[key].forEach((x,j)=>close(x,expected[key][j],'appended '+key));
      close(actual.weights.reduce((a,b)=>a+b),1,'appended weights normalize');
    }
  }
  for(const [index,suffix]of ['a','b'].entries()){
    assert.deepEqual(await page.locator('#position-append-'+suffix+' [data-vector]').evaluateAll(es=>es.map(e=>JSON.parse(e.dataset.vector))),appendedReference(sequences[index],1).rows);
  }
  const appendedColumns=await page.locator('#position-append-scores tbody tr').evaluateAll(rows=>rows.map(tr=>[...tr.querySelectorAll('[data-value]')].map(e=>Number(e.dataset.value))));
  appendedColumns.forEach((row,j)=>{
    const r=appendedReference(sequences[0],1);
    row.forEach((x,c)=>close(x,[r.wordTerms[j],r.positionTerms[j],r.rawScores[j]][c],'word / slot / raw score'));
  });
  const appendedSoftmax=await page.locator('#position-append-softmax tbody tr').evaluateAll(rows=>rows.map(tr=>[...tr.querySelectorAll('[data-value]')].map(e=>Number(e.dataset.value))));
  appendedSoftmax.forEach((row,j)=>{
    const r=appendedReference(sequences[0],1);
    row.forEach((x,c)=>close(x,[r.scores[j],Math.exp(r.scores[j]),r.weights[j]][c],'appended softmax working'));
  });
  await go('s17-position-append-scale');
  for(const scale of [1,.1,1,.1]){
    await page.locator('#position-append-scale').selectOption(String(scale));
    for(const [index,suffix]of ['a','b'].entries()){
      const r=appendedReference(sequences[index],scale);
      const actual=await page.locator('#position-append-weights-'+suffix+' tbody tr').evaluateAll(rows=>rows.map(tr=>[...tr.querySelectorAll('[data-value]')].map(e=>Number(e.dataset.value))));
      actual.forEach((row,j)=>{close(row[0],r.scores[j],'live appended score');close(row[1],r.weights[j],'live appended weight');});
      assert.deepEqual(await page.locator('#position-append-weights-'+suffix+' .weight').allTextContents(),r.weights.map(x=>(100*x).toFixed(1)+'%'));
      assert(scale===1?r.weights[3]>.89:r.weights[3]<.30,'Position scale controls the demonstrated preference.');
    }
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Appended-scale control fits at '+scale);
    await page.screenshot({path:path.join(shots,'append-scale-'+scale+'.png')});
  }
  await go('s17-position-append-tradeoffs');await go('s17-position-append-scale');
  assert.equal(await page.locator('#position-append-scale').inputValue(),'0.1','Appended scale survives navigation');
  await page.locator('#position-append-scale').selectOption('1');
  assert.equal(await page.evaluate(()=>JSON.stringify({e:AT.positionLesson.embeddings,p:AT.positionLesson.positions})),toyBefore,'Appended toy preserves existing embeddings and offsets.');
  assert.match(await page.locator('#s17-position-append-tradeoffs').innerText(),/Concatenation can work/);
  assert.match(await page.locator('#s17-position-updated').innerText(),/representation before attention/);
  assert.match(await page.locator('#s17-position-updated svg').textContent(),/e′₄ = e₄ \+ Δe₄/);
  assert(!/x[′_T]|XW/.test(await page.locator('#position-overview-map svg').textContent()),'Position map retains the established e/E notation.');
  // The geometric displacements and final updates share the worked arithmetic.
  for(const [index,suffix]of ['a','b'].entries()){
    const id='s17-position-move-'+suffix;
    const vectors=await page.locator('#'+id+' [data-endpoint]').evaluateAll(es=>es.map(e=>({token:e.dataset.endpoint,slot:Number(e.dataset.slot),v:JSON.parse(e.dataset.vector),x:Number(e.getAttribute('cx')),y:Number(e.getAttribute('cy'))})));
    vectors.forEach((point,j)=>{
      assert.equal(point.token,sequences[index][j]);assert.equal(point.slot,j+1);
      const expected=reference(sequences[index],true).rows[j];
      point.v.forEach((x,c)=>close(x,expected[c],'plotted endpoint coordinate'));
      close(point.x,85+200*expected[0],'geometric x coordinate');close(point.y,270-200*expected[1],'geometric y coordinate');
    });
    for(const build of [0,1,2,3,4,3,2,1,0]){
      await go(id,build);
      const visible=await page.locator('#'+id+' [data-move-slot]').evaluateAll(es=>es.filter(e=>getComputedStyle(e).visibility==='visible').map(e=>Number(e.dataset.moveSlot)));
      assert.deepEqual(visible,Array.from({length:build},(_,i)=>i),'One additional word displacement per build, reversible');
    }
  }
  const updates=await page.locator('[data-updated-vector]').evaluateAll(es=>es.map(e=>JSON.parse(e.dataset.updatedVector)));
  updates.forEach((v,j)=>{const r=reference(sequences[j],true);v.forEach((x,c)=>close(x,r.q[c]+r.message[c],'context update after position addition'));});
  await go('s17-position-clock');
  for(const i of [0,1,2,3,4,0]){
    await page.locator('#position-clock-index').fill(String(i));
    assert.equal(await page.locator('#position-clock-label').textContent(),String(i));
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Clock control fits');
  }
  await go('s17-position-repeat');
  for(const i of [0,4,8,12,0]){
    await page.locator('#position-pair-index').selectOption(String(i));
    const actual=JSON.parse(await page.locator('#position-pair-result').getAttribute('data-vector'));
    [Math.cos(i*Math.PI/2),Math.sin(i*Math.PI/2),Math.cos(i*Math.PI/6),Math.sin(i*Math.PI/6)].forEach((x,c)=>close(actual[c],x,'two-clock code'));
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Two-clock control fits');
  }
  const absScores=await page.locator('[data-absolute-score]').evaluateAll(es=>es.map(e=>Number(e.dataset.absoluteScore)));
  close(absScores[0],1.5+Math.sqrt(3)/2,'absolute additive match at 3,2');close(absScores[1],.5,'absolute additive match at 8,7');
  assert.deepEqual(await page.locator('[data-missing-position]').evaluateAll(es=>es.map(e=>Number(e.dataset.missingPosition))),[4,5],'Four-row learned table has no row at indices 4 or 5.');
  const shiftExamples=await page.locator('[data-shift-example]').evaluateAll(es=>es.map(e=>({example:Number(e.dataset.shiftExample),index:Number(e.dataset.position),word:e.dataset.word})));
  for(const example of [0,1]){
    const row=shiftExamples.filter(e=>e.example===example);
    const words=example?['At','the','park','after','lunch',...sequences[0]]:sequences[0];
    assert.deepEqual(row.map(e=>e.word),words,'Full sentence and prefix are visible.');
    assert.deepEqual(row.map(e=>e.index),words.map((_,i)=>i),'Every word has its correct zero-based slot.');
    const receiver=row.find(e=>e.word==='today').index,source=row.find(e=>e.word==='Ravi').index;
    assert.deepEqual([receiver,source],example?[8,7]:[3,2]);assert.equal(receiver-source,1,'Shared shift preserves the gap.');
  }
  const relativeRows=await page.locator('#s17-position-relative tbody tr').evaluateAll(es=>es.map(e=>[Number(e.cells[0].textContent),Number(e.cells[1].textContent)]));
  assert.deepEqual(relativeRows,[[3,2],[8,7]],'Relative bias reuses the exact absolute-position example.');
  await go('s17-position-overview');
  const mapGeometry=await page.locator('#position-overview-map [data-map-node]').evaluateAll(es=>es.map(e=>[e.dataset.mapNode,...['x','y','width','height'].map(a=>e.getAttribute(a))]));
  for(const focus of ['all','input','attention','output','all']){
    await page.locator('#position-map-focus').selectOption(focus);
    assert.equal(await page.locator('#position-overview-map').getAttribute('data-focus'),focus);
    assert.deepEqual(await page.locator('#position-overview-map [data-map-node]').evaluateAll(es=>es.map(e=>e.dataset.mapNode)),['tokens','positions','input','query','keys','values','scores','weights','message','projection','residual','hidden','prediction']);
    assert.deepEqual(await page.locator('#position-overview-map [data-map-node]').evaluateAll(es=>es.map(e=>[e.dataset.mapNode,...['x','y','width','height'].map(a=>e.getAttribute(a))])),mapGeometry,'Highlight changes preserve all node locations.');
    const emphasized=await page.locator('#position-overview-map [data-map-node]').evaluateAll(es=>[...new Set(es.filter(e=>e.getAttribute('stroke-width')==='2.5').map(e=>e.closest('[data-map-stage]').dataset.mapStage))]);
    assert.deepEqual(emphasized,focus==='all'?['input','attention','output']:[focus],'Highlight changes the intended stage.');
    const spillingLabels=await page.locator('#position-overview-map [data-map-node]').evaluateAll(es=>es.flatMap(rect=>{
      const b=rect.getBBox(),title=rect.nextElementSibling,caption=title.nextElementSibling;
      return [title,caption].filter(e=>{const t=e.getBBox();return t.x<b.x+3||t.x+t.width>b.x+b.width-3||t.y<b.y||t.y+t.height>b.y+b.height;}).map(e=>e.textContent);
    }));
    assert.deepEqual(spillingLabels,[],focus+' node labels fit their boxes');
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Map focus '+focus+' fits.');
    await page.screenshot({path:path.join(shots,'map-'+focus+'.png')});
  }
  await page.locator('#position-map-focus').selectOption('input');
  await go('s17-position-insertion');await go('s17-position-overview');
  assert.equal(await page.locator('#position-map-focus').inputValue(),'input','Map highlight survives navigation.');
  await page.locator('#position-map-focus').selectOption('all');
  for(const wave of await page.locator('[data-wave]').evaluateAll(es=>es.map(e=>({rate:Number(e.dataset.rate),end:Number(e.dataset.end),kind:e.dataset.wave,points:[...e.points].map(p=>[p.x,p.y])})))){
    const fn=wave.kind==='sine'?Math.sin:Math.cos;
    // SVG DOM stores coordinates as float32. Test the curve against its formula.
    const center=wave.points[0][1]+44*fn(0);
    wave.points.forEach(([x,y],i)=>{
      assert(Math.abs(x-(170+880*i/1200))<.0001,'wave horizontal scale');
      assert(Math.abs(y-(center-44*fn(wave.rate*wave.end*i/1200)))<.0001,'sinusoidal curve');
    });
  }
  assert(Math.abs(unpositioned[0].weights[0]-unpositioned[0].weights[2])>.05,'Use unequal weights so the example is not just averaging.');
  for(const [index,suffix]of ['a','b'].entries()){
    const r=unpositioned[index];
    const actual=await page.locator('#position-score-'+suffix+' tbody tr').evaluateAll(rows=>rows.map(tr=>[...tr.querySelectorAll('[data-value]')].map(td=>Number(td.dataset.value))));
    actual.forEach((row,j)=>{
      const expected=[r.rows[j].reduce((sum,x,c)=>sum+x*r.q[c],0),r.scores[j],Math.exp(r.scores[j]),r.weights[j]];
      row.forEach((x,c)=>close(x,expected[c],'worked score/softmax column'));
    });
    const terms=await page.locator('#position-contribution-'+suffix+' [data-vector]').evaluateAll(es=>es.map(e=>JSON.parse(e.dataset.vector)));
    terms.forEach((row,j)=>row.forEach((x,c)=>close(x,r.weights[j]*r.rows[j][c],'weighted value coordinate')));
    assert.equal(await page.locator('#position-contribution-'+suffix+' tfoot td:last-child').textContent(),'['+r.message.map(x=>x.toFixed(3)).join(', ')+']');
  }
  assert.deepEqual(await page.locator('#s17-position-break .position-credit a').evaluateAll(es=>es.map(e=>new URL(e.href).searchParams.get('v'))),['IHu3QehUmrQ','SMBkImDWOyQ'],'Both videos receive visible credit.');
  const mayaScores=await page.locator('#position-maya-slots [data-value]').evaluateAll(es=>es.map(e=>Number(e.dataset.value)));
  sequences.forEach((tokens,i)=>close(mayaScores[i],reference(tokens,true).scores[tokens.indexOf('Maya')],'Maya positional score'));
  for(const on of [false,true]){
    const results=[];
    for(const tokens of sequences){
      const actual=await page.evaluate(({tokens,on})=>AT.positionLesson.experiment(tokens,on),{tokens,on}),expected=reference(tokens,on);
      assert.deepEqual(actual.rows,expected.rows);
      for(const key of ['q','scores','weights','message'])actual[key].forEach((x,i)=>close(x,expected[key][i],key));
      close(actual.weights.reduce((a,b)=>a+b),1,'weights sum');results.push(actual);
    }
    if(on)assert(Math.abs(results[0].message[0]-results[1].message[0])>.01,'Positions must distinguish the orders.');
    else results[0].message.forEach((x,i)=>close(x,results[1].message[i],'no-position invariance'));
    const means=results.map(r=>r.rows[0].map((_,j)=>r.rows.reduce((s,row)=>s+row[j]/4,0)));
    means[0].forEach((x,i)=>close(x,means[1][i],'plain mean remains order blind'));
  }
  await go('s17-position-experiment');
  for(const state of ['off','on','off','on']){
    await page.locator('#position-enabled').selectOption(state);
    const expected=sequences.map(tokens=>reference(tokens,state==='on'));
    for(const [i,suffix]of ['a','b'].entries()){
      assert.deepEqual(await page.locator('#position-weights-'+suffix+' .num').allTextContents(),expected[i].weights.map(x=>x.toFixed(3)));
      assert.equal(await page.locator('#position-message-'+suffix).textContent(),'message ['+expected[i].message.map(x=>x.toFixed(3)).join(', ')+']');
    }
    assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,'Position control '+state+' fits');
    await page.screenshot({path:path.join(shots,'position-'+state+'.png')});
  }
  await go('s17-position-append');await go('s17-position-experiment');
  assert.equal(await page.locator('#position-enabled').inputValue(),'on');
  for(let i=0;i<4;i++){
    const values=[Math.sin(i),Math.cos(i),Math.sin(i/100),Math.cos(i/100)];
    assert.deepEqual(await page.locator('#position-sinusoids tr').nth(i).locator('td').allTextContents(),[String(i),...values.map(x=>x.toFixed(3))]);
  }
  for(const [i,j]of [[3,2],[8,7],[5,2]])close(await page.evaluate(([i,j])=>AT.positionLesson.rotaryDot(i,j),[i,j]),Math.cos((i-j)*Math.PI/6),'RoPE relative offset');
  const alibi=[1,1.5,2].map(Math.exp),denom=alibi.reduce((a,b)=>a+b);
  assert.deepEqual(await page.locator('#s17-position-alibi tbody tr td:last-child').allTextContents(),alibi.map(x=>(100*x/denom).toFixed(1)+'%'));
  // Every authored build must fit; reverse traversal must restore hidden math.
  let states=0;
  for(const id of ids){
    await go(id);const max=await page.evaluate(()=>AT.present.state().frame.maxBuild);
    assert(await page.locator('#'+id+' script[type="text/x-notes"]').count(),id+' has teaching notes');
    for(const build of [...Array(max+1).keys(),0,max]){
      await go(id,build);const fit=await page.evaluate(()=>AT.present.fitReport());assert(!fit.overflow,id+' '+build+': '+JSON.stringify(fit));states++;
      const wrongMath=await page.locator('#'+id+' [data-build] .katex-html').evaluateAll(es=>es.flatMap(e=>{
        const pending=!!e.closest('.is-pending');
        return [...e.querySelectorAll('*')].filter(x=>!x.children.length&&x.textContent.replace(/[\s\u200b]/g,'')&&((getComputedStyle(x).visibility==='visible')===pending)).map(x=>x.textContent);
      }));
      assert.deepEqual(wrongMath,[],id+' math layers must follow build '+build+' in both directions');
    }
    const undersized=await page.locator('#'+id).evaluate(e=>[...e.querySelectorAll('p,td,th,label,select')].filter(x=>getComputedStyle(x).visibility!=='hidden'&&parseFloat(getComputedStyle(x).fontSize)<21).map(x=>x.textContent));
    assert.deepEqual(undersized,[],id+' has readable classroom text');
    await page.screenshot({path:path.join(shots,id+'.png')});
    const clipped=await page.locator('#'+id+' .position-visual svg, #'+id+' .position-journey svg').evaluateAll(es=>es.flatMap(s=>{
      const v=s.viewBox.baseVal;return [...s.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>v.width+1||b.y+b.height>v.height+1;}).map(t=>t.textContent);
    }));
    assert.deepEqual(clipped,[],id+' diagram text is not clipped');
  }
  // Match both widescreen sharing and older 4:3 projector viewports.
  for(const viewport of [{width:1920,height:1080},{width:1024,height:768}]){
    await page.setViewportSize(viewport);
    for(const id of ids){await go(id);assert(!(await page.evaluate(()=>AT.present.fitReport())).overflow,id+' at '+viewport.width);}
    await go('s17-position-experiment');await page.screenshot({path:path.join(shots,'projector-'+viewport.width+'.png')});
  }
  await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
  for(const id of readingExtras)assert(await page.locator('#'+id).isVisible(),'Recap is available in reading mode: '+id);
  const spills=await page.locator('.context-lesson').evaluateAll(es=>es.filter(e=>e.getBoundingClientRect().right>innerWidth+2||e.scrollWidth>e.clientWidth+2).map(e=>({id:e.id,client:e.clientWidth,scroll:e.scrollWidth})));
  assert.deepEqual(spills,[],'Mobile reading frames remain contained.');
  await page.locator('#s17-position-experiment').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,'reading-phone.png')});
  assert.equal(await page.evaluate(()=>JSON.stringify({model:AT.model,result:AT.forward(AT.sentences.river)})),original,'Extensions must not mutate the bank model.');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({frames:ids.length,buildChecks:states,viewports:['1280×720','1920×1080','1024×768','390×844 reading'],checks:'MACs, grids, softmax, word-dot endpoints/reveals, appended-position rows/score terms/scales, e notation, position/context updates, order swap, mean invariance, clock collisions, sinusoid curves, absolute/rotary shifts, ALiBi, complete map, controls, model immutability',screenshots:shots},null,2));
}finally{await browser.close();}

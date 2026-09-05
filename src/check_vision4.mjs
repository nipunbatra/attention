// Regression checks for the Vision IV toy (toy8.json v2): the frozen Vision I encoder, the fitted
// connector and width-three prefix decoder, the saved NumPy reference, gradients and generation.
// Run: node src/check_vision4.mjs [--browser] [--screenshots /tmp/vision4-check]
// Reproduce the saved NumPy model separately: python3 src/train_vision4.py --check
// Browser mode builds the current source fragments in memory, never published HTML.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const src=path.dirname(fileURLToPath(import.meta.url));
const read=name=>fs.readFileSync(path.join(src,name),'utf8');
const toy=JSON.parse(read('toy8.json')),data=toy.vlm,original=JSON.stringify(toy);
const toy5=JSON.parse(read('toy5.json'));
const clone=x=>JSON.parse(JSON.stringify(x));
const sum=r=>r.reduce((a,b)=>a+b,0);
const mm=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0)));
const tr=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const add=(a,b)=>a.map((r,i)=>r.map((x,j)=>x+b[i][j]));
const zeros=(r,c)=>Array.from({length:r},()=>Array(c).fill(0));
const sm=r=>{const peak=Math.max(...r),ex=r.map(x=>x===-Infinity?0:Math.exp(x-peak)),total=sum(ex);return ex.map(x=>x/total);};
const NP=data.imageRows,VOC=data.vocab,PROMPT=data.prompt,D=toy.d_model;

// A minimal DOM so vision-shared.js and part8.js load outside a browser (they add one <style> each).
const fake=()=>({appendChild(){},setAttribute(){},addEventListener(){},childNodes:[],style:{},classList:{add(){},toggle(){}}});
const AT={h:()=>fake(),svg:()=>fake(),fmt:(v,n=2)=>Number(v).toFixed(n),escape:s=>String(s),range:n=>Array.from({length:n},(_,i)=>i),
  argmax:r=>r.indexOf(Math.max(...r)),notation:[],axes:{named:true,e:toy.axes.e,qk:toy.axes.qk,v:toy.axes.v,short:toy.axes.short},tex(){},ui:{notationCard(){return fake();},chips:()=>fake()}};
const sandbox={window:{AT,__TOY__:toy,__PART__:{notation:'vision4'}},document:{head:fake(),getElementById:()=>null}};
vm.createContext(sandbox);
vm.runInContext(read('vision-shared.js'),sandbox);
vm.runInContext(read('part8.js'),sandbox);
const T=AT.vlm,V=AT.vision;
assert(T&&V,'runtime loaded');
let comparisons=0,maximumReferenceError=0;
function compare(actual,expected,label='value',tolerance=1e-11){
  if(Array.isArray(expected)){
    assert(Array.isArray(actual),label);assert.equal(actual.length,expected.length,label+' length');
    expected.forEach((v,i)=>compare(actual[i],v,`${label}[${i}]`,tolerance));
  }else if(expected&&typeof expected==='object'){
    for(const[k,v]of Object.entries(expected))compare(actual[k],v,`${label}.${k}`,tolerance);
  }else if(expected===null){
    assert.equal(actual,-Infinity,`${label}: null is reserved for masked negative infinity`);
  }else if(typeof expected==='number'){
    assert(Number.isFinite(actual),label+' finite');const error=Math.abs(actual-expected);
    comparisons++;maximumReferenceError=Math.max(maximumReferenceError,error);
    assert(error<=tolerance,`${label}: ${actual} != ${expected}, error ${error}`);
  }else assert.equal(actual,expected,label);
}
function shape(m,rows,columns,label){assert.equal(m.length,rows,label);m.forEach(r=>{assert.equal(r.length,columns,label);assert(r.every(Number.isFinite),label);});}
function normalized(m,label){m.forEach((r,i)=>{assert(r.every(v=>Number.isFinite(v)&&v>=0&&v<=1),label);compare(sum(r),1,`${label} row ${i} sum`);});}

// 1. The frozen encoder is exactly Vision I's: toy8.encoder equals toy5.trained, and an independent
//    implementation of the shared worksheet (fixed patch map, CLS, positions in thirds, /sqrt 2) agrees with V.encode.
compare(toy.encoder,toy5.trained,'encoder copied from Vision I');
const W_PATCH=[[.25,.5,0,0],[.25,-.5,0,0],[.25,.5,0,0],[.25,-.5,0,0]],CLS=[1,0,0,0];
const POS=[[0,0,-1,-1]].concat(Array.from({length:16},(_,j)=>[0,0,Math.floor(j/4)/3,(j%4)/3]));
function encodeIndependently(grid){
  const R=[];for(let pr=0;pr<4;pr++)for(let pc=0;pc<4;pc++)R.push([grid[2*pr][2*pc],grid[2*pr][2*pc+1],grid[2*pr+1][2*pc],grid[2*pr+1][2*pc+1]]);
  const E=add([CLS].concat(mm(R,W_PATCH)),POS),e=toy.encoder;
  const Q=mm(E,e.W_Q),K=mm(E,e.W_K),Vv=mm(E,e.W_V);
  const A=Q.map(q=>sm(K.map(k=>(q[0]*k[0]+q[1]*k[1])/Math.SQRT2)));
  return add(E,mm(mm(A,Vv),e.W_O));
}
for(const scene of ['A','B','C']){
  compare(V.encode(scene),encodeIndependently(data.scenes[scene]),scene+' encoder');
  compare(V.scenes[scene],data.scenes[scene],scene+' scene pixels');
  compare(T.rows(scene),data.visualRows[scene],scene+' saved visual rows');
  shape(T.rows(scene),16,4,'sixteen visual rows of width four');
  compare(V.mugPatches(scene),data.mugPatches[scene],scene+' mug patches');
}
assert.equal(AT.axes.named,true,'the learned axes carry names');
assert(toy.axes.e.every(n=>!/coordinate/.test(n))&&new Set(toy.axes.qk).size===3&&new Set(toy.axes.v).size===3,'three distinct names per space');

// 2. Independent loss: only the receiving answer rows, with explicit allowed source lists. It does not
//    call the production attention or loss code.
function scalarLoss(scene,p){
  const answer=data.answers[scene],prefix=PROMPT.concat(answer.slice(0,-1)),ids=prefix.map(t=>VOC.indexOf(t));
  const G=data.visualRows[scene];
  const bridge=mm(G,p.W_bridge).map(r=>r.map((v,j)=>v+p.b_bridge[j]));
  const rows=bridge.concat(ids.map((id,i)=>p.E_tok[id].map((v,j)=>v+p.P[i][j])));
  const terms=[];
  for(let t=0;t<answer.length;t++){
    const i=NP+PROMPT.length-1+t,q=mm([rows[i]],p.W_Q)[0],keys=mm(rows.slice(0,i+1),p.W_K),values=mm(rows.slice(0,i+1),p.W_V);
    const scores=keys.map(k=>sum(q.map((v,j)=>v*k[j]))/Math.sqrt(D));
    const weights=sm(scores),message=[0,1,2].map(j=>sum(values.map((v,k)=>weights[k]*v[j])));
    const update=mm([message],p.W_O)[0],row=rows[i].map((v,j)=>v+update[j]);
    const logits=mm([row],p.W_vocab)[0].map((v,j)=>v+p.b_vocab[j]);
    const peak=Math.max(...logits),logZ=peak+Math.log(sum(logits.map(v=>Math.exp(v-peak))));
    terms.push(logZ-logits[VOC.indexOf(answer[t])]);
  }
  return sum(terms)/terms.length;
}
// 3. A separate backward pass for the gradient; the saved NumPy gradient is compared below.
function analyticGradient(f,p){
  const dz=zeros(f.logits.length,VOC.length),targets=f.targets;
  targets.forEach((target,t)=>{const i=f.targetRows[t];dz[i]=f.probs[i].map(v=>v/targets.length);dz[i][VOC.indexOf(target)]-=1/targets.length;});
  const g={W_vocab:mm(tr(f.out),dz),b_vocab:tr(dz).map(sum)};
  const dout=mm(dz,tr(p.W_vocab));
  g.W_O=mm(tr(f.message),dout);
  const dm=mm(dout,tr(p.W_O)),da=mm(dm,tr(f.V)),dv=mm(tr(f.A),dm);
  const ds=f.A.map((r,i)=>{const center=sum(r.map((a,j)=>a*da[i][j]));return r.map((a,j)=>a*(da[i][j]-center)/Math.sqrt(D));});
  const dq=mm(ds,f.K),dk=mm(tr(ds),f.Q);
  g.W_Q=mm(tr(f.E),dq);g.W_K=mm(tr(f.E),dk);g.W_V=mm(tr(f.E),dv);
  const de=add(add(dout,mm(dq,tr(p.W_Q))),add(mm(dk,tr(p.W_K)),mm(dv,tr(p.W_V))));
  g.W_bridge=mm(tr(f.G),de.slice(0,NP));g.b_bridge=tr(de.slice(0,NP)).map(sum);
  g.E_tok=zeros(p.E_tok.length,D);f.ids.forEach((id,i)=>de[i+NP].forEach((v,j)=>g.E_tok[id][j]+=v));
  g.P=zeros(p.P.length,D);f.ids.forEach((_,i)=>g.P[i]=de[i+NP].slice());
  return g;
}
let gradientChecks=0,maximumGradientError=0;
const reports={};
for(const snapshot of ['trained','step']){
  const p=T.params(snapshot);reports[snapshot]={};
  for(const scene of ['A','B']){
    const f=T.teacher(scene,{snapshot}),g=analyticGradient(f,p),n=f.n;
    compare(f,data.reference[snapshot][scene],`${snapshot}/${scene} NumPy reference`);
    shape(f.E,NP+PROMPT.length+1,D,'teacher-forced input');shape(f.Q,n,D,'Q');shape(f.K,n,D,'K');shape(f.V,n,D,'V');shape(f.out,n,D,'updated rows');
    normalized(f.A,'attention');normalized(f.probs,'vocabulary');
    f.allowed.forEach((row,i)=>row.forEach((allowed,j)=>{
      assert.equal(allowed,i<NP?j<NP:j<=i,'image-only prefix / causal text mask');
      if(!allowed){assert.equal(f.scores[i][j],-Infinity);assert.equal(f.A[i][j],0);}
    }));
    compare(f.B,mm(f.G,p.W_bridge).map(r=>r.map((v,j)=>v+p.b_bridge[j])),'connector');
    compare(f.E,f.B.concat(f.ids.map((id,i)=>p.E_tok[id].map((v,j)=>v+p.P[i][j]))),'image rows then text rows with positions');
    compare(f.Q,mm(f.E,p.W_Q),'query origin');compare(f.K,mm(f.E,p.W_K),'key origin');compare(f.V,mm(f.E,p.W_V),'value origin');
    compare(f.delta,mm(mm(f.A,f.V),p.W_O),'value sum and output projection');compare(f.out,add(f.E,f.delta),'residual addition');
    compare(f.loss,scalarLoss(scene,p),'independent answer-only mean loss');
    for(let length=1;length<=f.prefix.length;length++){
      const short=T.forward(scene,f.prefix.slice(0,length),{snapshot});
      compare(short.out,f.out.slice(0,NP+length),'extending the prefix leaves earlier rows unchanged');
      compare(short.probs,f.probs.slice(0,NP+length),'extending the prefix leaves earlier logits unchanged');
    }
    const other=scene==='A'?'one':'two';
    const changedFuture=T.forward(scene,PROMPT.concat([other]),{snapshot});
    compare(changedFuture.out.slice(0,NP+PROMPT.length),f.out.slice(0,NP+PROMPT.length),'a different answer token cannot leak into the first prediction');
    const generation=T.generate(scene,{snapshot}),saved=data.generation[snapshot][scene];
    compare(generation.tokens,saved.tokens,'greedy tokens');assert.equal(generation.stoppedBy,saved.stoppedBy);
    generation.trace.forEach((step,i)=>{
      compare({prefix:step.prefix,row:step.row,query:step.query,weights:step.weights,logits:step.logits,probs:step.probs,chosen:step.chosen},saved.trace[i],'saved greedy trace');
      compare(step.prefix,PROMPT.concat(generation.tokens.slice(0,i)),'self-generated prefix');
      assert.equal(step.row,NP+step.prefix.length-1,'last known text position');
      assert.equal(step.chosen,VOC[step.probs.indexOf(Math.max(...step.probs))],'actual argmax choice');
    });
    assert.equal(generation.stoppedBy,'eos','generation stops at the end marker');
    if(snapshot==='trained'&&scene==='A')compare(g,data.update.gradients,'independent backward versus NumPy gradient');
    let localMaximum=0;
    for(const epsilon of [1e-5,1e-6])for(const[key,value]of Object.entries(p)){
      const isMatrix=Array.isArray(value[0]);
      for(let i=0;i<value.length;i++)for(let j=0;j<(isMatrix?value[i].length:1);j++){
        const old=isMatrix?value[i][j]:value[i],set=v=>isMatrix?value[i][j]=v:value[i]=v;
        let high,low;
        try{set(old+epsilon);high=scalarLoss(scene,p);set(old-epsilon);low=scalarLoss(scene,p);}finally{set(old);}
        const analytic=isMatrix?g[key][i][j]:g[key][i],error=Math.abs((high-low)/(2*epsilon)-analytic);
        gradientChecks++;localMaximum=Math.max(localMaximum,error);maximumGradientError=Math.max(maximumGradientError,error);
        assert(error<1e-6,`${snapshot}/${scene}/${key}[${i},${j}] gradient error ${error}`);
      }
    }
    reports[snapshot][scene]={loss:f.loss,perToken:f.losses,targetProbs:f.targetProbs,greedy:generation.tokens,maxGradientError:localMaximum};
  }
  reports[snapshot].C={greedy:T.generate('C',{snapshot}).tokens,probs:T.forward('C',PROMPT,{snapshot}).probs[NP+PROMPT.length-1]};
  compare(T.generate('C',{snapshot}).tokens,data.generation[snapshot].C.tokens,'scene C probe');
}
// 4. The saved NumPy finite-difference checks and the exact SGD step.
assert(data.update.gradientChecks.every(c=>c.error<1e-7),'saved finite-difference checks');
assert.equal(data.update.gradientChecks.length,2*145,'two scenes times 145 trainable scalars');
const trained=T.params('trained'),step=T.params('step'),rate=data.update.rate;
let parameterCount=0;
for(const[key,values]of Object.entries(trained)){
  const isMatrix=Array.isArray(values[0]);
  for(let i=0;i<values.length;i++)for(let j=0;j<(isMatrix?values[i].length:1);j++){
    const get=x=>isMatrix?x[key][i][j]:x[key][i];
    compare(get(step),get(trained)-rate*get(data.update.gradients),'exact SGD parameter update');parameterCount++;
    assert.equal(Math.round(get(trained)*100)/100,get(trained),'the exported checkpoint is rounded to two decimals');
  }
}
assert.equal(parameterCount,145,'trainable scalar count');
assert.equal(gradientChecks,2*2*2*145,'145 scalars × two scenes × two snapshots × two finite-difference widths');
compare(reports.trained.A.loss,data.update.lossBefore.A,'saved loss before (A)');compare(reports.trained.B.loss,data.update.lossBefore.B,'saved loss before (B)');
compare(reports.step.A.loss,data.update.lossAfter.A,'saved loss after (A)');compare(reports.step.B.loss,data.update.lossAfter.B,'saved loss after (B)');
compare(reports.trained.A.perToken,data.update.perTokenBefore.A,'per-token losses before');compare(reports.step.A.perToken,data.update.perTokenAfter.A,'per-token losses after');
assert(reports.step.A.loss<reports.trained.A.loss,'the step lowers the loss of the scene it used');
// This regression is the teaching point of the section, not a quality pass.
assert(reports.step.B.loss>reports.trained.B.loss,'the other scene gets worse after a one-scene update');
assert(reports.step.A.perToken[1]>reports.trained.A.perToken[1],'the second target of scene A also rises while the mean falls');
// 5. Targets from VISION_AXES: p >= 0.9 on both fitted scenes, the answer token reads the mug patches.
assert(reports.trained.A.targetProbs[0]>=0.9&&reports.trained.B.targetProbs[0]>=0.9,'answers with p >= 0.9');
for(const scene of ['A','B']){
  const w=T.generate(scene).trace[0].weights,m=T.mass(w,scene);
  assert(m.mug>=0.6*m.image&&m.image>=0.9,`${scene}: the first answer token reads the mug patches (${m.mug.toFixed(3)} of ${m.image.toFixed(3)})`);
  compare(m.image+m.text,1,'weights account for every source');
}
const mA=T.mass(T.generate('A').trace[0].weights,'A');
assert(mA.left>0.3&&mA.right>0.3,'"two" reads both mugs of scene A');
const gC=T.generate('C');
reports.probe={C:gC.tokens,pTwo:gC.trace[0].probs[VOC.indexOf('two')],mugShare:T.mass(gC.trace[0].weights,'C').mug};
// 6. Bookkeeping of the two-versus-one logit difference, the axis names, and the readings.
for(const scene of ['A','B','C'])for(const prefix of [PROMPT,PROMPT.concat([data.generation.trained[scene].tokens[0]])]){
  const c=T.contrast(scene,prefix),f=T.forward(scene,prefix),i=f.last,p=T.params();
  compare(c.sources.map(s=>s.weight),f.A[i],'source weights retain all image and text mass');
  compare(tr(c.sources.map(s=>s.update)).map(sum),f.delta[i],'source updates sum to the contextual update');
  compare(c.residual+c.image+c.text,f.logits[i][VOC.indexOf('two')]-f.logits[i][VOC.indexOf('one')],'residual and source terms reproduce the logit contrast');
}
const two=T.forward('A'),one=T.forward('B');
compare(two.Q[two.last],one.Q[one.last],'same first query for the same known text in this one-layer decoder');
assert(Math.abs(two.probs[two.last][VOC.indexOf('two')]-one.probs[one.last][VOC.indexOf('two')])>.5,'the image changes the first answer distribution');
assert.equal(T.generate('A',{limit:1}).stoppedBy,'limit');assert.equal(T.generate('A',{limit:1}).trace.length,1);
for(const bad of [0,-1,4,1.5])assert.throws(()=>T.generate('A',{limit:bad}));
assert.throws(()=>T.forward('unknown'));assert.throws(()=>T.forward('A',[]));assert.throws(()=>T.forward('A',['unknown']));assert.throws(()=>T.params('missing'));
assert.match(T.readQuery(two.Q[two.last]),/asks for/);assert.match(T.readWeights('A',two.A[two.last]),/mug patches together/);
// The generator is not an answer-key animation: altering the answer labels must not change the greedy output.
const savedAnswers=clone(data.answers);
try{data.answers.A=['one','<eos>'];compare(T.generate('A').tokens,['two','<eos>'],'generation ignores the answer key');}finally{data.answers=savedAnswers;}
assert.equal(JSON.stringify(toy),original,'checks did not mutate saved data');
// 7. Fragments: every frame has notes; no "coordinate N"; no dangling references to an earlier article.
const sections=fs.readdirSync(path.join(src,'sections8')).filter(n=>/^sec\d\d\.html$/.test(n)).sort();
const fragments=sections.map(n=>read('sections8/'+n)).join('\n');
const frames=(fragments.match(/class="frame"/g)||[]).length;
assert.equal(sections.length,10,'ten sections');
assert.equal((fragments.match(/type="text\/x-notes"/g)||[]).length,frames,'every frame has presenter notes');
assert(!/coordinate \d/i.test(fragments),'no "coordinate N" labels');
assert(!/earlier article|earlier VLM article|thermal/i.test(fragments),'no dangling references to another article');
assert(!/[—–]/.test(fragments),'no em or en dashes in the fragments');
console.log(JSON.stringify({numeric:'pass',referenceComparisons:comparisons,maximumReferenceError,parameterCount,gradientChecks,maximumGradientError,frames,reports},null,2));
if(!process.argv.includes('--browser'))process.exit(0);

// Optional offline rendering, control and reveal checks using an installed browser.
const require=createRequire(import.meta.url),candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const d of fs.readdirSync(cache).sort())candidates.push(path.join(cache,d,'node_modules','playwright'));
let pw;for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
if(!pw)throw new Error('No installed Playwright runtime. No dependencies were installed by this check.');
const outIndex=process.argv.indexOf('--screenshots'),out=outIndex>=0?process.argv[outIndex+1]:fs.mkdtempSync(path.join(os.tmpdir(),'vision4-check-'));
fs.mkdirSync(out,{recursive:true});
const config=JSON.parse(read('part8.json'));if(config.prev)config.prev.available=false;if(config.next)config.next.available=false;
const json=value=>JSON.stringify(value).replace(/<\//g,'<\\/');
const sceneAssets=Object.fromEntries([['two','two-mugs.jpg'],['one','one-mug.jpg']].map(([variant,file])=>[variant,'data:image/jpeg;base64,'+fs.readFileSync(path.join(src,'..','figures','vision-scene',file)).toString('base64')]));
const shared='<script>window.__TOY__='+json(toy)+';window.__PART__='+json(config)+';</script><script>'+read('shared.js')+'</script><script>'+read('vision-shared.js')+'</script><script>'+read('part8.js')+'</script><script>window.__VISION_SCENES__='+json(sceneAssets)+';</script><script>'+read('vision-scene.js')+'</script>';
const html=read('shell.html').replace('<!--KATEX-->',()=>read('katex-bundle.html')).replace('<!--SHARED-->',()=>shared).replace('<!--SECTIONS-->',()=>fragments);
const browser=await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{});
const errors=[],issues=[];
try{
  const page=await browser.newPage({viewport:{width:1280,height:720}});
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'||m.type()==='warning')errors.push(m.text());});
  await page.route('http://vision4.test/**',route=>route.fulfill({contentType:'text/html',body:html}));
  await page.goto('http://vision4.test/vision4.html?present#s01');await page.waitForTimeout(300);
  const total=await page.evaluate(()=>AT.present.state().total);
  assert.equal(total,frames,'all authored frames registered');
  assert.equal(await page.locator('.katex-error').count(),0,'all math rendered');
  for(let i=0;i<total;i++){
    await page.evaluate(i=>{AT.present.go(i,null,999);document.querySelectorAll('.frame.is-live details.reveal').forEach(el=>el.open=true);},i);await page.waitForTimeout(40);
    const geometry=await page.evaluate(()=>{const f=document.querySelector('.frame.is-live');return{title:AT.present.state().frame.title,scroll:f.scrollHeight,client:f.clientHeight,wide:f.scrollWidth>f.clientWidth+2};});
    if(geometry.scroll>geometry.client+3||geometry.wide)issues.push({frame:i+1,...geometry});
    await page.screenshot({path:path.join(out,`frame-${String(i+1).padStart(2,'0')}.png`)});
  }
  await page.evaluate(()=>AT.present.exit());
  assert.equal(await page.locator('#s07-probe .voverlay').count(),1,'the scene C overlay is drawn');
  assert.equal(await page.locator('#s06-stepper .stepper').count(),1,'the generation stepper exists');
  await page.setViewportSize({width:390,height:844});await page.goto('http://vision4.test/vision4.html#s01');await page.waitForTimeout(300);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'no mobile document horizontal overflow');
  assert.equal(errors.length,0,errors.join('\n'));
  console.log(JSON.stringify({browser:'checked',frames:total,issues,errors,screenshots:out},null,2));
  if(issues.length)process.exitCode=1;
}finally{await browser.close();}

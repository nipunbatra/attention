// Regression checks for the exact image-conditioned prefix decoder.
// Run: node src/check_vision4.mjs [--browser] [--screenshots /tmp/vision4-check]
// Reproduce the saved NumPy model separately: python3 -B src/train_vision4.py --check
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
const clone=x=>JSON.parse(JSON.stringify(x));
const sum=r=>r.reduce((a,b)=>a+b,0);
const mm=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0)));
const tr=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const add=(a,b)=>a.map((r,i)=>r.map((x,j)=>x+b[i][j]));
const zeros=(r,c)=>Array.from({length:r},()=>Array(c).fill(0));
const sm=r=>{const peak=Math.max(...r),ex=r.map(x=>Math.exp(x-peak)),total=sum(ex);return ex.map(x=>x/total);};
const AT={matmul:mm,transpose:tr,softmax:sm,fmt:(v,n=3)=>Number(v).toFixed(n),
  argmax:r=>r.indexOf(Math.max(...r)),notation:[],axes:{named:true}};
const sandbox={window:{AT,__TOY__:toy}};
vm.runInNewContext(read('part8.js'),sandbox);
const T=AT.vlm;
let comparisons=0,maximumReferenceError=0;
function compare(actual,expected,label='value',tolerance=1e-11){
  if(Array.isArray(expected)){
    assert(Array.isArray(actual),label);assert.equal(actual.length,expected.length,label);
    expected.forEach((v,i)=>compare(actual[i],v,`${label}[${i}]`,tolerance));
  }else if(expected&&typeof expected==='object'){
    for(const[k,v]of Object.entries(expected))compare(actual[k],v,`${label}.${k}`,tolerance);
  }else if(expected===null){
    assert.equal(actual,-Infinity,`${label}: null is reserved for masked negative infinity`);
  }else if(typeof expected==='number'){
    assert(Number.isFinite(actual),label);const error=Math.abs(actual-expected);
    comparisons++;maximumReferenceError=Math.max(maximumReferenceError,error);
    assert(error<=tolerance,`${label}: ${actual} != ${expected}, error ${error}`);
  }else assert.equal(actual,expected,label);
}
function shape(m,rows,columns,label){
  assert.equal(m.length,rows,label);m.forEach(r=>{assert.equal(r.length,columns,label);assert(r.every(Number.isFinite),label);});
}
function normalized(m,label){m.forEach((r,i)=>{assert(r.every(v=>Number.isFinite(v)&&v>=0&&v<=1),label);compare(sum(r),1,`${label} row ${i} sum`);});}

// Vision I's encoder is the frozen visual front end, including CLS while it mixes.
const vision1=JSON.parse(read('toy5.json')).vision;
const imageOutputs={};
for(const name of ['two','one']){
  const image=data.images[name],patches=[];
  for(const y of [0,2])for(const x of [0,2])patches.push([image[y][x],image[y][x+1],image[y+1][x],image[y+1][x+1]]);
  const embedded=mm(patches,vision1.W_patch).map(r=>r.map((v,j)=>v+vision1.b_patch[j]));
  const E=add([vision1.cls].concat(embedded),vision1.positions);
  const Q=mm(E,vision1.W_Q),K=mm(E,vision1.W_K),V=mm(E,vision1.W_V);
  const scores=mm(Q,tr(K)).map(r=>r.map(v=>v/Math.sqrt(2))),A=scores.map(sm);
  const allUpdated=add(E,mm(mm(A,V),vision1.W_O));
  imageOutputs[name]=allUpdated.slice(1);
  compare(T.vision(name),{image,patches,embedded,E,Q,K,V,scores,A,G:imageOutputs[name]},name+' frozen Vision I');
  shape(T.vision(name).G,4,2,'four contextual patch rows');
}
compare(data.frozenVision.W_patch,vision1.W_patch,'same patch projection');
compare(data.frozenVision.positions,vision1.positions.slice(1),'same patch positions');
assert.equal(AT.axes.named,false,'no invented learned coordinate names');

// Independent loss: compute only the three receiving answer rows, with explicit
// allowed source lists. It does not call the production attention or loss code.
function scalarLoss(name,p){
  const answer=data.answers[name],prefix=data.prompt.concat(answer.slice(0,-1));
  const ids=prefix.map(t=>data.vocab.indexOf(t));
  const bridge=mm(imageOutputs[name],p.W_bridge).map(r=>r.map((v,j)=>v+p.b_bridge[j]));
  const rows=add(bridge.concat(ids.map(id=>p.E_tok[id])),p.P.slice(0,4+ids.length));
  const terms=[];
  for(let t=0;t<answer.length;t++){
    const i=6+t,q=mm([rows[i]],p.W_Q)[0],keys=mm(rows.slice(0,i+1),p.W_K),values=mm(rows.slice(0,i+1),p.W_V);
    const scores=keys.map(k=>sum(q.map((v,j)=>v*k[j]))/Math.sqrt(3));
    const weights=sm(scores),message=[0,1,2].map(j=>sum(values.map((v,k)=>weights[k]*v[j])));
    const update=mm([message],p.W_O)[0],row=rows[i].map((v,j)=>v+update[j]);
    const logits=mm([row],p.W_vocab)[0].map((v,j)=>v+p.b_vocab[j]);
    const peak=Math.max(...logits),logZ=peak+Math.log(sum(logits.map(v=>Math.exp(v-peak))));
    terms.push(logZ-logits[data.vocab.indexOf(answer[t])]);
  }
  return sum(terms)/terms.length;
}

// A separate backward calculation supplies a gradient for both images and both
// snapshots; the saved NumPy gradient is also checked below.
function analyticGradient(f,p){
  const dz=zeros(f.logits.length,data.vocab.length),targets=f.targets;
  targets.forEach((target,t)=>{
    const i=6+t;dz[i]=f.probs[i].map(v=>v/targets.length);dz[i][data.vocab.indexOf(target)]-=1/targets.length;
  });
  const g={W_vocab:mm(tr(f.out),dz),b_vocab:tr(dz).map(sum)};
  const dout=mm(dz,tr(p.W_vocab));
  g.W_O=mm(tr(f.message),dout);
  const dm=mm(dout,tr(p.W_O)),da=mm(dm,tr(f.V)),dv=mm(tr(f.A),dm);
  const ds=f.A.map((r,i)=>{const center=sum(r.map((a,j)=>a*da[i][j]));return r.map((a,j)=>a*(da[i][j]-center)/Math.sqrt(3));});
  const dq=mm(ds,f.K),dk=mm(tr(ds),f.Q);
  g.W_Q=mm(tr(f.E),dq);g.W_K=mm(tr(f.E),dk);g.W_V=mm(tr(f.E),dv);
  const de=add(add(dout,mm(dq,tr(p.W_Q))),add(mm(dk,tr(p.W_K)),mm(dv,tr(p.W_V))));
  g.W_bridge=mm(tr(f.vision.G),de.slice(0,4));g.b_bridge=tr(de.slice(0,4)).map(sum);
  g.E_tok=zeros(p.E_tok.length,3);f.ids.forEach((id,i)=>de[i+4].forEach((v,j)=>g.E_tok[id][j]+=v));
  g.P=zeros(p.P.length,3);de.forEach((r,i)=>g.P[i]=r.slice());
  return g;
}
let gradientChecks=0,maximumGradientError=0;
const reports={};
for(const snapshot of ['before','after']){
  const p=T.params(snapshot);reports[snapshot]={};
  compare(T.teacher('two',{snapshot}),data.reference[snapshot],snapshot+' Python reference');
  for(const name of ['two','one']){
    const f=T.teacher(name,{snapshot}),g=analyticGradient(f,p),n=f.E.length;
    shape(f.E,9,3,'training input');shape(f.Q,9,3,'Q');shape(f.K,9,3,'K');shape(f.V,9,3,'V');shape(f.out,9,3,'updated rows');
    normalized(f.A,'attention');normalized(f.probs,'vocabulary');
    f.allowed.forEach((row,i)=>row.forEach((allowed,j)=>{
      assert.equal(allowed,i<4?j<4:j<=i,'image-only prefix / causal text mask');
      if(!allowed){assert.equal(f.scores[i][j],-Infinity);assert.equal(f.A[i][j],0);}
    }));
    compare(f.bridged,mm(f.vision.G,p.W_bridge).map(r=>r.map((v,j)=>v+p.b_bridge[j])),'connector');
    compare(f.E,add(f.bridged.concat(f.ids.map(id=>p.E_tok[id])),p.P.slice(0,n)),'add full-width positions');
    compare(f.Q,mm(f.E,p.W_Q),'query origin');compare(f.K,mm(f.E,p.W_K),'key origin');compare(f.V,mm(f.E,p.W_V),'value origin');
    compare(f.delta,mm(mm(f.A,f.V),p.W_O),'value sum and output projection');compare(f.out,add(f.E,f.delta),'residual addition');
    compare(f.loss,scalarLoss(name,p),'independent answer-only mean loss');
    for(let length=1;length<=f.prefix.length;length++){
      const short=T.forward(name,f.prefix.slice(0,length),{snapshot});
      compare(short.out,f.out.slice(0,4+length),'extending prefix leaves past representations unchanged');
      compare(short.probs,f.probs.slice(0,4+length),'extending prefix leaves past logits unchanged');
    }
    const changedFuture=T.forward(name,['<bos>','count','?','block','one'],{snapshot});
    compare(changedFuture.out.slice(0,7),f.out.slice(0,7),'future-answer mutation cannot leak to first prediction');
    const generation=T.generate(name,{snapshot});
    compare(generation,data.generation[snapshot][name],'actual Python/JS greedy trace');
    generation.trace.forEach((step,i)=>{
      compare(step.prefix,data.prompt.concat(generation.tokens.slice(0,i)),'self-generated prefix');
      assert.equal(step.row,4+step.prefix.length-1,'last known text position');
      assert.equal(step.chosen,data.vocab[step.probs.indexOf(Math.max(...step.probs))],'actual argmax choice');
      const pass=T.forward(name,step.prefix,{snapshot});compare(step.query,pass.Q[step.row],'changing last-row query');
    });
    assert.equal(generation.stoppedBy,'eos');
    if(snapshot==='before'&&name==='two')compare(g,data.update.gradients,'independent backward versus NumPy gradient');
    let localMaximum=0;
    for(const epsilon of [1e-5,1e-6])for(const[key,value]of Object.entries(p)){
      const isMatrix=Array.isArray(value[0]);
      for(let i=0;i<value.length;i++)for(let j=0;j<(isMatrix?value[i].length:1);j++){
        const old=isMatrix?value[i][j]:value[i],set=v=>isMatrix?value[i][j]=v:value[i]=v;
        let high,low;
        try{set(old+epsilon);high=scalarLoss(name,p);set(old-epsilon);low=scalarLoss(name,p);}finally{set(old);}
        const analytic=isMatrix?g[key][i][j]:g[key][i],error=Math.abs((high-low)/(2*epsilon)-analytic);
        gradientChecks++;localMaximum=Math.max(localMaximum,error);maximumGradientError=Math.max(maximumGradientError,error);
        assert(error<1e-6,`${snapshot}/${name}/${key}[${i},${j}] gradient error ${error}`);
      }
    }
    reports[snapshot][name]={loss:f.loss,perToken:f.losses,greedy:generation.tokens,maxGradientError:localMaximum};
  }
}
const before=T.params('before'),after=T.params('after'),rate=data.update.rate;
let parameterCount=0;
for(const[key,values]of Object.entries(before)){
  const isMatrix=Array.isArray(values[0]);
  for(let i=0;i<values.length;i++)for(let j=0;j<(isMatrix?values[i].length:1);j++){
    const get=x=>isMatrix?x[key][i][j]:x[key][i];
    compare(get(after),get(before)-rate*get(data.update.gradients),'exact SGD parameter update');parameterCount++;
  }
}
assert.equal(parameterCount,137,'trainable scalar count');
assert.equal(gradientChecks,1096,'137 scalars × two images × two snapshots × two finite-difference widths');
compare(reports.before.two.loss,data.update.lossBefore,'saved loss before');compare(reports.after.two.loss,data.update.lossAfter,'saved loss after');
compare(reports.before.two.perToken,data.update.perTokenBefore,'saved per-token losses before');compare(reports.after.two.perToken,data.update.perTokenAfter,'saved per-token losses after');
assert(reports.after.two.loss<reports.before.two.loss,'SGD lowers the training example loss');
// This regression is an intentional teaching counterexample, not a quality pass.
assert(reports.after.one.loss>reports.before.one.loss,'the other image gets worse after a one-example update');
compare(reports.before.one.greedy,['one','block','<eos>'],'one-image answer before');
compare(reports.after.one.greedy,['two','blocks','<eos>'],'one-image answer regresses after update');
assert(reports.after.two.perToken[2]>reports.before.two.perToken[2],'even one target loss may rise while the mean improves');
const two=T.forward('two'),one=T.forward('one');
compare(two.Q[6],one.Q[6],'same first query for the same known text in this one-layer decoder');
for(const snapshot of ['before','after'])for(const name of ['two','one'])for(const prefix of [data.prompt,data.prompt.concat(data.answers[name].slice(0,1)),data.prompt.concat(data.answers[name].slice(0,2))]){
  const c=T.contributions(name,prefix,{snapshot}),f=T.forward(name,prefix,{snapshot}),i=f.E.length-1,p=T.params(snapshot);
  compare(c.sources.map(r=>r.weight),f.A[i],'source weights retain all image and text mass');
  for(const [j,r]of c.sources.entries()){
    compare(r.weighted,f.V[j].map(v=>v*f.A[i][j]),'individual weighted value');
    compare(r.update,mm([r.weighted],p.W_O)[0],'individual projected message');
    const logits=mm([r.update],p.W_vocab)[0];compare(r.contrast,logits[4]-logits[3],'source logit difference');
  }
  compare(tr(c.sources.map(r=>r.weighted)).map(sum),f.message[i],'weighted source values sum to message');
  compare(tr(c.sources.map(r=>r.update)).map(sum),f.delta[i],'source updates sum to contextual update');
  compare(c.residual+c.image+c.text,f.logits[i][4]-f.logits[i][3],'residual and source terms reproduce logit contrast');
}
assert(T.contributions('two').image<0&&T.contributions('one').image<T.contributions('two').image,'displayed signed image contributions reflect the actual fitted model');
assert(Math.abs(two.probs[6][4]-one.probs[6][4])>.5,'image changes the first answer distribution');
assert.equal(T.generate('two',{limit:1}).stoppedBy,'limit');assert.equal(T.generate('two',{limit:1}).trace.length,1);
for(const bad of [0,-1,7,1.5])assert.throws(()=>T.generate('two',{limit:bad}));
assert.throws(()=>T.forward('unknown'));assert.throws(()=>T.forward('two',[]));assert.throws(()=>T.forward('two',['unknown']));assert.throws(()=>T.params('missing'));
// The generator cannot be an answer-key animation: altering the answer labels
// must not change the greedy output from the same prompt, pixels and parameters.
const savedAnswers=clone(data.answers);
try{data.answers.two=['one','block','<eos>'];compare(T.generate('two').tokens,['two','blocks','<eos>'],'generation ignores answer key');}finally{data.answers=savedAnswers;}
assert.equal(JSON.stringify(toy),original,'checks did not mutate saved data');
console.log(JSON.stringify({numeric:'pass',referenceComparisons:comparisons,maximumReferenceError,parameterCount,gradientChecks,maximumGradientError,reports},null,2));
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
const sections=fs.readdirSync(path.join(src,'sections8')).filter(n=>/^sec\d\d\.html$/.test(n)).sort().map(n=>read('sections8/'+n)).join('\n');
const sceneAssets=Object.fromEntries([['two','two-mugs.jpg'],['one','one-mug.jpg']].map(([variant,file])=>[variant,'data:image/jpeg;base64,'+fs.readFileSync(path.join(src,'..','figures','vision-scene',file)).toString('base64')]));
const shared='<script>window.__TOY__='+json(toy)+';window.__PART__='+json(config)+';window.__VISION_SCENES__='+json(sceneAssets)+';</script><script>'+read('shared.js')+'</script><script>'+read('part8.js')+'</script><script>'+read('vision-scene.js')+'</script>';
const html=read('shell.html').replace('<!--KATEX-->',()=>read('katex-bundle.html')).replace('<!--SHARED-->',()=>shared).replace('<!--SECTIONS-->',()=>sections);
const browser=await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{});
const errors=[],issues=[];
try{
  const page=await browser.newPage({viewport:{width:1280,height:720}});
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.route('http://vision4.test/**',route=>route.fulfill({contentType:'text/html',body:html}));
  await page.goto('http://vision4.test/vision4.html?present#s01');await page.waitForTimeout(200);
  const frames=await page.evaluate(()=>AT.present.state().total);
  assert.equal(frames,(sections.match(/class="frame"/g)||[]).length,'all authored frames registered');
  assert.equal(await page.locator('.katex-error').count(),0,'all math rendered');
  const decodedScenes=await page.evaluate(async()=>Promise.all(Object.entries(window.__VISION_SCENES__).map(async([variant,src])=>{const image=new Image();image.src=src;await image.decode();return[variant,image.naturalWidth,image.naturalHeight];})));
  compare(decodedScenes.sort(),[['one',1536,1024],['two',1536,1024]],'both offline scene assets decode');
  for(let i=0;i<frames;i++){
    await page.evaluate(i=>{AT.present.go(i,null,999);document.querySelectorAll('.frame.is-live details.reveal').forEach(el=>el.open=true);},i);await page.waitForTimeout(30);
    const geometry=await page.evaluate(()=>{
      const f=document.querySelector('.frame.is-live'),r=f.getBoundingClientRect();
      const svg=[];
      f.querySelectorAll('svg.vlm-diagram').forEach(root=>{
        const v=root.viewBox.baseVal;
        root.querySelectorAll('[marker-end]').forEach(el=>{
          const id=el.getAttribute('marker-end').match(/^url\(#(.+)\)$/)?.[1];
          const target=id&&document.getElementById(id);
          if(!target||target.localName!=='marker')svg.push({tag:el.tagName,marker:id,error:'Arrow reference must resolve to its marker, not a duplicate description ID.'});
        });
        root.querySelectorAll('text,rect,path').forEach(el=>{
          if(el.closest('defs'))return;const b=el.getBBox();
          if(b.width&&b.height&&(b.x<v.x-2||b.y<v.y-2||b.x+b.width>v.x+v.width+2||b.y+b.height>v.y+v.height+2))svg.push({text:el.textContent,tag:el.tagName,bounds:{x:b.x,y:b.y,width:b.width,height:b.height}});
        });
      });
      return{title:AT.present.state().frame.title,scroll:f.scrollHeight,client:f.clientHeight,wide:f.scrollWidth>f.clientWidth+2,svg};
    });
    if(geometry.scroll>geometry.client+3||geometry.wide||geometry.svg.length)issues.push({frame:i+1,...geometry});
    await page.screenshot({path:path.join(out,`frame-${String(i+1).padStart(2,'0')}.png`)});
  }
  await page.evaluate(()=>AT.present.exit());
  assert.equal(await page.locator('#s07-reveal details.reveal').count(),1,'final quiz is actually appended through the reveal API');
  const reveal=page.locator('#s07-reveal details.reveal');
  await reveal.evaluate(el=>el.open=false);await reveal.locator('summary').click();
  assert.equal(await reveal.evaluate(el=>el.open),true,'final reveal opens');
  assert.match(await reveal.innerText(),/projected image rows/,'final answer text is present');
  assert.equal(await page.locator('#s05-pick button').count(),2,'two image-choice controls');
  assert.equal(await page.locator('#s05-step0 svg rect[data-pixel-value]').count(),16,'the image remains visible at the first generation step');
  assert.equal(await page.locator('#s05-step1 svg rect[data-pixel-value]').count(),16,'the image remains visible after appending a token');
  assert.equal(await page.locator('#s05-step2 svg rect[data-pixel-value]').count(),16,'the image remains visible until the end marker');
  await page.locator('#s05-pick button').filter({hasText:'one block'}).click();
  assert.match(await page.locator('#s05-compare').innerText(),/Actual greedy output: one block <eos>/,'one-image control recomputes output');
  await page.locator('#s05-pick button').filter({hasText:'two blocks'}).click();
  assert.match(await page.locator('#s05-compare').innerText(),/Actual greedy output: two blocks <eos>/,'two-image control restores output');
  await page.setViewportSize({width:390,height:844});await page.goto('http://vision4.test/vision4.html#s01');
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'no mobile document horizontal overflow');
  assert.equal(errors.length,0,errors.join('\n'));
  console.log(JSON.stringify({browser:'checked',frames,issues,errors,screenshots:out},null,2));
  if(issues.length)process.exitCode=1;
}finally{await browser.close();}

// Exact arithmetic and optional browser checks for the offline Vision I lesson (toy5.json v2: is there a mug on the right half?).
// node src/check_vision1.mjs [--browser] [--screenshots /tmp/vision1-check]
// Builds a temporary in-memory page in browser mode. Does not assemble or edit published HTML.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const src=path.dirname(fileURLToPath(import.meta.url));
const read=name=>fs.readFileSync(path.join(src,name),'utf8');
const data=JSON.parse(read('toy5.json'));
const config=JSON.parse(read('part5.json'));
// A minimal AT so vision-shared.js and part5.js load without a DOM; the figure builders are never called here.
const stub=()=>({appendChild(){},setAttribute(){},addEventListener(){},querySelector(){return null;},replaceChild(){}});
const fmt=(x,d=2)=>{const s=Number(x).toFixed(d);return s.charAt(0)==='-'?'−'+s.slice(1):s;};
const context={window:{__TOY__:data,AT:{h:stub,svg:stub,fmt,fmtSigned:(x,d)=>(x>=0?'+':'')+fmt(x,d),notation:[]}},document:{head:{appendChild(){}}}};
vm.createContext(context);
vm.runInContext(read('vision-shared.js'),context);
vm.runInContext(read('part5.js'),context);
const V=context.window.AT.vision;
const plain=x=>JSON.parse(JSON.stringify(x));
const near=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const round=(x,d)=>Math.round(x*10**d)/10**d;
const sum=a=>a.reduce((s,x)=>s+x,0);

// ---- the toy and the runtime agree with each other and with the stored rows
assert.deepEqual(plain(V.axes.e),data.axes.e);assert.deepEqual(plain(V.axes.qk),data.axes.qk);assert.deepEqual(plain(V.axes.v),data.axes.v);
assert.equal(data.d_model,4);assert.equal(data.d_k,2);assert.equal(data.d_v,2);
assert.deepEqual(data.classes,['mug on the right','no mug on the right']);
for(const k of ['A','B','C']){
  assert.deepEqual(plain(V.scenes[k]),data.scenes[k].pixels,'scene '+k+' pixels');
  assert.deepEqual(plain(V.regions(k)),data.scenes[k].regions,'scene '+k+' region names');
  assert.equal(V.labelOf(k),data.scenes[k].label);
  for(const which of ['initial','trained']){
    const f=V.forward(k,which);
    assert.equal(f.A.length,17);f.A.forEach(r=>{assert.equal(r.length,17);near(sum(r),1);r.forEach(a=>assert.ok(a>0&&a<1));});
    assert.deepEqual(plain(f.A[0]).map(x=>round(x,3)),data.scenes[k]['cls_attention_'+which],`${k} ${which} CLS attention row`);
    assert.deepEqual(plain(f.probs).map(x=>round(x,3)),data.scenes[k]['probs_'+which],`${k} ${which} probabilities`);
    assert.deepEqual(plain(V.attention(k,which)).map(x=>round(x,3)),data.scenes[k]['cls_attention_'+which]);
  }
}
// ---- independent scalar recomputation of one forward pass (scene A, trained), not the runtime's helpers
{
  const P=data.trained,R=V.patchify('A');
  const E=[[1,0,-1,-1]].concat(R.map((r,j)=>[sum(r)/4,(r[0]-r[1]+r[2]-r[3])/2,Math.floor(j/4)/3,(j%4)/3]));
  const mv=(x,W)=>W[0].map((_,c)=>x.reduce((s,v,i)=>s+v*W[i][c],0));
  const q=mv(E[0],P.W_Q),K=E.map(e=>mv(e,P.W_K)),Vv=E.map(e=>mv(e,P.W_V));
  const s=K.map(k=>(q[0]*k[0]+q[1]*k[1])/Math.SQRT2),z=sum(s.map(Math.exp)),a=s.map(x=>Math.exp(x)/z);
  const m=[0,1].map(c=>sum(a.map((w,j)=>w*Vv[j][c]))),d=mv(m,P.W_O),ep=E[0].map((x,i)=>x+d[i]);
  const l=mv(ep,P.W_cls).map((x,c)=>x+P.b_cls[c]),pz=sum(l.map(Math.exp)),p=l.map(x=>Math.exp(x)/pz);
  const f=V.forward('A','trained');
  E.forEach((row,i)=>row.forEach((x,c)=>near(f.E[i][c],x)));
  a.forEach((x,j)=>near(f.A[0][j],x));m.forEach((x,c)=>near(f.H[0][c],x));ep.forEach((x,c)=>near(f.Enew[0][c],x));
  l.forEach((x,c)=>near(f.logits[c],x));p.forEach((x,c)=>near(f.probs[c],x));
  near(V.loss('A','trained'),-Math.log(p[0]));
}
// ---- the targets of VISION_AXES.md, adjusted to the mug-on-the-right question
{
  const a0=V.attention('A','initial'),a1=V.attention('A','trained');
  const top0=V.topSources(a0,'A',1)[0];assert.match(top0.label,/mug/,'T1: untrained CLS reads a mug patch first');
  assert.ok(V.regionMass(a1,'A','mug')>=0.6,'T2: trained CLS puts at least 0.6 on the six mug patches of A: '+V.regionMass(a1,'A','mug'));
  assert.ok(V.regionMass(a1,'A','right mug')>V.regionMass(a0,'A','right mug'),'training moves mass to the right mug');
  for(const k of ['A','B']){const pr=V.predict(k,'trained');assert.equal(pr.label,V.labelOf(k));assert.ok(pr.probs[V.labelIndex(k)]>=0.9,'T3: '+k);}
  const pr0=V.predict('B','initial');assert.notEqual(pr0.label,V.labelOf('B'),'the untrained model is wrong on scene B; that is the lesson');
  const probe=V.predict('C','trained'),pC=probe.probs[V.labelIndex('C')];
  assert.equal(round(pC,3),data.scenes.C.probs_trained[V.labelIndex('C')],'T4: the probe is reported from the stored numbers');
  assert.ok(V.meanLoss('trained')<V.meanLoss('initial'));assert.ok(V.meanLoss('trained')<0.01);
  assert.ok(data.curve.length>=10&&data.curve[0][0]===1&&data.curve[data.curve.length-1][0]===1500);
  assert.ok(data.curve[0][1]<V.meanLoss('initial')&&data.curve[data.curve.length-1][1]<0.03);
  for(const which of ['initial','trained'])for(const M of ['W_Q','W_K','W_V','W_O','W_cls','b_cls'])data[which][M].flat().forEach(x=>assert.equal(x,round(x,2),'T5: two decimals in '+M));
}
// ---- permutation and value-only checks used by the page
{
  const sw=V.swapPatches('B',5,7);
  assert.equal(V.regions(sw)[7],'right mug centre');
  const withB=V.forward('B','initial').Enew[0],withS=V.forward(sw,'initial').Enew[0];
  const noB=V.forwardOpts('B','initial',{positions:false}).Enew[0],noS=V.forwardOpts(sw,'initial',{positions:false}).Enew[0];
  assert.ok(withB.some((x,c)=>Math.abs(x-withS[c])>1e-6),'positions make the swap visible');
  noB.forEach((x,c)=>near(noS[c],x));
  const orig=V.forward('A','initial'),alt=V.attendWith('A','initial',{W_V:data.initial.W_V.map(r=>r.map((x,c)=>c===0?0:x))});
  orig.A[0].forEach((x,j)=>near(alt.A[0][j],x));near(alt.H[0][0],0);near(alt.H[0][1],orig.H[0][1]);
  assert.deepEqual(plain(V.tokensPerImage(224,16)),{patches:196,rows:197,scores:38809});assert.equal(V.tokensPerImage(224,1).scores,2517731329);
}
// ---- fragments and config
const sections=fs.readdirSync(path.join(src,'sections5')).filter(n=>/^sec\d\d\.html$/.test(n)).sort();
const fragments=sections.map(n=>read('sections5/'+n)).join('\n');
const frames=(fragments.match(/class="frame"/g)||[]).length;
assert.equal(sections.length,11);assert.ok(frames>=60,'frames: '+frames);
assert.equal((fragments.match(/type="text\/x-notes"/g)||[]).length,frames,'every frame has presenter notes');
assert.deepEqual(config.sections.map(s=>s.id),sections.map(n=>'s'+n.slice(3,5)));
assert.equal(config.notation,'vision1');assert.match(config.provenance,/\{\{axes\}\}/);
assert.doesNotMatch(fragments,/coordinate \d/i,'no anonymous coordinates');
assert.doesNotMatch(fragments,/[—–]/,'no dashes in prose or string literals');
assert.equal((fragments.match(/AT\.visionScene\.mount/g)||[]).length,3,'the photo hook appears three times, in the scene section only');
assert.match(read('sections5/sec02.html'),/AT\.visionScene\.mount/);assert.doesNotMatch(read('sections5/sec08.html'),/visionScene/);
assert.equal(fs.existsSync(path.join(src,'part5-learning.js')),false);assert.equal(fs.existsSync(path.join(src,'part5-diagrams.js')),false);
assert.ok(context.window.AT.notation.filter(n=>n.parts&&n.parts.includes('vision1')).length>=12,'notation rows for the card');
const untrained=V.forward('A','initial'),trained=V.forward('A','trained');
console.log(JSON.stringify({arithmetic:'pass',sections:sections.length,frames,clsAttentionTop:{initial:V.topSources(untrained.A[0],'A',3),trained:V.topSources(trained.A[0],'A',3)},probs:{A:plain([V.forward('A','initial').probs,trained.probs]),B:plain([V.forward('B','initial').probs,V.forward('B','trained').probs]),C:plain([V.forward('C','initial').probs,V.forward('C','trained').probs])},meanLoss:{initial:V.meanLoss('initial'),trained:V.meanLoss('trained')},mugMassTrainedA:V.regionMass(trained.A[0],'A','mug')}));
if(!process.argv.includes('--browser'))process.exit(0);

// ---- browser mode: assemble in memory, walk every frame in presentation, exercise the controls
const require=createRequire(import.meta.url),candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean),cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const d of fs.readdirSync(cache).sort())candidates.push(path.join(cache,d,'node_modules','playwright'));
let pw;for(const c of candidates){try{pw=require(c);break;}catch{}}
if(!pw)throw new Error('No installed Playwright runtime; no dependencies were installed.');
const shotArg=process.argv.indexOf('--screenshots'),out=shotArg>=0?process.argv[shotArg+1]:fs.mkdtempSync(path.join(os.tmpdir(),'vision1-check-'));fs.mkdirSync(out,{recursive:true});
const json=x=>JSON.stringify(x).replace(/<\//g,'<\\/');
const cfg=JSON.parse(JSON.stringify(config));cfg.prev.available=false;cfg.next.available=false;
const sceneImages=Object.fromEntries([['two','two-mugs'],['one','one-mug']].map(([key,name])=>[key,'data:image/jpeg;base64,'+fs.readFileSync(path.join(src,'../figures/vision-scene',name+'.jpg')).toString('base64')]));
const shared='<script>window.__TOY__='+json(data)+';window.__PART__='+json(cfg)+';</script>'+['shared.js','vision-shared.js','part5.js'].map(name=>'<script>'+read(name)+'</script>').join('')+'<script>window.__VISION_SCENES__='+json(sceneImages)+';</script><script>'+read('vision-scene.js')+'</script>';
const html=read('shell.html').replace('<!--KATEX-->',()=>read('katex-bundle.html')).replace('<!--SHARED-->',()=>shared).replace('<!--SECTIONS-->',()=>fragments);
const browser=await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{});
const errors=[],issues=[];
try{
 const page=await browser.newPage({viewport:{width:1280,height:720}});
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'||m.type()==='warning')errors.push(m.text());});
 await page.route('http://vision1.test/**',route=>route.fulfill({contentType:'text/html',body:html}));
 await page.goto('http://vision1.test/vision1.html?present#s01');await page.waitForTimeout(300);
 const result=await page.evaluate(()=>({frames:AT.present.state().total,mathErrors:document.querySelectorAll('.katex-error').length,overlays:document.querySelectorAll('.voverlay').length,scatters:document.querySelectorAll('.vscatter').length,curves:document.querySelectorAll('.vcurve').length,thumbLabels:document.querySelectorAll('.dt tbody th .vthumb').length,notation:document.querySelector('#notation-card').textContent,provenance:document.querySelector('#hero-provenance').textContent}));
 assert.equal(result.frames,frames);assert.equal(result.mathErrors,0);assert.ok(result.overlays>=10,'attention overlays');assert.ok(result.scatters>=2);assert.ok(result.curves>=1);assert.ok(result.thumbLabels>=40,'thumbnail row labels');
 assert.ok(result.notation.includes('patch'));assert.ok(result.provenance.includes('brightness'),'the provenance names the axes from the toy');
 const scenes=await page.evaluate(()=>({count:document.querySelectorAll('[data-vision-scene]').length,modes:Array.from(document.querySelectorAll('#s02 [data-vision-scene]')).map(el=>el.dataset.visionScene),embedded:Array.from(document.querySelectorAll('[data-vision-scene] image')).every(el=>el.getAttribute('href').startsWith('data:image/jpeg;base64,'))}));
 assert.equal(scenes.count,3);assert.deepEqual(scenes.modes,['scene','patches','patch-crop']);assert.ok(scenes.embedded,'scene images are embedded for offline use');
 let visited=0;
 for(let fi=0;fi<frames;fi++){
   await page.evaluate(fi=>{AT.present.go(fi,null,999);document.querySelectorAll('.frame.is-live details.reveal').forEach(el=>el.open=true);},fi);await page.waitForTimeout(60);
   const state=await page.evaluate(()=>{const s=AT.present.state(),f=document.querySelector('.frame.is-live');return{fi:s.fi,id:s.frame.id,title:s.frame.title,scroll:f.scrollHeight,client:f.clientHeight,wide:f.scrollWidth>f.clientWidth+2};});
   if(state.scroll>state.client+3||state.wide)issues.push(state);
   await page.screenshot({path:path.join(out,`frame-${String(fi+1).padStart(2,'0')}.png`)});
   visited++;
 }
 // Interaction checks run in reading mode so every control is available.
 await page.evaluate(()=>AT.present.exit());
 await page.locator('#s09-pick .vpick-b').nth(8).click();
 assert.match(await page.locator('#s09-read').innerText(),/patch 8/);
 await page.locator('#s09-scenes button').nth(2).click();
 assert.match(await page.locator('#s09-ov').innerText(),/Scene C/);
 await page.locator('#s11-scenes button').nth(2).click();
 assert.match(await page.locator('#s11-close').innerText(),/Scene C \(never trained on\)/);
 await page.locator('#s06-mute').click();
 assert.match(await page.locator('#s06-altread').innerText(),/Every weight is unchanged/);
 await page.locator('#s03-pick .vpick-b').nth(0).click();
 assert.match(await page.locator('#s03-cread').innerText(),/patch 1/);
 await page.setViewportSize({width:390,height:844});await page.goto('http://vision1.test/vision1.html#s05');await page.waitForTimeout(300);
 const mobile=await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2);assert.ok(mobile,'no mobile document horizontal overflow');
 assert.equal(errors.length,0,errors.join('\n'));
 console.log(JSON.stringify({browser:'checked',visited,issues,errors,screenshots:out}));
 if(issues.length)process.exitCode=1;
}finally{await browser.close();}

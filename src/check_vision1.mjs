// Exact arithmetic and optional browser checks for the offline Vision I lesson.
// node src/check_vision1.mjs [--browser] [--screenshots /tmp/vision1-check]
// Builds a temporary in-memory page. Does not assemble or edit published HTML.
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
const context={window:{__TOY__:data,AT:{axes:{},objects:[],notation:[]}}};
vm.createContext(context);vm.runInContext(read('part5.js'),context);
vm.runInContext(read('part5-learning.js'),context);
const T=context.window.AT.vision;
const plain=x=>JSON.parse(JSON.stringify(x));
const near=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const matrixNear=(a,b)=>{assert.equal(a.length,b.length);a.forEach((r,i)=>{assert.equal(r.length,b[i].length);r.forEach((v,j)=>near(v,b[i][j]));});};
const f=T.forward();
assert.deepEqual(plain(f.patches),[[1,1,1,1],[0,0,0,0],[0,0,0,0],[2,2,2,2]]);
assert.deepEqual(plain(f.embeddings),[[2,2],[0,0],[0,0],[4,4]]);
assert.deepEqual(plain(f.E),[[1,1],[2,2],[0,1],[1,0],[5,5]]);
assert.deepEqual(plain(f.Q),[[.5,.5],[1,1],[0,.5],[.5,0],[2.5,2.5]]);
matrixNear(f.Q,f.K);matrixNear(f.V,f.E);matrixNear(f.message,f.delta);
// Independent scalar CLS calculation, not the runtime's matrix helpers.
const scores=[.5,1,.25,.25,2.5].map(x=>x/Math.sqrt(2));
const z=scores.reduce((sum,x)=>sum+Math.exp(x),0),weights=scores.map(x=>Math.exp(x)/z);
scores.forEach((s,i)=>near(f.S[0][i],s));weights.forEach((a,i)=>near(f.A[0][i],a));
const m=weights[0]+2*weights[1]+weights[3]+5*weights[4];
near(f.message[0][0],m);near(f.message[0][1],m);near(f.updated[0][0],m+1);
const za=(m+1)*.5,zb=(m+1)*.25,pa=1/(1+Math.exp(zb-za));
near(f.logits[0],za);near(f.logits[1],zb);near(f.probs[0],pa);near(f.loss,-Math.log(pa));
assert.equal(f.A.length,5);f.A.forEach(r=>{assert.equal(r.length,5);near(r.reduce((a,b)=>a+b,0),1);r.forEach(a=>assert.ok(a>0&&a<1));});
for(const P of [1,2,4]){const p=T.patchify(T.data.image,P);assert.equal(p.count,16/(P*P));assert.equal(p.width,P*P);assert.equal(p.patches.flat().reduce((a,b)=>a+b,0),12);}
assert.throws(()=>T.patchify(T.data.image,3));assert.throws(()=>T.forward({image:[[1]]}));assert.throws(()=>T.forward({order:[0,0,2,3]}));
// With positions disabled, reordering patch rows permutes outputs and leaves CLS fixed.
const perm=[3,1,2,0], a=T.forward({positions:false}), b=T.forward({positions:false,order:perm});
matrixNear(b.updated,[a.updated[0],...perm.map(i=>a.updated[i+1])]);
matrixNear(b.A,[0,...perm.map(i=>i+1)].map(i=>[0,...perm.map(j=>j+1)].map(j=>a.A[i][j])));
assert.ok(Math.abs(T.forward({order:perm}).updated[0][0]-f.updated[0][0])>.01);
assert.deepEqual(plain(a.updated[0].map(x=>x.toFixed(3))),['3.293','3.293']);
assert.deepEqual(plain(T.forward({order:perm}).updated[0].map(x=>x.toFixed(3))),['3.565','3.565']);
const modified=T.forward({image:T.editImage(3)});
assert.equal(modified.patches[3][0],3);assert.deepEqual(plain(modified.embeddings[3]),[5,4]);
near(modified.Q[0][0],f.Q[0][0]);assert.notEqual(modified.K[4][0],f.K[4][0]);assert.notEqual(modified.probs[0],f.probs[0]);
assert.deepEqual(plain(T.forward().image),plain(data.vision.image),'interaction did not mutate baseline');
const sections=fs.readdirSync(path.join(src,'sections5')).filter(n=>/^sec\d\d\.html$/.test(n)).sort();
const fragments=sections.map(n=>read('sections5/'+n)).join('\n');
const frames=(fragments.match(/class="frame"/g)||[]).length;
assert.equal(sections.length,8);assert.equal(frames,60);
assert.match(fragments,/s05-position-check/);assert.match(fragments,/s07-held-out/);
assert.match(fragments,/s01-scene-crop/);assert.match(fragments,/separate 4 × 4 image/);
assert.match(fragments,/grid:14,square:true/);
assert.equal((fragments.match(/type="text\/x-notes"/g)||[]).length,frames,'every frame has presenter notes');
// Counterfactual values: matching is unchanged, transmitted information differs.
const valueOnly=T.attention(f.E,{W_V:[[1,0],[0,2]]});
matrixNear(valueOnly.A,f.A);matrixNear(valueOnly.K,f.K);
near(valueOnly.message[0][0],f.message[0][0]);
near(valueOnly.message[0][1],2*f.message[0][1]);
const learn=T.learning,experiment=learn.experiment;
matrixNear(learn.forward(learn.initial,learn.images[0],0).updated,f.updated);
assert.ok(experiment.afterSingle.meanLoss<experiment.before.meanLoss);
assert.deepEqual(plain(experiment.afterTraining.predictions),[0,1]);
assert.ok(experiment.afterTraining.meanLoss<.01);
// Same single block, same brightness and count, different position.
const moved=Array.from({length:4},()=>Array(4).fill(0));
for(let y=0;y<2;y++)for(let x=0;x<2;x++)moved[y][x]=learn.images[1][y+2][x+2];
const heldOut=learn.forward(experiment.afterTraining.params,moved,1);
assert.equal(heldOut.probs[0].toFixed(4),'0.7915');
assert.equal(experiment.afterTraining.forwards[1].probs[1].toFixed(4),'0.9953');
assert.equal(moved.flat().reduce((a,b)=>a+b,0),learn.images[1].flat().reduce((a,b)=>a+b,0));
assert.equal(data.axes.named,false);
console.log(JSON.stringify({arithmetic:'pass',sections:sections.length,frames,CLS:plain(f.updated[0]),probs:plain(f.probs),loss:f.loss}));
if(!process.argv.includes('--browser'))process.exit(0);
const require=createRequire(import.meta.url),candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean),cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const d of fs.readdirSync(cache).sort())candidates.push(path.join(cache,d,'node_modules','playwright'));
let pw;for(const c of candidates){try{pw=require(c);break;}catch{}}
if(!pw)throw new Error('No installed Playwright runtime; no dependencies were installed.');
const shotArg=process.argv.indexOf('--screenshots'),out=shotArg>=0?process.argv[shotArg+1]:fs.mkdtempSync(path.join(os.tmpdir(),'vision1-check-'));fs.mkdirSync(out,{recursive:true});
const json=x=>JSON.stringify(x).replace(/<\//g,'<\\/');
const config=JSON.parse(read('part5.json'));config.prev.available=false;config.next.available=false;
const sceneImages=Object.fromEntries([['two','two-mugs'],['one','one-mug']].map(([key,name])=>[key,'data:image/jpeg;base64,'+fs.readFileSync(path.join(src,'../figures/vision-scene',name+'.jpg')).toString('base64')]));
const shared='<script>window.__TOY__='+json(data)+';window.__PART__='+json(config)+';window.__VISION_SCENES__='+json(sceneImages)+';</script>'+['shared.js','vision-scene.js','part5.js','part5-learning.js','part5-diagrams.js'].map(name=>'<script>'+read(name)+'</script>').join('');
const html=read('shell.html').replace('<!--KATEX-->',()=>read('katex-bundle.html')).replace('<!--SHARED-->',()=>shared).replace('<!--SECTIONS-->',()=>fragments);
const browser=await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{});
const errors=[],issues=[];
try{
 const page=await browser.newPage({viewport:{width:1280,height:720}});
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.route('http://vision1.test/**',route=>route.fulfill({contentType:'text/html',body:html}));
 await page.goto('http://vision1.test/vision1.html?present#s01');await page.waitForTimeout(250);
 const result=await page.evaluate(()=>({frames:AT.present.state().total,mathErrors:document.querySelectorAll('.katex-error').length,svgCount:document.querySelectorAll('[data-vision1-diagram],[data-vision-story]').length,notation:document.querySelector('#notation-card').textContent}));
 assert.equal(result.frames,frames);assert.equal(result.mathErrors,0);assert.ok(result.svgCount>=10);assert.ok(result.notation.includes('patch'));
 const scenes=await page.evaluate(()=>({count:document.querySelectorAll('[data-vision-scene]').length,modes:Array.from(document.querySelectorAll('#s01 [data-vision-scene]')).map(el=>el.dataset.visionScene),embedded:Array.from(document.querySelectorAll('[data-vision-scene] image')).every(el=>el.getAttribute('href').startsWith('data:image/jpeg;base64,'))}));
 assert.equal(scenes.count,5);assert.deepEqual(scenes.modes,['scene','patches','patch-crop']);assert.ok(scenes.embedded,'scene images are embedded for offline use');
 assert.ok(await page.evaluate(()=>Array.from(document.querySelectorAll('[data-vision-scene] svg svg')).every(el=>getComputedStyle(el).overflow==='hidden')),'each scene crop clips to its own SVG viewport');
 let visited=0;
 for(let fi=0;fi<frames;fi++){
   await page.evaluate(fi=>{AT.present.go(fi,null,999);document.querySelectorAll('.frame.is-live details.reveal').forEach(el=>el.open=true);},fi);await page.waitForTimeout(50);
   const state=await page.evaluate(()=>{const s=AT.present.state(),f=document.querySelector('.frame.is-live');return{fi:s.fi,id:s.frame.id,title:s.frame.title,scroll:f.scrollHeight,client:f.clientHeight,wide:f.scrollWidth>f.clientWidth+2};});
   if(state.scroll>state.client+3||state.wide)issues.push(state);
   const svgIssues=await page.evaluate(()=>Array.from(document.querySelectorAll('.frame.is-live svg[data-vision1-diagram],.frame.is-live svg[data-vision-story]')).flatMap(svg=>{const v=svg.viewBox.baseVal;return Array.from(svg.querySelectorAll('text,rect,path')).filter(el=>!el.closest('defs')).flatMap(el=>{const b=el.getBBox();return b.width&&b.height&&(b.x<v.x-2||b.y<v.y-2||b.x+b.width>v.x+v.width+2||b.y+b.height>v.y+v.height+2)?[{diagram:svg.dataset.vision1Diagram||svg.dataset.visionStory,element:el.tagName,text:el.textContent,bounds:{x:b.x,y:b.y,width:b.width,height:b.height}}]:[];});}));
   if(svgIssues.length)issues.push({frame:fi+1,svg:svgIssues});
   await page.screenshot({path:path.join(out,`frame-${String(fi+1).padStart(2,'0')}.png`)});
   visited++;
 }
 // Interaction checks run in reading mode so every control is available.
 await page.evaluate(()=>AT.present.exit());
 await page.locator('#s02-explore select').selectOption('1');
 assert.match(await page.locator('#s02-explore').innerText(),/16 patch tokens \+ 1 CLS = 17 rows; 289/);
 await page.locator('#s05-explore select').selectOption('4');
 assert.equal(await page.locator('#s05-explore svg').getAttribute('data-receiver'),'4');
 await page.setViewportSize({width:390,height:844});await page.goto('http://vision1.test/vision1.html#s02');
 const mobile=await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2);assert.ok(mobile,'no mobile document horizontal overflow');
 assert.equal(errors.length,0,errors.join('\n'));
 console.log(JSON.stringify({browser:'checked',visited,issues,errors,screenshots:out}));
 if(issues.length)process.exitCode=1;
}finally{await browser.close();}

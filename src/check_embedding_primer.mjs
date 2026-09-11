// Verify the embedding detour without fetching models or changing the user's tab.
// node src/check_embedding_primer.mjs [part1.html]
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const require=createRequire(import.meta.url);
const candidates=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core'].filter(Boolean);
const cache=path.join(os.homedir(),'.npm','_npx');
if(fs.existsSync(cache))for(const d of fs.readdirSync(cache))candidates.push(path.join(cache,d,'node_modules/playwright'));
let pw;for(const candidate of candidates){try{pw=require(candidate);break;}catch{}}
if(!pw)throw Error('Use an existing Playwright installation.');
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'embedding-primer-'));
const browser=await pw.chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:720}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.route(/^https?:\/\//,route=>route.abort());
await page.goto(pathToFileURL(path.resolve(process.argv[2]||'part1.html')).href);
await page.evaluate(()=>document.fonts.ready);
const inspected=await page.evaluate(()=>{
  const issues=[], charts=[...document.querySelectorAll('[data-embedding-diagram]')];
  const original=JSON.stringify(window.__TOY__);
  const q=[...document.querySelectorAll('#s04 .frame')];
  const table=q.findIndex(f=>f.dataset.title==='Look up a learned row');
  const point=q.findIndex(f=>f.dataset.title==='Two numbers locate one character');
  if(!(point>=0&&point<table))issues.push('Point motivation must precede the lookup table');
  for(const svg of charts){
    const vb=svg.viewBox.baseVal;
    if(svg.getAttribute('role')!=='img'||!svg.querySelector('title')||!svg.querySelector('desc'))issues.push('Missing accessible diagram labels');
    for(const n of svg.querySelectorAll('text')){
      const b=n.getBBox(),m=svg.getScreenCTM().inverse().multiply(n.getScreenCTM());
      const corners=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));
      if(corners.some(p=>p.x<-.5||p.y<-.5||p.x>vb.width+.5||p.y>vb.height+.5))issues.push(svg.dataset.embeddingDiagram+': label outside diagram: '+n.textContent);
    }
    for(const n of svg.querySelectorAll('[data-coordinates]')){
      const row=window.__TOY__.E[window.__TOY__.vocab.indexOf(n.dataset.token)];
      if(JSON.stringify(row)!==n.dataset.coordinates)issues.push('Plot changed the saved coordinates');
    }
  }
  if(charts.length!==11)issues.push('Expected 11 new primer diagrams');
  const svgImage=document.querySelector('#s04-image-flow image');
  if(!svgImage?.getAttribute('href').startsWith('data:image/jpeg;base64,'))issues.push('Scene is not embedded for offline use');
  const vectors=Object.fromEntries([...document.querySelectorAll('#s04-word-offsets [data-word]')].map(n=>[n.dataset.word,JSON.parse(n.dataset.vector)]));
  if(Object.keys(vectors).length!==4||vectors.king.some((v,j)=>v-vectors.man[j]+vectors.woman[j]!==vectors.queen[j]))issues.push('Word analogy coordinates disagree');
  const methods=AT.embeddingPrimer;
  Object.values(methods).forEach(draw=>draw());
  if(JSON.stringify(window.__TOY__)!==original)issues.push('Diagram construction mutated trained model');
  return {diagrams:charts.length,vectors,issues};
});
errors.push(...inspected.issues);
const frames=await page.evaluate(()=>[...document.querySelectorAll('#s04 .frame')].map((f,i)=>({index:i+1,title:f.dataset.title,primer:f.hasAttribute('data-embedding-primer')})).filter(f=>f.primer||f.title==='Look up a learned row'));
for(const f of frames){
  await page.evaluate(i=>{AT.present.enter();AT.present.go('s04',i,99);},f.index);
  await page.waitForTimeout(65);
  const fit=await page.evaluate(()=>AT.present.fitReport());
  if(fit.overflow)errors.push('Frame overflow: '+f.title);
  const undersized=await page.evaluate(i=>[...document.querySelectorAll('#s04 .frame')[i-1].querySelectorAll('[data-embedding-diagram]')]
    .some(svg=>svg.getBoundingClientRect().width<800),f.index);
  if(undersized)errors.push('Diagram too small for classroom viewing: '+f.title);
  await page.screenshot({path:path.join(dir,String(f.index).padStart(2,'0')+'.png')});
}
await page.evaluate(()=>AT.present.exit());await page.setViewportSize({width:390,height:844});
const mobile=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,
  stacked:getComputedStyle(document.querySelector('#s04 .primer-pair')).gridTemplateColumns.split(' ').length===1}));
if(mobile.overflow||!mobile.stacked)errors.push('Mobile reading layout does not fit');
await browser.close();
console.log(JSON.stringify({inspected,frames:frames.length,mobile,screenshots:dir,errors},null,2));
if(errors.length)process.exitCode=1;

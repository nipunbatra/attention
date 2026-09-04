// Exact illustrative arithmetic and authoring checks. No trained model is claimed.
// node src/check_vision2.mjs [--browser] [--page /absolute/path/vision2.html]
import assert from 'node:assert/strict';
import {readFileSync,readdirSync,existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';

const toy=JSON.parse(readFileSync(new URL('toy6.json',import.meta.url),'utf8'));
const runtime=readFileSync(new URL('part6.js',import.meta.url),'utf8');
const config=JSON.parse(readFileSync(new URL('part6.json',import.meta.url),'utf8'));
const pristine=JSON.stringify(toy);
class Element{
  constructor(tag){this.tagName=tag;this.attributes={};this.children=[];this.style={};this._text='';}
  setAttribute(k,v){this.attributes[k]=String(v);}getAttribute(k){return this.attributes[k];}
  set textContent(t){this._text=String(t);}get textContent(){return this._text+this.children.map(c=>c.textContent).join(' ');}
  appendChild(c){this.children.push(c);return c;}get firstChild(){return this.children[0];}
  removeChild(c){this.children.splice(this.children.indexOf(c),1);}
}
const document={createElementNS:(_,tag)=>new Element(tag)};
const window={__TOY__:toy,AT:{axes:{named:true},objects:[],notation:[],clear:e=>{e.children=[];}}};
// Load the actual shared image convention into this small DOM test harness.
const shadeSource=readFileSync(new URL('shared.js',import.meta.url),'utf8').match(/AT\.imageShade = function \(v\) \{[^\n]+\};/)[0];
vm.runInNewContext(shadeSource,{AT:window.AT});
vm.runInNewContext(runtime,{window,document,console});
const T=window.AT.visionSSL,clone=x=>JSON.parse(JSON.stringify(x));
const near=(a,b,label)=>assert(Math.abs(a-b)<1e-12,`${label}: ${a} != ${b}`);
const sum=a=>a.reduce((s,x)=>s+x,0);
assert.deepEqual(clone(T.patchify()),[[1,1,1,1],[0,0,0,0],[0,0,0,0],[2,2,2,2]]);
let comparisons=0;
for(const preset of toy.visionSSL.maskPresets)for(let guess=0;guess<=2;guess+=.25){
  const R=T.reconstruction(preset.indices,guess),pixels=preset.indices.flatMap(i=>T.patchify()[i]);
  const expected=sum(pixels.map(x=>(guess-x)**2))/pixels.length;
  near(R.loss,expected,'masked pixel MSE');assert.equal(R.count,pixels.length);assert.equal(R.visible,4-preset.indices.length);
  assert(R.rows.filter(r=>!r.hidden).every(r=>r.sse===0));comparisons++;
}
near(T.reconstruction([1,2,3],1).loss,1,'default');near(T.reconstruction([3],2).loss,0,'matching patch 4');near(T.reconstruction([0,3],1).loss,.5,'changed mask');
for(const bad of [[],[0,1,2,3],[1,1],[-1],[4],[.5],null])assert.throws(()=>T.reconstruction(bad));
assert.throws(()=>T.reconstruction([1],NaN));assert.throws(()=>T.patchify([[1]]));
const expectedTarget=[Math.exp(3),Math.exp(1),1].map(x=>x/(Math.exp(3)+Math.exp(1)+1));
let previous=Infinity;
for(let a=0;a<=3;a+=.25){const R=T.dino(a);near(sum(R.target),1,'target normalization');near(sum(R.prediction),1,'student normalization');R.target.forEach((p,i)=>near(p,expectedTarget[i],'center/sharpen target'));near(R.loss,-sum(R.target.map((p,i)=>p*Math.log(R.prediction[i]))),'CE');assert(R.loss<=previous+1e-12,'CE should decrease toward this chosen target');previous=R.loss;comparisons++;}
const match=T.dino(3);match.target.forEach((p,i)=>near(p,match.prediction[i],'matching target'));near(match.loss,-sum(match.target.map(p=>p*Math.log(p))),'CE at target is entropy');assert(match.loss>0);
near(T.ema(),1.2,'EMA parameter');near(T.ema(1,3,1),1,'EMA endpoint');near(T.ema(1,3,0),3,'EMA endpoint');assert.throws(()=>T.ema(1,3,2));
assert.deepEqual(clone(T.transform('flip'))[0],[0,0,1,1]);assert.deepEqual(clone(T.transform('dim'))[3],[0,0,1.5,1.5]);assert.throws(()=>T.transform('unknown'));
assert.equal(JSON.stringify(toy),pristine,'runtime must not mutate embedded data');assert.equal(window.AT.axes.named,false);assert.equal(window.AT.notation.filter(r=>r.parts.includes('vision2')).length,9);
assert.equal(config.part,2);assert.equal(config.series,'Vision to language');assert.equal(config.notation,'vision2');assert.equal(config.prev.href,'vision1.html');assert.equal(config.next.href,'vision3.html');
const stages=['image','mae-encoder','mae-decoder','mae-transfer','views','contrastive','dino','dino-update','jepa','probe'];
function nodes(e){return[e,...e.children.flatMap(nodes)];}
const svgIds=new Set();
for(const stage of stages){const svg=T.diagram(new Element('div'),stage),b=svg.getAttribute('viewBox').split(' ').map(Number);assert(b[2]===1100&&b[3]<=330);assert.equal(svg.getAttribute('role'),'img');assert(nodes(svg).some(e=>e.tagName==='title'&&e.textContent));for(const n of nodes(svg)){const id=n.getAttribute('id');if(id){assert(!svgIds.has(id));svgIds.add(id);}}
  if(stage==='mae-encoder')assert(svg.textContent.includes('No mask tokens enter this encoder.'));
  if(stage==='mae-decoder')assert(svg.textContent.includes('[M]   [M]   [M]'));
  if(stage==='jepa')assert(svg.textContent.includes('full image')&&svg.textContent.includes('after encoding')&&svg.textContent.includes('stop gradient'));
}
const dir=new URL('sections6/',import.meta.url),sections=readdirSync(dir).filter(x=>/^sec\d\d\.html$/.test(x)).sort();assert.equal(sections.length,7);
const ids=new Set();let frames=0;
for(const file of sections){const source=readFileSync(new URL(file,dir),'utf8');frames+=(source.match(/class="frame"/g)||[]).length;
  for(const m of source.matchAll(/\bid="([^"]+)"/g)){assert(!ids.has(m[1]),'duplicate ID '+m[1]);ids.add(m[1]);}
  for(const m of source.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(m[1],{filename:file});
  for(const m of source.matchAll(/<pre class="torch-snippet"[^>]*><code>([\s\S]*?)<\/code><\/pre>/g))assert(m[1].trim().split('\n').length<=4,'short code on '+file);
  assert(!source.includes('onChange:')&&!source.includes('overflow:auto'),'use shared widgets/no internal scroll');
}
assert(frames>=20&&frames<=26);console.log(`PASS: Vision II ${comparisons} numerical cases, invariants, ${stages.length} SVG stages, ${sections.length} sections / ${frames} frames, IDs, code syntax, and notebook-size snippets.`);

if(process.argv.includes('--browser')){
  const require=createRequire(import.meta.url);let pw;
  for(const candidate of [process.env.PLAYWRIGHT_PATH,'/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)){try{pw=require(candidate);break;}catch{}}
  if(!pw)throw new Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
  const browser=await pw.chromium.launch();
  try{
    const page=await browser.newPage({viewport:{width:1200,height:900}});
    await page.setContent('<style>:root{--font-ui:Arial,sans-serif;--ink:#20272c;--ink-3:#606a70;--c-e:#658169;--card:#fff;--paper:#fff;--line:#ddd;--t-e:#f0f5f0}body{margin:20px}svg{display:block;width:1100px}</style><main></main>');
    await page.evaluate(toy=>{window.__TOY__=toy;window.AT={axes:{},objects:[],notation:[],clear:e=>{e.innerHTML='';}};},clone(toy));
    await page.addScriptTag({content:shadeSource});
    await page.addScriptTag({content:runtime});
    const fails=await page.evaluate(stages=>{const fails=[];for(const stage of stages){const host=document.createElement('div');document.querySelector('main').appendChild(host);const svg=AT.visionSSL.diagram(host,stage),v=svg.viewBox.baseVal;for(const text of svg.querySelectorAll('text')){const b=text.getBBox();if(b.x<0||b.y<0||b.x+b.width>v.width||b.y+b.height>v.height)fails.push(stage+': '+text.textContent);}}return fails;},stages);
    assert.deepEqual(fails,[],'SVG text stays inside viewBox');console.log('PASS: browser SVG text bounds.');
    const pixelLabels=await page.evaluate(()=>[...document.querySelectorAll('text[data-pixel-label]')].flatMap(text=>{
      const b=text.getBBox(),cx=Number(text.getAttribute('x')),cy=Number(text.getAttribute('y'));
      const cell=[...text.closest('svg').querySelectorAll('rect[data-pixel-value]')].find(rect=>{const r=rect.getBBox();return cx>=r.x&&cx<=r.x+r.width&&cy>=r.y&&cy<=r.y+r.height;})?.getBBox();
      return !cell||b.x<cell.x+1||b.x+b.width>cell.x+cell.width-1?[text.textContent]:[];
    }));
    assert.deepEqual(pixelLabels,[],'pixel labels fit within their own cells, including fractional brightness values');
    const arg=process.argv.indexOf('--page');if(arg>=0){
      const path=process.argv[arg+1];assert(path&&existsSync(path),'assembled page required');const errors=[];page.on('pageerror',e=>errors.push(String(e)));
      await page.goto(pathToFileURL(path).href);await page.waitForFunction(()=>window.AT?.visionSSL&&document.querySelector('#s03-lab-loss')?.textContent.includes('1.0000'));
      const guess=page.locator('#s03-slider input');await page.selectOption('#s03-mask','1');await guess.fill('2');await guess.dispatchEvent('input');assert((await page.locator('#s03-lab-loss').textContent()).includes('0.0000'));
      await page.selectOption('#s03-mask','2');await guess.fill('1');await guess.dispatchEvent('input');assert((await page.locator('#s03-lab-loss').textContent()).includes('0.5000'));
      const before=await page.locator('#s05-ce-loss').textContent();await page.locator('#s05-slider input').fill('3');await page.locator('#s05-slider input').dispatchEvent('input');const after=await page.locator('#s05-ce-loss').textContent();assert.notEqual(after,before);assert(after.includes(T.dino(3).loss.toFixed(4)));assert.deepEqual(errors,[]);
      for(const preset of ['0','1','2','1','0']){
        await page.selectOption('#s03-mask',preset);
        for(const value of ['0','0.75','2','1']){
          await guess.fill(value);await guess.dispatchEvent('input');
          assert.equal(await page.locator('#s03-lab-table table').count(),1,'MAE redraw replaces the previous table');
          assert.equal(await page.locator('#s03-lab-table tbody tr').count(),toy.visionSSL.maskPresets[+preset].indices.length,'mask changes also replace rows');
          const expected=T.reconstruction(toy.visionSSL.maskPresets[+preset].indices,+value);
          assert((await page.locator('#s03-lab-loss').textContent()).includes(expected.loss.toFixed(4)));
          assert((await page.locator('#s03-lab-table').innerText()).includes((+value).toFixed(2)),'current guess reaches the visible table');
        }
      }
      for(const value of ['0','1','3','2','0.5']){
        const slider=page.locator('#s05-slider input');await slider.fill(value);await slider.dispatchEvent('input');
        assert.equal(await page.locator('#s05-ce-table table').count(),1,'DINO redraw replaces the previous table');
        const expected=T.dino(+value),text=await page.locator('#s05-ce-table').innerText();
        expected.prediction.forEach(p=>assert(text.includes(p.toFixed(4)),'current probability is visible'));
        assert((await page.locator('#s05-ce-loss').textContent()).includes(expected.loss.toFixed(4)));
      }
      assert(await page.evaluate(()=>AT.imageShade(1.5)<AT.imageShade(2)&&AT.imageShade(0)<AT.imageShade(1)),'dimming lowers displayed intensity');
      console.log('PASS: assembled page interactions update MAE and DINO results without JavaScript errors.');
    }
  }finally{await browser.close();}
}

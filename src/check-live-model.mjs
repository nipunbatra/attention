// Check the assembled article against the independent numerical reference.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {forward} from './toy_ref.mjs';
const require=createRequire(import.meta.url);
let pw;
for(const p of [process.env.PLAYWRIGHT_PATH,'/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)){try{pw=require(p);break;}catch{}}
if(!pw)throw Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
const model=JSON.parse(readFileSync(new URL('toy.json',import.meta.url),'utf8'));
let numbers=0,infinities=0,maxError=0;
function compare(actual,expected,label){
  if(Array.isArray(expected)){assert(Array.isArray(actual),label);assert.equal(actual.length,expected.length,label);expected.forEach((x,j)=>compare(actual[j],x,label+'['+j+']'));return;}
  if(!Number.isFinite(expected)){assert.equal(actual,expected,label);infinities++;return;}
  assert(Number.isFinite(actual),label);const err=Math.abs(actual-expected);maxError=Math.max(maxError,err);numbers++;assert(err<1e-12,label+': '+actual+' != '+expected);
}
const b=await pw.chromium.launch();
try{
  const p=await b.newPage(),errors=[];p.on('pageerror',e=>errors.push(e.message));
  await p.goto(pathToFileURL(path.resolve(process.argv[2]||'attention.html')).href);
  assert.deepEqual(await p.evaluate(()=>window.__TOY__),model,'assembled HTML must contain the current model');
  for(const sentence of ['river','cheque'])for(const mask of [false,true])for(const scale of [false,true]){
    const tokens=model.sentences[sentence], ref=forward(model,tokens,{mask,scale});
    const live=await p.evaluate(({tokens,mask,scale})=>AT.forward(tokens,{mask,scale}),{tokens,mask,scale});
    for(const k of ['E','Q','K','V','Sraw','S','A','Mmsg','Delta','Enew','logits','probs'])compare(live[k],ref[k],sentence+'.'+mask+'.'+scale+'.'+k);
    compare(live.Sfull,ref.Sraw.map(r=>r.map(x=>scale?x/Math.sqrt(model.d_k):x)),'scaled scores before mask');
  }
  assert.deepEqual(errors,[]);
  console.log(`PASS:8 live/reference cases; ${numbers} finite values, ${infinities} masked infinities; max error=${maxError}; assembled model current.`);
}finally{await b.close();}

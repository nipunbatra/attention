import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const browser=await chromium.launch({headless:!process.argv.includes('--headed')});
const shots=fs.mkdtempSync('/tmp/word-lab-');
const base=process.env.LAB_URL||'http://127.0.0.1:8778/word-lab/';
const page=await browser.newPage({viewport:{width:1360,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
  await page.goto(base);await page.locator('#generate:enabled').waitFor({timeout:60000});
  await page.screenshot({path:shots+'/desktop.png',fullPage:true});
  // Execute real model graphs and compare every logit to the PyTorch probe.
  const parity=await page.evaluate(async()=>{
    ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';ort.env.wasm.numThreads=1;
    const report={};
    for(const kind of ['mlp','attention','multihead']){
      const p=await (await fetch(`./models/${kind}-parity.json`)).json();
      const session=await ort.InferenceSession.create(`./models/${kind}.onnx`,{executionProviders:['wasm']});
      const input=new ort.Tensor('int32',Int32Array.from(p.input_ids.flat()),[3,64]);
      const r=await session.run({input_ids:input});let error=0;
      const expected=p.logits.flat();for(let i=0;i<expected.length;i++)error=Math.max(error,Math.abs(expected[i]-r.logits.data[i]));
      report[kind]=error;input.dispose();r.logits.dispose();await session.release();
    }return report;
  });
  for(const [kind,error] of Object.entries(parity))assert(error<5e-4,`${kind}: ${error}`);
  const saved=JSON.parse(fs.readFileSync(new URL('saved-examples.json',import.meta.url)));
  await page.locator('.settings summary').click();
  await page.selectOption('#backend','wasm');await page.selectOption('#token-count','24');
  async function generate(){await page.evaluate(()=>{globalThis.lastComparison=null;});await page.click('#generate');await page.waitForFunction(()=>globalThis.lastComparison,{timeout:120000});return page.evaluate(()=>globalThis.lastComparison);}
  const runs=[];for(let repeat=0;repeat<4;repeat++)runs.push(await generate());
  for(const kind of ['mlp','attention','multihead'])assert.deepEqual(runs[0].results[kind].ids,saved.examples.find(e=>e.prompt_id==='train-588307'&&e.model===kind).generated_ids);
  const median=values=>values.sort((a,b)=>a-b)[Math.floor(values.length/2)];
  const timing={device:'Apple M2 Max, '+await browser.version()+', Chromium headless',backend:'WASM CPU, one thread',repetitions:3,prompt:await page.inputValue('#prompt'),models:{},browser_parity_max_absolute_error:parity};
  for(const kind of ['mlp','attention','multihead']){const rs=runs.slice(1).map(r=>r.results[kind]);timing.models[kind]={new_tokens:rs[0].ids.length,calls:rs[0].calls,generation_ms_median:median(rs.map(r=>r.total_ms)),ms_per_call_median:median(rs.map(r=>r.inference_ms/r.calls)),runs:rs};}
  if(!process.env.LAB_URL)fs.writeFileSync(new URL('browser-timing.json',import.meta.url),JSON.stringify(timing,null,2)+'\n');
  await page.screenshot({path:shots+'/generated.png',fullPage:true});
  await page.selectOption('#example','science');assert.match(await page.textContent('#token-note'),/Unknown words/);const outside=await generate();
  assert.notDeepEqual(outside.results.mlp.ids,runs[0].results.mlp.ids,'a different prompt computes different output');
  await page.selectOption('#example','train-1992490');assert.equal(await page.locator('#source-details').isVisible(),true);await page.locator('#source-details summary').click();assert.match(await page.textContent('#source-text'),/old hotel/);
  await page.fill('#prompt','a new prompt');assert.equal(await page.locator('#source-details').isVisible(),false);
  await page.selectOption('#decoding','sample');assert.equal(await page.locator('#seed').isEnabled(),true);
  await page.selectOption('#decoding','greedy');assert.equal(await page.locator('#seed').isEnabled(),false);
  for(const width of [1024,390]){await page.setViewportSize({width,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:`${shots}/${width}.png`,fullPage:true});}
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({parity,timing,shots},null,2));
}finally{await browser.close();}

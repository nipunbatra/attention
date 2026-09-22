import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const browser=await chromium.launch({headless:!process.argv.includes('--headed')});
try {
  const page=await browser.newPage();
  await page.goto(process.env.LAB_URL||'http://127.0.0.1:8778/word-lab/');
  await page.locator('#generate:enabled').waitFor();
  const gpu=await page.evaluate(async()=>{
    const adapter=await navigator.gpu?.requestAdapter();
    if(!adapter)return {available:false};
    const info=adapter.info;
    const report={available:true,vendor:info.vendor,architecture:info.architecture,description:info.description,parity:{}};
    for(const kind of ['mlp','attention','multihead']){
      const probe=await (await fetch(`./models/${kind}-parity.json`)).json();
      const session=await ort.InferenceSession.create(`./models/${kind}.onnx`,{executionProviders:['webgpu']});
      const input=new ort.Tensor('int32',Int32Array.from(probe.input_ids.flat()),[3,64]);
      const result=await session.run({input_ids:input});
      const expected=probe.logits.flat();let error=0;
      for(let i=0;i<expected.length;i++)error=Math.max(error,Math.abs(expected[i]-result.logits.data[i]));
      report.parity[kind]=error;input.dispose();result.logits.dispose();await session.release();
    }
    return report;
  });
  if(gpu.available)for(const error of Object.values(gpu.parity))assert(error<2e-3);
  await page.locator('.settings summary').click();await page.selectOption('#token-count','24');
  await page.click('#generate');
  await page.waitForFunction(()=>globalThis.lastComparison,{timeout:120000});
  const results=await page.evaluate(()=>globalThis.lastComparison);
  for(const r of Object.values(results.results))assert.equal(r.backend,gpu.available?'WebGPU':'WASM');
  const report={browser:await browser.version(),gpu,automatic_generation:results};
  if(!process.env.LAB_URL)fs.writeFileSync(new URL('webgpu-check.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}

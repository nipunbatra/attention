// Reproduce the masked-image experiment. Use --check for a read-only check.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const dir=new URL('./',import.meta.url),window={AT:{}};
vm.runInNewContext(fs.readFileSync(new URL('part6-learning.js',dir),'utf8'),{window});
const result=window.AT.mae.run();
if(process.argv.includes('--check')){
  const saved=JSON.parse(fs.readFileSync(new URL('mae6.json',dir),'utf8'));
  let count=0,max=0;
  function compare(a,b){if(Array.isArray(b))return b.forEach((v,i)=>compare(a[i],v));if(b&&typeof b==='object')return Object.keys(b).forEach(k=>compare(a[k],b[k]));if(typeof b==='number'){max=Math.max(max,Math.abs(a-b));count++;assert(Math.abs(a-b)<1e-10);}else assert.equal(a,b);}
  compare(result,saved);console.log(JSON.stringify({reproduced:count,maxError:max,initial:result.history[0],final:result.history.at(-1)}));
}else{
  fs.writeFileSync(new URL('mae6.json',dir),JSON.stringify(result));
  console.log(JSON.stringify({written:'src/mae6.json',initial:result.history[0],final:result.history.at(-1)}));
}

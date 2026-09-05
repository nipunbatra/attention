import { createRequire } from 'module';const require=createRequire(import.meta.url);
const {chromium}=require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage();
for(const f of ['part1','attention','part3','part4','vision1','vision2','vision3','vision4']){
await p.goto('file:///Users/nipun/git/attention/'+f+'.html');await p.waitForTimeout(1500);
// click all controls? just read initial
const r=await p.evaluate(()=>{const out=[];document.querySelectorAll('table.dt thead th').forEach(th=>{const t=th.textContent.trim();if(/_/.test(t)&&!th.querySelector('.katex'))out.push(t);});return [...new Set(out)];});
console.log(f, JSON.stringify(r));}
await b.close();})();

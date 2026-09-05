import { createRequire } from 'module';const require=createRequire(import.meta.url);
const {chromium}=require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage();
for(const f of ['part1','attention','part3','part4','vision1','vision2','vision3','vision4']){
await p.goto('file:///Users/nipun/git/attention/'+f+'.html');await p.waitForTimeout(1200);
const btns=await p.$$('button, input[type=range]');
for(const bt of btns){try{await bt.click({timeout:400});}catch(e){}}
await p.waitForTimeout(1200);
const r=await p.evaluate(()=>{const k=[],a=[];document.querySelectorAll('table.dt thead th').forEach(th=>{const t=th.textContent.trim();if(!t)return;if(th.querySelector('.katex'))k.push(t);else a.push(t);});return {katex:[...new Set(k)].length, ascii_us:[...new Set(a.filter(x=>/_/.test(x)))]};});
console.log(f, JSON.stringify(r));}
await b.close();})();

// Rendered input pixels must mean the same thing across the four vision lessons.
// node src/check_vision_pixels.mjs /path/vision1.html ... /path/vision4.html
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const require=createRequire(import.meta.url);
let pw;
for(const candidate of [process.env.PLAYWRIGHT_PATH,'playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)){
  try{pw=require(candidate);break;}catch{}
}
if(!pw)throw Error('Set PLAYWRIGHT_PATH to an installed Playwright module.');
const files=process.argv.slice(2);assert(files.length>0,'Supply assembled vision HTML files.');
const browser=await pw.chromium.launch();
try{
  for(const file of files){
    const page=await browser.newPage(),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(pathToFileURL(path.resolve(file)).href);
    const result=await page.evaluate(()=>{
      const pixels=[...document.querySelectorAll('rect[data-pixel-value]')].filter(el=>el.dataset.pixelValue!=='');
      const issues=pixels.flatMap(el=>{
        const v=Number(el.dataset.pixelValue),shade=Math.round(35+69*Math.max(0,Math.min(3,v)));
        const expected='rgb('+shade+', '+shade+', '+shade+')',actual=getComputedStyle(el).fill;
        return expected===actual?[]:[{value:v,expected,actual}];
      });
      return {count:pixels.length,issues,ramp:[0,1,2,3].map(AT.imageShade),darker:AT.imageShade(1.5)<AT.imageShade(2)};
    });
    assert.deepEqual(errors,[],file+' runtime errors');
    assert(result.count>=16,file+' should render actual input pixels');
    assert.deepEqual(result.ramp,[35,104,173,242],file+' grayscale convention');
    assert.equal(result.darker,true,file+' dim means darker');
    assert.deepEqual(result.issues,[],file+' pixel rendering');
    console.log('PASS: '+path.basename(file)+' / '+result.count+' rendered pixels match the shared grayscale ramp.');
    await page.close();
  }
}finally{await browser.close();}

// Verify that the shared Vision scene renderer draws the actual 8x8 pixel
// values, in the same grayscale ramp, on every assembled vision page.
// node src/check_vision_pixels.mjs vision1.html vision2.html vision3.html vision4.html
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
      const V=AT.vision, scene=V.scene('A');
      const drawn=V.grid('A',{cell:20,labels:'none',patchLines:false});
      const pixels=[...drawn.querySelectorAll('svg rect')].slice(0,64);
      const issues=[];
      if(pixels.length!==64)issues.push('expected 64 pixel rectangles, got '+pixels.length);
      for(let r=0;r<8;r++)for(let c=0;c<8;c++){
        const i=r*8+c, value=scene[r][c], expected=V.ramp(value);
        if(pixels[i]?.getAttribute('fill')!==expected)issues.push({r,c,value,expected,actual:pixels[i]?.getAttribute('fill')});
      }
      const ramp=[0,1,2,3].map(V.ramp), brightness=ramp.map(hex=>parseInt(hex.slice(1,3),16));
      return {pixels:pixels.length,issues,ramp,brightness,renderedGrids:document.querySelectorAll('.vgrid svg').length};
    });
    assert.deepEqual(errors,[],file+' runtime errors');
    assert.equal(result.pixels,64,file+' should draw every input pixel');
    assert.equal(result.issues.length,0,file+' pixel values and fills: '+JSON.stringify(result.issues.slice(0,3)));
    assert.deepEqual(result.ramp,['#3A3A3A','#6E6E6E','#A3A3A3','#D8D8D8']);
    assert.ok(result.brightness.every((v,i,a)=>i===0||v>a[i-1]),'larger pixel values appear brighter');
    assert.ok(result.renderedGrids>0,file+' should show a scene in the article');
    console.log('PASS: '+path.basename(file)+' / 64 scene pixels, shared grayscale ramp, '+result.renderedGrids+' article grids');
    await page.close();
  }
}finally{await browser.close();}

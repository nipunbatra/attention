// Verify the embedded scene assets, crop geometry, accessible labels and bounds.
// node src/check_vision_scene.mjs [--shots /tmp/vision-scenes]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(import.meta.url);
let pw;
for(const candidate of [process.env.PLAYWRIGHT_MODULE,'playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)){
  try{pw=require(candidate);break;}catch{}
}
if(!pw)throw Error('Set PLAYWRIGHT_MODULE to an installed Playwright module.');
const args=process.argv.slice(2),si=args.indexOf('--shots'),shots=si>=0?args[si+1]:null;
if(shots)fs.mkdirSync(shots,{recursive:true});
const browser=await pw.chromium.launch();
const report=[];
try{
  for(let n=1;n<=4;n++){
    const errors=[],remote=[];
    const page=await browser.newPage({viewport:{width:1280,height:720}});
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',request=>{if(/^https?:/.test(request.url()))remote.push(request.url());});
    await page.goto(pathToFileURL(path.join(root,`vision${n}.html`)).href);
    await page.evaluate(()=>document.fonts.ready);
    const assets=await page.evaluate(async()=>{
      const dims=await Promise.all(Object.entries(window.__VISION_SCENES__).map(async([variant,url])=>{
        const image=new Image();image.src=url;await image.decode();
        return {variant,width:image.naturalWidth,height:image.naturalHeight,embedded:url.startsWith('data:image/jpeg;base64,')};
      }));
      return {dims,patch:AT.visionScene.patch};
    });
    assert.deepEqual(assets.patch,[512,512,256,256]);
    assert.equal(assets.dims.length,2);
    assets.dims.forEach(d=>{assert.equal(d.width,1536);assert.equal(d.height,1024);assert(d.embedded);});
    const frames=await page.evaluate(()=>AT.present.frames().map((f,index)=>({id:f.id,num:f.index+1,index,title:f.title,hasScene:!!f.el.querySelector('[data-vision-scene]')})).filter(f=>f.hasScene));
    assert(frames.length>=3,`Vision ${n} needs the opening and return-to-scene examples.`);
    const details=[];
    for(const frame of frames){
      await page.evaluate(({index})=>{AT.present.enter();AT.present.go(index,0);AT.present.setBuild(AT.present.state().frame.maxBuild);},frame);
      await page.waitForTimeout(90);
      const result=await page.evaluate(()=>{
        const frame=document.querySelector('.frame.is-live');
        const figures=[...frame.querySelectorAll('[data-vision-scene]')];
        return figures.map(fig=>{
          const svg=fig.querySelector('svg'),b=svg.getBoundingClientRect();
          const bad=[...svg.querySelectorAll('text')].flatMap(t=>{
            const r=t.getBoundingClientRect();
            return r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2?[t.textContent]:[];
          });
          const crops=[...svg.querySelectorAll('svg')].map(crop=>({
            width:crop.width.baseVal.value,actualWidth:crop.getScreenCTM().a/svg.getScreenCTM().a*crop.viewBox.baseVal.width,
            height:crop.height.baseVal.value,actualHeight:crop.getScreenCTM().d/svg.getScreenCTM().d*crop.viewBox.baseVal.height,
            overflow:getComputedStyle(crop).overflow
          }));
          return {mode:fig.dataset.visionScene,title:svg.querySelector('title')?.textContent,caption:fig.querySelector('figcaption')?.textContent,bad,crops};
        });
      });
      assert.equal(await page.evaluate(()=>AT.present.state().fi),frame.index);
      assert(result.length>0,frame.title+' must actually display its scene');
      result.forEach(r=>{
        assert(r.title?.length>10);assert.equal(r.caption,undefined,'No repeated production caption on the classroom figure');assert.deepEqual(r.bad,[],frame.title+' SVG text bounds');
        r.crops.forEach(crop=>{assert(Math.abs(crop.width-crop.actualWidth)<1,JSON.stringify(crop));assert(Math.abs(crop.height-crop.actualHeight)<1,JSON.stringify(crop));assert.equal(crop.overflow,'hidden');});
      });
      details.push({title:frame.title,modes:result.map(r=>r.mode)});
      if(shots)await page.screenshot({path:path.join(shots,`vision${n}-${frame.id}-${frame.num}.png`)});
    }
    await page.evaluate(()=>AT.present.exit());
    await page.setViewportSize({width:390,height:844});
    for(const frame of frames){
      // Use the runtime's own frame element rather than DOM sibling numbering.
      await page.evaluate(({index})=>AT.present.frames()[index].el.scrollIntoView({behavior:'instant'}),frame);
      const mobile=await page.evaluate(({index})=>{
        const f=AT.present.frames()[index].el;
        return {width:document.documentElement.clientWidth,bodyWidth:document.documentElement.scrollWidth,
          figures:[...f.querySelectorAll('.vision-scene')].map(fig=>({width:fig.clientWidth,scrollWidth:fig.scrollWidth,mode:fig.dataset.visionScene}))};
      },frame);
      assert(mobile.bodyWidth<=mobile.width+1);
      mobile.figures.forEach(fig=>assert(fig.scrollWidth<=fig.width+1,JSON.stringify(fig)));
      if(shots)await page.screenshot({path:path.join(shots,`vision${n}-${frame.id}-${frame.num}-phone.png`)});
    }
    assert.deepEqual(errors,[],`Vision ${n} runtime errors`);
    assert.deepEqual(remote,[],`Vision ${n} must work offline`);
    report.push({lesson:n,sceneFrames:frames.length,assets:assets.dims,frames:details});
    await page.close();
  }
}finally{await browser.close();}
console.log(JSON.stringify({passed:true,lessons:report},null,2));

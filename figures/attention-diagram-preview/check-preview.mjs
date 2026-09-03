import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
let pw;
for(const candidate of [process.env.PLAYWRIGHT_PATH,'/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright','/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)) {
  try {pw=require(candidate);break;} catch {}
}
if(!pw) throw new Error('Set PLAYWRIGHT_PATH to an existing Playwright package.');
const dir=path.dirname(fileURLToPath(import.meta.url));
const browser=await pw.chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(dir,'index.html')).href);
  for(let i=0;i<12;i++) {
    await page.locator(`#tab-${i}`).click();
    assert.equal(await page.locator('#counter').innerText(),`${String(i+1).padStart(2,'0')} / 12`);
    const outside=await page.locator('#canvas svg').evaluate(svg=>[...svg.querySelectorAll('text')].map(t=>({text:t.textContent,b:t.getBBox()})).filter(x=>x.b.x<0||x.b.y<0||x.b.x+x.b.width>1600||x.b.y+x.b.height>960).map(x=>x.text));
    assert.deepEqual(outside,[],`stage ${i+1}: text outside SVG`);
    const nodes=await page.locator('#canvas [data-node]').count();
    assert(nodes>0);
  }
  assert.match(await page.locator('#canvas svg').textContent(),/receiver 10/);
  await page.getByRole('button',{name:'All stages',exact:true}).click();
  assert.equal(await page.locator('#storyboard .thumbnail').count(),12);
  await page.getByRole('button',{name:'Open stage 6: Bank cannot read the future',exact:true}).click();
  assert.equal(await page.locator('#counter').innerText(),'06 / 12');
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#counter').innerText(),'07 / 12');
  await page.keyboard.press('ArrowLeft');
  assert.equal(await page.locator('#counter').innerText(),'06 / 12');
  const downloadPromise=page.waitForEvent('download');
  await page.getByRole('button',{name:'Download SVG',exact:true}).click();
  assert.equal((await downloadPromise).suggestedFilename(),'attention-stage-06-mask.svg');
  assert.deepEqual(errors,[]);
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,'mobile has no body-level overflow');
  console.log('PASS: 12 SVG stages, text bounds, receiver switch, overview, keyboard, SVG download, mobile containment; zero page errors.');
} finally {await browser.close();}

// node hovershot.mjs page.html out.png "css" [width]  — scroll to the element, hover it with a real mouse move, print tooltip state, screenshot the viewport
import { createRequire } from 'module'; import path from 'path';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const [file,out,sel,width='390'] = process.argv.slice(2);
const b = await pw.chromium.launch();
const p = await (await b.newContext({viewport:{width:+width,height:800}})).newPage();
await p.goto('file://'+path.resolve(file),{waitUntil:'load'}); await p.waitForTimeout(500);
const el = await p.$(sel); await el.scrollIntoViewIfNeeded(); await p.evaluate(()=>window.scrollBy({top:-250,behavior:"instant"})); await p.waitForTimeout(700);
const box = await el.boundingBox();
await p.mouse.move(box.x+box.width/2, box.y+box.height/2); await p.waitForTimeout(400);
console.log(await p.evaluate(s=>{const td=document.querySelector(s); const tip=td.closest('.dt-fig').querySelector('.dt-tip'); const r=tip.getBoundingClientRect(); return JSON.stringify({on:tip.classList.contains('is-on'), box:[Math.round(r.x),Math.round(r.y),Math.round(r.width),Math.round(r.height)], overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth+1, strip: document.getElementById('strip').getBoundingClientRect().height})}, sel));
await p.screenshot({path:out}); await b.close();

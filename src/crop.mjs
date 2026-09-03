// node crop.mjs page.html out.png y0 height [width] [scale]  — full-page clip screenshot at native size
import { createRequire } from 'module'; import path from 'path';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const [file, out, y0, h, w = '1280', scale = '1'] = process.argv.slice(2);
const b = await pw.chromium.launch();
const ctx = await b.newContext({ viewport: { width: +w, height: 800 }, deviceScaleFactor: +scale });
const p = await ctx.newPage();
await p.goto('file://' + path.resolve(file), { waitUntil: 'load' }); await p.waitForTimeout(700);
await p.screenshot({ path: out, fullPage: true, clip: { x: 0, y: +y0, width: +w, height: +h } });
await b.close(); console.log('wrote', out);

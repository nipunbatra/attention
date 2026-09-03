// node secshot.mjs page.html sNN out.png [width] [height] — viewport screenshot of a section
import { createRequire } from 'module'; import path from 'path';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const [file, id, out, w = '1280', h = '900'] = process.argv.slice(2);
const b = await pw.chromium.launch(); const p = await (await b.newContext({ viewport: { width: +w, height: +h } })).newPage();
await p.goto('file://' + path.resolve(file), { waitUntil: 'load' }); await p.waitForTimeout(600);
await p.evaluate(id => { document.documentElement.style.scrollBehavior = 'auto'; document.getElementById(id).scrollIntoView(); window.scrollBy(0, -90); }, id);
await p.waitForTimeout(400); await p.screenshot({ path: out }); await b.close(); console.log('wrote', out);

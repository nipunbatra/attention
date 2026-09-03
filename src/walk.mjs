// node walk.mjs page.html outdir [width height] [--only sNN]  — present-mode walk: screenshot every build of every frame
import { createRequire } from 'module'; import path from 'path'; import fs from 'fs';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const args = process.argv.slice(2);
const [file, outdir = 'frames'] = args;
const only = args.includes('--only') ? args[args.indexOf('--only') + 1] : null;
const width = +(args[2] && !isNaN(+args[2]) ? args[2] : 1280), height = +(args[3] && !isNaN(+args[3]) ? args[3] : 720);
fs.mkdirSync(outdir, { recursive: true });
const errs = [];
const b = await pw.chromium.launch();
const ctx = await b.newContext({ viewport: { width, height } });
const p = await ctx.newPage();
p.on('pageerror', e => errs.push('PAGEERROR: ' + String(e.message || e).split('\n')[0]));
p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push('CONSOLE ' + m.type() + ': ' + m.text().slice(0, 200)); });
await p.goto('file://' + path.resolve(file) + '?present' + (only ? '#' + only + '/1/0' : ''), { waitUntil: 'load' });
await p.waitForTimeout(700);
const shots = []; let last = null; let overflow = [];
for (let n = 0; n < 2000; n++) {
  const s = await p.evaluate(() => { const st = AT.present.state(); const f = document.querySelector('.frame.is-live'); return { hash: location.hash, fi: st.fi, total: st.total, step: st.stepper ? st.stepper.index : -1, id: st.frame && st.frame.id, tall: f ? f.scrollHeight > f.clientHeight + 2 : false, wide: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1 }; });
  const key = s.hash + (s.step >= 0 ? '_st' + (s.step + 1) : '');
  if (key === last) break;
  if (only && s.id !== only) break;
  last = key;
  const name = key.replace(/[#/]/g, '_').replace(/^_/, '') + '.png';
  await p.screenshot({ path: path.join(outdir, name) }); shots.push(name);
  if (s.tall) overflow.push(key + ' (scrolls)'); if (s.wide) overflow.push(key + ' (WIDE)');
  await p.keyboard.press('ArrowRight'); await p.waitForTimeout(220);
}
await b.close();
console.log(JSON.stringify({ shots: shots.length, errors: errs, tallFrames: overflow }, null, 2));

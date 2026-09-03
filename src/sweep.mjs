// node sweep.mjs page.html [width]  — click every control in every section, report errors + bad text
import { createRequire } from 'module'; import path from 'path';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const [file, width = '1280'] = process.argv.slice(2);
const b = await pw.chromium.launch();
const ctx = await b.newContext({ viewport: { width: +width, height: 800 } });
const p = await ctx.newPage();
const errs = [];
p.on('pageerror', e => errs.push('PAGEERROR: ' + String(e.message || e).split('\n')[0]));
p.on('console', m => { if (m.type() === 'error') errs.push('CONSOLE: ' + m.text().slice(0, 200)); });
await p.goto('file://' + path.resolve(file), { waitUntil: 'load' }); await p.waitForTimeout(800);
const secs = await p.$$eval('section.sec', els => els.map(e => e.id));
const report = {};
for (const id of secs) {
  const before = errs.length;
  await p.evaluate(id => document.getElementById(id).scrollIntoView(), id); await p.waitForTimeout(150);
  const counts = await p.evaluate(async id => {
    const S = document.getElementById(id); const c = { buttons: 0, sliders: 0, toggles: 0, details: 0 };
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    for (const d of S.querySelectorAll('details')) { d.open = true; c.details++; }
    for (const r of S.querySelectorAll('input[type=range]')) { for (const v of [r.min, (+r.min + +r.max) / 2, r.max]) { r.value = v; r.dispatchEvent(new Event('input', { bubbles: true })); r.dispatchEvent(new Event('change', { bubbles: true })); await sleep(20); } c.sliders++; }
    for (const t of S.querySelectorAll('[aria-pressed]')) { t.click(); await sleep(30); t.click(); await sleep(30); c.toggles++; }
    const btns = [...S.querySelectorAll('button:not([aria-pressed])')];
    for (const bt of btns) { if (bt.disabled) continue; const txt = (bt.textContent || '').trim().toLowerCase(); const n = /next/.test(txt) ? 22 : 1; for (let i = 0; i < n; i++) { if (bt.disabled) break; bt.click(); await sleep(25); } c.buttons++; }
    // reset steppers if present
    for (const bt of btns) { if (/reset/.test((bt.textContent || '').toLowerCase())) { bt.click(); await sleep(25); } }
    return c;
  }, id);
  await p.waitForTimeout(200);
  const bad = await p.evaluate(id => { const t = document.getElementById(id).innerText; const hits = []; for (const w of ['NaN', 'undefined', '[object', 'Infinity', 'null']) { const i = t.indexOf(w); if (i >= 0) hits.push(w + ' @ "' + t.slice(Math.max(0, i - 40), i + 30).replace(/\n/g, ' ') + '"'); } return hits; }, id);
  report[id] = { ...counts, errors: errs.slice(before), badText: bad };
}
const summary = Object.entries(report).filter(([k, v]) => v.errors.length || v.badText.length);
console.log(JSON.stringify({ sections: secs.length, controlsClicked: Object.values(report).reduce((a, v) => a + v.buttons + v.sliders + v.toggles + v.details, 0), problems: Object.fromEntries(summary), totalErrors: errs.length }, null, 2));
await b.close();

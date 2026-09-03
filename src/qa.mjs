// Headless QA harness. Usage:
//   node qa.mjs <file.html> [--width 1280] [--height 800] [--shot out.png] [--full] [--eval "js expr"] [--click "css selector"]... [--wait ms]
// Prints JSON: {pageErrors, consoleErrors, katexErrors, evalResult, docHeight}
import { createRequire } from 'module';
import path from 'path';
const require = createRequire(import.meta.url);
const pw = require('/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const args = process.argv.slice(2);
const file = args[0];
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const flag = k => args.includes(k);
const clicks = []; args.forEach((a, i) => { if (a === '--click') clicks.push(args[i + 1]); });
const b = await pw.chromium.launch();
const ctx = await b.newContext({ viewport: { width: +opt('--width', 1280), height: +opt('--height', 800) } });
const p = await ctx.newPage();
const pageErrors = [], consoleErrors = [];
p.on('pageerror', e => pageErrors.push(String(e && e.stack || e)));
p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') consoleErrors.push(m.type() + ': ' + m.text()); });
await p.goto('file://' + path.resolve(file), { waitUntil: 'load' });
await p.waitForTimeout(+opt('--wait', 600));
for (const sel of clicks) { try { await p.click(sel, { timeout: 3000 }); await p.waitForTimeout(250); } catch (e) { pageErrors.push('click failed: ' + sel + ' :: ' + e.message.split('\n')[0]); } }
let evalResult = null; const ev = opt('--eval', null);
if (ev) { try { evalResult = await p.evaluate(ev); } catch (e) { evalResult = 'EVAL ERROR: ' + e.message; } }
const katexErrors = await p.evaluate(() => Array.from(document.querySelectorAll('.katex-error')).map(e => e.textContent.slice(0, 120)));
const docHeight = await p.evaluate(() => document.documentElement.scrollHeight);
const overflowX = await p.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
const shot = opt('--shot', null);
if (shot) await p.screenshot({ path: shot, fullPage: flag('--full') });
console.log(JSON.stringify({ pageErrors, consoleErrors, katexErrors, overflowX, docHeight, evalResult }, null, 2));
await b.close();

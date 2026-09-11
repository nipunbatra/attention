// Check that introductory Part 1 equations have nearby prose in matching colours.
// node src/notation_test.mjs [part1.html] [screenshot-directory]
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) for (const dir of fs.readdirSync(cache)) candidates.push(path.join(cache, dir, 'node_modules/playwright'));
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('Set PLAYWRIGHT_MODULE to an existing Playwright installation.');
const out = process.argv[3] || fs.mkdtempSync(path.join(os.tmpdir(), 'attention-notation-'));
fs.mkdirSync(out, { recursive: true });
const browser = await pw.chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
await page.goto(pathToFileURL(path.resolve(process.argv[2] || 'part1.html')).href);
await page.waitForTimeout(250);
const checkGuides = () => {
  const issues = [], guides = [...document.querySelectorAll('[data-math-guide]')];
  const roles = ['input', 'target', 'param', 'prob', 'activation', 'score', 'loss'];
  for (const guide of guides) {
    for (const role of roles) {
      const cls = '.p1-' + role;
      const symbols = [...guide.querySelectorAll('.p1-math .katex-html ' + cls)];
      if (!symbols.length) continue;
      const prose = [...guide.querySelectorAll('.p1-key ' + cls + ', .p1-read' + cls + ', .p1-read ' + cls)]
        .filter(el => !el.closest('.katex') && /[A-Za-z]{3}/.test(el.textContent));
      if (!prose.length) issues.push(guide.closest('.sec').id + ': no prose for ' + role);
      for (const symbol of symbols) if (!prose.some(el => getComputedStyle(el).color === getComputedStyle(symbol).color)) {
        issues.push(guide.closest('.sec').id + ': colour mismatch for ' + role);
      }
    }
  }
  if (guides.length < 10) issues.push('Missing introductory equation guides');
  for (const row of document.querySelectorAll('#s06-shapes tbody tr, #s16-input-notation tbody tr, #s16-output-notation tbody tr')) {
    const symbol = row.querySelector('th .katex-html [class*="p1-"]');
    const meaning = row.querySelector('td');
    if (!symbol || !meaning || getComputedStyle(symbol).color !== getComputedStyle(meaning).color) {
      issues.push(row.closest('[id]').id + ': table symbol/meaning colour mismatch');
    }
  }
  return { guides: guides.length, issues };
};
const article = await page.evaluate(checkGuides);
errors.push(...article.issues);
// Include the fully revealed notation frames and the tables adjacent to them.
for (const [section, title, name] of [
  ['s03', 'Writing the same question as a probability', 'probability'],
  ['s04', 'Look up a learned row', 'lookup'],
  ['s05', 'Three rows become one row', 'concatenation'],
  ['s06', 'The hidden layer combines the input numbers', 'hidden-layer'],
  ['s06', 'The output layer makes one score per token', 'output-layer'],
  ['s06', 'Check the shapes before multiplying', 'shapes'],
  ['s07', 'Exponentiate, then divide by the total', 'softmax'],
  ['s07', 'The same softmax, with smaller intermediate numbers', 'stable-softmax'],
  ['s07', 'Softmax worksheet', 'softmax-table'],
  ['s08', 'One target probability', 'loss'],
  ['s08', 'Low target probability gives a large loss', 'loss-examples'],
  ['s09', 'What the optimizer changes', 'parameters'],
  ['s14', 'A larger window makes a larger weight matrix', 'window-shapes'],
  ['s16', 'The same model in symbols', 'summary'],
  ['s16', 'Notation: from tokens to the MLP input', 'input-notation'],
  ['s16', 'Notation: from the input to a probability', 'output-notation']
]) {
  await page.evaluate(({ section, title }) => {
    const frames = [...document.querySelectorAll('#' + section + ' .frame')];
    const index = frames.findIndex(frame => frame.dataset.title === title);
    if (index < 0) throw new Error('Missing notation frame: ' + title);
    AT.present.enter(); AT.present.go(section, index + 1, 99);
  }, { section, title });
  await page.waitForTimeout(80);
  const fit = await page.evaluate(() => AT.present.fitReport());
  if (fit.overflow) errors.push(name + ': frame overflow');
  await page.screenshot({ path: path.join(out, name + '.png') });
}
await page.evaluate(() => AT.present.exit());
await page.setViewportSize({ width: 390, height: 844 });
await page.evaluate(() => document.querySelectorAll('#s03 .frame')[1].scrollIntoView({ behavior: 'instant' }));
await page.waitForTimeout(80);
const phone = await page.evaluate(() => ({
  overflow: document.documentElement.scrollWidth > innerWidth,
  columns: getComputedStyle(document.querySelector('#s03 .p1-key')).gridTemplateColumns.split(' ').length
}));
if (phone.overflow || phone.columns !== 1) errors.push('Phone article legend does not fit one column');
await page.screenshot({ path: path.join(out, 'phone-probability.png') });
await page.emulateMedia({ media: 'print' });
const printed = await page.evaluate(checkGuides);
errors.push(...printed.issues.map(x => 'print: ' + x));
await browser.close();
console.log(JSON.stringify({ article, phone, print: printed, screenshots: out, errors }, null, 2));
if (errors.length) process.exitCode = 1;

// Focused, read-only regression checks for §§08, 12, 14.
// Uses the project's existing Playwright installation. No HTML/model files are written.
// Run: node src/check-routing-scaling.mjs [--shots-dir /absolute/existing/directory]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { forward, baseline, matmul, softmax } from './toy_ref.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const read = name => fs.readFileSync(path.join(here, name), 'utf8');
const toy = JSON.parse(read('toy.json'));
const original = JSON.stringify(toy);
const tokens = toy.sentences.river.slice(0, 7);
const F = forward(toy, tokens);
const financeIndex=toy.axes.v.findIndex(axis=>/finance/i.test(typeof axis==='string'?axis:JSON.stringify(axis)));
assert(financeIndex>=0,'the declared intervention needs a named finance value coordinate');
const alternate = structuredClone(toy);
alternate.W_V = matmul(toy.W_V, Array.from({ length: toy.d_v }, (_, i) => Array.from({ length: toy.d_v }, (_, j) => i === j && i !== financeIndex ? 1 : 0)));
const Fa = forward(alternate, tokens);
for (const key of ['E', 'Q', 'K', 'Sraw', 'S', 'A']) assert.deepEqual(Fa[key], F[key], key + ' is held fixed');
assert.equal(Fa.Mmsg[6][financeIndex], 0);
F.Mmsg[6].forEach((x,i)=>{if(i!==financeIndex)assert.equal(Fa.Mmsg[6][i],x);});
assert.deepEqual(Fa.Delta,matmul(Fa.Mmsg,toy.W_O),'map the changed message through the unchanged W_O');
toy.W_O[financeIndex].forEach((x,i)=>{if(x===0)assert.equal(Fa.Delta[6][i],F.Delta[6][i]);});
assert.notDeepEqual(Fa.probs[6], F.probs[6], 'a real changed value path changes the prediction');
assert.equal(JSON.stringify(toy), original, 'no model mutation');
assert.deepEqual(baseline(toy, toy.sentences.river).probs[9], baseline(toy, toy.sentences.cheque).probs[9]);

const names = ['sec08.html', 'sec12.html', 'sec14.html'];
const sections = names.map(name => read('sections/' + name));
for (let i = 0; i < sections.length; i++) {
  for (const match of sections[i].matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
    if (!match[1].includes('text/x-notes')) new vm.Script(match[2], { filename: names[i] });
  }
  assert(!/\\vq\{q[^}]*\}\^\\top\s*\\vk\{k/.test(sections[i]), 'no column-vector scoring remnant');
}

const require = createRequire(import.meta.url);
const playwright = require(process.env.ATTENTION_PLAYWRIGHT_PATH || '/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright');
const browser = await playwright.chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
const config = JSON.parse(read('part2.json'));
config.sections = config.sections.filter(section => ['s08', 's12', 's14'].includes(section.id));
const html = read('shell.html')
  .replace('<!--KATEX-->', () => read('katex-bundle.html'))
  .replace('<!--SHARED-->', () => '<script>window.__TOY__=' + JSON.stringify(toy) + ';window.__PART__=' + JSON.stringify(config) + ';</script><script>' + read('shared.js') + '</script>')
  .replace('<!--SECTIONS-->', () => sections.join('\n'));

try {
  await page.setContent(html, { waitUntil: 'load' });
  await page.waitForFunction(() => document.querySelectorAll('#s08-value-message table').length === 2);
  const bodyRows = selector => page.locator(selector + ' tbody tr').evaluateAll(rows => rows.map(row => Array.from(row.querySelectorAll('td')).map(cell => cell.textContent.trim())));
  const numberRows = async selector => (await bodyRows(selector)).map(row => row.map(value => Number(value.replaceAll('−', '-'))));
  const rounded = vector => vector.map(value => Number(value.toFixed(3)));

  const weightsBefore = await bodyRows('#s08-value-alpha');
  await page.click('#s08-values-alt');
  assert.deepEqual(await bodyRows('#s08-value-alpha'), weightsBefore, 'visible weights unchanged');
  let messageTables = await page.locator('#s08-value-message table').evaluateAll(tables => tables.map(table => Array.from(table.querySelectorAll('tbody tr')).map(row => Array.from(row.querySelectorAll('td')).map(cell => Number(cell.textContent.replaceAll('−', '-'))))));
  assert.deepEqual(messageTables[0][1], rounded(Fa.Mmsg[6]));
  assert.deepEqual(messageTables[1][1], rounded(Fa.Delta[6]));
  assert.match(await page.locator('#s08-value-result').innerText(), /finance message coordinate.*0\.000/);
  await page.click('#s08-values-baseline');
  messageTables = await page.locator('#s08-value-message table').evaluateAll(tables => tables.map(table => Array.from(table.querySelectorAll('tbody tr')).map(row => Array.from(row.querySelectorAll('td')).map(cell => Number(cell.textContent.replaceAll('−', '-'))))));
  assert.deepEqual(messageTables[0][1], rounded(F.Mmsg[6]));
  await page.click('#s08-runB');
  await page.waitForFunction(() => document.querySelector('#s08-bread').textContent.includes('Reading'));

  const illustrative = [];
  for (const dimension of [1, 16, 64]) {
    await page.locator('#s12-ctl input').evaluate((input, value) => { input.value = value; input.dispatchEvent(new Event('input', { bubbles: true })); }, String(dimension));
    illustrative.push(await numberRows('#s12-table'));
  }
  // This panel deliberately holds raw scores fixed. The separate simulation
  // below, not this divisor slider, demonstrates growth with dimension.
  const fixedRaw=[0,0.5,1.5,-0.5],rawProb=softmax(fixedRaw);
  [1,16,64].forEach((dimension,i)=>{
    const scaled=fixedRaw.map(x=>x/Math.sqrt(dimension)),scaledProb=softmax(scaled);
    const expected=fixedRaw.map((x,j)=>[x,Number(scaled[j].toFixed(2)),Number(rawProb[j].toFixed(3)),Number(scaledProb[j].toFixed(3))]);
    assert.deepEqual(illustrative[i],expected,'fixed scores, divisor, and both softmax columns are exact to displayed precision');
  });
  assert(illustrative[0][2][3] > illustrative[1][2][3] && illustrative[1][2][3] > illustrative[2][2][3], 'larger divisor flattens the fixed-score softmax');
  const varianceRows = await numberRows('#s12-sim');
  varianceRows.forEach(row => assert(row[1] > 0.8 && row[1] < 1.2, 'seeded scaled variance near 1'));

  const riverPrediction = await bodyRows('#s14-head');
  await page.click('#s14-ctx-cheque');
  assert.notDeepEqual(await bodyRows('#s14-head'), riverPrediction, 'attention responds to context');
  await page.click('#s14-toggle button');
  const offWeights = await bodyRows('#s14-alpha');
  assert(offWeights[0].slice(0,10).every(x=>x===''||x==='—'),'bypass must not show fabricated numeric weights');
  assert.equal(offWeights[0][10], 'not computed');
  assert.match(await page.locator('#s14-alpha').innerText(),/Attention bypassed: no weights are used/);
  const offPrediction = await bodyRows('#s14-head');
  await page.click('#s14-ctx-river');
  assert.deepEqual(await bodyRows('#s14-head'), offPrediction, 'bypass prediction does not depend on earlier context');
  assert.deepEqual((await numberRows('#s14-update'))[1], Array(toy.d_model).fill(0));
  await page.click('#s14-toggle button');
  assert.deepEqual(await bodyRows('#s14-head'), riverPrediction, 'toggle returns to exact original display');
  await page.click('#s14-head-calc button');
  assert.match(await page.locator('#s14-head-calc').innerText(), /logit for.*water/);
  assert(!/leads pos/.test(await page.locator('#s14-update').innerText()), 'position coordinate is not described as a semantic topic');

  const frameAudit = await page.evaluate(() => Array.from(document.querySelectorAll('section.sec')).map(section => ({
    id: section.id, count: section.querySelectorAll('.frame').length,
    notes: Array.from(section.querySelectorAll('.frame')).every(frame => frame.querySelector('script[type="text/x-notes"]')?.textContent.trim()),
    snippets: Array.from(section.querySelectorAll('pre.pytorch code')).every(code => code.textContent.trim().split('\n').length <= 4)
  })));
  frameAudit.forEach(section => { assert(section.count >= 1); assert(section.notes); assert(section.snippets); });
  const frameCounts = Object.fromEntries(frameAudit.map(section => [section.id, section.count]));
  const shotsIndex = process.argv.indexOf('--shots-dir');
  const shotsDir = shotsIndex < 0 ? null : path.resolve(process.argv[shotsIndex + 1]);
  if (shotsDir) assert(fs.statSync(shotsDir).isDirectory());
  await page.evaluate(() => AT.present.enter());
  for (const id of ['s08', 's12', 's14']) {
    for (let frame = 1; frame <= frameCounts[id]; frame++) {
      await page.evaluate(({ id, frame }) => { AT.present.go(id, frame, 99); }, { id, frame });
      assert.equal(await page.locator('.frame.is-live').count(), 1, 'one live frame');
      if (shotsDir) await page.screenshot({ path: path.join(shotsDir, `${id}-${frame}.png`) });
    }
  }
  await page.evaluate(() => AT.present.exit());
  await page.setViewportSize({ width: 390, height: 844 });
  const widthAudit = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth,
    wide: Array.from(document.querySelectorAll('section.sec *')).filter(el => {
      const box = el.getBoundingClientRect();
      return box.width > document.documentElement.clientWidth && !el.closest('.dt-scroll, .scroll-x');
    }).slice(0, 12).map(el => ({ tag: el.tagName, id: el.id, cls: el.className, width: el.getBoundingClientRect().width }))
  }));
  assert(widthAudit.page <= widthAudit.viewport + 1, 'no page-wide horizontal overflow on mobile: ' + JSON.stringify(widthAudit));
  assert.deepEqual(await page.locator('.katex-error').allTextContents(), []);
  assert.deepEqual(errors, []);
  console.log('PASS: exact value-only isolation; fixed-score divisor and softmax calculations; variance simulation; bypass vs self-attention UI; restore toggles; classroom frames; ≤2-line snippets; mobile width; no JS/KaTeX errors.');
  console.log(JSON.stringify({ bankAlpha: F.A[6], originalMessage: F.Mmsg[6], alternateMessage: Fa.Mmsg[6], originalDelta: F.Delta[6], alternateDelta: Fa.Delta[6], frames: frameAudit }, null, 2));
} finally {
  await browser.close();
}

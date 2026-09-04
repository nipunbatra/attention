// node src/check_vision3.mjs [--browser vision3.html]
// Companion reference: python3 src/train_vision3.py --check
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const toy = JSON.parse(readFileSync(new URL('toy7.json', import.meta.url), 'utf8'));
const source = readFileSync(new URL('part7.js', import.meta.url), 'utf8');
const baseline = JSON.stringify(toy), d = toy.clip;
const window = { __TOY__: toy, AT: { axes: {}, objects: [], notation: [], ui: { notationCard: x => x } } };
vm.runInNewContext(source, { window, console });
const C = window.AT.clip;
let scalars = 0, maximum = 0;
const clone = value => JSON.parse(JSON.stringify(value));
function close(actual, expected, label = 'value', tolerance = 1e-11) {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length, label); expected.forEach((v, i) => close(actual[i], v, label + '[' + i + ']', tolerance)); }
  else if (expected !== null && typeof expected === 'object') { for (const [k, v] of Object.entries(expected)) close(actual[k], v, label + '.' + k, tolerance); }
  else if (typeof expected === 'number') { assert(Number.isFinite(actual), label); const error = Math.abs(actual - expected); maximum = Math.max(maximum, error); assert(error < tolerance, `${label}: ${actual} != ${expected}`); scalars++; }
  else assert.equal(actual, expected, label);
}
const sum = row => row.reduce((a, b) => a + b, 0);
const transpose = rows => rows[0].map((_, j) => rows.map(row => row[j]));
let gradientChecks = 0, gradientError = 0;
for (const snapshot of ['initial', 'afterOne', 'trained']) {
  const p = C.params(snapshot), f = C.forward(p), g = C.gradients(p);
  close(f, d.reference[snapshot], snapshot);
  for (const rows of [f.imageUnit, f.textUnit]) rows.forEach(row => close(sum(row.map(x => x * x)), 1, 'unit norm'));
  for (const rows of [f.rowProb, transpose(f.columnProb)]) rows.forEach(row => { close(sum(row), 1, 'probability sum'); assert(row.every(x => x >= 0 && x <= 1)); });
  assert(f.cosine.flat().every(x => x >= -1 - 1e-12 && x <= 1 + 1e-12));
  close(f.loss, (f.rowLoss + f.columnLoss) / 2, 'symmetric mean');
  close(f.rowLoss, -sum(f.rowProb.map((row, i) => Math.log(row[i]))) / 3, 'row CE');
  close(f.columnLoss, -sum(f.columnProb.map((row, i) => Math.log(row[i]))) / 3, 'column CE');
  for (const check of d.gradientCheck.checks) {
    const key = check.parameter, [i, j] = check.index;
    const original = check.index.length ? p[key][i][j] : p[key];
    const set = v => { if (check.index.length) p[key][i][j] = v; else p[key] = v; };
    const eps = 1e-5;
    set(original + eps); const plus = C.forward(p).loss;
    set(original - eps); const minus = C.forward(p).loss;
    set(original);
    const analytic = check.index.length ? g[key][i][j] : g[key];
    const numerical = (plus - minus) / (2 * eps), error = Math.abs(analytic - numerical);
    gradientError = Math.max(gradientError, error);
    assert(error < 1e-7, snapshot + '.' + key + ' exact normalization gradient'); gradientChecks++;
  }
}
close(C.gradients(), d.initialGradients, 'independent NumPy gradients');
const first = C.params(), firstCopy = JSON.stringify(first);
close(C.step(first), d.snapshots.afterOne, 'first SGD update');
assert.equal(JSON.stringify(first), firstCopy, 'step must not mutate its input');
let trained = C.params();
for (let i = 0; i < d.trainedSteps; i++) { close(C.forward(trained).loss, d.history[i], 'training history'); trained = C.step(trained); }
close(trained, d.snapshots.trained, '60 simultaneous updates');
assert(C.forward(trained).loss < C.forward().loss);
assert(Math.abs(C.forward().rowLoss - C.forward().columnLoss) > .001, 'directions are distinct');

for (let imageIndex = 0; imageIndex < 3; imageIndex++) {
  let winner;
  for (const tau of [.05, .2, 1, 10]) {
    const r = C.classify({ imageIndex, tau });
    close(sum(r.probabilities), 1, 'candidate sum');
    if (winner == null) winner = r.best;
    assert.equal(r.best, winner, 'positive temperature preserves ranking');
    assert.equal(r.best, imageIndex, 'trained paired match');
  }
  const scaled = d.images[imageIndex].map(row => row.map(x => x * 7));
  close(C.classify({ image: scaled }).cosine, C.classify({ imageIndex }).cosine, 'positive brightness scaling in this linear toy');
  assert.equal(C.retrieve(d.captions[imageIndex]).order[0], imageIndex);
}
const r = C.classify(), duplicate = C.classify({ captions: d.captions.concat(d.captions[0]) });
close(duplicate.probabilities[0], duplicate.probabilities[3], 'identical candidates split equally');
close(duplicate.probabilities[0], r.probabilities[0] / (1 + r.probabilities[0]), 'duplicate changes denominator');
close(C.classify({ captions: ['horizontal stripes'] }).probabilities, [1], 'one wrong candidate still gets probability 1');
const missing = C.classify({ captions: d.captions.slice(1) });
assert.notEqual(missing.label, d.captions[0]); close(sum(missing.probabilities), 1);
const template = C.classify({ captions: d.captions.map(c => 'a grid with ' + c) });
assert(template.cosine.some((x, i) => Math.abs(x - r.cosine[i]) > 1e-4), 'template changes actual encoded vectors');
close(C.words(['two bright squares']), C.words(['bright squares two']), 'disclosed bag-of-words limitation');
close(C.unit([2, 1, 0]), [2 / Math.sqrt(5), 1 / Math.sqrt(5), 0]);
assert.throws(() => C.unit([0, 0, 0]));
for (const tau of [0, -1, NaN]) assert.throws(() => C.classify({ tau }));
assert.throws(() => C.classify({ captions: [] }));
assert.throws(() => C.classify({ captions: ['unseen word'] }));
assert.throws(() => C.classify({ image: [[1]] }));
assert.throws(() => C.step(C.params(), -1));
const bad = C.params(); bad.W_img[0][0] = NaN; assert.throws(() => C.forward(bad));
const p = C.params(); p.W_img[0][0] = 999; assert.notEqual(C.params().W_img[0][0], 999);
assert.equal(JSON.stringify(toy), baseline, 'readout and training do not mutate source data');
assert.equal(window.AT.axes.named, false);
assert.equal(window.AT.notation.filter(n => n.parts.includes('vision3')).length, 10);
const sections = Array.from({ length: 7 }, (_, i) => readFileSync(new URL(`sections7/sec0${i + 1}.html`, import.meta.url), 'utf8'));
assert.equal(sections.reduce((n, text) => n + (text.match(/class="frame"/g) || []).length, 0), 25);
assert(!sections.some(text => /[—–]/.test(text)), 'natural prose uses no em/en dashes');
console.log(`PASS: Vision III ${scalars} reference scalars (max error ${maximum}); ${gradientChecks} finite-difference gradients (max error ${gradientError}); training, normalization, row/column loss, prompt/candidate/temperature controls, source immutability, and 25-frame contract.`);

const browserFlag = process.argv.indexOf('--browser');
if (browserFlag >= 0) {
  const filename = process.argv[browserFlag + 1] || 'vision3.html';
  const require = createRequire(import.meta.url); let pw;
  for (const candidate of [process.env.PLAYWRIGHT_PATH, '/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright', '/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)) { try { pw = require(candidate); break; } catch {} }
  if (!pw) throw Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
  const b = await pw.chromium.launch();
  try {
    const page = await b.newPage({ viewport: { width: 1280, height: 900 } }), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('https://**/*', route => route.abort()); // The lesson must work offline.
    if (existsSync(filename)) await page.goto(pathToFileURL(path.resolve(filename)).href);
    else {
      // Exercise the same fragments in memory while the main build is pending.
      // This does not write or replace any assembled article.
      const read = name => readFileSync(new URL(name, import.meta.url), 'utf8');
      const inject = JSON.stringify(toy).replace(/<\//g, '<\\/');
      const part = JSON.stringify(JSON.parse(read('part7.json'))).replace(/<\//g, '<\\/');
      const scripts = `<script>window.__TOY__=${inject};window.__PART__=${part};</script><script>${read('shared.js')}</script><script>${source}</script>`;
      const html = read('shell.html').replace('<!--KATEX-->', () => read('katex-bundle.html')).replace('<!--SHARED-->', () => scripts).replace('<!--SECTIONS-->', () => sections.join('\n'));
      await page.setContent(html);
    }
    assert.deepEqual(errors, [], 'initial browser runtime errors');
    await page.waitForFunction(() => window.AT?.clip && document.querySelector('#s06-candidates table'));
    assert.deepEqual(errors, [], 'browser runtime errors');
    assert.equal(await page.locator('.frame').count(), 25);
    assert.equal(await page.locator('.katex-error').count(), 0, 'all lesson math renders');
    const bounds = await page.evaluate(() => {
      const failures = [];
      document.querySelectorAll('svg[data-clip-diagram]').forEach(svg => {
        const box = svg.viewBox.baseVal;
        svg.querySelectorAll('text').forEach(text => { const r = text.getBBox(); if (r.x < -1 || r.y < -1 || r.x + r.width > box.width + 1 || r.y + r.height > box.height + 1) failures.push(svg.dataset.clipDiagram + ': ' + text.textContent); });
      });
      return failures;
    });
    assert.deepEqual(bounds, [], 'SVG labels within viewBox');
    const status = page.locator('#s05-widget p[aria-live]');
    await page.locator('#s05-widget button', { hasText: 'One SGD step' }).click();
    assert.match(await status.textContent(), /Step 1: loss 0\.5088/);
    await page.locator('#s05-widget button', { hasText: 'Reset' }).click();
    assert.match(await status.textContent(), /Step 0: loss 0\.5780/);
    const choices = page.locator('#s06-candidates select[aria-label="Candidate caption set"]');
    await choices.selectOption('duplicate');
    assert.equal(await page.locator('#s06-candidates tbody tr').count(), 4);
    await choices.selectOption('missing');
    assert.equal(await page.locator('#s06-candidates tbody tr').count(), 2);
    await page.locator('#s06-candidates select[aria-label="Toy prompt template"]').selectOption('grid');
    assert.match(await page.locator('#s06-candidates').textContent(), /a grid with horizontal stripes/);
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.evaluate(() => AT.present.enter());
    const fit = async label => {
      await page.waitForTimeout(80);
      const report = await page.evaluate(() => AT.present.fitReport());
      assert.equal(report.overflow, false, label + ': ' + JSON.stringify(report));
      const scrolls = await page.evaluate(() => [...document.querySelectorAll('.frame.is-live *')].filter(el => {
        const css = getComputedStyle(el);
        return el.getClientRects().length && ((/(auto|scroll)/.test(css.overflowX) && el.scrollWidth > el.clientWidth + 2) || (/(auto|scroll)/.test(css.overflowY) && el.scrollHeight > el.clientHeight + 2));
      }).map(el => el.id || el.className));
      assert.deepEqual(scrolls, [], label + ': presentation must not scroll');
    };
    await page.evaluate(() => AT.present.go('s06', 3, 0));
    for (const template of ['bare', 'grid']) for (const candidates of ['all', 'missing', 'duplicate']) for (const tau of ['0.05', '1']) {
      await page.evaluate(({ template, candidates, tau }) => {
        for (const [label, value] of [['Toy prompt template', template], ['Candidate caption set', candidates], ['Inference temperature', tau]]) {
          const el = document.querySelector('#s06-candidates [aria-label="' + label + '"]'); el.value = value; el.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }, { template, candidates, tau });
      await fit('candidate controls ' + [template, candidates, tau].join('/'));
    }
    await page.evaluate(() => AT.present.go('s05', 3, 0));
    for (const n of [0, 1, 20, 200]) {
      await page.evaluate(n => {
        const buttons = [...document.querySelectorAll('#s05-widget button')]; buttons.find(b => b.textContent === 'Reset').click();
        const button = buttons.find(b => b.textContent === (n === 1 ? 'One SGD step' : '20 steps'));
        for (let i = 0; i < (n === 1 ? 1 : n / 20); i++) button.click();
      }, n);
      await fit('training controls step ' + n);
    }
    await page.evaluate(() => AT.present.exit());
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(100);
    const phone = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth, errors: document.querySelectorAll('.katex-error').length }));
    assert.ok(phone.page <= phone.viewport + 1, 'phone article must not overflow the page: ' + JSON.stringify(phone));
    assert.deepEqual(errors, []);
    console.log('PASS: offline assembled browser, SVG label bounds, math, 12 candidate-control layouts, 4 training-control layouts, and 390px phone article width.');
  } finally { await b.close(); }
}

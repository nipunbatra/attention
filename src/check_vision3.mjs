// node src/check_vision3.mjs [--browser vision3.html]
// Companion reference: python3 src/train_vision3.py --check
// Runs vision-shared.js and part7.js on a stub window (no DOM work is needed for the numbers) and checks every
// number the page shows against toy7.json, the finite-difference gradients, the training checkpoints, the frozen
// encoder, the zero-shot and batch probes, the named-axis promises and the section contract. With --browser it also
// loads the assembled page offline and checks frames, math, interactive fit and the phone width.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const read = name => readFileSync(new URL(name, import.meta.url), 'utf8');
const toy = JSON.parse(read('toy7.json')), d = toy.clip, baseline = JSON.stringify(toy);

/* ---- a stub window: enough for the two scripts to load; the numbers never touch the DOM ---- */
function el() { const e = { children: [], style: {}, classList: { add() {}, remove() {}, toggle() {} }, appendChild(c) { this.children.push(c); return c; }, insertBefore(c) { this.children.unshift(c); return c; }, setAttribute() {}, querySelector() { return null; }, addEventListener() {}, innerHTML: '' }; return e; }
const AT = { h: () => el(), svg: () => el(), fmt: (x, k) => Number(x).toFixed(k == null ? 2 : k), escape: s => String(s), axes: {}, objects: [], notation: [], ui: { notationCard: x => x }, reducedMotion: () => true };
const window = { AT, __TOY__: toy, __PART__: JSON.parse(read('part7.json')), matchMedia: () => ({ matches: true }) };
const document = { head: el(), body: el(), getElementById: () => null, createElement: () => el() };
const ctx = { window, document, console, Math, JSON, Number, Array, Object, String, Error, setInterval, clearInterval, setTimeout, clearTimeout };
vm.runInNewContext(read('vision-shared.js'), ctx);
vm.runInNewContext(read('part7.js'), ctx);
const V = AT.vision, C = AT.clip;

let scalars = 0, maximum = 0;
function close(actual, expected, label = 'value', tolerance = 1e-9) {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length, label + ' length'); expected.forEach((v, i) => close(actual[i], v, label + '[' + i + ']', tolerance)); }
  else if (expected !== null && typeof expected === 'object') { for (const [k, v] of Object.entries(expected)) close(actual[k], v, label + '.' + k, tolerance); }
  else if (typeof expected === 'number') { assert(Number.isFinite(actual), label + ' finite'); const err = Math.abs(actual - expected); maximum = Math.max(maximum, err); assert(err < tolerance, `${label}: ${actual} != ${expected}`); scalars++; }
  else assert.equal(actual, expected, label);
}
const sum = a => a.reduce((x, y) => x + y, 0);
const argmax = a => a.indexOf(Math.max(...a));
const N = d.trainPairs;

/* 1. the frozen encoder is Vision I's: CLS rows and menu probabilities for every scene */
for (const [scene, row] of Object.entries(d.clsRows)) {
  close(C.cls(scene), row, 'CLS row ' + scene);
  close(V.frozen(scene).probs, d.frozen[scene].probs, 'frozen menu ' + scene);
  assert.equal(V.encode(scene).length, 17, 'seventeen updated rows');
}
close(toy.encoder, JSON.parse(read('toy5.json')).trained, 'encoder is toy5 trained');

/* 2. forward passes at the three snapshots, with the invariants the page relies on */
for (const name of ['initial', 'afterOne', 'trained']) {
  const p = C.params(name), f = C.forward(p);
  close(f, d.reference[name], name);
  for (const rows of [f.imgUnit, f.txtUnit]) rows.forEach(r => close(sum(r.map(x => x * x)), 1, name + ' unit norm'));
  f.rowProb.forEach(r => close(sum(r), 1, name + ' row sum'));
  f.colProb[0].forEach((_, j) => close(sum(f.colProb.map(r => r[j])), 1, name + ' column sum'));
  close(f.loss, (f.rowLoss + f.colLoss) / 2, name + ' symmetric mean');
  close(f.rowLoss, -sum(f.rowProb.map((r, i) => Math.log(r[i]))) / N, name + ' row CE');
  close(f.colLoss, -sum(f.colProb.map((r, i) => Math.log(r[i]))) / N, name + ' column CE');
  assert(f.cosine.flat().every(x => x >= -1 - 1e-12 && x <= 1 + 1e-12));
}

/* 3. gradients: analytic equals the NumPy reference and central differences for all 55 parameters */
const p0 = C.params('initial'), g = C.gradients(p0);
close({ W_img: g.W_img, W_txt: g.W_txt, log_scale: g.log_scale }, d.initialGradients, 'independent NumPy gradients');
let gradientChecks = 0, gradientError = 0;
for (const check of d.gradientCheck.checks) {
  const key = check.parameter, idx = check.index, p = C.params('initial'), eps = 1e-5;
  const get = () => idx.length ? p[key][idx[0]][idx[1]] : p[key], set = v => { if (idx.length) p[key][idx[0]][idx[1]] = v; else p[key] = v; };
  const original = get();
  set(original + eps); const plus = C.forward(p).loss; set(original - eps); const minus = C.forward(p).loss; set(original);
  const numerical = (plus - minus) / (2 * eps), analytic = idx.length ? g[key][idx[0]][idx[1]] : g[key];
  const err = Math.abs(analytic - numerical); gradientError = Math.max(gradientError, err);
  assert(err < 1e-7, key + ' ' + idx + ' gradient'); gradientChecks++;
  close(analytic, check.analytic, key + ' saved analytic');
}
assert.equal(gradientChecks, 55);
const zeroWords = ['one', 'and', 'photo', 'of', 'the', 'mug'];
zeroWords.forEach(w => close(g.W_txt[d.vocab.indexOf(w)], [0, 0, 0], 'no gradient for an unseen word ' + w, 1e-15));

/* 4. sixty simultaneous steps reproduce every checkpoint, the history and the trajectory */
const first = C.params(), firstCopy = JSON.stringify(first);
close(C.step(first), d.snapshots.afterOne, 'first step');
assert.equal(JSON.stringify(first), firstCopy, 'step must not mutate its input');
let q = C.params();
for (let t = 0; t <= d.steps; t++) {
  close(q, d.checkpoints[t], 'checkpoint ' + t, 1e-9);
  const f = C.forward(q);
  close(f.loss, d.history[t], 'history ' + t);
  close(f.imgUnit, d.trajectory[t].img, 'trajectory img ' + t); close(f.txtUnit, d.trajectory[t].txt, 'trajectory txt ' + t);
  close(C.state(t).loss, d.history[t], 'state loss ' + t);
  if (t < d.steps) q = C.step(q);
}
close(q, d.snapshots.trained, 'trained snapshot');
assert(d.history.every((l, i) => i === 0 || l < d.history[i - 1]), 'loss falls at every step');
assert(C.forward(C.params('trained')).tau < C.forward(C.params('initial')).tau, 'temperature sharpened');

/* 5. the named axes are true: each image and each object word points most to its own axis, before and after */
for (const name of ['initial', 'trained']) {
  const f = C.forward(C.params(name));
  f.imgUnit.forEach((u, i) => assert.equal(argmax(u), i, name + ': image ' + i + ' argmax is its own axis'));
  f.txtUnit.forEach((u, i) => assert.equal(argmax(u), i, name + ': caption ' + i + ' argmax is its own axis'));
  f.cosine.forEach((r, i) => assert.equal(argmax(r), i, name + ': diagonal wins row ' + i));
  const p = C.params(name);
  [['mug', 0], ['mugs', 0], ['book', 1], ['plant', 2]].forEach(([w, k]) => assert.equal(argmax(p.W_txt[d.vocab.indexOf(w)]), k, name + ': row for ' + w));
}
const three = C.cls('A'), rows3 = ['A', 'D', 'E'].map(C.cls);
(function collinear() { // the frozen encoder's CLS rows lie on one line (reported honestly on the page)
  const dir = rows3[0].map((x, k) => x - rows3[2][k]), mid = rows3[1].map((x, k) => x - rows3[2][k]);
  const t = sum(mid.map((x, k) => x * dir[k])) / sum(dir.map(x => x * x));
  const residual = Math.sqrt(sum(mid.map((x, k) => (x - t * dir[k]) ** 2)));
  assert(residual < 1e-3, 'CLS rows are collinear up to ' + residual);
})();

/* 6. zero-shot and batch probes match the saved references, plus the candidate-list invariants */
for (const z of d.zeroShot) {
  const r = C.classify({ scene: z.scene, captions: z.captions, params: C.params('trained'), tau: z.tau });
  close(r.cosine, z.cosine, 'zero-shot cosine ' + z.scene); close(r.probs, z.probs, 'zero-shot probs ' + z.scene); assert.equal(r.best, z.best);
  close(sum(r.probs), 1, 'candidate sum');
}
for (const b of d.batches) { const f = C.batch(b.n); close(f.loss, b.loss, 'batch loss ' + b.n); close(f.cosine, b.cosine, 'batch cosine ' + b.n); close(Math.log(b.n), b.collapse, 'collapse'); }
const trained = C.params('trained');
for (const scene of ['A', 'D', 'E']) { let winner = null; for (const tau of [0.02, 0.2, 1, 10]) { const r = C.classify({ scene, tau, params: trained }); if (winner == null) winner = r.best; assert.equal(r.best, winner, 'temperature keeps the ranking'); } }
const dup = C.classify({ scene: 'A', captions: ['two mugs', 'a book', 'two mugs'], params: trained });
close(dup.probs[0], dup.probs[2], 'identical candidates split equally');
close(C.classify({ scene: 'A', captions: ['a plant'], params: trained }).probs, [1], 'one candidate gets probability 1');
assert.notEqual(C.classify({ scene: 'A', captions: ['a book', 'a plant'], params: trained }).label, 'two mugs on a table');
close(C.words(['two mugs on a table']), C.words(['a table on two mugs']), 'bag of words ignores order');
const collapsed = C.params('initial'); collapsed.W_img = [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]; collapsed.W_txt = d.vocab.map(() => [1, 1, 1]);
close(C.forward(collapsed).loss, Math.log(3), 'collapsed model scores log N');
assert.throws(() => C.classify({ scene: 'A', captions: ['unseen word'], params: trained }));
assert.throws(() => C.classify({ scene: 'A', captions: [], params: trained }));
for (const tau of [0, -1, NaN]) assert.throws(() => C.classify({ scene: 'A', tau, params: trained }));
assert.throws(() => C.step(C.params(), -1));
assert.throws(() => C.params('nope'));
const bad = C.params(); bad.W_img[0][0] = NaN; assert.throws(() => C.forward(bad));
const mut = C.params(); mut.W_img[0][0] = 999; assert.notEqual(C.params().W_img[0][0], 999, 'params are copies');
assert.equal(JSON.stringify(toy), baseline, 'nothing mutates the source data');
assert.equal(AT.notation.filter(n => n.parts.includes('vision3')).length, 13, 'notation rows');
assert(AT.axes.named !== false);

/* 7. the section contract: nine fragments, every frame titled, noted and built, no dashes, no anonymous coordinates */
const sections = Array.from({ length: 9 }, (_, i) => read(`sections7/sec0${i + 1}.html`));
const frames = sections.reduce((n, t) => n + (t.match(/class="frame"/g) || []).length, 0);
assert.equal(frames, 44, 'frame count');
sections.forEach((t, i) => {
  const blocks = t.split('class="frame"').slice(1);
  blocks.forEach((b, k) => { const head = b.slice(0, 200); assert(/data-title="[^"]+"/.test(head), `sec0${i + 1} frame ${k + 1} has a title`); assert(/text\/x-notes/.test(b), `sec0${i + 1} frame ${k + 1} has notes`); assert(/data-build="\d"/.test(b), `sec0${i + 1} frame ${k + 1} has a build`); });
  assert(!/[—–]/.test(t), `sec0${i + 1} uses no em or en dash`);
  assert(!/coordinate \d/i.test(t), `sec0${i + 1} names every axis`);
  assert(/class="companion"|companion scope|companion after/.test(t), `sec0${i + 1} has companion prose`);
});
assert(!/[—–]/.test(read('part7.js')), 'part7.js uses no em or en dash in its strings');
console.log(`PASS: Vision III ${scalars} reference scalars (max error ${maximum}); ${gradientChecks} finite-difference gradients (max error ${gradientError}); frozen encoder, 61 checkpoints, trajectory, named axes, zero-shot and batch probes, invariants, and the 44-frame contract.`);

/* ---- optional: the assembled page in headless Chromium, offline ---- */
const browserFlag = process.argv.indexOf('--browser');
if (browserFlag >= 0) {
  const filename = process.argv[browserFlag + 1] || 'vision3.html';
  const require = createRequire(import.meta.url); let pw;
  for (const candidate of [process.env.PLAYWRIGHT_PATH, '/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)) { try { pw = require(candidate); break; } catch {} }
  if (!pw) throw Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
  assert(existsSync(filename), 'build the page first: python3 src/assemble.py --part 7 --out vision3.html');
  const b = await pw.chromium.launch();
  try {
    const page = await b.newPage({ viewport: { width: 1280, height: 900 } }), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text()); });
    await page.route('https://**/*', route => route.abort());
    await page.goto(pathToFileURL(path.resolve(filename)).href);
    await page.waitForFunction(() => window.AT && window.AT.clip && document.querySelector('#s08-table table'));
    assert.deepEqual(errors, [], 'browser runtime errors');
    assert.equal(await page.locator('.frame').count(), 44);
    assert.equal(await page.locator('.katex-error').count(), 0, 'all math renders');
    assert.equal(await page.locator('section.sec').count(), 9);
    const text = await page.evaluate(() => document.querySelector('main').innerText);
    assert(!/coordinate \d/i.test(text), 'no anonymous coordinate on the page');
    assert(!/NaN|undefined|Infinity/.test(text), 'no bad text');
    // the page's own numbers agree with this script's runtime
    const shown = await page.evaluate(() => ({ loss0: AT.clip.forward(AT.clip.params('initial')).loss, lossT: AT.clip.forward(AT.clip.params('trained')).loss, cls: AT.clip.cls('A') }));
    close(shown.loss0, d.reference.initial.loss, 'page initial loss'); close(shown.lossT, d.reference.trained.loss, 'page trained loss'); close(shown.cls, d.clsRows.A, 'page CLS row');
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.evaluate(() => AT.present.enter());
    const fit = async label => {
      await page.waitForTimeout(80);
      const report = await page.evaluate(() => AT.present.fitReport());
      assert.equal(report.overflow, false, label + ': ' + JSON.stringify(report));
    };
    await page.evaluate(() => AT.present.go('s06', 2, 0));
    for (const t of [0, 1, 30, 60]) { await page.evaluate(t => { const r = document.querySelector('#s06-slider input[type=range]'); r.value = t; r.dispatchEvent(new Event('input', { bubbles: true })); }, t); await fit('training slider ' + t); }
    await page.evaluate(() => AT.present.go('s07', 1, 0));
    for (const n of [3, 5, 8]) { await page.evaluate(n => { const r = document.querySelector('#s07-slider input[type=range]'); r.value = n; r.dispatchEvent(new Event('input', { bubbles: true })); }, n); await fit('batch slider ' + n); }
    await page.evaluate(() => AT.present.go('s08', 1, 0));
    for (let s = 0; s < 5; s++) for (let l = 0; l < 4; l++) { await page.evaluate(([s, l]) => { document.querySelectorAll('#s08-scene .chip')[s].click(); document.querySelectorAll('#s08-list .btn')[l].click(); }, [s, l]); await fit('zero-shot ' + s + '/' + l); }
    await page.evaluate(() => AT.present.go('s08', 5, 0));
    for (const tau of ['0.02', '0.5', '1']) { await page.evaluate(tau => { const r = document.querySelector('#s08-tau input[type=range]'); r.value = tau; r.dispatchEvent(new Event('input', { bubbles: true })); }, tau); await fit('temperature ' + tau); }
    await page.evaluate(() => AT.present.exit());
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(150);
    const phone = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
    assert(phone.page <= phone.viewport + 1, 'phone article must not overflow: ' + JSON.stringify(phone));
    assert.deepEqual(errors, []);
    console.log('PASS: offline assembled page, 44 frames, math, training/batch/zero-shot/temperature control fit in presentation, and 390px width.');
  } finally { await b.close(); }
}

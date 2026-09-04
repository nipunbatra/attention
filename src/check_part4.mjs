// Numerical and diagram contracts for the separate Part IV translation toy.
// Run: node src/check_part4.mjs [--browser]
// Reproduce training and its independent gradient reference with:
//   python3 src/train_part4.py --check
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const toy = JSON.parse(readFileSync(new URL('toy4.json', import.meta.url), 'utf8'));
const runtime = readFileSync(new URL('part4.js', import.meta.url), 'utf8');
const data = toy.translation;
const unchanged = JSON.stringify(toy);

// A small DOM stand-in checks diagram content without a browser dependency.
class Element {
  constructor(tag) { this.tagName = tag; this.attributes = {}; this.children = []; this._text = ''; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  get id() { return this.attributes.id; }
  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(' '); }
  get firstChild() { return this.children[0]; }
  appendChild(child) { this.children.push(child); return child; }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
}
const document = { createElementNS: (_, tag) => new Element(tag) };
const window = { __TOY__: toy, AT: { axes: { named: true }, objects: [], notation: [] } };
vm.runInNewContext(runtime, { window, document, console });
const T = window.AT.translation;
let comparisons = 0, maximumError = 0;
function compare(actual, expected, name = 'value', tolerance = 1e-11) {
  if (Array.isArray(expected)) {
    assert(Array.isArray(actual), name);
    assert.equal(actual.length, expected.length, name);
    expected.forEach((value, i) => compare(actual[i], value, `${name}[${i}]`, tolerance));
  } else if (expected !== null && typeof expected === 'object') {
    for (const [key, value] of Object.entries(expected)) compare(actual[key], value, `${name}.${key}`, tolerance);
  } else if (expected === null) {
    assert.equal(actual, -Infinity, `${name}: null is reserved for a masked score`);
  } else if (typeof expected === 'number') {
    assert(Number.isFinite(actual), name);
    const error = Math.abs(actual - expected);
    maximumError = Math.max(maximumError, error); comparisons++;
    assert(error <= tolerance, `${name}: ${actual} != ${expected} (error ${error})`);
  } else assert.equal(actual, expected, name);
}
const clone = value => JSON.parse(JSON.stringify(value));
const sum = row => row.reduce((a, b) => a + b, 0);
const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
const mm = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((s, x, k) => s + x * b[k][j], 0)));
const add = (a, b) => a.map((row, i) => row.map((x, j) => x + b[i][j]));
function shape(values, rows, columns, name) {
  assert.equal(values.length, rows, name);
  for (const row of values) { assert.equal(row.length, columns, name); assert(row.every(Number.isFinite), name); }
}
function normalized(rows, name) {
  rows.forEach((row, i) => {
    assert(row.every(x => x >= 0 && x <= 1 && Number.isFinite(x)), name);
    compare(sum(row), 1, `${name}[${i}] sum`);
  });
}
for (const snapshot of ['before', 'after']) {
  const f = T.forward(T.targetInput, { snapshot });
  const p = T.parameters(snapshot);
  compare(f, data.reference[snapshot], snapshot);
  shape(f.sourceRows, 4, 3, 'English rows'); shape(f.targetRows, 3, 3, 'French rows');
  compare(f.sourceRows, add(f.sourceLookup, f.sourcePositions), 'add source position');
  compare(f.targetRows, add(f.targetLookup, f.targetPositions), 'add target position');
  for (const [name, block, queries, sources, causal] of [
    ['enc', f.encoder, 4, 4, false], ['dec', f.decoderSelf, 3, 3, true], ['cross', f.cross, 3, 4, false]
  ]) {
    shape(block.Q, queries, 3, `${name}.Q`); shape(block.K, sources, 3, `${name}.K`);
    shape(block.V, sources, 3, `${name}.V`); shape(block.A, queries, sources, `${name}.A`);
    compare(block.Q, mm(block.input, p[name + '_W_Q']), name + ' Q provenance');
    compare(block.K, mm(block.memory, p[name + '_W_K']), name + ' K provenance');
    compare(block.V, mm(block.memory, p[name + '_W_V']), name + ' V provenance');
    compare(block.Sraw, mm(block.Q, transpose(block.K)), name + ' raw scores');
    compare(block.Sfull, block.Sraw.map(row => row.map(x => x / Math.sqrt(3))), name + ' scaling');
    compare(block.Mmsg, mm(block.A, block.V), name + ' weighted message');
    compare(block.Delta, mm(block.Mmsg, p[name + '_W_O']), name + ' output projection');
    compare(block.Enew, add(block.input, block.Delta), name + ' residual');
    normalized(block.A, name + ' weights');
    block.S.forEach((row, i) => row.forEach((value, j) => {
      if (causal && j > i) { assert.equal(value, -Infinity); assert.equal(block.A[i][j], 0); }
      else assert(Number.isFinite(value));
    }));
  }
  compare(f.cross.input, f.decoderSelf.Enew, 'cross query input');
  compare(f.cross.memory, f.encoder.Enew, 'cross source memory');
  normalized(f.probs, 'vocabulary probabilities');
  const trained = T.teacherForced({ snapshot });
  compare(trained.meanLoss, data.update[snapshot === 'before' ? 'lossBefore' : 'lossAfter'], 'mean loss');
  compare(trained.perPosition.map(row => row.loss), data.update[snapshot === 'before' ? 'perPositionBefore' : 'perPositionAfter'], 'row losses');
  compare(trained.meanLoss, sum(trained.perPosition.map(row => -Math.log(row.probability))) / 3, 'cross entropy');
  for (let length = 1; length <= 3; length++) {
    const short = T.forward(T.targetInput.slice(0, length), { snapshot });
    compare(short.probs, f.probs.slice(0, length), 'prefix causality');
    compare(short.cross.K, f.cross.K, 'fixed source K'); compare(short.cross.V, f.cross.V, 'fixed source V');
  }
  const changedFuture = T.forward(['<bos>', 'la', 'banque'], { snapshot });
  compare(changedFuture.probs.slice(0, 2), f.probs.slice(0, 2), 'no future French leakage');
  for (const [label, source, desired] of [
    ['river', T.source, ['la', 'rive', '<eos>']], ['financial', T.sourceContrast, ['la', 'banque', '<eos>']]
  ]) {
    const g = T.generate({ snapshot, source });
    compare(g, data.reference.generation[snapshot][label], label + ' generation');
    compare(g.tokens, desired, label + ' toy prediction');
    assert.equal(g.stoppedBy, 'eos'); assert.equal(g.encoderEvaluations, 1);
    g.trace.forEach((step, i) => {
      assert.equal(step.prefix.length, i + 1);
      assert.equal(step.chosenId, step.probabilities.indexOf(Math.max(...step.probabilities)));
      compare(step.prefix, ['<bos>', ...g.tokens.slice(0, i)], 'generated prefix, not teacher forcing');
    });
  }
}
const river = T.forward(['<bos>', 'la']);
const finance = T.forward(['<bos>', 'la'], { source: T.sourceContrast });
const riveId = T.targetVocab.indexOf('rive');
compare(river.cross.Q, finance.cross.Q, 'same decoder query for fixed prefix');
assert(river.probs[1][riveId] - finance.probs[1][riveId] > 0.7, 'prediction must depend meaningfully on source');
assert.equal(T.generate({ maxTokens: 1 }).stoppedBy, 'limit');
assert.equal(T.generate({ maxTokens: 1 }).trace.length, 1);

// Check every saved parameter update, and independently finite-difference every
// scalar with the JavaScript forward pass against the NumPy analytic gradient.
let parameterCount = 0, gradientMaximumError = 0;
const before = T.parameters('before'), after = T.parameters('after'), epsilon = 1e-5;
for (const [name, value] of Object.entries(before)) {
  const matrixParameter = Array.isArray(value[0]);
  const rows = matrixParameter ? value : [value];
  rows.forEach((row, i) => row.forEach((initial, j) => {
    const gradient = matrixParameter ? data.update.gradients[name][i][j] : data.update.gradients[name][j];
    const updated = matrixParameter ? after[name][i][j] : after[name][j];
    compare(updated, initial - data.update.learningRate * gradient, `${name} SGD update`);
    row[j] = initial + epsilon;
    const plus = T.teacherForced({ parameters: before }).meanLoss;
    row[j] = initial - epsilon;
    const minus = T.teacherForced({ parameters: before }).meanLoss;
    row[j] = initial;
    const numeric = (plus - minus) / (2 * epsilon);
    gradientMaximumError = Math.max(gradientMaximumError, Math.abs(numeric - gradient));
    assert(Math.abs(numeric - gradient) < 1e-7, `${name}[${i},${j}] gradient`);
    parameterCount++;
  }));
}
assert.equal(parameterCount, 188);
assert.equal(sum(T.parameterRows().map(row => row.count)), 188);
assert(T.comparison().after.meanLoss < T.comparison().before.meanLoss);
// Do not claim every target improves: the visible update is river-only.
assert(T.comparison().after.perPosition[2].loss > T.comparison().before.perPosition[2].loss);

for (const bad of [[], ['la'], ['<bos>', 'unknown'], Array(7).fill('<bos>')]) assert.throws(() => T.forward(bad));
assert.throws(() => T.forward(undefined, { source: [] }));
assert.throws(() => T.forward(undefined, { source: ['unknown'] }));
assert.throws(() => T.forward(undefined, { snapshot: 'missing' }));
assert.throws(() => T.generate({ maxTokens: 0 }));
assert.throws(() => T.teacherForced({ targets: ['la'] }));
const invalid = T.parameters(); invalid.cross_W_Q[0][0] = NaN;
assert.throws(() => T.forward(undefined, { parameters: invalid }));
invalid.cross_W_Q = [[0]]; assert.throws(() => T.forward(undefined, { parameters: invalid }));
const disposable = T.parameters(); disposable.E_src[0][0] = 999;
assert.notEqual(T.parameters().E_src[0][0], 999);
assert.equal(JSON.stringify(toy), unchanged, 'runtime must not mutate saved model');
assert.equal(window.AT.axes.named, false);
assert.equal(window.AT.notation.filter(item => item.parts.includes('part4')).length, 22);
assert(!window.AT.notation.some(item => /src\},0|tgt\},0/.test(item.sym)));

function descendants(element) { return [element, ...element.children.flatMap(descendants)]; }
const stages = ['tokens', 'masks', 'provenance', 'encoder', 'decoder-self', 'cross', 'head', 'training', 'generation'];
const allIds = new Set();
for (const stage of stages) {
  const host = new Element('div'), svg = T.diagram(host, stage);
  const bounds = svg.getAttribute('viewBox').split(' ').map(Number);
  assert(bounds[2] <= 1100 && bounds[3] <= 350, stage + ' diagram size');
  assert.equal(svg.getAttribute('role'), 'img');
  assert(descendants(svg).some(node => node.tagName === 'title' && node.textContent));
  assert(descendants(svg).some(node => node.tagName === 'desc' && node.textContent));
  for (const node of descendants(svg)) if (node.id) { assert(!allIds.has(node.id), 'unique SVG IDs'); allIds.add(node.id); }
  if (stage === 'encoder') assert.equal(bounds[3], 180);
  if (stage === 'training') assert(svg.textContent.includes('Gradients reach both embedding tables'));
  if (stage === 'masks') {
    const cells = descendants(svg).filter(node => node.getAttribute('data-allowed'));
    assert.equal(cells.filter(node => node.getAttribute('data-mask-kind') === 'Cross-attention').length, 12);
    assert.equal(cells.filter(node => node.getAttribute('data-allowed') === 'false').length, 3);
  }
}
for (let step = 0; step < 3; step++) {
  const svg = T.diagram(new Element('div'), 'generation', { prefixTokens: T.targetInput.slice(0, step + 1) });
  const tokens = descendants(svg).filter(node => node.tagName === 'text').map(node => node._text);
  assert.equal(svg.getAttribute('viewBox'), '0 0 1100 170');
  assert.equal(tokens.filter(text => text.startsWith('p = ')).length, 1, 'only the requested generation row');
  assert.equal(svg.textContent.includes('The model predicted <eos>; stop.'), step === 2);
}
for (const step of [0, 1]) {
  const svg = T.diagram(new Element('div'), 'generation', { step });
  assert(!svg.textContent.includes('The model predicted <eos>; stop.'));
}
console.log(`PASS: Part IV references, masks, shapes, source dependence, causal prefixes, generation, ${parameterCount} updates/gradients, and all SVG stages. ${comparisons} reference scalars; max error ${maximumError}; gradient max error ${gradientMaximumError}.`);

if (process.argv.includes('--browser')) {
  const require = createRequire(import.meta.url);
  let pw;
  for (const candidate of [process.env.PLAYWRIGHT_PATH,
    '/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright',
    '/Users/nipun/.npm/_npx/360550e4913b8759/node_modules/playwright'].filter(Boolean)) {
    try { pw = require(candidate); break; } catch {}
  }
  if (!pw) throw new Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');
  const browser = await pw.chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1140, height: 900 } });
    await page.setContent('<style>:root{--font-ui:Arial,sans-serif;--ink:#20272c;--ink-3:#606a70;--c-e:#658169;--c-q:#2563ae;--c-k:#137965;--c-v:#ce7100;--c-a:#8462b1;--c-d:#128869;--card:#fff;--paper:#fff;--line:#ddd;--warn:#ad5011;--t-e:#f0f5f0;--t-q:#edf4fd;--t-k:#eff6f3;--t-v:#fff5e6;--t-a:#f5f1fa;--t-d:#edf8f2}body{margin:20px}svg{width:1100px;display:block;margin-bottom:24px}</style><main></main>');
    await page.evaluate(value => { window.__TOY__ = value; window.AT = { axes: {}, objects: [], notation: [] }; }, clone(toy));
    await page.addScriptTag({ content: runtime });
    const measured = await page.evaluate(stages => {
      const failures = [], result = [];
      for (const stage of stages) {
        const host = document.createElement('div'); document.querySelector('main').appendChild(host);
        const svg = AT.translation.diagram(host, stage), view = svg.viewBox.baseVal;
        for (const text of svg.querySelectorAll('text')) {
          const b = text.getBBox();
          if (b.x < -1 || b.y < -1 || b.x + b.width > view.width + 1 || b.y + b.height > view.height + 1) failures.push(stage + ': ' + text.textContent);
        }
        result.push({ stage, width: view.width, height: view.height });
      }
      return { failures, result };
    }, stages);
    assert.deepEqual(measured.failures, [], 'SVG labels must fit their view boxes');
    console.log('PASS: browser SVG label bounds: ' + JSON.stringify(measured.result));
  } finally { await browser.close(); }
}

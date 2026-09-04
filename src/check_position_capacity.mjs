// Read-only capacity and generation regression for the assembled Parts 2 and 3.
// Usage: node src/check_position_capacity.mjs [attention.html] [part3.html]
// Uses an existing Playwright installation; no dependencies or output files are written.
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { baseline, forward } from './toy_ref.mjs';

const src = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const candidates = [
  process.env.PLAYWRIGHT_MODULE,
  process.env.PLAYWRIGHT_PATH,
  'playwright',
  'playwright-core',
  '/Users/nipun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
].filter(Boolean);
const npmCache = path.join(os.homedir(), '.npm', '_npx');
if (existsSync(npmCache)) {
  for (const dir of readdirSync(npmCache).sort()) candidates.push(path.join(npmCache, dir, 'node_modules', 'playwright'));
}
let pw;
for (const candidate of candidates) {
  try { pw = require(candidate); break; } catch { /* Try another existing runtime. */ }
}
if (!pw) throw new Error('No existing Playwright runtime found. Set PLAYWRIGHT_MODULE; this test installs nothing.');

const files = [
  process.argv[2] || path.join(src, '..', 'attention.html'),
  process.argv[3] || path.join(src, '..', 'part3.html')
];
const model = JSON.parse(readFileSync(path.join(src, 'toy.json'), 'utf8'));
assert.equal(model.max_context, 20, 'The toy must declare its 20-token capacity.');
assert.equal(model.pos_emb.length, model.max_context, 'One positional row is required for every supported position.');
assert(model.pos_emb.every(row => Array.isArray(row) && row.length === model.d_model && row.every(Number.isFinite)), 'Every position row must be finite and d_model-wide.');
const baselineRiver = baseline(model, model.sentences.river).probs.at(-1);
const baselineFinance = baseline(model, model.sentences.cheque).probs.at(-1);
assert.deepEqual(baselineRiver, baselineFinance, 'Same final token and position give exactly the same baseline distribution.');
const peakBaseline = Math.max(...baselineRiver);
assert.equal(baselineRiver.filter(p => p === peakBaseline).length, 8, 'The eight displayed baseline candidates are exactly tied, not a ranked winner.');
assert(Math.abs(peakBaseline - 0.0936543386) < 1e-10, 'Baseline tie probability');
assert(baselineRiver.some(p => p !== peakBaseline), 'The whole vocabulary distribution is not uniform.');

const keys = ['E', 'Q', 'K', 'V', 'Sraw', 'S', 'A', 'Mmsg', 'Delta', 'Enew', 'logits', 'probs'];
let finite = 0, infinities = 0, maxError = 0, cases = 0, errorChecks = 0;
function compare(actual, expected, label) {
  if (Array.isArray(expected)) {
    assert(Array.isArray(actual), label + ': expected array');
    assert.equal(actual.length, expected.length, label + ': array length');
    expected.forEach((value, i) => compare(actual[i], value, `${label}[${i}]`));
    return;
  }
  if (!Number.isFinite(expected)) {
    assert.equal(actual, expected, label + ': masked score');
    infinities++;
    return;
  }
  assert(Number.isFinite(actual), label + ': non-finite result');
  const error = Math.abs(actual - expected);
  maxError = Math.max(maxError, error);
  finite++;
  assert(error < 1e-12, `${label}: ${actual} != ${expected}`);
}
function compareForward(actual, expected, label) {
  assert.equal(actual.T, expected.T, label + ': number of positions');
  for (const key of keys) compare(actual[key], expected[key], label + '.' + key);
}
function assertClearError(message, pattern, label) {
  assert.equal(typeof message, 'string', label + ': must reject the invalid input');
  assert(pattern.test(message), label + ': error must explain the invalid input, got ' + message);
  errorChecks++;
}
const capacityError = /position|context|capacity/i;
const tokenError = /token|vocab/i;
const tooLong = Array.from({ length: model.max_context + 1 }, (_, i) => model.sentences.river[i % model.sentences.river.length]);
const unknown = [...model.sentences.river, '__unknown_token__'];
assert.throws(() => forward(model, tooLong), capacityError, 'Reference must reject position 21.');
assert.throws(() => forward(model, unknown), tokenError, 'Reference must reject an unknown token.');
errorChecks += 2;

const browserErrors = [];
const browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
  : {});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on('pageerror', error => browserErrors.push('PAGEERROR: ' + error.message));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push('CONSOLE: ' + message.text());
  });

  for (const [partIndex, file] of files.entries()) {
    const name = path.basename(file);
    await page.goto(pathToFileURL(path.resolve(file)).href, { waitUntil: 'load' });
    await page.waitForFunction(() => window.AT && typeof AT.forward === 'function');
    const liveModel = await page.evaluate(() => window.__TOY__);
    for (const key of ['max_context', 'd_model', 'd_k', 'd_v', 'vocab', 'tok_emb', 'pos_emb', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_vocab', 'b_vocab', 'sentences']) {
      assert.deepEqual(liveModel[key], model[key], name + ': assembled model is stale at ' + key);
    }

    for (const sentence of ['river', 'cheque']) {
      const initial = model.sentences[sentence];
      assert.equal(initial.length, 10, sentence + ': regression starts from the original ten-token prefix');
      const initialResult = forward(model, initial);
      const finalProbabilities = initialResult.probs.at(-1);
      const chosen = model.vocab[finalProbabilities.indexOf(Math.max(...finalProbabilities))];
      const prefixes = [initial, [...initial, chosen], Array.from({ length: model.max_context }, (_, i) => initial[i % initial.length])];
      for (const tokens of prefixes) for (const mask of [false, true]) for (const scale of [false, true]) {
        const options = { mask, scale };
        const ref = forward(model, tokens, options);
        const live = await page.evaluate(({ tokens, options }) => AT.forward(tokens, options), { tokens, options });
        compareForward(live, ref, `${name}.${sentence}.${tokens.length}.mask=${mask}.scale=${scale}`);
        if (tokens.length === 10) {
          // Extra capacity must not affect the original prefix: truncate unused
          // rows to recreate the original ten-position reference model.
          const originalCapacity = { ...model, max_context: 10, pos_emb: model.pos_emb.slice(0, 10) };
          compareForward(live, forward(originalCapacity, tokens, options), `${name}.${sentence}.original-ten-position-model`);
        }
        cases++;
      }
    }

    for (const [label, tokens, pattern] of [
      ['21-token prefix', tooLong, capacityError],
      ['unknown token', unknown, tokenError]
    ]) {
      const message = await page.evaluate(tokens => {
        try { AT.forward(tokens); return null; } catch (error) { return String(error.message || error); }
      }, tokens);
      assertClearError(message, pattern, name + ': ' + label);
    }

    // Mutations exist only in this browser instance and are restored immediately.
    // They must fail loudly rather than silently becoming zero position vectors.
    for (const mode of ['missing', 'wrong-width', 'non-finite']) {
      const message = await page.evaluate(mode => {
        const rows = AT.model.pos_emb, original = rows[0];
        try {
          rows[0] = mode === 'missing' ? undefined
            : mode === 'wrong-width' ? original.slice(0, -1)
              : original.map((value, i) => i === 0 ? NaN : value);
          try { AT.embed(AT.sentences.river); return null; } catch (error) { return String(error.message || error); }
        } finally { rows[0] = original; }
      }, mode);
      assertClearError(message, /position|positional/i, name + ': ' + mode + ' position row');
    }

    if (partIndex === 1) {
      const stepper = page.locator('#s13-stepper .stepper');
      assert.equal(await stepper.count(), 1, 'Part 3 generation stepper must exist.');
      const initial = await page.evaluate(() => AT.train.sentence());
      assert.equal(initial.length, 10, 'Generation starts with ten tokens.');
      const reference = forward(model, initial), probs = reference.probs.at(-1);
      const chosen = model.vocab[probs.indexOf(Math.max(...probs))];
      await page.locator('#s13-stepper .btn-reset').click();
      await page.locator('#s13-stepper .btn-next').click();
      await page.waitForFunction(() => document.querySelector('#s13-stepper .stepper').stepperApi.index() === 1);
      const result = await page.evaluate(() => {
        const stage = document.querySelector('#s13-stepper .stepper-stage');
        return {
          tokens: [...stage.querySelectorAll('.chips .chip:not(.is-slot) .chip-t')].map(el => el.textContent),
          activePosition: stage.querySelector('.chips .chip.is-active .chip-i')?.textContent,
          slotPosition: stage.querySelector('.chips .chip.is-slot .chip-i')?.textContent,
          bars: [...stage.querySelectorAll('.bars .bl')].map((el, i) => ({
            token: el.textContent,
            probability: Number(stage.querySelectorAll('.bars .bv')[i].textContent)
          }))
        };
      });
      assert.deepEqual(result.tokens, [...initial, chosen], 'Generation must append the actual greedy prediction.');
      assert.equal(result.activePosition, '11', 'The new final query belongs to position 11.');
      assert.equal(result.slotPosition, '12', 'The next distribution predicts position 12.');
      const nextProbabilities = forward(model, [...initial, chosen]).probs.at(-1);
      const nextTopThree = model.vocab.map((token, i) => ({ token, probability: nextProbabilities[i] }))
        .sort((a, b) => b.probability - a.probability).slice(0, 3)
        .map(item => ({ token: item.token, probability: Number(item.probability.toFixed(4)) }));
      assert.deepEqual(result.bars, nextTopThree, 'The eleven-token forward pass must render the correct top-three probabilities.');
      console.log(`PASS: Part 3 generation appends "${chosen}", evaluates 11 positions, and predicts position 12.`);
    }
    assert.deepEqual(browserErrors, [], name + ': browser errors');
    console.log(`PASS: ${name}: 10/11/20-token forward passes, unchanged original prefix, and explicit invalid-input errors.`);
  }
  console.log(`PASS: ${cases} live/reference cases; ${finite} finite values, ${infinities} masked infinities; max error ${maxError.toExponential(3)}; ${errorChecks} rejection checks; no files written.`);
} finally {
  await browser.close();
}

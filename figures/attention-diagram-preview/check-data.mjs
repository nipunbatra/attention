// Read-only verification: node figures/attention-diagram-preview/check-data.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { forward } from '../../src/toy_ref.mjs';
import './toy-data.js';

const data = globalThis.ATTENTION_PREVIEW_DATA;
const source = readFileSync(new URL('../../src/toy.json', import.meta.url), 'utf8');
const model = JSON.parse(source);
assert.equal(createHash('sha256').update(source).digest('hex'), data.provenance.modelSha256);
assert.equal(data.provenance.handDesigned, true);
assert.equal(data.provenance.trained, false);
assert.deepEqual(data.tokens, model.sentences.river);
assert.deepEqual(data.vocabulary, model.vocab);
assert.deepEqual(data.dims, {
  T: 10, dModel: model.d_model, dKey: model.d_k,
  dValue: model.d_v, vocabSize: model.vocab.length,
});
for (const [name, key] of Object.entries({ WQ: 'W_Q', WK: 'W_K', WV: 'W_V', WO: 'W_O', WVocab: 'W_vocab' })) {
  assert.deepEqual(data.shapes[name], [model[key].length, model[key][0].length]);
}

let checkedNumbers = 0;
function close(actual, expected, path) {
  if (Array.isArray(expected)) {
    assert(Array.isArray(actual), path);
    assert.equal(actual.length, expected.length, path);
    expected.forEach((value, i) => close(actual[i], value, `${path}[${i}]`));
  } else if (typeof expected === 'number') {
    assert(Number.isFinite(actual), path);
    assert(Math.abs(actual - expected) < 1e-12, `${path}: ${actual} != ${expected}`);
    checkedNumbers++;
  } else {
    assert.deepEqual(actual, expected, path);
  }
}
function frozen(value) {
  if (value === null || typeof value !== 'object') return;
  assert(Object.isFrozen(value), 'Every nested object and row must be frozen');
  Object.values(value).forEach(frozen);
}
frozen(data);

const reference = forward(model, data.tokens);
for (const [name, index] of [['bank', 6], ['last', 9]]) {
  const actual = data[name];
  assert.equal(actual.index, index);
  assert.equal(actual.position, index + 1);
  assert.equal(actual.token, data.tokens[index]);
  const expected = {
    e: reference.E[index], q: reference.Q[index],
    keys: reference.K, values: reference.V,
    rawScores: reference.Sraw[index],
    scaledScores: reference.Sraw[index].map(s => s / Math.sqrt(model.d_k)),
    maskedScores: reference.S[index].map(s => Number.isFinite(s) ? s : null),
    alpha: reference.A[index], mixture: reference.Mmsg[index],
    delta: reference.Delta[index], updated: reference.Enew[index],
    logits: reference.logits[index], probabilities: reference.probs[index],
  };
  for (const [key, value] of Object.entries(expected)) close(actual[key], value, `${name}.${key}`);
  assert(Math.abs(actual.alpha.reduce((sum, x) => sum + x, 0) - 1) < 1e-12);
  assert(Math.abs(actual.probabilities.reduce((sum, x) => sum + x, 0) - 1) < 1e-12);
  for (let j = index + 1; j < data.tokens.length; j++) {
    assert.equal(actual.alpha[j], 0);
    assert.equal(actual.maskedScores[j], null);
  }
  const prefix = forward(model, data.tokens.slice(0, index + 1));
  close(actual.updated, prefix.Enew[index], `${name}.causalPrefixEquivalence`);
  const top = model.vocab.map((token, j) => ({
    token, index: j, logit: reference.logits[index][j], probability: reference.probs[index][j],
  })).sort((a, b) => b.probability - a.probability).slice(0, 6);
  assert.deepEqual(actual.topVocabulary, top);
}

console.log(`PASS: ${checkedNumbers} numeric comparisons; shapes, causal zeros, normalization, prefix equivalence, provenance, and deep freeze.`);
console.log('Bank: e7 =', data.bank.e, 'q7 =', data.bank.q);
console.log('Bank: mixture =', data.bank.mixture, 'delta =', data.bank.delta, 'updated =', data.bank.updated);
console.log('Final token:', data.last.token, 'at position', data.last.position, 'predicts', data.last.topVocabulary[0]);

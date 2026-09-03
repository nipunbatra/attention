// toy_ref.mjs - reference forward pass for the shared toy model (toy.json).
// Plain ES module, no dependencies. shared.js must reproduce these numbers exactly.
//
// Conventions (BRIEF.md section 4): row vectors. E[i] = tok_emb[lower(token_i)] + pos_emb[i];
// Q = E W_Q, K = E W_K, V = E W_V (each row = one token); Sraw[i][j] = Q[i].K[j];
// S = Sraw / sqrt(d_k) when scale; A = row-softmax(S) with j>i set to -Infinity when mask;
// Mmsg[i] = sum_j A[i][j] V[j]; Delta = Mmsg W_O; Enew = E + Delta;
// logits[i] = Enew[i] W_vocab + b_vocab; probs = row-softmax(logits).
//
// CLI:  node toy_ref.mjs [toy.json] [--compare py_check.json]
//   prints the check report and exits 1 if any HARD target fails (or the comparison exceeds 1e-6).

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function matmul(A, B) {
  // A: n x k (array of rows), B: k x m (array of rows) -> n x m
  const n = A.length, k = B.length, m = B[0].length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(m).fill(0);
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) row[j] += a * Bp[j];
    }
    out[i] = row;
  }
  return out;
}

export function softmax(arr) {
  let m = -Infinity;
  for (const x of arr) if (x > m) m = x;
  const ex = arr.map((x) => Math.exp(x - m));
  let s = 0;
  for (const x of ex) s += x;
  return ex.map((x) => x / s);
}

export function embed(toy, tokens) {
  const capacity = toy.max_context ?? toy.pos_emb.length;
  if (tokens.length > capacity) throw new Error(`context length ${tokens.length} exceeds positional capacity ${capacity}`);
  return tokens.map((t, i) => {
    const te = toy.tok_emb[t.toLowerCase()];
    if (!te) throw new Error(`token not in vocabulary: ${t}`);
    const pe = toy.pos_emb[i];
    if (!Array.isArray(te) || te.length !== toy.d_model || !te.every(Number.isFinite)) throw new Error(`invalid token embedding for ${t}`);
    if (!Array.isArray(pe) || pe.length !== toy.d_model || !pe.every(Number.isFinite)) throw new Error(`invalid positional vector for position ${i + 1}`);
    return te.map((x, d) => x + pe[d]);
  });
}

export function forward(toy, tokens, { mask = true, scale = true } = {}) {
  const T = tokens.length;
  const E = embed(toy, tokens);
  const Q = matmul(E, toy.W_Q);
  const K = matmul(E, toy.W_K);
  const V = matmul(E, toy.W_V);
  const Sraw = Q.map((qi) => K.map((kj) => dot(qi, kj)));
  const f = scale ? 1 / Math.sqrt(toy.d_k) : 1;
  const S = Sraw.map((row, i) => row.map((s, j) => (mask && j > i ? -Infinity : s * f)));
  const A = S.map((row) => softmax(row));
  const Mmsg = matmul(A, V);
  const Delta = matmul(Mmsg, toy.W_O);
  const Enew = E.map((e, i) => e.map((x, d) => x + Delta[i][d]));
  const logits = matmul(Enew, toy.W_vocab).map((row) => row.map((x, j) => x + toy.b_vocab[j]));
  const probs = logits.map((row) => softmax(row));
  return { tokens, T, E, Q, K, V, Sraw, S, A, Mmsg, Delta, Enew, logits, probs };
}

export function baseline(toy, tokens) {
  const E = embed(toy, tokens);
  const logits = matmul(E, toy.W_vocab).map((row) => row.map((x, j) => x + toy.b_vocab[j]));
  const probs = logits.map((row) => softmax(row));
  return { E, logits, probs };
}

// ------------------------------------------------------------------------------------------------
// Target checks (mirrors make_toy.py check()).  Returns [{name, ok, detail, hard}].
// ------------------------------------------------------------------------------------------------
export function checkTargets(toy) {
  const SA = toy.sentences.river, SB = toy.sentences.cheque;
  const CA = toy.candidates.river, CB = toy.candidates.cheque;
  const VI = Object.fromEntries(toy.vocab.map((w, i) => [w, i]));
  const BANK = 6, LAST = 9;
  const fa = forward(toy, SA), fb = forward(toy, SB), fu = forward(toy, SA, { mask: false });
  const ba = baseline(toy, SA), bb = baseline(toy, SB);
  const f2 = (x) => x.toFixed(2), f3 = (x) => x.toFixed(3);
  const res = [];
  const rec = (name, ok, detail, hard = true) => res.push({ name, ok: !!ok, detail, hard });
  const rowStr = (toks, row, n) => toks.slice(0, n).map((t, j) => `${t}=${f2(row[j])}`).join(' ');

  // Targets follow AXES.md (toy v2): interpretability first, so the thresholds are looser than the first toy's.
  const top3 = (row) => row.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => x[1]);
  const sameSet = (a, b) => a.length === b.length && a.every((x) => b.includes(x));
  const rA = fa.A[BANK];
  rec('T1 hard: bank(7) in S_A: river >= .40, fisherman second, bank(self) and glue words below fisherman',
    rA[5] >= 0.40 && rA[1] > Math.max(rA[0], rA[2], rA[3], rA[4], rA[6]),
    'bank(7) row S_A: ' + rowStr(SA, rA, 7));
  const rB = fb.A[BANK];
  rec('T2 hard: bank(7) in S_B: cheque highest, deposited second, others low',
    rB[3] > rB[1] && rB[1] > Math.max(rB[0], rB[2], rB[4], rB[5], rB[6]),
    'bank(7) row S_B: ' + rowStr(SB, rB, 7));
  const pA = fa.probs[LAST], aA = fa.A[LAST];
  const othA = toy.vocab.filter((w) => !CA.includes(w)).map((w) => pA[VI[w]]);
  const cA = CA.map((c) => pA[VI[c]]);
  rec('T3 hard: the(10) in S_A: water > boats > fish > ducks, every other token <= .04',
    cA[0] > cA[1] && cA[1] > cA[2] && cA[2] > cA[3] && othA.every((p) => p <= 0.04),
    CA.map((c) => `${c}=${f3(pA[VI[c]])}`).join(' ') + ` | max other=${f3(Math.max(...othA))}`);
  rec('T3 hard: attention of the(10) mostly on river, bank, fisherman (sum >= .60, they are the top 3)',
    aA[5] + aA[6] + aA[1] >= 0.60 && sameSet(top3(aA), [5, 6, 1]),
    'the(10) row S_A: ' + rowStr(SA, aA, 10));
  rec('T3 soft: top-5 next-token probs', true,
    top5(toy, pA).join('  '), false);
  const pB = fb.probs[LAST], aB = fb.A[LAST];
  const othB = toy.vocab.filter((w) => !CB.includes(w)).map((w) => pB[VI[w]]);
  const cB = CB.map((c) => pB[VI[c]]);
  rec('T4 hard: the(10) in S_B: teller > clerk > queue > money, every other token <= .04',
    cB[0] > cB[1] && cB[1] > cB[2] && cB[2] > cB[3] && othB.every((p) => p <= 0.04),
    CB.map((c) => `${c}=${f3(pB[VI[c]])}`).join(' ') + ` | max other=${f3(Math.max(...othB))}`);
  rec('T4 hard: attention of the(10) mostly on cheque, bank, deposited (sum >= .60, they are the top 3)',
    aB[3] + aB[6] + aB[1] >= 0.60 && sameSet(top3(aB), [3, 6, 1]),
    'the(10) row S_B: ' + rowStr(SB, aB, 10));
  rec('T4 soft: top-5 next-token probs', true, top5(toy, pB).join('  '), false);
  const pb = ba.probs[LAST], pb2 = bb.probs[LAST];
  const cands = [...CA, ...CB];
  const cp = cands.map((c) => pb[VI[c]]);
  const othAll = toy.vocab.filter((w) => !cands.includes(w)).map((w) => pb[VI[w]]);
  const same = pb.every((p, i) => Math.abs(p - pb2[i]) < 1e-12);
  rec('T5 hard: baseline candidates each in [.06,.18], others <= .04, identical for S_A/S_B',
    cp.every((p) => p >= 0.06 && p <= 0.18) && othAll.every((p) => p <= 0.04) && same,
    cands.map((c) => `${c}=${f3(pb[VI[c]])}`).join(' ') + ` | max other=${f3(Math.max(...othAll))}`);
  const leaks = [0, 1, 2, 3, 4].map((i) => [`${SA[i]}(${i + 1})`, fu.A[i][5] + fu.A[i][6], fu.A[i][5], fu.A[i][6]]);
  rec('T6 soft: mask off, some early token of S_A puts >= .25 on future river/bank',
    Math.max(...leaks.map((l) => l[1])) >= 0.25,
    leaks.map(([k, v, r, b]) => `${k}: ${f2(v)} (river ${f2(r)}, bank ${f2(b)})`).join('  '), false);
  const all = [];
  for (const w of toy.vocab) all.push(...toy.tok_emb[w]);
  for (const k of ['pos_emb', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_vocab']) for (const r of toy[k]) all.push(...r);
  all.push(...toy.b_vocab);
  rec('T7 hard: all |x| <= 3.0 and 1-decimal',
    all.every((x) => Math.abs(x) <= 3 && Math.abs(x * 10 - Math.round(x * 10)) < 1e-9),
    `${all.length} numbers, max|x|=${Math.max(...all.map(Math.abs)).toFixed(1)}`);
  const want = 'the fisherman sat beside river bank and watched she deposited cheque at water boats fish ducks teller clerk queue money'.split(' ');
  rec('T8 hard: vocabulary is the 20 tokens', toy.vocab.length === 20 && want.every((w, i) => toy.vocab[i] === w), '20 tokens');
  const posAxis = toy.axes && toy.axes.e ? toy.axes.e.indexOf('position') : -1;
  const thes = [0, 4, 9].map((i) => fa.E[i].join(','));
  const positionRowsOk = posAxis >= 0 && toy.pos_emb.every((row, i) => row.every((x, d) =>
    Math.abs(x - (d === posAxis ? 0.1 * (i + 1) : 0)) < 1e-9));
  const zeroAt = (M) => posAxis >= 0 && M[posAxis] && M[posAxis].every((x) => x === 0);
  rec("T9 hard: position has its own coordinate and the three 'the' rows differ",
    positionRowsOk && toy.vocab.every((w) => toy.tok_emb[w][posAxis] === 0) &&
      zeroAt(toy.W_Q) && zeroAt(toy.W_K) && zeroAt(toy.W_V) && zeroAt(toy.W_vocab) && new Set(thes).size === 3,
    `position rows 0.1 to ${(0.1 * toy.pos_emb.length).toFixed(1)}; token and projection position rows are zero`);
  const hasShape = (M, rows, cols) => Array.isArray(M) && M.length === rows && M.every((row) => Array.isArray(row) && row.length === cols);
  rec('T10 hard: parameter shapes match d_model, d_k and d_v',
    toy.axes.e.length === toy.d_model && toy.axes.qk.length === toy.d_k && toy.axes.v.length === toy.d_v &&
      toy.vocab.every((w) => Array.isArray(toy.tok_emb[w]) && toy.tok_emb[w].length === toy.d_model) &&
      toy.pos_emb.length >= toy.sentences.river.length && hasShape(toy.pos_emb, toy.max_context ?? toy.pos_emb.length, toy.d_model) &&
      hasShape(toy.W_Q, toy.d_model, toy.d_k) && hasShape(toy.W_K, toy.d_model, toy.d_k) &&
      hasShape(toy.W_V, toy.d_model, toy.d_v) && hasShape(toy.W_O, toy.d_v, toy.d_model) &&
      hasShape(toy.W_vocab, toy.d_model, toy.vocab.length) && toy.b_vocab.length === toy.vocab.length, 'ok');
  return { results: res, fa, fb, fu, ba };
}

function top5(toy, p) {
  return toy.vocab.map((w, i) => [w, p[i]]).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([w, v]) => `${w} ${v.toFixed(3)}`);
}

// ------------------------------------------------------------------------------------------------
// CLI
// ------------------------------------------------------------------------------------------------
async function cli() {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const url = await import('node:url');
  const here = path.dirname(url.fileURLToPath(import.meta.url));
  const args = process.argv.slice(2);
  const jsonPath = args.find((a) => !a.startsWith('--') && args[args.indexOf(a) - 1] !== '--compare') || path.join(here, 'toy.json');
  const toy = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  const { results, fa, fb, fu } = checkTargets(toy);
  const SA = toy.sentences.river, SB = toy.sentences.cheque;
  const f2 = (x) => x.toFixed(2);
  console.log(`toy_ref.mjs check report for ${jsonPath}\n`);
  console.log('Attention rows (causal, scaled):');
  console.log('  S_A bank(7):  ' + SA.slice(0, 7).map((t, j) => `${t}=${f2(fa.A[6][j])}`).join(' '));
  console.log('  S_B bank(7):  ' + SB.slice(0, 7).map((t, j) => `${t}=${f2(fb.A[6][j])}`).join(' '));
  console.log('  S_A the(10):  ' + SA.map((t, j) => `${t}=${f2(fa.A[9][j])}`).join(' '));
  console.log('  S_B the(10):  ' + SB.map((t, j) => `${t}=${f2(fb.A[9][j])}`).join(' '));
  console.log('Top-5 next-token probs after the(10):');
  console.log('  S_A: ' + top5(toy, fa.probs[9]).join('  '));
  console.log('  S_B: ' + top5(toy, fb.probs[9]).join('  '));
  console.log('Baseline (no attention) probs for the(10), top-9: ' + baseline(toy, SA).probs[9]
    .map((p, i) => [toy.vocab[i], p]).sort((a, b) => b[1] - a[1]).slice(0, 9).map(([w, v]) => `${w} ${v.toFixed(3)}`).join('  '));
  console.log('Unmasked S_A rows (leakage onto future positions):');
  for (const i of [0, 1, 2, 3, 4]) console.log(`  ${SA[i]}(${i + 1}): ` + SA.map((t, j) => `${t}=${f2(fu.A[i][j])}`).join(' '));
  console.log('');
  let hardFail = false;
  for (const r of results) {
    console.log(`[${r.ok ? 'PASS' : 'FAIL'}]${r.hard ? '' : ' (soft)'} ${r.name}\n        ${r.detail}`);
    if (r.hard && !r.ok) hardFail = true;
  }
  const ci = args.indexOf('--compare');
  if (ci >= 0) {
    const ref = JSON.parse(fs.readFileSync(args[ci + 1], 'utf8'));
    const cases = {
      river_masked: forward(toy, SA), cheque_masked: forward(toy, SB),
      river_unmasked: forward(toy, SA, { mask: false }), cheque_unmasked: forward(toy, SB, { mask: false }),
      river_noscale: forward(toy, SA, { scale: false }),
    };
    let worst = 0, count = 0;
    const cmp = (a, b) => {
      if (Array.isArray(a)) { a.forEach((x, i) => cmp(x, b[i])); return; }
      if (b === null) return; // masked S entry (python uses -1e9, JS -Infinity)
      worst = Math.max(worst, Math.abs(a - b)); count++;
    };
    for (const [name, f] of Object.entries(cases)) {
      for (const k of ['E', 'Q', 'K', 'V', 'Sraw', 'S', 'A', 'Mmsg', 'Delta', 'Enew', 'logits', 'probs']) cmp(f[k], ref[name][k]);
      const b = baseline(toy, name.startsWith('river') ? SA : SB);
      cmp(b.logits, ref[name].base_logits); cmp(b.probs, ref[name].base_probs);
    }
    console.log(`\ncompare with ${args[ci + 1]}: ${count} numbers, max |JS - python| = ${worst.toExponential(2)} -> ${worst <= 1e-6 ? 'AGREE' : 'MISMATCH'}`);
    if (worst > 1e-6) hardFail = true;
  }
  console.log(`\nHARD TARGETS: ${hardFail ? 'FAILURES' : 'ALL PASS'}`);
  process.exit(hardFail ? 1 : 0);
}

const isMain = typeof process !== 'undefined' && process.argv && process.argv[1] &&
  import.meta.url === (await import('node:url')).pathToFileURL(process.argv[1]).href;
if (isMain) await cli();

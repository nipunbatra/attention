// Independent checks for Vision II (toy6.json, part6.js, sections6/, part6.json).
//   node src/check_vision2.mjs [--no-browser] [--page /absolute/path/vision2.html]
// 1. A plain-JS reference implementation (not part6.js) recomputes the frozen Vision I encoder, the MAE decoder,
//    the I-JEPA predictor, the DINO checkpoints and the probe from the stored parameters and compares every stored
//    loss, prediction, attention row and probability with the JSON written by train_vision2.py.
// 2. Fragment and config invariants: frames, notes, builds, unique ids, no "coordinate N", no dashes, code parses.
// 3. With an existing Playwright runtime (default), the assembled page is loaded: the runtime's numbers match the
//    JSON, the controls redraw, every frame fits the 1280x720 stage at its last build, and the 390px article has no
//    horizontal overflow. Nothing is installed and nothing is written.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const src = path.dirname(fileURLToPath(import.meta.url));
const read = name => fs.readFileSync(path.join(src, name), 'utf8');
const toy = JSON.parse(read('toy6.json')), toy5 = JSON.parse(read('toy5.json')), config = JSON.parse(read('part6.json'));
const TOL = 5e-5;
let compared = 0, maxErr = 0;
const near = (a, b, label, tol = TOL) => { const d = Math.abs(a - b); compared++; maxErr = Math.max(maxErr, d); assert.ok(Number.isFinite(a) && d <= tol, `${label}: ${a} != ${b} (diff ${d})`); };
const rowsNear = (A, B, label) => A.forEach((r, i) => r.forEach((x, j) => near(x, B[i][j], label + '[' + i + '][' + j + ']')));

/* ---------- the reference implementation (vision-shared.js conventions, written independently) ---------- */
const SQ2 = Math.SQRT2;
function blank() { return Array.from({ length: 8 }, () => Array(8).fill(0)); }
function mug(g, c0) { for (let r = 1; r <= 3; r++) for (let c = c0; c < c0 + 3; c++) g[r][c] = 3; }
function book(g) { for (let r = 5; r <= 6; r++) for (let c = 1; c <= 5; c++) g[r][c] = 2; }
function plant(g) { g[4][7] = 1; }
const SC = {};
SC.A = blank(); mug(SC.A, 1); mug(SC.A, 5); book(SC.A); plant(SC.A);
SC.B = blank(); mug(SC.B, 1); book(SC.B); plant(SC.B);
SC.C = blank(); mug(SC.C, 4); book(SC.C); plant(SC.C);
SC.D = blank(); book(SC.D);
SC.E = blank(); plant(SC.E);
for (const k of Object.keys(toy5.scenes)) assert.deepEqual(SC[k], toy5.scenes[k].pixels, 'scene ' + k + ' matches toy5');
const W_PATCH = [[.25, .5, 0, 0], [.25, -.5, 0, 0], [.25, .5, 0, 0], [.25, -.5, 0, 0]], CLS = [1, 0, 0, 0];
const POS = [[0, 0, -1, -1]].concat(Array.from({ length: 16 }, (_, j) => [0, 0, Math.floor(j / 4) / 3, (j % 4) / 3]));
const ENC = toy.encoder;
assert.deepEqual(ENC, toy5.trained, 'the encoder is exactly the trained Vision I encoder');
const mm = (A, B) => A.map(r => B[0].map((_, c) => r.reduce((s, x, k) => s + x * B[k][c], 0)));
const mv = (v, M) => M[0].map((_, c) => v.reduce((s, x, k) => s + x * M[k][c], 0));
const add = (a, b) => a.map((x, i) => x + b[i]);
const softmax = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); };
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
function patchify(g) { const rows = []; for (let pr = 0; pr < 4; pr++) for (let pc = 0; pc < 4; pc++) rows.push([g[2 * pr][2 * pc], g[2 * pr][2 * pc + 1], g[2 * pr + 1][2 * pc], g[2 * pr + 1][2 * pc + 1]]); return rows; }
function embed(g) { return [CLS].concat(mm(patchify(g), W_PATCH)).map((r, i) => add(r, POS[i])); }
function attend(E, P) { const Q = mm(E, P.W_Q), K = mm(E, P.W_K), Vv = mm(E, P.W_V); const S = Q.map(q => K.map(k => (q[0] * k[0] + q[1] * k[1]) / SQ2)); const A = S.map(softmax); const D = mm(mm(A, Vv), P.W_O); return { E, A, Enew: E.map((r, i) => add(r, D[i])) }; }
const encode = g => attend(embed(g), ENC).Enew;
function encodeVisible(g, hidden) { const em = embed(g), idx = [0]; for (let j = 0; j < 16; j++) if (!hidden.includes(j)) idx.push(j + 1); return { vis: idx.slice(1).map(i => i - 1), out: attend(idx.map(i => em[i]), ENC) }; }
function readSlots(theta, g, hidden) {
  const ev = encodeVisible(g, hidden), Ev = ev.out.Enew.slice(1), K = mm(Ev, ENC.W_K), Vv = mm(Ev, ENC.W_V), res = {};
  for (const j of hidden) { const u = add(theta.m, POS[j + 1]), q = mv(u, theta.W_Qd), a = softmax(K.map(k => (q[0] * k[0] + q[1] * k[1]) / SQ2)); const msg = [0, 1].map(c => a.reduce((t, al, i) => t + al * Vv[i][c], 0)); res[j] = { alpha: a, z: [msg[0], msg[1], POS[j + 1][2], POS[j + 1][3]] }; }
  return { vis: ev.vis, res };
}
const views = {
  identity: g => g.map(r => r.slice()), flip: g => g.map(r => r.slice().reverse()), dim: g => g.map(r => r.map(x => .75 * x)),
  crop: g => { const idx = Array.from({ length: 8 }, (_, r) => 1 + Math.floor(r * 7 / 8)); return idx.map(rr => idx.map(cc => g[rr][cc])); }
};
views['crop+dim'] = g => views.dim(views.crop(g));

/* MAE and I-JEPA checkpoints */
const hidden = toy.hidden; assert.deepEqual(hidden, [2, 3, 6, 7]);
for (const [step, cp] of Object.entries(toy.mae.checkpoints)) {
  const per = [];
  for (const s of toy.mae.scenes) {
    const rd = readSlots(cp.params, SC[s], hidden), R = patchify(SC[s]), se = [];
    for (const j of hidden) { const r = add(mv(rd.res[j].z, cp.params.W_dec), cp.params.b_dec); r.forEach((x, c) => { near(x, cp.scenes[s].pred[j][c], `mae step ${step} ${s} pred ${j}`); se.push((x - R[j][c]) ** 2); }); rd.res[j].alpha.forEach((a, i) => near(a, cp.scenes[s].alpha[j][i], `mae alpha ${step} ${s} ${j}`)); rd.res[j].z.forEach((x, i) => near(x, cp.scenes[s].z[j][i], `mae z ${step} ${s} ${j}`)); }
    near(mean(se), cp.scenes[s].loss, `mae loss ${step} ${s}`); per.push(mean(se));
    assert.equal(rd.vis.length, 12);
  }
  near(mean(per), cp.mean_loss, 'mae mean ' + step); near(mean(per), toy.mae.curve[+step][1], 'mae curve ' + step);
}
assert.ok(toy.mae.curve[100][1] < toy.mae.curve[10][1] && toy.mae.curve[10][1] < toy.mae.curve[0][1], 'the MAE loss falls');
{ const a = toy.mae.checkpoints['100'].scenes.A.pred, b = toy.mae.checkpoints['100'].scenes.B.pred; for (const j of hidden) a[j].forEach((x, c) => near(x, b[j][c], 'A and B share one prediction', 1e-9)); }
for (const [step, cp] of Object.entries(toy.jepa.checkpoints)) {
  const per = [];
  for (const s of toy.jepa.scenes) {
    const rd = readSlots(cp.params, SC[s], hidden), full = encode(SC[s]), se = [];
    for (const j of hidden) { const y = add(mv(rd.res[j].z, cp.params.W_pred), cp.params.b_pred), t = full[j + 1].slice(0, 2); y.forEach((x, c) => { near(x, cp.scenes[s].pred[j][c], `jepa pred ${step} ${s} ${j}`); near(t[c], cp.scenes[s].target[j][c], `jepa target ${s} ${j}`); se.push((x - t[c]) ** 2); }); }
    near(mean(se), cp.scenes[s].loss, `jepa loss ${step} ${s}`); per.push(mean(se));
  }
  near(mean(per), cp.mean_loss, 'jepa mean ' + step);
}
/* DINO checkpoints: recompute the outputs from the encoder and the stored heads */
const D = toy.dino;
for (const [key, run] of Object.entries(D.runs)) {
  const tau = run.tau_t; assert.equal(tau, run.sharpen ? D.tau_t_sharpen : D.tau_s);
  for (const [step, cp] of Object.entries(run.checkpoints)) {
    cp.outputs.forEach(o => {
      const f1 = encode(views[D.views[0]](SC[o.scene]))[0], f2 = encode(views[D.views[1]](SC[o.scene]))[0];
      const zs = add(mv(f1, cp.student.W), cp.student.b), zt = add(mv(f2, cp.teacher.W), cp.teacher.b), c = run.center ? cp.center : [0, 0, 0];
      zt.forEach((x, i) => near(x, o.teacher_logits_view2[i], `${key} ${step} ${o.scene} zt`));
      softmax(zs.map(x => x / D.tau_s)).forEach((p, i) => near(p, o.student_view1[i], `${key} ${step} ${o.scene} ps`));
      softmax(zt.map((x, i) => (x - c[i]) / tau)).forEach((p, i) => near(p, o.teacher_view2[i], `${key} ${step} ${o.scene} pt`));
    });
  }
  const last = run.curve[run.curve.length - 1][1];
  if (key === 'nocenter_sharpen') assert.ok(last < 0.01, 'no centring: the loss collapses to zero');
  if (key === 'center_nosharpen') near(last, Math.log(3), 'no sharpening: the loss sits at log 3', 1e-4);
  if (key === 'center_sharpen') assert.ok(last > 0.3 && last < Math.log(3), 'the full recipe keeps a non-trivial loss');
}
{ const outs = D.runs.center_sharpen.checkpoints['200'].outputs.map(o => o.teacher_view2.indexOf(Math.max(...o.teacher_view2))); assert.ok(new Set(outs).size >= 2, 'the full recipe assigns different slots to different scenes'); }
{ const outs = D.runs.nocenter_sharpen.checkpoints['200'].outputs; assert.ok(outs.every(o => Math.max(...o.teacher_view2) > 0.99) && new Set(outs.map(o => o.teacher_view2.indexOf(Math.max(...o.teacher_view2)))).size === 1, 'no centring: one slot for every scene'); }
{ const outs = D.runs.center_nosharpen.checkpoints['200'].outputs; assert.ok(outs.every(o => Math.max(...o.teacher_view2) < 0.34), 'no sharpening: uniform for every scene'); }
/* the probe */
const P = toy.probe;
for (const p of P.points) {
  const E = encode(views[p.view](SC[p.scene])).slice(1), f = [mean(E.map(r => r[0])), mean(E.map(r => r[1]))];
  f.forEach((x, i) => near(x, p.feature[i], `probe feature ${p.scene} ${p.view}`));
  near(1 / (1 + Math.exp(-(P.w[0] * f[0] + P.w[1] * f[1] + P.b))), p.p_two, `probe p ${p.scene} ${p.view}`);
  assert.equal(p.correct, (p.p_two >= .5) === (p.label === 1) ? 1 : 0);
}
assert.equal(P.points.filter(p => p.split === 'train').length, 9); assert.equal(P.points.filter(p => p.split === 'test').length, 3);
near(P.train_accuracy, mean(P.points.filter(p => p.split === 'train').map(p => p.correct)), 'train accuracy');
near(P.test_accuracy, mean(P.points.filter(p => p.split === 'test').map(p => p.correct)), 'test accuracy');
near(P.all.accuracy, mean(P.points.map((p, i) => (P.all.p_two[i] >= .5) === (p.label === 1) ? 1 : 0)), 'all-views accuracy');

/* ---------- fragments and config ---------- */
const dir = path.join(src, 'sections6'), files = fs.readdirSync(dir).filter(n => /^sec\d\d\.html$/.test(n)).sort();
assert.equal(files.length, 10, 'ten sections');
assert.deepEqual(config.sections.map(s => s.id), files.map(n => 's' + n.slice(3, 5)), 'config lists the files');
const ids = new Set(); let frames = 0, notes = 0, builds = 0;
for (const name of files) {
  const s = fs.readFileSync(path.join(dir, name), 'utf8');
  const fr = s.match(/<div class="frame"[^>]*>/g) || []; frames += fr.length;
  notes += (s.match(/type="text\/x-notes"/g) || []).length;
  for (const m of s.matchAll(/<div class="frame"[^>]*>([\s\S]*?)(?=<div class="frame"|<div class="companion after"|<script>\s*\(function)/g)) { assert.match(m[1], /data-build="\d"/, name + ': every frame has a build'); builds++; }
  for (const m of s.matchAll(/\bid="([^"]+)"/g)) { assert.ok(!ids.has(m[1]), 'duplicate id ' + m[1]); ids.add(m[1]); }
  for (const m of s.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(m[1], { filename: name });
  assert.ok(!/coordinate \d/i.test(s), name + ': no "coordinate N"');
  assert.ok(!/[—–]/.test(s), name + ': no em or en dashes');
  assert.ok(!/overflow:auto/.test(s), name + ': no internal scrolling');
}
assert.equal(frames, 42, 'frame count'); assert.equal(notes, frames, 'every frame has presenter notes'); assert.equal(builds, frames);
assert.equal(config.part, 2); assert.equal(config.notation, 'vision2'); assert.equal(config.prev.href, 'vision1.html'); assert.equal(config.next.href, 'vision3.html');
assert.ok(config.provenance.includes('frozen') && config.provenance.includes('{{axes}}'), 'provenance states the scope with the named axes');
console.log(JSON.stringify({ reference: 'pass', compared, maxErr, sections: files.length, frames, mae: ['0', '10', '100'].map(k => toy.mae.checkpoints[k].mean_loss), jepa: ['0', '10', '100'].map(k => toy.jepa.checkpoints[k].mean_loss), dino: Object.fromEntries(Object.entries(D.runs).map(([k, r]) => [k, r.curve[r.curve.length - 1][1]])), probe: [P.train_accuracy, P.test_accuracy] }));

/* ---------- browser ---------- */
if (process.argv.includes('--no-browser')) process.exit(0);
const require = createRequire(import.meta.url), cands = [process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean), cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) for (const d of fs.readdirSync(cache).sort()) cands.push(path.join(cache, d, 'node_modules', 'playwright'));
let pw; for (const c of cands) { try { pw = require(c); break; } catch { /* next */ } }
if (!pw) throw new Error('No installed Playwright runtime; pass --no-browser or set PLAYWRIGHT_MODULE.');
const argPage = process.argv.indexOf('--page'), page_path = argPage >= 0 ? process.argv[argPage + 1] : path.join(src, '..', 'vision2.html');
assert.ok(fs.existsSync(page_path), 'assembled page: ' + page_path);
const browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {});
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('pageerror', e => errors.push(String(e.message || e))); page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text()); });
  await page.goto(pathToFileURL(page_path).href); await page.waitForTimeout(600);
  const live = await page.evaluate(() => {
    const V = AT.vision, T = AT.model, out = { mae: [], jepa: [], dino: [], probe: [], katex: document.querySelectorAll('.katex-error').length, sections: document.querySelectorAll('section.sec').length, coordinate: /coordinate \d/i.test(document.body.innerText) };
    for (const st of ['0', '10', '100']) { out.mae.push([+st, V.maeMeanLoss(+st), T.mae.checkpoints[st].mean_loss]); out.jepa.push([+st, T.jepa.scenes.map(s => V.jepa(+st, s).loss).reduce((a, b) => a + b, 0) / 3, T.jepa.checkpoints[st].mean_loss]); }
    for (const [c, s] of [[true, true], [false, true], [true, false], [false, false]]) V.dino.outputs(c, s, 200).forEach((o, i) => { const st = V.dino.run(c, s).checkpoints['200'].outputs[i]; out.dino.push(Math.max(...o.pt.map((x, k) => Math.abs(x - st.teacher_view2[k])), ...o.ps.map((x, k) => Math.abs(x - st.student_view1[k])))); });
    V.probe.points.forEach(p => { const f = V.probe.pooled(V.view(p.view, p.scene)); out.probe.push(Math.abs(V.probe.p(f) - p.p_two)); });
    out.encoderRows = V.encode('A').length; out.frames = AT.present.frames().length;
    return out;
  });
  live.mae.forEach(([st, a, b]) => near(a, b, 'page mae ' + st)); live.jepa.forEach(([st, a, b]) => near(a, b, 'page jepa ' + st));
  live.dino.forEach(d => near(d, 0, 'page dino outputs')); live.probe.forEach(d => near(d, 0, 'page probe'));
  assert.equal(live.katex, 0); assert.equal(live.sections, 10); assert.equal(live.encoderRows, 17); assert.equal(live.frames, frames); assert.ok(!live.coordinate, 'the page never says "coordinate N"');
  /* controls redraw */
  await page.click('#s03-btns-a button:nth-child(3)'); assert.match(await page.locator('#s03-curve-read').innerText(), /step 100/);
  await page.selectOption('#s04-scene', 'D'); assert.match(await page.locator('#s04-read').innerText(), /scene D/);
  const before = await page.locator('#s06-run-read').innerText(); await page.click('#s06-switches .toggle:nth-child(1)'); const after = await page.locator('#s06-run-read').innerText();
  assert.notEqual(before, after); assert.match(after, /Collapse to one slot/); await page.click('#s06-switches .toggle:nth-child(1)');
  await page.click('#s06-switches .toggle:nth-child(2)'); assert.match(await page.locator('#s06-run-read').innerText(), /log 3/); await page.click('#s06-switches .toggle:nth-child(2)');
  /* every frame fits at its last build */
  await page.goto(pathToFileURL(page_path).href + '?present#s01/1/0'); await page.waitForTimeout(600);
  const issues = [];
  for (let fi = 0; fi < frames; fi++) {
    await page.evaluate(fi => { AT.present.go(fi, null, 999); }, fi); await page.waitForTimeout(80);
    const st = await page.evaluate(() => { const s = AT.present.state(), f = document.querySelector('.frame.is-live'); return { id: s.frame.id + '/' + (s.frame.index + 1), tall: f.scrollHeight - f.clientHeight, wide: f.scrollWidth - f.clientWidth }; });
    if (st.tall > 2 || st.wide > 2) issues.push(st);
  }
  assert.deepEqual(issues, [], 'no presentation frame overflows its stage');
  await page.setViewportSize({ width: 390, height: 844 }); await page.goto(pathToFileURL(page_path).href + '#s03'); await page.waitForTimeout(400);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), 'no horizontal overflow at 390px');
  assert.deepEqual(errors, [], 'no page errors');
  console.log(JSON.stringify({ browser: 'pass', frames, compared, maxErr }));
} finally { await browser.close(); }

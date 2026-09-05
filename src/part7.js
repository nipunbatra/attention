/* Vision III: CLIP on the shared scene. The frozen Vision I encoder (toy7.json "encoder", a copy of toy5.json "trained")
   supplies each scene's updated CLS row; a learned 4x3 map puts it on the joint axes mug / book / plant; a bag-of-words
   text map puts captions on the same axes. AT.clip owns every CLIP number; AT.vision (vision-shared.js) owns the scene,
   the frozen encoder arithmetic and the figures. Reference numbers: train_vision3.py. */
(function () {
  'use strict';
  const AT = window.AT, V = AT.vision, toy = window.__TOY__, C = toy.clip, ENC = toy.encoder;
  if (!V || !C) return;
  const h = AT.h, svg = AT.svg;

  /* ---------- helpers ---------- */
  const clone = x => JSON.parse(JSON.stringify(x));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const transpose = a => a[0].map((_, j) => a.map(r => r[j]));
  const mm = (a, b) => { const bt = transpose(b); return a.map(r => bt.map(c => dot(r, c))); };
  const sum = a => a.reduce((s, x) => s + x, 0);
  const mean = a => sum(a) / a.length;
  const norm = a => Math.sqrt(dot(a, a));
  function unit(a) { const n = norm(a); if (!(n > 1e-12) || !Number.isFinite(n)) throw Error('A finite, nonzero vector is needed before normalisation.'); return a.map(x => x / n); }
  function softmax(row) { const m = Math.max(...row), ex = row.map(x => Math.exp(x - m)), z = sum(ex); return ex.map(x => x / z); }
  const logsumexp = row => { const m = Math.max(...row); return m + Math.log(sum(row.map(x => Math.exp(x - m)))); };

  /* ---------- extra scenes for the batch-size demonstration (F..I); A..E come from vision-shared.js ---------- */
  function mug(g, c0) { for (let r = 1; r <= 3; r++) for (let c = c0; c < c0 + 3; c++) g[r][c] = 3; }
  function book(g) { for (let r = 5; r <= 6; r++) for (let c = 1; c <= 5; c++) g[r][c] = 2; }
  function plant(g) { g[4][7] = 1; }
  function blank() { const g = []; for (let r = 0; r < 8; r++) g.push([0, 0, 0, 0, 0, 0, 0, 0]); return g; }
  if (!V.scenes.F) { const g = blank(); mug(g, 1); mug(g, 5); V.scenes.F = g; }
  if (!V.scenes.G) { const g = blank(); book(g); plant(g); V.scenes.G = g; }
  if (!V.scenes.H) { const g = blank(); mug(g, 1); plant(g); V.scenes.H = g; }
  if (!V.scenes.I) { const g = blank(); mug(g, 5); book(g); V.scenes.I = g; }
  Object.keys(C.sceneNames || {}).forEach(k => { if (!V.sceneNames[k]) V.sceneNames[k] = C.sceneNames[k]; });

  /* ---------- the frozen Vision I encoder, exactly as Vision I shows it ---------- */
  V.frozen = scene => V.attend(V.scene(scene), ENC);
  V.encode = scene => V.frozen(scene).Enew;                       // 17 updated rows, CLS first
  V.classes = toy.encoderClasses || [];
  V.clsAxes = C.clsAxes || (toy.encoder && toy.encoder.axes) || ['brightness', 'contrast (left minus right)', 'row', 'col'];
  V.clsAxesShort = ['bright', 'contrast', 'row', 'col'];

  /* ---------- parameters ---------- */
  function params(which) {
    if (which == null || which === 'initial') return clone(C.snapshots.initial);
    if (typeof which === 'number') { if (!C.checkpoints[which]) throw Error('No checkpoint ' + which); return clone(C.checkpoints[which]); }
    if (!C.snapshots[which]) throw Error('Unknown CLIP snapshot: ' + which);
    return clone(C.snapshots[which]);
  }
  function validate(p) {
    if (!Array.isArray(p.W_img) || p.W_img.length !== 4 || !p.W_img.every(r => r.length === 3 && r.every(Number.isFinite))) throw Error('W_img must be 4x3.');
    if (!Array.isArray(p.W_txt) || p.W_txt.length !== C.vocab.length || !p.W_txt.every(r => r.length === 3 && r.every(Number.isFinite))) throw Error('W_txt must be ' + C.vocab.length + 'x3.');
    if (!Number.isFinite(p.log_scale)) throw Error('log_scale must be finite.');
  }
  function words(captions) {
    return captions.map(caption => {
      const row = Array(C.vocab.length).fill(0);
      String(caption).trim().toLowerCase().split(/\s+/).filter(Boolean).forEach(w => { const i = C.vocab.indexOf(w); if (i < 0) throw Error('Outside the toy vocabulary: ' + w); row[i] += 1; });
      if (!row.some(x => x)) throw Error('An empty caption has no vector.');
      return row;
    });
  }
  const cls = scene => V.encode(scene)[0];

  /* ---------- forward, gradients, one step ---------- */
  function encode(p, scenes, captions) {
    validate(p);
    if (!scenes.length) throw Error('Choose at least one image.');
    if (!captions.length) throw Error('Choose at least one caption.');
    const clsRows = scenes.map(cls), wordRows = words(captions);
    const imgRaw = mm(clsRows, p.W_img), txtRaw = mm(wordRows, p.W_txt);
    return { scenes: scenes.slice(), captions: captions.slice(), clsRows, wordRows, imgRaw, txtRaw, imgNorm: imgRaw.map(norm), txtNorm: txtRaw.map(norm), imgUnit: imgRaw.map(unit), txtUnit: txtRaw.map(unit) };
  }
  function forward(p, opts) {
    p = p || params(); opts = opts || {};
    const n = opts.pairs || C.trainPairs;
    const scenes = opts.scenes || C.pairs.slice(0, n).map(x => x.scene), captions = opts.captions || C.pairs.slice(0, n).map(x => x.caption);
    const f = encode(p, scenes, captions);
    const cosine = mm(f.imgUnit, transpose(f.txtUnit)), scale = Math.exp(p.log_scale), tau = 1 / scale;
    const logits = cosine.map(r => r.map(x => x * scale));
    const rowProb = logits.map(softmax), colProb = transpose(transpose(logits).map(softmax));
    const out = Object.assign(f, { cosine, scale, tau, logits, rowProb, colProb });
    if (scenes.length === captions.length) {
      out.rowLosses = logits.map((r, i) => logsumexp(r) - r[i]);
      out.colLosses = transpose(logits).map((c, j) => logsumexp(c) - c[j]);
      out.rowLoss = mean(out.rowLosses); out.colLoss = mean(out.colLosses); out.loss = (out.rowLoss + out.colLoss) / 2;
      out.diag = rowProb.map((r, i) => r[i]);
    }
    return out;
  }
  function gradients(p, opts) {
    const f = forward(p, opts), n = f.logits.length;
    if (f.logits[0].length !== n) throw Error('The contrastive loss needs one caption per image.');
    const dlogits = f.logits.map((r, i) => r.map((_, j) => (f.rowProb[i][j] + f.colProb[i][j] - (i === j ? 2 : 0)) / (2 * n)));
    const dcos = dlogits.map(r => r.map(x => x * f.scale));
    const du = mm(dcos, f.txtUnit), dt = mm(transpose(dcos), f.imgUnit);
    const undo = (up, u, len) => up.map((r, i) => { const radial = dot(r, u[i]); return r.map((x, j) => (x - u[i][j] * radial) / len[i]); });
    const dgImg = undo(du, f.imgUnit, f.imgNorm), dgTxt = undo(dt, f.txtUnit, f.txtNorm);
    return { W_img: mm(transpose(f.clsRows), dgImg), W_txt: mm(transpose(f.wordRows), dgTxt), log_scale: sum(dlogits.map((r, i) => dot(r, f.logits[i]))), forward: f };
  }
  function step(p, lr, opts) {
    p = p || params(); lr = lr == null ? C.learningRate : lr;
    if (!Number.isFinite(lr) || lr <= 0) throw Error('The learning rate must be positive.');
    const g = gradients(p, opts);
    return { W_img: p.W_img.map((r, i) => r.map((x, j) => x - lr * g.W_img[i][j])), W_txt: p.W_txt.map((r, i) => r.map((x, j) => x - lr * g.W_txt[i][j])), log_scale: p.log_scale - lr * g.log_scale };
  }
  function classify(opts) {
    opts = opts || {};
    const p = opts.params || params(opts.snapshot || 'trained'), scene = opts.scene || 'A', captions = opts.captions || C.pairs.slice(0, C.trainPairs).map(x => x.caption);
    const f = encode(p, [scene], captions), tau = opts.tau == null ? Math.exp(-p.log_scale) : opts.tau;
    if (!Number.isFinite(tau) || tau <= 0) throw Error('Temperature must be positive.');
    const cosine = f.txtUnit.map(t => dot(f.imgUnit[0], t)), logits = cosine.map(x => x / tau), probs = softmax(logits), best = cosine.indexOf(Math.max(...cosine));
    return Object.assign(f, { scene, tau, cosine, logits, probs, best, label: captions[best] });
  }
  function retrieve(caption, opts) {
    opts = opts || {};
    const p = opts.params || params(opts.snapshot || 'trained'), scenes = opts.scenes || C.pairs.slice(0, C.trainPairs).map(x => x.scene);
    const f = encode(p, scenes, [caption]);
    const cosine = f.imgUnit.map(u => dot(u, f.txtUnit[0]));
    return { caption, scenes, cosine, order: cosine.map((score, i) => ({ i, score })).sort((a, b) => b.score - a.score).map(x => x.i) };
  }

  /* ---------- readings and labels ---------- */
  const fmt = (x, d) => AT.fmt(x, d == null ? 2 : d);
  function readVec(v, axes, decimals) { axes = axes || C.axes; return v.map((x, i) => axes[i] + ' ' + fmt(x, decimals)).join(', '); }
  function readDirection(u, who) {
    const order = u.map((x, i) => ({ x, i })).sort((a, b) => b.x - a.x), top = order[0], next = order[1];
    const lead = C.axes[top.i], gap = top.x - next.x;
    let so = gap < 0.15 ? who + ' points almost equally to ' + lead + ' and ' + C.axes[next.i] : who + ' points mostly to ' + lead;
    const neg = order.filter(o => o.x < -0.2).map(o => C.axes[o.i]);
    if (neg.length) so += ' and away from ' + neg.join(' and ');
    return readVec(u, C.axes) + ', so ' + so + '.';
  }
  function thumb(scene, cell) { const g = V.grid(scene, { cell: cell || 4, patchLines: false }); g.classList.add('clip-thumb'); return g; }
  /* a label string for AT.ui.table: it must start with <span so the table treats it as markup */
  function thumbHTML(scene, cell) { const s = h('span', { class: 'clip-thumb', title: 'scene ' + scene }); s.appendChild(V.grid(scene, { cell: cell || 4, patchLines: false }).querySelector('svg')); const d = h('div'); d.appendChild(s); return d.innerHTML; }
  const sceneName = scene => V.sceneNames[scene] || scene;
  const pairLabel = (i, cell) => thumbHTML(C.pairs[i].scene, cell) + ' ' + AT.escape(sceneName(C.pairs[i].scene));
  const caption = i => C.pairs[i].caption;
  const quote = s => '“' + AT.escape(s) + '”';

  /* ---------- the projected sphere: a fixed camera looking down the (1,1,1) diagonal so all three axes show ---------- */
  const R2 = Math.SQRT2, R6 = Math.sqrt(6);
  function project(u) { return [(u[0] - u[1]) / R2, (u[0] + u[1] - 2 * u[2]) / R6]; }
  function points(state, opts) {
    opts = opts || {};
    const n = state.img.length, out = [];
    const imgLabels = opts.imgLabels || state.img.map((_, i) => sceneName(C.pairs[i].scene));
    const txtLabels = opts.txtLabels || state.txt.map((_, i) => '“' + caption(i) + '”');
    for (let i = 0; i < n; i++) out.push({ v: project(state.img[i]), kind: 'img', label: opts.labels === false ? '' : imgLabels[i] });
    for (let i = 0; i < state.txt.length; i++) out.push({ v: project(state.txt[i]), kind: 'txt', label: opts.labels === false ? '' : txtLabels[i] });
    return out;
  }
  function sphere(state, opts) {
    opts = opts || {};
    const size = opts.size || 420, box = V.circle(points(state, opts), { size, caption: opts.caption, into: opts.into, labels: opts.labels });
    const s = box.querySelector('svg'), c = size / 2, R = size / 2 - 34;
    const triad = svg('g', { class: 'clip-triad' });
    [[0, 'mug'], [1, 'book'], [2, 'plant']].forEach(([k, name]) => {
      const e = [0, 0, 0]; e[k] = 1; const p = project(e);
      const x = c + p[0] * R, y = c - p[1] * R;
      triad.appendChild(svg('line', { x1: c, y1: c, x2: x, y2: y, stroke: 'var(--ink-3)', 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }));
      triad.appendChild(svg('text', { x: c + p[0] * (R + 16), y: c - p[1] * (R + 16) + 4, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: 'var(--ink-2)' }, name));
    });
    const marks = s.querySelector('rect, path');
    s.insertBefore(triad, marks || null);
    box.setState = st => box.setPoints(points(st, opts));
    return box;
  }
  const state = t => C.trajectory[Math.max(0, Math.min(C.steps, Math.round(t)))];
  const stateOf = f => ({ img: f.imgUnit, txt: f.txtUnit });

  /* ---------- batch of the first n pairs ---------- */
  function batch(n, p) { n = Math.max(1, Math.min(C.pairs.length, n | 0)); return forward(p || params('trained'), { pairs: n }); }

  AT.clip = { data: clone(C), axes: C.axes, vocab: C.vocab, pairs: C.pairs, N: C.trainPairs, lr: C.learningRate, steps: C.steps, history: C.history, trajectory: C.trajectory,
    params, words, cls, encode, forward, gradients, step, classify, retrieve, batch, unit, softmax, project, points, sphere, state, stateOf,
    readVec, readDirection, thumb, thumbHTML, sceneName, pairLabel, caption, quote, fmt };

  /* ---------- objects, notation ---------- */
  const defs = {
    e: ['g_img', 'image vector', 'One vector for the whole image: the frozen encoder’s CLS row times W_img, then unit-normalised. Not an attention value.', '\\hat g^{\\mathrm{img}}'],
    q: ['g_txt', 'caption vector', 'One vector for the whole caption: the sum of its word rows, unit-normalised. Not an attention query.', '\\hat g^{\\mathrm{txt}}'],
    a: ['p', 'match probability', 'A softmax over the captions offered for one image (or the images offered for one caption). It compares the candidates on the list, nothing else.', 'p'],
    d: ['L', 'contrastive loss', 'The mean of the image-to-text and text-to-image cross-entropies on the observed pairs.', 'L']
  };
  (AT.objects || []).forEach(o => { const d = defs[o.cls]; if (d) { o.sym = d[0]; o.symTex = d[3]; o.name = d[1]; o.def = d[2]; o.tip = d[2]; } });
  const G1 = 'Image-text matching', G2 = 'Named coordinates (mug, book, plant)';
  function note(sym, mean, shape, dims, g) { AT.notation.push({ g: g || G1, sym, mean, shape, dims: () => dims, parts: ['vision3'] }); }
  note('e_0^{\\prime}', 'The frozen Vision I encoder’s updated CLS row for a scene: brightness, contrast, row, col', '1\\times 4', '1×4');
  note('\\ve{g^{\\mathrm{img}}} = e_0^{\\prime} W_{\\mathrm{img}}', 'The image vector on the joint axes', '1\\times d', '1×3');
  note('\\vq{g^{\\mathrm{txt}}} = \\sum_{w \\in \\text{caption}} W_{\\mathrm{txt}}[w]', 'The caption vector: its word rows added up (word order is ignored)', '1\\times d', '1×3');
  note('\\hat g = g / \\lVert g \\rVert', 'Unit length; only the direction is compared', '1\\times d', '1×3');
  note('C_{ij} = \\ve{\\hat g_i^{\\mathrm{img}}} \\cdot \\vq{\\hat g_j^{\\mathrm{txt}}}', 'Cosine similarity of image i and caption j', '\\text{scalar}', '−1 to 1');
  note('S_{ij} = C_{ij}/\\tau', 'The logit; τ is the learned temperature', '\\text{scalar}', '');
  note('\\va{p(T_j \\mid I_i)}', 'Row softmax: image i chooses among the captions in the batch', '\\text{scalar}', '0 to 1');
  note('\\va{p(I_i \\mid T_j)}', 'Column softmax: caption j chooses among the images in the batch', '\\text{scalar}', '0 to 1');
  note('\\vd{L} = \\tfrac12 (L_{I \\to T} + L_{T \\to I})', 'Mean of the two cross-entropies over the N observed pairs', '\\text{scalar}', '');
  note('\\text{mug}, \\text{book}, \\text{plant}', 'The three joint coordinates. Rows of W_txt and columns of W_img are named by them; a caption vector reads as “mug 0.98, book −0.01, plant −0.13”', '', '', G2);
  note('N, d', 'Pairs in the batch; width of the joint space', '', '3 pairs (8 in the batch demo); 3', 'sizes');
  note('W_{\\mathrm{img}}, W_{\\mathrm{txt}}', 'Learned image map (rows: brightness, contrast, row, col) and word rows (one per vocabulary word)', '4\\times 3,\\; 14\\times 3', '54 parameters', 'sizes');
  note('a = \\log(1/\\tau)', 'Learned log scale; logits are exp(a) times cosine', '\\text{scalar}', '1 parameter', 'sizes');
  const originalCard = AT.ui.notationCard;
  AT.ui.notationCard = function (opts) { opts = opts || {}; const part = opts.part || (window.__PART__ && window.__PART__.notation); return originalCard(Object.assign({}, opts, part === 'vision3' && !opts.groups ? { groups: [G1, G2, 'sizes'] } : {})); };

  /* ---------- fragment-shared CSS ---------- */
  document.head.appendChild(h('style', {}, [
    '.clip-thumb{display:inline-block;vertical-align:middle;margin-right:4px}.clip-thumb svg{display:block;border:1px solid var(--line)}',
    '.dt th .clip-thumb{margin-right:6px}',
    'body.present .clip-thumb svg{width:40px;height:40px}',
    '.clip-triad text{font-family:var(--font-ui)}',
    '.clip-read{font-size:15px;color:var(--ink-2);margin:10px 0 0;max-width:72ch;line-height:1.5}',
    'body.present .clip-read{font-size:22px;line-height:1.35}',
    'body.present .dt-cap,body.present .dt-note{display:none}',
    '.clip-pairs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;align-items:start}',
    '@media (max-width:640px){.clip-pairs{grid-template-columns:minmax(0,1fr)}}',
    '.bars .bl{white-space:normal}',
    '.clip-pair{display:grid;gap:8px;justify-items:center;text-align:center}',
    '.clip-pair .clip-cap{font-weight:600;color:var(--c-q);font-size:15px}',
    '.clip-pair .clip-scene{font-size:13px;color:var(--ink-3)}',
    'body.present .clip-pair .clip-cap{font-size:24px}body.present .clip-pair .clip-scene{font-size:20px}',
    '.clip-two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;align-items:start}',
    '@media (max-width:820px){.clip-two{grid-template-columns:1fr}}',
    'body.present .clip-two{gap:28px}'
  ].join('\n')));
})();

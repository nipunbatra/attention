/* part8.js: Vision IV runtime on toy8.json v2. Requires shared.js and vision-shared.js.
   The image encoder is exactly the one Vision I shows: V.attend with toy8.encoder (a copy of toy5 "trained").
   Its sixteen updated patch rows enter a width-three prefix decoder through a fitted connector. */
(function () {
  'use strict';
  var AT = window.AT, V = AT.vision, T = window.__TOY__ || {}, M = T.vlm;
  if (!V || !M || !T.encoder) return;
  var h = AT.h, svg = AT.svg;
  var NP = M.imageRows, TP = M.textPositions, VOC = M.vocab, PROMPT = M.prompt, D = T.d_model;
  var cp = function (x) { return JSON.parse(JSON.stringify(x)); };

  /* ---------- the frozen encoder ---------- */
  V.model = T;
  V.encoder = T.encoder;
  V.encode = function (scene) { return V.attend(V.scene(scene), T.encoder).Enew; };   /* 17 rows, CLS first */
  V.encoderAxes = T.encoderAxes || { e: V.axes.e, short: V.axes.short.e };

  /* ---------- small exact linear algebra (row vectors) ---------- */
  function mm(A, B) { return A.map(function (r) { return B[0].map(function (_, c) { var s = 0; for (var k = 0; k < r.length; k++) s += r[k] * B[k][c]; return s; }); }); }
  function tr(A) { return A[0].map(function (_, j) { return A.map(function (r) { return r[j]; }); }); }
  function addRows(A, B) { return A.map(function (r, i) { return r.map(function (x, k) { return x + B[i][k]; }); }); }
  function sm(v) { var m = -Infinity; v.forEach(function (x) { if (x > m) m = x; }); var ex = v.map(function (x) { return x === -Infinity ? 0 : Math.exp(x - m); }); var z = ex.reduce(function (a, b) { return a + b; }, 0); return ex.map(function (x) { return x / z; }); }
  function sum(v) { return v.reduce(function (a, b) { return a + b; }, 0); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  /* ---------- the decoder ---------- */
  function params(s) { s = s || M.defaultSnapshot; if (!M.snapshots[s]) throw new Error('Unknown snapshot: ' + s); return cp(M.snapshots[s]); }
  function rows(scene) { return V.encode(scene).slice(1); }
  function checkPrefix(prefix) {
    if (!Array.isArray(prefix) || !prefix.length || prefix.length > TP || prefix.some(function (t) { return VOC.indexOf(t) < 0; })) throw new Error('Unsupported text prefix.');
  }
  function forward(scene, prefix, opts) {
    opts = opts || {}; prefix = prefix || PROMPT;
    if (typeof scene === 'string' && !V.scenes[scene]) throw new Error('Unknown scene: ' + scene);
    checkPrefix(prefix);
    var p = opts.params ? cp(opts.params) : params(opts.snapshot), G = rows(scene), ids = prefix.map(function (t) { return VOC.indexOf(t); }), Tn = ids.length;
    var B = mm(G, p.W_bridge).map(function (r) { return r.map(function (x, c) { return x + p.b_bridge[c]; }); });
    var Et = ids.map(function (id, i) { return p.E_tok[id].map(function (x, c) { return x + p.P[i][c]; }); });
    var E = B.concat(Et), n = NP + Tn;
    var Q = mm(E, p.W_Q), K = mm(E, p.W_K), Vv = mm(E, p.W_V);
    var allowed = E.map(function (_, i) { return E.map(function (_, j) { return i < NP ? j < NP : j <= i; }); });
    var raw = mm(Q, tr(K));
    var scores = raw.map(function (r, i) { return r.map(function (x, j) { return allowed[i][j] ? x / Math.sqrt(D) : -Infinity; }); });
    var A = scores.map(sm), msg = mm(A, Vv), delta = mm(msg, p.W_O), out = addRows(E, delta);
    var logits = mm(out, p.W_vocab).map(function (r) { return r.map(function (x, c) { return x + p.b_vocab[c]; }); });
    return { scene: scene, snapshot: opts.snapshot || M.defaultSnapshot, prefix: prefix.slice(), ids: ids, n: n, T: Tn, last: n - 1,
      G: G, B: B, E: E, Q: Q, K: K, V: Vv, raw: raw, scores: scores, allowed: allowed, A: A, message: msg, delta: delta, out: out, logits: logits, probs: logits.map(sm) };
  }
  function teacher(scene, opts) {
    var answer = M.answers[scene]; if (!answer) throw new Error('No answer for scene ' + scene);
    var f = forward(scene, PROMPT.concat(answer.slice(0, -1)), opts);
    var rowsOf = answer.map(function (_, t) { return NP + PROMPT.length - 1 + t; });
    var losses = answer.map(function (tok, t) { return -Math.log(f.probs[rowsOf[t]][VOC.indexOf(tok)]); });
    f.targets = answer.slice(); f.targetRows = rowsOf; f.losses = losses; f.loss = sum(losses) / losses.length;
    f.targetProbs = answer.map(function (tok, t) { return f.probs[rowsOf[t]][VOC.indexOf(tok)]; });
    return f;
  }
  function generate(scene, opts) {
    opts = opts || {};
    var limit = opts.limit == null ? TP - PROMPT.length : opts.limit;
    if (!Number.isInteger(limit) || limit < 1 || limit > TP - PROMPT.length) throw new Error('Generation limit must be 1 through ' + (TP - PROMPT.length) + '.');
    var prefix = (opts.prefix || PROMPT).slice(), trace = [], stoppedBy = 'limit';
    for (var i = 0; i < limit; i++) {
      var f = forward(scene, prefix, opts), row = f.last, chosen = VOC[AT.argmax(f.probs[row])];
      trace.push({ prefix: prefix.slice(), row: row, query: f.Q[row], weights: f.A[row], logits: f.logits[row], probs: f.probs[row], chosen: chosen, forward: f });
      prefix.push(chosen);
      if (chosen === '<eos>') { stoppedBy = 'eos'; break; }
    }
    return { tokens: prefix.slice((opts.prefix || PROMPT).length), trace: trace, stoppedBy: stoppedBy };
  }
  /* where one query's weight went: image versus text, mug patches versus the rest */
  function mass(weights, scene) {
    var img = weights.slice(0, NP), txt = weights.slice(NP), mugs = V.mugPatches(scene), left = V.mugPatches(scene, 'left'), right = V.mugPatches(scene, 'right');
    var pick = function (idx) { return sum(idx.map(function (j) { return img[j]; })); };
    var order = AT.range(NP).sort(function (a, b) { return img[b] - img[a]; });
    return { image: sum(img), text: sum(txt), mug: pick(mugs), left: pick(left), right: pick(right), order: order, top: order.slice(0, 4).map(function (j) { return { j: j, w: img[j], name: V.regionOf(scene, j) }; }) };
  }
  /* exact bookkeeping of logit(two) - logit(one) for the last row: residual term plus one term per source */
  function contrast(scene, prefix, opts) {
    var f = forward(scene, prefix, opts), p = params((opts || {}).snapshot), i = f.last, a = VOC.indexOf('two'), b = VOC.indexOf('one');
    var u = p.W_vocab.map(function (r) { return r[a] - r[b]; });
    var sources = f.V.map(function (v, j) { var w = v.map(function (x) { return x * f.A[i][j]; }), up = mm([w], p.W_O)[0]; return { j: j, weight: f.A[i][j], weighted: w, update: up, term: dot(up, u) }; });
    var residual = dot(f.E[i], u) + p.b_vocab[a] - p.b_vocab[b];
    return { f: f, row: i, sources: sources, residual: residual, image: sum(sources.slice(0, NP).map(function (s) { return s.term; })), text: sum(sources.slice(NP).map(function (s) { return s.term; })), total: f.logits[i][a] - f.logits[i][b] };
  }
  /* row labels for a stacked table: thumbnails for the image rows, then the text tokens with their positions */
  /* Image-row labels carry thumbnail markup (rendered as html); text-row labels are plain text. Escape a text label yourself when you embed it in html. */
  function labels(scene, prefix, size) { return V.rowLabels(scene, { size: size || 22 }).concat((prefix || PROMPT).map(function (t, i) { return textLabel(t, i); })); }
  function textLabel(t, i) { return (NP + i + 1) + ' ' + t; }

  /* ---------- readings: sentences computed from the numbers ---------- */
  var fmt = function (x, d) { return AT.fmt(x, d == null ? 2 : d); };
  function joinAnd(xs) { return xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]; }
  function readWeights(scene, weights) {
    var m = mass(weights, scene);
    var top = m.top.filter(function (t) { return t.w >= 0.05; });
    var s = 'image rows ' + fmt(m.image) + ', text rows ' + fmt(m.text);
    if (top.length) s += '; the largest: ' + joinAnd(top.map(function (t) { return t.name + ' ' + fmt(t.w); }));
    s += '; mug patches together ' + fmt(m.mug) + '.';
    return s;
  }
  function readQuery(q) {
    var ax = AT.axes.qk;
    var parts = q.map(function (x, c) { return { x: x, c: c }; }).filter(function (o) { return Math.abs(o.x) >= 0.5; }).sort(function (a, b) { return Math.abs(b.x) - Math.abs(a.x); });
    if (!parts.length) return 'every entry is small, so the query asks for little.';
    /* an axis named "−bright" with a negative entry asks for bright patches: fold the sign into the name and merge repeats */
    var want = {}, order = [];
    parts.forEach(function (o) { var neg = ax[o.c].charAt(0) === '−', name = neg ? ax[o.c].slice(1) : ax[o.c]; var s = neg ? -o.x : o.x; if (!(name in want)) { want[name] = 0; order.push(name); } want[name] += s; });
    order.sort(function (a, b) { return Math.abs(want[b]) - Math.abs(want[a]); });
    var asks = order.map(function (name) { return (want[name] > 0 ? '' : 'against ') + name + ' (' + fmt(Math.abs(want[name]), 1) + ')'; });
    return parts.map(function (o) { return ax[o.c] + ' ' + fmt(o.x); }).join(', ') + ', so it asks for ' + joinAnd(asks) + '.';
  }

  /* ---------- figures ---------- */
  document.head.appendChild(h('style', {}, [
    '.v4-row{display:flex;flex-wrap:wrap;align-items:flex-end;gap:10px 14px}',
    '.v4-thumbs{display:flex;flex-wrap:wrap;gap:3px;align-items:flex-end}',
    '.v4-th{display:flex;flex-direction:column;align-items:center;gap:1px;font-family:var(--font-mono);font-size:10px;color:var(--ink-3)}',
    '.v4-th svg{display:block;margin:0}',
    '.v4-th.is-hl svg{outline:3px solid var(--c-q);outline-offset:-1px}',
    '.v4-row .chips{margin:0}',
    '.v4-group{display:flex;flex-direction:column;gap:4px}',
    '.v4-group-lab{font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-weight:700;color:var(--ink-3)}',
    '.v4-answer{display:inline-flex;align-items:center;gap:6px;border:2px solid var(--c-d);background:var(--t-d);border-radius:999px;padding:4px 12px;font-weight:600}',
    '.v4-read{font-size:15px;color:var(--ink-2);margin:10px 0 0;max-width:72ch;line-height:1.5}',
    '.v4-mask{width:max-content;max-width:100%}',
    '.v4-mask svg{display:block;max-width:100%;height:auto}',
    '.v4-fig-col{max-width:460px;min-width:0}',
    '.v4-mask .cell{stroke:var(--card);stroke-width:1}',
    '.v4-mask .on{fill:var(--t-a)}.v4-mask .off{fill:var(--paper)}',
    '.v4-mask text{font-family:var(--font-ui);fill:var(--ink-2)}',
    'body.present .v4-read{font-size:22px;line-height:1.35}',
    'body.present .v4-th{font-size:14px}',
    'body.present .v4-group-lab{font-size:16px}',
    'body.present .v4-answer{font-size:24px}'
  ].join('\n')));
  function put(el, opts) { if (opts && opts.into) { var t = typeof opts.into === 'string' ? document.getElementById(opts.into) : opts.into; if (t) t.appendChild(el); } return el; }

  /* tokenRow(scene, tokens, {size, answer, slot, active, highlight:[patches], labels, into}): sixteen patch thumbnails then the text chips */
  function tokenRow(scene, tokens, opts) {
    opts = opts || {}; tokens = tokens || PROMPT;
    var row = h('div', { class: 'v4-row' });
    var thumbs = h('div', { class: 'v4-thumbs' });
    for (var j = 0; j < NP; j++) {
      var t = h('span', { class: 'v4-th' + ((opts.highlight || []).indexOf(j) >= 0 ? ' is-hl' : ''), title: V.patchLabel(scene, j) }, V.thumb(scene, j, { size: opts.size || 26 }));
      if (opts.labels !== false) t.appendChild(h('span', {}, String(j + 1)));
      thumbs.appendChild(t);
    }
    row.appendChild(h('div', { class: 'v4-group' }, opts.groupLabels === false ? null : h('span', { class: 'v4-group-lab' }, opts.imageLabel || (NP + ' image rows')), thumbs));
    var chipOpts = { numbered: false };
    if (opts.active != null) chipOpts.active = opts.active;
    if (opts.slot) chipOpts.slot = opts.slot;
    var chips = AT.ui.chips(tokens, chipOpts);
    row.appendChild(h('div', { class: 'v4-group' }, opts.groupLabels === false ? null : h('span', { class: 'v4-group-lab' }, opts.textLabel || (tokens.length + ' text rows')), chips));
    if (opts.answer) row.appendChild(h('div', { class: 'v4-group' }, opts.groupLabels === false ? null : h('span', { class: 'v4-group-lab' }, 'next token'), h('span', { class: 'v4-answer' }, opts.answer)));
    row.chips = chips; row.thumbs = thumbs;
    return put(row, opts);
  }

  /* maskFigure(scene, tokens, {cell}): who may read whom, with the patch thumbnails and tokens on both axes */
  function maskFigure(scene, tokens, opts) {
    opts = opts || {}; tokens = tokens || PROMPT;
    var f = forward(scene, tokens, opts), n = f.n, cell = opts.cell || 16, pad = 42, W = pad + n * cell + 4;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + W, width: W, height: W, role: 'img', 'aria-label': 'Attention mask: image rows read image rows only; text rows read every image row and the text so far' });
    var g = V.scene(scene);
    function mini(x, y, j, size) { var pr = Math.floor(j / 4), pc = j % 4, c = size / 2; for (var r = 0; r < 2; r++) for (var cc = 0; cc < 2; cc++) s.appendChild(svg('rect', { x: x + cc * c, y: y + r * c, width: c, height: c, fill: V.ramp(g[2 * pr + r][2 * pc + cc]) })); }
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) s.appendChild(svg('rect', { x: pad + j * cell, y: pad + i * cell, width: cell, height: cell, class: 'cell ' + (f.allowed[i][j] ? 'on' : 'off') }));
      if (i < NP) { mini(pad - cell - 4, pad + i * cell + 1, i, cell - 2); mini(pad + i * cell + 1, pad - cell - 4, i, cell - 2); }
      else {
        var tok = tokens[i - NP];
        s.appendChild(svg('text', { x: pad - 6, y: pad + i * cell + cell * 0.72, 'text-anchor': 'end', 'font-size': Math.round(cell * 0.7) }, tok));
        s.appendChild(svg('text', { x: pad + i * cell + cell / 2, y: pad - 6, 'text-anchor': 'end', 'font-size': Math.round(cell * 0.7), transform: 'rotate(-90 ' + (pad + i * cell + cell / 2) + ' ' + (pad - 6) + ')' }, tok));
      }
    }
    s.appendChild(svg('line', { x1: pad + NP * cell, y1: pad, x2: pad + NP * cell, y2: pad + n * cell, stroke: 'var(--ink)', 'stroke-width': 1.5 }));
    s.appendChild(svg('line', { x1: pad, y1: pad + NP * cell, x2: pad + n * cell, y2: pad + NP * cell, stroke: 'var(--ink)', 'stroke-width': 1.5 }));
    var box = h('div', { class: 'v4-mask' }, s);
    if (opts.caption) { var fig = h('figure', { class: 'vfig' }, box, h('figcaption', { html: opts.caption })); return put(fig, opts); }
    return put(box, opts);
  }

  /* ---------- notation ---------- */
  function note(g, sym, mean, shape, dims) { AT.notation.push({ g: g, sym: sym, mean: mean, shape: shape, dims: function () { return dims || ''; }, parts: ['vision4'] }); }
  if (AT.notation && AT.notation.push) {
    note('token', 'g_j', 'Frozen visual row of patch j: the Vision I encoder’s updated patch row, on the encoder axes brightness, contrast, row, col', '1\\times d_{\\mathrm{vision}}', '1×4');
    note('token', '\\ve{e_j}=g_jW_{\\mathrm{bridge}}+b_{\\mathrm{bridge}}', 'An image row after the connector: the same map for all sixteen patches', '1\\times d_{\\mathrm{model}}', '1×3');
    note('token', '\\ve{e_i}=E_{\\mathrm{tok}}[t_i]+p_i', 'A text row: vocabulary lookup plus a learned position vector for text position i', '1\\times d_{\\mathrm{model}}', '1×3');
    note('token', '\\vq{q_i}=\\ve{e_i}W_Q,\\ \\vk{k_j}=\\ve{e_j}W_K,\\ \\vv{v_j}=\\ve{e_j}W_V', 'Ask, offer, send; one set of matrices for image and text rows', '1\\times 3', '');
    note('token', 's_{ij}=\\vq{q_i}\\cdot\\vk{k_j}/\\sqrt{d_k},\\ \\va{\\alpha_{ij}}=\\operatorname{softmax}_j', 'Scores over the rows the mask allows; masked scores are −∞', '', '');
    note('token', 'm_i=\\sum_j\\va{\\alpha_{ij}}\\vv{v_j},\\ \\vd{\\Delta e_i}=m_iW_O,\\ \\vp{e_i^\\prime}=\\ve{e_i}+\\vd{\\Delta e_i}', 'The message, the update and the updated row', '1\\times d_{\\mathrm{model}}', '1×3');
    note('token', 'z_i=\\vp{e_i^\\prime}W_{\\mathrm{vocab}}+b', 'Next-token logits from the last known row; softmax over the vocabulary', '1\\times|\\mathcal V|', '1×10');
    note('matrix', 'G', 'The sixteen frozen visual rows (CLS dropped after the encoder’s attention)', 'N\\times d_{\\mathrm{vision}}', '16×4');
    note('matrix', '\\ve{E}', 'Image rows through the connector, then the known text rows', '(N+T)\\times d_{\\mathrm{model}}', '21×3 for the prompt');
    note('matrix', 'M', 'Who may read whom: image rows read image rows; text rows read all image rows and earlier text', '(N+T)\\times(N+T)', '21×21 for the prompt');
    note('matrix', '\\vq{Q},\\vk{K},\\vv{V},\\va{A},\\vd{\\Delta E},\\vp{E^\\prime}', 'As in Part 2, for all rows at once', '(N+T)\\times 3', '');
    note('sizes', 'N,\\ T,\\ d_{\\mathrm{vision}},\\ d_{\\mathrm{model}}=d_k=d_v', 'Image rows, text rows, encoder width, decoder width', '', '16, 5 to 8, 4, 3');
    note('sizes', 'W_{\\mathrm{bridge}},\\ b_{\\mathrm{bridge}}', 'The connector (fitted)', 'd_{\\mathrm{vision}}\\times d_{\\mathrm{model}}', '4×3, bias 1×3');
    note('sizes', 'E_{\\mathrm{tok}},\\ P', 'Text lookup table and text positions (fitted)', '|\\mathcal V|\\times d_{\\mathrm{model}},\\ 8\\times d_{\\mathrm{model}}', '10×3, 8×3');
    note('sizes', 'W_Q,W_K,W_V,W_O,\\ W_{\\mathrm{vocab}},\\ b', 'One attention head and the head (fitted); the encoder’s own matrices stay frozen', '3\\times3,\\ 3\\times|\\mathcal V|', '3×3, 3×10');
  }
  /* the shell's notation card: the standard groups, then the learned decoder axes with the rule that named them */
  var baseCard = AT.ui.notationCard;
  function learnedAxes() {
    var box = h('div', { class: 'notation-group' });
    box.appendChild(h('p', { class: 'notation-title' }, 'Learned coordinates (named after the fit)'));
    var table = h('table', { class: 'dt dt-notation' });
    table.appendChild(h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'object'), h('th', { scope: 'col' }, 'names'), h('th', { scope: 'col' }, 'what each one tracks'))));
    var body = h('tbody');
    var full = (T.axes && T.axes.full) || {};
    [['\\ve{e}', 'e'], ['\\vq{q},\\ \\vk{k}', 'qk'], ['\\vv{v}', 'v']].forEach(function (pair) {
      var ts = h('th', { scope: 'row', class: 'nt-sym' }); AT.tex(ts, pair[0]);
      body.appendChild(h('tr', {}, ts, h('td', { class: 'nt-mean' }, AT.axes[pair[1]].join(' · ')), h('td', { class: 'nt-mean' }, (full[pair[1]] || []).map(function (s) { return s.replace(/ across the 16 patches$/, ''); }).join('; '))));
    });
    table.appendChild(body);
    box.appendChild(h('div', { class: 'dt-scroll' }, table));
    box.appendChild(h('p', { class: 'notation-note' }, 'The decoder’s three coordinates were not designed. After fitting, each one is named by the pixel quantity it tracks across the sixteen image rows of scene A (brightness, contrast, patch row, patch column); r is the correlation. Text rows use the same coordinates without such a reading.'));
    return box;
  }
  AT.ui.notationCard = function (opts) {
    opts = opts || {};
    var part = opts.part || (window.__PART__ && window.__PART__.notation) || 'part2';
    if (part !== 'vision4' || opts.groups) return baseCard(opts);
    var root = baseCard({ part: part, groups: ['token', 'matrix', 'sizes'], only: opts.only, compact: opts.compact });
    root.appendChild(learnedAxes());
    return put(root, opts);
  };

  AT.vlm = { data: M, toy: T, NP: NP, TP: TP, vocab: VOC, prompt: PROMPT, answers: M.answers, sceneNames: M.sceneNames, scenes: ['A', 'B', 'C'],
    params: params, rows: rows, forward: forward, teacher: teacher, generate: generate, mass: mass, contrast: contrast, labels: labels, textLabel: textLabel,
    readWeights: readWeights, readQuery: readQuery, tokenRow: tokenRow, maskFigure: maskFigure, fmt: fmt, joinAnd: joinAnd, mm: mm, sum: sum, dot: dot, softmax: sm };
})();

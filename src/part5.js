/* part5.js: Vision I runtime on toy5.json (v2, named axes). Requires shared.js and vision-shared.js. */
(function () {
  'use strict';
  var AT = window.AT, V = AT.vision, T = window.__TOY__ || {};
  if (!V || !T.scenes) return;
  var h = AT.h;
  V.model = T; V.classes = T.classes; V.trainingCurve = T.curve || [];
  V.params = function (which) { return which === 'initial' ? T.initial : T.trained; };
  /* forward(scene, which): scene = 'A'..'E' or an 8x8 array; which = 'initial' | 'trained' (default trained) */
  V.forward = function (scene, which) { return V.attend(V.scene(scene), V.params(which || 'trained')); };
  V.encode = function (scene, which) { return V.forward(scene, which).Enew; };     // 17 updated rows (CLS first): the frozen encoder later parts reuse
  V.attention = function (scene, which, receiver) { var A = V.forward(scene, which).A; return receiver == null ? A[0] : A[receiver]; };
  V.predict = function (scene, which) { var f = V.forward(scene, which); var i = f.probs[0] >= f.probs[1] ? 0 : 1; return { label: T.classes[i], probs: f.probs, logits: f.logits }; };
  V.regions = function (scene) { var g = V.scene(scene), o = []; for (var j = 0; j < 16; j++) o.push(V.regionOf(g, j)); return o; };
  V.labelOf = function (key) { return T.scenes[key] && T.scenes[key].label; };
  V.labelIndex = function (key) { return T.classes.indexOf(V.labelOf(key)); };
  V.tokensPerImage = function (side, patch) { var n = Math.floor(side / patch); return { patches: n * n, rows: n * n + 1, scores: (n * n + 1) * (n * n + 1) }; };
  /* cross-entropy of the labelled class for a named scene, and the mean over the training scenes */
  V.loss = function (key, which) { var f = V.forward(key, which); return -Math.log(f.probs[V.labelIndex(key)]); };
  V.meanLoss = function (which, keys) { keys = keys || ['A', 'B']; return keys.reduce(function (s, k) { return s + V.loss(k, which); }, 0) / keys.length; };
  V.trainScenes = ['A', 'B']; V.probeScene = 'C';
  /* the four pixels of a patch in the order patchify reads them */
  V.pixelAxes = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
  V.pixelAxesShort = ['TL', 'TR', 'BL', 'BR'];

  /* ---------- small linear algebra, local to this file ---------- */
  var mm = function (A, B) { return A.map(function (r) { return B[0].map(function (_, c) { return r.reduce(function (s, x, k) { return s + x * B[k][c]; }, 0); }); }); };
  var add = function (A, B) { return A.map(function (r, i) { return r.map(function (x, k) { return x + B[i][k]; }); }); };
  var sm = function (v) { var m = Math.max.apply(null, v), ex = v.map(function (x) { return Math.exp(x - m); }), z = ex.reduce(function (a, b) { return a + b; }, 0); return ex.map(function (x) { return x / z; }); };

  /* swapPatches(scene, i, j): a copy of the scene with the pixels of patches i and j (0-based) exchanged */
  V.swapPatches = function (scene, i, j) {
    var g = V.copyScene(scene), pi = [Math.floor(i / 4), i % 4], pj = [Math.floor(j / 4), j % 4];
    for (var r = 0; r < 2; r++) for (var c = 0; c < 2; c++) {
      var a = g[2 * pi[0] + r][2 * pi[1] + c]; g[2 * pi[0] + r][2 * pi[1] + c] = g[2 * pj[0] + r][2 * pj[1] + c]; g[2 * pj[0] + r][2 * pj[1] + c] = a;
    }
    return g;
  };
  /* forwardOpts(scene, which, {positions:false, order:[16 patch indices]}): the same layer with the position rows left out
     and/or the content rows placed in another order (slot s receives patch order[s]); the check behind the swap frame */
  V.forwardOpts = function (scene, which, opts) {
    opts = opts || {}; var P = V.params(which || 'trained');
    var R = V.patchify(scene); if (opts.order) R = opts.order.map(function (j) { return R[j]; });
    var E = [V.cls.slice()].concat(mm(R, V.W_patch));
    if (opts.positions !== false) E = add(E, V.pos);
    var Q = mm(E, P.W_Q), K = mm(E, P.W_K), Vv = mm(E, P.W_V);
    var S = Q.map(function (q) { return K.map(function (k) { return (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2; }); });
    var A = S.map(sm), H = mm(A, Vv), D = mm(H, P.W_O), En = add(E, D);
    var out = { R: R, E: E, Q: Q, K: K, V: Vv, S: S, A: A, H: H, Delta: D, Enew: En };
    if (P.W_cls) { var l = P.b_cls.map(function (b, c) { return b + En[0].reduce(function (s, x, k) { return s + x * P.W_cls[k][c]; }, 0); }); out.logits = l; out.probs = sm(l); }
    return out;
  };
  /* attendWith(scene, which, overrides): the stored parameters with some matrices replaced (the value-only intervention) */
  V.attendWith = function (scene, which, overrides) {
    var P = V.params(which || 'trained'), Q = {}; Object.keys(P).forEach(function (k) { Q[k] = P[k]; }); Object.keys(overrides || {}).forEach(function (k) { Q[k] = overrides[k]; });
    return V.attend(V.scene(scene), Q);
  };

  /* ---------- readings computed from the numbers ---------- */
  /* rowName(scene, i): i is a row index of E (0 = CLS, 1..16 = patches) */
  V.rowName = function (scene, i) { return i === 0 ? 'CLS' : V.patchLabel(scene, i - 1); };
  V.rowLabelRow = function (scene, i, opts) { return i === 0 ? 'CLS' : V.rowLabel(scene, i - 1, opts); };
  V.rowLabelsE = function (scene, opts) { var o = ['CLS']; for (var j = 0; j < 16; j++) o.push(V.rowLabel(scene, j, opts)); return o; };
  /* topSources(alpha, scene, n): the n largest entries of a 17-entry attention row, largest first */
  V.topSources = function (alpha, scene, n) {
    var idx = alpha.map(function (a, i) { return i; }).sort(function (a, b) { return alpha[b] - alpha[a]; }).slice(0, n || 3);
    return idx.map(function (i) { return { i: i, label: V.rowName(scene, i), a: alpha[i] }; });
  };
  V.readAttention = function (alpha, scene, n, dec) {
    var top = V.topSources(alpha, scene, n || 3), d = dec == null ? 2 : dec;
    var parts = top.map(function (t) { return t.label + ' ' + AT.fmt(t.a, d); });
    var rest = alpha.length - top.length, restSum = 1 - top.reduce(function (s, t) { return s + t.a; }, 0);
    return parts.length < 2 ? parts.join('') : parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1] + (rest > 0 ? '; the other ' + rest + ' rows share ' + AT.fmt(restSum, d) : '');
  };
  /* the mass of an attention row on the patches of one region family ('mug' | 'right mug' | 'book' | 'table' | 'plant') */
  V.regionMass = function (alpha, scene, name) { var s = 0; for (var j = 0; j < 16; j++) if (V.regionOf(scene, j).indexOf(name) >= 0) s += alpha[j + 1]; return s; };

  /* ---------- a patch picker: CLS plus the sixteen patches as thumbnail buttons ---------- */
  var css = h('style', {}, [
    '.vpick{display:flex;gap:10px 14px;align-items:center;flex-wrap:wrap;margin:6px 0 10px}',
    '.vpick-grid{display:grid;grid-template-columns:repeat(4,auto);gap:3px}',
    '.vpick-b{display:inline-flex;align-items:center;gap:4px;padding:3px 6px 3px 4px;border:1.5px solid var(--line);border-radius:7px;background:var(--card);font:inherit;font-size:12px;color:var(--ink-2);cursor:pointer;font-variant-numeric:tabular-nums}',
    '.vpick-b .vthumb{margin:0}',
    '.vpick-b[aria-pressed=true]{border-color:var(--ink);color:var(--ink);box-shadow:0 0 0 1px var(--ink)}',
    '.vpick-b:focus-visible{outline:2px solid var(--ink);outline-offset:2px}',
    '.vpick-cls{min-height:34px;padding:3px 10px;font-weight:600}',
    '.vpick-lab{font-size:13px;color:var(--ink-3);font-weight:600;margin-right:2px}',
    'body.present .vpick-b{font-size:18px;padding:3px 8px 3px 5px}body.present .vpick-lab{font-size:20px}body.present .vpick{gap:10px 18px;margin:0 0 8px}body.present .vpick-grid{grid-template-columns:repeat(8,auto)}'
  ].join('\n'));
  document.head.appendChild(css);
  /* picker(scene, {cls:true, active:0, size:18, label, onClick(i), into}) → element with setActive(i), setScene(scene), buttons */
  V.picker = function (scene, opts) {
    opts = opts || {}; var g = V.scene(scene), size = opts.size || 18, active = opts.active == null ? 0 : opts.active, withCls = opts.cls !== false;
    var root = h('div', { class: 'vpick', role: 'group', 'aria-label': opts.label || 'Choose a row' }), buttons = [];
    if (opts.label) root.appendChild(h('span', { class: 'vpick-lab' }, opts.label));
    function press(i) { buttons.forEach(function (b, k) { if (b) b.setAttribute('aria-pressed', k === i ? 'true' : 'false'); }); }
    function make(i) {
      var b = h('button', { type: 'button', class: 'vpick-b' + (i === 0 ? ' vpick-cls' : ''), 'aria-pressed': 'false', title: V.rowName(g, i) });
      if (i === 0) b.textContent = 'CLS'; else { b.appendChild(V.thumb(g, i - 1, { size: size })); b.appendChild(h('span', {}, String(i))); }
      b.addEventListener('click', function () { active = i; press(i); if (opts.onClick) opts.onClick(i); });
      return b;
    }
    if (withCls) { buttons[0] = make(0); root.appendChild(buttons[0]); } else buttons[0] = null;
    var grid = h('div', { class: 'vpick-grid' }); for (var i = 1; i <= 16; i++) { buttons[i] = make(i); grid.appendChild(buttons[i]); }
    root.appendChild(grid);
    root.buttons = buttons; root.value = function () { return active; };
    root.setActive = function (i) { active = i; press(i); };
    root.setScene = function (ng) { g = V.scene(ng); for (var i = 1; i <= 16; i++) { var old = buttons[i].querySelector('svg'); if (old) buttons[i].replaceChild(V.thumb(g, i - 1, { size: size }), old); buttons[i].title = V.rowName(g, i); } };
    press(active);
    if (opts.into) (typeof opts.into === 'string' ? document.getElementById(opts.into) : opts.into).appendChild(root);
    return root;
  };

  /* ---------- notation rows for the card ---------- */
  function note(g, sym, mean, shape, dims) { AT.notation.push({ g: g, sym: sym, mean: mean, shape: shape, dims: function () { return dims || ''; }, parts: ['vision1'] }); }
  if (AT.notation && AT.notation.push) {
    note('token', 'r_j', 'the four pixels of patch $j$: top-left, top-right, bottom-left, bottom-right', '1\\times 4', '1×4');
    note('token', 'p_j', 'the position row of slot $j$: zeros on brightness and contrast, then row and col in 0..1 (CLS: $-1,-1$)', '1\\times d_{\\text{model}}', '1×' + T.d_model);
    note('token', '\\ve{e_j}=r_jW_{\\text{patch}}+p_j', 'the row of patch $j$ on the named axes brightness, contrast, row, col', '1\\times d_{\\text{model}}', '1×' + T.d_model);
    note('token', '\\ve{e_0}=c+p_0', 'the CLS row: a fixed start $c$ plus its own position, no pixels', '1\\times d_{\\text{model}}', '1×' + T.d_model);
    note('token', '\\vq{q_i}=\\ve{e_i}W_Q', 'query: what row $i$ asks for (bright region? on the right?)', '1\\times d_k', '1×' + T.d_k);
    note('token', '\\vk{k_j}=\\ve{e_j}W_K', 'key: what patch $j$ offers on the same two axes', '1\\times d_k', '1×' + T.d_k);
    note('token', '\\vv{v_j}=\\ve{e_j}W_V', 'value: what patch $j$ sends (brightness, contrast)', '1\\times d_v', '1×' + T.d_v);
    note('token', 's_{ij},\\ \\va{\\alpha_{ij}}', 'scaled score $\\vq{q_i}\\cdot\\vk{k_j}/\\sqrt{d_k}$ and its softmax weight over the 17 rows (no mask)', '', '');
    note('token', 'm_0=\\sum_j\\va{\\alpha_{0j}}\\vv{v_j}', 'the message CLS receives', '1\\times d_v', '1×' + T.d_v);
    note('token', '\\vd{\\Delta e_0}=m_0W_O,\\ \\vp{e_0^\\prime}=\\ve{e_0}+\\vd{\\Delta e_0}', 'the update CLS receives and its updated row', '1\\times d_{\\text{model}}', '1×' + T.d_model);
    note('token', '\\ell=\\vp{e_0^\\prime}W_{\\text{cls}}+b_{\\text{cls}},\\ p=\\operatorname{softmax}(\\ell)', 'the class head: one logit per answer, then probabilities over the two answers', '1\\times 2', '1×2');
    note('matrix', '\\ve{E},\\ \\vq{Q},\\ \\vk{K},\\ \\vv{V}', 'CLS plus sixteen patch rows; one row per patch, CLS first', '17\\times d', '17×4, 17×2, 17×2, 17×2');
    note('matrix', '\\va{A}=\\operatorname{softmax}(\\vq{Q}\\vk{K}^\\top/\\sqrt{d_k})', 'every row reads every row; no causal mask in an image encoder', '17\\times 17', '17×17');
    note('matrix', '\\vd{\\Delta E}=\\va{A}\\vv{V}W_O,\\ \\vp{E^\\prime}=\\ve{E}+\\vd{\\Delta E}', 'all seventeen updates at once; the head reads row 0 of $E^\\prime$', '17\\times d_{\\text{model}}', '17×4');
    note('sizes', 'N,\\ s_{\\text{patch}}', 'patches per image and the patch side (an $8\\times 8$ scene cut into $2\\times 2$ patches)', '', '16, 2');
    note('sizes', 'd_{\\text{model}},\\ d_k,\\ d_v', 'row width; matching width; value width', '', T.d_model + ', ' + T.d_k + ', ' + T.d_v);
    note('sizes', 'W_{\\text{patch}},\\ c,\\ P', 'fixed by the axis names: the patch projection, the CLS start row and the position rows (never trained here)', '4\\times 4,\\ 1\\times 4,\\ 17\\times 4', '');
    note('sizes', 'W_Q,W_K,W_V,W_O,W_{\\text{cls}},b_{\\text{cls}}', 'the trained numbers: 1,500 Adam steps on scenes A and B', '4\\times2,\\ 4\\times2,\\ 4\\times2,\\ 2\\times4,\\ 4\\times2,\\ 1\\times2', '');
  }
})();

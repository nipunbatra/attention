/* part6.js: Vision II runtime on toy6.json. Requires shared.js and vision-shared.js (V.scenes, V.embed, V.attend, the figures).
   The encoder is the frozen Vision I encoder: V.encode(scene) = V.attend(V.scene(scene), toy.encoder).Enew (17 rows, CLS first).
   Everything else here (the placeholder read, the MAE decoder, the I-JEPA predictor, the DINO head, the probe) recomputes the
   numbers stored by train_vision2.py from the parameters in toy6.json, so every displayed number is computed on the page. */
(function () {
  'use strict';
  var AT = window.AT, V = AT.vision, T = window.__TOY__ || {};
  if (!V || !T.encoder) return;
  var h = AT.h, svg = AT.svg;
  var SQ2 = Math.SQRT2;

  /* ---------- row-vector helpers ---------- */
  function mm(A, B) { return A.map(function (r) { return B[0].map(function (_, c) { return r.reduce(function (s, x, k) { return s + x * B[k][c]; }, 0); }); }); }
  function mv(v, M) { return M[0].map(function (_, c) { return v.reduce(function (s, x, k) { return s + x * M[k][c]; }, 0); }); }
  function add(a, b) { return a.map(function (x, i) { return x + b[i]; }); }
  function sm(v) { var m = Math.max.apply(null, v), ex = v.map(function (x) { return Math.exp(x - m); }), z = ex.reduce(function (a, b) { return a + b; }, 0); return ex.map(function (x) { return x / z; }); }
  function mean(a) { return a.reduce(function (s, x) { return s + x; }, 0) / a.length; }
  function copyRows(M) { return M.map(function (r) { return r.slice(); }); }
  V.mm = mm; V.mv = mv; V.softmaxRow = sm;

  /* ---------- the frozen encoder (ENCODER NOTE) ---------- */
  V.model = T; V.encoderParams = T.encoder; V.hidden = T.hidden.slice(); V.classesQuestion = T.question;
  V.forwardFull = function (scene) { return V.attend(V.scene(scene), T.encoder); };
  V.encode = function (scene) { return V.attend(V.scene(scene), T.encoder).Enew; };
  /* attention among an arbitrary subset of embedded rows, with the same formula as V.attend */
  function attendRows(E, P) {
    var Q = mm(E, P.W_Q), K = mm(E, P.W_K), Vv = mm(E, P.W_V);
    var S = Q.map(function (q) { return K.map(function (k) { return (q[0] * k[0] + q[1] * k[1]) / SQ2; }); });
    var A = S.map(sm), H = mm(A, Vv), D = mm(H, P.W_O);
    return { E: E, Q: Q, K: K, V: Vv, S: S, A: A, H: H, Delta: D, Enew: E.map(function (r, i) { return add(r, D[i]); }) };
  }
  V.attendRows = attendRows;
  /* MAE: only CLS and the visible patch rows enter the encoder; idx maps the rows back to 0 (CLS) or j+1 (patch j) */
  V.encodeVisible = function (scene, hidden) {
    hidden = hidden || T.hidden; var em = V.embed(V.scene(scene)), idx = [0];
    for (var j = 0; j < 16; j++) if (hidden.indexOf(j) < 0) idx.push(j + 1);
    var E = idx.map(function (i) { return em.E[i]; });
    return { idx: idx, vis: idx.slice(1).map(function (i) { return i - 1; }), out: attendRows(E, T.encoder) };
  };

  /* ---------- the placeholder read (MAE and I-JEPA share it) ---------- */
  V.read = function (theta, scene, hidden) {
    hidden = hidden || T.hidden;
    var ev = V.encodeVisible(scene, hidden), Ev = ev.out.Enew.slice(1);
    var K = mm(Ev, T.encoder.W_K), Vv = mm(Ev, T.encoder.W_V), res = {};
    hidden.forEach(function (j) {
      var u = add(theta.m, V.pos[j + 1]), q = mv(u, theta.W_Qd);
      var s = K.map(function (k) { return (q[0] * k[0] + q[1] * k[1]) / SQ2; }), a = sm(s);
      var msg = [0, 1].map(function (c) { return a.reduce(function (t, al, i) { return t + al * Vv[i][c]; }, 0); });
      res[j] = { u: u, q: q, scores: s, alpha: a, msg: msg, z: [msg[0], msg[1], V.pos[j + 1][2], V.pos[j + 1][3]] };
    });
    return { vis: ev.vis, Ev: Ev, K: K, V: Vv, encoded: ev.out, read: res };
  };
  /* a 16-entry weight row for the overlay: hidden patches get 0 */
  V.alpha16 = function (alphaVis, vis) { var o = []; for (var j = 0; j < 16; j++) o.push(0); vis.forEach(function (j, i) { o[j] = alphaVis[i]; }); return o; };

  /* ---------- MAE ---------- */
  V.maeSteps = Object.keys(T.mae.checkpoints).map(Number).sort(function (a, b) { return a - b; });
  V.mae = function (step, scene, hidden) {
    hidden = hidden || T.hidden;
    var cp = T.mae.checkpoints[String(step)], th = cp.params, rd = V.read(th, scene, hidden), R = V.patchify(scene), pred = {}, se = [];
    hidden.forEach(function (j) { var r = add(mv(rd.read[j].z, th.W_dec), th.b_dec); pred[j] = r; R[j].forEach(function (t, c) { se.push((r[c] - t) * (r[c] - t)); }); });
    return { step: step, params: th, vis: rd.vis, read: rd.read, K: rd.K, V: rd.V, Ev: rd.Ev, pred: pred, R: R, hidden: hidden.slice(), loss: mean(se) };
  };
  V.maeMeanLoss = function (step) { return mean(T.mae.scenes.map(function (s) { return V.mae(step, s).loss; })); };
  /* the scene with the hidden patches replaced by the decoder's guesses (visible pixels copied from the input) */
  V.maeImage = function (step, scene, hidden) {
    var f = V.mae(step, scene, hidden), g = V.copyScene(scene);
    f.hidden.forEach(function (j) { var pr = Math.floor(j / 4), pc = j % 4, r = f.pred[j]; g[2 * pr][2 * pc] = r[0]; g[2 * pr][2 * pc + 1] = r[1]; g[2 * pr + 1][2 * pc] = r[2]; g[2 * pr + 1][2 * pc + 1] = r[3]; });
    return g;
  };
  V.maskedImage = function (scene, hidden) {
    hidden = hidden || T.hidden; var g = V.copyScene(scene);
    hidden.forEach(function (j) { var pr = Math.floor(j / 4), pc = j % 4; for (var r = 0; r < 2; r++) for (var c = 0; c < 2; c++) g[2 * pr + r][2 * pc + c] = 0; });
    return g;
  };

  /* ---------- I-JEPA ---------- */
  V.jepaTarget = function (scene, j) { return V.encode(scene)[j + 1].slice(0, 2); };
  V.jepa = function (step, scene, hidden) {
    hidden = hidden || T.hidden;
    var cp = T.jepa.checkpoints[String(step)], th = cp.params, rd = V.read(th, scene, hidden), pred = {}, target = {}, se = [];
    hidden.forEach(function (j) { var y = add(mv(rd.read[j].z, th.W_pred), th.b_pred), t = V.jepaTarget(scene, j); pred[j] = y; target[j] = t; se.push((y[0] - t[0]) * (y[0] - t[0]), (y[1] - t[1]) * (y[1] - t[1])); });
    return { step: step, params: th, vis: rd.vis, read: rd.read, pred: pred, target: target, hidden: hidden.slice(), loss: mean(se) };
  };

  /* ---------- views ---------- */
  V.views = {
    identity: function (g) { return V.copyScene(g); },
    flip: function (g) { return V.scene(g).map(function (r) { return r.slice().reverse(); }); },
    dim: function (g) { return V.scene(g).map(function (r) { return r.map(function (x) { return 0.75 * x; }); }); },
    crop: function (g) { g = V.scene(g); var idx = []; for (var r = 0; r < 8; r++) idx.push(1 + Math.floor(r * 7 / 8)); return idx.map(function (rr) { return idx.map(function (cc) { return g[rr][cc]; }); }); }
  };
  V.views['crop+dim'] = function (g) { return V.views.dim(V.views.crop(g)); };
  V.viewNames = { identity: 'the scene as it is', flip: 'flipped left to right', dim: 'brightness × 0.75', crop: 'cropped one pixel in, stretched back', 'crop+dim': 'cropped, then × 0.75' };
  V.view = function (name, scene) { return V.views[name](V.scene(scene)); };

  /* ---------- DINO ---------- */
  var DN = T.dino;
  V.dino = {
    scenes: DN.scenes, viewNames: DN.views, tauS: DN.tau_s, log3: Math.log(3),
    feature: function (scene) { return V.encode(scene)[0]; },
    logits: function (f, head) { return add(mv(f, head.W), head.b); },
    target: function (z, c, tau) { return sm(z.map(function (x, i) { return (x - c[i]) / tau; })); },
    student: function (z, tau) { return sm(z.map(function (x) { return x / tau; })); },
    ce: function (pt, ps) { return -pt.reduce(function (s, p, i) { return s + p * Math.log(ps[i]); }, 0); },
    runKey: function (center, sharpen) { return (center ? 'center' : 'nocenter') + '_' + (sharpen ? 'sharpen' : 'nosharpen'); },
    run: function (center, sharpen) { return DN.runs[V.dino.runKey(center, sharpen)]; },
    /* recompute one checkpoint's outputs from the encoder and the stored heads (the JSON keeps a copy to check against) */
    outputs: function (center, sharpen, step) {
      var run = V.dino.run(center, sharpen), cp = run.checkpoints[String(step)], tau = run.tau_t, c = center ? cp.center : [0, 0, 0];
      return DN.scenes.map(function (s) {
        var f1 = V.dino.feature(V.view(DN.views[0], s)), f2 = V.dino.feature(V.view(DN.views[1], s));
        var zs = V.dino.logits(f1, cp.student), zt = V.dino.logits(f2, cp.teacher);
        var ps = V.dino.student(zs, DN.tau_s), pt = V.dino.target(zt, c, tau);
        return { scene: s, f1: f1, f2: f2, zs: zs, zt: zt, ps: ps, pt: pt, center: c, tau: tau, loss: V.dino.ce(pt, ps) };
      });
    },
    finalLoss: function (center, sharpen) { var cv = V.dino.run(center, sharpen).curve; return cv[cv.length - 1][1]; }
  };

  /* ---------- the linear probe ---------- */
  var PR = T.probe;
  V.probe = {
    axes: PR.axes, w: PR.w, b: PR.b, points: PR.points, all: PR.all, trainViews: ['identity', 'flip', 'dim'], testViews: ['crop'],
    pooled: function (scene) { var E = V.encode(scene).slice(1); return [mean(E.map(function (r) { return r[0]; })), mean(E.map(function (r) { return r[1]; }))]; },
    p: function (f, w, b) { w = w || PR.w; b = b == null ? PR.b : b; return 1 / (1 + Math.exp(-(w[0] * f[0] + w[1] * f[1] + b))); }
  };

  /* ---------- figures: a grid that draws fractional values with the same ramp, hatching, and a linear grey scale ---------- */
  var RAMP = [[0x3A, 0x3A, 0x3A], [0x6E, 0x6E, 0x6E], [0xA3, 0xA3, 0xA3], [0xD8, 0xD8, 0xD8]];
  V.rampLinear = function (v) {
    v = Math.max(0, Math.min(3, +v || 0)); var i = Math.min(2, Math.floor(v)), t = v - i;
    var c = RAMP[i].map(function (a, k) { return Math.round(a + t * (RAMP[i + 1][k] - a)); });
    return 'rgb(' + c.join(',') + ')';
  };
  var hatchCount = 0;
  function hatch(sv, cell, patches) {
    var id = 'v6h-' + (++hatchCount), d = svg('defs'), p = svg('pattern', { id: id, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    p.appendChild(svg('rect', { width: 6, height: 6, fill: '#F7F8FA' })); p.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: 'var(--ink-3)', 'stroke-width': 2 }));
    d.appendChild(p); sv.insertBefore(d, sv.firstChild);
    patches.forEach(function (j) { var pr = Math.floor(j / 4), pc = j % 4; sv.appendChild(svg('rect', { x: 2 * pc * cell, y: 2 * pr * cell, width: 2 * cell, height: 2 * cell, fill: 'url(#' + id + ')', class: 'v6-hatch' })); });
  }
  V.hatch = hatch;
  /* grid6: V.grid plus fractional values on a linear ramp, formatted labels, hatched hidden patches, a purple outline for a slot */
  V.grid6 = function (g, opts) {
    g = V.scene(g); opts = opts || {}; var cell = opts.cell || 28;
    var integral = g.every(function (r) { return r.every(function (x) { return x === Math.round(x); }); });
    var el = V.grid(g, { cell: cell, labels: opts.labels, patchLines: opts.patchLines, highlight: opts.highlight, names: opts.names, caption: opts.caption });
    var sv = el.querySelector('svg'), rects = sv.querySelectorAll('rect'), texts = sv.querySelectorAll('text'), k = 0, t = 0;
    for (var r = 0; r < 8; r++) for (var c = 0; c < 8; c++) {
      var v = g[r][c]; if (!integral || opts.linear) rects[k].setAttribute('fill', V.rampLinear(v)); k++;
      if (opts.labels === 'values') { var tx = texts[t++]; tx.textContent = integral ? String(v) : AT.fmt(v, opts.decimals == null ? 1 : opts.decimals); tx.setAttribute('fill', v <= 1.5 ? '#FFFFFF' : '#14171F'); if (!integral) tx.setAttribute('font-size', String(Math.round(cell * 0.38))); }
    }
    if (opts.hidden && opts.hidden.length) hatch(sv, cell, opts.hidden);
    if (opts.slot != null) { var pr = Math.floor(opts.slot / 4), pc = opts.slot % 4; sv.appendChild(svg('rect', { x: 2 * pc * cell + 1.5, y: 2 * pr * cell + 1.5, width: 2 * cell - 3, height: 2 * cell - 3, fill: 'none', stroke: 'var(--c-q)', 'stroke-width': 3 })); }
    (opts.outline || []).forEach(function (j) { var pr2 = Math.floor(j / 4), pc2 = j % 4; sv.appendChild(svg('rect', { x: 2 * pc2 * cell + 1.5, y: 2 * pr2 * cell + 1.5, width: 2 * cell - 3, height: 2 * cell - 3, fill: 'none', stroke: 'var(--c-a)', 'stroke-width': 3 })); });
    if (opts.into) { var host = typeof opts.into === 'string' ? document.getElementById(opts.into) : opts.into; if (host) host.appendChild(el); }
    return el;
  };
  /* a row of grids with captions (the triptych idiom for any number of panels) */
  V.gridRow = function (panels, opts) {
    opts = opts || {}; var row = h('div', { class: 'side-by-side v6-row', style: 'grid-template-columns:repeat(' + panels.length + ',auto);justify-content:start;gap:' + (opts.gap || 16) + 'px;align-items:start' });
    panels.forEach(function (p) { var f = h('figure', { class: 'vfig', style: 'margin:0;max-width:' + (8 * (opts.cell || 22) + 64) + 'px' }); f.appendChild(V.grid6(p.scene, { cell: opts.cell || 22, labels: p.labels || opts.labels, hidden: p.hidden, slot: p.slot, outline: p.outline, linear: p.linear, highlight: p.highlight })); f.appendChild(h('figcaption', { html: p.name })); row.appendChild(f); });
    if (opts.into) { var host = typeof opts.into === 'string' ? document.getElementById(opts.into) : opts.into; if (host) host.appendChild(row); }
    return row;
  };
  /* one 2x2 patch of fractional values as a small grid */
  V.patchTile = function (values, opts) {
    opts = opts || {}; var cell = opts.cell || 26, s = svg('svg', { viewBox: '0 0 ' + 2 * cell + ' ' + 2 * cell, width: 2 * cell, height: 2 * cell, role: 'img', class: 'v6-tile' });
    values.forEach(function (v, i) { var r = Math.floor(i / 2), c = i % 2; s.appendChild(svg('rect', { x: c * cell, y: r * cell, width: cell, height: cell, fill: V.rampLinear(v), stroke: '#F7F8FA' })); if (opts.labels !== false) s.appendChild(svg('text', { x: c * cell + cell / 2, y: r * cell + cell / 2 + 1, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-family': 'var(--font-mono)', 'font-size': Math.round(cell * 0.4), fill: v <= 1.5 ? '#fff' : '#14171F' }, AT.fmt(v, opts.decimals == null ? 1 : opts.decimals))); });
    s.appendChild(svg('rect', { x: .5, y: .5, width: 2 * cell - 1, height: 2 * cell - 1, fill: 'none', stroke: 'var(--ink-3)' }));
    var box = h('div', { class: 'vgrid' }, s); if (opts.caption) { var f = h('figure', { class: 'vfig', style: 'margin:0' }, box, h('figcaption', { html: opts.caption })); box = f; }
    if (opts.into) opts.into.appendChild(box); return box;
  };

  /* the placeholder's read painted on the scene: hidden patches hatched and unlabelled, the slot outlined in the query colour */
  V.readOverlay = function (scene, f, j, opts) {
    opts = opts || {}; var el = V.overlay(scene, V.alpha16(f.read[j].alpha, f.vis), { receiver: j, cell: opts.cell || 28, onHover: opts.onHover, onClick: opts.onClick, caption: opts.caption, decimals: opts.decimals });
    var sv = el.querySelector('svg'), cell = opts.cell || 28;
    var texts = sv.querySelectorAll('text'); f.hidden.forEach(function (hj) { if (texts[hj]) texts[hj].textContent = ''; });
    hatch(sv, cell, f.hidden.filter(function (hj) { return hj !== j; }));
    var pr = Math.floor(j / 4), pc = j % 4; sv.appendChild(svg('rect', { x: 2 * pc * cell + 1.5, y: 2 * pr * cell + 1.5, width: 2 * cell - 3, height: 2 * cell - 3, fill: 'var(--t-q)', stroke: 'var(--c-q)', 'stroke-width': 3, 'pointer-events': 'none' }));
    sv.appendChild(svg('text', { x: 2 * pc * cell + cell, y: 2 * pr * cell + cell + 1, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-family': 'var(--font-ui)', 'font-size': Math.round(cell * .5), 'font-weight': 700, fill: 'var(--c-q)', 'pointer-events': 'none' }, '?'));
    if (opts.into) opts.into.appendChild(el); return el;
  };

  /* the probability triangle: three slots at the corners; a point is a distribution p over the slots */
  V.simplex = function (points, opts) {
    opts = opts || {}; var W = opts.size || 340, m = 30, R = W - 2 * m;
    var A = [m, W - m], B = [W - m, W - m], C = [W / 2, W - m - R * Math.sqrt(3) / 2];
    var top = C[1] - 24;
    var s = svg('svg', { viewBox: '0 ' + top + ' ' + W + ' ' + (W - top), width: '100%', style: 'max-width:' + W + 'px', class: 'vsimplex', role: 'img' });
    s.appendChild(svg('path', { d: 'M' + A.join(' ') + 'L' + B.join(' ') + 'L' + C.join(' ') + 'Z', fill: 'var(--card)', stroke: 'var(--line)', 'stroke-width': 1.5 }));
    var cen = [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3];
    s.appendChild(svg('circle', { cx: cen[0], cy: cen[1], r: 3, fill: 'var(--line)' }));
    s.appendChild(svg('text', { x: cen[0] + 6, y: cen[1] + 4, 'font-size': 11, fill: 'var(--ink-3)' }, 'uniform'));
    var names = opts.slots || ['slot 1', 'slot 2', 'slot 3'];
    s.appendChild(svg('text', { x: A[0] - 4, y: A[1] + 16, 'font-size': 12, 'text-anchor': 'start' }, names[0]));
    s.appendChild(svg('text', { x: B[0] + 4, y: B[1] + 16, 'font-size': 12, 'text-anchor': 'end' }, names[1]));
    s.appendChild(svg('text', { x: C[0], y: C[1] - 8, 'font-size': 12, 'text-anchor': 'middle' }, names[2]));
    function XY(p) { return [p[0] * A[0] + p[1] * B[0] + p[2] * C[0], p[0] * A[1] + p[1] * B[1] + p[2] * C[1]]; }
    var marks = svg('g'); s.appendChild(marks);
    var defs = svg('defs'), mk = svg('marker', { id: 'v6-arrow-' + (++hatchCount), viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }); mk.appendChild(svg('path', { d: 'M0 0L10 5L0 10Z', fill: 'var(--ink-3)' })); defs.appendChild(mk); s.appendChild(defs);
    function draw(pts, arrows) {
      while (marks.firstChild) marks.removeChild(marks.firstChild);
      (arrows || []).forEach(function (ar) { var a = XY(ar.from), b = XY(ar.to); marks.appendChild(svg('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: 'var(--ink-3)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3', 'marker-end': 'url(#' + mk.id + ')' })); });
      pts.forEach(function (p) {
        var xy = XY(p.p), sz = p.size || 7;
        var el = p.kind === 'teacher' ? svg('path', { d: 'M' + xy[0] + ' ' + (xy[1] - sz) + 'L' + (xy[0] + sz) + ' ' + (xy[1] + sz) + 'L' + (xy[0] - sz) + ' ' + (xy[1] + sz) + 'Z', fill: p.fill || 'var(--c-k)', opacity: p.opacity == null ? .9 : p.opacity, stroke: '#fff', 'stroke-width': 1 })
                                         : svg('circle', { cx: xy[0], cy: xy[1], r: sz, fill: p.fill || 'var(--c-e)', opacity: p.opacity == null ? .9 : p.opacity, stroke: '#fff', 'stroke-width': 1.5 });
        el.appendChild(svg('title', {}, (p.label || '') + ' (' + p.p.map(function (x) { return AT.fmt(x, 2); }).join(', ') + ')')); marks.appendChild(el);
        if (p.label && opts.labels !== false) marks.appendChild(svg('text', { x: xy[0] + (p.dx == null ? 10 : p.dx), y: xy[1] + (p.dy == null ? 4 : p.dy), 'font-size': 12, fill: 'var(--ink)' }, p.label));
      });
    }
    draw(points, opts.arrows);
    var box = h('div', { class: 'vfig' }, s); if (opts.caption) box.appendChild(h('figcaption', { html: opts.caption }));
    box.setPoints = draw; box.svg = s;
    if (opts.into) opts.into.appendChild(box); return box;
  };

  /* the probe plane: points with two classes, an optional decision line w·f + b = 0, named axes */
  V.plane = function (points, opts) {
    opts = opts || {}; var W = opts.width || 460, H = opts.height || 300, m = { l: 52, r: 16, t: 14, b: 42 };
    var xs = points.map(function (p) { return p.x; }), ys = points.map(function (p) { return p.y; });
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs), ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    var padx = (xmax - xmin) * .15 + .05, pady = (ymax - ymin) * .2 + .05; xmin -= padx; xmax += padx; ymin -= pady; ymax += pady;
    var X = function (x) { return m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r); }, Y = function (y) { return H - m.b - (y - ymin) / (ymax - ymin) * (H - m.t - m.b); };
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', class: 'vscatter vplane', style: 'max-width:' + W + 'px', role: 'img' });
    s.appendChild(svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'var(--card)', stroke: 'var(--line)' }));
    if (ymin < 0 && ymax > 0) s.appendChild(svg('line', { x1: m.l, y1: Y(0), x2: W - m.r, y2: Y(0), stroke: 'var(--line)' }));
    var ax = opts.axes || ['axis 1', 'axis 2'];
    s.appendChild(svg('text', { x: (m.l + W - m.r) / 2, y: H - 10, 'text-anchor': 'middle', 'font-size': 13 }, ax[0]));
    s.appendChild(svg('text', { x: 14, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', 'font-size': 13, transform: 'rotate(-90 14 ' + (m.t + H - m.b) / 2 + ')' }, ax[1]));
    [xmin, xmax].forEach(function (v) { s.appendChild(svg('text', { x: X(v), y: H - m.b + 14, 'text-anchor': 'middle', 'font-size': 11, 'font-family': 'var(--font-mono)' }, AT.fmt(v, 1))); });
    [ymin, ymax].forEach(function (v) { s.appendChild(svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, 'font-family': 'var(--font-mono)' }, AT.fmt(v, 1))); });
    var lineEl = svg('line', { stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '6 4' }), shade = svg('polygon', { fill: 'var(--t-e)', opacity: .5 });
    s.appendChild(shade); s.appendChild(lineEl);
    function setLine(w, b) {
      if (!w) { lineEl.setAttribute('opacity', '0'); shade.setAttribute('points', ''); return; }
      /* w0 x + w1 y + b = 0 clipped to the box; the shaded side is w·f + b > 0 (the "two mugs" side) */
      var pts = [];
      [xmin, xmax].forEach(function (x) { if (Math.abs(w[1]) > 1e-9) { var y = -(w[0] * x + b) / w[1]; if (y >= ymin && y <= ymax) pts.push([x, y]); } });
      [ymin, ymax].forEach(function (y) { if (Math.abs(w[0]) > 1e-9) { var x = -(w[1] * y + b) / w[0]; if (x > xmin && x < xmax) pts.push([x, y]); } });
      if (pts.length < 2) { lineEl.setAttribute('opacity', '0'); return; }
      lineEl.setAttribute('opacity', '1'); lineEl.setAttribute('x1', X(pts[0][0])); lineEl.setAttribute('y1', Y(pts[0][1])); lineEl.setAttribute('x2', X(pts[1][0])); lineEl.setAttribute('y2', Y(pts[1][1]));
      var corners = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]].filter(function (c) { return w[0] * c[0] + w[1] * c[1] + b > 0; });
      var poly = corners.concat(pts).map(function (c) { return [X(c[0]), Y(c[1])]; });
      var cx = mean(poly.map(function (p) { return p[0]; })), cy = mean(poly.map(function (p) { return p[1]; }));
      poly.sort(function (a, b2) { return Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b2[1] - cy, b2[0] - cx); });
      shade.setAttribute('points', poly.map(function (p) { return p.join(','); }).join(' '));
    }
    setLine(opts.w, opts.b);
    /* nudge labels apart: points sorted by x, a label that would sit on an earlier one moves down */
    var placed = [];
    points.slice().sort(function (a, b2) { return a.x - b2.x; }).forEach(function (p) {
      var gx = X(p.x), gy = Y(p.y) + 4, tries = 0;
      while (tries < 6 && placed.some(function (q) { return Math.abs(q[0] - gx) < 62 && Math.abs(q[1] - gy) < 13; })) { gy += 13; tries++; }
      p._ly = gy - Y(p.y); placed.push([gx, gy]);
    });
    points.forEach(function (p) {
      var gx = X(p.x), gy = Y(p.y), sz = 6;
      var el = p.cls === 'two' ? svg('rect', { x: gx - sz, y: gy - sz, width: 2 * sz, height: 2 * sz, fill: 'var(--c-e)', stroke: p.test ? 'var(--warn)' : '#fff', 'stroke-width': p.test ? 2.5 : 1.5 })
                               : svg('circle', { cx: gx, cy: gy, r: sz, fill: 'var(--card)', stroke: p.test ? 'var(--warn)' : 'var(--ink)', 'stroke-width': p.test ? 2.5 : 2 });
      el.appendChild(svg('title', {}, (p.label || '') + ' (' + AT.fmt(p.x, 2) + ', ' + AT.fmt(p.y, 2) + ')')); s.appendChild(el);
      if (p.label) s.appendChild(svg('text', { x: gx + (p.dx == null ? 9 : p.dx), y: gy + (p.dy == null ? (p._ly == null ? 4 : p._ly) : p.dy), 'font-size': 11, fill: 'var(--ink)' }, p.label));
    });
    var box = h('div', { class: 'vfig' }, s); if (opts.caption) box.appendChild(h('figcaption', { html: opts.caption }));
    box.setLine = setLine; box.svg = s;
    if (opts.into) opts.into.appendChild(box); return box;
  };

  /* ---------- present-mode sizing for the shared vision figures inside this part ---------- */
  document.head.appendChild(h('style', {}, [
    'body.present .vfig figcaption{font-size:var(--present-caption);margin-top:8px}',
    'body.present .vscatter,body.present .vcurve{max-width:560px!important}',
    'body.present .vsimplex{max-width:380px!important}',
    'svg.vcurve,svg.vscatter,svg.vsimplex{overflow:visible}',
    '.vfig{max-width:100%}.vfig figcaption{max-width:100%}',
    '.v6-pair > :first-child{max-width:300px}',
    'body.present .v6-pair > :first-child{max-width:340px}',
    'body.present .vgrid-svg{width:224px;height:224px}',
    '.v6-row figcaption{font-size:14px;color:var(--ink-2)}',
    '.v6-read{font-size:15px;color:var(--ink-2);margin:10px 0 0;max-width:72ch;line-height:1.5}',
    'body.present .v6-read{font-size:22px;line-height:1.35;margin:6px 0 0}',
    'body.present .v6-row{margin:0}',
    '.v6-pair{display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px 28px;align-items:start}',
    '@media (max-width:820px){body:not(.present) .v6-pair{grid-template-columns:minmax(0,1fr)}}',
    'body.present .dt-cap,body.present .dt-note{display:none}',
    'body.present .dt td,body.present .dt th{padding:.22em .5em}',
    'body.present .dt tbody th .vthumb,body.present .dt tbody th .v6-tile{width:30px;height:30px}',
    '.v6-btns .btn[aria-pressed=true]{background:var(--ink);color:var(--card)}',
    'body.present .calc-rounding{display:none}',
    '.v6-cap{font-size:14px;color:var(--ink-2);margin:6px 0 0;max-width:68ch}',
    'body.present .v6-cap{font-size:22px;line-height:1.35}',
    'body.present .v6-hide{display:none}'
  ].join('\n')));

  /* the two content axes with their short labels and full names, for tables of 2-wide feature rows */
  V.contentCols = (AT.axesFor('e', 4) || []).slice(0, 2);

  /* ---------- notation rows ---------- */
  function note(g, sym, mean, shape, dims) { AT.notation.push({ g: g, sym: sym, mean: mean, shape: shape, dims: function () { return dims || ''; }, parts: ['vision2'] }); }
  if (AT.notation && AT.notation.push) {
    note('image', 'r_j', 'Raw pixels of patch $j$: top left, top right, bottom left, bottom right', '1\\times 4', '1×4');
    note('image', '\\ve{e_j}=r_jW_{\\mathrm{patch}}+p_j', 'Patch row on the named axes brightness, contrast, row, col (Vision I)', '1\\times d_{\\mathrm{model}}', '1×4');
    note('image', "\\vp{e_j'}", 'The frozen encoder’s output row for patch $j$ (or CLS for $j=0$): the feature other parts reuse', '1\\times d_{\\mathrm{model}}', '1×4');
    note('image', 'f', 'Pooled image feature for the probe: the mean of the sixteen $\\vp{e_j\'}$ rows on (brightness, contrast)', '1\\times 2', '');
    note('masked', '\\mathcal M', 'The hidden patches; here the four right-mug slots', '', '{3, 4, 7, 8}');
    note('masked', 'u_j=m+p_j', 'The placeholder for hidden slot $j$: one learned row $m$ plus the slot’s position', '1\\times d_{\\mathrm{model}}', '1×4');
    note('masked', '\\vq{q_j}=u_jW_Q^{\\mathrm{dec}}', 'What the placeholder asks the visible patches (learned)', '1\\times d_k', '1×2');
    note('masked', 'g_j=\\sum_i\\va{\\alpha_{ji}}\\vv{v_i}', 'What it reads: a mixture of the visible patches’ values (frozen $W_K$, $W_V$)', '1\\times d_v', '1×2');
    note('masked', 'z_j=(g_j,\\ \\mathrm{row}_j,\\ \\mathrm{col}_j)', 'The decoder row: what it read and where it is', '1\\times 4', '');
    note('masked', '\\hat r_j=z_jW_{\\mathrm{dec}}+b_{\\mathrm{dec}}', 'MAE: the four predicted pixels of slot $j$', '1\\times 4', '');
    note('masked', '\\hat y_j=z_jW_{\\mathrm{pred}}+b_{\\mathrm{pred}},\\ y_j', 'I-JEPA: the predicted and the target feature of slot $j$ on (brightness, contrast)', '1\\times 2', '');
    note('distill', 'z_s,\\ z_t', 'Student and teacher logits over three output slots: $\\vp{e_0\'}W+b$ on a view', '1\\times 3', '');
    note('distill', 'c,\\ \\tau_t,\\ \\tau_s', 'The centre (a running mean of teacher logits) and the two temperatures', '', '');
    note('distill', 'p_t=\\operatorname{softmax}((z_t-c)/\\tau_t),\\ p_s=\\operatorname{softmax}(z_s/\\tau_s)', 'The target and the student’s distribution; the loss is $-\\sum p_t\\log p_s$', '1\\times 3', '');
    note('distill', '\\phi\\leftarrow\\beta\\phi+(1-\\beta)\\theta,\\ \\operatorname{sg}[\\cdot]', 'The teacher follows the student by a moving average; no gradient passes through the target', '', '');
    note('sizes', 'N,\\ |\\mathcal M|,\\ d_{\\mathrm{model}},\\ d_k,\\ d_v,\\ K', 'Patches, hidden patches, widths, DINO slots', '', '16, 4, 4, 2, 2, 3');
    note('sizes', '\\text{decoder},\\ \\text{predictor},\\ \\text{head},\\ \\text{probe}', 'Trained parameters: MAE decoder, I-JEPA predictor, DINO head, probe line', '', '32, 22, 15, 3');
  }
  if (Array.isArray(AT.objects)) {
    var e = AT.objects.find(function (o) { return o.cls === 'e'; }), ep = AT.objects.find(function (o) { return o.cls === 'ep'; });
    if (e) { e.name = 'patch or CLS row'; e.def = 'e_j: the row for patch j (or the CLS row) before attention.'; e.tip = 'e_j is the input row of patch j: its pixels through the fixed W_patch plus its position. Named axes: brightness, contrast, row, col.'; }
    if (ep) { ep.name = 'encoder output'; ep.def = "e_j′ = e_j + Δe_j: the frozen encoder's output row, the feature."; ep.tip = "e_j′ is what the frozen Vision I encoder returns for patch j. MAE, DINO, I-JEPA and the probe all read these rows."; }
  }
})();

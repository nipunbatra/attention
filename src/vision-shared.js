/* vision-shared.js: the one scene, the fixed patch encoder, and the shared figures for the vision parts.
   Injected after shared.js for parts 5 to 8. Design: VISION_AXES.md. Plain script; extends window.AT.vision. */
(function () {
  'use strict';
  var AT = window.AT; if (!AT) return;
  var V = AT.vision = AT.vision || {};
  var h = AT.h, svg = AT.svg;

  /* ---------- scenes (8x8, values 0..3) ---------- */
  function blank() { var g = []; for (var r = 0; r < 8; r++) { g.push([0, 0, 0, 0, 0, 0, 0, 0]); } return g; }
  function mug(g, c0) { for (var r = 1; r <= 3; r++) for (var c = c0; c < c0 + 3; c++) g[r][c] = 3; }
  function book(g) { for (var r = 5; r <= 6; r++) for (var c = 1; c <= 5; c++) g[r][c] = 2; }
  function plant(g) { g[4][7] = 1; }
  var S = {};
  S.A = blank(); mug(S.A, 1); mug(S.A, 5); book(S.A); plant(S.A);
  S.B = blank(); mug(S.B, 1); book(S.B); plant(S.B);
  S.C = blank(); mug(S.C, 4); book(S.C); plant(S.C);
  S.D = blank(); book(S.D);
  S.E = blank(); plant(S.E);
  var NAMES = { A: 'two mugs', B: 'one mug', C: 'one mug, moved right', D: 'book only', E: 'plant only' };
  V.scenes = S; V.sceneNames = NAMES;
  V.scene = function (key) { return Array.isArray(key) ? key : S[key]; };
  V.copyScene = function (g) { return V.scene(g).map(function (r) { return r.slice(); }); };
  V.ramp = function (v) { return ['#3A3A3A', '#6E6E6E', '#A3A3A3', '#D8D8D8'][Math.max(0, Math.min(3, Math.round(v)))]; };
  V.rampText = function (v) { return Math.round(v) <= 1 ? '#FFFFFF' : '#14171F'; };

  /* ---------- patches and region names ---------- */
  V.patchify = function (g) {
    g = V.scene(g); var rows = [];
    for (var pr = 0; pr < 4; pr++) for (var pc = 0; pc < 4; pc++) rows.push([g[2 * pr][2 * pc], g[2 * pr][2 * pc + 1], g[2 * pr + 1][2 * pc], g[2 * pr + 1][2 * pc + 1]]);
    return rows;
  };
  V.regionOf = function (g, j) {
    g = V.scene(g); var pr = Math.floor(j / 4), pc = j % 4, objs = {};
    for (var r = 0; r < 2; r++) for (var c = 0; c < 2; c++) {
      var v = g[2 * pr + r][2 * pc + c], col = 2 * pc + c; if (!v) continue;
      var name = v === 3 ? (col <= 3 ? 'left mug' : 'right mug') : v === 2 ? 'book' : 'plant';
      objs[name] = (objs[name] || 0) + 1;
    }
    var best = null, n = 0; Object.keys(objs).forEach(function (k) { if (objs[k] > n) { best = k; n = objs[k]; } });
    if (!best) return 'table'; if (best === 'plant') return 'plant';
    return best + (n === 4 ? ' centre' : n === 1 ? ' corner' : ' edge');
  };
  V.patchLabel = function (g, j) { return 'patch ' + (j + 1) + ' · ' + V.regionOf(g, j); };
  V.mugPatches = function (g, side) { var o = []; for (var j = 0; j < 16; j++) { var r = V.regionOf(g, j); if (r.indexOf((side || '') + ' mug') >= 0 || (!side && r.indexOf('mug') >= 0)) o.push(j); } return o; };

  /* ---------- the fixed encoder (names first) ---------- */
  V.axes = { e: ['brightness', 'contrast (left minus right)', 'row', 'col'], qk: ['bright region?', 'on the right?'], v: ['sends: brightness', 'sends: contrast'],
    short: { e: ['bright', 'contrast', 'row', 'col'], qk: ['bright?', 'right?'], v: ['→bright', '→contrast'] } };
  V.W_patch = [[.25, .5, 0, 0], [.25, -.5, 0, 0], [.25, .5, 0, 0], [.25, -.5, 0, 0]];
  V.cls = [1, 0, 0, 0];
  V.pos = (function () { var P = [[0, 0, -1, -1]]; for (var j = 0; j < 16; j++) P.push([0, 0, Math.floor(j / 4) / 3, (j % 4) / 3]); return P; })();
  var mm = function (A, B) { return A.map(function (r) { return B[0].map(function (_, c) { return r.reduce(function (s, x, k) { return s + x * B[k][c]; }, 0); }); }); };
  var add = function (A, B) { return A.map(function (r, i) { return r.map(function (x, k) { return x + B[i][k]; }); }); };
  var sm = function (v) { var m = Math.max.apply(null, v), ex = v.map(function (x) { return Math.exp(x - m); }), z = ex.reduce(function (a, b) { return a + b; }, 0); return ex.map(function (x) { return x / z; }); };
  V.embed = function (g) { var R = V.patchify(g); var E = [V.cls.slice()].concat(mm(R, V.W_patch)); return { R: R, E: add(E, V.pos) }; };
  /* one attention layer over CLS + 16 patches with a parameter set {W_Q, W_K, W_V, W_O, W_cls, b_cls} */
  V.attend = function (g, P) {
    var em = V.embed(g), E = em.E, Q = mm(E, P.W_Q), K = mm(E, P.W_K), Vv = mm(E, P.W_V);
    var S = Q.map(function (q) { return K.map(function (k) { return (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2; }); });
    var A = S.map(sm), H = mm(A, Vv), D = mm(H, P.W_O), En = add(E, D);
    var out = { R: em.R, E: E, Q: Q, K: K, V: Vv, S: S, A: A, H: H, Delta: D, Enew: En };
    if (P.W_cls) { var l = P.b_cls.map(function (b, c) { return b + En[0].reduce(function (s, x, k) { return s + x * P.W_cls[k][c]; }, 0); }); out.logits = l; out.probs = sm(l); }
    return out;
  };
  V.readRow = function (row, axesKey, dec) { var ax = (V.axes[axesKey] || V.axes.e); return row.map(function (x, k) { return ax[k] + ' ' + AT.fmt(x, dec == null ? 2 : dec); }).join(', '); };

  /* ---------- figures ---------- */
  var css = h('style', {}, [
    '.vgrid{display:inline-block;vertical-align:middle}.vgrid svg{display:block}',
    '.vfig{margin:8px 0}.vfig figcaption{font-size:14px;color:var(--ink-3);margin-top:6px}',
    '.vthumb{display:inline-block;vertical-align:middle;margin-right:6px}',
    '.vscatter text,.vcircle text,.vcurve text{font-family:var(--font-ui);fill:var(--ink-2)}',
    '.voverlay .vcell:hover{stroke:var(--ink);stroke-width:2}'
  ].join('\n'));
  document.head.appendChild(css);
  function px(n) { return String(n); }

  /* grid: the 8x8 image; opts {cell, labels:'values'|'none', patchLines, highlight:[j], names:false, into, caption} */
  V.grid = function (g, opts) {
    g = V.scene(g); opts = opts || {}; var cell = opts.cell || 28, W = 8 * cell;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + W, width: W, height: W, class: 'vgrid-svg', role: 'img' });
    for (var r = 0; r < 8; r++) for (var c = 0; c < 8; c++) {
      s.appendChild(svg('rect', { x: c * cell, y: r * cell, width: cell, height: cell, fill: V.ramp(g[r][c]), stroke: '#F7F8FA', 'stroke-width': 1 }));
      if (opts.labels === 'values') s.appendChild(svg('text', { x: c * cell + cell / 2, y: r * cell + cell / 2 + 1, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-family': 'var(--font-mono)', 'font-size': Math.round(cell * .5), fill: V.rampText(g[r][c]) }, String(g[r][c])));
    }
    if (opts.patchLines !== false) for (var k = 1; k < 4; k++) { s.appendChild(svg('line', { x1: 2 * k * cell, y1: 0, x2: 2 * k * cell, y2: W, stroke: 'var(--ink-3)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' })); s.appendChild(svg('line', { x1: 0, y1: 2 * k * cell, x2: W, y2: 2 * k * cell, stroke: 'var(--ink-3)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' })); }
    (opts.highlight || []).forEach(function (j) { var pr = Math.floor(j / 4), pc = j % 4; s.appendChild(svg('rect', { x: 2 * pc * cell + 1, y: 2 * pr * cell + 1, width: 2 * cell - 2, height: 2 * cell - 2, fill: 'none', stroke: 'var(--c-e)', 'stroke-width': 3 })); });
    if (opts.names) for (var j = 0; j < 16; j++) { var pr2 = Math.floor(j / 4), pc2 = j % 4; s.appendChild(svg('text', { x: 2 * pc2 * cell + 3, y: 2 * pr2 * cell + 11, 'font-family': 'var(--font-ui)', 'font-size': 10, fill: '#fff', opacity: .9 }, String(j + 1))); }
    var box = h('div', { class: 'vgrid' }, s);
    if (opts.caption) { var f = h('figure', { class: 'vfig' }, box, h('figcaption', { html: opts.caption })); return put(f, opts); }
    return put(box, opts);
  };
  function put(el, opts) { if (opts && opts.into) { var t = typeof opts.into === 'string' ? document.getElementById(opts.into) : opts.into; if (t) t.appendChild(el); } return el; }

  /* thumb: one 2x2 patch as a tiny svg; thumbHTML for table labels */
  V.thumb = function (g, j, opts) {
    g = V.scene(g); opts = opts || {}; var size = opts.size || 22, cell = size / 2, pr = Math.floor(j / 4), pc = j % 4;
    var s = svg('svg', { viewBox: '0 0 ' + size + ' ' + size, width: size, height: size, class: 'vthumb', role: 'img' });
    for (var r = 0; r < 2; r++) for (var c = 0; c < 2; c++) s.appendChild(svg('rect', { x: c * cell, y: r * cell, width: cell, height: cell, fill: V.ramp(g[2 * pr + r][2 * pc + c]) }));
    s.appendChild(svg('rect', { x: .5, y: .5, width: size - 1, height: size - 1, fill: 'none', stroke: 'var(--ink-3)' }));
    return put(s, opts);
  };
  V.thumbHTML = function (g, j, size) { var t = V.thumb(g, j, { size: size || 22 }); var d = h('div'); d.appendChild(t); return d.innerHTML; };
  V.rowLabel = function (g, j, opts) { return V.thumbHTML(g, j, (opts && opts.size) || 22) + ' ' + V.patchLabel(g, j); };
  V.rowLabels = function (g, opts) { var o = []; for (var j = 0; j < 16; j++) o.push(V.rowLabel(g, j, opts)); return o; };

  /* overlay: attention painted on the image. alpha has 17 entries (CLS first) or 16 (patches only). */
  V.overlay = function (g, alpha, opts) {
    g = V.scene(g); opts = opts || {}; var cell = opts.cell || 28, W = 8 * cell;
    var a = alpha.length === 17 ? alpha.slice(1) : alpha.slice(), clsA = alpha.length === 17 ? alpha[0] : null;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + (W + (clsA != null ? 26 : 0)), width: W, height: W + (clsA != null ? 26 : 0), class: 'voverlay', role: 'img' });
    for (var r = 0; r < 8; r++) for (var c = 0; c < 8; c++) s.appendChild(svg('rect', { x: c * cell, y: r * cell, width: cell, height: cell, fill: V.ramp(g[r][c]), stroke: '#F7F8FA', 'stroke-width': 1 }));
    var tints = [], texts = [], cells = [];
    function paint() {
      var mx = Math.max.apply(null, a.concat([1e-9]));
      for (var j = 0; j < 16; j++) { var t = a[j] / mx; tints[j].setAttribute('opacity', String(0.08 + 0.72 * t)); texts[j].textContent = AT.fmt(a[j], opts.decimals == null ? 2 : opts.decimals); }
      if (clsText) clsText.textContent = 'CLS reads itself: ' + AT.fmt(clsA, 2);
    }
    for (var j = 0; j < 16; j++) {
      var pr = Math.floor(j / 4), pc = j % 4;
      var tint = svg('rect', { x: 2 * pc * cell, y: 2 * pr * cell, width: 2 * cell, height: 2 * cell, fill: 'var(--c-a)', opacity: .1, 'pointer-events': 'none' });
      var tx = svg('text', { x: 2 * pc * cell + cell, y: 2 * pr * cell + cell + 1, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-family': 'var(--font-mono)', 'font-size': Math.round(cell * .55), 'font-weight': 700, fill: '#fff', stroke: 'rgba(0,0,0,.55)', 'stroke-width': 3, 'paint-order': 'stroke', 'pointer-events': 'none' });
      var hit = svg('rect', { x: 2 * pc * cell, y: 2 * pr * cell, width: 2 * cell, height: 2 * cell, fill: 'transparent', stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '3 3', class: 'vcell', tabindex: 0 });
      hit.appendChild(svg('title', {}, V.patchLabel(g, j)));
      (function (jj, hh) { ['mouseenter', 'focus'].forEach(function (ev) { hh.addEventListener(ev, function () { if (opts.onHover) opts.onHover(jj); }); }); hh.addEventListener('click', function () { if (opts.onClick) opts.onClick(jj); }); })(j, hit);
      s.appendChild(tint); s.appendChild(tx); s.appendChild(hit); tints.push(tint); texts.push(tx); cells.push(hit);
    }
    var clsText = null;
    if (clsA != null) { clsText = svg('text', { x: 4, y: W + 17, 'font-family': 'var(--font-ui)', 'font-size': 13, fill: 'var(--ink-2)' }); s.appendChild(clsText); }
    if (opts.receiver != null && opts.receiver !== 'CLS') { var rp = Math.floor(opts.receiver / 4), rc = opts.receiver % 4; s.appendChild(svg('rect', { x: 2 * rc * cell + 1.5, y: 2 * rp * cell + 1.5, width: 2 * cell - 3, height: 2 * cell - 3, fill: 'none', stroke: 'var(--c-q)', 'stroke-width': 3, 'pointer-events': 'none' })); }
    paint();
    var box = h('div', { class: 'vgrid' }, s);
    box.setAlpha = function (na) { a = na.length === 17 ? na.slice(1) : na.slice(); if (na.length === 17) clsA = na[0]; paint(); };
    box.setScene = function (ng) { g = V.scene(ng); var i = 0; for (var r = 0; r < 8; r++) for (var c = 0; c < 8; c++) s.childNodes[i++].setAttribute('fill', V.ramp(g[r][c])); };
    if (opts.caption) { var f = h('figure', { class: 'vfig' }, box, h('figcaption', { html: opts.caption })); f.setAlpha = box.setAlpha; f.setScene = box.setScene; return put(f, opts); }
    return put(box, opts);
  };

  /* scatter: 2-D embedding plot with named axes; points [{x,y,label,thumb:{scene,j}|null,cls}], arrows [{from:[x,y],to:[x,y],label}] */
  V.scatter = function (points, opts) {
    opts = opts || {}; var W = opts.width || 420, H = opts.height || 300, m = { l: 48, r: 16, t: 14, b: 40 };
    var xs = points.map(function (p) { return p.x; }), ys = points.map(function (p) { return p.y; });
    (opts.arrows || []).forEach(function (ar) { xs.push(ar.from[0], ar.to[0]); ys.push(ar.from[1], ar.to[1]); });
    var xmin = Math.min.apply(null, xs.concat([0])), xmax = Math.max.apply(null, xs.concat([1])), ymin = Math.min.apply(null, ys.concat([0])), ymax = Math.max.apply(null, ys.concat([1]));
    var padx = (xmax - xmin) * .12 + .05, pady = (ymax - ymin) * .12 + .05; xmin -= padx; xmax += padx; ymin -= pady; ymax += pady;
    var X = function (x) { return m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r); }, Y = function (y) { return H - m.b - (y - ymin) / (ymax - ymin) * (H - m.t - m.b); };
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', class: 'vscatter', style: 'max-width:' + W + 'px', role: 'img' });
    s.appendChild(svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'var(--card)', stroke: 'var(--line)' }));
    if (xmin < 0 && xmax > 0) s.appendChild(svg('line', { x1: X(0), y1: m.t, x2: X(0), y2: H - m.b, stroke: 'var(--line)' }));
    if (ymin < 0 && ymax > 0) s.appendChild(svg('line', { x1: m.l, y1: Y(0), x2: W - m.r, y2: Y(0), stroke: 'var(--line)' }));
    var ax = opts.axes || ['axis 1', 'axis 2'];
    s.appendChild(svg('text', { x: (m.l + W - m.r) / 2, y: H - 10, 'text-anchor': 'middle', 'font-size': 13 }, ax[0]));
    var yl = svg('text', { x: 14, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', 'font-size': 13, transform: 'rotate(-90 14 ' + (m.t + H - m.b) / 2 + ')' }, ax[1]); s.appendChild(yl);
    [xmin, xmax].forEach(function (v) { s.appendChild(svg('text', { x: X(v), y: H - m.b + 14, 'text-anchor': 'middle', 'font-size': 11, 'font-family': 'var(--font-mono)' }, AT.fmt(v, 1))); });
    [ymin, ymax].forEach(function (v) { s.appendChild(svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, 'font-family': 'var(--font-mono)' }, AT.fmt(v, 1))); });
    var defs = svg('defs'), mk = svg('marker', { id: 'vsc-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }); mk.appendChild(svg('path', { d: 'M0 0L10 5L0 10Z', fill: 'var(--c-d)' })); defs.appendChild(mk); s.appendChild(defs);
    (opts.arrows || []).forEach(function (ar) {
      s.appendChild(svg('line', { x1: X(ar.from[0]), y1: Y(ar.from[1]), x2: X(ar.to[0]), y2: Y(ar.to[1]), stroke: 'var(--c-d)', 'stroke-width': 2.5, 'marker-end': 'url(#vsc-arrow)' }));
      if (ar.label) s.appendChild(svg('text', { x: (X(ar.from[0]) + X(ar.to[0])) / 2 + 6, y: (Y(ar.from[1]) + Y(ar.to[1])) / 2 - 6, 'font-size': 12, fill: 'var(--c-d)' }, ar.label));
    });
    points.forEach(function (p) {
      var gx = X(p.x), gy = Y(p.y);
      if (p.thumb) { var t = V.thumb(p.thumb.scene, p.thumb.j, { size: 16 }); t.setAttribute('x', gx - 8); t.setAttribute('y', gy - 8); s.appendChild(t); }
      else s.appendChild(svg('circle', { cx: gx, cy: gy, r: p.r || 6, fill: p.cls === 'ep' ? 'var(--c-e)' : p.cls === 'd' ? 'var(--c-d)' : 'var(--c-e)', stroke: p.cls === 'ep' ? 'var(--c-d)' : '#fff', 'stroke-width': p.cls === 'ep' ? 3 : 1.5 }));
      if (p.label) s.appendChild(svg('text', { x: gx + 10, y: gy + 4, 'font-size': 12, fill: 'var(--ink)' }, p.label));
    });
    var box = h('div', { class: 'vfig' }, s); if (opts.caption) box.appendChild(h('figcaption', { html: opts.caption }));
    return put(box, opts);
  };

  /* circle: unit vectors (2-D exact; 3-D drawn by orthographic projection on the first two coordinates, radius shows the third) */
  V.circle = function (points, opts) {
    opts = opts || {}; var W = opts.size || 320, c = W / 2, R = W / 2 - 34;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + W, width: '100%', style: 'max-width:' + W + 'px', class: 'vcircle', role: 'img' });
    s.appendChild(svg('circle', { cx: c, cy: c, r: R, fill: 'var(--card)', stroke: 'var(--line)', 'stroke-width': 1.5 }));
    s.appendChild(svg('line', { x1: c - R, y1: c, x2: c + R, y2: c, stroke: 'var(--line)' })); s.appendChild(svg('line', { x1: c, y1: c - R, x2: c, y2: c + R, stroke: 'var(--line)' }));
    var ax = opts.axes || []; if (ax[0]) s.appendChild(svg('text', { x: c + R + 4, y: c + 4, 'font-size': 12 }, ax[0])); if (ax[1]) s.appendChild(svg('text', { x: c, y: c - R - 8, 'text-anchor': 'middle', 'font-size': 12 }, ax[1]));
    var marks = [];
    function draw(pts) {
      marks.forEach(function (m) { s.removeChild(m); }); marks = [];
      pts.forEach(function (p) {
        var x = c + p.v[0] * R, y = c - p.v[1] * R, sz = 7 + (p.v.length > 2 ? p.v[2] * 3 : 0);
        var el = p.kind === 'txt' ? svg('path', { d: 'M' + x + ' ' + (y - sz) + 'L' + (x + sz) + ' ' + (y + sz) + 'L' + (x - sz) + ' ' + (y + sz) + 'Z', fill: 'var(--c-q)', opacity: .9 })
                                   : svg('rect', { x: x - sz, y: y - sz, width: 2 * sz, height: 2 * sz, fill: 'var(--c-e)', opacity: .9 });
        el.appendChild(svg('title', {}, p.label || '')); s.appendChild(el); marks.push(el);
        if (p.label && opts.labels !== false) { var t = svg('text', { x: x + sz + 3, y: y + 4, 'font-size': 11, fill: 'var(--ink)' }, p.label); s.appendChild(t); marks.push(t); }
      });
    }
    draw(points);
    var box = h('div', { class: 'vfig' }, s); if (opts.caption) box.appendChild(h('figcaption', { html: opts.caption }));
    box.setPoints = draw;
    box.animate = function (states, ms) { var i = 0, red = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches; if (red) { draw(states[states.length - 1]); return; } var tm = setInterval(function () { draw(states[i++]); if (i >= states.length) clearInterval(tm); }, ms || 120); return tm; };
    return put(box, opts);
  };

  /* triptych: masked, prediction, original (each an 8x8 array, values may be fractional) */
  V.triptych = function (masked, pred, orig, opts) {
    opts = opts || {}; var names = opts.names || ['masked input', 'prediction', 'original'];
    var row = h('div', { class: 'side-by-side', style: 'grid-template-columns:repeat(3,auto);justify-content:start;gap:16px;align-items:start' });
    [masked, pred, orig].forEach(function (g, i) {
      var f = h('figure', { class: 'vfig', style: 'margin:0' });
      var s = V.grid(g, { cell: opts.cell || 22, labels: opts.labels, patchLines: true }); f.appendChild(s);
      if (opts.hidden && i === 0) { var sv = s.querySelector('svg'), cell = opts.cell || 22; opts.hidden.forEach(function (j) { var pr = Math.floor(j / 4), pc = j % 4; var r = svg('rect', { x: 2 * pc * cell, y: 2 * pr * cell, width: 2 * cell, height: 2 * cell, fill: 'url(#vhatch)' }); sv.appendChild(r); }); if (!sv.querySelector('#vhatch')) { var d = svg('defs'), p = svg('pattern', { id: 'vhatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }); p.appendChild(svg('rect', { width: 6, height: 6, fill: '#F7F8FA' })); p.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: 'var(--ink-3)', 'stroke-width': 2 })); d.appendChild(p); sv.insertBefore(d, sv.firstChild); } }
      f.appendChild(h('figcaption', {}, names[i])); row.appendChild(f);
    });
    return put(row, opts);
  };

  /* curve: [[step, value], ...] with an optional marker step */
  V.curve = function (data, opts) {
    opts = opts || {}; var W = opts.width || 420, H = opts.height || 180, m = { l: 46, r: 12, t: 10, b: 30 };
    var xs = data.map(function (d) { return d[0]; }), ys = data.map(function (d) { return d[1]; });
    var xmax = Math.max.apply(null, xs), ymax = Math.max.apply(null, ys.concat([1e-6])), ymin = 0;
    var X = function (x) { return m.l + x / xmax * (W - m.l - m.r); }, Y = function (y) { return H - m.b - (y - ymin) / (ymax - ymin) * (H - m.t - m.b); };
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', style: 'max-width:' + W + 'px', class: 'vcurve', role: 'img' });
    s.appendChild(svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'var(--card)', stroke: 'var(--line)' }));
    s.appendChild(svg('path', { d: data.map(function (d, i) { return (i ? 'L' : 'M') + X(d[0]) + ' ' + Y(d[1]); }).join(''), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }));
    s.appendChild(svg('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 12 }, opts.xlabel || 'training steps'));
    s.appendChild(svg('text', { x: m.l - 6, y: Y(ymax) + 4, 'text-anchor': 'end', 'font-size': 11, 'font-family': 'var(--font-mono)' }, AT.fmt(ymax, 2)));
    s.appendChild(svg('text', { x: m.l - 6, y: Y(0) + 4, 'text-anchor': 'end', 'font-size': 11, 'font-family': 'var(--font-mono)' }, '0'));
    s.appendChild(svg('text', { x: X(xmax), y: H - m.b + 13, 'text-anchor': 'end', 'font-size': 11, 'font-family': 'var(--font-mono)' }, String(xmax)));
    var mark = svg('g'); s.appendChild(mark);
    var box = h('div', { class: 'vfig' }, s); if (opts.label) box.appendChild(h('figcaption', {}, opts.label));
    box.setMark = function (step) { while (mark.firstChild) mark.removeChild(mark.firstChild); if (step == null) return; var y = null; for (var i = 0; i < data.length; i++) if (data[i][0] <= step) y = data[i][1]; if (y == null) return; mark.appendChild(svg('line', { x1: X(step), y1: m.t, x2: X(step), y2: H - m.b, stroke: 'var(--c-a)', 'stroke-dasharray': '4 3' })); mark.appendChild(svg('circle', { cx: X(step), cy: Y(y), r: 5, fill: 'var(--c-a)' })); mark.appendChild(svg('text', { x: X(step) + 8, y: Y(y) - 6, 'font-size': 12, fill: 'var(--c-a)' }, 'loss ' + AT.fmt(y, 3))); };
    if (opts.mark != null) box.setMark(opts.mark);
    return put(box, opts);
  };
})();

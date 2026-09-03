/* shared.js: the AT namespace, with toy-model math, KaTeX helpers, UI components, motif and flow.
   The ONLY global this file creates is window.AT. See CONTRACT.md for the API. */
(function () {
  'use strict';
  var AT = {};

  /* ======================================================================
     0. small utilities
     ====================================================================== */
  function isNum(x) { return typeof x === 'number' && !isNaN(x); }
  function arr(x) { return Array.isArray(x) ? x : []; }
  function toEl(x) { return x instanceof Node ? x : (typeof x === 'string' ? document.getElementById(x) : null); }
  function reducedMotion() {
    try { return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; }
  }
  /** h(tag, attrs, ...children): tiny element builder. attrs: {class, id, style(string|obj), dataset, on{event:fn}, html, text, ...attributes}. Children: Node | string(text) | array | null. */
  function h(tag, attrs) {
    var el = document.createElement(tag || 'div');
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v == null || v === false) return;
      if (k === 'class' || k === 'className') el.className = v;
      else if (k === 'style') { if (typeof v === 'string') el.style.cssText = v; else Object.keys(v).forEach(function (p) { el.style[p] = v[p]; }); }
      else if (k === 'dataset') Object.keys(v).forEach(function (d) { el.dataset[d] = v[d]; });
      else if (k === 'on') Object.keys(v).forEach(function (ev) { el.addEventListener(ev, v[ev]); });
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'text') el.textContent = v;
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    });
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function append(el, c) {
    if (c == null || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { append(el, x); }); return; }
    if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
  function clear(el) { el = toEl(el); if (el) while (el.firstChild) el.removeChild(el.firstChild); return el; }
  function debounce(fn, ms) {
    var t = null;
    return function () { var a = arguments, self = this; clearTimeout(t); t = setTimeout(function () { fn.apply(self, a); }, ms == null ? 120 : ms); };
  }
  function onVisible(el, cb, opts) {
    el = toEl(el);
    if (!el || typeof cb !== 'function') return;
    if (!('IntersectionObserver' in window)) { cb(el); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { io.disconnect(); cb(el); } });
    }, opts || { threshold: 0.15 });
    io.observe(el);
  }
  AT.h = h; AT.append = append; AT.clear = clear; AT.debounce = debounce; AT.onVisible = onVisible; AT.reducedMotion = reducedMotion;

  /* ======================================================================
     1. toy model + math
     ====================================================================== */
  var model = (typeof window !== 'undefined' && window.__TOY__) ? window.__TOY__ : {};
  AT.model = model;
  AT.d_model = isNum(model.d_model) ? model.d_model : arr(model.axes && model.axes.e).length;
  AT.d_k = isNum(model.d_k) ? model.d_k : arr(model.axes && model.axes.qk).length;
  AT.d_v = isNum(model.d_v) ? model.d_v : arr(model.axes && model.axes.v).length;
  AT.sqrt_dk = Math.sqrt(AT.d_k);
  AT.vocab = arr(model.vocab);
  AT.sentences = model.sentences || {};
  if (!AT.sentences.river) AT.sentences.river = ["The", "fisherman", "sat", "beside", "the", "river", "bank", "and", "watched", "the"];
  if (!AT.sentences.cheque) AT.sentences.cheque = ["She", "deposited", "the", "cheque", "at", "the", "bank", "and", "watched", "the"];
  AT.candidates = model.candidates || { river: [], cheque: [] };

  /* named coordinates (toy.json "axes"); falls back to index labels so every component still renders without them */
  function idxLabels(n) { var o = []; for (var i = 0; i < (n | 0); i++) o.push(String(i + 1)); return o; }
  function strList(list, n) { list = arr(list).map(function (s) { return s == null ? '' : String(s); }); return list.length === n ? list : null; }
  (function () {
    var ax = (model.axes && typeof model.axes === 'object') ? model.axes : {};
    var sh = (ax.short && typeof ax.short === 'object') ? ax.short : {};
    var e = strList(ax.e, AT.d_model), qk = strList(ax.qk, AT.d_k), v = strList(ax.v, AT.d_v);
    AT.axes = {
      named: !!(e && qk && v),
      e: e || idxLabels(AT.d_model), qk: qk || idxLabels(AT.d_k), v: v || idxLabels(AT.d_v),
      short: { e: strList(sh.e, AT.d_model) || e || idxLabels(AT.d_model), qk: strList(sh.qk, AT.d_k) || qk || idxLabels(AT.d_k), v: strList(sh.v, AT.d_v) || v || idxLabels(AT.d_v) }
    };
  })();
  /** axisKind(cls, n) → 'e' | 'qk' | 'v' | null: which named axes a vector of class `cls` and width `n` lives on */
  AT.axisKind = function (cls, n) {
    cls = cls || 'neutral';
    if (n === AT.d_model && (cls === 'e' || cls === 'ep' || cls === 'd')) return 'e';
    if (n === AT.d_k && (cls === 'q' || cls === 'k')) return 'qk';
    if (n === AT.d_v && (cls === 'v' || cls === 'm')) return 'v';
    var matches = [];
    if (n === AT.d_model) matches.push('e');
    if (n === AT.d_k) matches.push('qk');
    if (n === AT.d_v) matches.push('v');
    return matches.length === 1 ? matches[0] : null;
  };
  /** axesFor(kind | cls, n?, short=true) → [{label, title}] or null. kind: 'e' | 'qk' | 'v' | 'vocab' | array of labels | false */
  AT.axesFor = function (kind, n, short) {
    if (kind === false) return null;
    if (Array.isArray(kind)) return kind.map(function (s) { return (s && typeof s === 'object') ? s : { label: s == null ? '' : String(s), title: '' }; });
    if (kind === 'vocab') return AT.vocab.map(function (w) { return { label: w, title: '' }; });
    if (kind !== 'e' && kind !== 'qk' && kind !== 'v') kind = AT.axisKind(kind, n);
    if (!kind) return null;
    var full = AT.axes[kind], sh = short === false ? full : AT.axes.short[kind];
    return sh.map(function (s, j) { return { label: s, title: full[j] !== s ? full[j] : '' }; });
  };
  AT.axisLabels = function (kind, n, short) { var a = AT.axesFor(kind, n, short); return a ? a.map(function (x) { return x.label; }) : null; };

  AT.lower = function (tok) { return String(tok == null ? '' : tok).toLowerCase(); };
  AT.zeros = function (n) { var o = []; for (var i = 0; i < (n | 0); i++) o.push(0); return o; };
  AT.range = function (n) { var o = []; for (var i = 0; i < (n | 0); i++) o.push(i); return o; };
  AT.dot = function (a, b) { a = arr(a); b = arr(b); var s = 0, n = Math.min(a.length, b.length); for (var i = 0; i < n; i++) s += (+a[i] || 0) * (+b[i] || 0); return s; };
  AT.add = function (a, b) { a = arr(a); b = arr(b); var n = Math.max(a.length, b.length), o = []; for (var i = 0; i < n; i++) o.push((+a[i] || 0) + (+b[i] || 0)); return o; };
  AT.sub = function (a, b) { return AT.add(a, AT.scale(b, -1)); };
  AT.scale = function (v, s) { return arr(v).map(function (x) { return (+x || 0) * (+s || 0); }); };
  AT.transpose = function (M) {
    M = arr(M); if (!M.length) return [];
    var cols = arr(M[0]).length, o = [];
    for (var j = 0; j < cols; j++) { var r = []; for (var i = 0; i < M.length; i++) r.push(+arr(M[i])[j] || 0); o.push(r); }
    return o;
  };
  /** matmul(A, B): rows × rows. Also accepts a vector as A (1×n) and returns a vector. */
  AT.matmul = function (A, B) {
    B = arr(B); if (!B.length) return [];
    var vec = Array.isArray(A) && A.length && !Array.isArray(A[0]);
    var rows = vec ? [A] : arr(A);
    var cols = arr(B[0]).length, out = [];
    for (var i = 0; i < rows.length; i++) {
      var r = arr(rows[i]), o = [];
      for (var j = 0; j < cols; j++) { var s = 0; for (var k = 0; k < r.length && k < B.length; k++) s += (+r[k] || 0) * (+arr(B[k])[j] || 0); o.push(s); }
      out.push(o);
    }
    return vec ? out[0] : out;
  };
  AT.softmax = function (a) {
    a = arr(a).map(function (x) { return x === -Infinity ? -Infinity : (isNum(x) ? x : 0); });
    var m = -Infinity; a.forEach(function (x) { if (x > m) m = x; });
    if (m === -Infinity) return a.map(function () { return 0; });
    var ex = a.map(function (x) { return x === -Infinity ? 0 : Math.exp(x - m); });
    var z = ex.reduce(function (s, x) { return s + x; }, 0);
    return z > 0 ? ex.map(function (x) { return x / z; }) : ex.map(function () { return 0; });
  };
  AT.argmax = function (a) { a = arr(a); var b = -1, bv = -Infinity; a.forEach(function (x, i) { if (x > bv) { bv = x; b = i; } }); return b; };
  AT.sum = function (a) { return arr(a).reduce(function (s, x) { return s + (+x || 0); }, 0); };
  AT.mean = function (rows) { rows = arr(rows); if (!rows.length) return []; var o = AT.zeros(arr(rows[0]).length); rows.forEach(function (r) { o = AT.add(o, r); }); return AT.scale(o, 1 / rows.length); };
  AT.norm = function (a) { return Math.sqrt(AT.dot(a, a)); };

  /** initial representation e_i^(0) = tok_emb[lower(token)] + pos_emb[i]  (rows) */
  AT.embed = function (tokens) {
    tokens = arr(tokens);
    var te = model.tok_emb || {}, pe = arr(model.pos_emb), d = AT.d_model;
    return tokens.map(function (t, i) {
      var a = arr(te[AT.lower(t)]), p = arr(pe[i]);
      if (!a.length) a = AT.zeros(d);
      if (!p.length) p = AT.zeros(d);
      return AT.add(a, p);
    });
  };
  /** single-head causal attention forward pass (row-vector convention). See BRIEF §4. */
  AT.forward = function (tokens, opts) {
    opts = opts || {};
    var mask = opts.mask !== false, scale = opts.scale !== false;
    tokens = arr(tokens);
    var T = tokens.length;
    var E = AT.embed(tokens);
    var WQ = arr(model.W_Q), WK = arr(model.W_K), WV = arr(model.W_V), WO = arr(model.W_O), WVo = arr(model.W_vocab), b = arr(model.b_vocab);
    var Q = WQ.length ? AT.matmul(E, WQ) : E.map(function () { return AT.zeros(AT.d_k); });
    var K = WK.length ? AT.matmul(E, WK) : E.map(function () { return AT.zeros(AT.d_k); });
    var V = WV.length ? AT.matmul(E, WV) : E.map(function () { return AT.zeros(AT.d_v); });
    var Sraw = Q.map(function (q) { return K.map(function (k) { return AT.dot(q, k); }); });
    var sc = scale ? 1 / AT.sqrt_dk : 1;
    var Sfull = Sraw.map(function (r) { return r.map(function (x) { return x * sc; }); });
    var S = Sfull.map(function (r, i) { return r.map(function (x, j) { return (mask && j > i) ? -Infinity : x; }); });
    var A = S.map(function (r) { return AT.softmax(r); });
    var Mmsg = A.map(function (a) { var m = AT.zeros(V.length ? V[0].length : AT.d_v); a.forEach(function (w, j) { if (w) m = AT.add(m, AT.scale(V[j], w)); }); return m; });
    var Delta = WO.length ? AT.matmul(Mmsg, WO) : Mmsg.map(function () { return AT.zeros(AT.d_model); });
    var Enew = E.map(function (e, i) { return AT.add(e, Delta[i]); });
    var logits = WVo.length ? AT.matmul(Enew, WVo).map(function (r) { return AT.add(r, b); }) : Enew.map(function () { return AT.zeros(AT.vocab.length); });
    var probs = logits.map(function (r) { return AT.softmax(r); });
    return { tokens: tokens, T: T, E: E, Q: Q, K: K, V: V, Sraw: Sraw, S: S, Sfull: Sfull, A: A, Mmsg: Mmsg, Delta: Delta, Enew: Enew, logits: logits, probs: probs, mask: mask, scale: scale };
  };
  /** output head applied directly to e^(0) (no attention) */
  AT.baseline = function (tokens) {
    var E = AT.embed(arr(tokens));
    var WVo = arr(model.W_vocab), b = arr(model.b_vocab);
    var logits = WVo.length ? AT.matmul(E, WVo).map(function (r) { return AT.add(r, b); }) : E.map(function () { return AT.zeros(AT.vocab.length); });
    return { E: E, logits: logits, probs: logits.map(function (r) { return AT.softmax(r); }) };
  };
  /** output head applied to any single vector (row) → {logits, probs} */
  AT.head = function (e) {
    var WVo = arr(model.W_vocab), b = arr(model.b_vocab);
    var logits = WVo.length ? AT.add(AT.matmul(arr(e), WVo), b) : AT.zeros(AT.vocab.length);
    return { logits: logits, probs: AT.softmax(logits) };
  };
  AT.topk = function (probRow, k) {
    probRow = arr(probRow);
    var items = probRow.map(function (p, i) { return { tok: AT.vocab[i] != null ? AT.vocab[i] : String(i), p: +p || 0, i: i }; });
    items.sort(function (a, b) { return b.p - a.p; });
    return items.slice(0, k == null ? 5 : Math.max(0, k | 0));
  };
  /** probabilities for a list of candidate tokens (in the given order) */
  AT.probsFor = function (probRow, toks) {
    probRow = arr(probRow);
    return arr(toks).map(function (t) { var i = AT.vocab.indexOf(AT.lower(t)); return { tok: t, p: i >= 0 ? (+probRow[i] || 0) : 0, i: i }; });
  };
  AT.fmt = function (x, decimals) {
    var d = decimals == null ? 2 : decimals;
    if (x === -Infinity) return '−∞';
    if (x === Infinity) return '∞';
    if (!isNum(x)) return '·';
    var s = Math.abs(x).toFixed(d);
    var neg = x < 0 && +s !== 0;
    return (neg ? '−' : '') + s;
  };
  AT.fmtSigned = function (x, decimals) { var s = AT.fmt(x, decimals); return (isNum(x) && x >= 0 && s !== '·') ? '+' + s : s; };
  AT.heatColor = function (a) {
    a = isNum(a) ? Math.max(0, Math.min(1, a)) : 0;
    var r = Math.round(255 + (225 - 255) * a), g = Math.round(255 + (29 - 255) * a), b = Math.round(255 + (72 - 255) * a);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  };
  AT.objColor = function (cls) { return ({ e: 'var(--c-e)', q: 'var(--c-q)', k: 'var(--c-k)', v: 'var(--c-v)', a: 'var(--c-a)', d: 'var(--c-d)', m: 'var(--c-v)', ep: 'var(--c-e)', neutral: 'var(--ink-3)' })[cls] || 'var(--ink-3)'; };
  AT.objTint = function (cls) { return ({ e: 'var(--t-e)', q: 'var(--t-q)', k: 'var(--t-k)', v: 'var(--t-v)', a: 'var(--t-a)', d: 'var(--t-d)', m: 'var(--t-v)', ep: 'var(--t-e)', neutral: 'var(--t-neutral)' })[cls] || 'var(--t-neutral)'; };

  /* ======================================================================
     2. KaTeX
     ====================================================================== */
  var MACROS = {
    "\\ve": "\\htmlClass{m-e}{#1}", "\\vq": "\\htmlClass{m-q}{#1}", "\\vk": "\\htmlClass{m-k}{#1}", "\\vv": "\\htmlClass{m-v}{#1}",
    "\\va": "\\htmlClass{m-a}{#1}", "\\vd": "\\htmlClass{m-d}{#1}", "\\vp": "\\htmlClass{m-ep}{#1}"
  };
  AT.macros = MACROS;
  AT.katexOpts = function (extra) {
    var o = { throwOnError: false, trust: true, strict: false, macros: Object.assign({}, MACROS) };
    return extra ? Object.assign(o, extra) : o;
  };
  AT.tex = function (el, latex, opts) {
    el = toEl(el); if (!el) return null;
    opts = opts || {};
    latex = latex == null ? '' : String(latex);
    if (typeof katex === 'undefined' || !katex.render) { el.textContent = latex; return el; }
    try { katex.render(latex, el, AT.katexOpts({ displayMode: !!opts.display })); }
    catch (e) { el.textContent = latex; }
    if (opts.display) el.classList.add('tex-display');
    return el;
  };
  AT.texStr = function (latex, opts) {
    opts = opts || {};
    latex = latex == null ? '' : String(latex);
    if (typeof katex === 'undefined' || !katex.renderToString) return '<span class="no-math">' + escapeHtml(latex) + '</span>';
    try { return katex.renderToString(latex, AT.katexOpts({ displayMode: !!opts.display })); }
    catch (e) { return '<span class="no-math">' + escapeHtml(latex) + '</span>'; }
  };
  /** auto-render $…$ / $$…$$ inside an element (for markup created after boot) */
  AT.renderMath = function (el) {
    el = toEl(el); if (!el) return el;
    if (typeof renderMathInElement !== 'function') return el;
    try {
      renderMathInElement(el, AT.katexOpts({
        delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
        ignoredClasses: ["no-math"]
      }));
    } catch (e) { /* ignore */ }
    return el;
  };
  function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  AT.escape = escapeHtml;

  /* ======================================================================
     3. UI components
     ====================================================================== */
  var ui = {};
  AT.ui = ui;
  function put(el, opts) { if (opts && opts.into) { var t = toEl(opts.into); if (t) t.appendChild(el); } return el; }
  function objClass(cls) { return cls ? 'obj-' + cls : ''; }

  /* ---- chips ---- */
  ui.chips = function (tokens, opts) {
    opts = opts || {};
    tokens = arr(tokens);
    var el = h('div', { class: 'chips' + (opts.size === 'lg' ? ' lg' : '') + (opts.cls ? ' ' + opts.cls : ''), role: opts.onClick ? 'group' : null });
    var numbered = opts.numbered !== false, muted = arr(opts.muted), interactive = typeof opts.onClick === 'function';
    var chipEls = tokens.map(function (t, i) {
      var c = h(interactive ? 'button' : 'span', { class: 'chip', type: interactive ? 'button' : null, dataset: { i: i }, 'aria-pressed': interactive ? 'false' : null },
        numbered ? h('span', { class: 'chip-i' }, String(i + 1)) : null,
        h('span', { class: 'chip-t' }, String(t)));
      if (interactive) c.addEventListener('click', function () { opts.onClick(i, c); });
      el.appendChild(c);
      return c;
    });
    el.chips = chipEls;
    el.setActive = function (i) {
      chipEls.forEach(function (c, j) { var on = (Array.isArray(i) ? i.indexOf(j) >= 0 : j === i); c.classList.toggle('is-active', on); if (interactive) c.setAttribute('aria-pressed', on ? 'true' : 'false'); });
      return el;
    };
    el.setMuted = function (list) { list = arr(list); chipEls.forEach(function (c, j) { c.classList.toggle('is-muted', list.indexOf(j) >= 0); }); return el; };
    el.setMark = function (list, kind) { list = arr(list); chipEls.forEach(function (c, j) { c.classList.toggle('is-hl-' + (kind || 'k'), list.indexOf(j) >= 0); }); return el; };
    if (opts.active != null) el.setActive(opts.active);
    if (muted.length) el.setMuted(muted);
    if (opts.slot) el.appendChild(h('span', { class: 'chip is-slot' }, numbered ? h('span', { class: 'chip-i' }, String(tokens.length + 1)) : null, h('span', { class: 'chip-t' }, opts.slot === true ? '___' : String(opts.slot))));
    return put(el, opts);
  };

  /* ---- vec ---- */
  ui.vec = function (values, opts) {
    opts = opts || {};
    values = arr(values);
    var cls = opts.cls || 'neutral', dec = opts.decimals == null ? 2 : opts.decimals;
    var el = h('div', { class: 'vec ' + objClass(cls) + (opts.size ? ' ' + opts.size : '') });
    var lab = h('span', { class: 'vec-label' });
    if (opts.label) { AT.tex(lab, opts.label); el.appendChild(lab); }
    var body = h('div', { class: 'vec-body' });
    var axesRow = h('div', { class: 'vec-axes', 'aria-hidden': 'true' });
    var cells = h('div', { class: 'vec-cells', role: 'list' });
    var cellEls = values.map(function (v, i) {
      var c = h('span', { class: 'cell' + (i === opts.highlight ? ' is-hl' : ''), role: 'listitem' }, AT.fmt(v, dec));
      cells.appendChild(c); return c;
    });
    body.appendChild(axesRow); body.appendChild(cells);
    el.appendChild(body);
    if (opts.dims) el.appendChild(h('span', { class: 'vec-dim' }, typeof opts.dims === 'string' ? opts.dims : ('1×' + values.length)));
    /* axis names above the cells: default = the named axes of this object when the width matches; axes:false hides them; axes: [..] | 'e' | 'qk' | 'v' overrides */
    function setAxes(spec) {
      var ax = spec === false ? null : AT.axesFor(spec == null ? cls : spec, cellEls.length, opts.shortAxes !== false);
      clear(axesRow);
      if (!ax || ax.length !== cellEls.length) { axesRow.hidden = true; el.classList.remove('has-axes'); el.axes = null; body.style.removeProperty('grid-template-columns'); return el; }
      ax.forEach(function (a, i) { axesRow.appendChild(h('span', { class: 'vec-ax', title: a.title || null }, a.label)); cellEls[i].setAttribute('aria-label', (a.title || a.label) + ': ' + cellEls[i].textContent); });
      body.style.gridTemplateColumns = 'repeat(' + ax.length + ', auto)';
      axesRow.hidden = false; el.classList.add('has-axes'); el.axes = ax.map(function (a) { return a.label; });
      return el;
    }
    el.cells = cellEls; el.labelEl = lab; el.axesEl = axesRow;
    el.update = function (vals) {
      vals = arr(vals);
      var n0 = cellEls.length;
      vals.forEach(function (v, i) { if (cellEls[i]) cellEls[i].textContent = AT.fmt(v, dec); else { var c = h('span', { class: 'cell', role: 'listitem' }, AT.fmt(v, dec)); cells.appendChild(c); cellEls.push(c); } });
      while (cellEls.length > vals.length) cells.removeChild(cellEls.pop());
      if (cellEls.length !== n0 || el.axes) setAxes(opts.axes);
      return el;
    };
    el.highlight = function (i) { cellEls.forEach(function (c, j) { c.classList.toggle('is-hl', j === i); }); return el; };
    el.setLabel = function (latex) { if (!lab.parentNode) el.insertBefore(lab, body); AT.tex(lab, latex); return el; };
    el.setAxes = function (spec) { opts.axes = spec; return setAxes(spec); };
    setAxes(opts.axes);
    return put(el, opts);
  };

  /* ---- mat ---- */
  ui.mat = function (rows, opts) {
    opts = opts || {};
    rows = arr(rows).map(arr);
    var m = rows.length, n = m ? Math.max.apply(null, rows.map(function (r) { return r.length; })) : 0;
    var cls = opts.cls || 'neutral', dec = opts.decimals == null ? 2 : opts.decimals;
    var rl = opts.rowLabels ? arr(opts.rowLabels) : null, cl = opts.colLabels ? arr(opts.colLabels) : null;
    /* default labels from the named axes: columns when the width matches the object; rows for the projection matrices (W_Q, W_K, W_V: e axes; W_O: v axes) */
    var axo = opts.axes, rlT = null, clT = null;
    if (axo !== false) {
      var axc = (axo && typeof axo === 'object' && !Array.isArray(axo)) ? axo.cols : axo, axr = (axo && typeof axo === 'object' && !Array.isArray(axo)) ? axo.rows : null;
      var cdef = null, rdef = null;
      if (!cl) {
        if (axc != null) cdef = AT.axesFor(axc, n);
        else cdef = AT.axesFor(cls, n) || ((cls === 'neutral' || cls === 'd') && m === AT.d_v && n === AT.d_model ? AT.axesFor('e', n) : null);
        if (cdef && cdef.length === n) { cl = cdef.map(function (a) { return a.label; }); clT = cdef.map(function (a) { return a.title; }); }
      }
      if (!rl) {
        if (axr != null) rdef = AT.axesFor(axr, m);
        else if (m === AT.d_model && ((cls === 'q' || cls === 'k') && n === AT.d_k || cls === 'v' && n === AT.d_v)) rdef = AT.axesFor('e', m);
        else if ((cls === 'neutral' || cls === 'd') && m === AT.d_v && n === AT.d_model) rdef = AT.axesFor('v', m);
        if (rdef && rdef.length === m) { rl = rdef.map(function (a) { return a.label; }); rlT = rdef.map(function (a) { return a.title; }); }
      }
    }
    var fig = h('figure', { class: 'mat ' + objClass(cls) + (opts.size ? ' ' + opts.size : '') + (opts.heat ? ' heat' : ''), style: 'margin:6px 0' });
    var scroll = h('div', { class: 'mat-scroll' });
    var grid = h('div', { class: 'mat-grid', role: 'table' });
    grid.style.gridTemplateColumns = 'auto 8px repeat(' + Math.max(1, n) + ', minmax(3.6em, auto)) 8px';
    var cellEls = [], rlEls = [], clEls = [];
    function hlKind(i, j) {
      var hl = opts.highlight; if (!hl) return '';
      if (isNum(hl.r) && isNum(hl.c)) return (hl.r === i && hl.c === j) ? ' is-hl' : '';
      var s = '';
      if (isNum(hl.row) && hl.row === i) s += ' is-hl-row';
      if (isNum(hl.col) && hl.col === j) s += ' is-hl-col';
      return s;
    }
    if (cl) {
      grid.appendChild(h('div', { class: 'mat-corner' }));
      grid.appendChild(h('div'));
      for (var j = 0; j < n; j++) { var c = h('div', { class: 'mat-cl' + (opts.highlight && isNum(opts.highlight.col) && opts.highlight.col === j ? ' is-hl' : ''), role: 'columnheader', title: clT && clT[j] ? clT[j] : null }, cl[j] != null ? String(cl[j]) : ''); clEls.push(c); grid.appendChild(c); }
      grid.appendChild(h('div'));
    }
    for (var i = 0; i < m; i++) {
      var rlab = h('div', { class: 'mat-rl' + (opts.highlight && isNum(opts.highlight.row) && opts.highlight.row === i ? ' is-hl' : ''), role: 'rowheader', title: rlT && rlT[i] ? rlT[i] : null }, rl && rl[i] != null ? String(rl[i]) : '');
      rlEls.push(rlab); grid.appendChild(rlab);
      if (i === 0) grid.appendChild(h('div', { class: 'mat-lb', style: 'grid-row: span ' + Math.max(1, m), 'aria-hidden': 'true' }));
      var rowEls = [];
      for (var jj = 0; jj < n; jj++) {
        var v = rows[i][jj];
        var cell = h('span', { class: 'cell' + hlKind(i, jj), role: 'cell', dataset: { r: i, c: jj } });
        styleCell(cell, v, i, jj);
        rowEls.push(cell); grid.appendChild(cell);
      }
      cellEls.push(rowEls);
      if (i === 0) grid.appendChild(h('div', { class: 'mat-rb', style: 'grid-row: span ' + Math.max(1, m), 'aria-hidden': 'true' }));
    }
    function masked(i, j) {
      var mk = opts.mask;
      if (!mk) return false;
      if (mk === true || mk === 'causal') return j > i;
      return !!(arr(mk[i])[j]);
    }
    function styleCell(cell, v, i, j) {
      cell.classList.remove('is-masked', 'is-heat', 'is-dark', 'is-leak');
      cell.style.removeProperty('--heat');
      if (masked(i, j)) { cell.classList.add('is-masked'); if (opts.leak) cell.classList.add('is-leak'); cell.textContent = opts.maskText != null ? opts.maskText : '×'; cell.title = 'masked (future token)'; return; }
      cell.textContent = AT.fmt(v, dec);
      cell.removeAttribute('title');
      if (opts.leak && j > i) { cell.classList.add('is-leak'); cell.title = 'future token: visible only because the mask is off'; }
      if (opts.heat) {
        var a = isNum(v) ? v : 0;
        var mx = isNum(opts.heatMax) ? opts.heatMax : 1;
        var t = mx > 0 ? a / mx : 0;
        cell.classList.add('is-heat');
        cell.style.setProperty('--heat', AT.heatColor(t));
        if (t > 0.55) cell.classList.add('is-dark');
      }
    }
    scroll.appendChild(grid);
    fig.appendChild(scroll);
    var cap = null;
    if (opts.caption) { cap = h('figcaption', { class: 'mat-cap', html: opts.caption }); AT.renderMath(cap); fig.appendChild(cap); }
    if (opts.dims) { var dm = h('span', { class: 'mat-dim' }, typeof opts.dims === 'string' ? opts.dims : (m + '×' + n)); (cap || fig).appendChild(dm); }
    fig.cells = cellEls; fig.rowLabelEls = rlEls; fig.colLabelEls = clEls; fig.grid = grid;
    fig.update = function (newRows, newOpts) {
      if (newOpts) Object.assign(opts, newOpts);
      newRows = arr(newRows).map(arr); rows = newRows;
      for (var i = 0; i < cellEls.length; i++) for (var j = 0; j < cellEls[i].length; j++) styleCell(cellEls[i][j], arr(newRows[i])[j], i, j);
      return fig;
    };
    fig.setHighlight = function (hl) {
      opts.highlight = hl || null;
      cellEls.forEach(function (r, i) { r.forEach(function (c, j) { c.classList.remove('is-hl', 'is-hl-row', 'is-hl-col'); var k = hlKind(i, j).trim(); if (k) k.split(' ').forEach(function (x) { c.classList.add(x); }); }); });
      rlEls.forEach(function (r, i) { r.classList.toggle('is-hl', !!(hl && isNum(hl.row) && hl.row === i)); });
      clEls.forEach(function (c, j) { c.classList.toggle('is-hl', !!(hl && isNum(hl.col) && hl.col === j)); });
      return fig;
    };
    fig.setMask = function (mask) { opts.mask = mask; return fig.update(rows); };
    fig.onCell = function (event, fn) { cellEls.forEach(function (r, i) { r.forEach(function (c, j) { c.addEventListener(event, function (ev) { fn(i, j, c, ev); }); }); }); return fig; };
    /* calc: { Q, K, scaled, d_k, axes } → clicking cell (i, j) opens a dotCalc popover for q_i · k_j (the score behind the cell) */
    if (opts.calc && typeof opts.calc === 'object') {
      var CQ = arr(opts.calc.Q).map(arr), CK = arr(opts.calc.K).map(arr), pop = popover(fig, { label: 'Score arithmetic' });
      fig.calcPop = pop;
      var openCalc = function (i, j, c) {
        if (!CQ[i] || !CK[j]) return;
        var ri = rl && rl[i] != null ? String(rl[i]) : String(i + 1), cj = cl && cl[j] != null ? String(cl[j]) : String(j + 1);
        var calc = ui.dotCalc(CQ[i], CK[j], { axes: opts.calc.axes != null ? opts.calc.axes : 'qk', qLabel: 'q · ' + ri, kLabel: 'k · ' + cj, scale: opts.calc.scaled !== false, d_k: opts.calc.d_k, decimals: dec });
        var extra = [];
        if (masked(i, j)) extra.push(h('p', { class: 'calc-line calc-masked' }, 'masked: token ' + cj + ' comes after token ' + ri + ', so this score is replaced by −∞ and its weight is 0'));
        else if (opts.heat) extra.push(h('p', { class: 'calc-line obj-a' }, 'after the softmax over row ' + ri + ': weight ', h('span', { class: 'calc-r' }, AT.fmt(rows[i] ? rows[i][j] : NaN, dec))));
        pop.show(c, 'q · k for row ' + ri + ', column ' + cj, [calc].concat(extra));
      };
      fig.onCell('click', openCalc);
      fig.onCell('keydown', function (i, j, c, ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openCalc(i, j, c); } });
      cellEls.forEach(function (r) { r.forEach(function (c) { c.tabIndex = 0; c.classList.add('has-calc'); }); });
    }
    return put(fig, opts);
  };

  /* ---- popover: a small dialog anchored below a cell, appended to the figure (outside the scroll container) ---- */
  function placeBelow(fig, anchor, el) {
    var fr = fig.getBoundingClientRect(), ar = anchor.getBoundingClientRect();
    var vw = document.documentElement.clientWidth || window.innerWidth || 1280;
    el.style.top = (ar.bottom - fr.top + 6) + 'px';
    var w = el.offsetWidth, left = ar.left - fr.left, minLeft = 12 - fr.left, maxLeft = vw - 12 - fr.left - w;
    if (maxLeft < minLeft) maxLeft = minLeft;
    el.style.left = Math.max(minLeft, Math.min(left, maxLeft)) + 'px';
  }
  function popover(fig, opts) {
    opts = opts || {};
    var close = h('button', { type: 'button', class: 'calc-close', 'aria-label': 'Close' }, '×');
    var titleEl = h('p', { class: 'calc-pop-title' }), bodyEl = h('div', { class: 'calc-pop-body' });
    var pop = h('div', { class: 'calc-pop', role: 'dialog', 'aria-label': opts.label || 'Arithmetic', hidden: true }, close, titleEl, bodyEl);
    fig.appendChild(pop);
    var anchor = null;
    function onOutside(ev) { if (pop.contains(ev.target) || (anchor && anchor.contains(ev.target))) return; hide(); }
    function onKey(ev) { if (ev.key === 'Escape') { hide(true); ev.stopPropagation(); } }
    function hide(refocus) {
      if (pop.hidden) return;
      pop.hidden = true;
      document.removeEventListener('mousedown', onOutside, true); document.removeEventListener('keydown', onKey, true);
      var a = anchor; anchor = null;
      if (refocus && a && a.focus) a.focus();
    }
    function show(td, title, content) {
      if (anchor === td && !pop.hidden) { hide(); return; }
      anchor = td; titleEl.textContent = title == null ? '' : String(title); clear(bodyEl); append(bodyEl, content);
      pop.hidden = false; placeBelow(fig, td, pop);
      if (fig.hideTip) fig.hideTip();
      document.addEventListener('mousedown', onOutside, true); document.addEventListener('keydown', onKey, true);
    }
    close.addEventListener('click', function () { hide(true); });
    return { el: pop, body: bodyEl, show: show, hide: hide, isOpen: function () { return !pop.hidden; } };
  }
  ui.popover = popover;

  /* ---- heat (attention heatmap) ---- */
  ui.heat = function (A, opts) {
    opts = opts || {};
    A = arr(A).map(arr);
    var rows = opts.rows ? arr(opts.rows) : null, cols = opts.cols ? arr(opts.cols) : rows;
    var mask = opts.mask == null ? true : opts.mask;
    var fig = ui.mat(A, {
      cls: 'a', heat: true, decimals: opts.decimals == null ? 2 : opts.decimals, mask: mask, leak: !!opts.leak,
      rowLabels: rows, colLabels: cols, caption: opts.caption, size: opts.size, dims: opts.dims, axes: opts.axes, calc: opts.calc,
      highlight: isNum(opts.highlightRow) ? { row: opts.highlightRow } : opts.highlight
    });
    fig.classList.add('heat');
    if (typeof opts.onHover === 'function') {
      fig.onCell('mouseenter', function (i, j, c) { c.classList.add('is-hover'); opts.onHover(i, j, c); });
      fig.onCell('mouseleave', function (i, j, c) { c.classList.remove('is-hover'); });
      fig.onCell('focus', function (i, j, c) { opts.onHover(i, j, c); });
      fig.cells.forEach(function (r) { r.forEach(function (c) { c.tabIndex = 0; }); });
    }
    if (typeof opts.onClick === 'function') { fig.onCell('click', function (i, j, c) { opts.onClick(i, j, c); }); fig.cells.forEach(function (r) { r.forEach(function (c) { c.style.cursor = 'pointer'; }); }); }
    fig.setHighlightRow = function (i) { return fig.setHighlight(isNum(i) ? { row: i } : null); };
    return put(fig, opts);
  };

  /* ---- bars ---- */
  ui.bars = function (items, opts) {
    opts = opts || {};
    var cls = opts.cls || 'neutral', dec = opts.decimals == null ? 2 : opts.decimals, max = isNum(opts.max) ? opts.max : 1;
    var el = h('div', { class: 'bars ' + objClass(cls) + (opts.size ? ' ' + opts.size : ''), role: 'list' });
    var rowsEls = [];
    function build(list) {
      clear(el); rowsEls = [];
      list = arr(list).map(function (it) { return { label: it && it.label != null ? String(it.label) : (it && it.tok != null ? String(it.tok) : ''), p: it ? (+it.p || 0) : 0 }; });
      if (opts.sorted !== false) list.sort(function (a, b) { return b.p - a.p; });
      list.forEach(function (it) {
        var isHl = opts.highlight != null && (Array.isArray(opts.highlight) ? opts.highlight.indexOf(it.label) >= 0 : it.label === opts.highlight);
        var bl = h('span', { class: 'bl' + (isHl ? ' is-hl' : ''), role: 'listitem' }, it.label);
        var fill = h('div', { class: 'bf' });
        var bt = h('div', { class: 'bt' + (isHl ? ' is-hl' : '') }, fill);
        var bv = h('span', { class: 'bv' + (isHl ? ' is-hl' : '') }, AT.fmt(it.p, dec));
        el.appendChild(bl); el.appendChild(bt); el.appendChild(bv);
        var w = max > 0 ? Math.max(0, Math.min(1, it.p / max)) : 0;
        if (reducedMotion()) fill.style.width = (w * 100) + '%'; else requestAnimationFrame(function () { fill.style.width = (w * 100) + '%'; });
        rowsEls.push({ label: it.label, bl: bl, bt: bt, fill: fill, bv: bv });
      });
    }
    build(items);
    el.rows = function () { return rowsEls; };
    el.update = function (newItems, newOpts) {
      if (newOpts) Object.assign(opts, newOpts);
      var list = arr(newItems).map(function (it) { return { label: it && it.label != null ? String(it.label) : (it && it.tok != null ? String(it.tok) : ''), p: it ? (+it.p || 0) : 0 }; });
      if (opts.sorted !== false) list.sort(function (a, b) { return b.p - a.p; });
      var same = !newOpts && list.length === rowsEls.length && list.every(function (it, i) { return rowsEls[i].label === it.label; });
      if (same) { list.forEach(function (it, i) { var r = rowsEls[i]; r.fill.style.width = (max > 0 ? Math.max(0, Math.min(1, it.p / max)) * 100 : 0) + '%'; r.bv.textContent = AT.fmt(it.p, dec); }); }
      else build(list);
      return el;
    };
    return put(el, opts);
  };

  /* ---- slider ---- */
  ui.slider = function (opts) {
    opts = opts || {};
    var id = 'at-sl-' + Math.random().toString(36).slice(2, 8);
    var input = h('input', { type: 'range', id: id, min: opts.min == null ? 0 : opts.min, max: opts.max == null ? 1 : opts.max, step: opts.step == null ? 1 : opts.step, value: opts.value == null ? (opts.min == null ? 0 : opts.min) : opts.value });
    var val = h('output', { class: 'slider-val', for: id });
    var el = h('div', { class: 'slider' }, h('label', { class: 'slider-label', for: id }, opts.label != null ? String(opts.label) : ''), input, val);
    function fmt(v) { return typeof opts.format === 'function' ? opts.format(v) : String(v); }
    function num() { return +input.value; }
    val.textContent = fmt(num());
    input.addEventListener('input', function () { val.textContent = fmt(num()); if (typeof opts.onInput === 'function') opts.onInput(num()); });
    el.input = input; el.output = val;
    el.value = function () { return num(); };
    el.setValue = function (v, fire) { input.value = v; val.textContent = fmt(num()); if (fire !== false && typeof opts.onInput === 'function') opts.onInput(num()); return el; };
    if (opts.labelTex) AT.tex(el.firstChild, opts.labelTex);
    return put(el, opts);
  };

  /* ---- toggle ---- */
  ui.toggle = function (opts) {
    opts = opts || {};
    var on = !!opts.on;
    var state = h('span', { class: 'toggle-state' });
    var btn = h('button', { type: 'button', class: 'toggle', 'aria-pressed': on ? 'true' : 'false' },
      h('span', { class: 'toggle-track', 'aria-hidden': 'true' }, h('span', { class: 'toggle-knob' })),
      h('span', { class: 'toggle-text' }, opts.label != null ? String(opts.label) : ''), state);
    function paint() { btn.setAttribute('aria-pressed', on ? 'true' : 'false'); state.textContent = on ? (opts.onText || 'on') : (opts.offText || 'off'); }
    paint();
    btn.addEventListener('click', function () { on = !on; paint(); if (typeof opts.onChange === 'function') opts.onChange(on); });
    btn.get = function () { return on; };
    btn.set = function (v, fire) { on = !!v; paint(); if (fire !== false && typeof opts.onChange === 'function') opts.onChange(on); return btn; };
    return put(btn, opts);
  };

  /* ---- stepper ---- */
  ui.stepper = function (opts) {
    opts = opts || {};
    var steps = arr(opts.steps);
    var root = toEl(opts.el) || h('div');
    root.classList.add('stepper'); if (opts.big) root.classList.add('is-big');
    clear(root);
    var prev = h('button', { type: 'button', class: 'btn btn-quiet btn-prev' }, 'Previous');
    var next = h('button', { type: 'button', class: 'btn btn-primary btn-next' + (opts.big ? ' btn-big' : '') }, opts.nextLabel || 'Next step');
    var reset = h('button', { type: 'button', class: 'btn btn-quiet btn-reset' }, 'Reset');
    var count = h('span', { class: 'stepper-count', 'aria-live': 'polite' });
    var bar = h('div', { class: 'stepper-bar' }, prev, next, reset, count);
    var list = h('ol', { class: 'stepper-list' + (opts.compact ? ' compact' : '') });
    var stage = h('div', { class: 'stepper-stage', 'aria-live': 'polite' });
    var lis = steps.map(function (s, i) {
      var li = h('li', { dataset: { i: i } }, h('span', { class: 'sn' }, String(i + 1)), h('span', { class: 'st' }, s && s.title != null ? String(s.title) : ('Step ' + (i + 1))));
      if (s && s.tex && opts.texInList !== false) { var t = h('span', { class: 'stex' }); AT.tex(t, s.tex); li.appendChild(t); }
      li.addEventListener('click', function () { go(i); });
      li.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { go(i); ev.preventDefault(); } });
      li.tabIndex = 0; li.setAttribute('role', 'button');
      li.style.cursor = 'pointer';
      list.appendChild(li); return li;
    });
    var body = h('div', { class: 'stepper-body' }, opts.hideList ? null : list, stage);
    root.appendChild(bar); root.appendChild(body);
    var idx = -1;
    var api = { el: root, stage: stage, list: list, index: function () { return idx; }, steps: steps };
    function go(i) {
      if (!steps.length) return;
      i = Math.max(0, Math.min(steps.length - 1, i | 0));
      idx = i;
      lis.forEach(function (li, j) { li.classList.toggle('is-current', j === i); li.classList.toggle('is-done', j < i); if (j === i) li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current'); });
      prev.disabled = i === 0; next.disabled = i === steps.length - 1;
      count.textContent = 'Step ' + (i + 1) + ' of ' + steps.length;
      var s = steps[i] || {};
      if (!s.keep) clear(stage);
      if (s.title && opts.stageTitle !== false) stage.appendChild(h('p', { class: 'stage-title' }, String(s.title)));
      if (s.tex && opts.texInStage !== false) { var t = h('div', { class: 'stage-tex' }); AT.tex(t, s.tex, { display: true }); stage.appendChild(t); }
      if (s.note) { var nEl = h('p', { class: 'stage-note', html: String(s.note) }); stage.appendChild(nEl); AT.renderMath(nEl); }
      if (typeof s.render === 'function') { try { s.render(stage, { index: i, step: s, steps: steps, stepper: api }); } catch (e) { console.error('stepper render failed at step ' + i, e); } }
      AT.renderMath(stage);
      if (lis[i] && lis[i].scrollIntoView && opts.scrollList !== false) { try { lis[i].scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (e) { /* ignore */ } }
      if (typeof opts.onChange === 'function') opts.onChange(i, s);
      root.dispatchEvent(new CustomEvent('at-stepchange', { bubbles: true, detail: { index: i } }));
    }
    prev.addEventListener('click', function () { go(idx - 1); });
    next.addEventListener('click', function () { go(idx + 1); });
    reset.addEventListener('click', function () { go(0); });
    root.addEventListener('keydown', function (ev) {
      if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA')) return;
      if (ev.key === 'ArrowRight') { go(idx + 1); ev.preventDefault(); }
      else if (ev.key === 'ArrowLeft') { go(idx - 1); ev.preventDefault(); }
    });
    api.go = go; api.next = function () { go(idx + 1); }; api.prev = function () { go(idx - 1); }; api.reset = function () { go(0); };
    api.buttons = { prev: prev, next: next, reset: reset };
    root.stepperApi = api;
    if (opts.start !== false) go(isNum(opts.start) ? opts.start : 0);
    put(root, opts);
    return api;
  };

  /* ---- reveal ---- */
  ui.reveal = function (questionHTML, answerHTML, opts) {
    opts = opts || {};
    var body = h('div', { class: 'reveal-body' });
    if (answerHTML instanceof Node) body.appendChild(answerHTML);
    else if (Array.isArray(answerHTML)) answerHTML.forEach(function (c) { if (c instanceof Node) body.appendChild(c); else if (c != null) body.insertAdjacentHTML('beforeend', String(c)); });
    else if (answerHTML != null) body.innerHTML = String(answerHTML);
    var det = h('details', { class: 'reveal', open: !!opts.open },
      h('summary', {}, h('span', { class: 'q', html: questionHTML == null ? '' : String(questionHTML) }), h('span', { class: 'hint' }, opts.hint || 'Reveal')),
      body);
    AT.renderMath(det);
    det.addEventListener('toggle', function () { AT.renderMath(det); if (typeof opts.onToggle === 'function') opts.onToggle(det.open); });
    det.body = body;
    return put(det, opts);
  };

  /* ---- callout ---- */
  ui.callout = function (html, opts) {
    opts = opts || {};
    var kind = opts.kind || 'key';
    var el = h('div', { class: 'callout callout-' + kind, html: html == null ? '' : String(html) });
    AT.renderMath(el);
    return put(el, opts);
  };

  /* ---- card ---- */
  ui.card = function (title) {
    var el = h('div', { class: 'card' });
    var opts = null;
    if (title) el.appendChild(h('h3', { class: 'card-title', html: String(title) }));
    for (var i = 1; i < arguments.length; i++) {
      var c = arguments[i];
      if (c && !(c instanceof Node) && typeof c === 'object' && !Array.isArray(c)) { opts = c; continue; }
      if (typeof c === 'string') { var d = h('div', { html: c }); AT.renderMath(d); el.appendChild(d); }
      else append(el, c);
    }
    if (opts && opts.cls) el.classList.add(opts.cls);
    return put(el, opts);
  };

  /* ---- table: spreadsheet-like rows with named coordinate columns ---- */
  /** labelInto(el, s): "$…$" → KaTeX; text containing $ → html + renderMath; otherwise plain text. */
  function labelInto(el, s) {
    s = s == null ? '' : String(s);
    if (s.length > 1 && s.charAt(0) === '$' && s.charAt(s.length - 1) === '$' && s.indexOf('$', 1) === s.length - 1) AT.tex(el, s.slice(1, -1));
    else if (s.indexOf('$') >= 0) { el.innerHTML = s; AT.renderMath(el); }
    else el.textContent = s;
    return el;
  }
  function factor(x, dec) { var s = AT.fmt(x, dec); return s.charAt(0) === '−' ? '(' + s + ')' : s; }
  function isRounded(x, dec) { var k = Math.pow(10, dec == null ? 2 : dec); return isFinite(x) && Math.abs(x - Math.round(x * k) / k) > 1e-10 * Math.max(1, Math.abs(x)); }
  function numericRelation(values, dec) { return values.some(function (v) { return isRounded(v, dec); }) ? ' ≈ ' : ' = '; }
  /** productLine(a, b, dec): "a1×b1 + a2×b2 + … = Σ" for two equal-length vectors */
  function productLine(a, b, dec) {
    a = arr(a); b = arr(b);
    var n = Math.min(a.length, b.length), terms = [];
    for (var i = 0; i < n; i++) terms.push(factor(+a[i] || 0, dec) + '×' + factor(+b[i] || 0, dec));
    var sum = AT.dot(a, b);
    return terms.join(' + ') + numericRelation(a.concat(b, [sum]), dec) + AT.fmt(sum, dec);
  }
  AT.productLine = productLine;
  ui.table = function (rows, opts) {
    opts = opts || {};
    rows = arr(rows);
    var cols = arr(opts.cols).map(function (c) { return (c && typeof c === 'object') ? c : { label: c }; });
    if (!cols.length && opts.axes !== false) {
      /* no cols given: name the columns after the object's axes when the width matches, else 1..n */
      var n0 = rows.length && Array.isArray(rows[0]) ? rows[0].length : 0;
      var rc0 = Array.isArray(opts.rowCls) ? opts.rowCls[0] : opts.rowCls;
      var adef = AT.axesFor(opts.axes != null ? opts.axes : (opts.cls || rc0 || 'neutral'), n0);
      cols = (adef && adef.length === n0 ? adef : AT.axesFor(idxLabels(n0), n0)).map(function (a) { return { label: a.label, title: a.title }; });
    }
    var lead = arr(opts.lead).map(function (c) { return c || {}; }), comp = arr(opts.computed).map(function (c) { return c || {}; });
    var dec = opts.decimals == null ? 2 : opts.decimals;
    var rl = arr(opts.rowLabels);
    var heatCols = arr(opts.heatCols), heatMax = isNum(opts.heatMax) ? opts.heatMax : 1;
    var dims = arr(opts.dimRows);
    var fig = h('figure', { class: 'dt-fig' + (opts.size === 'lg' ? ' lg' : '') + (opts.cls ? ' ' + objClass(opts.cls) : '') });
    var scroll = h('div', { class: 'dt-scroll' });
    var table = h('table', { class: 'dt' + (opts.sticky === false ? '' : ' dt-sticky') });
    var thead = h('thead'), tbody = h('tbody'), tfoot = null;
    function rowCls(i) { var rc = opts.rowCls; if (!rc) return ''; return Array.isArray(rc) ? (rc[i] || '') : String(rc); }
    function val(row, j) {
      if (Array.isArray(row)) return row[j];
      if (row && typeof row === 'object') { var k = cols[j] && cols[j].key != null ? cols[j].key : j; return row[k]; }
      return undefined;
    }
    function fillCell(td, v, d, heat, hmax) {
      td.classList.remove('is-heat', 'is-dark', 'is-masked');
      td.style.removeProperty('--heat');
      td.removeAttribute('title');
      if (v === '×' || v === 'masked') { td.classList.add('is-masked'); td.textContent = '×'; td.title = 'masked (future token)'; return; }
      if (typeof v === 'string') { td.textContent = v; return; }
      if (v instanceof Node) { clear(td); td.appendChild(v); return; }
      td.textContent = AT.fmt(v, d);
      if (heat && isNum(v)) {
        var t = hmax > 0 ? v / hmax : 0;
        td.classList.add('is-heat'); td.style.setProperty('--heat', AT.heatColor(t));
        if (t > 0.55) td.classList.add('is-dark');
      }
    }
    function colTh(def, extra) {
      var t = h('th', { scope: 'col', class: 'dt-ch' + (def.cls ? ' ' + objClass(def.cls) + ' dt-tint' : '') + (extra || ''), title: def.title || null });
      labelInto(t, def.label);
      return t;
    }
    /* header */
    var hr = h('tr');
    hr.appendChild(labelInto(h('th', { scope: 'col', class: 'dt-corner' }), opts.cornerLabel));
    lead.forEach(function (d, k) { hr.appendChild(colTh(d, ' dt-lead' + (k === lead.length - 1 ? ' dt-lead-last' : ''))); });
    cols.forEach(function (d) { hr.appendChild(colTh(d)); });
    comp.forEach(function (d, k) { hr.appendChild(colTh(d, ' dt-comp' + (k === 0 ? ' dt-comp-first' : ''))); });
    thead.appendChild(hr);
    /* body */
    var trEls = [], cellEls = [], leadEls = [], compEls = [], rlEls = [];
    function leadValue(def, row, i) { if (typeof def.fn === 'function') return def.fn(row, i); if (Array.isArray(def.values)) return def.values[i]; return undefined; }
    rows.forEach(function (row, i) {
      var rc = rowCls(i);
      var tr = h('tr', { class: (rc ? objClass(rc) + ' dt-tint' : '') + (i === opts.highlightRow ? ' is-hl' : '') + (dims.indexOf(i) >= 0 ? ' is-dim' : ''), dataset: { r: i } });
      var th = h('th', { scope: 'row', class: 'dt-rl' }); labelInto(th, rl[i]); tr.appendChild(th); rlEls.push(th);
      var lds = lead.map(function (def, k) {
        var td = h('td', { class: 'dt-num dt-lead' + (k === lead.length - 1 ? ' dt-lead-last' : '') + (def.cls ? ' ' + objClass(def.cls) + ' dt-tint' : ''), dataset: { r: i, l: k } });
        fillCell(td, leadValue(def, row, i), def.decimals == null ? dec : def.decimals, !!def.heat, isNum(def.heatMax) ? def.heatMax : heatMax);
        tr.appendChild(td); return td;
      });
      var cs = cols.map(function (def, j) {
        var td = h('td', { class: 'dt-num' + (def.cls ? ' ' + objClass(def.cls) + ' dt-tint' : ''), dataset: { r: i, c: j } });
        fillCell(td, val(row, j), def.decimals == null ? dec : def.decimals, heatCols.indexOf(j) >= 0, heatMax);
        tr.appendChild(td); return td;
      });
      var cps = comp.map(function (def, k) {
        var td = h('td', { class: 'dt-num dt-comp' + (k === 0 ? ' dt-comp-first' : '') + (def.cls ? ' ' + objClass(def.cls) + ' dt-tint' : ''), dataset: { r: i, k: k } });
        var v = typeof def.fn === 'function' ? def.fn(row, i) : (Array.isArray(def.values) ? def.values[i] : undefined);
        fillCell(td, v, def.decimals == null ? dec : def.decimals, !!def.heat, isNum(def.heatMax) ? def.heatMax : heatMax);
        tr.appendChild(td); return td;
      });
      tbody.appendChild(tr);
      trEls.push(tr); cellEls.push(cs); leadEls.push(lds); compEls.push(cps);
    });
    table.appendChild(thead); table.appendChild(tbody);
    /* footer */
    var footEls = { label: null, cells: [], lead: [], computed: [] };
    if (opts.footer) {
      var f = opts.footer;
      tfoot = h('tfoot');
      var ftr = h('tr', { class: 'dt-foot' + (f.cls ? ' ' + objClass(f.cls) + ' dt-tint' : '') });
      var fth = h('th', { scope: 'row', class: 'dt-rl' }); labelInto(fth, f.label); ftr.appendChild(fth); footEls.label = fth;
      var fdec = f.decimals == null ? dec : f.decimals;
      lead.forEach(function (def, k) {
        var td = h('td', { class: 'dt-num dt-lead' + (k === lead.length - 1 ? ' dt-lead-last' : '') });
        fillCell(td, Array.isArray(f.lead) ? f.lead[k] : '', fdec, false, 1); ftr.appendChild(td); footEls.lead.push(td);
      });
      if (typeof f.values === 'string' || f.values == null) {
        var span = h('td', { class: 'dt-text', colspan: String(Math.max(1, cols.length)) }); labelInto(span, f.values); ftr.appendChild(span); footEls.cells.push(span);
      } else {
        cols.forEach(function (def, j) { var td = h('td', { class: 'dt-num' }); fillCell(td, arr(f.values)[j], fdec, false, 1); ftr.appendChild(td); footEls.cells.push(td); });
      }
      comp.forEach(function (def, k) {
        var td = h('td', { class: 'dt-num dt-comp' + (k === 0 ? ' dt-comp-first' : '') });
        fillCell(td, Array.isArray(f.computed) ? f.computed[k] : '', fdec, false, 1); ftr.appendChild(td); footEls.computed.push(td);
      });
      tfoot.appendChild(ftr); table.appendChild(tfoot);
    }
    scroll.appendChild(table); fig.appendChild(scroll);
    var cap = null;
    if (opts.caption) { cap = h('figcaption', { class: 'dt-cap', html: String(opts.caption) }); AT.renderMath(cap); fig.appendChild(cap); }
    var note = null;
    if (opts.note) { note = h('p', { class: 'dt-note', html: String(opts.note) }); AT.renderMath(note); fig.appendChild(note); }
    /* tooltip (one per table, positioned inside the figure so the scroll container cannot clip it) */
    var tip = h('div', { class: 'dt-tip', role: 'status', 'aria-live': 'polite' });
    fig.appendChild(tip);
    fig.showTip = function (td, text) {
      tip.textContent = text == null ? '' : String(text);
      tip.classList.add('is-on');
      placeBelow(fig, td, tip);
      return fig;
    };
    fig.hideTip = function () { tip.classList.remove('is-on'); return fig; };
    /** fig.tipOn(td, textFn): hover/focus shows textFn() in the tooltip; the cell becomes keyboard focusable */
    var tipCells = [];
    fig.refreshTips = function () {
      tipCells.forEach(function (p) { p[0].setAttribute('aria-label', (p[0].textContent || '') + ': ' + (typeof p[1] === 'function' ? p[1]() : p[1])); });
      return fig;
    };
    fig.tipOn = function (td, textFn) {
      if (!td) return fig;
      td.tabIndex = 0; td.classList.add('has-tip');
      var show = function () { fig.showTip(td, typeof textFn === 'function' ? textFn() : textFn); };
      td.addEventListener('mouseenter', show); td.addEventListener('focus', show);
      td.addEventListener('mouseleave', fig.hideTip); td.addEventListener('blur', fig.hideTip);
      tipCells.push([td, textFn]); fig.refreshTips();
      return fig;
    };
    /* api */
    fig.table = table; fig.tip = tip; fig.cap = cap; fig.note = note; fig.cols = cols; fig.colLabels = cols.map(function (c) { return c.label; });
    fig.rowEls = trEls; fig.cells = cellEls; fig.leadCells = leadEls; fig.computedCells = compEls; fig.rowLabelEls = rlEls; fig.footer = footEls;
    fig.update = function (newRows) {
      newRows = arr(newRows);
      rows = newRows;
      newRows.forEach(function (row, i) {
        if (!cellEls[i]) return;
        cols.forEach(function (def, j) { fillCell(cellEls[i][j], val(row, j), def.decimals == null ? dec : def.decimals, heatCols.indexOf(j) >= 0, heatMax); });
        lead.forEach(function (def, k) { fillCell(leadEls[i][k], leadValue(def, row, i), def.decimals == null ? dec : def.decimals, !!def.heat, isNum(def.heatMax) ? def.heatMax : heatMax); });
        comp.forEach(function (def, k) { var v = typeof def.fn === 'function' ? def.fn(row, i) : (Array.isArray(def.values) ? def.values[i] : undefined); fillCell(compEls[i][k], v, def.decimals == null ? dec : def.decimals, !!def.heat, isNum(def.heatMax) ? def.heatMax : heatMax); });
      });
      return fig.refreshTips();
    };
    fig.setFooter = function (values, leadVals, compVals) {
      var fdec = opts.footer && opts.footer.decimals != null ? opts.footer.decimals : dec;
      arr(values).forEach(function (v, j) { if (footEls.cells[j]) fillCell(footEls.cells[j], v, fdec, false, 1); });
      arr(leadVals).forEach(function (v, k) { if (footEls.lead[k]) fillCell(footEls.lead[k], v, fdec, false, 1); });
      arr(compVals).forEach(function (v, k) { if (footEls.computed[k]) fillCell(footEls.computed[k], v, fdec, false, 1); });
      return fig.refreshTips();
    };
    fig.setHighlightRow = function (i) { trEls.forEach(function (tr, j) { tr.classList.toggle('is-hl', j === i); }); return fig; };
    fig.setDim = function (list) { list = arr(list); trEls.forEach(function (tr, j) { tr.classList.toggle('is-dim', list.indexOf(j) >= 0); }); return fig; };
    fig.onRow = function (event, fn) { trEls.forEach(function (tr, i) { tr.addEventListener(event, function (ev) { fn(i, tr, ev); }); }); return fig; };
    fig.onCell = function (event, fn) { cellEls.forEach(function (r, i) { r.forEach(function (td, j) { td.addEventListener(event, function (ev) { fn(i, j, td, ev); }); }); }); return fig; };
    return put(fig, opts);
  };

  /* ---- dotTable: the query row above the key rows, same columns, a score column on the right ---- */
  ui.dotTable = function (q, K, opts) {
    opts = opts || {};
    q = arr(q); K = arr(K).map(arr);
    var dec = opts.decimals == null ? 2 : opts.decimals;
    var n = q.length || (K[0] ? K[0].length : 0);
    var scaled = !!opts.scaled;
    var dk = isNum(opts.d_k) ? opts.d_k : n;
    var sq = Math.sqrt(dk);
    var mask = arr(opts.mask);
    var adefs = opts.cols ? null : (opts.axes !== false ? AT.axesFor(opts.axes != null ? opts.axes : 'q', n) : null);
    var cols = opts.cols ? arr(opts.cols) : (adefs && adefs.length === n ? adefs : AT.range(n).map(function (j) { return String(j + 1); }));
    var colText = cols.map(function (c) { return (c && typeof c === 'object') ? String(c.label == null ? '' : c.label) : String(c == null ? '' : c); });
    var keyLabels = opts.rowLabels ? arr(opts.rowLabels) : K.map(function (_, j) { return '$\\vk{k_{' + (j + 1) + '}}$'; });
    var rowLabels = [opts.queryLabel != null ? opts.queryLabel : '$\\vq{q}$'].concat(keyLabels);
    var raw = K.map(function (k) { return AT.dot(q, k); });
    var scores = raw.map(function (s, j) { return mask.indexOf(j) >= 0 ? -Infinity : (scaled ? s / sq : s); });
    var scoreLabel = opts.scoreLabel != null ? opts.scoreLabel : (scaled ? '$\\text{score}\\,/\\sqrt{d_k}$' : 'score');
    var fig = ui.table([q].concat(K), {
      cols: cols, rowLabels: rowLabels, rowCls: ['q'].concat(K.map(function () { return 'k'; })),
      highlightRow: 0, dimRows: mask.map(function (j) { return j + 1; }), decimals: dec, cornerLabel: opts.cornerLabel, size: opts.size, sticky: opts.sticky, cls: opts.cls,
      computed: [{ label: scoreLabel, cls: 'a', decimals: opts.scoreDecimals == null ? dec : opts.scoreDecimals,
        fn: function (row, i) { if (i === 0) return ''; var j = i - 1; return mask.indexOf(j) >= 0 ? '×' : scores[j]; } }],
      caption: opts.caption,
      note: opts.note !== undefined ? opts.note : 'Hover or focus a score to see the products, column by column.',
      into: opts.into
    });
    var sdec = opts.scoreDecimals == null ? dec : opts.scoreDecimals;
    var pop = popover(fig, { label: 'Score arithmetic' });
    fig.calcPop = pop;
    function plain(lab) { lab = lab == null ? '' : String(lab); return lab.replace(/\$[^$]*\$/g, '').replace(/\s+/g, ' ').trim(); }
    function openCalc(j, td) {
      var qName = plain(rowLabels[0]) || 'q', kName = plain(keyLabels[j]) || ('k ' + (j + 1));
      var calc = ui.dotCalc(q, K[j], { axes: colText, qLabel: 'q · ' + qName, kLabel: 'k · ' + kName, scale: scaled, d_k: dk, decimals: dec, resultDecimals: sdec });
      var extra = mask.indexOf(j) >= 0 ? [h('p', { class: 'calc-line calc-masked' }, 'masked: this key belongs to a later token, so the score is replaced by −∞ and its weight is 0')] : [];
      pop.show(td, 'score of key ' + kName + ' for query ' + qName, [calc].concat(extra));
    }
    K.forEach(function (k, j) {
      var td = fig.computedCells[j + 1][0];
      if (mask.indexOf(j) >= 0) fig.tipOn(td, 'masked: this key belongs to a later token, so its score is set to −∞ and its weight becomes 0');
      else fig.tipOn(td, function () {
        var s = productLine(q, k, dec);
        if (scaled) s += ', then ' + AT.fmt(raw[j], dec) + ' ÷ √' + dk + (isRounded(raw[j], dec) || isRounded(scores[j], sdec) ? ' ≈ ' : ' = ') + AT.fmt(scores[j], sdec);
        return s + ' (click for the worksheet)';
      });
      td.classList.add('has-calc');
      td.addEventListener('click', function () { openCalc(j, td); });
      td.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openCalc(j, td); } });
    });
    fig.scores = scores; fig.raw = raw; fig.scoreCells = K.map(function (_, j) { return fig.computedCells[j + 1][0]; });
    fig.queryRow = fig.rowEls[0]; fig.keyRows = fig.rowEls.slice(1);
    fig.openCalc = function (j) { var td = fig.scoreCells[j]; if (td) openCalc(j, td); return fig; };
    return fig;
  };

  /* ---- mixTable: value rows with a left α column and a weighted-sum footer ---- */
  ui.mixTable = function (alpha, V, opts) {
    opts = opts || {};
    alpha = arr(alpha); V = arr(V).map(arr);
    var dec = opts.decimals == null ? 2 : opts.decimals;
    var weighted = !!opts.weighted;
    var d = V[0] ? V[0].length : 0;
    var vdefs = opts.cols ? null : (opts.axes !== false ? AT.axesFor(opts.axes != null ? opts.axes : 'v', d) : null);
    var cols = opts.cols ? arr(opts.cols) : (vdefs && vdefs.length === d ? vdefs : AT.range(d).map(function (j) { return String(j + 1); }));
    var rows = weighted ? V.map(function (v, j) { return AT.scale(v, +alpha[j] || 0); }) : V;
    var sum = AT.zeros(d);
    V.forEach(function (v, j) { sum = AT.add(sum, AT.scale(v, +alpha[j] || 0)); });
    var rowLabels = opts.rowLabels ? arr(opts.rowLabels) : V.map(function (_, j) { return weighted ? '$\\va{\\alpha_{' + (j + 1) + '}}\\vv{v_{' + (j + 1) + '}}$' : '$\\vv{v_{' + (j + 1) + '}}$'; });
    var dimRows = opts.dimRows ? arr(opts.dimRows) : (opts.dimZero === false ? [] : V.map(function (_, j) { return j; }).filter(function (j) { return !(+alpha[j] > 0.0005); }));
    var fig = ui.table(rows, {
      cols: cols, rowLabels: rowLabels, rowCls: 'v', decimals: dec, dimRows: dimRows, highlightRow: opts.highlightRow,
      cornerLabel: opts.cornerLabel, size: opts.size, sticky: opts.sticky, cls: opts.cls,
      lead: [{ label: opts.alphaLabel != null ? opts.alphaLabel : '$\\va{\\alpha_j}$', cls: 'a', heat: opts.heat !== false, heatMax: 1, fn: function (_, j) { return +alpha[j] || 0; }, decimals: opts.alphaDecimals == null ? dec : opts.alphaDecimals }],
      footer: opts.footer === false ? null : { label: opts.footerLabel != null ? opts.footerLabel : '$\\sum_j \\va{\\alpha_j}\\,\\vv{v_j}$', values: sum, cls: opts.footerCls || 'm', lead: [AT.sum(alpha)], decimals: dec },
      caption: opts.caption,
      note: opts.note !== undefined ? opts.note : 'Hover or focus a sum to see the weighted terms, row by row.',
      into: opts.into
    });
    if (opts.footer !== false) cols.forEach(function (_, c) {
      var td = fig.footer.cells[c];
      fig.tipOn(td, function () {
        var terms = [];
        var values = [sum[c]], adec = opts.alphaDecimals == null ? dec : opts.alphaDecimals, roundedAlpha = false;
        V.forEach(function (v, j) { if (!alpha[j]) return; terms.push(factor(+alpha[j] || 0, adec) + '×' + factor(+v[c] || 0, dec)); values.push(+v[c] || 0); roundedAlpha = roundedAlpha || isRounded(+alpha[j] || 0, adec); });
        return (terms.length ? terms.join(' + ') : '0') + (roundedAlpha ? ' ≈ ' : numericRelation(values, dec)) + AT.fmt(sum[c], dec);
      });
    });
    fig.sum = sum; fig.alpha = alpha;
    /* setAlpha stores the new weights (a copy, so later edits of the caller's array do not leak in until the next call) and refills
       the α column, the weighted cells, the dims and the footer from them; passing the creation array mutated in place also works */
    fig.setAlpha = function (newAlpha) {
      alpha = arr(newAlpha).slice();
      sum = AT.zeros(d); V.forEach(function (v, j) { sum = AT.add(sum, AT.scale(v, +alpha[j] || 0)); });
      fig.sum = sum; fig.alpha = alpha;
      fig.update(weighted ? V.map(function (v, j) { return AT.scale(v, +alpha[j] || 0); }) : V);
      if (opts.dimZero !== false && !opts.dimRows) { dimRows = V.map(function (_, j) { return j; }).filter(function (j) { return !(+alpha[j] > 0.0005); }); fig.setDim(dimRows); }
      if (opts.footer !== false) fig.setFooter(sum, [AT.sum(alpha)]);
      return fig;
    };
    return fig;
  };

  /* ---- wTable: a projection matrix as a table, rows = input axes, columns = output axes, zeros greyed ---- */
  var AXIS_SYM = { e: '\\ve{e}', q: '\\vq{q}', k: '\\vk{k}', v: '\\vv{v}', vocab: '\\text{word}' };
  function axisCls(kind, cls) { return kind === 'e' ? 'e' : kind === 'qk' ? (cls === 'k' ? 'k' : 'q') : kind === 'v' ? 'v' : 'neutral'; }
  function axisDefs(kind, n, short) { var a = AT.axesFor(kind, n, short); return (a && a.length === n) ? a : AT.axesFor(idxLabels(n), n); }
  /** wTable(W, { from: 'e'|'v'|'qk', to: 'qk'|'v'|'e'|'vocab', cls, decimals: 1, cornerLabel, caption, note, size, sticky, into }) */
  ui.wTable = function (W, opts) {
    opts = opts || {};
    W = arr(W).map(arr);
    var m = W.length, n = m ? Math.max.apply(null, W.map(function (r) { return r.length; })) : 0;
    var from = opts.from || 'e', to = opts.to || 'qk';
    var cls = opts.cls || axisCls(to, null), fromCls = axisCls(from, null);
    var inAx = axisDefs(from, m, opts.shortAxes), outAx = axisDefs(to, n, opts.shortAxes);
    var dec = opts.decimals == null ? 1 : opts.decimals;
    var corner = opts.cornerLabel != null ? opts.cornerLabel : '$' + (AXIS_SYM[fromCls] || '') + ' \\rightarrow ' + (to === 'vocab' ? AXIS_SYM.vocab : AXIS_SYM[cls] || '') + '$';
    var fig = ui.table(W, {
      cols: outAx.map(function (a) { return { label: a.label, title: a.title, cls: cls }; }),
      rowLabels: inAx.map(function (a) { return a.label; }), cornerLabel: corner, decimals: dec,
      caption: opts.caption, note: opts.note, size: opts.size, sticky: opts.sticky, cls: opts.figCls, into: opts.into
    });
    fig.classList.add('dt-w');
    var readings = [];
    fig.rowLabelEls.forEach(function (th, i) { th.classList.add(objClass(fromCls), 'dt-tint'); if (inAx[i].title) th.title = inAx[i].title; });
    fig.cells.forEach(function (r, i) {
      r.forEach(function (td, j) {
        var v = +W[i][j] || 0;
        var fi = inAx[i].title || inAx[i].label, fo = outAx[j].title || outAx[j].label;
        if (v === 0) { td.classList.add('is-zero'); td.title = fi + ' → ' + fo + ': 0 (no effect)'; }
        else { td.title = fi + ' → ' + fo + ': ' + AT.fmt(v, dec); readings.push(inAx[i].label + ' → ' + outAx[j].label + ' ' + AT.fmt(v, dec)); }
      });
    });
    fig.readings = readings; fig.inAxes = inAx; fig.outAxes = outAx;
    return fig;
  };

  /* ---- dotCalc: the two vectors being dot-producted, one column per axis, products underneath, then the sum (and the scaling) ---- */
  /** dotCalc(q, k, { axes: 'qk' | 'e' | 'v' | [labels] | false, qLabel: 'q', kLabel: 'k', scale: true, d_k, decimals: 2, resultDecimals, into }) */
  ui.dotCalc = function (q, k, opts) {
    opts = opts || {};
    q = arr(q); k = arr(k);
    var n = Math.max(q.length, k.length);
    var dec = opts.decimals == null ? 2 : opts.decimals, rdec = opts.resultDecimals == null ? dec : opts.resultDecimals;
    var scale = opts.scale !== false, dk = isNum(opts.d_k) ? opts.d_k : (n || AT.d_k);
    var ax = opts.axes === false ? AT.axesFor(idxLabels(n), n) : axisDefs(opts.axes == null ? 'q' : opts.axes, n, opts.shortAxes);
    var root = h('div', { class: 'calc calc-dot' });
    var scroll = h('div', { class: 'calc-scroll' }), table = h('table', { class: 'calc-t' });
    var thead = h('thead'), hr = h('tr', {}, h('th', { class: 'calc-corner' }));
    ax.forEach(function (a) { hr.appendChild(h('th', { class: 'calc-ax', scope: 'col', title: a.title || null }, a.label)); });
    thead.appendChild(hr); table.appendChild(thead);
    var tbody = h('tbody');
    function row(cls, labelText) {
      var tr = h('tr', { class: objClass(cls) });
      var th = h('th', { scope: 'row', class: 'calc-rl' }, labelText);
      tr.appendChild(th);
      var tds = [];
      for (var i = 0; i < n; i++) { var td = h('td', { class: cls === 'neutral' ? '' : 'tint' }); tr.appendChild(td); tds.push(td); }
      tbody.appendChild(tr);
      return { tr: tr, th: th, tds: tds };
    }
    var qr = row('q', opts.qLabel != null ? String(opts.qLabel) : 'q'), kr = row('k', opts.kLabel != null ? String(opts.kLabel) : 'k'), pr = row('neutral', 'q × k');
    table.appendChild(tbody); scroll.appendChild(table); root.appendChild(scroll);
    var sumLine = h('p', { class: 'calc-line calc-sum' }), scaleLine = h('p', { class: 'calc-line calc-scale obj-a' });
    root.appendChild(sumLine); if (scale) root.appendChild(scaleLine);
    function fill() {
      var prods = [], dot = 0;
      for (var i = 0; i < n; i++) {
        var a = +q[i] || 0, b = +k[i] || 0, pv = a * b; dot += pv; prods.push(pv);
        qr.tds[i].textContent = AT.fmt(a, dec); kr.tds[i].textContent = AT.fmt(b, dec); pr.tds[i].textContent = AT.fmt(pv, dec);
        pr.tds[i].classList.toggle('calc-zero', pv === 0);
      }
      clear(sumLine);
      append(sumLine, ['sum = ' + prods.map(function (x) { return factor(x, dec); }).join(' + ') + (prods.some(function (x) { return isRounded(x, dec); }) || isRounded(dot, scale ? dec : rdec) ? ' ≈ ' : ' = '), h('span', { class: 'calc-r' }, AT.fmt(dot, scale ? dec : rdec))]);
      if (scale) { clear(scaleLine); append(scaleLine, [AT.fmt(dot, dec) + ' / √' + dk + (isRounded(dot, dec) || isRounded(dot / Math.sqrt(dk), rdec) ? ' ≈ ' : ' = '), h('span', { class: 'calc-r' }, AT.fmt(dot / Math.sqrt(dk), rdec))]); }
      root.dot = dot; root.score = scale ? dot / Math.sqrt(dk) : dot;
    }
    root.appendChild(h('p', { class: 'calc-rounding small muted' }, 'Displayed operands and products are rounded; results use the unrounded values.'));
    fill();
    root.update = function (nq, nk, o) {
      o = o || {};
      if (nq != null) q = arr(nq); if (nk != null) k = arr(nk);
      if (o.qLabel != null) qr.th.textContent = String(o.qLabel); if (o.kLabel != null) kr.th.textContent = String(o.kLabel);
      fill(); return root;
    };
    root.rows = { q: qr, k: kr, prod: pr }; root.sumLine = sumLine; root.scaleLine = scale ? scaleLine : null; root.axes = ax.map(function (a) { return a.label; });
    return put(root, opts);
  };

  /* ---- matVecCalc: x W as a worksheet, one line per output coordinate, zero weights of the sparse W greyed ---- */
  /** matVecCalc(x, W, { from: 'e', to: 'qk'|'v'|'e', xLabel, outLabel, cls, decimals: 2, wDecimals, into }) */
  ui.matVecCalc = function (x, W, opts) {
    opts = opts || {};
    x = arr(x); W = arr(W).map(arr);
    var nIn = x.length, nOut = W[0] ? W[0].length : 0;
    var from = opts.from || 'e', to = opts.to || 'qk';
    var cls = opts.cls || (to === 'e' ? 'd' : axisCls(to, null)), fromCls = opts.fromCls || axisCls(from, null);
    var dec = opts.decimals == null ? 2 : opts.decimals, wdec = opts.wDecimals == null ? 1 : opts.wDecimals;
    var inAx = axisDefs(from, nIn, opts.shortAxes), outAx = axisDefs(to, nOut, opts.shortAxes);
    var root = h('div', { class: 'calc calc-mv ' + objClass(cls) });
    var scroll = h('div', { class: 'calc-scroll' }), table = h('table', { class: 'calc-t' });
    var hr = h('tr', {}, h('th', { class: 'calc-corner' }));
    inAx.forEach(function (a) { hr.appendChild(h('th', { class: 'calc-ax', scope: 'col', title: a.title || null }, a.label)); });
    hr.appendChild(h('th', { class: 'calc-ax calc-res-h', scope: 'col' }, 'result'));
    table.appendChild(h('thead', {}, hr));
    var tbody = h('tbody');
    var xtr = h('tr', { class: objClass(fromCls) + ' calc-x' }), xth = h('th', { scope: 'row', class: 'calc-rl' }, opts.xLabel != null ? String(opts.xLabel) : 'x');
    xtr.appendChild(xth);
    var xtds = [];
    for (var c = 0; c < nIn; c++) { var xtd = h('td', { class: 'tint' }); xtr.appendChild(xtd); xtds.push(xtd); }
    xtr.appendChild(h('td', { class: 'calc-blank' }));
    tbody.appendChild(xtr);
    var lines = [];
    for (var o = 0; o < nOut; o++) {
      var tr = h('tr', { class: objClass(cls) + ' calc-out' });
      tr.appendChild(h('th', { scope: 'row', class: 'calc-rl', title: outAx[o].title || null }, outAx[o].label));
      var terms = [];
      for (var cc = 0; cc < nIn; cc++) { var td = h('td', { class: 'calc-term' }); tr.appendChild(td); terms.push(td); }
      var res = h('td', { class: 'tint calc-res' }); tr.appendChild(res);
      tbody.appendChild(tr); lines.push({ tr: tr, terms: terms, res: res });
    }
    table.appendChild(tbody); scroll.appendChild(table); root.appendChild(scroll);
    var outRow = h('div', { class: 'calc-outrow' });
    root.appendChild(outRow);
    var outVec = ui.vec([], { cls: cls, axes: outAx.map(function (a) { return a; }), decimals: dec });
    var outLab = h('span', { class: 'calc-rl calc-outlab ' + objClass(cls) }, opts.outLabel != null ? String(opts.outLabel) : 'x W');
    outRow.appendChild(outLab); outRow.appendChild(outVec);
    function fill() {
      var out = [];
      for (var c = 0; c < nIn; c++) xtds[c].textContent = AT.fmt(+x[c] || 0, dec);
      lines.forEach(function (ln, o) {
        var sum = 0, rounded = false;
        ln.terms.forEach(function (td, c) {
          var xv = +x[c] || 0, wv = +arr(W[c])[o] || 0; sum += xv * wv;
          rounded = rounded || isRounded(xv, dec) || isRounded(wv, wdec);
          clear(td);
          if (c > 0) td.appendChild(h('span', { class: 'calc-plus' }, '+ '));
          td.appendChild(document.createTextNode(factor(xv, dec) + '×' + factor(wv, wdec)));
          td.classList.toggle('calc-zero', wv === 0);
          td.title = (inAx[c].title || inAx[c].label) + ' → ' + (outAx[o].title || outAx[o].label) + ': ' + AT.fmt(xv, dec) + ' × ' + AT.fmt(wv, wdec);
        });
        ln.res.textContent = (rounded || isRounded(sum, dec) ? '≈ ' : '= ') + AT.fmt(sum, dec); out.push(sum);
      });
      outVec.update(out); root.out = out;
    }
    root.appendChild(h('p', { class: 'calc-rounding small muted' }, 'Displayed operands are rounded; results use the unrounded values.'));
    fill();
    root.update = function (nx, o) { o = o || {}; if (nx != null) x = arr(nx); if (o.xLabel != null) xth.textContent = String(o.xLabel); if (o.outLabel != null) outLab.textContent = String(o.outLabel); fill(); return root; };
    root.lines = lines; root.xCells = xtds; root.outVec = outVec;
    return put(root, opts);
  };

  /* ---- notationCard: symbol / meaning / shape, in three groups ---- */
  var NOTATION = [
    { g: 'token', sym: '\\ve{e_i^{(0)}}', mean: 'starting representation of token $i$: token embedding plus position', shape: '1\\times d_{\\text{model}}', dims: function () { return '1×' + AT.d_model; } },
    { g: 'token', sym: '\\ve{e_i}', mean: 'current representation of token $i$', shape: '1\\times d_{\\text{model}}', dims: function () { return '1×' + AT.d_model; } },
    { g: 'token', sym: '\\vq{q_i} = \\ve{e_i} W_Q', mean: 'query: what token $i$ is looking for', shape: '1\\times d_k', dims: function () { return '1×' + AT.d_k; } },
    { g: 'token', sym: '\\vk{k_j} = \\ve{e_j} W_K', mean: 'key: when token $j$ should be read', shape: '1\\times d_k', dims: function () { return '1×' + AT.d_k; } },
    { g: 'token', sym: '\\vv{v_j} = \\ve{e_j} W_V', mean: 'value: what token $j$ sends if it is read', shape: '1\\times d_v', dims: function () { return '1×' + AT.d_v; } },
    { g: 'token', sym: 's_{ij} = \\vq{q_i}\\cdot\\vk{k_j}/\\sqrt{d_k}', mean: 'scaled score of key $j$ for query $i$', shape: '\\text{scalar}', dims: function () { return ''; } },
    { g: 'token', sym: '\\va{\\alpha_{ij}} = \\operatorname{softmax}_j(s_{ij})', mean: 'attention weight: how much $i$ reads from $j$; each row sums to 1', shape: '\\text{scalar}', dims: function () { return ''; } },
    { g: 'token', sym: 'm_i = \\sum_j \\va{\\alpha_{ij}}\\,\\vv{v_j}', mean: 'retrieved message: the weighted mixture of values', shape: '1\\times d_v', dims: function () { return '1×' + AT.d_v; } },
    { g: 'token', sym: '\\vd{\\Delta e_i} = \\big(\\sum_j \\va{\\alpha_{ij}}\\,\\vv{v_j}\\big) W_O = m_i W_O', mean: 'contextual update (this page\u2019s name for the attention output)', shape: '1\\times d_{\\text{model}}', dims: function () { return '1×' + AT.d_model; } },
    { g: 'token', sym: "\\vp{e_i'} = \\ve{e_i} + \\vd{\\Delta e_i}", mean: 'updated representation (the residual addition)', shape: '1\\times d_{\\text{model}}', dims: function () { return '1×' + AT.d_model; } },
    { g: 'matrix', sym: '\\ve{E}', mean: 'all current representations, one row per token', shape: 'T\\times d_{\\text{model}}', dims: function () { return AT.T + '×' + AT.d_model; } },
    { g: 'matrix', sym: '\\vq{Q} = \\ve{E} W_Q,\\; \\vk{K} = \\ve{E} W_K', mean: 'all queries and all keys', shape: 'T\\times d_k', dims: function () { return AT.T + '×' + AT.d_k; } },
    { g: 'matrix', sym: '\\vv{V} = \\ve{E} W_V', mean: 'all values', shape: 'T\\times d_v', dims: function () { return AT.T + '×' + AT.d_v; } },
    { g: 'matrix', sym: 'S = \\vq{Q}\\vk{K}^{\\top}/\\sqrt{d_k}', mean: 'all scaled scores', shape: 'T\\times T', dims: function () { return AT.T + '×' + AT.T; } },
    { g: 'matrix', sym: 'M', mean: 'causal mask: $0$ where $j \\le i$, $-\\infty$ where $j > i$', shape: 'T\\times T', dims: function () { return AT.T + '×' + AT.T; } },
    { g: 'matrix', sym: '\\va{A} = \\operatorname{softmax}(S + M)', mean: 'all attention weights, row by row', shape: 'T\\times T', dims: function () { return AT.T + '×' + AT.T; } },
    { g: 'matrix', sym: 'H = \\va{A}\\vv{V}', mean: 'all retrieved messages', shape: 'T\\times d_v', dims: function () { return AT.T + '×' + AT.d_v; } },
    { g: 'matrix', sym: '\\vd{\\Delta E} = H W_O', mean: 'all contextual updates', shape: 'T\\times d_{\\text{model}}', dims: function () { return AT.T + '×' + AT.d_model; } },
    { g: 'matrix', sym: "\\vp{E'} = \\ve{E} + \\vd{\\Delta E}", mean: 'all updated representations', shape: 'T\\times d_{\\text{model}}', dims: function () { return AT.T + '×' + AT.d_model; } },
    { g: 'sizes', sym: 'T', mean: 'number of tokens in the sequence', shape: '', dims: function () { return String(AT.T); } },
    { g: 'sizes', sym: 'd_{\\text{model}}', mean: 'width of a token representation', shape: '', dims: function () { return String(AT.d_model); } },
    { g: 'sizes', sym: 'd_k,\\; d_v', mean: 'width of queries and keys; width of values', shape: '', dims: function () { return AT.d_k + ', ' + AT.d_v; } },
    { g: 'sizes', sym: 'W_Q,\\; W_K', mean: 'query and key projections (learned)', shape: 'd_{\\text{model}}\\times d_k', dims: function () { return AT.d_model + '×' + AT.d_k; } },
    { g: 'sizes', sym: 'W_V', mean: 'value projection (learned)', shape: 'd_{\\text{model}}\\times d_v', dims: function () { return AT.d_model + '×' + AT.d_v; } },
    { g: 'sizes', sym: 'W_O', mean: 'output projection: message space back to representation space', shape: 'd_v\\times d_{\\text{model}}', dims: function () { return AT.d_v + '×' + AT.d_model; } },
    { g: 'sizes', sym: '\\ve{E_{\\text{tok}}}', mean: 'learned token lookup table; distinct from the current sequence stack $\\ve{E}$', shape: '|\\mathcal V|\\times d_{\\text{model}}', dims: function () { return (AT.vocab.length || 20) + '×' + AT.d_model; } },
    { g: 'sizes', sym: 'W_{\\text{vocab}}', mean: 'output-head weights: $\\ell = \\vp{e_t\'}\\,W_{\\text{vocab}} + b$, then softmax', shape: 'd_{\\text{model}}\\times |\\mathcal V|', dims: function () { return AT.d_model + '×' + (AT.vocab.length || 20); } },
    { g: 'sizes', sym: 'b', mean: 'one learned bias per vocabulary logit', shape: '1\\times |\\mathcal V|', dims: function () { return '1×' + (AT.vocab.length || 20); } }
  ];
  NOTATION.forEach(function (n) { n.parts = ['part2', 'part3']; });
  /* Part 1 (the character model) and Part 3 (learning, heads, the block) rows; groups 'mlp', 'train', 'block' */
  var NOTATION_1 = [
    { g: 'mlp', sym: 't_i', mean: 'token id of character $i$ (an index into the vocabulary)', shape: '\\text{integer}', dims: function () { return ''; } },
    { g: 'mlp', sym: '\\mathcal V', mean: 'the vocabulary: the boundary token and the letters', shape: '|\\mathcal V|', dims: function () { return String(AT.vocab.length || ''); } },
    { g: 'mlp', sym: 'w', mean: 'the window: how many previous characters the model sees', shape: '', dims: function () { return isNum(model.w) ? String(model.w) : ''; } },
    { g: 'mlp', sym: '\\ve{E_{\\text{tok}}}', mean: 'the learned lookup table: one row per vocabulary token', shape: '|\\mathcal V| \\times d', dims: function () { return (AT.vocab.length || '?') + '×' + AT.d_model; } },
    { g: 'mlp', sym: '\\ve{e_i} = \\ve{E_{\\text{tok}}}[t_i]', mean: 'the embedding at position $i$: the token\'s row of the table', shape: '1 \\times d', dims: function () { return '1×' + AT.d_model; } },
    { g: 'mlp', sym: 'a_0 = [\\ve{e_1}, \\ldots, \\ve{e_w}]', mean: 'the concatenated window (the input of the network)', shape: '1 \\times wd', dims: function () { return isNum(model.w) ? '1×' + (model.w * AT.d_model) : ''; } },
    { g: 'mlp', sym: 'a_1 = \\sigma(a_0 W_1 + b_1)', mean: 'the hidden layer', shape: '1 \\times d_h', dims: function () { return isNum(model.d_h) ? '1×' + model.d_h : ''; } },
    { g: 'mlp', sym: 'z = a_1 W_2 + b_2', mean: 'the logits: one score per vocabulary entry', shape: '1 \\times |\\mathcal V|', dims: function () { return '1×' + (AT.vocab.length || '?'); } },
    { g: 'mlp', sym: 'p = \\operatorname{softmax}(z)', mean: 'the next-character probabilities', shape: '1 \\times |\\mathcal V|', dims: function () { return '1×' + (AT.vocab.length || '?'); } },
    { g: 'mlp', sym: '\\text{loss} = -\\log p(\\text{target})', mean: 'cross-entropy on the observed next character', shape: '\\text{scalar}', dims: function () { return ''; } },
    { g: 'sizes', sym: 'd', mean: 'width of an embedding row', shape: '', dims: function () { return String(AT.d_model); } },
    { g: 'sizes', sym: 'd_h', mean: 'width of the hidden layer', shape: '', dims: function () { return isNum(model.d_h) ? String(model.d_h) : ''; } },
    { g: 'sizes', sym: 'W_1', mean: 'first-layer weights (learned)', shape: 'wd \\times d_h', dims: function () { return (isNum(model.w) && isNum(model.d_h)) ? (model.w * AT.d_model) + '×' + model.d_h : ''; } },
    { g: 'sizes', sym: 'b_1', mean: 'one learned bias per hidden unit', shape: '1 \\times d_h', dims: function () { return isNum(model.d_h) ? '1×' + model.d_h : ''; } },
    { g: 'sizes', sym: 'W_2', mean: 'output-layer weights (learned)', shape: 'd_h \\times |\\mathcal V|', dims: function () { return isNum(model.d_h) ? model.d_h + '×' + (AT.vocab.length || '?') : ''; } },
    { g: 'sizes', sym: 'b_2', mean: 'one learned bias per vocabulary logit', shape: '1 \\times |\\mathcal V|', dims: function () { return '1×' + (AT.vocab.length || '?'); } }
  ];
  NOTATION_1.forEach(function (n) { n.parts = ['part1']; });
  var NOTATION_3 = [
    { g: 'train', sym: '\\theta', mean: 'all the parameters together', shape: '', dims: function () { return ''; } },
    { g: 'train', sym: 'L', mean: 'the loss: $-\\log p(\\text{target})$, averaged over positions when training in parallel', shape: '\\text{scalar}', dims: function () { return ''; } },
    { g: 'train', sym: '\\partial L / \\partial \\theta', mean: 'the gradient: how the loss changes with each parameter', shape: '\\text{same shape as } \\theta', dims: function () { return ''; } },
    { g: 'train', sym: '\\eta', mean: 'the learning rate: the step size of $\\theta \\leftarrow \\theta - \\eta\\, \\partial L / \\partial \\theta$', shape: '\\text{scalar}', dims: function () { return ''; } },
    { g: 'block', sym: 'n_h', mean: 'number of heads, each with its own $W_Q, W_K, W_V$', shape: '', dims: function () { return ''; } },
    { g: 'block', sym: '\\operatorname{LN}(\\ve{e})', mean: 'layer normalisation of a row', shape: '1 \\times d_{\\text{model}}', dims: function () { return '1×' + AT.d_model; } },
    { g: 'block', sym: 'd_{\\text{ff}}', mean: 'width of the feed-forward hidden layer', shape: '', dims: function () { return ''; } },
    { g: 'block', sym: 'L\\ \\text{(blocks)}', mean: 'number of blocks stacked', shape: '', dims: function () { return ''; } },
    { g: 'block', sym: 'T^2', mean: 'the cost of the score matrix: every token against every token', shape: '', dims: function () { return AT.T + '² = ' + (AT.T * AT.T); } }
  ];
  NOTATION_3.forEach(function (n) { n.parts = ['part3']; });
  AT.notation = NOTATION.concat(NOTATION_1, NOTATION_3);
  AT.T = arr(AT.sentences.river).length || 10;
  var GROUP_TITLES = { token: 'One token at a time', matrix: 'All tokens at once', sizes: 'Sizes and learned weights', axes: 'Named coordinates (illustrative)', mlp: 'The character model, step by step', train: 'Learning', block: 'The Transformer block' };
  var PART_GROUPS = { part1: ['mlp', 'sizes', 'axes'], part2: ['token', 'matrix', 'sizes', 'axes'], part3: ['token', 'matrix', 'train', 'block', 'sizes', 'axes'] };
  /* the "axes" group is built at call time from AT.axes so the names are never retyped */
  function axesNotation() {
    if (!AT.axes.named) return [];
    var q = function (list) { return list.map(function (s) { return '\u201c' + AT.escape(s) + '\u201d'; }).join(', '); };
    var hasPosition = AT.axes.e.some(function (s) { return String(s).toLowerCase() === 'position'; });
    return [
      { g: 'axes', sym: '\\ve{e}', mean: 'coordinates ' + q(AT.axes.e) + (hasPosition ? ': meaning coordinates plus a separate position coordinate; this toy carries position but does not use it in attention' : ': coordinates of the current token representation'), shape: '1\\times d_{\\text{model}}', dims: function () { return '1×' + AT.d_model; } },
      { g: 'axes', sym: '\\vq{q},\\; \\vk{k}', mean: 'coordinates ' + q(AT.axes.qk) + ': a query row reads \u201cwhat I ask for\u201d, a key row \u201cwhat I offer\u201d', shape: '1\\times d_k', dims: function () { return '1×' + AT.d_k; } },
      { g: 'axes', sym: '\\vv{v}', mean: 'coordinates ' + q(AT.axes.v) + ': what the token sends if it is read; values have their own width and $W_O$ maps them back onto the $\\ve{e}$ coordinates', shape: '1\\times d_v', dims: function () { return '1×' + AT.d_v; } }
    ];
  }
  ui.notationCard = function (opts) {
    opts = opts || {};
    var part = opts.part || (window.__PART__ && window.__PART__.notation) || 'part2';
    var groups = opts.groups ? arr(opts.groups) : (PART_GROUPS[part] || PART_GROUPS.part2).filter(function (g) { return g !== 'axes' || AT.axes.named; });
    var root = h('div', { class: 'notation-card' + (opts.compact ? ' compact' : '') });
    groups.forEach(function (g) {
      var items = (g === 'axes' ? axesNotation() : AT.notation).filter(function (n) { return n.g === g && (g === 'axes' || !n.parts || n.parts.indexOf(part) >= 0) && (!opts.only || arr(opts.only).indexOf(n.sym) >= 0); });
      if (!items.length) return;
      var box = h('div', { class: 'notation-group' });
      box.appendChild(h('p', { class: 'notation-title' }, GROUP_TITLES[g] || g));
      var scroll = h('div', { class: 'dt-scroll' });
      var table = h('table', { class: 'dt dt-notation' });
      var thead = h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'symbol'), h('th', { scope: 'col' }, 'meaning'), h('th', { scope: 'col' }, g === 'sizes' ? 'value' : 'shape')));
      var tbody = h('tbody');
      items.forEach(function (n) {
        var ts = h('th', { scope: 'row', class: 'nt-sym' }); AT.tex(ts, n.sym);
        var tm = h('td', { class: 'nt-mean', html: n.mean }); AT.renderMath(tm);
        var tsh = h('td', { class: 'nt-shape' });
        var d = n.dims();
        if (n.shape) { var sh = h('span', { class: 'nt-shape-tex' }); AT.tex(sh, n.shape); tsh.appendChild(sh); if (d) tsh.appendChild(h('span', { class: 'nt-dims' }, '= ' + d)); }
        else if (d) tsh.appendChild(h('span', { class: 'nt-dims' }, d));
        tbody.appendChild(h('tr', {}, ts, tm, tsh));
      });
      table.appendChild(thead); table.appendChild(tbody); scroll.appendChild(table); box.appendChild(scroll);
      if (g === 'axes') box.appendChild(h('p', { class: 'notation-note' }, 'The coordinates were given illustrative names, and the toy model was written by hand so that the names are true: every row of a weight matrix reads as \u201cthis input axis feeds that output axis\u201d.'));
      root.appendChild(box);
    });
    return put(root, opts);
  };

  /* ---- legend ---- */
  var OBJECTS = [
    { cls: 'e', sym: 'e', name: 'current representation', def: 'e_i: the current representation of token i.', tip: 'e_i is the current representation of token i. At layer 0 it is token embedding + position; after attention it is contextual.' },
    { cls: 'q', sym: 'Q', name: 'query', def: 'q_i = e_i W_Q: what token i is looking for.', tip: 'q_i = e_i W_Q is the query: what token i is looking for.' },
    { cls: 'k', sym: 'K', name: 'key', def: 'k_j = e_j W_K: when token j should be retrieved.', tip: 'k_j = e_j W_K is the key: when token j should be retrieved.' },
    { cls: 'v', sym: 'V', name: 'value', def: 'v_j = e_j W_V: what token j sends if retrieved.', tip: 'v_j = e_j W_V is the value: what token j sends if it is retrieved.' },
    { cls: 'a', sym: 'α', name: 'attention weight', def: 'α_ij: how much token i reads from token j.', tip: 'α_ij is the attention weight: how much token i reads from token j. It is the softmax of the scaled scores, and each row sums to 1.' },
    { cls: 'd', sym: 'Δe', name: 'contextual update', def: 'Δe_i: the update produced by attention (a pedagogical name).', tip: 'Δe_i = (Σ_j α_ij v_j) W_O = m_i W_O is the contextual update, using row vectors. Pedagogical name for the attention output.' },
    { cls: 'ep', sym: 'e+Δe', name: 'updated representation', def: 'e_i′ = e_i + Δe_i: the residual addition is standard.', tip: 'e_i′ = e_i + Δe_i is the updated representation. The addition is the standard residual connection.' }
  ];
  AT.objects = OBJECTS;
  ui.legend = function (opts) {
    opts = opts || {};
    var ul = h('ul', { class: 'legend', 'aria-label': 'Colour legend' });
    OBJECTS.forEach(function (o) {
      if (opts.only && arr(opts.only).indexOf(o.cls) < 0) return;
      ul.appendChild(h('li', { class: 'obj-' + o.cls, title: o.def }, h('span', { class: 'sw' }), h('span', { class: 'sy' }, o.sym), o.name));
    });
    return put(ul, opts);
  };

  /* ---- motif: e → (Q,K,V) → attention → Δe → ⊕ → e' ---- */
  var SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { if (attrs[k] != null) el.setAttribute(k, String(attrs[k])); });
    for (var i = 2; i < arguments.length; i++) { var c = arguments[i]; if (c == null) continue; if (c instanceof Node) el.appendChild(c); else el.appendChild(document.createTextNode(String(c))); }
    return el;
  }
  AT.svg = sv;
  var motifCount = 0;
  AT.motif = function (el, opts) {
    el = toEl(el); opts = opts || {};
    var box = el || h('div');
    box.classList.add('motif'); if (opts.size === 'sm') box.classList.add('sm');
    clear(box);
    var id = 'at-motif-' + (++motifCount);
    var labels = opts.labels !== false;
    var W = 780, H = labels ? 172 : 150, cy = 78;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Pipeline: e to Q, K, V to attention to delta e, added back to give e prime' });
    var defs = sv('defs');
    function marker(mid, color) {
      var m = sv('marker', { id: mid, viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse' });
      m.appendChild(sv('path', { d: 'M0 0L10 5L0 10z', fill: color }));
      return m;
    }
    defs.appendChild(marker(id + '-ink', '#8A91A0'));
    defs.appendChild(marker(id + '-e', '#2563EB'));
    svg.appendChild(defs);
    var stages = {};
    function stage(name, cls) { var g = sv('g', { 'data-stage': name, class: 'stage-' + cls }); stages[name] = g; svg.appendChild(g); return g; }
    function edge(d, res) { var p = sv('path', { d: d, class: 'edge' + (res ? ' edge-res' : ''), 'marker-end': 'url(#' + id + (res ? '-e' : '-ink') + ')' }); return p; }
    function rect(g, x, y, w, hh, extra) { g.appendChild(sv('rect', Object.assign({ x: x, y: y, width: w, height: hh, class: 'box' }, extra || {}))); }
    function text(g, x, y, s, cls, anchor) { g.appendChild(sv('text', { x: x, y: y, class: cls || 'st', 'text-anchor': anchor || 'middle', 'dominant-baseline': 'middle' }, s)); }
    function sub(g, x, y, s) { if (labels) g.appendChild(sv('text', { x: x, y: y, class: 'sub', 'text-anchor': 'middle' }, s)); }
    // geometry
    var eX = 12, eW = 66, bh = 44;
    var pX = 150, pW = 56, ph = 28, pys = [cy - 46, cy - 14, cy + 18];
    var aX = 290, aW = 118;
    var dX = 478, dW = 74;
    var oX = 618, oR = 17;
    var xX = 690, xW = 78;
    // edges (drawn first so boxes sit on top)
    var edges = sv('g', { class: 'edges' });
    pys.forEach(function (py) { edges.appendChild(edge('M' + (eX + eW) + ' ' + cy + ' C ' + (eX + eW + 36) + ' ' + cy + ', ' + (pX - 36) + ' ' + (py + ph / 2) + ', ' + (pX - 2) + ' ' + (py + ph / 2))); });
    pys.forEach(function (py) { edges.appendChild(edge('M' + (pX + pW) + ' ' + (py + ph / 2) + ' C ' + (pX + pW + 40) + ' ' + (py + ph / 2) + ', ' + (aX - 40) + ' ' + cy + ', ' + (aX - 2) + ' ' + cy)); });
    edges.appendChild(edge('M' + (aX + aW) + ' ' + cy + ' L ' + (dX - 2) + ' ' + cy));
    edges.appendChild(edge('M' + (dX + dW) + ' ' + cy + ' L ' + (oX - oR - 2) + ' ' + cy));
    edges.appendChild(edge('M' + (oX + oR) + ' ' + cy + ' L ' + (xX - 2) + ' ' + cy));
    // residual: from e up over the top to the plus circle
    edges.appendChild(edge('M' + (eX + eW / 2) + ' ' + (cy - bh / 2) + ' L ' + (eX + eW / 2) + ' 16 L ' + oX + ' 16 L ' + oX + ' ' + (cy - oR - 2), true));
    if (labels) edges.appendChild(sv('text', { x: (eX + eW / 2 + oX) / 2, y: 11, class: 'sub', 'text-anchor': 'middle', fill: '#2563EB' }, 'residual: keep e, add Δe'));
    svg.appendChild(edges);
    // stages
    var gE = stage('e', 'e'); rect(gE, eX, cy - bh / 2, eW, bh); text(gE, eX + eW / 2, cy, 'e'); sub(gE, eX + eW / 2, cy + bh / 2 + 18, 'current');
    sub(gE, eX + eW / 2, cy + bh / 2 + 31, 'representation');
    var gP = stage('qkv', 'qkv');
    rect(gP, pX, pys[0], pW, ph, { class: 'box box-q' }); text(gP, pX + pW / 2, pys[0] + ph / 2, 'Q', 'st st-q');
    rect(gP, pX, pys[1], pW, ph, { class: 'box box-k' }); text(gP, pX + pW / 2, pys[1] + ph / 2, 'K', 'st st-k');
    rect(gP, pX, pys[2], pW, ph, { class: 'box box-v' }); text(gP, pX + pW / 2, pys[2] + ph / 2, 'V', 'st st-v');
    sub(gP, pX + pW / 2, pys[2] + ph + 18, 'three projections');
    sub(gP, pX + pW / 2, pys[2] + ph + 31, 'of e');
    var gA = stage('att', 'att'); rect(gA, aX, cy - bh / 2, aW, bh); text(gA, aX + aW / 2 - 10, cy, 'attention', 'st ui'); text(gA, aX + aW / 2 + 40, cy, 'α', 'st');
    sub(gA, aX + aW / 2, cy + bh / 2 + 18, 'Q·K decide where to read,');
    sub(gA, aX + aW / 2, cy + bh / 2 + 31, 'V carries the message');
    var gD = stage('delta', 'delta'); rect(gD, dX, cy - bh / 2, dW, bh); text(gD, dX + dW / 2, cy, 'Δe'); sub(gD, dX + dW / 2, cy + bh / 2 + 18, 'contextual');
    sub(gD, dX + dW / 2, cy + bh / 2 + 31, 'update');
    var gO = stage('add', 'add'); gO.appendChild(sv('circle', { cx: oX, cy: cy, r: oR, class: 'box' })); text(gO, oX, cy + 1, '+'); sub(gO, oX, cy + oR + 18, 'add');
    var gX = stage('ep', 'ep'); rect(gX, xX, cy - bh / 2, xW, bh); text(gX, xX + xW / 2, cy, 'e + Δe'); sub(gX, xX + xW / 2, cy + bh / 2 + 18, 'updated');
    sub(gX, xX + xW / 2, cy + bh / 2 + 31, 'representation');
    box.appendChild(svg);
    var api = {
      el: box, svg: svg, stages: stages,
      setActive: function (name) {
        var any = false;
        Object.keys(stages).forEach(function (k) { var on = k === name; stages[k].classList.toggle('is-active', on); if (on) any = true; });
        box.classList.toggle('has-active', any);
        return api;
      }
    };
    api.setActive(opts.active || null);
    return api;
  };

  /* ---- flow: dots travelling from source elements to a target ---- */
  ui.flow = function (container, opts) {
    opts = opts || {};
    container = toEl(container);
    var from = arr(opts.from).map(toEl).filter(Boolean), to = toEl(opts.to);
    if (!container || !to || !from.length) return Promise.resolve();
    if (reducedMotion() || typeof container.animate !== 'function' && typeof Element.prototype.animate !== 'function') {
      if (typeof opts.onDone === 'function') opts.onDone();
      return Promise.resolve();
    }
    var cs = getComputedStyle(container);
    if (cs.position === 'static') container.style.position = 'relative';
    var cr = container.getBoundingClientRect(), tr = to.getBoundingClientRect();
    var tx = tr.left - cr.left + tr.width / 2, ty = tr.top - cr.top + tr.height / 2;
    var weights = arr(opts.weights), color = opts.color || 'var(--c-v)', dur = isNum(opts.duration) ? opts.duration : 900, stagger = isNum(opts.stagger) ? opts.stagger : 70;
    var proms = from.map(function (f, i) {
      var w = isNum(weights[i]) ? Math.max(0, Math.min(1, weights[i])) : 0.6;
      if (opts.skipZero !== false && isNum(weights[i]) && weights[i] <= 0.001) return Promise.resolve();
      var size = Math.round(7 + 15 * w);
      var fr = f.getBoundingClientRect();
      var sx = fr.left - cr.left + fr.width / 2, sy = fr.top - cr.top + fr.height / 2;
      var dot = h('div', { class: 'flow-dot', style: { width: size + 'px', height: size + 'px', background: color, opacity: String(0.35 + 0.65 * w), marginLeft: (-size / 2) + 'px', marginTop: (-size / 2) + 'px' } });
      container.appendChild(dot);
      var anim = dot.animate([
        { transform: 'translate(' + sx + 'px,' + sy + 'px) scale(.6)', opacity: 0 },
        { transform: 'translate(' + sx + 'px,' + sy + 'px) scale(1)', opacity: 0.35 + 0.65 * w, offset: 0.12 },
        { transform: 'translate(' + tx + 'px,' + ty + 'px) scale(1)', opacity: 0.35 + 0.65 * w, offset: 0.9 },
        { transform: 'translate(' + tx + 'px,' + ty + 'px) scale(.3)', opacity: 0 }
      ], { duration: dur, delay: i * stagger, easing: 'cubic-bezier(.4,0,.2,1)', fill: 'forwards' });
      return new Promise(function (res) { anim.onfinish = function () { if (dot.parentNode) dot.parentNode.removeChild(dot); res(); }; anim.oncancel = anim.onfinish; });
    });
    var all = Promise.all(proms).then(function () { if (typeof opts.onDone === 'function') opts.onDone(); });
    return all;
  };

  /* ---- an arrow glyph (for “→” between components in a .row) ---- */
  ui.op = function (text, opts) { return put(h('span', { class: 'mat-op', 'aria-hidden': 'true' }, text == null ? '→' : String(text)), opts); };

  /* ======================================================================
     8. netSketch: a node-network sketch (input column e_1..e_w, hidden column, output column = vocabulary)
     ====================================================================== */
  AT.netSketch = function (el, opts) {
    el = toEl(el); opts = opts || {};
    var box = el || h('div');
    box.classList.add('netsk');
    clear(box);
    var hidden = Math.max(1, opts.hidden == null ? 5 : (opts.hidden | 0));
    var outputs = arr(opts.outputs).map(function (s) { return String(s); });
    if (!outputs.length) outputs = AT.vocab.length ? AT.vocab.slice() : ['a', 'b', 'c'];
    var collapseAbove = isNum(opts.collapseAbove) ? opts.collapseAbove : 8;
    var hl = opts.highlightOutput == null ? null : String(opts.highlightOutput);
    var labels = opts.labels || {};
    var inputLabel = typeof labels.input === 'function' ? labels.input : function (i) { return '\\ve{e_{' + i + '}}'; };
    var outputLabel = typeof labels.output === 'function' ? labels.output : function (s) { return s; };
    var w = Math.max(1, (opts.inputs | 0) || 3);
    // geometry (constant, so the column can grow without the drawing jumping)
    var SP = 34, R = 11, X = [92, 290, 470], W = 560;
    var outRows = outputRows();
    var maxRows = Math.max(Math.min(collapseAbove, 12), hidden, outRows.length, 6);
    var H = maxRows * SP + 64, CY = H / 2 + 12;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'A network that reads ' + w + ' input rows through a hidden layer and scores every vocabulary entry' });
    var gEdges = sv('g', { class: 'edges' }), gIn = sv('g', { class: 'col col-in' }), gHid = sv('g', { class: 'col col-hid' }), gOut = sv('g', { class: 'col col-out' }), gCap = sv('g', { class: 'caps' });
    svg.appendChild(gEdges); svg.appendChild(gIn); svg.appendChild(gHid); svg.appendChild(gOut); svg.appendChild(gCap);
    box.appendChild(svg);
    function ys(n) { var o = []; for (var i = 0; i < n; i++) o.push(CY + (i - (n - 1) / 2) * SP); return o; }
    function inputRows() {
      if (w <= collapseAbove) return AT.range(w).map(function (i) { return { key: 'i' + (i + 1), i: i + 1 }; });
      return [{ key: 'i1', i: 1 }, { key: 'i2', i: 2 }, { key: 'i3', i: 3 }, { key: 'ell', ell: true }, { key: 'i' + (w - 1), i: w - 1 }, { key: 'i' + w, i: w }];
    }
    function outputRows() {
      var n = outputs.length;
      var idx = AT.range(n);
      if (n > collapseAbove) {
        var hi = hl == null ? -1 : outputs.indexOf(hl);
        idx = [0, 1, 2];
        if (hi > 2 && hi < n - 2) idx.push(-1, hi); else idx.push(-1);
        idx.push(n - 2, n - 1);
        if (hi >= 0 && hi <= 2) { /* already shown */ }
      }
      return idx.map(function (i) { return i < 0 ? { key: 'ell', ell: true } : { key: 'o' + i, i: i, label: outputs[i] }; });
    }
    function node(cls, key) {
      var g = sv('g', { class: 'node ' + cls, 'data-key': key });
      g.appendChild(sv('circle', { cx: 0, cy: 0, r: R }));
      return g;
    }
    function ellipsis(cls) {
      var g = sv('g', { class: 'node ell ' + cls });
      g.appendChild(sv('text', { x: 0, y: 0, 'text-anchor': 'middle', 'dominant-baseline': 'middle' }, '⋮'));
      return g;
    }
    function place(g, x, y) { g.setAttribute('transform', 'translate(' + x + ' ' + y + ')'); }
    function labelIn(g, i) {
      var fo = sv('foreignObject', { x: -R - 74, y: -14, width: 68, height: 28, class: 'lab-in' });
      var d = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
      d.setAttribute('class', 'lab-in-box');
      AT.tex(d, inputLabel(i));
      fo.appendChild(d); g.appendChild(fo);
    }
    // hidden column (fixed)
    var hy = ys(hidden);
    AT.range(hidden).forEach(function (j) { var g = node('hid', 'h' + j); place(g, X[1], hy[j]); gHid.appendChild(g); });
    // output column (fixed)
    var oy = ys(outRows.length), outNodes = [];
    outRows.forEach(function (r, k) {
      var g = r.ell ? ellipsis('out') : node('out', r.key);
      if (!r.ell) { g.appendChild(sv('text', { x: R + 9, y: 0, class: 'out-l', 'dominant-baseline': 'middle' }, outputLabel(r.label))); if (hl != null && r.label === hl) g.classList.add('is-hl'); g.setAttribute('data-label', r.label); }
      place(g, X[2], oy[k]); gOut.appendChild(g); if (!r.ell) outNodes.push({ y: oy[k], g: g });
    });
    // captions
    var capIn = sv('text', { x: X[0], y: 22, class: 'cap', 'text-anchor': 'middle' }, '');
    gCap.appendChild(capIn);
    gCap.appendChild(sv('text', { x: X[1], y: 22, class: 'cap', 'text-anchor': 'middle' }, opts.hiddenCaption || 'hidden'));
    gCap.appendChild(sv('text', { x: X[2], y: 22, class: 'cap', 'text-anchor': 'middle' }, opts.outputCaption || 'next token'));
    var inNodes = {};
    function drawEdges(rows, iy) {
      var g = sv('g', { class: 'edges-set' });
      rows.forEach(function (r, k) { if (r.ell) return; hy.forEach(function (y) { g.appendChild(sv('line', { x1: X[0] + R, y1: iy[k], x2: X[1] - R, y2: y })); }); });
      hy.forEach(function (y) { outNodes.forEach(function (o) { var l = sv('line', { x1: X[1] + R, y1: y, x2: X[2] - R, y2: o.y }); if (o.g.classList.contains('is-hl')) l.setAttribute('class', 'hl'); g.appendChild(l); }); });
      return g;
    }
    var edgeSet = null;
    function render(animate) {
      var rows = inputRows(), iy = ys(rows.length);
      var keep = {};
      rows.forEach(function (r, k) {
        keep[r.key] = true;
        var g = inNodes[r.key];
        var fresh = !g;
        if (fresh) {
          g = r.ell ? ellipsis('e') : node('e', r.key);
          if (!r.ell) labelIn(g, r.i);
          inNodes[r.key] = g; gIn.appendChild(g);
          if (animate) { g.style.opacity = '0'; place(g, X[0], iy[k]); void g.getBoundingClientRect(); g.style.opacity = '1'; }
        }
        place(g, X[0], iy[k]);
      });
      Object.keys(inNodes).forEach(function (k) {
        if (keep[k]) return;
        var g = inNodes[k]; delete inNodes[k];
        if (animate && !reducedMotion()) { g.style.opacity = '0'; setTimeout(function () { if (g.parentNode) g.parentNode.removeChild(g); }, 260); }
        else if (g.parentNode) g.parentNode.removeChild(g);
      });
      var ne = drawEdges(rows, iy);
      if (edgeSet) {
        var old = edgeSet;
        if (animate && !reducedMotion()) { old.style.opacity = '0'; setTimeout(function () { if (old.parentNode) old.parentNode.removeChild(old); }, 260); }
        else if (old.parentNode) old.parentNode.removeChild(old);
      }
      if (animate) { ne.style.opacity = '0'; gEdges.appendChild(ne); void ne.getBoundingClientRect(); ne.style.opacity = '1'; } else gEdges.appendChild(ne);
      edgeSet = ne;
      capIn.textContent = (typeof opts.inputCaption === 'function' ? opts.inputCaption(w) : (w + (w === 1 ? ' input' : ' inputs')));
      svg.setAttribute('aria-label', 'A network that reads ' + w + ' input rows through ' + hidden + ' hidden nodes and scores ' + outputs.length + ' vocabulary entries');
    }
    render(false);
    var api = {
      el: box, svg: svg,
      inputs: function () { return w; },
      setInputs: function (n) { n = Math.max(1, n | 0); if (n === w) return api; w = n; render(true); return api; },
      setHighlight: function (label) {
        hl = label == null ? null : String(label);
        Array.prototype.slice.call(gOut.querySelectorAll('.node.out')).forEach(function (g) { g.classList.toggle('is-hl', hl != null && g.getAttribute('data-label') === hl); });
        Array.prototype.slice.call(gEdges.querySelectorAll('line')).forEach(function (l) { l.removeAttribute('class'); });
        outNodes.forEach(function (o) { if (!o.g.classList.contains('is-hl')) return; Array.prototype.slice.call(gEdges.querySelectorAll('line')).forEach(function (l) { if (+l.getAttribute('y2') === o.y && +l.getAttribute('x2') === X[2] - R) l.setAttribute('class', 'hl'); }); });
        return api;
      }
    };
    return api;
  };

  /* ======================================================================
     9. present mode: frames, builds, keys, hash, overview, blank, notes, presenter window, print
     ====================================================================== */
  AT.present = (function () {
    var P = { active: false, discovered: false, frames: [], fi: -1, build: 0, listeners: {}, blank: false, overview: false, notes: false, help: false, presWin: null, isPresenter: false, printState: null, startedAt: null, fit: null, preflightPromise: null };
    var U = {}, pendingState = new WeakMap(), overviewFocus = null, overviewBackground = [], entryFocus = null, fitRaf = 0, svgPaintRaf = 0, svgPaintFrame = null;
    var UNIT_SEL = '.card, .callout, .tex-display, .prose, p:not(.companion), .chips, table, .stepper, .reveal, .dt-fig, figure, .motif, .netsk, .row, .stack, .btn-row, .scroll-x, ul, ol, h3, blockquote';
    var SPLIT_SEL = '.side-by-side, .grid-2, .grid-3';
    function emit(ev, data) { arr(P.listeners[ev]).forEach(function (f) { try { f(data); } catch (e) { console.error('present listener failed', e); } }); }
    function on(ev, fn) { if (typeof fn === 'function') (P.listeners[ev] = P.listeners[ev] || []).push(fn); }
    function children(el) { return Array.prototype.slice.call(el.children); }
    function all(el, sel) { return Array.prototype.slice.call(el.querySelectorAll(sel)); }

    /* ---- notes: <script type="text/x-notes"> (blank line = paragraph) or data-notes ---- */
    function notesOf(frame) {
      var sc = frame.querySelector('script[type="text/x-notes"]');
      var txt = sc ? sc.textContent : (frame.getAttribute('data-notes') || '');
      txt = txt.replace(/\r/g, '').replace(/^\s*\n/, '').replace(/\s+$/, '');
      if (!txt.trim()) return [];
      var lines = txt.split('\n'), ind = null;
      lines.forEach(function (l) { if (l.trim()) { var m = l.match(/^\s*/)[0].length; ind = ind == null ? m : Math.min(ind, m); } });
      lines = lines.map(function (l) { return l.slice(ind || 0); });
      /* the first line is the question to ask before the reveal: it is always its own paragraph */
      var first = lines.shift().trim();
      var rest = lines.join('\n').split(/\n\s*\n/).map(function (p) { return p.replace(/\s*\n\s*/g, ' ').trim(); }).filter(Boolean);
      return (first ? [first] : []).concat(rest);
    }

    /* ---- builds ---- */
    function buildOf(el) {
      var n = 0;
      for (var p = el; p && !p.classList.contains('frame'); p = p.parentElement) n = Math.max(n, parseInt(p.getAttribute('data-build'), 10) || 0);
      return n;
    }
    function prepBuilds(f) {
      if (f.querySelector('[data-build]') || f.getAttribute('data-autobuild') === 'off') return;
      var n = 0;
      children(f).forEach(function (c) {
        if (c.tagName === 'SCRIPT' || c.tagName === 'STYLE' || c.classList.contains('companion') || c.classList.contains('sec-head')) return;
        var units = c.matches(SPLIT_SEL) ? children(c) : (c.matches(UNIT_SEL) ? [c] : []);
        units.forEach(function (u) { u.setAttribute('data-build', String(n)); u.setAttribute('data-build-auto', ''); n++; });
      });
    }
    function maxBuild(f) { var m = 0; all(f, '[data-build]').forEach(function (e) { m = Math.max(m, parseInt(e.getAttribute('data-build'), 10) || 0); }); return m; }
    function pending(el, on) {
      if (on && !pendingState.has(el)) pendingState.set(el, { inert: el.inert, hidden: el.getAttribute('aria-hidden') });
      el.classList.toggle('is-pending', on);
      if (on) { el.inert = true; el.setAttribute('aria-hidden', 'true'); }
      else if (pendingState.has(el)) {
        var saved = pendingState.get(el); el.inert = saved.inert;
        if (saved.hidden == null) el.removeAttribute('aria-hidden'); else el.setAttribute('aria-hidden', saved.hidden);
        pendingState.delete(el);
      }
    }
    function applyBuild(fr, b, edge) {
      all(fr.el, '[data-build]').forEach(function (e) {
        var pend = buildOf(e) > b;
        if (pend && e.contains(document.activeElement) && U.controls) U.controls.focus({ preventScroll: true });
        pending(e, pend);
      });
      steppersOf(fr).forEach(function (s) {
        var nb = buildOf(s.el);
        if (nb > b || (nb === b && edge === 'start')) { if (s.api.index() !== 0) s.api.go(0); }
        else if (nb < b || (nb === b && edge === 'end')) { var last = s.api.steps.length - 1; if (s.api.index() !== last) s.api.go(last); }
      });
    }
    function steppersOf(fr) { return all(fr.el, '.stepper').filter(function (e) { return e.stepperApi && e.stepperApi.steps.length && !e.closest('[data-present="manual"]'); }).map(function (e) { return { el: e, api: e.stepperApi }; }); }
    function activeStepper(fr, b, direction) {
      var list = steppersOf(fr).filter(function (s) { return buildOf(s.el) === b && !s.el.closest('[hidden]'); });
      if (direction < 0) list.reverse();
      var found = list.filter(function (s) { return direction < 0 ? s.api.index() > 0 : s.api.index() < s.api.steps.length - 1; })[0];
      return (found || list[0] || {}).api || null;
    }

    /* ---- per-frame control state: snapshot on first entry, restore on leave (unless data-keep-state) ---- */
    function snapshot(el) {
      var s = { ranges: [], toggles: [], details: [] };
      all(el, 'input[type=range]').forEach(function (r) { if (!r.closest('[data-keep-state], [data-present="manual"]')) s.ranges.push([r, r.value]); });
      all(el, '.toggle[aria-pressed]').forEach(function (t) { if (!t.closest('[data-keep-state], [data-present="manual"]')) s.toggles.push([t, t.getAttribute('aria-pressed')]); });
      all(el, 'details').forEach(function (d) { if (!d.closest('[data-keep-state], [data-present="manual"]')) s.details.push([d, d.open]); });
      return s;
    }
    function restore(s) {
      if (!s) return;
      s.ranges.forEach(function (p) { if (p[0].value !== p[1]) { p[0].value = p[1]; p[0].dispatchEvent(new Event('input', { bubbles: true })); p[0].dispatchEvent(new Event('change', { bubbles: true })); } });
      s.toggles.forEach(function (p) { if (p[0].getAttribute('aria-pressed') !== p[1]) { if (typeof p[0].set === 'function') p[0].set(p[1] === 'true'); else p[0].click(); } });
      s.details.forEach(function (p) { if (p[0].open !== p[1]) p[0].open = p[1]; });
    }

    /* ---- discovery: every section becomes one or more frames ---- */
    function discover() {
      if (P.discovered) return;
      P.discovered = true;
      P.frames = [];
      var secs = Array.prototype.slice.call(document.querySelectorAll('main .sec'));
      secs.forEach(function (sec, si) {
        var head = sec.querySelector(':scope > .sec-head');
        var frames = all(sec, '.frame').filter(function (f) { return f.parentNode === sec || !f.parentNode.closest('.frame'); });
        if (!frames.length) {
          var f = h('div', { class: 'frame frame-auto' });
          var kids = children(sec).filter(function (c) { return c !== head && c.tagName !== 'STYLE' && c.tagName !== 'SCRIPT'; });
          if (head && head.nextSibling) sec.insertBefore(f, head.nextSibling); else sec.appendChild(f);
          kids.forEach(function (k) { f.appendChild(k); });
          frames = [f];
        }
        sec.classList.add('has-frames');
        if (head && !head.querySelector('.frame-sub')) head.appendChild(h('h3', { class: 'frame-sub' }));
        var title = sec.getAttribute('data-title') || (sec.querySelector('h2') ? sec.querySelector('h2').textContent : sec.id);
        frames.forEach(function (f, k) {
          prepBuilds(f);
          if (!f.hasAttribute('tabindex')) f.setAttribute('tabindex', '-1');
          P.frames.push({ sec: sec, secIndex: si, id: sec.id, num: (sec.id || '').replace(/^s/, ''), secTitle: title, el: f, index: k, count: frames.length, title: f.getAttribute('data-title') || '', notes: notesOf(f), maxBuild: maxBuild(f), snapshot: null });
        });
      });
    }

    /* ---- chrome (counter, notes strip, overview, blank, help) ---- */
    function ensureUi() {
      if (U.counter) return;
      U.counter = h('div', { id: 'at-counter', role: 'status', 'aria-live': 'polite', 'aria-atomic': 'true' });
      U.prev = h('button', { id: 'at-prev', type: 'button', class: 'pbtn', on: { click: prev } }, 'Previous');
      U.next = h('button', { id: 'at-next', type: 'button', class: 'pbtn', on: { click: next } }, 'Next');
      U.overviewButton = h('button', { id: 'at-overview-btn', type: 'button', class: 'pbtn', 'aria-haspopup': 'dialog', on: { click: function () { setOverview(true); } } }, 'Frames');
      U.fullscreen = h('button', { id: 'at-fullscreen', type: 'button', class: 'pbtn', 'aria-pressed': 'false', on: { click: fullscreen } }, 'Full screen');
      U.controls = h('nav', { id: 'at-controls', tabindex: '-1', 'aria-label': 'Classroom navigation' }, U.counter, U.prev, U.next, U.overviewButton, U.fullscreen,
        h('button', { id: 'at-exit', type: 'button', class: 'pbtn', on: { click: exit } }, 'Exit'));
      U.announcement = h('p', { id: 'at-announcement', role: 'status', 'aria-live': 'polite', 'aria-atomic': 'true' });
      U.fitWarning = h('p', { id: 'at-fit-warning', role: 'status', 'aria-live': 'polite', 'aria-atomic': 'true' });
      U.notes = h('div', { id: 'at-notes', role: 'complementary', 'aria-label': 'Presenter notes' });
      U.overview = h('div', { id: 'at-overview', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'All frames' }, h('div', { class: 'ov-head' }, h('p', { class: 'ov-title' }, 'All frames: select one to jump'), h('button', { id: 'at-overview-close', type: 'button', class: 'pbtn', on: { click: function () { setOverview(false); } } }, 'Close overview')), h('div', { class: 'ov-grid' }));
      U.blank = h('div', { id: 'at-blank', 'aria-hidden': 'true' });
      U.help = h('div', { id: 'at-help', role: 'note' }, h('div', { html: '<kbd>→</kbd> <kbd>Space</kbd> <kbd>PgDn</kbd> next build · <kbd>←</kbd> <kbd>PgUp</kbd> <kbd>Backspace</kbd> back · <kbd>Home</kbd> <kbd>End</kbd> first / last frame<br><kbd>O</kbd> overview · <kbd>B</kbd> blank · <kbd>S</kbd> notes · <kbd>?</kbd> this help · <kbd>Esc</kbd> exit' }));
      U.blank.addEventListener('click', function () { setBlank(false); });
      document.body.appendChild(U.controls); document.body.appendChild(U.announcement); document.body.appendChild(U.fitWarning); document.body.appendChild(U.notes); document.body.appendChild(U.overview); document.body.appendChild(U.blank); document.body.appendChild(U.help);
      window.addEventListener('resize', measureChrome);
      document.addEventListener('fullscreenchange', function () { updateChrome(); measureChrome(); });
      if (window.ResizeObserver) { U.resize = new ResizeObserver(measureChrome); U.resize.observe(U.controls); var strip = document.getElementById('strip'); if (strip) U.resize.observe(strip); }
      document.addEventListener('input', scheduleFitCheck, true);
      document.addEventListener('change', scheduleFitCheck, true);
      document.addEventListener('click', scheduleFitCheck, true);
      document.addEventListener('toggle', scheduleFitCheck, true);
      document.addEventListener('load', scheduleFitCheck, true);
    }
    function measureChrome() {
      if (!P.active) return;
      var strip = document.getElementById('strip');
      document.body.style.setProperty('--present-strip-h', (strip ? strip.getBoundingClientRect().height : 0) + 'px');
      document.body.style.setProperty('--present-controls-h', (U.controls ? U.controls.getBoundingClientRect().height : 0) + 'px');
      var main = document.querySelector('main');
      if (main) {
        var r = main.getBoundingClientRect(), cs = getComputedStyle(document.body);
        var sw = parseFloat(cs.getPropertyValue('--present-stage-w')) || 1280;
        var sh = parseFloat(cs.getPropertyValue('--present-stage-h')) || 720;
        var gutter = parseFloat(cs.getPropertyValue('--present-gutter')) || 0;
        var scale = Math.min(Math.max(1, r.width - gutter * 2) / sw, Math.max(1, r.height - gutter * 2) / sh);
        document.body.style.setProperty('--present-scale', String(Math.max(0.05, scale)));
      }
      repaintSvgs(cur());
      scheduleFitCheck();
    }
    /* Chromium can retain SVG text at coordinates from the previously hidden
       layout after a fixed stage is scaled. Reinsert each live SVG into paint
       without replacing the node, listeners, or component state. */
    function repaintSvgs(fr) {
      if (!fr || !fr.el) return;
      svgPaintFrame = fr;
      if (svgPaintRaf) cancelAnimationFrame(svgPaintRaf);
      svgPaintRaf = requestAnimationFrame(function () {
        svgPaintRaf = 0;
        var active = svgPaintFrame; svgPaintFrame = null;
        if (!active || !active.el.classList.contains('is-live')) return;
        var svgs = all(active.el, 'svg');
        var displays = svgs.map(function (svg) { return svg.style.display; });
        svgs.forEach(function (svg) { svg.style.display = 'none'; });
        if (svgs[0]) svgs[0].getBoundingClientRect();
        svgs.forEach(function (svg, i) { svg.style.display = displays[i]; });
        if (svgs[0]) svgs[0].getBoundingClientRect();
      });
    }
    function frameHeading(fr) {
      if (!fr || !fr.sec) return;
      var head = fr.sec.querySelector(':scope > .sec-head');
      var sub = head && head.querySelector('.frame-sub');
      var title = (fr.title || '').trim();
      var distinct = title && title !== fr.secTitle;
      if (head) head.classList.toggle('has-frame-title', !!distinct);
      if (sub) sub.textContent = distinct ? title : '';
    }
    function measureFit(fr) {
      if (!fr || !fr.el || !fr.el.classList.contains('is-live')) return null;
      var el = fr.el, stage = fr.sec;
      var frameX = Math.max(0, el.scrollWidth - el.clientWidth);
      var frameY = Math.max(0, el.scrollHeight - el.clientHeight);
      var stageX = Math.max(0, stage.scrollWidth - stage.clientWidth);
      var stageY = Math.max(0, stage.scrollHeight - stage.clientHeight);
      var overX = Math.max(frameX, stageX);
      var overY = Math.max(frameY, stageY);
      var overflow = overX > 2 || overY > 2;
      var report = {
        overflow: overflow,
        horizontal: Math.round(overX),
        vertical: Math.round(overY),
        section: fr.id,
        frame: fr.index + 1,
        title: fr.title || fr.secTitle,
        width: el.clientWidth,
        height: el.clientHeight,
        contentWidth: el.scrollWidth,
        contentHeight: el.scrollHeight,
        frameHorizontal: Math.round(frameX),
        frameVertical: Math.round(frameY),
        stageHorizontal: Math.round(stageX),
        stageVertical: Math.round(stageY)
      };
      el.setAttribute('data-overflow', overflow ? 'true' : 'false');
      return report;
    }
    function reportFit(fr) {
      if (!P.active || document.body.classList.contains('is-preflighting')) return null;
      var report = measureFit(fr || cur());
      P.fit = report;
      if (U.fitWarning) {
        if (report && report.overflow) {
          var extra = [];
          if (report.vertical) extra.push(report.vertical + 'px too tall');
          if (report.horizontal) extra.push(report.horizontal + 'px too wide');
          U.fitWarning.textContent = 'Authoring check — frame ' + report.section.replace(/^s/, '') + '.' + report.frame + ' is ' + extra.join(' and ') + '. Move content to a continuation frame; presentation frames never scroll.';
        } else U.fitWarning.textContent = '';
      }
      if (report) emit('fit', report);
      return report;
    }
    function scheduleFitCheck() {
      if (!P.active || document.body.classList.contains('is-preflighting')) return;
      if (fitRaf) cancelAnimationFrame(fitRaf);
      fitRaf = requestAnimationFrame(function () {
        fitRaf = requestAnimationFrame(function () { fitRaf = 0; reportFit(); });
      });
    }
    function nextPaint() {
      return new Promise(function (resolve) { requestAnimationFrame(function () { requestAnimationFrame(resolve); }); });
    }
    function worseFit(a, b) {
      if (!a) return b;
      if (!b) return a;
      return b.horizontal + b.vertical > a.horizontal + a.vertical ? b : a;
    }
    async function samplePreflightState(fr, worst) {
      repaintSvgs(fr); await nextPaint();
      worst = worseFit(worst, measureFit(fr));
      var details = all(fr.el, 'details');
      if (!details.length) return worst;
      var states = details.map(function (e) { return [e, e.open]; });
      details.forEach(function (e) { e.open = true; });
      await nextPaint();
      worst = worseFit(worst, measureFit(fr));
      states.forEach(function (s) { s[0].open = s[1]; });
      await nextPaint();
      return worst;
    }
    /* Authoring preflight: reveal each frame at its fullest state and measure it on the
       canonical stage. It never scrolls or shrinks a frame, and restores the live view. */
    function preflight() {
      if (P.preflightPromise) return P.preflightPromise;
      P.preflightPromise = (async function () {
        discover(); ensureUi();
        var wasActive = P.active, wasFi = P.fi, wasBuild = P.build, wasFit = P.fit;
        var wasScroll = [window.scrollX, window.scrollY];
        var wasFocus = document.activeElement;
        var liveSecs = all(document, '.sec.is-live'), liveFrames = all(document, '.frame.is-live');
        var buildStates = all(document, '[data-build]').map(function (e) { return [e, e.classList.contains('is-pending'), e.inert, e.getAttribute('aria-hidden')]; });
        var stepStates = all(document, '.stepper').filter(function (e) { return e.stepperApi && !e.closest('[data-present="manual"]'); }).map(function (e) { return [e.stepperApi, e.stepperApi.index()]; });
        var detailStates = all(document, 'details').map(function (e) { return [e, e.open]; });
        var reports = [];
        try {
          if (!wasActive) { P.active = true; document.body.classList.add('present'); }
          document.body.classList.add('is-preflighting');
          liveFrames.forEach(function (e) { e.classList.remove('is-live'); });
          liveSecs.forEach(function (e) { e.classList.remove('is-live'); });
          measureChrome(); await nextPaint();
          for (var i = 0; i < P.frames.length; i++) {
            var fr = P.frames[i];
            all(document, '.frame.is-live').forEach(function (e) { e.classList.remove('is-live'); });
            all(document, '.sec.is-live').forEach(function (e) { e.classList.remove('is-live'); });
            fr.sec.classList.add('is-live'); fr.el.classList.add('is-live'); frameHeading(fr); repaintSvgs(fr);
            all(fr.el, '[data-build]').forEach(function (e) { pending(e, false); });
            var steps = steppersOf(fr);
            steps.forEach(function (s) { s.api.go(s.api.steps.length - 1); });
            var worst = await samplePreflightState(fr, null);
            for (var j = 0; j < steps.length; j++) {
              for (var k = 0; k < steps[j].api.steps.length; k++) {
                steps[j].api.go(k);
                worst = await samplePreflightState(fr, worst);
              }
            }
            if (worst) reports.push(worst);
          }
        } finally {
          all(document, '.frame.is-live').forEach(function (e) { e.classList.remove('is-live'); });
          all(document, '.sec.is-live').forEach(function (e) { e.classList.remove('is-live'); });
          buildStates.forEach(function (s) {
            pending(s[0], s[1]); s[0].inert = s[2];
            if (s[3] == null) s[0].removeAttribute('aria-hidden'); else s[0].setAttribute('aria-hidden', s[3]);
          });
          document.body.classList.remove('is-preflighting');
          P.active = wasActive; P.fi = wasFi; P.build = wasBuild; P.fit = wasFit;
          detailStates.forEach(function (s) { s[0].open = s[1]; });
          if (wasActive && wasFi >= 0 && P.frames[wasFi]) {
            var active = P.frames[wasFi]; active.sec.classList.add('is-live'); active.el.classList.add('is-live'); frameHeading(active); applyBuild(active, wasBuild); repaintSvgs(active);
            stepStates.forEach(function (s) { s[0].go(s[1]); }); scheduleFitCheck();
          } else {
            document.body.classList.remove('present');
            document.body.style.removeProperty('--present-scale');
            if (U.fitWarning) U.fitWarning.textContent = '';
            stepStates.forEach(function (s) { s[0].go(s[1]); });
            window.scrollTo(wasScroll[0], wasScroll[1]);
          }
          if (wasFocus && wasFocus.isConnected && typeof wasFocus.focus === 'function') wasFocus.focus({ preventScroll: true });
        }
        return { total: reports.length, overflow: reports.filter(function (r) { return r.overflow; }), frames: reports };
      })().finally(function () { P.preflightPromise = null; });
      return P.preflightPromise;
    }
    function fullscreen() {
      var fallback = function () { U.announcement.textContent = 'Full screen is unavailable here. Use your browser’s full-screen command; classroom navigation still works.'; };
      U.announcement.textContent = '';
      try {
        var result;
        if (document.fullscreenElement && document.exitFullscreen) result = document.exitFullscreen();
        else if (document.documentElement.requestFullscreen) { P.ownsFullscreen = true; result = document.documentElement.requestFullscreen(); }
        else { fallback(); return; }
        if (result && result.catch) result.catch(fallback);
      } catch (e) { fallback(); }
    }
    function notesHtml(list, empty) {
      if (!list || !list.length) return '<p class="at-notes-empty">' + (empty || 'No notes for this frame.') + '</p>';
      return list.map(function (p) { return '<p>' + escapeHtml(p) + '</p>'; }).join('');
    }
    function updateChrome() {
      var fr = P.frames[P.fi]; if (!fr || !U.counter) return;
      U.counter.textContent = fr.num + '.' + (fr.index + 1) + '/' + fr.count + (fr.maxBuild ? '  b' + P.build + '/' + fr.maxBuild : '');
      var st = activeStepper(fr, P.build), back = activeStepper(fr, P.build, -1);
      if (st) U.counter.textContent += ' · step ' + (st.index() + 1) + '/' + st.steps.length;
      U.counter.setAttribute('aria-label', fr.secTitle + (fr.title ? ': ' + fr.title : '') + '. Frame ' + (fr.index + 1) + ' of ' + fr.count + ', build ' + P.build + ' of ' + fr.maxBuild + (st ? ', step ' + (st.index() + 1) + ' of ' + st.steps.length : ''));
      U.prev.disabled = P.fi === 0 && P.build === 0 && !(back && back.index() > 0);
      U.next.disabled = P.fi === P.frames.length - 1 && P.build === fr.maxBuild && !(st && st.index() < st.steps.length - 1);
      U.prev.setAttribute('aria-label', back && back.index() > 0 ? 'Previous step in this build' : P.build > 0 ? 'Previous build' : 'Previous frame');
      U.next.setAttribute('aria-label', st && st.index() < st.steps.length - 1 ? 'Next step in this build' : P.build < fr.maxBuild ? 'Next build' : 'Next frame');
      U.fullscreen.textContent = document.fullscreenElement ? 'Leave full screen' : 'Full screen';
      U.fullscreen.setAttribute('aria-pressed', document.fullscreenElement ? 'true' : 'false');
      frameHeading(fr);
      if (P.notes) U.notes.innerHTML = notesHtml(fr.notes);
      var bar = document.getElementById('progress');
      if (bar) bar.style.width = (100 * (P.fi + (P.build + 1) / (fr.maxBuild + 1)) / P.frames.length).toFixed(2) + '%';
    }
    function renderOverview() {
      var grid = U.overview.querySelector('.ov-grid'); clear(grid);
      P.frames.forEach(function (fr, i) {
        var b = h('button', { type: 'button', class: 'ov-item' + (i === P.fi ? ' is-current' : ''), 'aria-current': i === P.fi ? 'true' : null, on: { click: function () { setOverview(false); showFrame(i, 0); cur().el.focus({ preventScroll: true }); } } },
          h('span', { class: 'ov-n' }, fr.num + '.' + (fr.index + 1) + (fr.maxBuild ? ' · ' + (fr.maxBuild + 1) + ' builds' : '')),
          h('span', { class: 'ov-t' }, fr.secTitle), fr.title ? h('span', { class: 'ov-s' }, fr.title) : null);
        grid.appendChild(b);
      });
    }
    function setOverview(on) {
      on = !!on; if (on === P.overview || !U.overview) return;
      P.overview = on; U.overview.classList.toggle('is-on', on);
      if (on) {
        overviewFocus = document.activeElement;
        overviewBackground = children(document.body).filter(function (e) { return e !== U.overview && e.tagName !== 'SCRIPT' && e.tagName !== 'STYLE'; }).map(function (e) { var saved = [e, e.inert]; e.inert = true; return saved; });
        renderOverview(); var c = U.overview.querySelector('.is-current'); if (c) c.focus();
      } else {
        overviewBackground.forEach(function (p) { p[0].inert = p[1]; }); overviewBackground = [];
        var focus = overviewFocus && overviewFocus.isConnected && overviewFocus.getClientRects().length ? overviewFocus : U.overviewButton;
        if (focus) focus.focus({ preventScroll: true }); overviewFocus = null;
      }
      emit('overview', P.overview);
    }
    function setBlank(on) { P.blank = !!on; if (U.blank) U.blank.classList.toggle('is-on', P.blank); }
    function setNotes(on) { P.notes = !!on; if (U.notes) { U.notes.classList.toggle('is-on', P.notes); if (P.notes) updateChrome(); } }
    function setHelp(on) { P.help = !!on; if (U.help) U.help.classList.toggle('is-on', P.help); }

    /* ---- frames ---- */
    function cur() { return P.frames[P.fi] || null; }
    function leaveFrame(fr) {
      if (!fr) return;
      restore(fr.snapshot);
      steppersOf(fr).forEach(function (s) { if (s.api.index() !== 0) s.api.go(0); });
      all(fr.el, '[data-build]').forEach(function (e) { pending(e, false); });
      fr.el.classList.remove('is-live');
    }
    function showFrame(i, b, edge) {
      i = Math.max(0, Math.min(P.frames.length - 1, i | 0));
      var prev = cur(), fr = P.frames[i];
      if (!fr) return;
      if (prev && prev !== fr) { leaveFrame(prev); if (prev.sec !== fr.sec) prev.sec.classList.remove('is-live'); }
      var moveFocus = prev && prev !== fr && prev.el.contains(document.activeElement);
      fr.sec.classList.add('is-live'); fr.el.classList.add('is-live');
      repaintSvgs(fr);
      if (prev !== fr) {
        if (!fr.snapshot) fr.snapshot = snapshot(fr.el);
        if (AT.strip) AT.strip.setCurrent(fr.secIndex);
      }
      P.fi = i;
      P.build = Math.max(0, Math.min(fr.maxBuild, b | 0));
      applyBuild(fr, P.build, edge || 'start');
      if (moveFocus) fr.el.focus({ preventScroll: true });
      updateChrome(); syncHash(); postState(); emit('frame', state()); scheduleFitCheck();
    }
    function reveal(el, block) { if (P.active) { scheduleFitCheck(); return; } if (!el || !el.scrollIntoView) return; try { el.scrollIntoView({ block: block || 'nearest', inline: 'nearest' }); } catch (e) { /* ignore */ } }
    function setBuild(b) {
      var fr = cur(); if (!fr) return;
      var was = P.build;
      P.build = Math.max(0, Math.min(fr.maxBuild, b | 0)); applyBuild(fr, P.build, P.build < was ? 'end' : P.build > was ? 'start' : null);
      if (P.build > was) { var first = fr.el.querySelector('[data-build="' + P.build + '"]'); if (first) { var st = activeStepper(fr, P.build); reveal(st ? st.el : first, st ? 'start' : 'nearest'); } }
      updateChrome(); syncHash(); postState(); emit('build', state()); scheduleFitCheck();
    }
    function next() {
      var fr = cur(); if (!P.active || !fr) return;
      var st = activeStepper(fr, P.build);
      if (st && st.index() < st.steps.length - 1) { st.next(); reveal(st.el, 'start'); updateChrome(); postState(); emit('step', state()); return; }
      if (P.build < fr.maxBuild) setBuild(P.build + 1);
      else if (P.fi < P.frames.length - 1) showFrame(P.fi + 1, 0);
    }
    function prev() {
      var fr = cur(); if (!P.active || !fr) return;
      var st = activeStepper(fr, P.build, -1);
      if (st && st.index() > 0) { st.prev(); reveal(st.el, 'start'); updateChrome(); postState(); emit('step', state()); return; }
      if (P.build > 0) setBuild(P.build - 1);
      else if (P.fi > 0) showFrame(P.fi - 1, P.frames[P.fi - 1].maxBuild, 'end');
    }
    function frameIndexFor(id, f) { var k = -1; P.frames.forEach(function (fr, i) { if (k < 0 && fr.id === id && fr.index === Math.max(0, (f | 0) - 1)) k = i; }); if (k < 0) P.frames.forEach(function (fr, i) { if (k < 0 && fr.id === id) k = i; }); return k; }
    function go(id, f, b) { if (typeof id === 'number') { showFrame(id, b || 0); return; } var i = frameIndexFor(String(id), f == null ? 1 : f); if (i >= 0) showFrame(i, b || 0); }
    function parseHash(hsh) { var m = /^#?(s\d+)\/(\d+)\/(\d+)$/.exec(hsh || ''); return m ? { id: m[1], f: +m[2], b: +m[3] } : null; }
    function hashOf() { var fr = cur(); return fr ? '#' + fr.id + '/' + (fr.index + 1) + '/' + P.build : ''; }
    var syncing = false;
    function syncHash() { if (!P.active) return; var hh = hashOf(); if (hh && location.hash !== hh) { syncing = true; try { history.replaceState(null, '', hh); } catch (e) { location.hash = hh; } setTimeout(function () { syncing = false; }, 0); } }
    function sectionAtViewport() {
      var secs = Array.prototype.slice.call(document.querySelectorAll('main .sec')), y = window.scrollY + window.innerHeight * 0.4, best = 0;
      secs.forEach(function (s, i) { if (s.getBoundingClientRect().top + window.scrollY <= y) best = i; });
      return secs[best] ? secs[best].id : null;
    }

    /* ---- enter / exit ---- */
    function enter(target) {
      if (P.active) { if (target) goTarget(target); return; }
      discover(); ensureUi();
      if (!P.frames.length) return;
      var t = target || parseHash(location.hash);
      if (!t) { var id = sectionAtViewport(); t = id ? { id: id, f: 1, b: 0 } : { index: 0 }; }
      if (t.id && frameIndexFor(t.id, t.f) < 0) t = { index: 0 };
      entryFocus = document.activeElement;
      P.frames.forEach(function (fr) { fr.snapshot = null; }); P.fi = -1;
      P.active = true; P.startedAt = P.startedAt || Date.now();
      document.body.classList.add('present');
      goTarget(t); measureChrome();
      emit('change', state());
    }
    function goTarget(t) { if (t.index != null) showFrame(t.index, t.b || 0); else go(t.id, t.f, t.b); }
    function exit() {
      if (!P.active) return;
      var fr = cur();
      setOverview(false); setBlank(false); setNotes(false); setHelp(false);
      if (fr) { leaveFrame(fr); fr.sec.classList.remove('is-live'); }
      P.active = false;
      document.body.classList.remove('present');
      U.announcement.textContent = '';
      if (U.fitWarning) U.fitWarning.textContent = '';
      document.body.style.removeProperty('--present-scale');
      try { var url = new URL(location.href); url.searchParams.delete('present'); url.hash = fr ? fr.id : ''; history.replaceState(null, '', url.href); } catch (e) { /* ignore */ }
      if (P.ownsFullscreen && document.fullscreenElement && document.exitFullscreen) { var result = document.exitFullscreen(); if (result && result.catch) result.catch(function () {}); } P.ownsFullscreen = false;
      if (entryFocus && entryFocus.isConnected && entryFocus !== document.body) entryFocus.focus({ preventScroll: true });
      if (fr) fr.sec.scrollIntoView({ block: 'start' });
      emit('change', state()); postState();
    }
    function toggle() { if (P.active) exit(); else enter(); }

    /* ---- print: every build visible, steppers at their last step, read-mode layout; restored afterwards ---- */
    function prepareForPrint() {
      if (P.printState) return;
      var st = { active: P.active, fi: P.fi, build: P.build, steppers: [] };
      all(document, '.stepper').forEach(function (e) { if (e.stepperApi && e.stepperApi.steps.length) { st.steppers.push([e.stepperApi, e.stepperApi.index()]); e.stepperApi.go(e.stepperApi.steps.length - 1); } });
      if (P.active) {
        var fr = cur();
        all(document, '[data-build].is-pending').forEach(function (e) { pending(e, false); });
        if (fr) { fr.el.classList.remove('is-live'); fr.sec.classList.remove('is-live'); }
        document.body.classList.remove('present');
      }
      document.documentElement.classList.add('printing');
      P.printState = st;
    }
    function restoreAfterPrint() {
      var st = P.printState; if (!st) return;
      P.printState = null;
      document.documentElement.classList.remove('printing');
      st.steppers.forEach(function (p) { p[0].go(p[1]); });
      if (st.active) { document.body.classList.add('present'); var fr = P.frames[st.fi]; if (fr) { fr.sec.classList.add('is-live'); fr.el.classList.add('is-live'); applyBuild(fr, st.build); } }
    }

    /* ---- keys ---- */
    function inField(t) { if (!t || !t.tagName) return false; var tag = t.tagName; return tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'INPUT' || t.isContentEditable; }
    function overviewKey(key, ev) {
      if (key === 'Escape' || key === 'o' || key === 'O') { setOverview(false); return true; }
      var buttons = all(U.overview, 'button:not([disabled])'), i = buttons.indexOf(document.activeElement), to;
      if (key === 'Tab') to = (i + (ev && ev.shiftKey ? -1 : 1) + buttons.length) % buttons.length;
      else if (key === 'ArrowRight' || key === 'ArrowDown') to = (i + 1) % buttons.length;
      else if (key === 'ArrowLeft' || key === 'ArrowUp') to = (i - 1 + buttons.length) % buttons.length;
      else if (key === 'Home') to = 0;
      else if (key === 'End') to = buttons.length - 1;
      if (to != null && buttons[to]) { buttons[to].focus(); return true; }
      return false;
    }
    function handleKey(key, target, ev) {
      if (!P.active) { if ((key === 'p' || key === 'P') && !inField(target)) { enter(); return true; } return false; }
      if (P.overview) return overviewKey(key, ev);
      if (P.blank) { if (key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta') return false; setBlank(false); return true; }
      if (key === 'Escape') { if (P.help) setHelp(false); else exit(); return true; }
      if (inField(target) || (target && target.closest && target.closest('[data-present="manual"]'))) return false;
      switch (key) {
        case 'ArrowRight': next(); return true;
        case ' ': if (target && (target.tagName === 'BUTTON' || target.tagName === 'SUMMARY' || target.tagName === 'A' || target.getAttribute('role') === 'button')) return false; next(); return true;
        case 'PageDown': case 'n': case 'N': next(); return true;
        case 'ArrowLeft': prev(); return true;
        case 'PageUp': case 'Backspace': prev(); return true;
        case 'Home': showFrame(0, 0); return true;
        case 'End': showFrame(P.frames.length - 1, 0); return true;
        case 'o': case 'O': setOverview(!P.overview); return true;
        case 'b': case 'B': case '.': setBlank(!P.blank); return true;
        case 's': case 'S': setNotes(!P.notes); return true;
        case '?': setHelp(!P.help); return true;
        case 'f': case 'F': fullscreen(); return true;
      }
      return false;
    }
    function onKeydown(ev) {
      if (ev.metaKey || ev.ctrlKey || ev.altKey || ev.defaultPrevented) return;
      if (handleKey(ev.key, ev.target, ev)) { ev.preventDefault(); ev.stopPropagation(); }
    }

    /* ---- presenter window ---- */
    function state() {
      var fr = cur(), nx = P.frames[P.fi + 1] || null, st = fr ? activeStepper(fr, P.build) : null;
      function fs(f) { return f ? { id: f.id, num: f.num, secTitle: f.secTitle, title: f.title, index: f.index, count: f.count, maxBuild: f.maxBuild, notes: f.notes } : null; }
      return { active: P.active, fi: P.fi, total: P.frames.length, build: P.build, frame: fs(fr), next: fs(nx), stepper: st ? { index: st.index(), count: st.steps.length } : null, fit: P.fit, hash: hashOf(), startedAt: P.startedAt, part: (window.__PART__ && window.__PART__.title) || document.title };
    }
    function postState() {
      var w = P.presWin;
      if (!w) return;
      try { if (w.closed) { P.presWin = null; return; } w.postMessage({ type: 'at-state', state: state() }, '*'); } catch (e) { /* ignore */ }
    }
    function openPresenter() {
      var url = new URL(location.href); url.hash = ''; url.searchParams.delete('present'); var base = url.href;
      var w = null;
      try { w = window.open(base + '#presenter', 'at-presenter', 'width=1100,height=720'); } catch (e) { w = null; }
      if (w) P.presWin = w;
      if (!P.active) enter();
      postState();
      return w;
    }
    function onMessage(ev) {
      var d = ev.data; if (!d || typeof d.type !== 'string' || d.type.indexOf('at-') !== 0) return;
      if (P.isPresenter) { if (d.type === 'at-state' && d.state) renderPresenter(d.state); return; }
      if (d.type === 'at-presenter-ready') { P.presWin = ev.source; if (!P.active) enter(); postState(); }
      else if (d.type === 'at-key') { if (!P.active) enter(); handleKey(String(d.key || ''), null, null); }
      else if (d.type === 'at-go') { if (!P.active) enter(); go(d.id, d.f, d.b); }
    }
    var PR = {};
    function bootPresenter() {
      P.isPresenter = true;
      document.body.classList.add('presenter');
      document.title = 'Presenter: ' + document.title;
      var root = h('div', { id: 'at-presenter' });
      PR.part = h('span', { class: 'pr-part' }, (window.__PART__ && window.__PART__.title) || document.title);
      PR.clock = h('span', { class: 'pr-clock' }, '--:--');
      PR.elapsed = h('span', { class: 'pr-elapsed' }, '00:00');
      PR.num = h('p', { class: 'pr-num' }, ''); PR.title = h('h2', {}, 'Waiting for the presentation window'); PR.sub = h('p', { class: 'pr-sub' }, '');
      PR.build = h('p', { class: 'pr-build' }, ''); PR.dots = h('div', { class: 'pr-dots' });
      PR.notes = h('div', { class: 'pr-notes' }, h('p', { class: 'at-notes-empty' }, 'Open this window with the Presenter view button of the presentation, or press P there.'));
      PR.nextT = h('h3', {}, ''); PR.nextS = h('p', { class: 'pr-sub' }, ''); PR.nextN = h('div', { class: 'pr-notes' });
      PR.status = h('span', { class: 'pr-status' }, window.opener ? 'connected' : 'no presentation window');
      function key(k) { return function () { send({ type: 'at-key', key: k }); }; }
      root.appendChild(h('div', { class: 'pr-top' }, h('span', { class: 'pr-lab', style: 'margin:0' }, 'Presenter'), PR.part, PR.elapsed, PR.clock));
      root.appendChild(h('div', { class: 'pr-panes' },
        h('div', { class: 'pr-pane pr-cur' }, h('p', { class: 'pr-lab' }, 'Now'), PR.num, PR.title, PR.sub, PR.build, PR.dots),
        h('div', { class: 'pr-pane' }, h('p', { class: 'pr-lab' }, 'Notes'), PR.notes),
        h('div', { class: 'pr-pane pr-next' }, h('p', { class: 'pr-lab' }, 'Next frame'), PR.nextT, PR.nextS, PR.nextN)));
      root.appendChild(h('div', { class: 'pr-bottom' },
        h('button', { type: 'button', class: 'btn btn-quiet', on: { click: key('ArrowLeft') } }, 'Back'),
        h('button', { type: 'button', class: 'btn btn-primary', on: { click: key('ArrowRight') } }, 'Next'),
        h('button', { type: 'button', class: 'btn btn-quiet', on: { click: key('b') } }, 'Blank'),
        h('span', {}, 'Keys here drive the presentation: arrows, Space, PageUp/PageDown, Home, End, B.'), PR.status));
      document.body.appendChild(root);
      function send(m) { if (window.opener && !window.opener.closed) { try { window.opener.postMessage(m, '*'); } catch (e) { /* ignore */ } } }
      window.addEventListener('keydown', function (ev) {
        if (ev.metaKey || ev.ctrlKey || ev.altKey || inField(ev.target)) return;
        var k = ev.key;
        if (['ArrowRight', 'ArrowLeft', ' ', 'PageDown', 'PageUp', 'Home', 'End', 'Backspace', 'n', 'N', 'b', 'B', '.'].indexOf(k) >= 0) { send({ type: 'at-key', key: k }); ev.preventDefault(); }
      });
      function tick() {
        var d = new Date(); PR.clock.textContent = (d.getHours() < 10 ? '0' : '') + d.getHours() + ':' + (d.getMinutes() < 10 ? '0' : '') + d.getMinutes();
        if (PR.startedAt) { var s = Math.max(0, Math.round((Date.now() - PR.startedAt) / 1000)); PR.elapsed.textContent = 'elapsed ' + (s / 60 | 0) + ':' + ((s % 60) < 10 ? '0' : '') + (s % 60); }
      }
      tick(); setInterval(tick, 1000);
      send({ type: 'at-presenter-ready' });
    }
    function renderPresenter(st) {
      if (!st.active || !st.frame) { PR.title.textContent = 'The presentation has ended'; PR.status.textContent = 'read mode'; return; }
      PR.status.textContent = 'connected';
      PR.startedAt = st.startedAt || PR.startedAt || Date.now();
      var f = st.frame;
      PR.num.textContent = f.num + ' · frame ' + (f.index + 1) + ' of ' + f.count + ' · ' + (st.fi + 1) + '/' + st.total;
      PR.title.textContent = f.secTitle; PR.sub.textContent = f.title;
      PR.build.textContent = 'build ' + st.build + ' of ' + f.maxBuild + (st.stepper ? ' · step ' + (st.stepper.index + 1) + ' of ' + st.stepper.count : '');
      clear(PR.dots); for (var i = 0; i <= f.maxBuild; i++) PR.dots.appendChild(h('span', { class: i < st.build ? 'on' : (i === st.build ? 'cur' : '') }));
      PR.notes.innerHTML = notesHtml(f.notes);
      if (st.next) { PR.nextT.textContent = st.next.num + '.' + (st.next.index + 1) + '  ' + st.next.secTitle; PR.nextS.textContent = st.next.title; PR.nextN.innerHTML = notesHtml(st.next.notes.slice(0, 1), ''); }
      else { PR.nextT.textContent = 'Last frame'; PR.nextS.textContent = ''; PR.nextN.innerHTML = ''; }
    }

    /* ---- boot ---- */
    function boot() {
      if (location.hash === '#presenter') { bootPresenter(); window.addEventListener('message', onMessage); return; }
      window.addEventListener('keydown', onKeydown, true);
      window.addEventListener('message', onMessage);
      window.addEventListener('beforeprint', prepareForPrint);
      window.addEventListener('afterprint', restoreAfterPrint);
      document.addEventListener('at-stepchange', function (ev) { if (P.active && cur() && cur().el.contains(ev.target)) { updateChrome(); postState(); repaintSvgs(cur()); scheduleFitCheck(); } });
      window.addEventListener('hashchange', function () { if (syncing) return; var t = parseHash(location.hash); if (t) { if (P.active) goTarget(t); else enter(t); } });
      var wantsPresent = /(^|[?&])present(=|&|$)/.test(location.search) || !!parseHash(location.hash);
      if (wantsPresent) { var t = parseHash(location.hash); if (!t && /^#s\d+$/.test(location.hash)) t = { id: location.hash.slice(1), f: 1, b: 0 }; enter(t); }
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();

    return {
      enter: enter, exit: exit, toggle: toggle, next: next, prev: prev, go: go, first: function () { showFrame(0, 0); }, last: function () { showFrame(P.frames.length - 1, 0); },
      setBuild: setBuild, overview: setOverview, blank: setBlank, notes: setNotes, help: setHelp, fullscreen: fullscreen, openPresenter: openPresenter,
      prepareForPrint: prepareForPrint, restoreAfterPrint: restoreAfterPrint, discover: discover, fitReport: function () { return reportFit(); }, preflight: preflight,
      isActive: function () { return P.active; }, state: state, frames: function () { discover(); return P.frames.slice(); }, on: on, parseHash: parseHash
    };
  })();

  window.AT = AT;
})();

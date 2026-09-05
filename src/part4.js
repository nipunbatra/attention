/* Part IV: a numerical encoder-decoder model and its teaching diagrams.
   Data lives in toy4.json. This runtime never uses the older AT.forward toy.

   AT.translation.forward(prefixTokens, {snapshot:'before', source?})
     -> sourceLookup, sourcePositions, sourceRows; matching target fields;
        encoder, decoderSelf, cross; logits, probs, last.
   Each attention block has input,memory,Q,K,V,Sraw,Sfull,S,A,Mmsg,Delta,Enew.
   teacherForced({snapshot}) adds targets,perPosition,meanLoss.
   generate({snapshot,maxTokens,source?}) runs greedy decoding from BOS.
   parameters(snapshot) returns a copy; update and training describe snapshots.
   diagram(host, stage, {prefixTokens?,snapshot?,index?,step?}) draws an SVG.
*/
(function () {
  'use strict';
  var AT = window.AT;
  var data = (window.__TOY__ || {}).translation;
  if (!AT || !data) throw new Error('Part IV translation data is missing.');
  var D = 3;
  var sourceVocab = data.sourceVocab.slice();
  var targetVocab = data.targetVocab.slice();
  var sourceIndex = {}, targetIndex = {};
  sourceVocab.forEach(function (token, i) { sourceIndex[token] = i; });
  targetVocab.forEach(function (token, i) { targetIndex[token] = i; });
  function copy(value) { return JSON.parse(JSON.stringify(value)); }
  function checkedTokens(tokens, index, capacity, label) {
    if (!Array.isArray(tokens) || !tokens.length || tokens.length > capacity) {
      throw new Error(label + ' needs 1 to ' + capacity + ' tokens.');
    }
    return tokens.map(function (token) {
      if (typeof token !== 'string' || !Object.prototype.hasOwnProperty.call(index, token)) {
        throw new Error('Unknown ' + label.toLowerCase() + ' token: ' + token);
      }
      return token;
    });
  }
  function snapshotName(name) {
    name = name == null ? data.defaultSnapshot : name;
    if (!Object.prototype.hasOwnProperty.call(data.snapshots, name)) throw new Error('Unknown snapshot: ' + name);
    return name;
  }
  function parameters(name) { return copy(data.snapshots[snapshotName(name)]); }
  function matrix(values, rows, columns, name) {
    if (!Array.isArray(values) || values.length !== rows || !values.every(function (row) {
      return Array.isArray(row) && row.length === columns && row.every(Number.isFinite);
    })) throw new Error('Invalid ' + name + ': expected ' + rows + ' × ' + columns + ' finite numbers.');
  }
  function validateParameters(p) {
    matrix(p.E_src, sourceVocab.length, D, 'E_src');
    matrix(p.P_src, data.maxSource, D, 'P_src');
    matrix(p.E_tgt, targetVocab.length, D, 'E_tgt');
    matrix(p.P_tgt, data.maxTarget, D, 'P_tgt');
    ['enc', 'dec', 'cross'].forEach(function (block) {
      ['Q', 'K', 'V', 'O'].forEach(function (role) { matrix(p[block + '_W_' + role], D, D, block + '_W_' + role); });
    });
    matrix(p.W_vocab, D, targetVocab.length, 'W_vocab');
    if (!Array.isArray(p.b_vocab) || p.b_vocab.length !== targetVocab.length || !p.b_vocab.every(Number.isFinite)) {
      throw new Error('Invalid b_vocab.');
    }
    return p;
  }
  function matmul(a, b) {
    return a.map(function (row) {
      return b[0].map(function (_, j) {
        return row.reduce(function (sum, value, k) { return sum + value * b[k][j]; }, 0);
      });
    });
  }
  function transpose(a) { return a[0].map(function (_, j) { return a.map(function (row) { return row[j]; }); }); }
  function add(a, b) { return a.map(function (row, i) { return row.map(function (v, j) { return v + b[i][j]; }); }); }
  function softmax(row) {
    var max = Math.max.apply(null, row);
    var ex = row.map(function (value) { return value === -Infinity ? 0 : Math.exp(value - max); });
    var sum = ex.reduce(function (a, b) { return a + b; }, 0);
    if (!Number.isFinite(sum) || sum <= 0) throw new Error('Softmax needs at least one finite score.');
    return ex.map(function (v) { return v / sum; });
  }
  function argmax(row) { return row.reduce(function (best, value, i) { return value > row[best] ? i : best; }, 0); }
  function attention(p, input, memory, name, causal) {
    var Q = matmul(input, p[name + '_W_Q']);
    var K = matmul(memory, p[name + '_W_K']);
    var V = matmul(memory, p[name + '_W_V']);
    var Sraw = matmul(Q, transpose(K));
    var Sfull = Sraw.map(function (row) { return row.map(function (v) { return v / Math.sqrt(D); }); });
    var S = Sfull.map(function (row, i) { return row.map(function (v, j) { return causal && j > i ? -Infinity : v; }); });
    var A = S.map(softmax);
    var Mmsg = matmul(A, V);
    var Delta = matmul(Mmsg, p[name + '_W_O']);
    return { input: input, memory: memory, Q: Q, K: K, V: V, Sraw: Sraw, Sfull: Sfull,
      S: S, A: A, Mmsg: Mmsg, Delta: Delta, Enew: add(input, Delta), causal: !!causal };
  }
  function encodeWith(p, source) {
    source = checkedTokens(source || data.source, sourceIndex, data.maxSource, 'Source');
    var ids = source.map(function (t) { return sourceIndex[t]; });
    var lookup = ids.map(function (id) { return p.E_src[id].slice(); });
    var positions = p.P_src.slice(0, source.length).map(function (row) { return row.slice(); });
    var rows = add(lookup, positions);
    return { source: source, sourceIds: ids, sourceLookup: lookup,
      sourcePositions: positions, sourceRows: rows, encoder: attention(p, rows, rows, 'enc', false) };
  }
  function decodeWith(p, prefix, encoded) {
    prefix = checkedTokens(prefix, targetIndex, data.maxTarget, 'Target prefix');
    if (prefix[0] !== '<bos>') throw new Error('A target prefix must begin with <bos>.');
    var ids = prefix.map(function (t) { return targetIndex[t]; });
    var lookup = ids.map(function (id) { return p.E_tgt[id].slice(); });
    var positions = p.P_tgt.slice(0, prefix.length).map(function (row) { return row.slice(); });
    var rows = add(lookup, positions);
    var self = attention(p, rows, rows, 'dec', true);
    // Cross-attention asks from the post-self-attention decoder rows. Keys and
    // values come from the separate, fully encoded English source.
    var cross = attention(p, self.Enew, encoded.encoder.Enew, 'cross', false);
    var logits = matmul(cross.Enew, p.W_vocab).map(function (row) {
      return row.map(function (v, i) { return v + p.b_vocab[i]; });
    });
    var probs = logits.map(softmax);
    var lastIndex = prefix.length - 1;
    var chosenId = argmax(probs[lastIndex]);
    return Object.assign({}, encoded, { prefix: prefix, targetIds: ids,
      targetLookup: lookup, targetPositions: positions, targetRows: rows,
      decoderSelf: self, cross: cross, logits: logits, probs: probs,
      last: { index: lastIndex, position: lastIndex + 1, token: prefix[lastIndex],
        query: cross.Q[lastIndex].slice(), weights: cross.A[lastIndex].slice(),
        logits: logits[lastIndex].slice(), probs: probs[lastIndex].slice(),
        chosen: targetVocab[chosenId], chosenId: chosenId } });
  }
  function forwardWith(p, prefix, source) {
    validateParameters(p);
    return decodeWith(p, prefix, encodeWith(p, source));
  }
  function forward(prefix, options) {
    options = options || {};
    var name = snapshotName(options.snapshot);
    var p = options.parameters || data.snapshots[name];
    var result = forwardWith(p, prefix || data.targetInput, options.source);
    result.snapshot = name;
    return result;
  }
  function teacherForced(options) {
    options = options || {};
    var prefix = options.input || data.targetInput;
    var targets = options.targets || data.targets;
    if (!Array.isArray(targets) || targets.length !== prefix.length) throw new Error('Each input row needs one target.');
    var result = forward(prefix, options);
    result.targets = checkedTokens(targets, targetIndex, data.maxTarget, 'Target');
    result.perPosition = targets.map(function (target, i) {
      var id = targetIndex[target], logits = result.logits[i], max = Math.max.apply(null, logits);
      var logSumExp = max + Math.log(logits.reduce(function (sum, v) { return sum + Math.exp(v - max); }, 0));
      return { position: i + 1, input: prefix[i], target: target, targetId: id,
        probability: result.probs[i][id], loss: logSumExp - logits[id] };
    });
    result.meanLoss = result.perPosition.reduce(function (sum, row) { return sum + row.loss; }, 0) / prefix.length;
    return result;
  }
  function generate(options) {
    options = options || {};
    var name = snapshotName(options.snapshot), p = data.snapshots[name];
    var maxTokens = options.maxTokens == null ? data.maxTarget : options.maxTokens;
    if (!Number.isInteger(maxTokens) || maxTokens < 1) throw new Error('maxTokens must be a positive integer.');
    validateParameters(p);
    var encoded = encodeWith(p, options.source), prefix = ['<bos>'], trace = [], reason = 'limit';
    for (var step = 0; step < maxTokens; step++) {
      if (prefix.length > data.maxTarget) { reason = 'capacity'; break; }
      var f = decodeWith(p, prefix, encoded), last = f.last;
      trace.push({ step: step + 1, prefix: prefix.slice(), query: last.query,
        weights: last.weights, logits: last.logits, probabilities: last.probs,
        chosen: last.chosen, chosenId: last.chosenId });
      prefix = prefix.concat([last.chosen]);
      if (last.chosen === '<eos>') { reason = 'eos'; break; }
    }
    var tokens = prefix.slice(1);
    return { source: encoded.source, snapshot: name, tokens: tokens,
      text: tokens.filter(function (t) { return t !== '<eos>' && t !== '<bos>'; }).join(' '),
      prefix: prefix, trace: trace, stoppedBy: reason, encoderEvaluations: 1 };
  }
  function topk(row, count) {
    return row.map(function (p, i) { return { tok: targetVocab[i], p: p, i: i }; })
      .sort(function (a, b) { return b.p - a.p; }).slice(0, count == null ? row.length : count);
  }
  function parameterRows() {
    var p = data.snapshots.before;
    return Object.keys(p).map(function (name) {
      var value = p[name], grad = data.update.gradients[name];
      var flat = value.flat(), g = grad.flat();
      return { name: name, rows: Array.isArray(value[0]) ? value.length : 1,
        columns: Array.isArray(value[0]) ? value[0].length : value.length,
        count: flat.length, gradientNorm: Math.sqrt(g.reduce(function (sum, v) { return sum + v * v; }, 0)) };
    });
  }

  /* Accessible SVGs use the same role colours as Parts I-III. All nodes and
     arrow markers are scoped to the returned SVG instance. */
  var serial = 0;
  var NS = 'http://www.w3.org/2000/svg';
  var palette = { e: 'var(--c-e)', q: 'var(--c-q)', k: 'var(--c-k)', v: 'var(--c-v)',
    a: 'var(--c-a)', d: 'var(--c-d)', ink: 'var(--ink)', muted: 'var(--ink-3)' };
  function sv(tag, attributes, text) {
    var element = document.createElementNS(NS, tag);
    Object.keys(attributes || {}).forEach(function (name) { element.setAttribute(name, String(attributes[name])); });
    if (text != null) element.textContent = text;
    return element;
  }
  function canvas(height, title, description) {
    var id = 'translation-diagram-' + (++serial);
    var svg = sv('svg', { id: id, viewBox: '0 0 1100 ' + height, role: 'img',
      'aria-labelledby': id + '-title ' + id + '-desc', 'data-part4-diagram': title });
    svg.appendChild(sv('title', { id: id + '-title' }, title));
    svg.appendChild(sv('desc', { id: id + '-desc' }, description));
    var defs = sv('defs');
    Object.keys(palette).forEach(function (role) {
      var marker = sv('marker', { id: id + '-' + role, viewBox: '0 0 10 10', refX: 9, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
      marker.appendChild(sv('path', { d: 'M0 0 L10 5 L0 10 Z', fill: palette[role] }));
      defs.appendChild(marker);
    });
    svg.appendChild(defs);
    function text(x, y, value, role, size, anchor) {
      var t = sv('text', { x: x, y: y, fill: palette[role || 'ink'], 'font-size': size || 22,
        'font-family': 'var(--font-ui)', 'text-anchor': anchor || 'middle', 'dominant-baseline': 'middle' }, value);
      svg.appendChild(t); return t;
    }
    function box(x, y, width, height, label, subtitle, role) {
      svg.appendChild(sv('rect', { x: x - width / 2, y: y - height / 2, width: width,
        height: height, rx: 8, fill: role ? 'var(--t-' + role + ')' : 'var(--card)',
        stroke: palette[role || 'muted'], 'stroke-width': 1.8 }));
      text(x, subtitle ? y - 10 : y, label, role, Math.min(21, (width - 18) / Math.max(1, label.length) * 1.7));
      if (subtitle) text(x, y + 17, subtitle, 'muted', Math.min(16, (width - 16) / Math.max(1, subtitle.length) * 1.85));
    }
    function arrow(path, role, dashed) {
      svg.appendChild(sv('path', { d: path, fill: 'none', stroke: palette[role || 'muted'],
        'stroke-width': 2.3, 'stroke-dasharray': dashed ? '6 4' : 'none',
        'marker-end': 'url(#' + id + '-' + (role || 'muted') + ')' }));
    }
    return { svg: svg, text: text, box: box, arrow: arrow };
  }
  function tokensDiagram(f) {
    var c = canvas(210, 'English source and French prefix', 'The complete English source ends with EOS. The French prefix starts with BOS; the next French token is still unknown.');
    c.text(20, 26, 'Complete English source', 'e', 22, 'start');
    c.text(20, 125, 'Known French prefix', 'q', 22, 'start');
    function row(tokens, y, role, slot) {
      var width = Math.min(138, 850 / (tokens.length + (slot ? 1 : 0)) - 12), gap = 12;
      var left = 220;
      tokens.forEach(function (t, i) { c.box(left + width / 2 + i * (width + gap), y, width, 48, t, null, role); c.text(left + width / 2 + i * (width + gap), y + 37, String(i + 1), 'muted', 16); });
      if (slot) c.box(left + width / 2 + tokens.length * (width + gap), y, width, 48, '?', null, 'd');
    }
    row(f.source, 58, 'e', false); row(f.prefix, 157, 'q', true);
    return c.svg;
  }
  function masksDiagram(f) {
    var c = canvas(340, 'Three attention masks', 'Rows are query positions and columns are key and value positions. Encoder self-attention reads every source position. Decoder self-attention blocks future French positions. Cross-attention reads all encoded English positions.');
    var targetCount = f.prefix.length, sourceCount = f.source.length;
    function grid(center, rows, columns, title, subtitle, causal, role) {
      c.text(center, 24, title, role, 22);
      c.text(center, 53, rows + ' × ' + columns, 'muted', 18);
      var cell = Math.min(44, 185 / Math.max(rows, columns)), left = center - columns * cell / 2, top = 93;
      for (var i = 0; i < rows; i++) {
        c.text(left - 19, top + (i + .5) * cell, String(i + 1), 'muted', 16);
        for (var j = 0; j < columns; j++) {
          var blocked = causal && j > i;
          c.svg.appendChild(sv('rect', { x: left + j * cell + 2, y: top + i * cell + 2,
            width: cell - 4, height: cell - 4, rx: 3, fill: blocked ? 'var(--paper)' : 'var(--t-' + role + ')',
            stroke: blocked ? 'var(--line)' : palette[role], 'data-allowed': String(!blocked),
            'data-mask-row': i, 'data-mask-column': j, 'data-mask-kind': title }));
          c.text(left + (j + .5) * cell, top + (i + .5) * cell, blocked ? '×' : '•', blocked ? 'muted' : role, 21);
        }
      }
      for (var j = 0; j < columns; j++) c.text(left + (j + .5) * cell, top - 14, String(j + 1), 'muted', 16);
      c.text(center, 288, subtitle, 'muted', 18);
    }
    grid(180, sourceCount, sourceCount, 'Encoder self-attention', 'All English positions', false, 'e');
    grid(550, targetCount, targetCount, 'Decoder self-attention', 'Current and earlier French', true, 'q');
    grid(920, targetCount, sourceCount, 'Cross-attention', 'All encoded English positions', false, 'a');
    c.text(550, 323, 'Rows ask. Columns provide keys and values. A cross-attention map need not be square.', 'muted', 18);
    return c.svg;
  }
  function provenanceDiagram(f, options) {
    var i = options.index == null ? f.prefix.length - 1 : options.index;
    if (!Number.isInteger(i) || i < 0 || i >= f.prefix.length) throw new Error('Invalid decoder row index.');
    var c = canvas(310, 'Where cross-attention queries, keys and values come from', 'The selected French row after masked self-attention supplies the query. Every encoded English row supplies a separate key and value.');
    c.box(190, 67, 340, 70, 'French row after masked self', f.prefix[i] + ' at position ' + (i + 1) + ' · 1 × 3', 'e');
    c.box(190, 212, 340, 70, 'Encoded English rows', 'E_enc · ' + f.source.length + ' × 3', 'e');
    c.arrow('M365 67 L690 67', 'q'); c.text(523, 43, 'W_Q, cross', 'q', 19);
    c.arrow('M365 212 L530 212 L530 167 L690 167', 'k'); c.text(590, 141, 'W_K, cross', 'k', 19);
    c.arrow('M365 212 L530 212 L530 260 L690 260', 'v'); c.text(590, 285, 'W_V, cross', 'v', 19);
    c.box(885, 67, 340, 68, 'Query q_cross,' + (i + 1), '1 × 3 · asks for source information', 'q');
    c.box(885, 167, 340, 68, 'Keys K_cross', f.source.length + ' × 3 · used to compute matches', 'k');
    c.box(885, 260, 340, 68, 'Values V_cross', f.source.length + ' × 3 · information to mix', 'v');
    return c.svg;
  }
  function graphDiagram(f, stage) {
    var names = ['encoder', 'decoder-self', 'cross', 'head', 'training'];
    var level = names.indexOf(stage);
    var trainingGraph = stage === 'training';
    var c = canvas(level === 0 ? 180 : trainingGraph ? 350 : 330, 'Encoder-decoder calculation: ' + stage,
      'English embeddings plus positions feed bidirectional encoder self-attention and a residual sum. French embeddings plus positions feed masked self-attention and a residual sum. Cross-attention takes queries from those French rows and keys and values from the English encoder. Its residual output goes to vocabulary logits. The numerical toy omits FFNs and LayerNorm.');
    c.text(20, 19, 'English: all source tokens are known', 'e', 19, 'start');
    c.box(105, 78, 180, 66, 'Source tokens', f.source.join(' '), 'e');
    c.box(310, 78, 180, 66, 'E_source', trainingGraph ? 'E_tok,src + P_src' : 'lookup + position · ' + f.source.length + ' × 3', 'e');
    c.box(520, 78, 190, 66, 'Encoder self-attn', trainingGraph ? 'Q/K/V/O + residual' : 'bidirectional + residual', 'a');
    c.box(752, 78, 210, 66, 'Encoded source', 'E_enc · ' + f.source.length + ' × 3', 'e');
    c.arrow('M197 78 L217 78'); c.arrow('M402 78 L422 78'); c.arrow('M617 78 L644 78', 'e');
    if (level >= 1) {
      c.text(20, 151, 'French: only the known prefix is available', 'q', 19, 'start');
      c.box(105, 219, 180, 66, 'Target prefix', f.prefix.join(' '), 'q');
      c.box(310, 219, 180, 66, 'E_target', trainingGraph ? 'E_tok,tgt + P_tgt' : 'lookup + position · ' + f.prefix.length + ' × 3', 'e');
      c.box(520, 219, 190, 66, 'Decoder self-attn', trainingGraph ? 'Q/K/V/O + residual' : 'causal + residual', 'a');
      c.arrow('M197 219 L217 219'); c.arrow('M402 219 L422 219');
    }
    if (level >= 2) {
      c.box(752, 219, 210, 66, 'Cross-attention', trainingGraph ? 'Q/K/V/O + residual' : 'source mixture + residual', 'd');
      c.arrow('M617 219 L644 219', 'q'); c.text(632, 194, 'Q', 'q', 17);
      c.arrow('M752 113 L752 183', 'muted');
      c.text(776, 149, 'K', 'k', 19); c.text(790, 149, ',', 'muted', 19); c.text(805, 149, 'V', 'v', 19);
    }
    if (level >= 3) {
      c.box(983, 219, 180, 66, 'Vocabulary head', trainingGraph ? 'W_vocab, b' : 'logits → probabilities', null);
      c.arrow('M859 219 L890 219', 'd');
      if (!trainingGraph) c.text(983, 279, 'Next: ' + f.last.chosen, 'ink', 20);
    }
    if (trainingGraph) {
      c.box(983, 78, 180, 66, 'Loss L', 'cross-entropy', null);
      c.text(983, 17, data.targets.join(' · '), 'ink', 16);
      c.arrow('M983 30 L983 43', 'ink');
      c.arrow('M983 183 L983 114', 'ink');
      c.svg.appendChild(sv('rect', { x: 20, y: 283, width: 882, height: 45, rx: 8,
        fill: 'var(--card)', stroke: 'var(--warn)', 'stroke-dasharray': '6 4' }));
      c.text(461, 299, 'Gradients reach both embedding tables, both position tables,', 'ink', 18);
      c.text(461, 319, 'all attention projection matrices, and the vocabulary head.', 'ink', 18);
      c.svg.appendChild(sv('path', { d: 'M1075 78 L1087 78 L1087 305 L906 305', fill: 'none',
        stroke: 'var(--warn)', 'stroke-width': 2.5, 'stroke-dasharray': '6 4',
        'marker-end': 'url(#' + c.svg.id + '-ink)' }));
      c.text(975, 285, 'autograd', 'ink', 18);
    } else c.text(550, level === 0 ? 157 : 316, 'Numerical toy: one head, output projections and residuals; no FFN or LayerNorm.', 'muted', 18);
    return c.svg;
  }
  function generationDiagram(options) {
    var g = generate(options), shown = Math.min(3, g.trace.length);
    if (options.step != null) shown = Math.min(shown, Math.max(1, options.step + 1));
    var prefix = options.prefixTokens || options.prefix;
    var visible = g.trace.slice(0, shown);
    if (prefix) {
      var match = g.trace.find(function (row) { return row.prefix.length === prefix.length && row.prefix.every(function (token, i) { return token === prefix[i]; }); });
      if (!match) throw new Error('The generation diagram prefix must match a prefix the model generated.');
      visible = [match];
    }
    var height = visible.length === 1 ? 170 : 320;
    var c = canvas(height, 'Generate the French translation one token at a time', 'Each row shows the actual greedy prediction for the current prefix. The source encoder runs once. The French prefix grows, so the final decoder query changes. Generation stops when EOS is predicted.');
    c.text(135, 24, 'Known prefix', 'e', 21); c.text(495, 24, 'Cross-attention query', 'q', 21);
    c.text(850, 24, 'Chosen next token', 'ink', 21);
    for (var i = 0; i < visible.length; i++) {
      var r = visible[i], y = 85 + i * 76;
      c.box(135, y, 235, 55, r.prefix.join(' '), null, 'e');
      c.arrow('M257 ' + y + ' L325 ' + y, 'q');
      c.box(495, y, 315, 55, '[' + r.query.map(function (v) { return v.toFixed(2); }).join(', ') + ']', null, 'q');
      c.arrow('M658 ' + y + ' L724 ' + y, 'd');
      c.box(850, y, 230, 55, r.chosen, 'p = ' + r.probabilities[r.chosenId].toFixed(3), 'd');
      if (i < visible.length - 1) c.arrow('M970 ' + y + ' L1050 ' + y + ' L1050 ' + (y + 48) + ' L25 ' + (y + 48) + ' L25 ' + (y + 49), 'muted', true);
    }
    var visibleLast = visible[visible.length - 1];
    var footer = visibleLast && visibleLast.chosen === '<eos>'
      ? 'The model predicted <eos>; stop. The English encoder was evaluated once.'
      : visibleLast && visibleLast.step < g.trace.length
        ? 'Append the chosen token and compute the next French query. Reuse the encoded English source.'
        : 'Stop reason: ' + g.stoppedBy + '. Each choice above came from the model.';
    c.text(550, height - 16, footer, 'muted', 18);
    return c.svg;
  }
  /* ---- The held whiteboard picture of Part IV ----
     English source tokens above, the French answer so far below. Every
     classroom frame draws this same picture and adds one mark per build:
     numbers under a token, arcs inside a row, the decoder query's arrows
     across to the English keys, scores on those arrows, softmax weights as
     arrow thickness, the message, the update, the prediction bars, the loss.
     board(host, options) -> svg. options:
       prefixTokens, snapshot, source, index (receiving French row; default last)
       slot:false            no dashed "?" box after the prefix
       srcLines / tgtLines   [{label, cls, rows:[vector|null per token] | text:[string per token],
                               build: n | [n per token], only:[token indices], decimals}]
       self                  [{row:'src'|'tgt', from:j, build, thick:build, numbers:build}]
       cross                 {from:i | [i..], build, thick:build,
                              labels:[{kind:'raw'|'scaled'|'weight'|'max', slot:0|1, build, role, decimals}]}
       bars                  {kind:'probs'|'logits', build, highlight:token, row:i}
       chosen                {build, token}        loss {build, target}
       backward              {build, lineQ, lineM, lineEp, lineK, lineV}
       receiver              build (thick outline on the receiving French row)
       dim                   'src' | 'tgt' (fade the row that is not in use)
     Numbers are read from the same forward pass as every table (2 decimals by default). */
  var B = { x0: 190, pitch: 220, boxW: 150, boxH: 46, slotW: 70, srcY: 76, tgtY: 300, srcLine0: 122, tgtLine0: 346,
    gap: 24, barsLabelX: 945, barsX: 955, barW: 80, barsY0: 277, barsGap: 27, barsTitleY: 250 };
  var boardPalette = { e: 'var(--c-e)', q: 'var(--c-q)', k: 'var(--c-k)', v: 'var(--c-v)', m: 'var(--c-v)', a: 'var(--c-a)',
    d: 'var(--c-d)', ep: 'var(--c-e)', ink: 'var(--ink)', ink2: 'var(--ink-2)', muted: 'var(--ink-3)', loss: 'var(--c-loss, #C62828)' };
  function boardDiagram(f, o) {
    o = o || {};
    var prefix = f.prefix, source = f.source, n = prefix.length;
    var i = o.index == null ? n - 1 : o.index;
    if (!Number.isInteger(i) || i < 0 || i >= n) throw new Error('Invalid receiving row index.');
    var srcLines = o.srcLines || [], tgtLines = o.tgtLines || [];
    var showSlot = o.slot !== false;
    function sx(j) { return B.x0 + B.pitch * j; }
    function tx(k) { return B.x0 + B.pitch * k; }
    var slotX = Math.min(tx(n), 830);
    function srcLineY(r) { return B.srcLine0 + B.gap * r; }
    function tgtLineY(r) { return B.tgtLine0 + B.gap * r; }
    var srcTop = B.srcY - B.boxH / 2, tgtTop = B.tgtY - B.boxH / 2;
    var srcBottom = srcLines.length ? srcLineY(srcLines.length - 1) + 12 : B.srcY + B.boxH / 2;
    var height = B.tgtY + B.boxH / 2 + 22;
    if (tgtLines.length) height = Math.max(height, tgtLineY(tgtLines.length - 1) + 22);
    if (o.bars) height = Math.max(height, B.barsY0 + B.barsGap * (targetVocab.length - 1) + (o.loss ? 56 : 26));
    var id = 'translation-board-' + (++serial);
    var svg = sv('svg', { id: id, viewBox: '0 0 1100 ' + height, role: 'img', 'class': 'p4-board',
      'aria-labelledby': id + '-title ' + id + '-desc', 'data-part4-diagram': 'board' });
    svg.appendChild(sv('title', { id: id + '-title' }, o.title || 'English source above, French answer below'));
    svg.appendChild(sv('desc', { id: id + '-desc' }, o.desc || ('The English source ' + source.join(' ') + ' is drawn as one row of tokens; the French prefix ' + prefix.join(' ') + ' below it. Marks added to the picture show numbers computed by the model.')));
    var defs = sv('defs');
    Object.keys(boardPalette).forEach(function (role) {
      var marker = sv('marker', { id: id + '-' + role, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 11, markerHeight: 11,
        markerUnits: 'userSpaceOnUse', orient: 'auto-start-reverse' });
      marker.appendChild(sv('path', { d: 'M0 0 L10 5 L0 10 Z', fill: boardPalette[role] }));
      defs.appendChild(marker);
    });
    svg.appendChild(defs);
    function grp(build, opacity) {
      var attrs = {};
      if (build) attrs['data-build'] = String(build);
      if (opacity != null) attrs.opacity = String(opacity);
      var g = sv('g', attrs); svg.appendChild(g); return g;
    }
    function txt(g, x, y, s, role, size, anchor, opts) {
      opts = opts || {};
      var attrs = { x: x, y: y, fill: boardPalette[role || 'ink'], 'font-size': size || 18, 'font-family': 'var(--font-ui)',
        'text-anchor': anchor || 'middle', 'dominant-baseline': 'middle', style: 'font-variant-numeric:tabular-nums' };
      if (opts.bold) attrs['font-weight'] = 700;
      if (opts.halo) { attrs['paint-order'] = 'stroke'; attrs.stroke = 'var(--paper)'; attrs['stroke-width'] = 7; attrs['stroke-linejoin'] = 'round'; }
      var t = sv('text', attrs, s); g.appendChild(t); return t;
    }
    function tokenBox(g, x, y, w, label, role, opts) {
      opts = opts || {};
      g.appendChild(sv('rect', { x: x - w / 2, y: y - B.boxH / 2, width: w, height: B.boxH, rx: 9,
        fill: opts.dashed ? 'var(--card)' : 'var(--t-' + role + ')', stroke: opts.dashed ? boardPalette.muted : boardPalette[role],
        'stroke-width': opts.strong ? 3.5 : 1.8, 'stroke-dasharray': opts.dashed ? '6 4' : 'none' }));
      txt(g, x, y + 1, label, opts.dashed ? 'muted' : role, 22, 'middle', { bold: !!opts.strong });
      if (opts.pos != null) txt(g, x - w / 2 + 12, y - B.boxH / 2 + 11, String(opts.pos), 'muted', 12);
    }
    function fmtRow(values, decimals) { return values.map(function (v) { return AT.fmt(v, decimals == null ? 2 : decimals); }).join('  '); }
    function drawLines(lines, xs, count, y0) {
      lines.forEach(function (line, r) {
        var y = y0 + B.gap * r;
        for (var k = 0; k < count; k++) {
          if (line.only && line.only.indexOf(k) < 0) continue;
          var value = line.text ? line.text[k] : (line.rows ? line.rows[k] : null);
          if (value == null) continue;
          var g = grp(Array.isArray(line.build) ? line.build[k] : line.build);
          var x = xs(k);
          if (line.label) txt(g, x - 98, y, line.label, line.cls || 'ink', 17, 'start', { bold: true });
          txt(g, x + 22, y, Array.isArray(value) ? fmtRow(value, line.decimals == null ? o.decimals : line.decimals) : String(value),
            line.valueCls || line.cls || 'ink', 18, 'middle', { bold: !!line.bold });
        }
      });
    }
    function arcPath(x1, x2, yTop) {
      if (x1 === x2) return 'M' + (x1 - 16) + ' ' + yTop + ' C ' + (x1 - 36) + ' ' + (yTop - 36) + ' ' + (x1 + 36) + ' ' + (yTop - 36) + ' ' + (x1 + 16) + ' ' + yTop;
      var h = 34 + Math.abs(x2 - x1) * 0.05;
      return 'M' + x1 + ' ' + yTop + ' Q ' + ((x1 + x2) / 2) + ' ' + (yTop - h) + ' ' + x2 + ' ' + yTop;
    }
    function arcPeak(x1, x2, yTop) {
      if (x1 === x2) return { x: x1, y: yTop - 27 };
      return { x: (x1 + x2) / 2, y: yTop - (34 + Math.abs(x2 - x1) * 0.05) / 2 };
    }
    function stroke(g, d, role, width, dashed) {
      g.appendChild(sv('path', { d: d, fill: 'none', stroke: boardPalette[role], 'stroke-width': width, 'stroke-linecap': 'round',
        'stroke-dasharray': dashed ? '7 5' : 'none', 'marker-end': 'url(#' + id + '-' + role + ')' }));
    }
    /* rows of tokens */
    var gSrc = grp(null, o.dim === 'src' ? 0.35 : null), gTgt = grp(null, o.dim === 'tgt' ? 0.35 : null);
    txt(gSrc, 20, B.srcY - 10, 'English', 'e', 19, 'start', { bold: true });
    txt(gSrc, 20, B.srcY + 11, 'source', 'muted', 15, 'start');
    source.forEach(function (t, j) { tokenBox(gSrc, sx(j), B.srcY, B.boxW, t, 'e', { pos: j + 1 }); });
    txt(gTgt, 20, B.tgtY - 10, 'French', 'q', 19, 'start', { bold: true });
    txt(gTgt, 20, B.tgtY + 11, 'so far', 'muted', 15, 'start');
    prefix.forEach(function (t, k) { tokenBox(gTgt, tx(k), B.tgtY, B.boxW, t, 'q', { pos: k + 1 }); });
    if (showSlot) tokenBox(gTgt, slotX, B.tgtY, B.slotW, '?', 'q', { dashed: true });
    if (o.receiver != null) {
      var gR = grp(o.receiver);
      gR.appendChild(sv('rect', { x: tx(i) - B.boxW / 2, y: tgtTop, width: B.boxW, height: B.boxH, rx: 9, fill: 'none', stroke: boardPalette.q, 'stroke-width': 4 }));
    }
    /* arcs inside a row (self-attention) */
    (o.self || []).forEach(function (spec) {
      var src = spec.row === 'src', A = src ? f.encoder.A : f.decoderSelf.A, xs = src ? sx : tx, yTop = src ? srcTop : tgtTop;
      var from = spec.from == null ? i : spec.from, count = src ? source.length : n;
      var gThin = grp(spec.build), gThick = spec.thick != null ? grp(spec.thick) : null, gNum = spec.numbers != null ? grp(spec.numbers) : null;
      for (var k = 0; k < count; k++) {
        if (!src && k > from) continue;
        var d = arcPath(xs(from), xs(k), yTop), w = A[from][k];
        stroke(gThin, d, 'muted', 2);
        if (gThick) stroke(gThick, d, 'a', 1.5 + 10 * w);
        if (gNum) { var p = arcPeak(xs(from), xs(k), yTop); txt(gNum, p.x, p.y - 12, AT.fmt(w, 2), 'a', 18, 'middle', { halo: true, bold: true }); }
      }
    });
    /* the decoder query's arrows across to the English keys */
    if (o.cross) {
      var spec = o.cross, froms = Array.isArray(spec.from) ? spec.from : [spec.from == null ? i : spec.from];
      var gThinX = grp(spec.build), gThickX = spec.thick != null ? grp(spec.thick) : null;
      froms.forEach(function (fi) {
        source.forEach(function (_, j) {
          var d = 'M' + tx(fi) + ' ' + tgtTop + ' L' + sx(j) + ' ' + srcBottom;
          stroke(gThinX, d, 'muted', 2);
          if (gThickX) stroke(gThickX, d, 'a', 1.5 + 10 * f.cross.A[fi][j]);
        });
      });
      (spec.labels || []).forEach(function (lab) {
        var g = grp(lab.build);
        froms.forEach(function (fi) {
          var row = lab.kind === 'raw' ? f.cross.Sraw[fi] : lab.kind === 'scaled' ? f.cross.S[fi] : f.cross.A[fi];
          var best = argmax(f.cross.A[fi]);
          source.forEach(function (_, j) {
            if (lab.kind === 'max' && j !== best) return;
            var t = lab.slot === 1 ? 0.36 : 0.14;
            var x = sx(j) + t * (tx(fi) - sx(j)), y = srcBottom + t * (tgtTop - srcBottom);
            var score = lab.kind === 'raw' || lab.kind === 'scaled';
            txt(g, x, y, AT.fmt(row[j], lab.decimals == null ? 2 : lab.decimals), lab.role || (score ? 'ink' : 'a'), 18, 'middle', { halo: true, bold: !score });
          });
        });
      });
    }
    drawLines(srcLines, sx, source.length, B.srcLine0);
    drawLines(tgtLines, tx, n, B.tgtLine0);
    /* the prediction for the "?" slot */
    if (o.bars) {
      var bs = o.bars, gB = grp(bs.build), row = bs.row == null ? i : bs.row;
      var values = bs.kind === 'logits' ? f.logits[row] : f.probs[row];
      txt(gB, 1020, B.barsTitleY, bs.title || (bs.kind === 'logits' ? 'next token: logit' : 'next token: p'), 'ink2', 16, 'middle');
      targetVocab.forEach(function (tok, r) {
        var y = B.barsY0 + B.barsGap * r, hl = tok === bs.highlight;
        txt(gB, B.barsLabelX, y, tok, hl ? 'ink' : 'muted', 18, 'end', { bold: hl });
        if (bs.kind === 'logits') txt(gB, B.barsX + 4, y, AT.fmt(values[r], 2), hl ? 'ink' : 'ink2', 18, 'start', { bold: hl });
        else {
          gB.appendChild(sv('rect', { x: B.barsX, y: y - 8, width: B.barW, height: 16, rx: 4, fill: 'var(--t-neutral)' }));
          gB.appendChild(sv('rect', { x: B.barsX, y: y - 8, width: Math.max(1.5, B.barW * values[r]), height: 16, rx: 4, fill: hl ? 'var(--ink)' : 'var(--ink-3)' }));
          txt(gB, B.barsX + B.barW + 8, y, AT.fmt(values[r], 3), hl ? 'ink' : 'ink2', 18, 'start', { bold: hl });
        }
      });
    }
    var lossY = B.barsY0 + B.barsGap * targetVocab.length + 8;
    if (o.loss) {
      var gL = grp(o.loss.build), tid = targetIndex[o.loss.target], pT = f.probs[i][tid];
      if (tid == null) throw new Error('Unknown loss target: ' + o.loss.target);
      txt(gL, 1090, lossY, 'loss = −log ' + AT.fmt(pT, 3) + ' = ' + AT.fmt(-Math.log(pT), 3), 'loss', 18, 'end', { bold: true });
    }
    if (o.chosen) {
      var gC = grp(o.chosen.build), w = Math.max(B.slotW, o.chosen.token.length * 13 + 24);
      tokenBox(gC, slotX, B.tgtY, w, o.chosen.token, 'q', { strong: true });
    }
    /* red = the gradient: autograd retraces the picture backward */
    if (o.backward) {
      var bw = o.backward, gW = grp(bw.build), xi = tx(i), best = argmax(f.cross.A[i]);
      var yEp = tgtLineY(bw.lineEp == null ? tgtLines.length - 1 : bw.lineEp), yM = tgtLineY(bw.lineM == null ? 1 : bw.lineM), yQ = tgtLineY(bw.lineQ == null ? 0 : bw.lineQ);
      var yK = srcLineY(bw.lineK == null ? 0 : bw.lineK), yV = srcLineY(bw.lineV == null ? 1 : bw.lineV);
      stroke(gW, 'M 845 ' + (lossY - 4) + ' L ' + (xi + 120) + ' ' + (yEp + 2), 'loss', 2.4, true);
      stroke(gW, 'M ' + (xi - 114) + ' ' + (yEp + 4) + ' L ' + (xi - 114) + ' ' + (yM - 4), 'loss', 2.4, true);
      stroke(gW, 'M ' + (xi - 60) + ' ' + (yM - 10) + ' L ' + (sx(best) + 40) + ' ' + (yV + 12), 'loss', 2.4, true);
      var t0 = 0.14, ax = sx(best) + t0 * (xi - sx(best)) + 24, ay = srcBottom + t0 * (tgtTop - srcBottom);
      stroke(gW, 'M ' + (xi - 40) + ' ' + (yM - 10) + ' L ' + (ax + 6) + ' ' + (ay + 14), 'loss', 2.4, true);
      stroke(gW, 'M ' + (ax + 10) + ' ' + (ay + 16) + ' L ' + (xi + 60) + ' ' + (yQ - 10), 'loss', 2.4, true);
      stroke(gW, 'M ' + (ax + 10) + ' ' + (ay - 12) + ' L ' + (sx(best) + 60) + ' ' + (yK + 12), 'loss', 2.4, true);
    }
    return svg;
  }
  function diagram(host, stage, options) {
    options = options || {};
    if (typeof host === 'string') host = document.getElementById(host);
    if (!host || typeof host.appendChild !== 'function') throw new Error('A diagram needs a host element.');
    var prefix = options.prefixTokens || options.prefix || data.targetInput;
    var f = forward(prefix, options), svg;
    if (stage === 'tokens') svg = tokensDiagram(f);
    else if (stage === 'masks') svg = masksDiagram(f);
    else if (stage === 'provenance') svg = provenanceDiagram(f, options);
    else if (stage === 'generation') svg = generationDiagram(options);
    else if (['encoder', 'decoder-self', 'cross', 'head', 'training'].indexOf(stage) >= 0) svg = graphDiagram(f, stage);
    else throw new Error('Unknown translation diagram stage: ' + stage);
    while (host.firstChild) host.removeChild(host.firstChild);
    host.appendChild(svg);
    return svg;
  }
  AT.translation = {
    config: { d_model: D, d_k: D, d_v: D, maxSource: data.maxSource, maxTarget: data.maxTarget,
      defaultSnapshot: data.defaultSnapshot, omitted: ['FFN', 'LayerNorm', 'dropout'], heads: 1 },
    d_model: D, d_k: D, d_v: D,
    source: data.source.slice(), sourceContrast: data.sourceContrast.slice(),
    sourceVocab: sourceVocab.slice(), targetVocab: targetVocab.slice(), vocab: targetVocab.slice(),
    targetInput: data.targetInput.slice(), targets: data.targets.slice(), contrastTargets: data.contrastTargets.slice(),
    axes: { e: ['coordinate 1', 'coordinate 2', 'coordinate 3'],
      qk: ['matching 1', 'matching 2', 'matching 3'], v: ['message 1', 'message 2', 'message 3'] },
    snapshots: copy(data.snapshots), training: copy(data.training), update: copy(data.update),
    notes: data.notes.slice(), parameters: parameters, params: parameters, parameterRows: parameterRows,
    forward: forward, teacherForced: teacherForced, generate: generate, topk: topk, diagram: diagram,
    comparison: function () { return { before: teacherForced({ snapshot: 'before' }), after: teacherForced({ snapshot: 'after' }), learningRate: data.update.learningRate }; },
    _forwardWith: forwardWith
  };

  // The shell builds its legend and notation reference after this runtime.
  // Part IV has generic learned coordinates, not the named Part II axes.
  if (AT.axes) AT.axes.named = false;
  if (Array.isArray(AT.objects)) {
    var definitions = {
      e: ['current representation', 'e carries a current token row. English and French use separate rows and embedding tables.'],
      q: ['query', 'In cross-attention, q comes from a French decoder row after masked self-attention.'],
      k: ['key', 'In cross-attention, each encoded English row supplies a key used to compute a match.'],
      v: ['value', 'In cross-attention, each encoded English row supplies a value containing the information to mix.'],
      a: ['attention weight', 'Cross-attention weights say how much a French query reads from each English source row. Each row sums to 1.'],
      d: ['attention update', 'The weighted value mixture is projected by W_O to update the receiving French row.'],
      ep: ['updated representation', 'The receiving French row adds the cross-attention update through a residual connection.']
    };
    AT.objects.forEach(function (object) {
      if (definitions[object.cls]) {
        object.name = definitions[object.cls][0];
        object.def = definitions[object.cls][1];
        object.tip = definitions[object.cls][1];
      }
    });
  }
  if (Array.isArray(AT.notation)) {
    function notation(group, symbol, meaning, shape, dimensions) {
      AT.notation.push({ g: group, sym: symbol, mean: meaning, shape: shape,
        dims: function () { return dimensions || ''; }, parts: ['part4'] });
    }
    notation('token', '\\ve{e_j^{\\mathrm{src}}}', 'English token embedding plus its source position vector', '1\\times d_{\\mathrm{model}}', '1×3');
    notation('token', '\\vp{e_j^{\\mathrm{enc}}}', 'English row after bidirectional encoder self-attention and residual addition', '1\\times d_{\\mathrm{model}}', '1×3');
    notation('token', '\\ve{e_i^{\\mathrm{tgt}}}', 'French token embedding plus its target position vector', '1\\times d_{\\mathrm{model}}', '1×3');
    notation('token', '\\vp{e_i^{\\mathrm{self}}}', 'French row after causal decoder self-attention and residual addition', '1\\times d_{\\mathrm{model}}', '1×3');
    notation('token', '\\vq{q_i^{\\mathrm{cross}}}=\\vp{e_i^{\\mathrm{self}}}W_Q^{\\mathrm{cross}}', 'Query from the receiving French row', '1\\times d_k', '1×3');
    notation('token', '\\vk{k_j^{\\mathrm{cross}}}=\\vp{e_j^{\\mathrm{enc}}}W_K^{\\mathrm{cross}}', 'Key from encoded English row j, used to compute its match', '1\\times d_k', '1×3');
    notation('token', '\\vv{v_j^{\\mathrm{cross}}}=\\vp{e_j^{\\mathrm{enc}}}W_V^{\\mathrm{cross}}', 'Value from the same encoded English row j, used in the mixture', '1\\times d_v', '1×3');
    notation('token', '\\vd{\\Delta e_i^{\\mathrm{cross}}}=m_iW_O^{\\mathrm{cross}}', 'Project the weighted source-value message back to representation width', '1\\times d_{\\mathrm{model}}', '1×3');
    notation('token', '\\vp{e_i^{\\mathrm{cross}}}=\\vp{e_i^{\\mathrm{self}}}+\\vd{\\Delta e_i^{\\mathrm{cross}}}', 'French row after adding the cross-attention update', '1\\times d_{\\mathrm{model}}', '1×3');
    notation('matrix', '\\vp{E_{\\mathrm{enc}}}', 'One encoded row per English source position, including EOS', 'T_{\\mathrm{src}}\\times d_{\\mathrm{model}}', '4×3');
    notation('matrix', '\\vq{Q_{\\mathrm{cross}}}', 'Queries projected from all post-self-attention French rows', 'T_{\\mathrm{tgt}}\\times d_k', '3×3 in training');
    notation('matrix', '\\vk{K_{\\mathrm{cross}}},\\;\\vv{V_{\\mathrm{cross}}}', 'Keys and values projected separately from encoded English rows', 'T_{\\mathrm{src}}\\times d_k,\\;T_{\\mathrm{src}}\\times d_v', '4×3 each');
    notation('matrix', 'S=\\vq{Q_{\\mathrm{cross}}}\\vk{K_{\\mathrm{cross}}}^{\\top}/\\sqrt{d_k}', 'Scaled scores: French query rows by English source columns', 'T_{\\mathrm{tgt}}\\times T_{\\mathrm{src}}', '3×4 in training');
    notation('matrix', '\\va{A}=\\operatorname{softmax}_{\\mathrm{row}}(S)', 'Cross-attention weights; every source position is available here', 'T_{\\mathrm{tgt}}\\times T_{\\mathrm{src}}', '3×4 in training');
    notation('matrix', 'H=\\va{A}\\vv{V_{\\mathrm{cross}}}', 'One weighted source-value message per receiving French row', 'T_{\\mathrm{tgt}}\\times d_v', '3×3 in training');
    notation('matrix', '\\vd{\\Delta E}=HW_O^{\\mathrm{cross}}', 'Updates added to the post-self-attention French rows', 'T_{\\mathrm{tgt}}\\times d_{\\mathrm{model}}', '3×3 in training');
    notation('sizes', 'T_{\\mathrm{src}},\\;T_{\\mathrm{tgt}}', 'English source length; current French prefix length', '', '4; 3 in teacher-forced training, 1 then 2 then 3 during generation');
    notation('sizes', 'd_{\\mathrm{model}},\\;d_k,\\;d_v', 'Representation, matching, and value widths are all three in this toy', '', '3, 3, 3');
    notation('sizes', 'E_{\\mathrm{tok}}^{\\mathrm{src}},\\;E_{\\mathrm{tok}}^{\\mathrm{tgt}}', 'Separate learned token lookup tables, distinct from the current sequence rows', '|\\mathcal V_{\\mathrm{src}}|\\times d_{\\mathrm{model}},\\;|\\mathcal V_{\\mathrm{tgt}}|\\times d_{\\mathrm{model}}', '5×3 each');
    notation('sizes', 'P_{\\mathrm{src}},\\;P_{\\mathrm{tgt}}', 'Separate learned position tables; positions are added, not appended', '', '4×3 and 6×3');
    notation('sizes', 'W_Q,W_K,W_V,W_O', 'Each of the three attention blocks has its own four learned matrices', '3\\times3', '12 matrices');
    notation('sizes', 'W_{\\mathrm{vocab}},\\;b', 'French vocabulary projection and bias', 'd_{\\mathrm{model}}\\times|\\mathcal V_{\\mathrm{tgt}}|,\\;1\\times|\\mathcal V_{\\mathrm{tgt}}|', '3×5 and 1×5');
  }
})();

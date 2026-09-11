(function () {
  'use strict';

  var AT = window.AT;
  var M = window.__TOY__ || {};
  var NS = 'http://www.w3.org/2000/svg';
  var diagramCount = 0;

  function sv(tag, attrs, text) {
    var el = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (attrs[key] != null) el.setAttribute(key, String(attrs[key]));
    });
    if (text != null) el.textContent = String(text);
    return el;
  }

  function add(parent, tag, attrs, text) {
    var el = sv(tag, attrs, text);
    parent.appendChild(el);
    return el;
  }

  function clampStage(value, max) {
    value = Number(value);
    return Number.isFinite(value) ? Math.max(0, Math.min(max, Math.floor(value))) : max;
  }

  function fmt(value) {
    value = Number(value) || 0;
    return (Math.abs(value) < 0.005 ? 0 : value).toFixed(2).replace('-', '\u2212');
  }

  function exact(value) {
    return Number(value).toString();
  }

  function arrowMarker(defs, id, color) {
    var marker = add(defs, 'marker', {
      id: id, viewBox: '0 0 10 10', refX: 9, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse'
    });
    add(marker, 'path', { d: 'M0 0 L10 5 L0 10 z', fill: color });
  }

  function baseSvg(name, title, description, width, height) {
    var id = 'at-part1-' + name + '-' + (++diagramCount);
    var svg = sv('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      role: 'img', 'aria-labelledby': id + '-title ' + id + '-desc',
      preserveAspectRatio: 'xMidYMid meet', 'data-part1-diagram': name,
      width: '100%', style: 'height:auto', focusable: 'false'
    });
    add(svg, 'title', { id: id + '-title' }, title);
    add(svg, 'desc', { id: id + '-desc' }, description);
    var defs = add(svg, 'defs');
    arrowMarker(defs, id + '-arrow', '#4A5160');
    arrowMarker(defs, id + '-blue-arrow', '#2563EB');
    arrowMarker(defs, id + '-grad-arrow', '#B45309');
    var css = [
      '.p1d{font-family:var(--font-ui,"Avenir Next","Segoe UI",sans-serif);color:var(--ink,#14171F)}',
      '.p1d text{fill:var(--ink,#14171F)}',
      '.p1d .main{font-size:22px;font-weight:700}',
      '.p1d .label{font-size:22px;font-weight:650}',
      '.p1d .small{font-size:18px;fill:var(--ink-2,#4A5160)}',
      '.p1d .mono{font-family:var(--font-mono,"SF Mono",Menlo,monospace);font-variant-numeric:tabular-nums}',
      '.p1d .box{fill:var(--card,#fff);stroke:var(--line,#D9DDE5);stroke-width:2}',
      '.p1d .act{fill:var(--t-e,#E4ECFF);stroke:var(--c-e,#2563EB);stroke-width:2}',
      '.p1d .param{fill:var(--t-neutral,#EEF0F4);stroke:var(--ink-3,#8A91A0);stroke-width:2;stroke-dasharray:8 6}',
      '.p1d .edge{fill:none;stroke:var(--ink-2,#4A5160);stroke-width:2.4;marker-end:url(#' + id + '-arrow)}',
      '.p1d .blue-edge{fill:none;stroke:var(--c-e,#2563EB);stroke-width:2.5;marker-end:url(#' + id + '-blue-arrow)}',
      '.p1d .grad-edge{fill:none;stroke:var(--warn,#B45309);stroke-width:2.5;stroke-dasharray:8 6;marker-end:url(#' + id + '-grad-arrow)}',
      '.p1d .guide{fill:none;stroke:var(--grid,#E3E6EC);stroke-width:1.5}',
      '.p1d .faint{fill:var(--ink-3,#8A91A0)}',
      '.p1d .blue{fill:var(--c-e,#2563EB)}'
    ].join('').replace(/\.p1d/g, '#' + id);
    add(svg, 'style', {}, css);
    svg.setAttribute('class', 'p1d');
    svg.setAttribute('id', id);
    return { svg: svg, defs: defs, id: id };
  }

  function box(g, x, y, w, h, cls, radius) {
    return add(g, 'rect', { x: x, y: y, width: w, height: h, rx: radius == null ? 10 : radius, class: cls || 'box' });
  }

  function text(g, x, y, value, cls, anchor) {
    return add(g, 'text', {
      x: x, y: y, class: cls || 'small', 'text-anchor': anchor || 'middle',
      'dominant-baseline': 'middle'
    }, value);
  }

  function line(g, x1, y1, x2, y2, cls, extra) {
    var attrs = { x1: x1, y1: y1, x2: x2, y2: y2, class: cls || 'guide' };
    Object.keys(extra || {}).forEach(function (key) { attrs[key] = extra[key]; });
    return add(g, 'line', attrs);
  }

  function path(g, d, cls, extra) {
    var attrs = { d: d, class: cls || 'edge' };
    Object.keys(extra || {}).forEach(function (key) { attrs[key] = extra[key]; });
    return add(g, 'path', attrs);
  }

  function tokenId(token) {
    var vocab = Array.isArray(M.vocab) ? M.vocab : [];
    return vocab.indexOf(token);
  }

  function embeddingSpace(options) {
    options = options || {};
    var highlight = Array.isArray(options.highlight) ? options.highlight : ['a', 'b', 'i'];
    var focus = {};
    highlight.forEach(function (token) { focus[token] = true; });
    var rows = Array.isArray(M.E) ? M.E : [];
    var vocab = Array.isArray(M.vocab) ? M.vocab : [];
    var b = baseSvg(
      'embedding-space', 'The trained two-dimensional character embedding table',
      'All 27 learned embedding rows plotted at their actual coordinates. The horizontal axis was sign-constrained positive for vowels and negative for consonants. The boundary token is near zero and has no such sign constraint. The vertical axis was learned without a semantic name.',
      1100, 420
    );
    var svg = b.svg;
    if (!rows.length) {
      text(svg, 550, 210, 'Embedding data unavailable', 'main');
      return svg;
    }

    var left = 116, top = 35, plotW = 900, plotH = 300;
    var xs = rows.map(function (row) { return Number(row[0]) || 0; });
    var ys = rows.map(function (row) { return Number(row[1]) || 0; });
    var minX = Math.min.apply(null, xs) - 0.22, maxX = Math.max.apply(null, xs) + 0.22;
    var minY = Math.min.apply(null, ys) - 0.25, maxY = Math.max.apply(null, ys) + 0.25;
    function px(value) { return left + (value - minX) / (maxX - minX) * plotW; }
    function py(value) { return top + plotH - (value - minY) / (maxY - minY) * plotH; }
    var zeroX = px(0);

    add(svg, 'rect', { x: left, y: top, width: Math.max(0, zeroX - left), height: plotH, fill: 'var(--t-neutral,#EEF0F4)', opacity: 0.55 });
    add(svg, 'rect', { x: zeroX, y: top, width: left + plotW - zeroX, height: plotH, fill: 'var(--t-e,#E4ECFF)', opacity: 0.5 });
    [-2, -1, 0, 1, 2].forEach(function (tick) {
      if (tick < minX || tick > maxX) return;
      line(svg, px(tick), top, px(tick), top + plotH, tick === 0 ? 'guide' : 'guide', tick === 0 ? { stroke: 'var(--ink-2,#4A5160)', 'stroke-width': 2.2 } : {});
      text(svg, px(tick), top + plotH + 23, tick, 'small mono');
    });
    [-2, -1, 0, 1, 2].forEach(function (tick) {
      if (tick < minY || tick > maxY) return;
      line(svg, left, py(tick), left + plotW, py(tick), 'guide');
      text(svg, left - 18, py(tick), tick, 'small mono', 'end');
    });
    line(svg, left, top + plotH, left + plotW, top + plotH, 'guide', { stroke: 'var(--ink,#14171F)', 'stroke-width': 2 });
    line(svg, left, top, left, top + plotH, 'guide', { stroke: 'var(--ink,#14171F)', 'stroke-width': 2 });

    text(svg, left + plotW / 2, 393, 'vowel-ness  (sign-constrained during training)', 'main');
    var yl = text(svg, 31, top + plotH / 2, 'learned axis 2', 'main');
    yl.setAttribute('transform', 'rotate(-90 31 ' + (top + plotH / 2) + ')');
    text(svg, left + 18, 56, 'consonants  <  0', 'small', 'start');
    text(svg, left + plotW - 18, 56, 'vowels  >  0', 'small blue', 'end');
    text(svg, zeroX + 17, 78, 'boundary \u201c-\u201d: no sign constraint', 'small', 'start');

    rows.forEach(function (row, i) {
      var token = vocab[i] == null ? String(i) : vocab[i];
      var isFocus = !!focus[token];
      var point = add(svg, 'circle', {
        cx: px(row[0]), cy: py(row[1]), r: isFocus ? 9 : 5.5,
        fill: isFocus ? 'var(--c-e,#2563EB)' : 'var(--ink-3,#8A91A0)',
        opacity: isFocus ? 1 : 0.48,
        stroke: isFocus ? 'var(--card,#fff)' : 'none', 'stroke-width': isFocus ? 3 : 0,
        'data-token': token, 'data-row-id': i, 'data-coordinates': JSON.stringify(row)
      });
      add(point, 'title', {}, token + ': [' + exact(row[0]) + ', ' + exact(row[1]) + ']');
      if (isFocus) {
        var above = token !== 'i';
        text(svg, px(row[0]) + (token === 'a' ? 8 : token === 'b' ? -8 : 10), py(row[1]) + (above ? -19 : 22), token, 'main blue', token === 'b' ? 'end' : 'start');
      }
    });
    return svg;
  }

  function lookupConcat(options) {
    options = options || {};
    var stage = clampStage(options.stage, 2);
    var tokens = ['a', 'a', 'b'];
    var ids = tokens.map(tokenId);
    var rows = AT.mlp.embed(ids);
    var concatenated = AT.mlp.concat(rows);
    var colors = ['#2563EB', '#497DD0', '#7296C8'];
    var dashes = ['', '9 6', '2 6'];
    var b = baseSvg(
      'lookup-concat', 'One lookup table supplies three ordered embedding occurrences',
      'The context a a b selects the same stored a row twice and the stored b row once, then concatenates the six actual coordinates in position order.',
      1100, 400
    );
    var svg = b.svg;
    svg.setAttribute('data-stage', String(stage));

    text(svg, 45, 30, 'context', 'small', 'start');
    tokens.forEach(function (token, i) {
      var x = 45 + i * 76;
      box(svg, x, 48, 66, 52, 'box');
      text(svg, x + 33, 74, token, 'main');
      text(svg, x + 33, 119, 'id ' + ids[i], 'small mono');
      text(svg, x + 33, 143, 'pos ' + (i + 1), 'small', 'middle').setAttribute('fill', colors[i]);
    });

    if (stage >= 1) {
      text(svg, 390, 31, 'one table  E_tok', 'main');
      box(svg, 280, 49, 220, 212, 'param');
      text(svg, 366, 77, 'axis 1', 'small');
      text(svg, 454, 77, 'axis 2', 'small');
      [['a', rows[0]], ['b', rows[2]]].forEach(function (entry, i) {
        var y = 102 + i * 84;
        line(svg, 290, y + 46, 490, y + 46, 'guide');
        text(svg, 304, y + 23, entry[0], 'main blue');
        text(svg, 366, y + 23, fmt(entry[1][0]), 'small mono');
        text(svg, 454, y + 23, fmt(entry[1][1]), 'small mono');
      });
      text(svg, 390, 241, 'one a row, two lookups', 'small', 'middle');

      tokens.forEach(function (token, i) {
        var gx = 545 + i * 182;
        text(svg, gx + 69, 74, 'position ' + (i + 1) + '  /  ' + token, 'small', 'middle').setAttribute('fill', colors[i]);
        rows[i].forEach(function (value, j) {
          add(svg, 'rect', { x: gx + j * 72, y: 96, width: 66, height: 54, rx: 8, fill: i === 0 ? '#E4ECFF' : i === 1 ? '#EBF1FC' : '#F0F4FA', stroke: colors[i], 'stroke-width': 2.2, 'stroke-dasharray': dashes[i] });
          text(svg, gx + j * 72 + 33, 123, fmt(value), 'small mono');
        });
        var sy = token === 'a' ? 125 : 209;
        var branch = i === 2
          ? 'M500 ' + sy + ' C545 260 1048 260 1051 123'
          : 'M500 ' + sy + ' C' + (517 + i * 36) + ' ' + sy + ' ' + (515 + i * 90) + ' 65 ' + (gx + 3) + ' 101';
        path(svg, branch, 'blue-edge', { stroke: colors[i], 'stroke-dasharray': dashes[i], 'marker-end': 'url(#' + b.id + '-blue-arrow)' });
      });
    }

    if (stage >= 2) {
      text(svg, 45, 319, 'a\u2080: concatenate', 'main', 'start');
      text(svg, 45, 349, 'position order, 1 \u00d7 6', 'small', 'start');
      concatenated.forEach(function (value, j) {
        var i = Math.floor(j / 2), x = 535 + j * 90;
        add(svg, 'rect', { x: x, y: 307, width: 86, height: 54, rx: 7, fill: i === 0 ? '#E4ECFF' : i === 1 ? '#EBF1FC' : '#F0F4FA', stroke: colors[i], 'stroke-width': 2.2, 'stroke-dasharray': dashes[i], 'data-concat-index': j, 'data-value': exact(value) });
        text(svg, x + 43, 334, fmt(value), 'label mono');
        text(svg, x + 43, 382, 'slot ' + (j + 1), 'small mono');
        var sourceX = 545 + i * 182 + (j % 2) * 72 + 33;
        path(svg, 'M' + sourceX + ' 151 C' + sourceX + ' 225 ' + (x + 43) + ' 235 ' + (x + 43) + ' 302', 'blue-edge', { stroke: colors[i], 'stroke-dasharray': dashes[i], 'marker-end': 'url(#' + b.id + '-blue-arrow)' });
      });
    }
    return svg;
  }

  function learningGraph(options) {
    options = options || {};
    var stage = clampStage(options.stage, 3);
    var ids = ['a', 'a', 'b'].map(tokenId);
    var target = tokenId('i');
    var result = AT.mlp.forward(ids);
    var probability = result.p[target] || 0;
    var loss = -Math.log(Math.max(probability, 1e-12));
    var b = baseSvg(
      'learning-graph', 'Forward prediction and reverse parameter learning for the trained character MLP',
      'The shared embedding table feeds a six-coordinate input through two learned affine layers. The observed i enters only at cross-entropy loss. At the last stage, gradients return to every shared parameter.',
      1100, 420
    );
    var svg = b.svg;
    svg.setAttribute('data-stage', String(stage));
    svg.setAttribute('data-target-probability', exact(probability));
    svg.setAttribute('data-loss', exact(loss));
    box(svg, 24, 177, 118, 84, 'box');
    text(svg, 83, 202, 'a  a  b', 'main');
    text(svg, 83, 239, 'IDs ' + ids.join(', '), 'small mono');
    box(svg, 62, 333, 154, 58, 'box');
    text(svg, 139, 352, 'observed: i', 'main');
    text(svg, 139, 380, 'target, not lookup', 'small');
    if (stage >= 1) {
      box(svg, 172, 174, 130, 90, 'param');
      text(svg, 237, 199, 'E_tok', 'main');
      text(svg, 237, 235, '27 \u00d7 2', 'small mono');
      text(svg, 237, 150, 'row a used twice', 'small');
      path(svg, 'M142 204 C154 204 157 204 170 204', 'blue-edge');
      path(svg, 'M142 238 C154 238 157 234 170 230', 'blue-edge', { 'stroke-dasharray': '8 6' });

      box(svg, 338, 181, 112, 76, 'act');
      text(svg, 394, 205, 'a\u2080', 'main blue');
      text(svg, 394, 236, '1 \u00d7 6', 'small mono');
      path(svg, 'M302 219 L334 219', 'blue-edge');

      box(svg, 480, 177, 148, 84, 'act');
      text(svg, 554, 201, 'affine \u2192 ReLU', 'label');
      text(svg, 554, 238, 'a\u2081 : 1 \u00d7 32', 'small mono');
      box(svg, 477, 56, 154, 68, 'param');
      text(svg, 554, 80, 'W\u2081  6 \u00d7 32', 'label mono');
      text(svg, 554, 108, 'b\u2081  1 \u00d7 32', 'small mono');
      path(svg, 'M450 219 L476 219', 'blue-edge');
      path(svg, 'M554 124 L554 172', 'edge');

      box(svg, 674, 177, 158, 84, 'act');
      text(svg, 753, 201, 'logits \u2192 probs', 'label');
      text(svg, 753, 238, '1 \u00d7 27', 'small mono');
      box(svg, 674, 56, 158, 68, 'param');
      text(svg, 753, 80, 'W\u2082  32 \u00d7 27', 'label mono');
      text(svg, 753, 108, 'b\u2082  1 \u00d7 27', 'small mono');
      path(svg, 'M628 219 L670 219', 'blue-edge');
      path(svg, 'M753 124 L753 172', 'edge');
    }

    if (stage >= 2) {
      box(svg, 886, 174, 184, 90, 'box');
      text(svg, 978, 194, 'cross-entropy', 'label');
      text(svg, 978, 225, 'p(i) = ' + probability.toFixed(3), 'small mono');
      text(svg, 978, 251, 'loss = ' + fmt(loss), 'main mono');
      path(svg, 'M832 219 L882 219', 'edge');
      path(svg, 'M216 362 C590 362 1080 362 1080 280 L1080 248 L1074 248', 'edge');
      text(svg, 510, 385, 'target i is not looked up as an input', 'small');
    }

    if (stage >= 3) {
      path(svg, 'M978 268 L978 312 L237 312 L237 268', 'grad-edge');
      path(svg, 'M648 312 L648 112 L634 112', 'grad-edge');
      path(svg, 'M853 312 L853 112 L836 112', 'grad-edge');
      text(svg, 670, 335, 'reverse gradients to shared parameters', 'small');
    }
    return svg;
  }

  function embeddingGradients() {
    var rows = M.E, vocab = M.vocab, input = ['a', 'a', 'b'];
    var b = baseSvg(
      'embedding-gradients', 'Only the looked-up embedding rows receive gradients from this example',
      'The complete 27 by 2 embedding table, displayed in three blocks with no omitted rows. Cross-entropy for the input a a b and target i sends two lookup-gradient contributions to the shared a row and one to b. All other embedding rows, including i, have zero gradient from this loss. Additional regularization terms are excluded.',
      1100, 420
    );
    var svg = b.svg;
    svg.setAttribute('data-stage', '0');
    svg.setAttribute('data-gradient-scope', 'single-example-cross-entropy');
    add(svg, 'style', {}, '#' + b.id + ' .lookup-gradient{fill:var(--t-e,#E4ECFF);stroke:var(--warn,#B45309);stroke-width:2}');
    text(svg, 18, 25, 'E_tok: ' + rows.length + ' \u00d7 ' + rows[0].length, 'main blue', 'start');
    text(svg, 18, 52, 'One table, split into three blocks.', 'small', 'start');
    var perBlock = Math.ceil(rows.length / 3), blockWidth = 252, rowHeight = 28;
    for (var block = 0; block < 3; block++) {
      var left = 18 + block * blockWidth;
      text(svg, left + 16, 80, 'ID', 'small');
      text(svg, left + 66, 80, 'token', 'small');
      text(svg, left + 132, 80, 'coord 1', 'small');
      text(svg, left + 208, 80, 'coord 2', 'small');
      line(svg, left, 93, left + 238, 93);
      for (var n = 0; n < perBlock; n++) {
        var id = block * perBlock + n;
        if (id >= rows.length) break;
        var count = input.filter(function (token) { return token === vocab[id]; }).length;
        var top = 96 + n * rowHeight, center = top + rowHeight / 2;
        var g = add(svg, 'g', {
          'data-embedding-row': id, 'data-token': vocab[id],
          'data-coordinates': JSON.stringify(rows[id]), 'data-lookup-contributions': count
        });
        add(g, 'title', {}, vocab[id] + ': ' + JSON.stringify(rows[id]) + '; ' + count + ' lookup-gradient contributions');
        if (count) box(g, left, top, 238, rowHeight, 'lookup-gradient', 3);
        else line(g, left, top + rowHeight, left + 238, top + rowHeight);
        text(g, left + 16, center, String(id), 'small mono');
        text(g, left + 66, center, vocab[id], count ? 'main blue mono' : 'label mono');
        rows[id].forEach(function (value, axis) {
          var node = text(g, left + 132 + axis * 76, center, fmt(value), 'label blue mono');
          node.setAttribute('data-coordinate', String(axis));
        });
      }
    }
    text(svg, 792, 25, 'One example: a a b \u2192 i', 'main', 'start');
    text(svg, 792, 98, 'Row a: two contributions', 'label', 'start');
    text(svg, 792, 127, 'One from each a position.', 'small', 'start');
    text(svg, 792, 154, 'Autograd adds them.', 'small', 'start');
    text(svg, 792, 203, 'Row b: one contribution', 'label', 'start');
    text(svg, 792, 253, 'Other 25 rows: zero gradient.', 'small', 'start');
    text(svg, 792, 280, 'That includes row i.', 'small', 'start');
    text(svg, 792, 307, 'i labels the correct output.', 'small', 'start');
    text(svg, 18, 375, 'Outlined rows receive the embedding gradient from this loss.', 'small', 'start');
    text(svg, 18, 403, 'Cross-entropy for this example only. Extra regularization can affect other rows.', 'small', 'start');
    return svg;
  }

  function trainingVsGeneration(options) {
    options = options || {};
    var stage = clampStage(options.stage, 2);
    var ids = ['a', 'a', 'b'].map(tokenId);
    var result = AT.mlp.forward(ids);
    var target = tokenId('i');
    var sampleSeed = 1;
    var sample = AT.mlp.generate({ ids: ['a', 'a', 'b'], seed: sampleSeed, temperature: 1, maxLength: 1, greedy: false });
    var chosen = sample.trace[0].chosen;
    var nextContext = sample.trace[0].next_context;
    var pTarget = result.p[target] || 0;
    var b = baseSvg(
      'training-vs-generation', 'Training and generation reuse the same learned character MLP',
      'Two views of the same MLP architecture. Training compares the distribution for a a b with observed i. Autograd computes gradients and the optimizer updates the embedding table, both weights and both biases. In generation, a lock and five snowflakes mark those parameters as frozen. Activations and probabilities are recomputed for each input. A real reproducible sample at temperature 1 with seed ' + sampleSeed + ' draws ' + chosen + ', then shifts the context to ' + nextContext.join(' ') + '. The boundary token would instead stop generation.',
      1100, 420
    );
    var svg = b.svg;
    svg.setAttribute('data-stage', String(stage));
    svg.setAttribute('data-sample-seed', String(sampleSeed));
    svg.setAttribute('data-sample-temperature', '1');
    svg.setAttribute('data-sample-token', chosen);
    svg.setAttribute('data-next-context', nextContext.join(' '));
    add(svg, 'style', {}, [
      '#' + b.id + ' .parameter-name{font-size:20px;fill:var(--c-q);font-family:var(--font-mono)}',
      '#' + b.id + ' .parameter-icon{fill:none;stroke:var(--c-e);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}',
      '#' + b.id + ' .state-heading{font-size:20px;font-weight:700}',
      '#' + b.id + ' .trainable-heading{fill:var(--warn)}',
      '#' + b.id + ' .frozen-heading{fill:var(--c-e)}'
    ].join(''));

    function parameters(y, frozen) {
      var group = add(svg, 'g', { 'data-parameter-state': frozen ? 'frozen' : 'trainable' });
      // Both rows name the same parameter types; icons belong only to parameters.
      [['E_tok', 'E_tok', 416, 0, 376], ['W1', 'W\u2081', 510, 0, 486], ['b1', 'b\u2081', 588, 0, 564],
        ['W2', 'W\u2082', 456, 28, 432], ['b2', 'b\u2082', 550, 28, 526]].forEach(function (item) {
        var label = text(group, item[2], y + item[3], item[1], 'parameter-name');
        label.setAttribute('data-parameter', item[0]);
        if (frozen) {
          var snow = add(group, 'g', { transform: 'translate(' + item[4] + ' ' + (y + item[3]) + ')', class: 'parameter-icon', 'data-frozen-symbol': item[0], 'aria-hidden': 'true' });
          // A small vector snowflake stays crisp in slides and PDF exports.
          for (var angle = 0; angle < 360; angle += 60) {
            add(snow, 'path', { d: 'M0 0 L0 -8 M-3 -5 L0 -3 L3 -5', transform: 'rotate(' + angle + ')' });
          }
        }
      });
      return group;
    }

    add(svg, 'rect', { x: 18, y: 26, width: 1064, height: 168, rx: 13, fill: 'var(--card,#fff)', stroke: 'var(--line,#D9DDE5)', 'stroke-width': 2 });
    if (stage >= 1) add(svg, 'rect', { x: 18, y: 202, width: 1064, height: 198, rx: 13, fill: 'var(--t-e,#E4ECFF)', opacity: 0.38, stroke: 'var(--line,#D9DDE5)', 'stroke-width': 2 });
    text(svg, 42, 53, 'TRAINING', 'small mono', 'start');
    box(svg, 56, 83, 152, 64, 'box');
    text(svg, 132, 104, 'context  a a b', 'label');
    text(svg, 132, 133, 'input only', 'small');

    box(svg, 344, 73, 284, 113, 'param');
    text(svg, 508, 96, 'TRAINABLE MLP', 'state-heading trainable-heading');
    var updateIcon = add(svg, 'g', { transform: 'translate(376 95)', 'data-update-symbol': 'optimizer', 'aria-hidden': 'true' });
    add(updateIcon, 'path', { d: 'M8 -3 A9 9 0 1 0 8 5 M3 -3 L8 -3 L8 -8', fill: 'none', stroke: 'var(--warn)', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
    parameters(130, false);

    box(svg, 660, 77, 168, 72, 'act');
    text(svg, 744, 100, 'distribution', 'label');
    text(svg, 744, 132, 'p(i) = ' + pTarget.toFixed(3), 'small mono');
    box(svg, 892, 77, 156, 72, 'box');
    text(svg, 970, 100, 'loss', 'main');
    text(svg, 970, 132, '\u2212log p(i)', 'small mono');
    box(svg, 748, 157, 126, 32, 'box', 7);
    text(svg, 811, 173, 'observed i', 'small');
    path(svg, 'M874 173 C910 173 949 173 949 152', 'edge');
    path(svg, 'M208 115 L340 115', 'blue-edge');
    path(svg, 'M628 115 L656 115', 'blue-edge');
    path(svg, 'M828 115 L888 115', 'edge');
    path(svg, 'M970 73 C970 31 653 31 618 76', 'grad-edge', { 'data-loop': 'parameter-update' });
    text(svg, 782, 22, 'gradients + optimizer update', 'small mono');

    if (stage >= 1) {
      text(svg, 42, 229, 'GENERATION', 'small mono', 'start');
      box(svg, 56, 259, 152, 64, 'box');
      text(svg, 132, 280, 'context  a a b', 'label');
      text(svg, 132, 309, 'no observed label', 'small');
      text(svg, 486, 212, 'same MLP operations', 'small');
      box(svg, 344, 239, 284, 100, 'param');
      text(svg, 508, 259, 'FROZEN MLP', 'state-heading frozen-heading');
      var lock = add(svg, 'g', { transform: 'translate(376 257)', class: 'parameter-icon', 'data-lock-symbol': 'parameters', 'aria-hidden': 'true' });
      add(lock, 'path', { d: 'M-6 -2 V-7 A6 6 0 0 1 6 -7 V-2' });
      add(lock, 'rect', { x: -9, y: -2, width: 18, height: 15, rx: 2 });
      add(lock, 'path', { d: 'M0 4 V8' });
      parameters(287, true);
      box(svg, 660, 255, 168, 72, 'act');
      text(svg, 744, 278, 'distribution', 'label');
      text(svg, 744, 310, 'recomputed p', 'small mono');
      box(svg, 877, 255, 178, 72, 'box');
      text(svg, 966, 278, 'sample  ' + chosen, 'main');
      text(svg, 966, 310, 'one possible draw', 'small');
      path(svg, 'M208 291 L340 291', 'blue-edge');
      path(svg, 'M628 291 L656 291', 'blue-edge');
      path(svg, 'M828 291 L873 291', 'edge');
      if (stage === 1) text(svg, 590, 378, 'All five parameter groups stay fixed during generation.', 'small');
    }

    if (stage >= 2) {
      path(svg, 'M966 331 C966 374 274 374 132 328', 'blue-edge', { 'data-loop': 'next-input' });
      text(svg, 590, 383, 'append ' + chosen + ', shift window \u2192 ' + nextContext.join(' '), 'main');
      text(svg, 590, 409, 'weights fixed during generation; boundary \u201c-\u201d would stop', 'small');
    }
    return svg;
  }

  // Add parameter labels to the existing editable MLP sketch. Node positions
  // come from netSketch so that labels remain tied to the actual columns.
  function annotateMLP(net) {
    var svg = net.svg;
    if (svg.classList.contains('p1-mlp-labelled')) return net;
    svg.classList.add('p1-mlp-labelled');
    var columns = ['in', 'hid', 'out'].map(function (name) {
      return svg.querySelector('.col-' + name + ' .node').transform.baseVal.getItem(0).matrix.e;
    });
    add(svg, 'style', {}, [
      '.p1-mlp-labelled .parameter-label{font-family:var(--font-mono);font-size:17px;fill:var(--c-q)}',
      '.p1-mlp-labelled .edges line{stroke:var(--c-q);opacity:.24}',
      '.p1-mlp-labelled .edges line.hl{stroke:var(--c-q);opacity:.85}',
      '.netsk .p1-mlp-labelled .col-hid circle{fill:var(--t-v);stroke:var(--c-v)}',
      '.p1-mlp-labelled .col-hid .unit-l{fill:var(--c-v)}',
      '.p1-mlp-labelled .lab-in-box{color:var(--c-e)}',
      '.p1-mlp-labelled .cap{text-transform:none;font-size:12px}',
      '.p1-mlp-labelled .cap:nth-child(1){fill:var(--c-e)}',
      '.p1-mlp-labelled .cap:nth-child(2){fill:var(--c-v)}'
    ].join(''));
    [
      ['W1', 'W₁ weights', (columns[0] + columns[1]) / 2, 48, '6 × 32 learned weights, one per input-to-hidden connection.'],
      ['W2', 'W₂ weights', (columns[1] + columns[2]) / 2, 48, '32 × 27 learned weights, one per hidden-to-output connection.'],
      ['b1', '+ b₁: 32 biases', columns[1], svg.viewBox.baseVal.height - 10, 'One learned bias added at each hidden unit, before ReLU.'],
      ['b2', '+ b₂: 27 biases', columns[2], svg.viewBox.baseVal.height - 10, 'One learned bias added at each output logit.']
    ].forEach(function (label) {
      var node = text(svg, label[2], label[3], label[1], 'parameter-label');
      node.setAttribute('data-parameter', label[0]);
      add(node, 'title', {}, label[4]);
    });
    svg.setAttribute('aria-label', svg.getAttribute('aria-label') + ' W1 and W2 are connection weights. b1 adds one bias per hidden unit before ReLU. b2 adds one bias per output logit.');
    return net;
  }

  // A complete, reproducible run of the saved model. Keep the random generator
  // in AT.mlp.generate as the single source of choices and probabilities.
  function generationExample() {
    return AT.mlp.generate({ seed: 3, temperature: 1, maxLength: 18 });
  }

  function generationRun(options) {
    options = options || {};
    var run = generationExample(), stage = clampStage(options.stage, run.trace.length * 3 - 1);
    var round = Math.floor(stage / 3), phase = stage % 3, t = run.trace[round];
    var before = run.trace.slice(0, round).map(function (r) { return r.chosen; }).join('');
    var stopped = t.chosen === '-', after = before + (stopped ? '' : t.chosen);
    var b = baseSvg('generation-run', 'Generation round ' + (round + 1),
      'Context ' + t.context.join(' ') + ' supplies the same trained model. ' +
      (phase ? 'Sampling chooses ' + (stopped ? 'the boundary and stops' : t.chosen) +
      ' with probability ' + t.probabilities[t.chosen_id].toFixed(3) + '.' : 'The chart shows next-character probabilities before a draw.') +
      (phase === 2 && !stopped ? ' Append the letter to the name and keep only the newest three tokens for the next call.' : ''), 1100, 410);
    var svg = b.svg;
    svg.setAttribute('data-stage', stage);
    svg.setAttribute('data-round', round);
    svg.setAttribute('data-phase', phase);
    svg.setAttribute('data-context', t.context.join(','));
    svg.setAttribute('data-generated', phase === 2 ? after : before);
    svg.setAttribute('data-stopped', String(phase === 2 && stopped));
    add(svg, 'style', {}, [
      '#' + b.id + ' .picked{fill:var(--warn,#B45309)}',
      '#' + b.id + ' .picked-box{fill:var(--t-warn,#FFF4E5);stroke:var(--warn,#B45309);stroke-width:2}',
      '#' + b.id + ' .token{font-size:30px;font-weight:700}',
      '@keyframes p1-generation-move{from{transform:translate(var(--from-x),var(--from-y));opacity:.35}to{transform:translate(0,0);opacity:1}}',
      '#' + b.id + ' .token-move{animation:p1-generation-move .75s ease-out both}',
      '@media(prefers-reduced-motion:reduce){#' + b.id + ' .token-move{animation:none}}',
      '@media print{#' + b.id + ' .token-move{animation:none}}'
    ].join(''));
    text(svg, 165, 24, 'Context for this prediction', 'label');
    t.context.forEach(function (token, j) {
      var x = 34 + j * 82;
      box(svg, x, 60, 66, 60, 'act');
      text(svg, x + 33, 90, token, 'mono token blue');
    });
    text(svg, 67, 141, 'oldest', 'small');
    text(svg, 231, 141, 'newest', 'small');
    line(svg, 276, 90, 334, 90, 'blue-edge');
    box(svg, 344, 60, 170, 60);
    text(svg, 429, 90, 'Same MLP', 'main');
    text(svg, 429, 141, 'Weights stay fixed', 'small');
    line(svg, 524, 90, 602, 90, 'edge');

    text(svg, 835, 24, 'Probabilities for ' + t.context.join(' '), 'label');
    // Keep the sampled token visible even when it lies outside the top five.
    var top = AT.topk(t.probabilities, 5);
    var sixth = top.some(function (r) { return r.i === t.chosen_id; })
      ? AT.topk(t.probabilities, 6)[5] : { tok: t.chosen, i: t.chosen_id, p: t.probabilities[t.chosen_id] };
    var shown = top.concat([sixth]).sort(function (a, c) { return c.p - a.p; });
    var rest = 1 - shown.reduce(function (sum, r) { return sum + r.p; }, 0);
    shown.forEach(function (r, j) {
      var y = 65 + j * 34, selected = phase > 0 && r.i === t.chosen_id;
      var row = add(svg, 'g', { 'data-prob-token': r.tok, 'data-probability': exact(r.p), 'data-selected': String(selected) });
      text(row, 642, y, r.tok === '-' ? 'END' : r.tok, 'mono label' + (selected ? ' picked' : ''), 'end');
      add(row, 'rect', { x: 660, y: y - 9, width: 310, height: 18, rx: 3, fill: 'var(--t-neutral,#EEF0F4)' });
      add(row, 'rect', { x: 660, y: y - 9, width: 310 * r.p / .5, height: 18, rx: 3,
        fill: selected ? 'var(--warn,#B45309)' : 'var(--c-e,#2563EB)' });
      text(row, 1074, y, r.p.toFixed(3), 'mono label' + (selected ? ' picked' : ''), 'end');
    });
    [0, .25, .5].forEach(function (p) { text(svg, 660 + 620 * p, 266, p.toFixed(2), 'small mono'); });
    text(svg, 835, 300, 'Other 21 characters: ' + rest.toFixed(3), 'small mono');
    text(svg, 835, 328, 'Total: 1 before rounding to 3 decimals.', 'small');
    text(svg, 18, 191, phase ? 'Sampled: ' + (stopped ? 'END (the “-” token)' : t.chosen) : 'Next: draw one of the 27 characters',
      phase ? 'label picked' : 'label', 'start');
    if (phase) text(svg, 18, 222, 'Probability of this choice: ' + t.probabilities[t.chosen_id].toFixed(3), 'small', 'start');
    if (phase === 2 && !stopped) {
      text(svg, 165, 256, 'Window for the next call', 'label');
      t.next_context.forEach(function (token, j) {
        var g = add(svg, 'g', { transform: 'translate(' + (34 + j * 82) + ',280)', 'data-next-token': token });
        var inner = add(g, 'g', { class: 'token-move', style: '--from-x:' + (j < 2 ? 82 : 140) + 'px;--from-y:' + (j < 2 ? -220 : -90) + 'px' });
        box(inner, 0, 0, 66, 60, j === 2 ? 'picked-box' : 'act');
        text(inner, 33, 30, token, 'mono token ' + (j === 2 ? 'picked' : 'blue'));
      });
      text(svg, 367, 282, 'Drop the oldest token.', 'small', 'start');
      text(svg, 367, 311, 'Keep two, append ' + t.chosen + '.', 'small', 'start');
    } else if (phase === 2) {
      text(svg, 18, 280, 'Stop. The name is complete.', 'main picked', 'start');
      text(svg, 18, 315, 'Do not append END or run another prediction.', 'small', 'start');
    }
    line(svg, 18, 358, 1080, 358);
    text(svg, 18, 389, 'Name so far: ' + ((phase === 2 ? after : before) || '(empty)'), 'main mono', 'start');
    text(svg, 1080, 389, 'Original probabilities, sampling seed 3', 'small', 'end');
    return svg;
  }

  function generationTrace() {
    var run = generationExample();
    var b = baseSvg('generation-trace', 'Every call in the sampled name sam',
      'Starting from three boundary tokens, sample s, a, m, and then the boundary. Append only letters. Each new window contains the newest three tokens. All probabilities come from the saved model.', 1100, 330);
    var svg = b.svg;
    svg.setAttribute('data-stage', 0);
    [[35, 'Call'], [160, 'Input window'], [425, 'Chosen'], [626, 'Probability'], [815, 'Name'], [985, 'Next window']].forEach(function (c) { text(svg, c[0], 26, c[1], 'label'); });
    var name = '';
    run.trace.forEach(function (t, j) {
      var y = 82 + j * 49;
      if (t.chosen !== '-') name += t.chosen;
      var row = add(svg, 'g', { 'data-trace-round': j, 'data-probability': exact(t.probabilities[t.chosen_id]) });
      line(row, 10, y - 25, 1090, y - 25);
      [[35, String(j + 1)], [160, t.context.join(' ')], [425, t.chosen === '-' ? 'END (−)' : t.chosen],
        [626, t.probabilities[t.chosen_id].toFixed(3)], [815, name], [985, t.chosen === '-' ? 'stop' : t.next_context.join(' ')]].forEach(function (c) {
        text(row, c[0], y, c[1], 'label mono');
      });
    });
    text(svg, 550, 291, 'Four model calls, three letters. The weights never change during generation.', 'label');
    return svg;
  }

  function generationChoice() {
    var run = generationExample(), t = run.trace[2], best = AT.argmax(t.probabilities);
    var b = baseSvg('generation-choice', 'Greedy and sampling from the same probability row',
      'At context minus s a, n has the largest probability. Greedy chooses n. The fixed sampling run draws m instead. Greedy completes san and this sample completes sam. Sampling can also choose n.', 1100, 230);
    var svg = b.svg;
    svg.setAttribute('data-stage', 0);
    text(svg, 550, 22, 'Same input window: − s a. Same model. Same probabilities.', 'label');
    text(svg, 260, 75, 'Greedy: choose the largest', 'main');
    text(svg, 840, 75, 'Sampling: draw using probabilities', 'main');
    text(svg, 260, 117, 'n   probability ' + t.probabilities[best].toFixed(3), 'label mono blue');
    text(svg, 840, 117, 'm   probability ' + t.probabilities[t.chosen_id].toFixed(3), 'label mono blue');
    text(svg, 260, 157, 'Next window: s a n', 'label mono');
    text(svg, 840, 157, 'Next window: s a m', 'label mono');
    text(svg, 260, 202, 'Greedy run finishes as san.', 'small');
    text(svg, 840, 202, 'Seed 3 finishes as sam. Another draw can differ.', 'small');
    line(svg, 550, 63, 550, 215);
    return svg;
  }

  function temperatureComparison() {
    var context = ['-', 's', 'a'], tokens = ['n', 'r', 'h', 'm', 'a'];
    var b = baseSvg('temperature-comparison', 'One prediction at three temperatures',
      'Keep the input minus s a and the model fixed. Temperature 0.5 concentrates more probability on the highest-scoring character n. Temperature 1 preserves the original probabilities. Temperature 1.5 makes the full distribution more even. Each chart uses the same probability scale and shows the same five characters out of 27.', 1100, 335);
    var svg = b.svg;
    svg.setAttribute('data-stage', 0);
    [.5, 1, 1.5].forEach(function (temperature, col) {
      var x = col * 370, p = AT.mlp.distribution(context, temperature).p;
      var panel = add(svg, 'g', { 'data-temperature': temperature, transform: 'translate(' + x + ',0)' });
      text(panel, 175, 24, 'Temperature ' + temperature.toFixed(1), 'main');
      text(panel, 175, 57, ['Lower: more concentrated', '1.0: original probabilities', 'Higher: more even'][col], 'small');
      tokens.forEach(function (token, j) {
        var prob = p[tokenId(token)], y = 98 + j * 36;
        var row = add(panel, 'g', { 'data-temperature-token': token, 'data-probability': exact(prob) });
        text(row, 22, y, token, 'label mono');
        add(row, 'rect', { x: 46, y: y - 9, width: 215, height: 18, rx: 3, fill: 'var(--t-neutral,#EEF0F4)' });
        add(row, 'rect', { x: 46, y: y - 9, width: 215 * prob / .4, height: 18, rx: 3, fill: 'var(--c-e,#2563EB)' });
        text(row, 345, y, prob.toFixed(3), 'label mono', 'end');
      });
      [0, .2, .4].forEach(function (p) { text(panel, 46 + 215 * p / .4, 271, p.toFixed(1), 'small mono'); });
      var rest = 1 - tokens.reduce(function (sum, t) { return sum + p[tokenId(t)]; }, 0);
      text(panel, 175, 308, 'Other 22: ' + rest.toFixed(3) + ' in total', 'small');
      if (col < 2) line(svg, x + 361, 10, x + 361, 320);
    });
    return svg;
  }

  AT.part1Diagrams = {
    annotateMLP: annotateMLP,
    embeddingSpace: embeddingSpace,
    lookupConcat: lookupConcat,
    learningGraph: learningGraph,
    embeddingGradients: embeddingGradients,
    trainingVsGeneration: trainingVsGeneration,
    generationExample: generationExample,
    generationRun: generationRun,
    generationTrace: generationTrace,
    generationChoice: generationChoice,
    temperatureComparison: temperatureComparison
  };
})();

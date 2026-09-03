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
      text(svg, 554, 201, 'affine \u2192 tanh', 'label');
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
      'Training compares the distribution for a a b with observed i and updates shared parameters. Generation holds those parameters fixed. A real reproducible sample at temperature 1 with seed 1 draws i, then shifts the context to a b i. The boundary token would instead stop generation.',
      1100, 420
    );
    var svg = b.svg;
    svg.setAttribute('data-stage', String(stage));
    svg.setAttribute('data-sample-seed', String(sampleSeed));
    svg.setAttribute('data-sample-temperature', '1');
    svg.setAttribute('data-sample-token', chosen);
    svg.setAttribute('data-next-context', nextContext.join(' '));

    add(svg, 'rect', { x: 18, y: 26, width: 1064, height: 168, rx: 13, fill: 'var(--card,#fff)', stroke: 'var(--line,#D9DDE5)', 'stroke-width': 2 });
    if (stage >= 1) add(svg, 'rect', { x: 18, y: 202, width: 1064, height: 198, rx: 13, fill: 'var(--t-e,#E4ECFF)', opacity: 0.38, stroke: 'var(--line,#D9DDE5)', 'stroke-width': 2 });
    text(svg, 42, 53, 'TRAINING', 'small mono', 'start');
    box(svg, 56, 83, 152, 64, 'box');
    text(svg, 132, 104, 'context  a a b', 'label');
    text(svg, 132, 133, 'input only', 'small');

    box(svg, 384, 73, 220, 248, 'param');
    text(svg, 494, 105, 'same trained MLP', 'main');
    text(svg, 494, 137, 'E_tok, W\u2081, b\u2081', 'small mono');
    text(svg, 494, 163, 'W\u2082, b\u2082', 'small mono');
    line(svg, 405, 194, 583, 194, 'guide');
    text(svg, 494, 218, 'one parameter set', 'small');
    text(svg, 494, 248, 'for both lanes', 'small');

    box(svg, 660, 77, 168, 72, 'act');
    text(svg, 744, 100, 'distribution', 'label');
    text(svg, 744, 132, 'p(i) = ' + pTarget.toFixed(3), 'small mono');
    box(svg, 892, 77, 156, 72, 'box');
    text(svg, 970, 100, 'loss', 'main');
    text(svg, 970, 132, '\u2212log p(i)', 'small mono');
    box(svg, 748, 157, 126, 32, 'box', 7);
    text(svg, 811, 173, 'observed i', 'small');
    path(svg, 'M874 173 C910 173 949 173 949 152', 'edge');
    path(svg, 'M208 115 L380 115', 'blue-edge');
    path(svg, 'M604 115 L656 115', 'blue-edge');
    path(svg, 'M828 115 L888 115', 'edge');
    path(svg, 'M970 73 C970 31 653 31 598 76', 'grad-edge');
    text(svg, 782, 22, 'update shared parameters', 'small mono');

    if (stage >= 1) {
      text(svg, 42, 229, 'GENERATION', 'small mono', 'start');
      box(svg, 56, 259, 152, 64, 'box');
      text(svg, 132, 280, 'context  a a b', 'label');
      text(svg, 132, 309, 'no observed label', 'small');
      box(svg, 660, 255, 168, 72, 'act');
      text(svg, 744, 278, 'distribution', 'label');
      text(svg, 744, 310, 'same weights', 'small mono');
      box(svg, 877, 255, 178, 72, 'box');
      text(svg, 966, 278, 'sample  ' + chosen, 'main');
      text(svg, 966, 310, 'one possible draw', 'small');
      path(svg, 'M208 291 L380 291', 'blue-edge');
      path(svg, 'M604 291 L656 291', 'blue-edge');
      path(svg, 'M828 291 L873 291', 'edge');
      text(svg, 494, 303, 'fixed during generation', 'small');
    }

    if (stage >= 2) {
      path(svg, 'M966 331 C966 354 274 354 132 328', 'blue-edge');
      text(svg, 590, 375, 'append ' + chosen + ', shift window \u2192 ' + nextContext.join(' '), 'main');
      text(svg, 590, 409, 'reuse the same model; boundary \u201c-\u201d would stop', 'small');
    }
    return svg;
  }

  AT.part1Diagrams = {
    embeddingSpace: embeddingSpace,
    lookupConcat: lookupConcat,
    learningGraph: learningGraph,
    trainingVsGeneration: trainingVsGeneration
  };
})();

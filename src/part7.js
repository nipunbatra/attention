/* Vision III. A deliberately small, fully disclosed CLIP-style calculation.
   AT.clip owns this model; the attention runtime is not used for its numbers. */
(function () {
  'use strict';
  const AT = window.AT, data = window.__TOY__.clip;
  const clone = x => JSON.parse(JSON.stringify(x));
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
  const mm = (a, b) => a.map(row => transpose(b).map(column => dot(row, column)));
  const sum = a => a.reduce((s, x) => s + x, 0);
  const mean = a => sum(a) / a.length;
  const norm = a => Math.sqrt(dot(a, a));
  function unit(a) { const n = norm(a); if (!(n > 1e-12) || !Number.isFinite(n)) throw Error('A finite, nonzero embedding is required.'); return a.map(x => x / n); }
  function softmax(row) {
    const m = Math.max(...row), ex = row.map(x => Math.exp(x - m)), z = sum(ex);
    if (!Number.isFinite(z) || z <= 0) throw Error('Invalid softmax scores.');
    return ex.map(x => x / z);
  }
  const logsumexp = row => { const m = Math.max(...row); return m + Math.log(sum(row.map(x => Math.exp(x - m)))); };
  function params(snapshot = 'initial') {
    if (!data.snapshots[snapshot]) throw Error('Unknown CLIP toy snapshot.');
    return clone(data.snapshots[snapshot]);
  }
  function validate(p) {
    for (const [name, rows] of [['W_img', 16], ['W_txt', data.vocab.length]]) {
      if (!Array.isArray(p[name]) || p[name].length !== rows || !p[name].every(r => Array.isArray(r) && r.length === 3 && r.every(Number.isFinite))) throw Error('Invalid ' + name + ' shape.');
    }
    if (!Number.isFinite(p.log_scale) || !Number.isFinite(Math.exp(p.log_scale))) throw Error('Invalid log scale.');
  }
  function words(captions) {
    return captions.map(caption => {
      const row = Array(data.vocab.length).fill(0), tokens = String(caption).trim().toLowerCase().split(/\s+/);
      tokens.forEach(token => { const i = data.vocab.indexOf(token); if (i < 0) throw Error('Outside the toy vocabulary: ' + token); row[i]++; });
      return row;
    });
  }
  function encode(p, images, captions) {
    validate(p);
    const pixels = images.map(image => image.flat());
    if (!pixels.length || !pixels.every(row => row.length === 16 && row.every(Number.isFinite))) throw Error('Each toy image needs 16 finite pixels.');
    if (!captions.length) throw Error('Choose at least one caption.');
    const wordCounts = words(captions), imageRaw = mm(pixels, p.W_img), textRaw = mm(wordCounts, p.W_txt);
    return { pixels, wordCounts, imageRaw, textRaw, imageNorm: imageRaw.map(norm), textNorm: textRaw.map(norm), imageUnit: imageRaw.map(unit), textUnit: textRaw.map(unit) };
  }
  function forward(p = params()) {
    const f = encode(p, data.images, data.captions);
    const cosine = mm(f.imageUnit, transpose(f.textUnit)), scale = Math.exp(p.log_scale), tau = 1 / scale;
    const logits = cosine.map(row => row.map(x => x * scale));
    const rowProb = logits.map(softmax), columnProb = transpose(transpose(logits).map(softmax));
    const rowLosses = logits.map((row, i) => logsumexp(row) - row[i]);
    const columnLosses = transpose(logits).map((column, j) => logsumexp(column) - column[j]);
    return Object.assign(f, { cosine, scale, tau, logits, rowProb, columnProb, rowLosses, columnLosses, rowLoss: mean(rowLosses), columnLoss: mean(columnLosses), loss: (mean(rowLosses) + mean(columnLosses)) / 2 });
  }
  function gradients(p = params()) {
    const f = forward(p), n = data.images.length;
    const dlogits = f.logits.map((row, i) => row.map((_, j) => (f.rowProb[i][j] + f.columnProb[i][j] - (i === j ? 2 : 0)) / (2 * n)));
    const dcosine = dlogits.map(row => row.map(x => x * f.scale));
    const di = mm(dcosine, f.textUnit), dt = mm(transpose(dcosine), f.imageUnit);
    function undoNormalization(upstream, normalized, lengths) {
      return upstream.map((row, i) => { const radial = dot(row, normalized[i]); return row.map((x, j) => (x - normalized[i][j] * radial) / lengths[i]); });
    }
    const dimage = undoNormalization(di, f.imageUnit, f.imageNorm), dtext = undoNormalization(dt, f.textUnit, f.textNorm);
    return { W_img: mm(transpose(f.pixels), dimage), W_txt: mm(transpose(f.wordCounts), dtext), log_scale: sum(dlogits.map((row, i) => dot(row, f.logits[i]))) };
  }
  function step(p = params(), rate = data.learningRate) {
    if (!Number.isFinite(rate) || rate <= 0) throw Error('The learning rate must be positive.');
    const g = gradients(p);
    // Read all gradients from one frozen pre-update state. Return new arrays.
    return { W_img: p.W_img.map((row, i) => row.map((x, j) => x - rate * g.W_img[i][j])),
      W_txt: p.W_txt.map((row, i) => row.map((x, j) => x - rate * g.W_txt[i][j])), log_scale: p.log_scale - rate * g.log_scale };
  }
  function classify(options = {}) {
    const p = options.parameters || params(options.snapshot || 'trained');
    const image = options.image || data.images[options.imageIndex == null ? 0 : options.imageIndex];
    if (!image) throw Error('Unknown image.');
    const captions = options.captions || data.captions;
    const f = encode(p, [image], captions), tau = options.tau == null ? Math.exp(-p.log_scale) : options.tau;
    if (!Number.isFinite(tau) || tau <= 0) throw Error('Temperature must be positive.');
    const cosine = f.textUnit.map(row => dot(f.imageUnit[0], row)), logits = cosine.map(x => x / tau), probabilities = softmax(logits);
    const best = cosine.indexOf(Math.max(...cosine));
    return Object.assign(f, { captions: captions.slice(), tau, cosine, logits, probabilities, best, label: captions[best] });
  }
  function retrieve(caption, options = {}) {
    const p = options.parameters || params(options.snapshot || 'trained');
    const f = encode(p, data.images, [caption]);
    const cosine = f.imageUnit.map(row => dot(row, f.textUnit[0]));
    return { caption, cosine, order: cosine.map((score, i) => ({ i, score })).sort((a, b) => b.score - a.score).map(x => x.i) };
  }

  let serial = 0;
  const ns = 'http://www.w3.org/2000/svg';
  const colors = { image: 'var(--c-e)', text: 'var(--c-q)', match: 'var(--ink-2)', probability: 'var(--c-a)', loss: 'var(--c-d)', ink: 'var(--ink)', muted: 'var(--ink-3)' };
  const roleClass = { image: 'e', text: 'q', probability: 'a', loss: 'd' };
  function node(tag, attrs = {}, value) {
    const e = document.createElementNS(ns, tag);
    for (const [key, val] of Object.entries(attrs)) e.setAttribute(key, String(val));
    if (value != null) e.textContent = String(value);
    return e;
  }
  function canvas(height, title, description, width = 1100) {
    const id = 'clip-diagram-' + (++serial), svg = node('svg', { id, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-labelledby': id + '-title ' + id + '-desc', 'data-clip-diagram': title });
    svg.append(node('title', { id: id + '-title' }, title), node('desc', { id: id + '-desc' }, description));
    const defs = node('defs');
    for (const [role, color] of Object.entries(colors)) { const m = node('marker', { id: id + '-' + role, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }); m.append(node('path', { d: 'M0 0 L10 5 L0 10 Z', fill: color })); defs.append(m); }
    svg.append(defs);
    const text = (x, y, value, role = 'ink', size = 22, anchor = 'middle') => { const e = node('text', { x, y, fill: colors[role], 'font-size': size, 'font-family': 'var(--font-ui)', 'text-anchor': anchor, 'dominant-baseline': 'middle' }, value); svg.append(e); return e; };
    function box(x, y, width, height, label, subtitle, role = 'ink') {
      svg.append(node('rect', { x: x - width / 2, y: y - height / 2, width, height, rx: 8, fill: roleClass[role] ? 'var(--t-' + roleClass[role] + ')' : 'var(--card)', stroke: colors[role], 'stroke-width': 1.8 }));
      text(x, subtitle ? y - 13 : y, label, role, 26);
      if (subtitle) text(x, y + 18, subtitle, 'muted', 22);
    }
    const arrow = (path, role = 'muted', dashed = false) => svg.append(node('path', { d: path, fill: 'none', stroke: colors[role], 'stroke-width': 2.3, 'stroke-dasharray': dashed ? '6 4' : 'none', 'marker-end': `url(#${id}-${role})` }));
    return { svg, text, box, arrow };
  }
  function imageGrid(c, image, left, top, cell = 27) {
    image.forEach((row, i) => row.forEach((value, j) => {
      const shade=AT.imageShade(value);
      c.svg.append(node('rect', { x: left + j * cell, y: top + i * cell, width: cell - 2, height: cell - 2, rx: 2,
        'data-pixel-value':value,fill: `rgb(${shade},${shade},${shade})`, stroke: 'var(--line)' }));
      c.text(left + j * cell + (cell - 2) / 2, top + i * cell + (cell - 2) / 2, value, 'ink', cell * .52).setAttribute('fill',shade<118?'#FFFFFF':'#000000');
    }));
  }
  const vectorLabel = row => '(' + row.map(x => Math.abs(x) < 1e-12 ? '0' : Number.isInteger(x) ? String(x) : x.toFixed(3)).join(', ') + ')';
  // Two fixed, orthonormal viewing axes. This is an orthographic view of 3D,
  // not PCA, a refitted camera, or a 2D substitute for the cosine calculation.
  function projectDirection(row) {
    return [(row[0] - row[1]) / Math.sqrt(2), (row[0] + row[1] - 2 * row[2]) / Math.sqrt(6)];
  }
  function checkpoint(count) {
    if (![0, 1, 20, 60].includes(count)) throw Error('Choose a displayed training checkpoint.');
    let p = params();
    for (let i = 0; i < count; i++) p = step(p);
    return p;
  }
  function pairedFigure(className) {
    const el = document.createElement('figure'); el.className = className; return el;
  }
  function normalizationDiagram(f) {
    const figure = pairedFigure('clip-plot-pair');
    for (const normalized of [false, true]) {
      const c = canvas(330, normalized ? 'After normalization' : 'Before normalization', 'The first image and caption have zero third coordinates, so this is their exact coordinate 1 and coordinate 2 plane. Both vectors keep the same direction after division by their own length.', 500);
      const ox = 110, oy = 247, scale = 110;
      c.text(250, 23, normalized ? 'Divide each vector by its own length' : 'Start with the encoder outputs', 'ink', 24);
      c.svg.append(node('path', { d: `M${ox + scale} ${oy} A${scale} ${scale} 0 0 0 ${ox} ${oy - scale}`, fill: 'none', stroke: 'var(--line)', 'stroke-dasharray': '5 5' }));
      c.arrow('M68 247 L426 247', 'muted'); c.arrow('M110 277 L110 60', 'muted');
      c.text(445, 247, '1', 'muted', 22); c.text(110, 49, '2', 'muted', 22);
      const rows = normalized ? [f.imageUnit[0], f.textUnit[0]] : [f.imageRaw[0], f.textRaw[0]];
      rows.forEach((row, i) => {
        const x = ox + scale * row[0], y = oy - scale * row[1], role = i ? 'text' : 'image';
        c.arrow(`M${ox} ${oy} L${x} ${y}`, role);
        if (!i) {
          const lx = normalized ? 297 : 347, ly = normalized ? 160 : 109;
          c.text(lx, ly, normalized ? '(0.894, 0.447, 0)' : '(2, 1, 0)', role, 22);
          if (normalized) c.svg.append(node('path', { d: `M${x} ${y} L254 171`, stroke: colors[role], fill: 'none' }));
        } else c.text(x + 12, oy + 32, normalized ? '(1, 0, 0)' : '(2, 0, 0)', role, 22);
      });
      const angle = Math.atan2(f.imageRaw[0][1], f.imageRaw[0][0]);
      c.svg.append(node('path', { d: `M${ox + 65} ${oy} A65 65 0 0 0 ${ox + 65 * Math.cos(angle)} ${oy - 65 * Math.sin(angle)}`, stroke: colors.match, fill: 'none', 'stroke-width': 2 }));
      c.text(286, 207, '26.6°', 'match', 24);
      c.text(250, 315, normalized ? 'Both lengths = 1' : 'Lengths: image √5; caption 2', 'ink', 24);
      if (normalized) c.svg.dataset.build = '1';
      figure.append(c.svg);
    }
    return figure;
  }
  function geometryDiagram(f, label = 'Initial directions') {
    const figure = pairedFigure('clip-geometry');
    const c = canvas(350, label, 'Fixed orthographic view of the actual three-dimensional unit vectors. Blue circles are images; purple diamonds are captions. Numbers link to the same image-caption pairs in the legend. Page angles are foreshortened; reported cosines use all three coordinates.', 460);
    const ox = 228, oy = 188, scale = 127;
    const point = row => { const p = projectDirection(row); return [ox + scale * p[0], oy + scale * p[1]]; };
    c.text(230, 22, label, 'ink', 26);
    for (const plane of [[0, 1], [0, 2], [1, 2]]) {
      const points = Array.from({ length: 97 }, (_, i) => { const a = i * Math.PI / 48, v = [0, 0, 0]; v[plane[0]] = Math.cos(a); v[plane[1]] = Math.sin(a); return point(v); });
      c.svg.append(node('path', { d: points.map((p, i) => (i ? 'L' : 'M') + p.join(' ')).join(' '), fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.3 }));
    }
    for (let j = 0; j < 3; j++) {
      const a = [0, 0, 0], b = [0, 0, 0]; a[j] = -1.2; b[j] = 1.2;
      const start = point(a), end = point(b);
      c.arrow(`M${start.join(' ')} L${end.join(' ')}`, 'muted');
      c.text(end[0] + (j === 0 ? 19 : j === 1 ? -19 : 0), end[1] + (j === 2 ? -15 : 6), String(j + 1), 'muted', 22);
    }
    const vectors = f.imageUnit.map((row, i) => ({ row, i, role: 'image' })).concat(f.textUnit.map((row, i) => ({ row, i, role: 'text' })));
    // Label locations stay fixed across checkpoints. Leaders keep nearby ends distinct.
    const labels = { image: [[404, 303], [54, 308], [327, 83]], text: [[403, 218], [55, 215], [143, 66]] };
    vectors.forEach(({ row, i, role }) => {
      const [x, y] = point(row), [lx, ly] = labels[role][i];
      c.svg.append(node('path', { d: `M${ox} ${oy} L${x} ${y}`, fill: 'none', stroke: colors[role], 'stroke-width': 2.2, 'data-vector': JSON.stringify(row), 'data-pair': i + 1, 'data-modality': role, 'data-projected': JSON.stringify(projectDirection(row)) }));
      c.svg.append(role === 'image' ? node('circle', { cx: x, cy: y, r: 6, fill: colors[role] }) : node('path', { d: `M${x} ${y - 8} L${x + 8} ${y} L${x} ${y + 8} L${x - 8} ${y} Z`, fill: 'var(--card)', stroke: colors[role], 'stroke-width': 2 }));
      c.svg.append(node('path', { d: `M${x} ${y} L${lx + (lx < ox ? 13 : -13)} ${ly}`, fill: 'none', stroke: colors[role], 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
      c.text(lx, ly, (role === 'image' ? 'I' : 'T') + (i + 1), role, 23);
    });
    c.text(230, 333, 'Axes: coordinates 1, 2, 3', 'muted', 22);
    const legend = document.createElement('div'); legend.className = 'clip-geometry-legend';
    data.images.forEach((im, i) => {
      const row = document.createElement('div'); row.className = 'clip-geometry-row';
      const thumb = canvas(100, 'Image ' + (i + 1) + ': ' + data.captions[i], 'The actual four by four pixel input for this pair.', 100);
      imageGrid(thumb, im, 2, 2, 24);
      const words = document.createElement('div'), title = document.createElement('p'), score = document.createElement('p');
      title.textContent = (i + 1) + ' · ' + data.captions[i];
      score.textContent = `I${i + 1} ↔ T${i + 1}: cosine ${f.cosine[i][i].toFixed(3)}`;
      score.className = 'clip-pair-score'; words.append(title, score); row.append(thumb.svg, words); legend.append(row);
    });
    figure.append(c.svg, legend); return figure;
  }
  function temperatureDiagram() {
    const figure = pairedFigure('clip-plot-pair');
    for (const tau of [0.5, 0.1]) {
      const r = classify({ snapshot: 'initial', tau });
      const c = canvas(300, 'Fixed directions, temperature ' + tau, 'Candidate probabilities for the unchanged two-square image and three initial caption vectors. Only the temperature changes; the cosine scores and ranking stay fixed.', 500);
      c.text(250, 25, 'Temperature τ = ' + tau.toFixed(1), 'ink', 26);
      for (const p of [0, 0.5, 1]) {
        const y = 213 - 145 * p;
        c.svg.append(node('path', { d: `M57 ${y} H467`, stroke: 'var(--line)', fill: 'none' }));
        c.text(44, y, p.toFixed(1), 'muted', 20, 'end');
      }
      r.probabilities.forEach((p, i) => {
        const x = 120 + 142 * i, y = 213 - 145 * p;
        c.svg.append(node('rect', { x: x - 36, y, width: 72, height: 145 * p, rx: 3, fill: i === r.best ? colors.probability : 'var(--line)', 'data-probability': p, 'data-temperature': tau, 'data-candidate': i }));
        c.text(x, y - 16, p.toFixed(3), 'probability', 24);
        c.text(x, 244, ['squares', 'stripes', 'dot'][i], 'text', 23);
        c.text(x, 278, 'cos ' + r.cosine[i].toFixed(3), 'match', 22);
      });
      if (tau === 0.1) c.svg.dataset.build = '1';
      figure.append(c.svg);
    }
    return figure;
  }
  function geometryWidget(host) {
    const controls = document.createElement('div'); controls.className = 'clip-controls';
    const plot = document.createElement('div'), status = document.createElement('p'); status.className = 'clip-checkpoint-status'; status.setAttribute('aria-live', 'polite');
    const buttons = [];
    function render(count) {
      const f = forward(checkpoint(count));
      plot.replaceChildren(geometryDiagram(f, 'After ' + count + ' SGD ' + (count === 1 ? 'step' : 'steps')));
      status.textContent = `Loss ${f.loss.toFixed(4)} · learned temperature ${f.tau.toFixed(3)}. Same axes and scale at every step.`;
      buttons.forEach(({ button, n }) => button.setAttribute('aria-pressed', String(n === count)));
    }
    for (const n of [0, 1, 20, 60]) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = n + (n === 1 ? ' step' : ' steps'); button.addEventListener('click', () => render(n));
      buttons.push({ button, n }); controls.append(button);
    }
    host.append(controls, plot, status); render(60);
  }
  function batchDiagram() {
    const c = canvas(235, 'Three observed image-caption pairs', 'A two-square grid, horizontal stripes, and a single dot have one observed caption each. All image pixels are shown.');
    data.images.forEach((im, i) => {
      const x = 185 + 365 * i;
      c.text(x, 20, 'Pair ' + (i + 1), 'image', 26); imageGrid(c, im, x - 56, 49, 28);
      c.text(x, 189, data.captions[i], 'text', 26); c.text(x, 220, 'I' + (i + 1) + ' ↔ T' + (i + 1), 'muted', 22);
    });
    return c.svg;
  }
  function anchorDiagram() {
    const c = canvas(225, 'The familiar two-square image', 'A four by four grid has a bright square of value one at top left and a brighter square of value two at bottom right. The other pixels are zero.');
    imageGrid(c, data.images[0], 150, 5, 50);
    c.text(720, 70, 'Class: pattern', 'image', 32);
    c.text(720, 130, 'Caption: two bright squares', 'text', 29);
    return c.svg;
  }
  function falseNegativeDiagram() {
    const c = canvas(235, 'Valid descriptions can be unpaired in a batch', 'Two copies of the two-square grid are paired with two valid descriptions. The crossed image-caption pairings are also valid, although a diagonal-only training target treats them as negatives.');
    imageGrid(c, data.images[0], 30, 12, 23); imageGrid(c, data.images[0], 30, 129, 23);
    c.text(213, 60, 'Image 1', 'image', 23); c.text(213, 177, 'Image 2', 'image', 23);
    c.box(843, 60, 455, 63, 'two bright squares', 'Observed with image 1', 'text');
    c.box(843, 177, 455, 63, 'a grid with bright squares', 'Observed with image 2', 'text');
    c.arrow('M280 60 L610 60', 'match'); c.arrow('M280 177 L610 177', 'match');
    c.arrow('M293 70 L600 167', 'loss', true); c.arrow('M293 167 L600 70', 'loss', true);
    c.text(485, 221, 'Dashed: valid, but not the recorded pair.', 'loss', 22);
    return c.svg;
  }
  function encodersDiagram(f, normalized) {
    const c = canvas(normalized ? 295 : 245, 'Two encoders produce comparable vectors', 'The image branch reads pixels. The text branch reads word counts. Each learned map returns one three-coordinate vector; normalization puts both on the unit sphere.');
    imageGrid(c, data.images[0], 60, 10, 25);
    c.box(110, 186, 214, 80, 'Caption T1', 'two bright squares', 'text');
    const encoderX = normalized ? 385 : 465, rawX = normalized ? 650 : 886;
    c.box(encoderX, 62, 248, 80, 'Image encoder', 'W_img: 16 × 3', 'image');
    c.box(encoderX, 186, 248, 80, 'Text encoder', 'W_txt: 11 × 3', 'text');
    c.arrow('M164 62 L' + (encoderX - 127) + ' 62', 'image'); c.arrow('M220 186 L' + (encoderX - 127) + ' 186', 'text');
    c.box(rawX, 62, 228, 80, 'g_img,1', vectorLabel(f.imageRaw[0]), 'image');
    c.box(rawX, 186, 228, 80, 'g_txt,1', vectorLabel(f.textRaw[0]), 'text');
    c.arrow('M' + (encoderX + 127) + ' 62 L' + (rawX - 117) + ' 62', 'image'); c.arrow('M' + (encoderX + 127) + ' 186 L' + (rawX - 117) + ' 186', 'text');
    if (normalized) {
      c.box(950, 62, 280, 80, 'Unit image vector', vectorLabel(f.imageUnit[0]), 'image');
      c.box(950, 186, 280, 80, 'Unit text vector', vectorLabel(f.textUnit[0]), 'text');
      c.arrow('M767 62 L807 62', 'image'); c.arrow('M767 186 L807 186', 'text');
      c.text(787, 24, '÷ norm', 'muted', 22); c.text(787, 145, '÷ norm', 'muted', 22);
      c.text(550, 270, 'Shared width: 3. Learned coordinates have no assigned names.', 'muted', 22);
    }
    return c.svg;
  }
  function matrixDiagram(f, mode) {
    const c = canvas(315, 'Image-caption matching matrix: ' + mode, 'Rows are images I1 to I3. Columns are captions T1 to T3. Diagonal entries are the observed pairings. Row probabilities sum across captions; column probabilities sum across images.');
    const values = mode === 'row' ? f.rowProb : mode === 'column' ? f.columnProb : mode === 'cosine' ? f.cosine : f.logits;
    const labels = { row: 'Image → text', column: 'Text → image', cosine: 'Cosine C', logits: 'Logits S = C / τ' };
    c.text(165, 31, labels[mode], mode === 'row' || mode === 'column' ? 'probability' : 'match', 26);
    c.text(700, 25, 'Caption candidates', 'text', 26);
    const x0 = 470, y0 = 87, w = 160, h = 54;
    ['T1 · squares', 'T2 · stripes', 'T3 · dot'].forEach((s, j) => c.text(x0 + j * w + w / 2, 63, s, 'text', 22));
    values.forEach((row, i) => {
      c.text(x0 - 24, y0 + i * h + h / 2, ['I1 · squares', 'I2 · stripes', 'I3 · dot'][i], 'image', 22, 'end');
      row.forEach((x, j) => {
        c.svg.append(node('rect', { x: x0 + j * w + 3, y: y0 + i * h + 3, width: w - 6, height: h - 6, rx: 5,
          fill: i === j ? 'var(--t-d)' : 'var(--card)', stroke: i === j ? 'var(--c-d)' : 'var(--line)', 'data-diagonal': i === j }));
        c.text(x0 + j * w + w / 2, y0 + i * h + h / 2, x.toFixed(3), i === j ? 'loss' : 'ink', 28);
      });
    });
    if (mode === 'row') { c.arrow('M478 268 L940 268', 'probability'); c.text(198, 141, 'One image;', 'ink', 23); c.text(198, 176, 'choose among captions.', 'ink', 20); }
    if (mode === 'column') { c.arrow('M983 92 L983 242', 'probability'); c.text(198, 141, 'One caption;', 'ink', 23); c.text(198, 176, 'choose among images.', 'ink', 20); }
    if (mode === 'cosine' || mode === 'logits') { c.text(182, 141, 'Green cells:', 'loss', 22); c.text(182, 176, 'observed matches', 'ink', 20); }
    c.text(550, 299, mode === 'row' ? 'Each row sums to 1.' : mode === 'column' ? 'Each column sums to 1.' : 'All nine scores come from the displayed vectors.', 'muted', 22);
    return c.svg;
  }
  function trainingDiagram() {
    const c = canvas(290, 'Loss sends gradients into both encoders', 'Pixels and word counts pass through separate learned encoders and exact unit normalization. Their cosine matrix and learned scale produce two cross-entropies. Autograd differentiates all paths before the optimizer updates the parameters.');
    c.box(122, 58, 220, 62, 'Images', '3 × 16 pixels', 'image'); c.box(122, 186, 220, 62, 'Captions', '3 × 11 word counts', 'text');
    c.box(389, 58, 240, 72, 'Image map', 'W_img → normalize', 'image'); c.box(389, 186, 240, 72, 'Text map', 'W_txt → normalize', 'text');
    c.box(675, 122, 230, 82, 'Cosine / τ', 'learned log_scale', 'match'); c.box(970, 122, 235, 82, 'Mean of two CEs', 'image ↔ text', 'loss');
    c.arrow('M235 58 L275 58', 'image'); c.arrow('M235 186 L275 186', 'text');
    c.arrow('M502 58 L546 58 L546 104 L557 104', 'image'); c.arrow('M502 186 L546 186 L546 144 L557 144', 'text'); c.arrow('M793 122 L849 122', 'match');
    c.arrow('M966 167 L966 243 L392 243 L392 221', 'loss', true);
    c.arrow('M676 243 L676 273 L261 273 L261 95 L281 95', 'loss', true);
    c.text(782, 215, 'autograd: both branches', 'loss', 22);
    return c.svg;
  }
  function promptsDiagram(result) {
    const c = canvas(290, 'Text prompts provide the candidate classifier', 'Frozen encoders turn an image and supplied prompt strings into normalized vectors. Their cosine scores rank only those candidate prompts. No encoder update happens here.');
    imageGrid(c, data.images[0], 35, 17, 28);
    c.box(321, 71, 250, 80, 'Image encoder', 'frozen weights', 'image'); c.arrow('M151 71 L192 71', 'image');
    c.box(321, 207, 250, 80, 'Text encoder', 'frozen weights', 'text');
    c.box(674, 137, 280, 80, 'Cosine → softmax', result.captions.length + ' supplied candidates', 'match');
    c.arrow('M449 71 L500 71 L500 119 L531 119', 'image'); c.arrow('M449 207 L500 207 L500 155 L531 155', 'text');
    c.box(957, 137, 235, 80, 'Highest score', result.label, 'probability'); c.arrow('M817 137 L836 137', 'match');
    c.text(100, 174, 'Prompt strings', 'text', 22); c.arrow('M150 207 L192 207', 'text');
    c.text(550, 271, 'Change the candidate text; keep the encoder weights fixed.', 'muted', 22);
    return c.svg;
  }
  function retrievalDiagram(caption) {
    const r = retrieve(caption), c = canvas(260, 'Retrieve images with a caption', 'One caption vector is compared with the cached image vectors. Images are ordered by computed cosine similarity.');
    c.box(196, 70, 370, 80, caption, 'one text vector', 'text');
    c.text(196, 154, 'Rank cached image vectors', 'ink', 22); c.arrow('M385 70 L434 70', 'text');
    r.order.forEach((index, rank) => { const x = 532 + rank * 215; imageGrid(c, data.images[index], x - 50, 23, 25); c.text(x, 155, (rank + 1) + '. ' + ['squares','stripes','dot'][index], 'image', 26); c.text(x, 190, 'cos = ' + r.cosine[index].toFixed(3), 'match', 22); });
    c.text(550, 238, 'Retrieval selects an existing item. It does not write a caption.', 'muted', 22);
    return c.svg;
  }
  function diagram(host, stage, options = {}) {
    if (typeof host === 'string') host = document.getElementById(host);
    if (!host) throw Error('Diagram host missing.');
    const f = forward(options.parameters || params(options.snapshot || 'initial'));
    let svg;
    if (stage === 'batch') svg = batchDiagram();
    else if (stage === 'anchor') svg = anchorDiagram();
    else if (stage === 'false-negatives') svg = falseNegativeDiagram();
    else if (stage === 'normalization-geometry') svg = normalizationDiagram(f);
    else if (stage === 'directions') svg = geometryDiagram(f, options.label);
    else if (stage === 'temperature') svg = temperatureDiagram();
    else if (stage === 'encoders' || stage === 'normalized') svg = encodersDiagram(f, stage === 'normalized');
    else if (['cosine', 'logits', 'row', 'column'].includes(stage)) svg = matrixDiagram(f, stage);
    else if (stage === 'training') svg = trainingDiagram();
    else if (stage === 'prompts') svg = promptsDiagram(classify(options));
    else if (stage === 'retrieval') svg = retrievalDiagram(options.caption || data.captions[0]);
    else throw Error('Unknown CLIP diagram stage: ' + stage);
    host.replaceChildren(svg); return svg;
  }
  function table(host, values, options = {}) {
    host.replaceChildren();
    const figure = AT.ui.table(values, Object.assign({ into: host, decimals: 3 }, options));
    figure.classList.add('clip-table');
    const cards = document.createElement('div'); cards.className = 'clip-phone-table';
    if (options.cornerLabel) { const title = document.createElement('h4'); title.textContent = options.cornerLabel; cards.append(title); }
    values.forEach((row, i) => {
      const card = document.createElement('div'); card.className = 'clip-phone-row';
      const title = document.createElement('h4'); title.textContent = options.rowLabels?.[i] || 'Row ' + (i + 1); card.append(title);
      const fields = document.createElement('dl');
      row.forEach((value, j) => {
        const label = document.createElement('dt'), number = document.createElement('dd');
        const col = options.cols?.[j]; label.textContent = typeof col === 'object' ? col.label : col || 'Coordinate ' + (j + 1);
        number.textContent = typeof value === 'number' ? value.toFixed(options.decimals ?? 3) : String(value);
        fields.append(label, number);
      });
      card.append(fields); cards.append(card);
    });
    figure.append(cards); return figure;
  }
  function trainingWidget(host) {
    let p = params(), count = 0;
    const controls = document.createElement('div'); controls.className = 'clip-controls';
    const output = document.createElement('p'), chart = document.createElement('div');
    const status = document.createElement('p'); status.setAttribute('aria-live', 'polite');
    function render() { const f = forward(p); status.textContent = `Step ${count}: loss ${f.loss.toFixed(4)}; temperature ${f.tau.toFixed(3)}. Same three training pairs.`; diagram(chart, 'cosine', { parameters: p }); output.textContent = 'Correct-caption probabilities: ' + f.rowProb.map((r, i) => 'pair ' + (i + 1) + ': ' + r[i].toFixed(3)).join(' · '); }
    for (const [label, steps] of [['One SGD step', 1], ['20 steps', 20], ['Reset', 0]]) {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
      b.addEventListener('click', () => { if (!steps) { p = params(); count = 0; } else for (let n = 0; n < steps && count < 200; n++) { p = step(p); count++; } render(); }); controls.append(b);
    }
    host.append(controls, status, chart, output); render();
  }
  function candidateWidget(host) {
    const controls = document.createElement('div'); controls.className = 'clip-controls';
    const template = document.createElement('select'); template.setAttribute('aria-label', 'Toy prompt template');
    for (const [value, label] of [['bare', 'Bare description'], ['grid', 'a grid with {description}']]) { const o = document.createElement('option'); o.value = value; o.textContent = label; template.append(o); }
    const set = document.createElement('select'); set.setAttribute('aria-label', 'Candidate caption set');
    for (const [value, label] of [['all', 'All three candidates'], ['missing', 'Omit the correct description'], ['duplicate', 'Duplicate the correct description']]) { const o = document.createElement('option'); o.value = value; o.textContent = label; set.append(o); }
    const tauLabel = document.createElement('label'); tauLabel.textContent = 'Temperature τ ';
    const learnedTau=classify().tau;
    const tau = document.createElement('input'); tau.type = 'range'; tau.min = '.05'; tau.max = '1'; tau.step = 'any'; tau.value = String(learnedTau); tau.setAttribute('aria-label', 'Inference temperature'); tauLabel.append(tau);
    const value = document.createElement('output'); tauLabel.append(value);
    const output = document.createElement('div'), summary = document.createElement('p'); summary.setAttribute('aria-live', 'polite');
    function render() {
      let captions = set.value === 'missing' ? data.captions.slice(1) : data.captions.slice();
      if (set.value === 'duplicate') captions.push(data.captions[0]);
      if (template.value === 'grid') captions = captions.map(c => 'a grid with ' + c);
      const r = classify({ captions, tau: Number(tau.value) }); value.textContent = Number(tau.value).toFixed(3);
      table(output, r.cosine.map((score, i) => [score, r.logits[i], r.probabilities[i]]), { cols: ['cosine', 'logit', 'candidate p'], rowLabels: captions, cornerLabel: 'Two-square image vs prompt' });
      const sameTemperature=Math.abs(Number(tau.value)-learnedTau)<1e-12;
      summary.textContent = `Highest-scoring candidate: ${r.label}. ${sameTemperature?'Using the learned temperature, τ = '+learnedTau.toFixed(3)+'.':'Inference temperature changed from the learned τ = '+learnedTau.toFixed(3)+'.'} The image and encoder weights are unchanged.`;
    }
    for (const input of [template, set, tau]) input.addEventListener('input', render);
    controls.append(template, set, tauLabel); host.append(controls, output, summary); render();
  }
  AT.clip = { data: clone(data), params, forward, gradients, step, classify, retrieve, words, unit, softmax, checkpoint, projectDirection, diagram, table, trainingWidget, geometryWidget, candidateWidget };
  AT.axes.named = false;
  const defs = { e: ['g_img', 'image embedding', 'One global image vector after its image encoder. A hat marks unit normalization.'], q: ['g_txt', 'text embedding', 'One global caption or prompt vector after its text encoder. This is not an attention query.'], a: ['p', 'candidate probability', 'A softmax over the supplied captions or images, not confidence over all possible descriptions.'], d: ['L', 'contrastive loss', 'Average image-to-text and text-to-image cross-entropy on the observed pairs.'] };
  AT.objects.forEach(object => { if (defs[object.cls]) { const d = defs[object.cls]; object.sym = d[0]; object.symTex=object.cls==='e'?'\\hat g^{\\mathrm{img}}':object.cls==='q'?'\\hat g^{\\mathrm{txt}}':d[0]; object.name = d[1]; object.def = d[2]; object.tip = d[2]; } });
  function note(sym, meaning, shape, size, group = 'Image-text matching') { AT.notation.push({ g: group, sym, mean: meaning, shape, dims: () => size, parts: ['vision3'] }); }
  note('\\ve{g_i^{\\mathrm{img}}},\\vq{g_j^{\\mathrm{txt}}}', 'Global image and text vectors before unit normalization; not attention values', '1\\times d', '1×3 in the toy');
  note('\\hat g = g/\\lVert g\\rVert_2', 'A unit-length vector; its direction is compared with the other modality', '1\\times d', '1×3');
  note('C_{ij}=\\ve{\\hat g_i^{\\mathrm{img}}}(\\vq{\\hat g_j^{\\mathrm{txt}}})^\\top', 'Cosine similarity of image i and caption j', '\\text{scalar}', '−1 to 1');
  note('S_{ij}=C_{ij}/\\tau', 'Temperature-scaled logit; it is not itself a probability', '\\text{scalar}', '');
  note('\\va{p(T_j\\mid I_i,\\mathcal B)}', 'Row softmax over the captions in this batch or candidate set', '\\text{scalar}', '0 to 1');
  note('\\va{p(I_i\\mid T_j,\\mathcal B)}', 'Column softmax over images in this batch', '\\text{scalar}', '0 to 1');
  note('\\vd{L}=(L_{I\\to T}+L_{T\\to I})/2', 'Mean of the two batch-average cross-entropies', '\\text{scalar}', '');
  note('N,d', 'Number of paired examples; shared embedding width', '', '3 pairs; 3 coordinates', 'sizes');
  note('W_{\\mathrm{img}},W_{\\mathrm{txt}}', 'Toy learned pixel map and word-row table', '16\\times3,\\;11\\times3', '81 parameters', 'sizes');
  note('a=\\log(1/\\tau)', 'Learned log scale; logits equal exp(a) times cosine', '\\text{scalar}', '1 parameter', 'sizes');
  const originalNotation = AT.ui.notationCard;
  AT.ui.notationCard = function (options = {}) { return originalNotation(Object.assign({}, options, options.part === 'vision3' ? { groups: ['Image-text matching', 'sizes'] } : {})); };
})();

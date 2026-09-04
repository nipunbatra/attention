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
  const colors = { image: 'var(--c-e)', text: 'var(--c-q)', match: 'var(--c-k)', probability: 'var(--c-a)', loss: 'var(--c-d)', ink: 'var(--ink)', muted: 'var(--ink-3)' };
  const roleClass = { image: 'e', text: 'q', match: 'k', probability: 'a', loss: 'd' };
  function node(tag, attrs = {}, value) {
    const e = document.createElementNS(ns, tag);
    for (const [key, val] of Object.entries(attrs)) e.setAttribute(key, String(val));
    if (value != null) e.textContent = String(value);
    return e;
  }
  function canvas(height, title, description) {
    const id = 'clip-diagram-' + (++serial), svg = node('svg', { id, viewBox: `0 0 1100 ${height}`, role: 'img', 'aria-labelledby': id + '-title ' + id + '-desc', 'data-clip-diagram': title });
    svg.append(node('title', { id: id + '-title' }, title), node('desc', { id: id + '-desc' }, description));
    const defs = node('defs');
    for (const [role, color] of Object.entries(colors)) { const m = node('marker', { id: id + '-' + role, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }); m.append(node('path', { d: 'M0 0 L10 5 L0 10 Z', fill: color })); defs.append(m); }
    svg.append(defs);
    const text = (x, y, value, role = 'ink', size = 22, anchor = 'middle') => { const e = node('text', { x, y, fill: colors[role], 'font-size': size, 'font-family': 'var(--font-ui)', 'text-anchor': anchor, 'dominant-baseline': 'middle' }, value); svg.append(e); return e; };
    function box(x, y, width, height, label, subtitle, role = 'ink') {
      svg.append(node('rect', { x: x - width / 2, y: y - height / 2, width, height, rx: 8, fill: roleClass[role] ? 'var(--t-' + roleClass[role] + ')' : 'var(--card)', stroke: colors[role], 'stroke-width': 1.8 }));
      text(x, subtitle ? y - 11 : y, label, role, Math.min(22, (width - 20) / Math.max(1, label.length) * 1.7));
      if (subtitle) text(x, y + 17, subtitle, 'muted', Math.min(17, (width - 20) / subtitle.length * 1.8));
    }
    const arrow = (path, role = 'muted', dashed = false) => svg.append(node('path', { d: path, fill: 'none', stroke: colors[role], 'stroke-width': 2.3, 'stroke-dasharray': dashed ? '6 4' : 'none', 'marker-end': `url(#${id}-${role})` }));
    return { svg, text, box, arrow };
  }
  function imageGrid(c, image, left, top, cell = 27) {
    image.forEach((row, i) => row.forEach((value, j) => {
      c.svg.append(node('rect', { x: left + j * cell, y: top + i * cell, width: cell - 2, height: cell - 2, rx: 2,
        fill: value === 0 ? 'var(--card)' : value === 1 ? 'var(--t-q)' : 'var(--t-v)', stroke: 'var(--line)' }));
      c.text(left + j * cell + (cell - 2) / 2, top + i * cell + (cell - 2) / 2, value, value === 0 ? 'muted' : 'ink', cell * .52);
    }));
  }
  const vectorLabel = row => '(' + row.map(x => Math.abs(x) < 1e-12 ? '0' : Number.isInteger(x) ? String(x) : x.toFixed(3)).join(', ') + ')';
  function batchDiagram() {
    const c = canvas(235, 'Three observed image-caption pairs', 'A two-square grid, horizontal stripes, and a single dot have one observed caption each. All image pixels are shown.');
    data.images.forEach((im, i) => {
      const x = 185 + 365 * i;
      c.text(x, 20, 'Pair ' + (i + 1), 'image', 22); imageGrid(c, im, x - 56, 49, 28);
      c.text(x, 189, data.captions[i], 'text', 23); c.text(x, 220, 'I' + (i + 1) + ' ↔ T' + (i + 1), 'muted', 17);
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
    c.text(485, 221, 'Dashed: also valid, but not the recorded pair.', 'loss', 19);
    return c.svg;
  }
  function encodersDiagram(f, normalized) {
    const c = canvas(normalized ? 295 : 245, 'Two encoders produce comparable vectors', 'The image branch reads pixels. The text branch reads word counts. Each learned map returns one three-coordinate vector; normalization puts both on the unit sphere.');
    imageGrid(c, data.images[0], 18, 10, 25);
    c.box(82, 186, 158, 68, 'Caption T1', 'two bright squares', 'text');
    const encoderX = normalized ? 306 : 430, rawX = normalized ? 596 : 886;
    c.box(encoderX, 62, 240, 74, 'Image encoder', 'toy: W_img is 16 × 3', 'image');
    c.box(encoderX, 186, 240, 74, 'Text encoder', 'toy: W_txt is 11 × 3', 'text');
    c.arrow('M123 62 L' + (encoderX - 123) + ' 62', 'image'); c.arrow('M164 186 L' + (encoderX - 123) + ' 186', 'text');
    c.box(rawX, 62, 252, 74, 'g_img,1', vectorLabel(f.imageRaw[0]), 'image');
    c.box(rawX, 186, 252, 74, 'g_txt,1', vectorLabel(f.textRaw[0]), 'text');
    c.arrow('M' + (encoderX + 123) + ' 62 L' + (rawX - 129) + ' 62', 'image'); c.arrow('M' + (encoderX + 123) + ' 186 L' + (rawX - 129) + ' 186', 'text');
    if (normalized) {
      c.box(934, 62, 290, 74, 'Unit image vector', vectorLabel(f.imageUnit[0]), 'image');
      c.box(934, 186, 290, 74, 'Unit text vector', vectorLabel(f.textUnit[0]), 'text');
      c.arrow('M725 62 L785 62', 'image'); c.arrow('M725 186 L785 186', 'text');
      c.text(757, 34, '÷ norm', 'muted', 16); c.text(757, 157, '÷ norm', 'muted', 16);
      c.text(550, 270, 'Shared width: 3. The learned coordinates are not named image attributes.', 'muted', 20);
    }
    return c.svg;
  }
  function matrixDiagram(f, mode) {
    const c = canvas(300, 'Image-caption matching matrix: ' + mode, 'Rows are images I1 to I3. Columns are captions T1 to T3. Diagonal entries are the observed pairings. Row probabilities sum across captions; column probabilities sum across images.');
    const values = mode === 'row' ? f.rowProb : mode === 'column' ? f.columnProb : mode === 'cosine' ? f.cosine : f.logits;
    const labels = { row: 'Image → text probabilities', column: 'Text → image probabilities', cosine: 'Cosine C', logits: 'Logits S = C / τ' };
    c.text(165, 31, labels[mode], mode === 'row' || mode === 'column' ? 'probability' : 'match', 23);
    c.text(700, 25, 'Caption candidates', 'text', 22);
    const x0 = 470, y0 = 87, w = 160, h = 54;
    ['T1 · squares', 'T2 · stripes', 'T3 · dot'].forEach((s, j) => c.text(x0 + j * w + w / 2, 63, s, 'text', 19));
    values.forEach((row, i) => {
      c.text(x0 - 24, y0 + i * h + h / 2, ['I1 · squares', 'I2 · stripes', 'I3 · dot'][i], 'image', 20, 'end');
      row.forEach((x, j) => {
        c.svg.append(node('rect', { x: x0 + j * w + 3, y: y0 + i * h + 3, width: w - 6, height: h - 6, rx: 5,
          fill: i === j ? 'var(--t-d)' : 'var(--card)', stroke: i === j ? 'var(--c-d)' : 'var(--line)', 'data-diagonal': i === j }));
        c.text(x0 + j * w + w / 2, y0 + i * h + h / 2, x.toFixed(3), i === j ? 'loss' : 'ink', 24);
      });
    });
    if (mode === 'row') { c.arrow('M478 268 L940 268', 'probability'); c.text(198, 141, 'One image;', 'ink', 23); c.text(198, 176, 'choose among captions.', 'ink', 20); }
    if (mode === 'column') { c.arrow('M983 92 L983 242', 'probability'); c.text(198, 141, 'One caption;', 'ink', 23); c.text(198, 176, 'choose among images.', 'ink', 20); }
    if (mode === 'cosine' || mode === 'logits') { c.text(182, 141, 'Green cells:', 'loss', 22); c.text(182, 176, 'observed matches', 'ink', 20); }
    c.text(550, 289, mode === 'row' ? 'Each row sums to 1.' : mode === 'column' ? 'Each column sums to 1.' : 'All nine scores come from the displayed vectors.', 'muted', 18);
    return c.svg;
  }
  function trainingDiagram() {
    const c = canvas(290, 'Loss sends gradients into both encoders', 'Pixels and word counts pass through separate learned encoders and exact unit normalization. Their cosine matrix and learned scale produce two cross-entropies. Autograd differentiates all paths before the optimizer updates the parameters.');
    c.box(122, 58, 220, 62, 'Images', '3 × 16 pixels', 'image'); c.box(122, 186, 220, 62, 'Captions', '3 × 11 word counts', 'text');
    c.box(389, 58, 220, 62, 'W_img → normalize', 'learned image map', 'image'); c.box(389, 186, 220, 62, 'W_txt → normalize', 'learned text map', 'text');
    c.box(675, 122, 230, 82, 'Cosine / τ', 'learned log_scale', 'match'); c.box(970, 122, 235, 82, 'Symmetric loss', '½(row CE + column CE)', 'loss');
    c.arrow('M235 58 L275 58', 'image'); c.arrow('M235 186 L275 186', 'text');
    c.arrow('M502 58 L546 58 L546 104 L557 104', 'image'); c.arrow('M502 186 L546 186 L546 144 L557 144', 'text'); c.arrow('M793 122 L849 122', 'match');
    c.arrow('M966 167 L966 243 L392 243 L392 221', 'loss', true);
    c.arrow('M676 243 L676 273 L261 273 L261 95 L281 95', 'loss', true);
    c.text(782, 215, 'autograd through both branches', 'loss', 19);
    return c.svg;
  }
  function promptsDiagram(result) {
    const c = canvas(290, 'Text prompts provide the candidate classifier', 'Frozen encoders turn an image and supplied prompt strings into normalized vectors. Their cosine scores rank only those candidate prompts. No encoder update happens here.');
    imageGrid(c, data.images[0], 35, 17, 28);
    c.box(321, 71, 250, 68, 'Frozen image encoder', 'one image vector', 'image'); c.arrow('M151 71 L192 71', 'image');
    c.box(321, 207, 250, 68, 'Frozen text encoder', 'one vector per prompt', 'text');
    c.box(674, 137, 280, 80, 'Cosine → softmax', result.captions.length + ' supplied candidates', 'match');
    c.arrow('M449 71 L500 71 L500 119 L531 119', 'image'); c.arrow('M449 207 L500 207 L500 155 L531 155', 'text');
    c.box(957, 137, 235, 80, 'Highest score', result.label, 'probability'); c.arrow('M817 137 L836 137', 'match');
    c.text(110, 194, 'Prompt strings', 'text', 21); c.arrow('M150 207 L192 207', 'text');
    c.text(550, 271, 'A new candidate set changes the readout. The encoder weights stay fixed.', 'muted', 19);
    return c.svg;
  }
  function retrievalDiagram(caption) {
    const r = retrieve(caption), c = canvas(260, 'Retrieve images with a caption', 'One caption vector is compared with the cached image vectors. Images are ordered by computed cosine similarity.');
    c.box(196, 70, 370, 80, caption, 'one text vector', 'text');
    c.text(196, 154, 'Rank cached image vectors', 'ink', 22); c.arrow('M385 70 L434 70', 'text');
    r.order.forEach((index, rank) => { const x = 532 + rank * 215; imageGrid(c, data.images[index], x - 50, 23, 25); c.text(x, 155, (rank + 1) + '. ' + data.imageNames[index], 'image', 17); c.text(x, 190, 'cos = ' + r.cosine[index].toFixed(3), 'match', 22); });
    c.text(550, 238, 'Retrieval selects an existing item. CLIP does not write a new caption.', 'muted', 20);
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
    const tau = document.createElement('input'); tau.type = 'range'; tau.min = '.05'; tau.max = '1'; tau.step = '.01'; tau.value = '.2'; tau.setAttribute('aria-label', 'Inference temperature'); tauLabel.append(tau);
    const value = document.createElement('output'); tauLabel.append(value);
    const output = document.createElement('div'), summary = document.createElement('p'); summary.setAttribute('aria-live', 'polite');
    function render() {
      let captions = set.value === 'missing' ? data.captions.slice(1) : data.captions.slice();
      if (set.value === 'duplicate') captions.push(data.captions[0]);
      if (template.value === 'grid') captions = captions.map(c => 'a grid with ' + c);
      const r = classify({ captions, tau: Number(tau.value) }); value.textContent = Number(tau.value).toFixed(2);
      table(output, r.cosine.map((score, i) => [score, r.logits[i], r.probabilities[i]]), { cols: ['cosine', 'logit', 'candidate p'], rowLabels: captions, cornerLabel: 'Two-square image vs prompt' });
      summary.textContent = `Highest-scoring candidate: ${r.label}. The image and encoder weights are unchanged.`;
    }
    for (const input of [template, set, tau]) input.addEventListener('input', render);
    controls.append(template, set, tauLabel); host.append(controls, output, summary); render();
  }
  AT.clip = { data: clone(data), params, forward, gradients, step, classify, retrieve, words, unit, softmax, diagram, table, trainingWidget, candidateWidget };
  AT.axes.named = false;
  const defs = { e: ['g_img', 'image embedding', 'One global image vector after its image encoder. A hat marks unit normalization.'], q: ['g_txt', 'text embedding', 'One global caption or prompt vector after its text encoder. This is not an attention query.'], a: ['p', 'candidate probability', 'A softmax over the supplied captions or images, not confidence over all possible descriptions.'], d: ['L', 'contrastive loss', 'Average image-to-text and text-to-image cross-entropy on the observed pairs.'] };
  AT.objects.forEach(object => { if (defs[object.cls]) { const d = defs[object.cls]; object.sym = d[0]; object.name = d[1]; object.def = d[2]; object.tip = d[2]; } });
  function note(sym, meaning, shape, size, group = 'Image-text matching') { AT.notation.push({ g: group, sym, mean: meaning, shape, dims: () => size, parts: ['vision3'] }); }
  note('g_i^{\\mathrm{img}},g_j^{\\mathrm{txt}}', 'Global image and text vectors before unit normalization; not attention values', '1\\times d', '1×3 in the toy');
  note('\\hat g = g/\\lVert g\\rVert_2', 'A unit-length vector; its direction is compared with the other modality', '1\\times d', '1×3');
  note('C_{ij}=\\hat g_i^{\\mathrm{img}}(\\hat g_j^{\\mathrm{txt}})^\\top', 'Cosine similarity of image i and caption j', '\\text{scalar}', '−1 to 1');
  note('S_{ij}=C_{ij}/\\tau', 'Temperature-scaled logit; it is not itself a probability', '\\text{scalar}', '');
  note('p(T_j\\mid I_i,\\mathcal B)', 'Row softmax over the captions in this batch or candidate set', '\\text{scalar}', '0 to 1');
  note('p(I_i\\mid T_j,\\mathcal B)', 'Column softmax over images in this batch', '\\text{scalar}', '0 to 1');
  note('L=(L_{I\\to T}+L_{T\\to I})/2', 'Mean of the two batch-average cross-entropies', '\\text{scalar}', '');
  note('N,d', 'Number of paired examples; shared embedding width', '', '3 pairs; 3 coordinates', 'sizes');
  note('W_{\\mathrm{img}},W_{\\mathrm{txt}}', 'Toy learned pixel map and word-row table', '16\\times3,\\;11\\times3', '81 parameters', 'sizes');
  note('a=\\log(1/\\tau)', 'Learned log scale; logits equal exp(a) times cosine', '\\text{scalar}', '1 parameter', 'sizes');
  const originalNotation = AT.ui.notationCard;
  AT.ui.notationCard = function (options = {}) { return originalNotation(Object.assign({}, options, options.part === 'vision3' ? { groups: ['Image-text matching', 'sizes'] } : {})); };
})();

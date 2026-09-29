/* Trained ViT measurements. Pure math is shared with the numerical checks. */
(function () {
  'use strict';
  const ROWS = 197, WIDTH = 64, FEATURES = 192, QSIZE = 3 * ROWS * WIDTH;
  function decode(buffer) {
    if (buffer.byteLength !== (2 * QSIZE + ROWS * FEATURES) * 4) throw new Error('Incomplete block data');
    const view = new DataView(buffer), values = new Float32Array(buffer.byteLength / 4);
    for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
    if (!values.every(Number.isFinite)) throw new Error('Invalid block values');
    return {q: values.subarray(0, QSIZE), k: values.subarray(QSIZE, 2 * QSIZE), features: values.subarray(2 * QSIZE)};
  }
  function attention(data, head, query) {
    const scores = [], offset = head * ROWS * WIDTH, qi = offset + query * WIDTH;
    for (let j = 0; j < ROWS; j++) {
      let sum = 0;
      for (let d = 0; d < WIDTH; d++) sum += data.q[qi + d] * data.k[offset + j * WIDTH + d];
      scores.push(sum / Math.sqrt(WIDTH));
    }
    const maximum = Math.max(...scores), exps = scores.map(x => Math.exp(x - maximum));
    const total = exps.reduce((a, b) => a + b, 0);
    return {scores, values: exps.map(x => x / total)};
  }
  function similarity(data, query) {
    const values = [], f = data.features, qi = query * FEATURES;
    let qnorm = 0;
    for (let d = 0; d < FEATURES; d++) qnorm += f[qi + d] ** 2;
    for (let j = 0; j < ROWS; j++) {
      let dot = 0, norm = 0;
      for (let d = 0; d < FEATURES; d++) { dot += f[qi + d] * f[j * FEATURES + d]; norm += f[j * FEATURES + d] ** 2; }
      values.push(Math.max(-1, Math.min(1, dot / Math.sqrt(qnorm * norm))));
    }
    return {values};
  }
  if (typeof module !== 'undefined' && module.exports) { module.exports = {decode, attention, similarity}; return; }
  document.addEventListener('DOMContentLoaded', () => {
    const root = document.getElementById('vit-explorer');
    if (!root) return;
    const frame = root.closest('.frame'), find = s => root.querySelector(s);
    const mode = find('[data-control="mode"]'), block = find('[data-control="block"]'), head = find('[data-control="head"]');
    const status = find('[data-status]'), play = find('[data-play]'), retry = find('[data-retry]');
    const canvas = find('canvas'), ctx = canvas.getContext('2d');
    const base = root.dataset.base, cache = new Map();
    let meta, data, row, query = 74, source = 60, version = 0, timer, playing = false, ready = false;
    const name = j => j === 0 ? 'CLS' : `P${j}`;
    const location = j => j === 0 ? 'image summary' : `row ${Math.floor((j - 1) / 14) + 1}, column ${(j - 1) % 14 + 1}`;
    const percent = n => n * 100 < .01 ? `${(n * 100).toPrecision(2)}%` : `${(n * 100).toFixed(2)}%`;
    const fixed = n => n.toFixed(3).replace('-', '−');
    const isAttention = () => mode.value === 'attention';
    const queryButtons = [], keyButtons = [];
    function grid(target, buttons, type) {
      for (let j = 1; j <= 196; j++) {
        const b = document.createElement('button'); b.type = 'button'; b.tabIndex = j === 74 ? 0 : -1;
        b.setAttribute('aria-label', `${type} ${name(j)}, ${location(j)}`);
        b.addEventListener('click', () => { stop(); if (type === 'Query') { query = j; render(); } else { source = j; renderSource(); } });
        b.addEventListener('keydown', e => {
          const delta = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -14, ArrowDown: 14}[e.key];
          if (!delta) return;
          e.preventDefault(); e.stopPropagation();
          const next = Math.max(1, Math.min(196, j + delta));
          buttons.forEach((el, i) => { el.tabIndex = i === next - 1 ? 0 : -1; });
          buttons[next - 1].focus();
        });
        target.append(b); buttons.push(b);
      }
    }
    grid(find('[data-query-grid]'), queryButtons, 'Query');
    grid(find('[data-key-grid]'), keyButtons, 'Source');
    function stop() { playing = false; clearTimeout(timer); play.textContent = '▶ Play blocks'; play.setAttribute('aria-pressed', 'false'); }
    async function read(url, binary = false) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Could not load data (${response.status})`);
      return binary ? response.arrayBuffer() : response.json();
    }
    function load(n) {
      if (!cache.has(n)) {
        cache.set(n, read(`${base}/${meta.files[n - 1].file}`, true).then(decode).catch(error => { cache.delete(n); throw error; }));
      }
      return cache.get(n);
    }
    function thumbnail(el, j) {
      el.style.backgroundImage = j ? `url(${root.dataset.image})` : 'none';
      el.style.backgroundSize = '1400% 1400%';
      el.style.backgroundPosition = j ? `${((j - 1) % 14) / 13 * 100}% ${Math.floor((j - 1) / 14) / 13 * 100}%` : '';
      el.textContent = j ? '' : 'CLS';
    }
    function renderSource() {
      if (!ready || !row) return;
      keyButtons.forEach((b, i) => { b.classList.toggle('vix-source', i + 1 === source); b.setAttribute('aria-pressed', String(i + 1 === source)); });
      find('[data-source-name]').textContent = `${name(source)} · ${location(source)}`;
      thumbnail(find('[data-source-crop]'), source);
      if (isAttention()) {
        find('[data-calculation]').textContent = `q(${name(query)}) · k(${name(source)}) / √64 = ${fixed(row.scores[source])}`;
        find('[data-result]').textContent = `Softmax across all 197 sources → ${percent(row.values[source])}`;
        find('[data-source-note]').textContent = `This is the weight multiplying ${name(source)}’s 64-number value row.`;
      } else {
        find('[data-calculation]').textContent = `Compare the two 192-number patch representations.`;
        find('[data-result]').textContent = `Cosine similarity = ${fixed(row.values[source])}`;
        find('[data-source-note]').textContent = '1 means the feature vectors point in the same direction.';
      }
    }
    function render() {
      if (!ready) return;
      row = isAttention() ? attention(data, Number(head.value) - 1, query) : similarity(data, query);
      const values = row.values;
      find('[data-query-name]').textContent = `${name(query)} · ${location(query)}`;
      thumbnail(find('[data-query-crop]'), query);
      find('[data-query-readout]').textContent = `Query: ${name(query)}`;
      head.disabled = !isAttention();
      find('[data-preset="0"]').disabled = !isAttention();
      queryButtons.forEach((b, i) => {
        b.classList.toggle('vix-query', i + 1 === query); b.setAttribute('aria-pressed', String(i + 1 === query));
        b.tabIndex = i + 1 === (query || 1) ? 0 : -1;
      });
      const patchValues = values.slice(1), maximum = Math.max(...patchValues);
      ctx.clearRect(0, 0, 280, 280);
      patchValues.forEach((v, i) => {
        // Attention contrast rescales per map; cosine always uses the fixed −1..1 scale.
        const strength = isAttention() ? v / maximum : Math.abs(v);
        ctx.fillStyle = !isAttention() && v < 0 ? `rgba(37,99,235,${strength * .85})` : `rgba(255,177,30,${strength * .9})`;
        ctx.fillRect(i % 14 * 20, Math.floor(i / 14) * 20, 20, 20);
        keyButtons[i].setAttribute('aria-label', `Source ${name(i + 1)}, ${location(i + 1)}, ${isAttention() ? 'weight ' + percent(v) : 'cosine ' + fixed(v)}`);
        keyButtons[i].title = `${name(i + 1)}: ${isAttention() ? percent(v) : fixed(v)}`;
        keyButtons[i].classList.toggle('vix-query-location', i + 1 === query);
      });
      const ranked = Array.from({length: 196}, (_, i) => i + 1).filter(j => isAttention() || j !== query).sort((a, b) => values[b] - values[a]);
      source = ranked[0];
      find('[data-map-title]').textContent = isAttention() ? '2 · See source weights' : '2 · Find similar patches';
      find('[data-ranking-label]').textContent = isAttention() ? 'Strongest patch sources' : 'Most similar other patches';
      const top = find('[data-top]'); top.replaceChildren();
      ranked.slice(0, 3).forEach(j => {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'vix-match';
        const crop = document.createElement('span'); crop.className = 'vix-crop'; thumbnail(crop, j);
        const label = document.createElement('span'); label.textContent = `${name(j)} · ${isAttention() ? percent(values[j]) : fixed(values[j])}`;
        button.append(crop, label); button.addEventListener('click', () => { stop(); source = j; renderSource(); }); top.append(button);
      });
      find('[data-scale]').classList.toggle('vix-cosine-scale', !isAttention());
      find('[data-scale-label]').textContent = isAttention() ? `0 → ${percent(maximum)} · contrast adapts to this map` : '−1 (blue) → 0 (clear) → 1 (gold) · fixed scale';
      const cls = find('[data-cls-weight]'); cls.hidden = !isAttention();
      cls.textContent = `CLS source: ${percent(values[0])}`;
      find('[data-accounting]').textContent = isAttention() ? '196 patch weights + the CLS weight = 100%.' : 'After this block’s attention + MLP. The selected patch matches itself at 1; it is omitted from the ranking.';
      find('[data-meaning]').textContent = isAttention() ? 'Gold = more of that source’s value enters the query’s message. Attention can favor background; it is not object matching.' : 'Gold = similar patch features. This resembles the DINO interaction, using our trained classification ViT; it is not attention.';
      find('[data-guide]').textContent = isAttention() ? 'Keep the query fixed. Change the head or play through the blocks.' : 'Try Ear at block 12: inspect patches on the other side of the dog.';
      status.textContent = `Trained ViT · Block ${block.value}${isAttention() ? ' · Head ' + head.value : ' · feature similarity'} · ${name(query)}`;
      renderSource();
    }
    async function update() {
      const request = ++version;
      ready = false; root.classList.add('vix-loading'); root.setAttribute('aria-busy', 'true'); retry.hidden = true;
      status.textContent = `Loading trained measurements for block ${block.value}…`;
      try {
        if (!meta) meta = await read(`${base}/manifest.json`);
        const next = await load(Number(block.value));
        if (request !== version) return;
        data = next; ready = true; root.classList.add('vix-ready');
        root.classList.remove('vix-loading'); root.setAttribute('aria-busy', 'false'); render();
        if (playing) timer = setTimeout(tick, 1400);
      } catch (error) {
        if (request !== version) return;
        stop(); root.classList.remove('vix-loading'); root.setAttribute('aria-busy', 'false');
        status.textContent = 'Interactive data could not load. The saved example below is still available. Retry with a connection.';
        root.classList.remove('vix-ready'); retry.hidden = false;
      }
    }
    function tick() {
      if (!playing) return;
      if (Number(block.value) === 12) { stop(); return; }
      block.value = String(Number(block.value) + 1); update();
    }
    mode.addEventListener('change', () => { stop(); if (!isAttention() && query === 0) query = 74; render(); });
    block.addEventListener('change', () => { stop(); update(); });
    head.addEventListener('change', () => { stop(); render(); });
    root.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => { stop(); query = Number(button.dataset.preset); render(); }));
    find('[data-cls-weight]').addEventListener('click', () => { stop(); source = 0; renderSource(); });
    play.addEventListener('click', () => {
      if (playing) { stop(); return; }
      playing = true; play.textContent = 'Ⅱ Pause'; play.setAttribute('aria-pressed', 'true'); block.value = '1'; update();
    });
    retry.addEventListener('click', update);
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
    new MutationObserver(() => { if (document.body.classList.contains('present') && !frame.classList.contains('is-live')) stop(); }).observe(frame, {attributes: true, attributeFilter: ['class']});
    if (window.IntersectionObserver) {
      const observer = new IntersectionObserver(entries => { if (entries[0].isIntersecting) { update(); observer.disconnect(); } });
      observer.observe(root);
    } else update();
  });
})();

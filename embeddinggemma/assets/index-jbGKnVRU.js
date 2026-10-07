(function polyfill() {
  const relList = document.createElement("link").relList;
  if (relList && relList.supports && relList.supports("modulepreload")) return;
  for (const link of document.querySelectorAll('link[rel="modulepreload"]')) processPreload(link);
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type !== "childList") continue;
      for (const node of mutation.addedNodes) if (node.tagName === "LINK" && node.rel === "modulepreload") processPreload(node);
    }
  }).observe(document, {
    childList: true,
    subtree: true
  });
  function getFetchOpts(link) {
    const fetchOpts = {};
    if (link.integrity) fetchOpts.integrity = link.integrity;
    if (link.referrerPolicy) fetchOpts.referrerPolicy = link.referrerPolicy;
    if (link.crossOrigin === "use-credentials") fetchOpts.credentials = "include";
    else if (link.crossOrigin === "anonymous") fetchOpts.credentials = "omit";
    else fetchOpts.credentials = "same-origin";
    return fetchOpts;
  }
  function processPreload(link) {
    if (link.ep) return;
    link.ep = true;
    const fetchOpts = getFetchOpts(link);
    fetch(link.href, fetchOpts);
  }
})();
function unit(vector, dimensions = vector.length) {
  const v = Array.from(vector).slice(0, dimensions);
  if (!v.length || !v.every(Number.isFinite))
    throw Error("Expected a finite, non-empty vector.");
  const norm = Math.hypot(...v);
  if (norm < 1e-10)
    throw Error("These vectors have almost no difference to compare.");
  return v.map((x) => x / norm);
}
function dot(a, b) {
  if (a.length !== b.length)
    throw Error("Both vectors must have the same dimension.");
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}
function rank(query, items, vectors, dimensions = 768, excluded = []) {
  const q = unit(query, dimensions);
  return items.filter((item) => vectors[item.id] && !excluded.includes(item.id)).map((item) => ({
    item,
    score: dot(q, unit(vectors[item.id].vector, dimensions))
  })).sort((a, b) => b.score - a.score || a.item.id.localeCompare(b.item.id));
}
function difference(after, before) {
  return unit(after.map((v, i) => v - before[i]));
}
function kmeans(rows, k = 4) {
  const centers = [rows[0].slice()];
  while (centers.length < k) {
    let best = 0, score = -Infinity;
    rows.forEach((r, i) => {
      const d = Math.min(
        ...centers.map((c) => r.reduce((s, x, j) => s + (x - c[j]) ** 2, 0))
      );
      if (d > score) {
        best = i;
        score = d;
      }
    });
    centers.push(rows[best].slice());
  }
  let labels = [];
  for (let iteration = 0; iteration < 30; iteration++) {
    const next = rows.map(
      (r) => centers.map((c) => r.reduce((s, x, j) => s + (x - c[j]) ** 2, 0)).reduce((best, d, i, a) => d < a[best] ? i : best, 0)
    );
    if (next.every((v, i) => v === labels[i])) break;
    labels = next;
    centers.forEach((c, g) => {
      const members = rows.filter((_, i) => labels[i] === g);
      if (members.length)
        centers[g] = c.map(
          (_, j) => members.reduce((s, r) => s + r[j], 0) / members.length
        );
    });
  }
  return labels;
}
function pca(rows) {
  const d = rows[0].length, mean = rows[0].map(
    (_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length
  ), x = rows.map((r) => r.map((v, j) => v - mean[j]));
  const axes = [];
  for (let a = 0; a < 2; a++) {
    let v = unit(
      Array.from({ length: d }, (_, i) => Math.sin((i + 1) * (a + 1.137)))
    );
    for (let it = 0; it < 60; it++) {
      let next = Array(d).fill(0);
      x.forEach((r) => {
        const projection = dot(r, v);
        r.forEach((z, j) => next[j] += z * projection);
      });
      axes.forEach((axis) => {
        const component = dot(next, axis);
        next = next.map((z, j) => z - component * axis[j]);
      });
      if (Math.hypot(...next) < 1e-10) break;
      v = unit(next);
    }
    axes.push(v);
  }
  return x.map((r) => axes.map((a) => dot(r, a)));
}
function pcaProjection(rows) {
  if (!rows.length || rows.some((r) => r.length !== rows[0].length || !r.every(Number.isFinite)))
    throw Error("PCA needs a rectangular finite matrix.");
  const mean = rows[0].map(
    (_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length
  );
  const centered = rows.map((r) => r.map((v, j) => v - mean[j]));
  const total = centered.reduce((s, r) => s + dot(r, r), 0);
  const points = pca(rows);
  const variance = [0, 1].map(
    (j) => total ? points.reduce((s, p) => s + p[j] ** 2, 0) / total : 0
  );
  return { points, variance };
}
function softmax(logits) {
  const max = Math.max(...logits), e = logits.map((x) => Math.exp(x - max)), sum = e.reduce((s, x) => s + x, 0);
  return e.map((x) => x / sum);
}
function makeHead(dimensions, classes) {
  return {
    weights: Array.from({ length: classes }, () => Array(dimensions).fill(0)),
    bias: Array(classes).fill(0)
  };
}
function headPredict(head, x) {
  return softmax(head.weights.map((w, c) => dot(w, x) + head.bias[c]));
}
function headMetrics(head, rows, labels) {
  const predictions = rows.map((x) => headPredict(head, x));
  return {
    loss: predictions.reduce(
      (s, p, i) => s - Math.log(Math.max(p[labels[i]], 1e-30)),
      0
    ) / rows.length,
    accuracy: predictions.reduce(
      (s, p, i) => s + Number(p.indexOf(Math.max(...p)) === labels[i]),
      0
    ) / rows.length,
    predictions
  };
}
function trainHeadStep(head, rows, labels, rate = 2, decay = 1e-3) {
  const gw = head.weights.map((w) => w.map(() => 0)), gb = head.bias.map(() => 0);
  rows.forEach((x, i) => {
    const p = headPredict(head, x);
    p.forEach((v, c) => {
      const error2 = (v - Number(c === labels[i])) / rows.length;
      gb[c] += error2;
      x.forEach((a, j) => gw[c][j] += error2 * a);
    });
  });
  head.weights.forEach(
    (w, c) => w.forEach((a, j) => w[j] -= rate * (gw[c][j] + decay * a))
  );
  head.bias.forEach((a, c) => head.bias[c] -= rate * gb[c]);
}
const MODEL = "onnx-community/embeddinggemma-2-ONNX";
const REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";
const escapeHTML = (s) => String(s ?? "").replace(
  /[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
);
const esc$1 = escapeHTML;
const colours = {
  image: "#bc5538",
  text: "#476b9b",
  audio: "#238178",
  video: "#835f9f"
};
function preview(item) {
  if (item.type === "image")
    return `<img src="${esc$1(item.src)}" alt="${esc$1(item.title)}" loading="lazy">`;
  if (item.type === "audio")
    return `<div class="audio-label">AUDIO · ${item.duration || 5} SECONDS</div><audio controls preload="none" src="${esc$1(item.src)}"></audio>`;
  if (item.type === "video")
    return `<video controls playsinline preload="metadata" src="${esc$1(item.src)}#t=${item.start || 0.05},${item.end || 20}" aria-label="${esc$1(item.title)}"></video>`;
  return `<blockquote>${esc$1(item.text)}</blockquote>`;
}
function saveJSON(data, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
  );
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1e3);
}
function drawVector(canvas, v, max) {
  const ctx = canvas.getContext("2d"), cols = 48, w = canvas.clientWidth || 500, h = Math.ceil(v.length / cols) * 10, dpr = devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + "px";
  ctx.scale(dpr, dpr);
  v.forEach((x, i) => {
    ctx.fillStyle = x >= 0 ? `rgba(35,129,120,${0.08 + 0.92 * Math.abs(x) / max})` : `rgba(188,85,56,${0.08 + 0.92 * Math.abs(x) / max})`;
    ctx.fillRect(
      i % cols * w / cols,
      Math.floor(i / cols) * 10,
      w / cols - 1,
      9
    );
  });
}
function mountExplorer(root, state2) {
  const items = state2.gallery.filter((i) => state2.vectors[i.id]);
  let selected = "sound-dog", compare = "newfoundland", dims = 768, projection, visible = new Set(Object.keys(colours)), zoom = 1, pan = [0, 0];
  const opts = () => Object.keys(colours).map(
    (type) => `<optgroup label="${type.toUpperCase()}">${items.filter((i) => i.type === type).map((i) => `<option value="${esc$1(i.id)}">${esc$1(i.title)}</option>`).join("")}</optgroup>`
  ).join("");
  root.innerHTML = `<div class="explorer-intro"><p><strong>Start with a bark.</strong> Select its point, play the recording, then compare it with a dog photograph. Do their closest neighbours agree?</p><span class="saved-tag">Stored real embeddings · no model download</span></div>
 <div class="space-toolbar"><label>Choose an item<select id="space-item">${opts()}</select></label><label>Compare with<select id="space-compare"><option value="">No comparison</option>${opts()}</select></label><label>Vector size<select id="space-dim"><option value="768">768</option><option>512</option><option>256</option><option>128</option></select></label></div>
 <div class="space-layout"><section class="space-plot"><header><div><p class="step-label">01 / SEE THE COLLECTION</p><h3>A map of the shared space</h3></div><div class="zoom-tools"><button class="quiet" id="zoom-out" aria-label="Zoom out">−</button><button class="quiet" id="zoom-in" aria-label="Zoom in">+</button><button class="quiet" id="zoom-reset">Reset view</button></div></header><div class="modality-legend">${Object.entries(
    colours
  ).map(
    ([t, c]) => `<label style="--modality:${c}"><input type="checkbox" value="${t}" checked><i></i>${t}<span>${items.filter((i) => i.type === t).length}</span></label>`
  ).join(
    ""
  )}</div><div id="space-chart"></div><p id="space-variance" class="hint"></p><div class="pan-tools" aria-label="Pan the map"><span>Drag to pan, or use</span><button class="quiet" data-pan="left" aria-label="Pan left">←</button><button class="quiet" data-pan="right" aria-label="Pan right">→</button><button class="quiet" data-pan="up" aria-label="Pan up">↑</button><button class="quiet" data-pan="down" aria-label="Pan down">↓</button></div><details class="space-explanation"><summary>How do 768 numbers become two coordinates?</summary><ol><li>Stack all ${items.length} unit vectors into one matrix.</li><li>Subtract the mean vector. PCA finds the two directions with the most variation.</li><li>Project each centred vector onto those directions. Colour records the input modality; it does not change the calculation.</li></ol><p>The two axes lose information. Close points on this map need not be nearest neighbours in the original space. The neighbour list uses cosine in all selected dimensions. Text candidates keep their retrieval document prefix; this is a view of our retrieval index.</p><p>Filters hide points without moving the axes. Changing vector size recomputes PCA.</p></details></section><aside id="space-detail" class="space-detail"></aside></div>
 <section class="full-vector"><header><div><p class="step-label">02 / FOLLOW THE NUMBERS</p><h3 id="space-vector-title"></h3></div><button id="space-export" class="quiet">Download these vectors</button></header><div id="space-vector-summary"></div><div class="vector-pair"><div><h4 id="space-a-title"></h4><canvas id="space-a" aria-label="Selected embedding heatmap"></canvas></div><div id="space-b-wrap"><h4 id="space-b-title"></h4><canvas id="space-b" aria-label="Comparison embedding heatmap"></canvas></div></div><p class="hint">Teal = positive · rust = negative · the same scale for both vectors. These coordinates are learned numbers, not named concepts.</p><details id="all-coordinates"><summary id="coordinate-summary">See every coordinate</summary><div class="coordinate-search"><label>Jump to coordinate<input id="jump-coordinate" type="number" min="1" max="768" value="1"></label><button class="quiet" id="jump-go">Go</button></div><div class="full-coordinate-table" tabindex="0"><table><thead id="space-table-head"></thead><tbody id="space-table"></tbody></table></div></details><details><summary>Actual model inputs and tensor shapes</summary><pre id="space-shapes"></pre></details></section>`;
  const $2 = (s) => root.querySelector(s), item = (id) => items.find((i) => i.id === id), vector = (id) => unit(state2.vectors[id].vector, dims);
  $2("#space-item").value = selected;
  $2("#space-compare").value = compare;
  function calculate() {
    projection = pcaProjection(items.map((i) => vector(i.id)));
    zoom = 1;
    pan = [0, 0];
  }
  function chart() {
    const p = projection.points, maxX = Math.max(...p.map((x) => Math.abs(x[0])), 0.01), maxY = Math.max(...p.map((x) => Math.abs(x[1])), 0.01), sx = (x) => 360 + x / maxX * 290, sy = (y) => 220 - y / maxY * 170;
    const selectedIndex = items.findIndex((i) => i.id === selected), compareIndex = items.findIndex((i) => i.id === compare);
    const line = compareIndex >= 0 && visible.has(item(selected).type) && visible.has(item(compare).type) ? `<line class="comparison-line" x1="${sx(p[selectedIndex][0])}" y1="${sy(p[selectedIndex][1])}" x2="${sx(p[compareIndex][0])}" y2="${sy(p[compareIndex][1])}" stroke="#424b50" stroke-dasharray="5 5"/>` : "";
    $2("#space-chart").innerHTML = `<svg viewBox="0 0 720 440" role="group" aria-label="Interactive PCA map, coloured by modality"><defs><clipPath id="plot-clip"><rect x="28" y="15" width="664" height="400"/></clipPath></defs><path d="M28 220H692M360 15V415" stroke="#e0dfd7"/><text x="674" y="437" class="axis-label">PC1</text><text x="6" y="22" class="axis-label">PC2</text><g clip-path="url(#plot-clip)"><g id="pan-layer" transform="translate(${360 + pan[0]},${220 + pan[1]}) scale(${zoom}) translate(-360,-220)">${line}${items.map((i, n) => visible.has(i.type) ? `<g class="space-point ${i.id === selected ? "chosen" : ""}" role="button" tabindex="0" aria-label="${esc$1(i.type + ": " + i.title)}" data-point="${esc$1(i.id)}"><circle cx="${sx(p[n][0])}" cy="${sy(p[n][1])}" r="${i.id === selected || i.id === compare ? 9 : 5.5}" fill="${colours[i.type]}" stroke="${i.id === selected ? "#202621" : i.id === compare ? "#fff" : "transparent"}" stroke-width="${i.id === selected || i.id === compare ? 2.5 : 0}"/><title>${esc$1(i.type + " · " + i.title)}</title></g>` : "").join("")}</g></g></svg>`;
    $2("#space-variance").textContent = `PCA of ${items.length} × ${dims} numbers · PC1 ${(projection.variance[0] * 100).toFixed(1)}% + PC2 ${(projection.variance[1] * 100).toFixed(1)}% = ${(projection.variance.reduce((a, b) => a + b) * 100).toFixed(1)}% of variation shown. Zoom ${zoom.toFixed(1)}×.`;
    const svg = $2("#space-chart svg");
    let drag = null;
    svg.onpointerdown = (e) => {
      if (e.target.closest("[data-point]")) return;
      drag = [e.clientX, e.clientY, ...pan];
      svg.setPointerCapture(e.pointerId);
    };
    svg.onpointermove = (e) => {
      if (!drag) return;
      const scale = 720 / svg.getBoundingClientRect().width;
      pan = [
        drag[2] + (e.clientX - drag[0]) * scale,
        drag[3] + (e.clientY - drag[1]) * scale
      ];
      $2("#pan-layer").setAttribute(
        "transform",
        `translate(${360 + pan[0]},${220 + pan[1]}) scale(${zoom}) translate(-360,-220)`
      );
    };
    svg.onpointerup = () => {
      drag = null;
    };
    svg.onpointercancel = () => {
      drag = null;
    };
  }
  function details() {
    const a = item(selected), b = item(compare), va = vector(selected), vb = b ? vector(compare) : null;
    const neighbours = rank(
      va,
      items.filter((i) => visible.has(i.type)),
      state2.vectors,
      dims,
      [selected]
    ).slice(0, 5);
    $2("#space-detail").innerHTML = `<span class="type-badge" style="color:${colours[a.type]}">${a.type.toUpperCase()}</span><h3>${esc$1(a.title)}</h3><div class="selected-media">${preview(a)}</div><p class="fineprint">${esc$1(a.credit)}</p><h4>Closest in ${dims} dimensions</h4><p class="hint">Cosine similarity, using the visible modalities.</p><div class="space-neighbours">${neighbours.map((r) => `<button data-neighbour="${r.item.id}"><i style="background:${colours[r.item.type]}"></i><span>${esc$1(r.item.title)}<small>${r.item.type}</small></span><b>${r.score.toFixed(4)}</b></button>`).join("") || "<p>No visible neighbours. Turn on a modality.</p>"}</div>${b ? `<div class="pair-score">${b.type === "image" ? `<img src="${esc$1(b.src)}" alt="${esc$1(b.title)}">` : ""}<span>${esc$1(a.title)}<br>↕<br>${esc$1(b.title)}</span><strong>${dot(va, vb).toFixed(4)}</strong><small>cosine in ${dims} dimensions</small></div>` : ""}`;
    $2("#space-vector-title").textContent = `One item → ${dims} numbers`;
    $2("#space-vector-summary").innerHTML = `<div class="under-hood"><span>${esc$1(a.type)} input</span><span>→</span><span>EmbeddingGemma 2</span><span>→</span><span>[1, 768]</span><span>→</span><span>${dims === 768 ? "L2-normalize" : `keep ${dims} + normalize`}</span></div><p>Length of the selected vector = <b>${Math.hypot(...va).toFixed(6)}</b>${vb ? ` · Σ aₖbₖ = <b>${dot(va, vb).toFixed(6)}</b> across all ${dims} coordinates.` : "."}</p>`;
    $2("#space-a-title").textContent = a.title;
    $2("#space-b-wrap").hidden = !b;
    if (b) $2("#space-b-title").textContent = b.title;
    const max = Math.max(...va.map(Math.abs), ...(vb || []).map(Math.abs));
    drawVector($2("#space-a"), va, max);
    if (b) drawVector($2("#space-b"), vb, max);
    $2("#coordinate-summary").textContent = `See all ${dims} coordinates${b ? " and their dot-product contributions" : ""}`;
    $2("#space-table-head").innerHTML = `<tr><th>Coordinate</th><th>Selected vector</th>${b ? "<th>Comparison vector</th><th>aₖ × bₖ</th>" : ""}</tr>`;
    $2("#space-table").innerHTML = va.map(
      (v, k) => `<tr id="coordinate-${k + 1}"><td>${k + 1}</td><td>${v.toFixed(7)}</td>${b ? `<td>${vb[k].toFixed(7)}</td><td>${(v * vb[k]).toFixed(7)}</td>` : ""}</tr>`
    ).join("");
    $2("#jump-coordinate").max = dims;
    $2("#space-shapes").textContent = JSON.stringify(
      {
        model: MODEL,
        revision: REVISION,
        selected: { title: a.title, ...state2.vectors[a.id], vector: void 0 },
        ...b ? {
          comparison: {
            title: b.title,
            ...state2.vectors[b.id],
            vector: void 0
          }
        } : {},
        outputShape: [1, 768],
        displayedShape: [1, dims]
      },
      null,
      2
    );
  }
  function render() {
    chart();
    details();
  }
  $2("#space-item").onchange = (e) => {
    selected = e.target.value;
    render();
  };
  $2("#space-compare").onchange = (e) => {
    compare = e.target.value;
    render();
  };
  $2("#space-dim").onchange = (e) => {
    dims = +e.target.value;
    calculate();
    render();
  };
  root.querySelectorAll(".modality-legend input").forEach(
    (n) => n.onchange = () => {
      n.checked ? visible.add(n.value) : visible.delete(n.value);
      render();
    }
  );
  $2("#zoom-in").onclick = () => {
    zoom = Math.min(5, zoom * 1.4);
    chart();
  };
  $2("#zoom-out").onclick = () => {
    zoom = Math.max(0.5, zoom / 1.4);
    chart();
  };
  $2("#zoom-reset").onclick = () => {
    zoom = 1;
    pan = [0, 0];
    chart();
  };
  root.onclick = (e) => {
    const p = e.target.closest("[data-point]"), n = e.target.closest("[data-neighbour]"), arrow = e.target.closest("[data-pan]");
    if (p) {
      selected = p.dataset.point;
      $2("#space-item").value = selected;
      render();
    }
    if (n) {
      compare = n.dataset.neighbour;
      $2("#space-compare").value = compare;
      render();
    }
    if (arrow) {
      const moves = {
        left: [40, 0],
        right: [-40, 0],
        up: [0, 40],
        down: [0, -40]
      };
      pan = pan.map((v, i) => v + moves[arrow.dataset.pan][i]);
      chart();
    }
  };
  root.onkeydown = (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches("[data-point]")) {
      e.preventDefault();
      e.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
  };
  $2("#jump-go").onclick = () => {
    const k = Math.max(1, Math.min(dims, +$2("#jump-coordinate").value || 1));
    $2(`#coordinate-${k}`).scrollIntoView({ block: "nearest" });
  };
  $2("#space-export").onclick = () => saveJSON(
    {
      model: MODEL,
      revision: REVISION,
      dimensions: dims,
      selected: { id: selected, vector: vector(selected) },
      ...compare ? {
        comparison: { id: compare, vector: vector(compare) },
        cosine: dot(vector(selected), vector(compare))
      } : {},
      projection: {
        method: "PCA",
        explainedVariance: projection.variance,
        points: items.map((i, n) => ({
          id: i.id,
          type: i.type,
          xy: projection.points[n]
        }))
      }
    },
    "embedding-space.json"
  );
  calculate();
  render();
  return () => {
    root.onclick = null;
    root.onkeydown = null;
  };
}
function mountTraining(root, state2) {
  let head, epoch = 0, history2 = [], training = false, alive = true, classes = [], train = [], test = [], d = 768;
  const $2 = (s) => root.querySelector(s);
  root.innerHTML = `<div class="training-intro"><p><strong>Can a few labelled sounds teach a classifier?</strong> EmbeddingGemma has already represented each recording. Now learn a small output layer on those fixed vectors.</p><span class="saved-tag">Real gradient descent · runs locally · no model download</span></div><div class="training-architecture"><div><b>Sound recording</b><span>16 kHz mono waveform</span></div><i>→</i><div class="frozen-block"><b>EmbeddingGemma 2</b><span>Frozen · does not change</span></div><i>→</i><div><b id="train-vector-shape">768 numbers</b><span>L2-normalized embedding</span></div><i>→</i><div class="learning-block"><b>Linear layer + softmax</b><span>W and b learn from labels</span></div></div>
 <div class="training-controls"><label>Classification task<select id="train-task"><option value="3">3 sounds · dog, waves, fire</option><option value="10">All 10 sound categories</option></select></label><label>Embedding dimensions<select id="train-dim"><option>768</option><option>256</option><option>128</option></select></label><div class="training-buttons"><button id="train-step" class="quiet">Take one step</button><button id="train-run" class="primary">Train 100 steps</button><button id="train-reset" class="quiet">Reset</button></div></div>
 <div class="training-metrics" id="train-metrics" aria-live="polite"></div><div class="training-grid"><section><h3>Watch the loss change</h3><div id="training-chart"></div><p class="hint">Solid rust: training cross-entropy · dashed blue: held-out cross-entropy. A lower loss means more probability assigned to the recorded label. Training loss can fall while held-out predictions get worse.</p><div id="train-formula" class="training-formula"></div></section><section><h3>Exactly what is learning?</h3><p>Start W and b at zero: every class gets the same probability. Each step computes the training loss, differentiates it, and updates only W and b.</p><pre class="training-code">x = frozen_embeddings          # [N, d]
logits = x @ W.T + b            # [N, C]
loss = cross_entropy(logits, y)
loss.backward()                # gradients for W, b
optimizer.step()</pre><p id="train-shape" class="hint"></p><details><summary>Learning rate and update rule</summary><p>Full-batch gradient descent · learning rate 2 · weight penalty λ = 0.001.</p><p>∂L/∂s = (p − one_hot(y)) / N<br>∂L/∂W = (p − Y)ᵀX / N + λW<br>W ← W − 2 × ∂L/∂W</p><p>The chart shows cross-entropy alone. The small weight penalty is used in the update.</p></details><button class="quiet" id="train-export">Download learned weights</button></section></div>
 <section class="test-predictions"><div class="section-heading"><h3>Try the held-out recordings</h3><p>These recordings never enter the gradient update. Play them before revealing their labels.</p></div><p class="hint" id="split-note"></p><div id="test-cards"></div></section>
 <details class="training-split"><summary>See every training example and the split</summary><div id="split-table"></div></details>
 <section class="finetune-section"><p class="eyebrow">THE NEXT STEP</p><h3>What if we want to change the embedding space itself?</h3><p>Fine-tuning updates the encoder. Use positive matches and in-batch negatives so related inputs move closer in the representation. That is a different objective from our small classifier above.</p><div class="fine-tuning-path"><span>Image + caption<br>or audio + text pairs</span><b>→</b><span>EmbeddingGemma<br><small>weights can update</small></span><b>→</b><span>Contrastive loss<br><small>compare candidates in a batch</small></span></div><p>The official notebooks below run model fine-tuning on a GPU runtime. This browser lab only trains the small output layer.</p><div class="training-links"><a href="https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers" target="_blank" rel="noopener">Google: text fine-tuning ↗</a><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma2_%28300M%29-Image_Text.ipynb" target="_blank" rel="noopener">Image + text Colab ↗</a><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma2_%28300M%29-Audio.ipynb" target="_blank" rel="noopener">Audio Colab ↗</a></div></section>`;
  function reset() {
    epoch = 0;
    history2 = [];
    d = +$2("#train-dim").value;
    classes = $2("#train-task").value === "3" ? ["dog", "sea_waves", "crackling_fire"] : [
      ...new Set(
        state2.gallery.filter((x) => x.label).map((x) => x.label)
      )
    ];
    const data = state2.gallery.filter(
      (x) => x.type === "audio" && classes.includes(x.label) && state2.vectors[x.id]
    );
    train = data.filter((x) => x.split === "train");
    test = data.filter((x) => x.split === "test");
    head = makeHead(d, classes.length);
    record();
    render();
  }
  const rows = (data) => data.map((x) => unit(state2.vectors[x.id].vector, d)), labels = (data) => data.map((x) => classes.indexOf(x.label));
  function record() {
    history2.push({
      epoch,
      train: headMetrics(head, rows(train), labels(train)),
      test: headMetrics(head, rows(test), labels(test))
    });
  }
  function chart() {
    const width = 680, height = 250, max = Math.max(
      Math.log(classes.length),
      ...history2.flatMap((h) => [h.train.loss, h.test.loss])
    ) * 1.12, end = Math.max(100, epoch);
    const path = (type) => history2.map(
      (h, i) => `${i ? "L" : "M"}${45 + h.epoch / end * 610},${218 - h[type].loss / max * 190}`
    ).join(" ");
    $2("#training-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and held-out cross-entropy over ${epoch} steps"><path d="M45 20V218H660" fill="none" stroke="#d0cec6"/><path d="M45 ${218 - Math.log(classes.length) / max * 190}H660" stroke="#c5c2b8" stroke-dasharray="3 4"/><text x="50" y="${211 - Math.log(classes.length) / max * 190}">uniform guess: ln(${classes.length}) = ${Math.log(classes.length).toFixed(3)}</text><path d="${path("train")}" fill="none" stroke="#bc5538" stroke-width="3"/><path d="${path("test")}" fill="none" stroke="#476b9b" stroke-width="2.5" stroke-dasharray="6 4"/><text x="12" y="220">0</text><text x="43" y="242">0</text><text x="582" y="242">${end} steps</text></svg>`;
  }
  function render() {
    const m = history2.at(-1);
    $2("#train-metrics").innerHTML = `<div><span>Gradient steps</span><b>${epoch}</b></div><div><span>Training loss</span><b>${m.train.loss.toFixed(4)}</b></div><div><span>Held-out loss</span><b>${m.test.loss.toFixed(4)}</b></div><div><span>Held-out accuracy</span><b>${Math.round(m.test.accuracy * test.length)} / ${test.length}</b></div>`;
    chart();
    $2("#train-vector-shape").textContent = d + " numbers";
    $2("#train-shape").textContent = `X: [${train.length}, ${d}] · W: [${classes.length}, ${d}] · b: [${classes.length}] · ${classes.length * (d + 1)} trainable parameters.`;
    $2("#train-formula").innerHTML = `<b>L = −(1/N) Σ log p(correct class)</b><span>N = ${train.length} training recordings · ${classes.length} classes · uniform baseline = ${Math.log(classes.length).toFixed(4)}</span>`;
    $2("#split-note").textContent = `3 training recordings + 1 held-out recording per class, all from different original source recordings. Only ${test.length} held-out examples: an inspectable demonstration, not a reliable performance estimate. Repeatedly choosing settings against this tiny test set would bias the result.`;
    $2("#test-cards").innerHTML = test.map((item, i) => {
      const probs = m.test.predictions[i], best = probs.indexOf(Math.max(...probs));
      return `<article><p class="small-label">HELD-OUT RECORDING ${i + 1}</p><audio controls preload="none" src="${escapeHTML(item.src)}" aria-label="Held-out recording ${i + 1}"></audio><strong>Prediction: ${escapeHTML(classes[best].replaceAll("_", " "))}</strong><p class="hint">Softmax output · not calibrated confidence</p><div class="class-bars">${classes.map((c, j) => `<div><span>${escapeHTML(c.replaceAll("_", " "))}</span><i><b style="width:${probs[j] * 100}%"></b></i><code>${(probs[j] * 100).toFixed(1)}%</code></div>`).join("")}</div><details><summary>Reveal the dataset label</summary><p>${escapeHTML(item.label.replaceAll("_", " "))} · ${classes[best] === item.label ? "Correct prediction" : "Different from the prediction"}</p></details></article>`;
    }).join("");
    $2("#split-table").innerHTML = `<p>Original source IDs are kept separate across the split. Audio never uses filenames as model input.</p><div class="table-scroll"><table><thead><tr><th>Recording</th><th>Listen</th><th>Split</th><th>Source recording ID</th></tr></thead><tbody>${[...train, ...test].map((i) => `<tr><td>${escapeHTML(i.title)}</td><td><audio controls preload="none" src="${escapeHTML(i.src)}"></audio></td><td>${i.split === "test" ? "Held out" : "Training"}</td><td>${escapeHTML(i.sourceRecording)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  function step() {
    trainHeadStep(head, rows(train), labels(train));
    epoch++;
    record();
  }
  $2("#train-step").onclick = () => {
    step();
    render();
  };
  $2("#train-reset").onclick = reset;
  $2("#train-task").onchange = reset;
  $2("#train-dim").onchange = reset;
  $2("#train-run").onclick = async () => {
    if (training) return;
    training = true;
    root.querySelectorAll(".training-controls button,.training-controls select").forEach((n) => n.disabled = true);
    const x = rows(train), y = labels(train);
    for (let n = 0; n < 100 && alive; n++) {
      trainHeadStep(head, x, y);
      epoch++;
      record();
      if (n % 10 === 9) {
        render();
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }
    training = false;
    if (alive) {
      render();
      root.querySelectorAll(".training-controls button,.training-controls select").forEach((n) => n.disabled = false);
    }
  };
  $2("#train-export").onclick = () => saveJSON(
    {
      model: MODEL,
      revision: REVISION,
      frozenEncoder: true,
      dimensions: d,
      classes,
      head,
      steps: epoch,
      learningRate: 2,
      weightPenalty: 1e-3,
      trainIds: train.map((i) => i.id),
      heldOutIds: test.map((i) => i.id),
      history: history2.map((h) => ({
        step: h.epoch,
        trainLoss: h.train.loss,
        heldOutLoss: h.test.loss,
        heldOutAccuracy: h.test.accuracy
      }))
    },
    "trained-linear-head.json"
  );
  reset();
  return () => {
    alive = false;
  };
}
class Engine {
  constructor(onProgress = () => {
  }) {
    this.onProgress = onProgress;
    this.pending = /* @__PURE__ */ new Map();
    this.sequence = 0;
  }
  start() {
    if (this.worker) return;
    this.worker = new Worker(new URL(
      /* @vite-ignore */
      "" + new URL("model-worker-BFElXQrI.js", import.meta.url).href,
      import.meta.url
    ), {
      type: "module"
    });
    this.worker.onmessage = ({ data }) => {
      if (data.type === "progress") return this.onProgress(data.progress);
      const entry = this.pending.get(data.id);
      if (entry) {
        this.pending.delete(data.id);
        data.error ? entry.reject(Error(data.error)) : entry.resolve(data.result);
      }
    };
    this.worker.onerror = (event) => this.stop(
      Error(event.message || "The model worker stopped. Try loading again.")
    );
  }
  request(action, input) {
    this.start();
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, action, input });
    });
  }
  load() {
    return this.request("load");
  }
  embed(input) {
    return this.request("embed", input);
  }
  stop(reason = Error("Stopped. Your results are kept; run again when ready.")) {
    this.worker?.terminate();
    this.worker = null;
    for (const entry of this.pending.values()) entry.reject(reason);
    this.pending.clear();
  }
}
async function decodeAudio(blob) {
  const ctx = new AudioContext();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  } finally {
    await ctx.close();
  }
  const duration = Math.min(20, decoded.duration);
  const mono = new OfflineAudioContext(1, Math.ceil(duration * 16e3), 16e3);
  const source = mono.createBufferSource();
  source.buffer = decoded;
  source.connect(mono.destination);
  source.start();
  const rendered = await mono.startRendering();
  return {
    audio: rendered.getChannelData(0),
    info: {
      sampleRate: 16e3,
      duration,
      originalDuration: decoded.duration,
      channels: 1
    }
  };
}
async function decodeVideo(blob, start = 0, end = 20) {
  const url = URL.createObjectURL(blob), el = document.createElement("video");
  el.muted = true;
  el.preload = "auto";
  try {
    await new Promise((resolve, reject) => {
      el.onloadedmetadata = resolve;
      el.onerror = () => reject(Error("This video could not be decoded. Try an MP4 file."));
      el.src = url;
    });
    const stop = Math.min(el.duration, end), duration = stop - start;
    if (!(duration > 0)) throw Error("The video has no readable frames.");
    const canvas = document.createElement("canvas");
    const ratio = Math.min(1, 384 / Math.max(el.videoWidth, el.videoHeight));
    canvas.width = Math.max(1, Math.round(el.videoWidth * ratio));
    canvas.height = Math.max(1, Math.round(el.videoHeight * ratio));
    const ctx = canvas.getContext("2d", { willReadFrequently: true }), frames = [];
    for (let t = start + 0.05; t < stop; t += 1) {
      await new Promise((resolve, reject) => {
        el.onseeked = resolve;
        el.onerror = () => reject(Error("Could not read a video frame."));
        el.currentTime = t;
      });
      ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
      frames.push({
        data: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
        width: canvas.width,
        height: canvas.height,
        timestamp: t - start
      });
    }
    return {
      video: { frames, duration },
      info: {
        duration,
        originalDuration: el.duration,
        frames: frames.length,
        fps: 1
      }
    };
  } finally {
    el.removeAttribute("src");
    el.load();
    URL.revokeObjectURL(url);
  }
}
async function prepare(item, task = "search result", role = "query", extraText = "") {
  if (item.type === "text") {
    const text = role === "document" ? `title: ${item.group === "caption" ? "none" : item.title} | text: ${item.text}` : `task: ${task} | query: ${item.text}`;
    return { input: { text }, info: { text, characters: text.length } };
  }
  const blob = item.blob || await fetch(item.src).then((r) => {
    if (!r.ok) throw Error("Sample could not be loaded.");
    return r.blob();
  });
  if (item.type === "image")
    return {
      input: {
        image: blob,
        ...extraText ? { text: `${extraText} <|image|>` } : {}
      },
      info: { modality: "pixels", ...extraText ? { text: extraText } : {} }
    };
  const decoded = item.type === "audio" ? await decodeAudio(blob) : await decodeVideo(blob, item.start || 0, item.end || 20);
  return { input: { [item.type]: decoded[item.type] }, info: decoded.info };
}
const experiments = [
  {
    id: "photos",
    group: "Find",
    name: "Words → pictures",
    title: "Find a picture without naming the file",
    description: "Describe what you want to see. Which image would you expect to rank first?",
    mode: "search",
    type: "text",
    query: "a pet resting at home",
    filter: "image",
    ideas: [
      "a pet resting at home",
      "something that produces electricity",
      "a drink served hot",
      "a person ready for space travel"
    ],
    lesson: "Only pixels were embedded for the images. Their filenames and captions never entered the image encoder."
  },
  {
    id: "sounds",
    group: "Find",
    name: "Words → sounds",
    title: "Can words find a sound?",
    description: "Play a recording, then try describing the sound in a different way.",
    mode: "search",
    type: "text",
    query: "a dog barking",
    filter: "audio",
    ideas: [
      "a dog barking",
      "water washing onto a beach",
      "a ticking clock",
      "a machine cutting wood"
    ],
    lesson: "The audio input is a waveform. There is no intermediate speech transcript in this lab."
  },
  {
    id: "listen",
    group: "Find",
    name: "Sound → pictures & words",
    title: "Use a bark as the query",
    description: "Listen first. Can the recording retrieve a dog without any text query?",
    mode: "search",
    type: "audio",
    sample: "sound-dog",
    filter: "image",
    lesson: "Both queries and candidates live in the same 768-dimensional space. Start with image candidates, then try captions or everything. Raw cosine ranges can differ across modalities."
  },
  {
    id: "captions",
    group: "Find",
    name: "Picture → captions",
    title: "Give the image a caption menu",
    description: "Choose an image. The model ranks the supplied captions; it does not write a new caption.",
    mode: "search",
    type: "image",
    sample: "chelsea",
    filter: "caption",
    lesson: "This is the CLIP application you already know, now using EmbeddingGemma 2. Compare the rankings, not individual coordinates across different models."
  },
  {
    id: "neighbors",
    group: "Find",
    name: "Picture → pictures",
    title: "What makes two images similar?",
    description: "Will a dog photo find the same animal, the same colour, or a sketch?",
    mode: "search",
    type: "image",
    sample: "retriever-photo",
    filter: "image",
    lesson: "Similarity is learned. A neighbour can share subject, style or background; inspect what the model actually chose."
  },
  {
    id: "languages",
    group: "Find",
    name: "Search in another language",
    title: "Does the query have to be in English?",
    description: "Keep the gallery fixed. Ask for a cat in Hindi, then try your own language.",
    mode: "search",
    type: "text",
    query: "एक बिल्ली की तस्वीर",
    filter: "image",
    ideas: [
      "एक बिल्ली की तस्वीर",
      "કોફીનો કપ",
      "un perro",
      "une fusée dans le ciel"
    ],
    lesson: "The text changes language; the image vectors stay fixed. Compare the top results with the equivalent English query."
  },
  {
    id: "mixed",
    group: "Find",
    name: "Picture + words",
    title: "Two inputs, one embedding",
    description: "Combine an image with a short note, then compare it with an image-only search.",
    mode: "mixed",
    type: "image",
    sample: "red-mug",
    query: "a drink for breakfast",
    filter: "all",
    lesson: "The image and your note are encoded together in one forward pass. This is different from averaging two independently computed embeddings."
  },
  {
    id: "moments",
    group: "Find",
    name: "Find a video moment",
    title: "Which part of the video matches?",
    description: "Search short excerpts from real videos of a puppy, coffee, waves and geese, alongside the original teaching slideshow.",
    mode: "search",
    type: "text",
    query: "a rocket launching",
    filter: "moment",
    ideas: [
      "a puppy playing indoors",
      "water moving in waves",
      "a rocket launching",
      "a cat looking at the camera",
      "a cup of coffee"
    ],
    lesson: "Each window has its own video embedding, using one sampled frame per second without the soundtrack. The same clip can match several queries. Video frames and extracted stills are related examples, not independent evaluation data."
  },
  {
    id: "classify",
    group: "Use",
    name: "Choose your own labels",
    title: "Turn descriptions into a classifier",
    description: "Change the labels and run again. There is no new classifier training.",
    mode: "classify",
    type: "image",
    sample: "newfoundland",
    filter: "all",
    labels: "a dog\na cat\na rocket\na cup of coffee",
    lesson: "The highest cosine wins among your supplied labels. These scores are not calibrated probabilities, and the correct answer may be missing from the menu."
  },
  {
    id: "documents",
    group: "Use",
    name: "Retrieve a course passage",
    title: "Find the evidence before answering",
    description: "Ask a question about the lecture. Inspect which passage a RAG system would receive.",
    mode: "search",
    type: "text",
    query: "Why do we normalize an embedding?",
    filter: "document",
    task: "question answering",
    ideas: [
      "Why do we normalize an embedding?",
      "How do we compute the CLIP loss?",
      "Can the end-of-text token see the whole sentence?"
    ],
    lesson: "This is the retrieval step of RAG. The displayed passages are source text; no generative model is loaded to write an answer."
  },
  {
    id: "code",
    group: "Use",
    name: "Search code by its job",
    title: "Describe the function you need",
    description: "Search by meaning instead of matching a function name.",
    mode: "search",
    type: "text",
    query: "find the five closest items to a query",
    filter: "code",
    task: "code retrieval",
    ideas: [
      "find the five closest items to a query",
      "make a vector have length one",
      "compute the symmetric contrastive loss"
    ],
    lesson: "The query uses the code-retrieval task prefix. The code snippets are embedded as documents with filenames as titles."
  },
  {
    id: "delta",
    group: "Inspect",
    name: "What changed?",
    title: "Return to the hat experiment",
    description: "Subtract the earlier image vector from the later one. Which descriptions align with that direction?",
    mode: "delta",
    type: "image",
    sample: "portrait",
    after: "portrait-hat",
    filter: "caption",
    lesson: "Δ = normalize(after − before). This is an exploratory direction in embedding space, not proof of what caused the change. Swap the images: every dot-product sign reverses."
  },
  {
    id: "clusters",
    group: "Inspect",
    name: "Group the collection",
    title: "Do different modalities group together?",
    description: "Group the stored embeddings without using their titles or class labels.",
    mode: "clusters",
    type: "text",
    filter: "all",
    lesson: "K-means uses the full selected-dimensional vectors. The plot uses PCA to show two dimensions, so its distances are only an approximation. Group numbers have no predefined meaning."
  },
  {
    id: "explorer",
    group: "Inspect",
    name: "Explore every embedding",
    title: "Where do pictures, words and sounds meet?",
    description: "Choose any item, open its complete vector, and compare it with another modality. Use the map to ask questions, then check the actual cosine.",
    mode: "explorer",
    lesson: "PCA keeps two directions of variation. The neighbour list uses the full selected-dimensional vectors."
  },
  {
    id: "training",
    group: "Learn",
    name: "Train a small classifier",
    title: "Teach a new task using frozen embeddings",
    description: "Take one gradient step at a time. Watch a small linear layer learn to recognise sounds from a handful of labelled examples.",
    mode: "training",
    lesson: "The encoder stays fixed. Only the classifier weights learn."
  }
];
let unmountAdvanced = () => {
};
const $ = (s) => document.querySelector(s), esc = (s) => String(s ?? "").replace(
  /[&<>"']/g,
  (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[c]
);
const state = {
  experiment: experiments[0],
  dimension: 768,
  busy: false,
  ready: false,
  gallery: [],
  vectors: {},
  results: [],
  query: null,
  selected: null,
  upload: null,
  limit: 6
};
let engine = new Engine((progress) => {
  if (progress.status === "progress_total")
    status(
      `Downloading model · ${Math.round(progress.progress)}%`,
      progress.progress
    );
  else if (progress.status === "progress")
    status(
      `Loading ${progress.file || "model"} · ${Math.round(progress.progress || 0)}%`
    );
  else if (progress.status === "initiate")
    status(`Loading ${progress.file || "model files"}…`);
});
function status(text, percent) {
  $("#status").textContent = text;
  $("#progress").hidden = !state.busy;
  percent == null ? $("#progress").removeAttribute("value") : $("#progress").value = percent;
}
function error(e) {
  $("#error").textContent = e.message || String(e);
  $("#error").hidden = false;
}
function busy(value) {
  state.busy = value;
  document.querySelectorAll(
    "#query-area input,#query-area select,#query-area textarea,#query-area button,#dimension"
  ).forEach((n) => {
    if (value) {
      n.dataset.preDisabled = String(n.disabled);
      n.disabled = true;
    } else {
      n.disabled = n.dataset.preDisabled === "true";
    }
  });
  $("#run").disabled = value;
  $("#workspace").setAttribute("aria-busy", String(value));
  $("#stop").hidden = !value;
  $("#experiment").disabled = value;
  document.querySelectorAll("[data-experiment]").forEach((b) => b.disabled = value);
  $("#progress").hidden = !value;
}
function clearResults() {
  state.query = null;
  state.results = [];
  state.selected = null;
  $("#inspect").hidden = true;
  $("#context").hidden = true;
  $("#map").hidden = true;
  $("#result-source").textContent = "";
  $("#results-title").textContent = "Make a prediction";
  $("#result-meta").textContent = "A few candidates from the collection. Which will match?";
  const filter = $("#filter")?.value || "all";
  let examples = filterItems(filter);
  if (filter === "image") {
    const featured = [
      "chelsea",
      "coffee",
      "rocket",
      "retriever-photo",
      "solar",
      "astronaut"
    ];
    examples = [
      ...featured.map(itemById).filter(Boolean),
      ...examples.filter((i) => !featured.includes(i.id))
    ];
  }
  $("#results").innerHTML = examples.slice(0, 6).map((item, i) => card(item, i, null)).join("");
  $("#result-step").textContent = "02 / PREDICT THE MATCH";
  $("#more").hidden = true;
}
function itemById(id) {
  return state.gallery.find((x) => x.id === id);
}
function filterItems(filter) {
  return state.gallery.filter(
    (i) => filter === "all" || i.type === filter || i.group === filter
  );
}
function thumb(item, controls = false) {
  if (item.type === "image")
    return `<img src="${esc(item.src)}" alt="${esc(item.title)}" loading="lazy">`;
  if (item.type === "audio")
    return `<div class="audio-preview"><svg viewBox="0 0 100 40" aria-hidden="true"><path d="M8 16v8m8-12v16m8-17v18m8-24v30m8-20v10m8-14v18m8-25v32m8-23v14m8-18v22m8-16v10m8-15v20"/></svg>${controls ? `<audio controls preload="metadata" src="${esc(item.src)}"></audio>` : "<span>5-second recording</span>"}</div>`;
  if (item.type === "video")
    return `<video ${controls ? "controls" : "muted"} playsinline preload="metadata" src="${esc(item.src)}#t=${item.start || 0.05},${item.end || 20}" aria-label="${esc(item.title)}"></video>`;
  return `<div class="text-preview ${item.group === "code" ? "code-preview" : ""}">${esc(item.text)}</div>`;
}
function renderQuery() {
  const e = state.experiment;
  $("#query-area").innerHTML = `<div class="input-row"><label>Query input<select id="query-type">${["text", "image", "audio", "video"].map((t) => `<option ${t === e.type ? "selected" : ""}>${t}</option>`).join("")}</select></label><label>Search within<select id="filter">${[
    ["all", "Everything"],
    ["image", "Images"],
    ["audio", "Sounds"],
    ["caption", "Captions"],
    ["moment", "Video moments"],
    ["video", "Videos"],
    ["document", "Course passages"],
    ["code", "Code"]
  ].map(
    ([v, t]) => `<option value="${v}" ${e.filter === v ? "selected" : ""}>${t}</option>`
  ).join(
    ""
  )}</select></label></div><div id="query-fields"></div><div id="special-fields"></div>`;
  $("#query-type").onchange = () => {
    renderFields();
    clearResults();
  };
  $("#filter").onchange = () => {
    if (state.query) showRanking();
    else clearResults();
  };
  renderFields();
  if (e.mode === "classify")
    $("#special-fields").innerHTML = `<label class="field">Your labels <span>one per line · up to 12</span><textarea id="labels" rows="4">${esc(e.labels)}</textarea></label>`;
  if (e.mode === "delta") {
    $("#query-type").value = "image";
    $("#query-type").disabled = true;
    renderFields();
    $("#special-fields").innerHTML = `<label class="field">After image<select id="after">${state.gallery.filter((i) => i.type === "image").map(
      (i) => `<option value="${esc(i.id)}" ${i.id === e.after ? "selected" : ""}>${esc(i.title)}</option>`
    ).join(
      ""
    )}</select></label><div id="after-preview" class="query-preview"></div><button id="swap" class="quiet" type="button">Swap before and after</button>`;
    const afterPreview = () => $("#after-preview").innerHTML = thumb(itemById($("#after").value));
    $("#after").onchange = () => {
      afterPreview();
      clearResults();
    };
    afterPreview();
    $("#swap").onclick = () => {
      const a = $("#sample").value;
      $("#sample").value = $("#after").value;
      $("#after").value = a;
      updatePreview();
      afterPreview();
      clearResults();
    };
  }
  if (e.mode === "clusters") {
    $("#query-area").innerHTML = '<p class="hint">The collection is already embedded. Grouping needs no model download.</p><label class="field">Number of groups<select id="groups"><option>3</option><option selected>4</option><option>5</option><option>6</option></select></label>';
  }
  $("#query-area").oninput = (event) => {
    if (!["filter", "groups"].includes(event.target.id)) clearResults();
  };
  if (e.mode === "mixed") $("#query-type").disabled = true;
  $("#run").textContent = e.mode === "clusters" ? "Group the collection" : e.mode === "delta" ? "Compare the change" : e.mode === "classify" ? "Compare with labels" : "Run this search";
}
function renderFields() {
  const type = $("#query-type").value, e = state.experiment;
  $("#query-fields").innerHTML = type === "text" ? `<label class="field">What are you looking for?<textarea id="query" rows="2" maxlength="2000" placeholder="Describe an image, sound, idea or moment…">${esc(e.query || "")}</textarea></label><div class="ideas">${(e.ideas || []).map((t) => `<button type="button" class="chip" data-idea="${esc(t)}">${esc(t)}</button>`).join("")}</div>` : `<label class="field">${e.mode === "delta" ? "Before image" : "Choose a sample"}<select id="sample">${state.gallery.filter((i) => i.type === type).map(
    (i) => `<option value="${esc(i.id)}" ${e.sample === i.id ? "selected" : ""}>${esc(i.title)}</option>`
  ).join(
    ""
  )}${state.upload?.type === type ? `<option value="${esc(state.upload.id)}" selected>Your upload</option>` : ""}</select></label><div id="query-preview" class="query-preview"></div><div class="sample-strip" id="sample-strip" aria-label="Try another sample"></div><details class="input-options"><summary>Upload or re-encode an input</summary><label class="upload">Or use your own ${type}<input id="upload" type="file" accept="${type}/*"></label><p class="hint">${type === "audio" ? "Audio is mixed to mono, resampled to 16 kHz and limited to the first 20 seconds." : type === "video" ? "We read the first 20 seconds at one frame per second. Video audio is not included." : "Your pixels are processed in this browser."} Upload limit: 30 MB.</p><label class="check"><input type="checkbox" id="recompute"> Re-encode this sample on my device</label></details>${e.mode === "mixed" ? `<label class="field">Add a note<textarea id="query" rows="2" maxlength="1000">${esc(e.query)}</textarea></label>` : ""}`;
  if (type !== "text") {
    $("#sample").onchange = () => {
      updatePreview();
      clearResults();
    };
    $("#upload").onchange = upload;
    updatePreview();
  }
}
function updatePreview() {
  const item = selectedInput();
  if (item) {
    $("#query-preview").innerHTML = thumb(item, true);
    const strip = $("#sample-strip");
    if (strip)
      strip.innerHTML = state.gallery.filter((i) => i.type === item.type).slice(0, 6).map(
        (i) => `<button type="button" class="sample-pick sample-${i.type}" data-sample="${esc(i.id)}" aria-label="Choose ${esc(i.title)}" aria-pressed="${i.id === item.id}">${thumb(i)}<span>${esc(i.title)}</span></button>`
      ).join("");
  }
}
function selectedInput() {
  const type = $("#query-type")?.value;
  if (type === "text")
    return {
      id: "query",
      type: "text",
      title: "Your query",
      text: $("#query")?.value.trim() || ""
    };
  const id = $("#sample")?.value;
  return id === state.upload?.id ? state.upload : itemById(id);
}
async function upload(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 30 * 1024 * 1024) {
    error(Error("Please use a file smaller than 30 MB."));
    return;
  }
  if (state.upload?.src) URL.revokeObjectURL(state.upload.src);
  state.upload = {
    id: "upload-" + Date.now(),
    type: $("#query-type").value,
    title: file.name,
    blob: file,
    src: URL.createObjectURL(file),
    credit: "Your local file"
  };
  renderFields();
  clearResults();
}
function activate(id) {
  if (state.busy) return;
  state.experiment = experiments.find((e2) => e2.id === id) || experiments[0];
  const e = state.experiment;
  $("#experiment").value = e.id;
  document.querySelectorAll("[data-experiment]").forEach(
    (b) => b.setAttribute("aria-current", String(b.dataset.experiment === e.id))
  );
  $("#experiment-title").textContent = e.title;
  $("#experiment-number").textContent = `EXPERIMENT ${String(experiments.indexOf(e) + 1).padStart(2, "0")} / ${experiments.length}`;
  $("#experiment-description").textContent = e.description;
  $("#lesson").textContent = e.lesson;
  unmountAdvanced();
  const advanced = ["explorer", "training"].includes(e.mode);
  $("#advanced-area").hidden = !advanced;
  $(".workbench").hidden = advanced;
  $("#inspect").hidden = true;
  if (advanced) {
    state.query = null;
    state.selected = null;
    unmountAdvanced = e.mode === "explorer" ? mountExplorer($("#advanced-area"), state) : mountTraining($("#advanced-area"), state);
  } else {
    $("#advanced-area").innerHTML = "";
    renderQuery();
    clearResults();
    unmountAdvanced = () => {
    };
  }
  history.replaceState(null, "", "#" + e.id);
}
async function ensureModel() {
  if (state.ready) return;
  if (!navigator.gpu || !await navigator.gpu.requestAdapter())
    throw Error(
      "WebGPU is unavailable here. Try current Chrome or Edge with hardware acceleration. You can still explore stored sample vectors and groups."
    );
  status("Loading EmbeddingGemma 2 · about 473 MB once, then browser-cached…");
  await engine.load();
  state.ready = true;
  $("#model-state").textContent = "WebGPU model ready";
}
async function embedItem(item, { role = "query", task = "search result", force = false, mixed = "" } = {}) {
  if (state.vectors[item.id] && !force && !mixed)
    return {
      ...state.vectors[item.id],
      origin: "Saved WebGPU run · same pinned model"
    };
  await ensureModel();
  status(`Encoding ${item.title}…`);
  const { input, info } = await prepare(item, task, role, mixed);
  const result = {
    ...await engine.embed(input),
    info,
    origin: "Computed in this browser · WebGPU"
  };
  if (!mixed && item.type !== "text") state.vectors[item.id] = result;
  return result;
}
async function run() {
  if (state.busy) return;
  $("#error").hidden = true;
  busy(true);
  const e = state.experiment;
  try {
    if (e.mode === "clusters") {
      showGroups();
      return;
    }
    const item = selectedInput();
    if (!item) throw Error("Choose an input first.");
    if (item.type === "text" && !item.text) throw Error("Enter a query first.");
    if (e.mode === "classify" && !$("#labels").value.trim())
      throw Error("Add at least one label.");
    const forced = $("#recompute")?.checked, query = await embedItem(item, {
      task: e.mode === "classify" ? "classification" : e.task || "search result",
      force: forced,
      mixed: e.mode === "mixed" ? $("#query").value.trim() : ""
    });
    let queryVector = query.vector, exclude = [item.id], origin = query.origin;
    if (e.mode === "delta") {
      const after = itemById($("#after").value);
      const other = await embedItem(after, { force: forced });
      queryVector = difference(other.vector, query.vector);
      origin = other.origin;
      exclude.push(after.id);
      query.info = {
        formula: "unit(after − before)",
        before: item.title,
        after: after.title
      };
    }
    let candidates = null;
    if (e.mode === "classify") {
      const labels = [
        ...new Set(
          $("#labels").value.split("\n").map((x) => x.trim()).filter(Boolean)
        )
      ];
      if (labels.length > 12)
        throw Error("Use at most 12 labels for this small lab.");
      candidates = [];
      for (let i = 0; i < labels.length; i++) {
        const label = {
          id: "label-" + i,
          type: "text",
          title: labels[i],
          text: labels[i],
          group: "label",
          credit: "Your candidate label"
        };
        const result = await embedItem(label, {
          task: "classification",
          force: true
        });
        state.vectors[label.id] = result;
        candidates.push(label);
      }
    }
    state.query = {
      ...query,
      vector: queryVector,
      item,
      exclude,
      candidates,
      origin
    };
    state.limit = 6;
    showRanking();
    status(
      state.ready ? "Ready · queries run on this device" : "Ready · comparing stored sample vectors"
    );
  } catch (e2) {
    error(e2);
    state.query = null;
    status("Not completed. You can change the input and try again.");
  } finally {
    busy(false);
  }
}
function showRanking() {
  if (!state.query) return;
  const q = state.query, filter = $("#filter")?.value || "all";
  state.results = rank(
    q.vector,
    q.candidates || filterItems(filter),
    state.vectors,
    state.dimension,
    q.exclude
  );
  $("#map").hidden = true;
  $("#result-meta").textContent = `${state.results.length} candidates · ${state.dimension} dimensions · raw cosine similarity${filter === "all" ? " · cross-modality score ranges may differ" : ""}`;
  renderResults();
  $("#inspect").hidden = true;
  state.selected = null;
  $("#results-title").textContent = "The closest matches";
  $("#result-step").textContent = "02 / COMPARE THE RESULTS";
  $("#context").hidden = !["document", "code"].includes(filter);
  if (!$("#context").hidden) {
    $("#context-text").textContent = state.results.slice(0, 3).map((r, i) => `[${i + 1}] ${r.item.title}
${r.item.text}`).join("\n\n");
  }
}
function renderResults() {
  const q = state.query;
  $("#results").innerHTML = state.results.slice(0, state.limit).map((r, i) => card(r.item, i, r.score)).join("") || '<p class="empty">No candidates in this selection. Choose another collection.</p>';
  $("#more").hidden = state.limit >= state.results.length;
  $("#result-source").textContent = q ? `${q.origin}${q.elapsed ? " · last encoding " + Math.round(q.elapsed) + " ms" : ""}. Candidates use the same pinned model. Re-encode recomputes sample inputs locally.` : "";
}
function card(item, index, score) {
  return `<article class="result-card ${score == null ? "preview-card" : ""}"><div class="card-media">${thumb(item, true)}</div><div class="card-body"><div class="card-top"><span class="small-label">${esc(item.group || item.type)}</span>${score == null ? "" : `<span class="score" title="Cosine similarity, not a probability">${score.toFixed(4)}</span>`}</div><h3>${score == null ? "" : `<span class="rank">${index + 1}</span> `}${esc(item.title)}</h3>${score == null ? "" : `<div class="score-track" aria-label="Cosine ${score.toFixed(4)}"><span style="width:${Math.max(0, (score + 1) / 2 * 100)}%"></span></div>`}<div class="card-actions"><button data-inspect="${esc(item.id)}" class="quiet">${score == null ? "See vector" : "Explain score ↗"}</button><button data-query="${esc(item.id)}" class="quiet">Use as query</button></div></div></article>`;
}
function findItem(id) {
  return state.query?.candidates?.find((i) => i.id === id) || itemById(id);
}
function inspect(id) {
  const item = findItem(id), data = state.vectors[id];
  if (!item || !data) return;
  state.selected = id;
  $("#inspect").hidden = false;
  $("#inspect-title").textContent = item.title;
  $("#inspect-pair").innerHTML = `${state.query ? `<div><span class="small-label">Your ${state.experiment.mode === "delta" ? "change direction" : "query"}</span><strong>${esc(state.experiment.mode === "delta" ? state.query.info.before + " → " + state.query.info.after : state.query.item.text || state.query.item.title)}</strong></div><span class="pair-arrow">→</span>` : ""}<div><span class="small-label">The candidate · ${esc(item.type)}</span><strong>${esc(item.title)}</strong></div>`;
  $("#candidate-text").textContent = item.type === "text" ? item.text : item.credit;
  const query = state.query ? unit(state.query.vector, state.dimension) : null, v = unit(data.vector, state.dimension);
  const colourMax = Math.max(
    ...v.map(Math.abs),
    ...(query || []).map(Math.abs)
  );
  $("#candidate-canvas").hidden = false;
  heatmap($("#candidate-canvas"), v, colourMax);
  $("#query-canvas").hidden = !query;
  if (query) heatmap($("#query-canvas"), query, colourMax);
  $("#query-vector-label").textContent = query ? "Query vector q" : "Run a query to compare two vectors";
  $("#candidate-vector-label").textContent = `Candidate vector v · ${state.dimension} numbers`;
  $("#shape-details").textContent = `${state.query ? "QUERY INPUT\n" + JSON.stringify(state.query.info, null, 2) + "\nQuery tensor shapes: " + JSON.stringify(state.query.shapes) + "\n\n" : ""}CANDIDATE INPUT
${JSON.stringify(data.info, null, 2)}

Model input tensor shapes:
${JSON.stringify(data.shapes, null, 2)}

Model output: [1, 768]
Selected: [1, ${state.dimension}] → L2-normalize
Norm = ${Math.hypot(...v).toFixed(6)}`;
  $("#dot-area").hidden = !query;
  $("#coordinates").max = state.dimension - 8;
  $("#coordinates").value = 0;
  if (query) renderDot();
  $("#vector-note").textContent = "The two maps use the same colour scale: dark green = positive, brown = negative. Coordinates are learned numbers, not named concepts. Each row below shows an exact multiplication.";
  $("#download-vector").onclick = () => download(
    {
      model: MODEL,
      revision: REVISION,
      dimension: state.dimension,
      item: item.title,
      vector: v,
      ...query ? { queryVector: query, cosine: dot(query, v) } : {}
    },
    "embedding.json"
  );
}
function heatmap(canvas, v, colourMax) {
  const ctx = canvas.getContext("2d"), cols = 64, rows = Math.ceil(v.length / cols), dpr = devicePixelRatio || 1, w = canvas.clientWidth || 500, h = rows * 9;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + "px";
  ctx.scale(dpr, dpr);
  const max = colourMax || Math.max(...v.map(Math.abs));
  v.forEach((x, i) => {
    ctx.fillStyle = x >= 0 ? `rgba(57,103,73,${0.08 + 0.92 * Math.abs(x) / max})` : `rgba(156,90,48,${0.08 + 0.92 * Math.abs(x) / max})`;
    ctx.fillRect(
      i % cols * w / cols,
      Math.floor(i / cols) * 9,
      w / cols - 1,
      8
    );
  });
}
function renderDot() {
  const q = unit(state.query.vector, state.dimension), v = unit(state.vectors[state.selected].vector, state.dimension), start = +$("#coordinates").value;
  $("#coordinate-range").textContent = `Coordinates ${start + 1}–${start + 8} of ${state.dimension}`;
  $("#coordinate-table").innerHTML = Array.from({ length: 8 }, (_, i) => {
    const j = start + i;
    return `<tr><td>${j + 1}</td><td>${q[j].toFixed(6)}</td><td>×</td><td>${v[j].toFixed(6)}</td><td>${(q[j] * v[j]).toFixed(6)}</td></tr>`;
  }).join("");
  const partial = q.slice(start, start + 8).reduce((s, x, i) => s + x * v[start + i], 0);
  $("#dot-total").innerHTML = `<span>These 8 products sum to <b>${partial.toFixed(6)}</b>.</span><strong>All ${state.dimension} products sum to ${dot(q, v).toFixed(6)}</strong><span>q · v = Σ qₖvₖ = cosine(q, v), because both vectors have length 1.</span>`;
}
function showGroups() {
  const items = state.gallery.filter(
    (i) => !["moment", "document", "code"].includes(i.group)
  ), rows = items.map((i) => unit(state.vectors[i.id].vector, state.dimension)), labels = kmeans(rows, +$("#groups").value), points = pca(rows);
  $("#inspect").hidden = true;
  $("#context").hidden = true;
  $("#map").hidden = false;
  $("#results-title").textContent = "Neighbours in the shared space";
  $("#result-step").textContent = "02 / EXPLORE THE GROUPS";
  $("#result-meta").textContent = `${items.length} items · ${$("#groups").value} groups · ${state.dimension} dimensions`;
  $("#result-source").textContent = "Computed now from the stored WebGPU embeddings. No titles or class labels are used for clustering.";
  const width = 700, height = 340, minX = Math.min(...points.map((p) => p[0])), maxX = Math.max(...points.map((p) => p[0])), minY = Math.min(...points.map((p) => p[1])), maxY = Math.max(...points.map((p) => p[1]));
  $("#map").innerHTML = `<p class="hint">PCA projection · colour = group · click a point to inspect it</p><div class="map-legend">${Array.from({ length: +$("#groups").value }, (_, g) => `<span><i style="background:${["#376448", "#a56538", "#627999", "#874d60", "#888338", "#635e85"][g]}"></i>Group ${g + 1}</span>`).join("")}</div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Two dimensional projection of the collection">${points.map((p, i) => `<g class="map-point" tabindex="0" role="button" aria-label="${esc(items[i].title)}, group ${labels[i] + 1}" data-inspect="${items[i].id}"><circle cx="${30 + (p[0] - minX) / (maxX - minX || 1) * (width - 60)}" cy="${30 + (p[1] - minY) / (maxY - minY || 1) * (height - 60)}" r="7" fill="${["#376448", "#a56538", "#627999", "#874d60", "#888338", "#635e85"][labels[i]]}"/><title>${esc(items[i].title)} · group ${labels[i] + 1}</title></g>`).join("")}</svg>`;
  $("#results").innerHTML = Array.from(
    { length: +$("#groups").value },
    (_, g) => `<div class="cluster" style="border-color:${["#376448", "#a56538", "#627999", "#874d60", "#888338", "#635e85"][g]}"><h3>Group ${g + 1}</h3>${items.filter((_2, i) => labels[i] === g).map(
      (item) => `<button class="cluster-item" data-inspect="${item.id}"><span class="small-label">${item.type}</span>${esc(item.title)}</button>`
    ).join("")}</div>`
  ).join("");
  $("#more").hidden = true;
  status("Groups calculated. Inspect the neighbours and the exceptions.");
}
function download(value, name) {
  const u = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" })
  ), a = document.createElement("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1e3);
}
function renderLibrary() {
  const filter = $("#library-filter").value;
  $("#library-grid").innerHTML = filterItems(filter).map(
    (item) => `<article class="library-item"><div class="card-media">${thumb(item, true)}</div><h3>${esc(item.title)}</h3><span class="small-label">${esc(item.group || item.type)}</span><div class="card-actions"><button class="quiet" data-query="${item.id}">Use as query</button><button class="quiet" data-inspect="${item.id}">Inspect</button></div><details><summary>Source & credit</summary><p>${esc(item.credit)}</p>${item.source ? `<a href="${esc(item.source)}" target="_blank" rel="noreferrer">Source</a>` : ""}</details></article>`
  ).join("");
}
function useAsQuery(id) {
  if (state.busy) return;
  const item = findItem(id);
  if (!item) return;
  $("#library").close();
  activate("photos");
  $("#query-type").value = item.type;
  renderFields();
  if (item.type === "text") $("#query").value = item.text;
  else {
    $("#sample").value = id;
    updatePreview();
  }
  $("#filter").value = "all";
  $("#experiment-title").textContent = "Search with " + item.title;
  $("#experiment-description").textContent = "Compare this input with the collection. Choose a candidate type, predict a match, then run.";
  $("#lesson").textContent = "The same comparison works for different inputs: encode, normalize and take a dot product. Similarity scores can have different ranges across modalities.";
  $("#workspace").scrollIntoView({ behavior: "smooth" });
  clearResults();
}
async function init() {
  $("#app").innerHTML = `<header class="site-header"><a class="brand" href="https://nipunbatra.github.io/"><span class="brand-mark">nb.</span> Nipun Batra <span class="brand-slash">/</span> <span class="brand-course">Learning labs</span></a><nav aria-label="Course links"><a href="https://nipunbatra.github.io/attention/clip/">CLIP lecture ↗</a><button class="quiet" id="open-about">How it works</button></nav></header>
<main><section class="intro"><div class="intro-copy"><p class="eyebrow">BEYOND CLIP · EMBEDDINGGEMMA 2</p><h1>One space.<br><em>Many ways to search.</em></h1><p>A photo, a sentence, even a bark. Find out what happens when different kinds of input share the same representation.</p><a class="intro-link" href="#workspace">Start exploring <span>↓</span></a><div class="intro-facts"><span><b>${experiments.length}</b> experiments</span><span><b id="sample-count">…</b> samples</span><span><b>768</b> dimensions</span></div></div>
<div class="intro-media"><div class="media-caption"><span>A few ways in</span><span>Pick one to begin ↙</span></div><div class="media-tiles"><button class="hero-tile image-tile" data-experiment="captions"><img src="./media/chelsea.jpg" alt="An orange cat looking at the camera"><span><b>Start with a picture</b>Find its words <i>↗</i></span></button><button class="hero-tile sound-tile" data-experiment="listen"><span class="sound-drawing" aria-hidden="true"><svg viewBox="0 0 150 90"><path d="M9 39v12m11-21v30m11-39v48m11-32v16m11-52v88m11-69v50m11-38v26m11-44v62m11-46v30m11-40v50m11-32v14m11-25v36m11-27v18"/></svg><small>Dog bark · 5 seconds</small></span><span><b>Start with a sound</b>Find its picture <i>↗</i></span></button><button class="hero-tile text-tile" data-experiment="moments"><span class="text-drawing">“a rocket<br>launching”<small>Text → video</small></span><span><b>Start with a thought</b>Find a moment <i>↗</i></span></button></div><p class="media-footnote">Different inputs. The same encode → compare idea.</p></div></section><div class="runtime-bar"><span class="status-dot"></span><strong id="model-state">Collection ready · model loads on demand</strong><span>WebGPU · q4 · ~473 MB first download</span><button class="quiet" id="open-library">Explore the collection</button></div><div class="lab-layout"><nav class="experiment-nav" aria-label="Experiments"><div class="nav-intro"><span class="eyebrow">THE EXPERIMENTS</span><span>Choose a question to investigate</span></div>${[
    "Find",
    "Use",
    "Inspect",
    "Learn"
  ].map(
    (g) => `<div class="experiment-group"><p class="small-label">${g}</p><div>${experiments.filter((e) => e.group === g).map(
      (e) => `<button data-experiment="${e.id}" aria-current="false">${e.name}</button>`
    ).join("")}</div></div>`
  ).join(
    ""
  )}</nav><section id="workspace"><label class="mobile-experiment">Experiment<select id="experiment">${experiments.map((e) => `<option value="${e.id}">${e.name}</option>`).join("")}</select></label><div class="experiment-head"><span class="eyebrow" id="experiment-number"></span><h2 id="experiment-title"></h2><p id="experiment-description"></p></div><div id="advanced-area" hidden></div><div class="workbench"><div class="query-column"><div class="query-panel"><p class="step-label">01 / CHOOSE YOUR INPUT</p><div id="query-area"></div><div class="run-row"><button id="run" class="primary">Run this search</button><button id="stop" class="quiet" hidden>Stop & unload</button><span id="status" role="status" aria-live="polite">No API key. Your inputs stay in this browser.</span></div><progress id="progress" max="100" hidden></progress><p id="error" role="alert" hidden></p></div><p class="lesson"><b>What to notice</b><span id="lesson"></span></p></div><div class="results-column"><p class="step-label" id="result-step">02 / PREDICT THE MATCH</p><div class="results-heading"><div><h2 id="results-title">The matches</h2><p id="result-meta"></p></div><label>Vector size<select id="dimension"><option value="768">768 · full</option><option value="512">512</option><option value="256">256</option><option value="128">128</option></select></label></div><p id="dimension-note" class="hint">Keep the first d coordinates, then normalize again. Both sides use the same d.</p><div id="map" hidden></div><div id="results" class="results-grid"></div><button id="more" class="quiet" hidden>Show more results</button><p id="result-source" class="fineprint"></p><details id="context" hidden><summary>The context a generative model could receive</summary><pre id="context-text"></pre><p>This lab stops at retrieval. Check whether these sources actually answer the question.</p></details></div></div><section id="inspect" class="inspector" hidden><div class="inspector-head"><div><p class="eyebrow">03 / FOLLOW THE NUMBERS</p><h2 id="inspect-title"></h2></div><div class="inspector-actions"><button class="quiet" id="download-vector">Download vectors</button><button class="quiet" id="close-inspect" aria-label="Close vector inspector">Close ×</button></div></div><div id="inspect-pair" class="inspect-pair"></div><p id="candidate-text"></p><div class="vector-pair"><div><h3 id="query-vector-label"></h3><canvas id="query-canvas" aria-label="Query embedding coordinates"></canvas></div><div><h3 id="candidate-vector-label"></h3><canvas id="candidate-canvas" aria-label="Candidate embedding coordinates"></canvas></div></div><p class="hint" id="vector-note"></p><div id="dot-area"><label id="coordinate-range" for="coordinates"></label><input type="range" id="coordinates" min="0" max="760" step="8" value="0"><div class="table-scroll"><table><thead><tr><th>k</th><th>Query qₖ</th><th></th><th>Candidate vₖ</th><th>Product</th></tr></thead><tbody id="coordinate-table"></tbody></table></div><div id="dot-total" class="dot-total"></div></div><details><summary>Show model inputs, shapes and normalization</summary><pre id="shape-details"></pre></details></section></section></div><footer><p class="eyebrow">TAKE IT BACK TO THE LECTURE</p><strong>The same idea as CLIP, with more kinds of input.</strong><p>Encode → normalize → compare. The model returns embeddings; the application decides how to use them.</p><div><a href="https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/">Google announcement</a><a href="https://www.youtube.com/watch?v=anPsS6huQk0">Introduction video</a><a href="https://huggingface.co/onnx-community/embeddinggemma-2-ONNX">Model & implementation</a><button class="quiet" id="footer-library">Sample credits</button><a href="https://github.com/nipunbatra/dl-teaching/tree/master/attention-followups/embeddinggemma-lab">Source code</a></div></footer></main><dialog id="library"><div class="dialog-head"><div><p class="eyebrow">THE SHARED COLLECTION</p><h2 id="library-title">Samples you can inspect</h2></div><button class="quiet" data-close="library" aria-label="Close collection">Close ×</button></div><p>Images, recordings and video are embedded from their media, without their titles. Some images are generated teaching examples, identified in the credits.</p><label>Show<select id="library-filter"><option value="all">All samples</option><option value="image">Images</option><option value="audio">Sounds</option><option value="text">Text & code</option><option value="video">Video</option></select></label><div id="library-grid" class="library-grid"></div></dialog><dialog id="about"><div class="dialog-head"><h2>From CLIP to a multimodal index</h2><button class="quiet" data-close="about">Close ×</button></div><ol class="explanation"><li><strong>Represent each item.</strong> Media encoders feed a shared model. Mean pooling and a projection produce a 768-number embedding. This architecture is not CLIP’s two independent towers.</li><li><strong>Store the collection.</strong> The supplied vectors were computed from these exact samples using this pinned ONNX model, q4 precision and WebGPU. They let us inspect the collection without re-encoding it on every search.</li><li><strong>Encode a new query on your device.</strong> Model files download from Hugging Face. Typed text and uploaded media go to a local worker; there is no inference server or API key.</li><li><strong>Compare.</strong> Unit vectors give a cosine through a dot product. Scores are similarities, not probabilities. There is no learned threshold that guarantees a match.</li><li><strong>Shorten, carefully.</strong> The size control keeps leading coordinates and re-normalizes. It reduces index storage, not the size of the model download. The 128-dimensional option can hurt multimodal retrieval.</li></ol><p class="hint">Text search uses a task prefix; text candidates use a document prefix. Classification uses the classification prefix. Inspect the input details to see the exact text supplied.</p><p>Local uploads are kept only in this tab. Audio is limited to 20 seconds. Video uses up to 20 seconds of sampled frames without its soundtrack. Close or stop the model to release its worker.</p><p class="fineprint">${MODEL}<br>Revision ${REVISION}<br>Transformers.js 4.3.1 · q4 · Apache 2.0 model</p></dialog>`;
  try {
    const [gallery, embeddings] = await Promise.all([
      fetch("./gallery.json").then((r) => {
        if (!r.ok) throw Error("Could not load sample collection.");
        return r.json();
      }),
      fetch("./embeddings.json").then((r) => {
        if (!r.ok) throw Error("Could not load the stored embeddings.");
        return r.json();
      })
    ]);
    state.gallery = gallery;
    $("#sample-count").textContent = gallery.length;
    state.vectors = embeddings.items;
    $("#library-title").textContent = `${gallery.length} samples, one shared space`;
    if (embeddings.revision !== REVISION)
      throw Error("The sample embeddings and model version do not match.");
    activate(location.hash.slice(1));
  } catch (e) {
    error(e);
    return;
  }
  $("#run").onclick = run;
  $("#stop").onclick = () => {
    engine.stop();
    state.ready = false;
    $("#model-state").textContent = "Model unloaded";
  };
  $("#experiment").onchange = (e) => activate(e.target.value);
  $("#coordinates").oninput = renderDot;
  $("#dimension").onchange = (e) => {
    state.dimension = +e.target.value;
    $("#dimension-note").textContent = `${state.dimension * 4} bytes per float32 vector · ${Math.round(768 / state.dimension * 10) / 10}× smaller than 768d. ${state.dimension === 128 ? "128d can lose multimodal quality. " : ""}Keep the first d values and re-normalize both vectors.`;
    if (state.experiment.mode === "clusters" && $("#map").hidden === false)
      showGroups();
    else if (state.query) showRanking();
    else if (state.selected) inspect(state.selected);
  };
  $("#close-inspect").onclick = () => {
    $("#inspect").hidden = true;
    state.selected = null;
  };
  $("#more").onclick = () => {
    state.limit += 12;
    renderResults();
  };
  const openLibrary = () => {
    renderLibrary();
    $("#library").showModal();
  };
  $("#open-library").onclick = openLibrary;
  $("#footer-library").onclick = openLibrary;
  $("#library-filter").onchange = renderLibrary;
  $("#open-about").onclick = () => $("#about").showModal();
  document.addEventListener("click", (event) => {
    const t = event.target.closest("button,[data-inspect]");
    if (!t) return;
    if (t.dataset.experiment) {
      activate(t.dataset.experiment);
      $("#workspace").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (t.dataset.sample && !state.busy) {
      $("#sample").value = t.dataset.sample;
      updatePreview();
      clearResults();
    }
    if (t.dataset.idea) {
      $("#query").value = t.dataset.idea;
      clearResults();
    }
    if (t.dataset.inspect && !state.busy) {
      $("#library").close();
      inspect(t.dataset.inspect);
      $("#inspect").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (t.dataset.query) {
      if (["explorer", "training"].includes(state.experiment.mode))
        activate("photos");
      useAsQuery(t.dataset.query);
    }
    if (t.dataset.close) $("#" + t.dataset.close).close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches(".map-point"))
      event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  window.embeddingLab = {
    state,
    run,
    activate,
    inspect,
    engine,
    prepare,
    showRanking
  };
  if (!navigator.gpu)
    $("#model-state").textContent = "WebGPU unavailable · stored sample experiments still work";
}
init();

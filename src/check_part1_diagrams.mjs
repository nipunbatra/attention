// Live-model and SVG-layout regressions for Part I's incremental diagrams.
// Uses an existing Playwright installation; no dependencies are installed.
// node src/check_part1_diagrams.mjs [part1.html]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_PATH, process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
const cache = path.join(os.homedir(), '.npm', '_npx');
if (fs.existsSync(cache)) {
  for (const dir of fs.readdirSync(cache).sort()) candidates.push(path.join(cache, dir, 'node_modules/playwright'));
}
let pw;
for (const candidate of candidates) { try { pw = require(candidate); break; } catch {} }
if (!pw) throw new Error('Set PLAYWRIGHT_PATH to an existing Playwright installation.');

const browser = await pw.chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto(pathToFileURL(path.resolve(process.argv[2] || 'part1.html')).href);
  await page.waitForFunction(() => !!window.AT?.part1Diagrams);
  await page.evaluate(() => document.fonts.ready);
  const report = await page.evaluate(() => {
    const fail = [], near = (a, b, tolerance = 1e-10) => Math.abs(a - b) <= tolerance;
    const check = (condition, description) => { if (!condition) fail.push(description); };
    const pretty = value => (Math.abs(value) < 0.005 ? 0 : value).toFixed(2).replace('-', '−');
    const methods = { windowDimensions: 0, embeddingSpace: 0, lookupConcat: 2, learningGraph: 3, embeddingGradients: 0, trainingVsGeneration: 2, knownTrainingWindows: 0, chosenGenerationWindows: 0,
      generationRun: 11, generationTrace: 0, generationChoice: 0, temperatureComparison: 0 };
    const toy = window.__TOY__, mlp = AT.mlp;
    const original = JSON.stringify(toy);
    for (const id of ['s06-net']) {
      const svg = document.querySelector(`#${id} svg`);
      check(!!svg, `${id}: missing network sketch`);
      if (!svg) continue;
      const hidden = [...svg.querySelectorAll('.col-hid .node:not(.ell)')];
      check(toy.d_h === 32 && hidden.length === 8, `${id}: abbreviate 32 hidden units using eight visible units`);
      check(JSON.stringify(hidden.map(node => Number(node.dataset.unit))) === JSON.stringify([1, 2, 3, 4, 29, 30, 31, 32]), `${id}: hidden endpoints must retain their real unit numbers`);
      check(hidden.every(node => node.querySelector('.unit-l')?.textContent === node.dataset.unit), `${id}: hidden numbers must be visible`);
      check(svg.querySelectorAll('.col-hid .ell').length === 1 && svg.querySelectorAll('.col-out .ell').length === 2, `${id}: every omitted hidden/output range needs dots`);
      check(svg.querySelector('.col-out .is-hl')?.dataset.label === 'i', `${id}: preserve the observed target highlight`);
      check(svg.getAttribute('aria-label').includes('32 hidden nodes') && svg.getAttribute('aria-label').includes('27 vocabulary entries'), `${id}: accessibility label must give actual widths`);
      const shownOutputs = [...svg.querySelectorAll('.col-out .node')].map(node => node.classList.contains('ell') ? '…' : node.dataset.label);
      check(JSON.stringify(shownOutputs) === JSON.stringify(['-', 'a', 'b', '…', 'i', '…', 'y', 'z']), `${id}: maintain vocabulary order with both gaps`);
      const inputs = svg.querySelectorAll('.col-in circle').length;
      const outputs = svg.querySelectorAll('.col-out circle').length;
      check(svg.querySelectorAll('.edges line').length === inputs * hidden.length + hidden.length * outputs, `${id}: connect every visible real unit, never a gap`);
      const span = selector => {
        const ys = [...svg.querySelectorAll(selector)].map(node => node.transform.baseVal.getItem(0).matrix.f);
        return Math.max(...ys) - Math.min(...ys);
      };
      check(span('.col-hid .node') > span('.col-out .node'), `${id}: hidden layer should look larger than the abbreviated output layer`);
    }
    check(document.querySelectorAll('#s06-net .col-in circle').length === toy.w * toy.d_model
      && !document.querySelector('#s06-net .col-in .ell'), 'full-network sketch must show all six input coordinates');
    const net = document.querySelector('#s06-net svg');
    const params = [...net.querySelectorAll('[data-parameter]')];
    check(params.map(node => node.dataset.parameter).join(',') === 'W1,W2,b1,b2', 'label both weight matrices and both biases on the network');
    const proseColour = getComputedStyle(document.querySelector('#s06 .p1-param')).color;
    check(params.every(node => getComputedStyle(node).fill === proseColour), 'network parameters and their prose must share a colour');
    for (const label of params) {
      const b = label.getBBox(), v = net.viewBox.baseVal;
      check(b.x >= 0 && b.y >= 0 && b.x + b.width <= v.width && b.y + b.height <= v.height, `network parameter ${label.dataset.parameter} must fit its SVG`);
    }
    const oneHotProducts = [...document.querySelectorAll('#s04-dot1 tbody tr:last-child th, #s04-dot2 tbody tr:last-child th')];
    check(oneHotProducts.length === 2 && oneHotProducts.every(th => th.textContent === 'products'), 'Part I one-hot worksheets must not introduce query/key notation');
    const host = document.createElement('div');
    host.id = 'part1-diagram-regression-fixtures';
    host.style.cssText = 'position:absolute;top:0;left:0;width:1100px;background:white;z-index:99999';
    document.body.appendChild(host);
    const instances = [];
    for (let copy = 0; copy < 2; copy++) {
      for (const [method, last] of Object.entries(methods)) {
        check(typeof AT.part1Diagrams[method] === 'function', `missing ${method} helper`);
        if (typeof AT.part1Diagrams[method] !== 'function') continue;
        for (const stage of [...Array(last + 1).keys(), 'default']) {
          const svg = stage === 'default' ? AT.part1Diagrams[method]() : AT.part1Diagrams[method]({ stage });
          const label = `${method}/${stage}/copy${copy}`;
          check(svg instanceof SVGSVGElement, `${label}: helper must return an SVG element`);
          if (!(svg instanceof SVGSVGElement)) continue;
          svg.style.width = '1100px';
          svg.style.display = 'block';
          host.appendChild(svg);
          instances.push({ method, stage, label, svg });
          if (method !== 'embeddingSpace') check(Number(svg.getAttribute('data-stage')) === (stage === 'default' ? last : stage), `${label}: wrong stage metadata`);
        }
      }
    }

    const ids = [...document.querySelectorAll('[id]')].map(node => node.id);
    const duplicates = ids.filter((id, i) => ids.indexOf(id) !== i);
    check(!duplicates.length, `duplicate document IDs: ${[...new Set(duplicates)].join(', ')}`);
    let labels = 0, markers = 0;
    for (const { svg, label } of instances) {
      check(svg.getAttribute('role') === 'img', `${label}: role=img is required`);
      const namedBy = (svg.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean);
      check(namedBy.length >= 2, `${label}: title and description must label the graphic`);
      for (const id of namedBy) {
        const target = document.getElementById(id);
        check(target && svg.contains(target) && target.textContent.trim(), `${label}: missing local accessible label ${id}`);
      }
      check(svg.querySelector(':scope > title')?.textContent.trim(), `${label}: missing accessible title`);
      check(svg.querySelector(':scope > desc')?.textContent.trim(), `${label}: missing accessible description`);
      check(!/\b(?:undefined|NaN|Infinity)\b/.test(svg.outerHTML), `${label}: invalid generated content`);
      const vb = svg.viewBox.baseVal;
      check(vb.width > 0 && vb.height > 0, `${label}: invalid viewBox`);
      for (const node of svg.querySelectorAll('text')) {
        labels++;
        // Transform the four corners back to this SVG's coordinate system so
        // rotated labels and viewBox scaling are checked correctly.
        const box = node.getBBox(), matrix = svg.getScreenCTM().inverse().multiply(node.getScreenCTM());
        const corners = [[box.x, box.y], [box.x + box.width, box.y], [box.x, box.y + box.height], [box.x + box.width, box.y + box.height]]
          .map(([x, y]) => new DOMPoint(x, y).matrixTransform(matrix));
        const tolerance = 1.5;
        check(corners.every(p => p.x >= vb.x - tolerance && p.y >= vb.y - tolerance && p.x <= vb.x + vb.width + tolerance && p.y <= vb.y + vb.height + tolerance),
          `${label}: label outside viewBox: ${node.textContent}`);
      }
      for (const node of svg.querySelectorAll('path,line,polyline')) {
        for (const name of ['markerStart', 'markerMid', 'markerEnd']) {
          const value = getComputedStyle(node)[name];
          if (!value || value === 'none') continue;
          markers++;
          const ref = value.match(/#([^\s)"']+)/)?.[1];
          const target = ref && document.getElementById(ref);
          check(target?.tagName.toLowerCase() === 'marker' && svg.contains(target), `${label}: ${name} must reference a marker in its own SVG, got ${value}`);
        }
      }
    }
    for (const [method, last] of Object.entries(methods)) {
      const lastSvg = instances.find(d => d.method === method && d.stage === last)?.svg;
      const defaultSvg = instances.find(d => d.method === method && d.stage === 'default')?.svg;
      const texts = svg => [...svg.querySelectorAll('text')].map(t => t.textContent);
      check(JSON.stringify(texts(lastSvg)) === JSON.stringify(texts(defaultSvg)), `${method}: default does not show the final stage`);
    }

    const final = method => instances.find(d => d.method === method && d.stage === 'default').svg;
    const scatter = final('embeddingSpace'), points = [...scatter.querySelectorAll('circle[data-row-id]')];
    check(points.length === toy.vocab.length && points.length === 27, 'scatter must show all 27 learned embedding rows');
    const coordinates = points.map((point, index) => {
      const title = point.querySelector('title')?.textContent || '';
      const match = title.match(/^(.*): \[([^,]+),\s*([^\]]+)\]$/);
      check(match && match[1] === toy.vocab[index] && near(Number(match[2]), toy.E[index][0]) && near(Number(match[3]), toy.E[index][1]), `scatter row ${index}: title does not match the stored embedding`);
      check(Number(point.getAttribute('data-row-id')) === index && point.getAttribute('data-token') === toy.vocab[index]
        && JSON.stringify(JSON.parse(point.getAttribute('data-coordinates'))) === JSON.stringify(toy.E[index]), `scatter row ${index}: metadata does not match the stored embedding`);
      return [Number(point.getAttribute('cx')), Number(point.getAttribute('cy'))];
    });
    // Each plotted coordinate must be an affine mapping of the actual learned
    // coordinate, not a hand-positioned conceptual cluster.
    for (let axis = 0; axis < 2; axis++) {
      const second = toy.E.findIndex(row => Math.abs(row[axis] - toy.E[0][axis]) > 1e-8);
      const slope = (coordinates[second][axis] - coordinates[0][axis]) / (toy.E[second][axis] - toy.E[0][axis]);
      check(axis === 0 ? slope > 0 : slope < 0, `scatter axis ${axis}: invalid orientation`);
      for (let i = 0; i < points.length; i++) {
        check(near(coordinates[i][axis], coordinates[0][axis] + slope * (toy.E[i][axis] - toy.E[0][axis]), 1e-7), `scatter row ${i}, axis ${axis}: position is not the actual embedding coordinate`);
      }
    }
    check(/sign-constrained/.test(scatter.textContent) && /learned axis 2/.test(scatter.textContent), 'scatter must distinguish constrained vowel-ness from the unnamed learned axis');

    const input = ['a', 'a', 'b'], rows = mlp.embed(input), concat = mlp.concat(rows);
    const lookup = final('lookupConcat');
    const shownConcat = [...lookup.querySelectorAll('text.label.mono')].map(node => node.textContent);
    const concatCells = [...lookup.querySelectorAll('[data-concat-index]')];
    check(concatCells.length === concat.length && concatCells.every((cell, i) => Number(cell.getAttribute('data-concat-index')) === i && near(Number(cell.getAttribute('data-value')), concat[i])), 'lookup output cells must retain their actual values and ordered indices');
    check(concat.length === 6 && JSON.stringify(shownConcat) === JSON.stringify(concat.map(pretty)), 'lookup diagram must display the six actual concatenated coordinates in order');
    check(JSON.stringify(rows[0]) === JSON.stringify(rows[1]) && /one a row, two lookups/.test(lookup.textContent), 'repeated a occurrences must share the same stored embedding row');

    const forward = mlp.forward(input), p = forward.p[toy.vocab.indexOf('i')], loss = mlp.loss(input, 'i');
    const graph = final('learningGraph'), graphText = graph.textContent;
    check(near(Number(graph.getAttribute('data-target-probability')), p) && near(Number(graph.getAttribute('data-loss')), loss), 'learning graph metadata must preserve the exact probability and loss');
    const dims = [`${toy.E.length} × ${toy.E[0].length}`, `1 × ${forward.a0.length}`, `${toy.W1.length} × ${toy.W1[0].length}`, `1 × ${forward.a1.length}`, `${toy.W2.length} × ${toy.W2[0].length}`, `1 × ${forward.p.length}`];
    for (const dim of dims) check(graphText.includes(dim), `learning graph: missing actual model shape ${dim}`);
    check(graphText.includes(`p(i) = ${p.toFixed(3)}`) && graphText.includes(`loss = ${pretty(loss)}`), 'learning graph probability/loss must agree with the actual forward pass');
    check(/target i is not looked up as an input/.test(graphText) && /reverse gradients/.test(graphText) && /ReLU/.test(graphText), 'learning graph must keep target, gradient direction, and hidden activation explicit');

    const expanded = final('embeddingGradients'), embeddingRows = [...expanded.querySelectorAll('[data-embedding-row]')];
    check(embeddingRows.length === toy.vocab.length && embeddingRows.length === 27, 'expanded embedding table must show every row without ellipses');
    check(expanded.dataset.gradientScope === 'single-example-cross-entropy', 'row-gradient claim must be scoped to this example’s loss');
    embeddingRows.forEach((row, i) => {
      check(Number(row.dataset.embeddingRow) === i && row.dataset.token === toy.vocab[i], `expanded row ${i}: wrong row ID or token`);
      check(JSON.stringify(JSON.parse(row.dataset.coordinates)) === JSON.stringify(toy.E[i]), `expanded row ${i}: changed stored coordinates`);
      check(JSON.stringify([...row.querySelectorAll('[data-coordinate]')].map(n => n.textContent)) === JSON.stringify(toy.E[i].map(pretty)), `expanded row ${i}: displayed coordinates must match the actual table`);
      const count = input.filter(token => token === toy.vocab[i]).length;
      check(Number(row.dataset.lookupContributions) === count && !!row.querySelector('.lookup-gradient') === (count > 0), `expanded row ${i}: only a and b should receive lookup gradients`);
    });
    check(/Autograd adds them/.test(expanded.textContent) && /Other 25 rows: zero gradient/.test(expanded.textContent) && /Extra regularization/.test(expanded.textContent), 'expanded table must explain accumulation, untouched rows, and scope');
    const learning = document.querySelector('#s09-learning .stepper') || document.querySelector('#s09-learning');
    check(learning.stepperApi.steps.length === 4 && !!document.querySelector('#s09-embedding-gradients svg').closest('.frame'), 'full table needs its own follow-on frame so final-state PDF exports retain both the graph and table');

    const lanes = final('trainingVsGeneration');
    const seed = Number(lanes.getAttribute('data-sample-seed'));
    const temperature = Number(lanes.getAttribute('data-sample-temperature'));
    const sample = mlp.generate({ ids: input, seed, temperature, maxLength: 1, greedy: false });
    const draw = sample.trace[0];
    check(seed === 1 && temperature === 1 && draw.chosen === 'h', 'generation example must use the ReLU model’s actual seed-1, temperature-1 draw of h');
    check(lanes.getAttribute('data-sample-token') === draw.chosen && lanes.textContent.includes(`sample  ${draw.chosen}`), 'generation label must match the sampled token');
    check(JSON.stringify(draw.next_context) === JSON.stringify(['a', 'b', 'h']) && lanes.textContent.includes(`append ${draw.chosen}, shift window → ${draw.next_context.join(' ')}`), 'generation must append h and shift a a b to a b h');
    check(lanes.getAttribute('data-next-context') === draw.next_context.join(' '), 'generation metadata must retain the next input window');
    check(lanes.textContent.includes('fixed during generation') && lanes.textContent.includes('boundary “-” would stop'), 'generation must explain fixed parameters and the boundary stop');
    const parameterNames = ['E_tok', 'W1', 'b1', 'W2', 'b2'];
    for (const { svg, method, stage } of instances.filter(item => item.method === 'trainingVsGeneration')) {
      const phase = stage === 'default' ? 2 : stage;
      const trainable = svg.querySelector('[data-parameter-state="trainable"]');
      const frozen = svg.querySelector('[data-parameter-state="frozen"]');
      check(JSON.stringify([...trainable.querySelectorAll('[data-parameter]')].map(el => el.dataset.parameter)) === JSON.stringify(parameterNames), 'training must name the embedding table, both weights, and both biases');
      check(!!svg.querySelector('[data-update-symbol]') && !!svg.querySelector('[data-loop="parameter-update"]'), 'training needs a visible update symbol and return arrow');
      check(!!frozen === (phase >= 1) && !!svg.querySelector('[data-lock-symbol]') === (phase >= 1), 'frozen parameters and lock must appear only when generation is revealed');
      check(svg.querySelectorAll('[data-frozen-symbol]').length === (phase >= 1 ? 5 : 0), 'show exactly five snowflakes for generation parameters');
      if (frozen) {
        check(JSON.stringify([...frozen.querySelectorAll('[data-parameter]')].map(el => el.dataset.parameter)) === JSON.stringify(parameterNames), 'generation must freeze all five parameter groups');
        check(JSON.stringify([...frozen.querySelectorAll('[data-frozen-symbol]')].map(el => el.dataset.frozenSymbol)) === JSON.stringify(parameterNames), 'each snowflake must belong to its named parameter');
        check(!frozen.querySelector('.grad-edge') && !svg.querySelector('[data-frozen-symbol]:not([data-parameter-state="frozen"] *)'), 'updates never enter the frozen group, and activations never get snowflakes');
      }
      check(!!svg.querySelector('[data-loop="next-input"]') === (phase === 2), 'the next-input loop must reveal last, separate from parameter updates');
      check(svg.querySelector('desc').textContent.includes('Activations and probabilities are recomputed'), 'accessible description must distinguish frozen parameters from recomputed activations');
    }
    const knownWindows = final('knownTrainingWindows');
    const trainingRows = [...knownWindows.querySelectorAll('[data-training-row]')];
    check(trainingRows.length === 6, 'batch explanation must show all six input/target pairs');
    trainingRows.forEach((row, i) => {
      check(Number(row.dataset.trainingRow) === i && row.dataset.context === toy.aabid_rows[i].context.join(' ')
        && row.dataset.target === toy.aabid_rows[i].target, `batch row ${i}: input and target must come from the observed name`);
      check(Number(row.dataset.outputWidth) === toy.vocab.length && row.textContent.includes('27 scores'), `batch row ${i}: each example must receive its own vocabulary score row`);
    });
    check(trainingRows[3].dataset.context === 'a a b' && trainingRows[3].dataset.target === 'i'
      && trainingRows[4].dataset.context === 'a b i', 'observed i must enter the next training window');
    const chosenWindows = final('chosenGenerationWindows');
    const dependencyRun = mlp.generate({ ids: input, seed: Number(chosenWindows.dataset.sampleSeed), temperature: Number(chosenWindows.dataset.sampleTemperature), maxLength: 2 });
    const calls = [...chosenWindows.querySelectorAll('[data-generation-call]')];
    check(calls.length === 2 && dependencyRun.trace.length === 2 && dependencyRun.trace[0].chosen === 'h'
      && dependencyRun.trace[1].chosen === '-', 'generation dependency must show the real h then stop run');
    let generatedName = input.join('');
    calls.forEach((call, i) => {
      const step = dependencyRun.trace[i];
      if (step.chosen !== '-') generatedName += step.chosen;
      check(call.dataset.context === step.context.join(' ') && call.dataset.chosen === step.chosen, `dependency call ${i}: wrong sampled input/output`);
      check(near(Number(call.dataset.chosenProbability), step.probabilities[step.chosen_id])
        && call.textContent.includes(step.probabilities[step.chosen_id].toFixed(3)), `dependency call ${i}: show real sampling probability`);
      check(call.dataset.name === generatedName && call.querySelector('[data-lock-symbol]'), `dependency call ${i}: preserve prefix, omit stop token, freeze parameters`);
    });
    check(calls[1].dataset.context === dependencyRun.trace[0].next_context.join(' ') && calls[1].dataset.context === 'a b h', 'sampled h must enter the next generation window');
    check(calls[1].dataset.name === 'aabh' && chosenWindows.querySelector('[data-loop="chosen-input"]'), 'generation must stop after the real dependency chain');
    const section11Frames = [...document.querySelectorAll('#s11 .frame')];
    check(section11Frames[1].contains(document.querySelector('#s11-known-windows'))
      && section11Frames[2].querySelector('[data-torch="batch"]')
      && section11Frames[3].contains(document.querySelector('#s11-chosen-windows')), 'explain known windows before batch code, then contrast generation');
    check(section11Frames[3].textContent.includes('several names in a batch'), 'sequential within one name must not imply that generation cannot batch independent names');
    let boundary;
    for (let boundarySeed = 1; boundarySeed <= 20000; boundarySeed++) {
      const candidate = mlp.generate({ ids: input, seed: boundarySeed, temperature: 1, maxLength: 8 });
      if (candidate.trace.at(-1)?.chosen === '-') { boundary = candidate; break; }
    }
    check(boundary && boundary.trace.at(-1).chosen === '-' && !boundary.name.includes('-') && boundary.trace.length === boundary.name.length + 1, 'a sampled boundary must stop generation and must not enter the generated name');
    const worked = AT.part1Diagrams.generationExample();
    check(worked.name === 'sam' && worked.trace.length === 4 && worked.trace.at(-1).chosen === '-', 'worked run must reach the boundary after the three letters sam');
    check(worked.trace[0].context.join('') === '---', 'worked run must start from three boundary tokens');
    const walk = document.querySelector('#s10-walk');
    check(walk.stepperApi.steps.length === worked.trace.length * 3, 'every call needs predict, choose, and append/stop phases');
    for (const {svg, stage} of instances.filter(d => d.method === 'generationRun')) {
      const s = stage === 'default' ? 11 : stage, round = Math.floor(s / 3), phase = s % 3, t = worked.trace[round];
      const before = worked.trace.slice(0, round).map(t => t.chosen).join('');
      const expectedName = before + (phase === 2 && t.chosen !== '-' ? t.chosen : '');
      check(svg.dataset.context === t.context.join(',') && svg.dataset.generated === expectedName, `generation stage ${s}: context and growing name must remain distinct`);
      const rows = [...svg.querySelectorAll('[data-prob-token]')];
      check(rows.length === 6 && new Set(rows.map(r => r.dataset.probToken)).size === 6, `generation stage ${s}: show six distinct probability entries`);
      rows.forEach(r => check(near(Number(r.dataset.probability), mlp.distribution(t.context, 1).p[mlp.stoi[r.dataset.probToken]]), `generation stage ${s}: probabilities must come from the displayed context`));
      check(rows.some(r => r.dataset.probToken === t.chosen), `generation stage ${s}: chosen token must not be hidden in 'other'`);
      check(rows.filter(r => r.dataset.selected === 'true').length === (phase ? 1 : 0), `generation stage ${s}: highlight only after the draw`);
      check(t.probabilities.every(p => p <= .5), `generation stage ${s}: fixed chart axis must cover all probabilities`);
      const next = [...svg.querySelectorAll('[data-next-token]')].map(n => n.dataset.nextToken);
      check(JSON.stringify(next) === JSON.stringify(phase === 2 && t.chosen !== '-' ? t.next_context : []), `generation stage ${s}: shift only after appending a letter, never after END`);
      check(svg.dataset.stopped === String(s === 11), `generation stage ${s}: stop only at the boundary`);
    }
    const traceRows = [...final('generationTrace').querySelectorAll('[data-trace-round]')];
    check(traceRows.length === 4 && traceRows.every((r, i) => near(Number(r.dataset.probability), worked.trace[i].probabilities[worked.trace[i].chosen_id])), 'printable trace must retain all four real chosen-token probabilities');
    const greedy = mlp.generate({greedy:true, temperature:1, maxLength:18});
    check(greedy.name === 'san' && greedy.trace[2].context.join(',') === worked.trace[2].context.join(','), 'comparison must diverge from the same - s a probability row');
    check(greedy.trace[2].chosen === 'n' && worked.trace[2].chosen === 'm', 'greedy picks n while this sampling run draws m');
    check(final('generationChoice').textContent.includes(worked.trace[2].probabilities[mlp.stoi.n].toFixed(3)) && final('generationChoice').textContent.includes(worked.trace[2].probabilities[mlp.stoi.m].toFixed(3)), 'comparison probabilities must match the saved model');
    const frames = [...document.querySelectorAll('#s10 .frame')];
    check(frames.findIndex(f => f.contains(document.querySelector('#s10-choice'))) + 1 === frames.findIndex(f => f.contains(document.querySelector('#s10-next'))), 'choice comparison must immediately precede the original live generator');
    const temperaturePanels = [...final('temperatureComparison').querySelectorAll('[data-temperature]')];
    check(temperaturePanels.map(p => p.dataset.temperature).join(',') === '0.5,1,1.5', 'temperature comparison needs low, unchanged, and high settings');
    const temperatureInput = ['-', 's', 'a'], temperatureForward = mlp.forward(temperatureInput);
    const entropies = [], nProbabilities = [];
    temperaturePanels.forEach((panel, i) => {
      const t = Number(panel.dataset.temperature), distribution = mlp.distribution(temperatureInput, t);
      const manual = AT.softmax(temperatureForward.z.map(z => z / t));
      check(distribution.p.every((p, j) => near(p, manual[j])) && near(distribution.p.reduce((a,b)=>a+b,0),1), `temperature ${t}: normalize divided logits, not probabilities`);
      check(distribution.z.every((z, j) => z === temperatureForward.z[j]), `temperature ${t}: original model scores must remain unchanged`);
      check(AT.argmax(distribution.p) === mlp.stoi.n, `temperature ${t}: greedy's favourite must remain n`);
      nProbabilities.push(distribution.p[mlp.stoi.n]);
      entropies.push(-distribution.p.reduce((h, p) => h + p * Math.log(p), 0));
      const rows = [...panel.querySelectorAll('[data-temperature-token]')];
      check(rows.map(r=>r.dataset.temperatureToken).join(',') === 'n,r,h,m,a', `temperature ${t}: compare the same five labelled tokens`);
      rows.forEach(r => check(near(Number(r.dataset.probability), distribution.p[mlp.stoi[r.dataset.temperatureToken]]), `temperature ${t}: bars must agree with live generation`));
      const tableRow = document.querySelectorAll('#s10-temperature-table tbody tr')[i];
      const expected = [temperatureForward.z[mlp.stoi.n]/t, temperatureForward.z[mlp.stoi.m]/t, distribution.p[mlp.stoi.n]].map(n=>AT.fmt(n,3));
      check(JSON.stringify([...tableRow.querySelectorAll('td')].map(c=>c.textContent)) === JSON.stringify(expected), `temperature ${t}: worksheet must use actual logits and full 27-token softmax`);
    });
    check(entropies[0]<entropies[1] && entropies[1]<entropies[2] && nProbabilities[0]>nProbabilities[1] && nProbabilities[1]>nProbabilities[2], 'higher temperature must flatten this distribution and lower its maximum');
    const codeIndex = frames.findIndex(f => f.querySelector('[data-torch="sample-function"]'));
    check(['s10-temperature-bars','s10-temperature-guide'].every(id => {
      const index = frames.findIndex(f=>f.contains(document.getElementById(id)));
      return index >= 0 && index < codeIndex;
    }), 'explain temperature before both its code and the live control');
    check(codeIndex > 0 && frames[codeIndex-1].contains(document.getElementById('s10-temperature-guide')), 'temperature formula must lead directly into sampling code');
    const arithmetic = document.getElementById('s10-temperature-table');
    check(!!arithmetic.closest('.companion') && !arithmetic.closest('.frame'), 'duplicate temperature arithmetic belongs only in the article companion');
    check(JSON.stringify(toy) === original, 'rendering and generation must not modify learned model parameters');
    host.remove();
    return { failures: fail, instances: instances.length, labels, markers, points: points.length, boundarySeed: boundary?.seed };
  });
  assert.deepEqual(errors, [], 'assembled Part I must have no browser errors');
  assert.deepEqual(report.failures, [], `Part I diagram regressions failed:\n${report.failures.join('\n')}`);
  // Both architecture controls must change the diagram and the same worksheet.
  const dimensions = await page.evaluate(() => {
    const failures = [], original = JSON.stringify(__TOY__);
    const windows = [1, 3, 5, 10, 100], widths = [1, 2, 4, 8, 256];
    const set = (id, value) => {
      const input = document.querySelector(id + ' input');
      input.value = value; input.dispatchEvent(new Event('input', { bubbles: true }));
    };
    AT.present.enter();
    for (const [wi, w] of windows.entries()) for (const [di, d] of widths.entries()) {
      set('#s14-window', wi); set('#s14-dimension', di);
      const expected = [27*d, w*d*32, 32, 32*27, 27];
      const counts = [...document.querySelectorAll('#s14-counts tbody tr')].map(row => Number(row.lastElementChild.textContent.replace(/,/g, '')));
      if (JSON.stringify(counts) !== JSON.stringify(expected)) failures.push(`${w}/${d}: parameter rows disagree`);
      if (Number(document.querySelector('#s14').dataset.parameterCount) !== expected.reduce((a,b)=>a+b,0)) failures.push(`${w}/${d}: total disagrees`);
      const svg = document.querySelector('#s14-net svg');
      if (Number(svg.dataset.inputWidth) !== w*d || Number(svg.dataset.window) !== w || Number(svg.dataset.dimension) !== d) failures.push(`${w}/${d}: diagram metadata differs`);
      const indices = (n, limit) => n <= limit ? Array.from({length:n},(_,i)=>i+1) : [1,2,n-1,n];
      const expectedCells = indices(w,5).flatMap(r=>indices(d,5).map(c=>`${r},${c}`));
      if (JSON.stringify([...svg.querySelectorAll('[data-coordinate]')].map(el=>el.dataset.coordinate)) !== JSON.stringify(expectedCells)) failures.push(`${w}/${d}: wrong embedding tiles`);
      if (JSON.stringify([...svg.querySelectorAll('[data-flat-coordinate]')].map(el=>Number(el.dataset.flatCoordinate))) !== JSON.stringify(indices(w*d,8))) failures.push(`${w}/${d}: wrong flattened tiles`);
      const vb = svg.viewBox.baseVal;
      for (const node of svg.querySelectorAll('text')) {
        const b = node.getBBox();
        if (b.x < -1 || b.y < -1 || b.x+b.width > vb.width+1 || b.y+b.height > vb.height+1) failures.push(`${w}/${d}: clipped label ${node.textContent}`);
      }
      for (const f of [1,2]) {
        AT.present.go('s14', f, 0);
        if (AT.present.fitReport().overflow) failures.push(`${w}/${d}: frame ${f} overflows`);
      }
    }
    if (JSON.stringify(__TOY__) !== original) failures.push('Explorer mutated trained parameters');
    AT.present.go('s14',1,0);
    if (document.querySelector('#s14-window input').value !== '4' || document.querySelector('#s14-dimension input').value !== '4') failures.push('Navigation reset control values');
    set('#s14-window',1);set('#s14-dimension',1);
    return failures;
  });
  assert.deepEqual(dimensions, [], dimensions.join('\n'));
  await page.locator('#s14-dimension input').focus();
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#s14').getAttribute('data-dimension'), '4');
  await page.keyboard.press('ArrowLeft');
  assert.equal(await page.locator('#s14').getAttribute('data-dimension'), '2');
  // Actual classroom controls: play, pause on manual navigation, rewind, stop,
  // leave-frame cancellation, and reduced-motion support.
  await page.evaluate(() => { AT.present.enter(); AT.present.go('s10', 1, 0); });
  await page.clock.install();
  await page.locator('#s10-play').click();
  await page.clock.runFor(2450);
  assert.equal(await page.locator('#s10-walk').evaluate(el => el.stepperApi.index()), 1);
  await page.evaluate(() => document.querySelector('#s10-walk').stepperApi.prev());
  assert.equal(await page.locator('#s10-play').getAttribute('aria-pressed'), 'false');
  await page.clock.runFor(3000);
  assert.equal(await page.locator('#s10-walk').evaluate(el => el.stepperApi.index()), 0);
  await page.locator('#s10-play').click();
  await page.clock.runFor(27000);
  assert.equal(await page.locator('#s10-walk').evaluate(el => el.stepperApi.index()), 11);
  assert.equal(await page.locator('#s10-play').getAttribute('aria-pressed'), 'false');
  assert.equal(await page.locator('#s10-walk svg').getAttribute('data-generated'), 'sam');
  await page.locator('#s10-play').click();
  assert.equal(await page.locator('#s10-walk').evaluate(el => el.stepperApi.index()), 0);
  await page.evaluate(() => AT.present.go('s10', 2, 0));
  await page.clock.runFor(3000);
  assert.equal(await page.locator('#s10-play').getAttribute('aria-pressed'), 'false');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.evaluate(() => { AT.present.go('s10', 1, 0); document.querySelector('#s10-walk').stepperApi.go(8); });
  assert.equal(await page.locator('#s10-walk .token-move').first().evaluate(el => getComputedStyle(el).animationName), 'none');
  await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
  await page.emulateMedia({ media: 'print' });
  assert.equal(await page.locator('#s10-walk').evaluate(el => el.stepperApi.index()), 11);
  assert.equal(await page.locator('#s10-trace [data-trace-round]').count(), 4);
  assert.equal(await page.locator('#s10-play').evaluate(el => getComputedStyle(el.parentElement).display), 'none');
  await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
  await page.emulateMedia({ media: 'screen' });
  assert.equal(await page.locator('#s10-walk').evaluate(el => el.stepperApi.index()), 8);
  console.log(`PASS: MLP widths and all 25 window/embedding choices; ${report.instances} Part I SVG instances, ${report.labels} bounded labels, ${report.markers} local marker references; all ${report.points} embedding coordinates, lookup values, model shapes, probability/loss, seeded generation, and boundary stop (seed ${report.boundarySeed}) agree with the live model.`);
} finally {
  await browser.close();
}

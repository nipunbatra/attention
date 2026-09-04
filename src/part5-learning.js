/* Vision I's exact 44-parameter learning experiment. No dependencies or sampling.
   This is a separate parameter snapshot: AT.vision.data and the encoder shared
   with Vision IV are never mutated. Verify with verify_vision1_learning.py. */
(function () {
  'use strict';
  var T = window.AT && window.AT.vision;
  if (!T) throw new Error('Load Vision I before its learning experiment.');
  var names = ['W_patch', 'b_patch', 'cls', 'positions', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_class', 'b_class'];
  function copy(x) { return JSON.parse(JSON.stringify(x)); }
  function map(x, fn) { return Array.isArray(x) ? x.map(function (v) { return map(v, fn); }) : fn(x); }
  function zip(a, b, fn) { return Array.isArray(a) ? a.map(function (v, i) { return zip(v, b[i], fn); }) : fn(a, b); }
  function zeros(n, d) { return Array.from({length:n}, function () { return Array(d).fill(0); }); }
  function transpose(a) { return a[0].map(function (_, j) { return a.map(function (r) { return r[j]; }); }); }
  function mm(a, b) { return a.map(function (r) { return b[0].map(function (_, j) { return r.reduce(function (s, v, k) { return s + v * b[k][j]; }, 0); }); }); }
  function add(a, b) { return zip(a, b, function (x, y) { return x + y; }); }
  function sumRows(a) { return a.reduce(function (s, r) { return add(s, r); }, a[0].map(function () { return 0; })); }
  function sm(a) { var m = Math.max.apply(null, a), e = a.map(function (x) { return Math.exp(x-m); }), z = e.reduce(function (s, x) { return s+x; }, 0); return e.map(function (x) { return x/z; }); }
  var initial = {};
  names.forEach(function (name) { initial[name] = copy(T.data[name]); });
  var images = [copy(T.data.image), copy(T.data.image)];
  for (var y=0; y<2; y++) for (var x=0; x<2; x++) images[1][y][x]=0;
  var targets = [0, 1], classes = ['two blocks', 'one block'];

  function forward(p, image, target) {
    image = image || images[0]; target = target == null ? 0 : target;
    if (target !== 0 && target !== 1) throw new Error('The target must be class index 0 or 1.');
    var patches = T.patchify(image, 2).patches;
    var embeddings = mm(patches, p.W_patch).map(function (r) { return add(r, p.b_patch); });
    var content = [p.cls.slice()].concat(embeddings), positions = copy(p.positions), E = add(content, positions);
    var Q = mm(E, p.W_Q), K = mm(E, p.W_K), V = mm(E, p.W_V), raw = mm(Q, transpose(K));
    var S = map(raw, function (v) { return v/Math.sqrt(2); }), A = S.map(sm);
    var message = mm(A, V), delta = mm(message, p.W_O), updated = add(E, delta);
    var logits = add(mm([updated[0]], p.W_class)[0], p.b_class), probs = sm(logits), peak = Math.max.apply(null, logits);
    var loss = peak + Math.log(logits.reduce(function (s, v) { return s+Math.exp(v-peak); }, 0)) - logits[target];
    return {image:copy(image), patches:patches, embeddings:embeddings, content:content, positions:positions, E:E, Q:Q, K:K, V:V, raw:raw, S:S, A:A,
      message:message, delta:delta, updated:updated, logits:logits, probs:probs, loss:loss, target:target};
  }

  // Reverse-mode differentiation through the complete graph, including both
  // the attention-weight branch (Q,K) and the value branch (V).
  function backward(p, image, target) {
    var f = forward(p, image, target), g = {}, dz = f.probs.slice(); dz[f.target] -= 1;
    g.W_class = mm(transpose([f.updated[0]]), [dz]); g.b_class = dz.slice();
    var du = zeros(5, 2); du[0] = mm([dz], transpose(p.W_class))[0];
    g.W_O = mm(transpose(f.message), du);
    var dm = mm(du, transpose(p.W_O)), da = mm(dm, transpose(f.V)), dv = mm(transpose(f.A), dm);
    var ds = f.A.map(function (r, i) {
      var weighted = r.reduce(function (s, v, j) { return s+v*da[i][j]; }, 0);
      return r.map(function (v, j) { return v*(da[i][j]-weighted); });
    });
    var draw = map(ds, function (v) { return v/Math.sqrt(2); });
    var dq = mm(draw, f.K), dk = mm(transpose(draw), f.Q);
    g.W_Q = mm(transpose(f.E), dq); g.W_K = mm(transpose(f.E), dk); g.W_V = mm(transpose(f.E), dv);
    var deQ = mm(dq, transpose(p.W_Q)), deK = mm(dk, transpose(p.W_K)), deV = mm(dv, transpose(p.W_V));
    var de = add(add(du, deQ), add(deK, deV)), dembed = de.slice(1);
    g.positions = copy(de); g.cls = de[0].slice();
    g.W_patch = mm(transpose(f.patches), dembed); g.b_patch = sumRows(dembed);
    var dp = mm(dembed, transpose(p.W_patch)), dimage = zeros(4, 4);
    for (var j=0; j<4; j++) for (var k=0; k<4; k++) dimage[2*Math.floor(j/2)+Math.floor(k/2)][2*(j%2)+k%2] = dp[j][k];
    return {loss:f.loss, grads:g, forward:f, graph:{logits:dz, updated:du, delta:copy(du), message:dm, A:da, V:dv, S:ds, raw:draw, Q:dq, K:dk,
      E:de, EResidual:copy(du), EQuery:deQ, EKey:deK, EValue:deV, content:copy(de), embeddings:copy(dembed), patches:dp, image:dimage}};
  }
  function batch(p) {
    var examples = images.map(function (im, i) { return backward(p, im, targets[i]); }), grads = {};
    names.forEach(function (name) { grads[name] = map(add(examples[0].grads[name], examples[1].grads[name]), function (v) { return v/2; }); });
    var losses = examples.map(function (e) { return e.loss; });
    return {loss:(losses[0]+losses[1])/2, losses:losses, grads:grads, examples:examples};
  }
  // Out-of-place SGD: every parameter, including CLS and all five positions.
  function step(p, grads, lr) {
    if (!Number.isFinite(lr) || lr <= 0) throw new Error('Learning rate must be positive and finite.');
    var next = {}; names.forEach(function (name) { next[name] = zip(p[name], grads[name], function (v, g) { return v-lr*g; }); }); return next;
  }
  function stage(p) {
    var fs = images.map(function (im, i) { return forward(p, im, targets[i]); }), losses = fs.map(function (f) { return f.loss; });
    return {params:copy(p), forwards:fs, losses:losses, meanLoss:(losses[0]+losses[1])/2,
      predictions:fs.map(function (f) { return f.probs[0] >= f.probs[1] ? 0 : 1; })};
  }
  function runExperiment(lr, nsteps) {
    lr = lr == null ? 0.05 : lr; nsteps = nsteps == null ? 600 : nsteps;
    if (!Number.isInteger(nsteps) || nsteps < 1) throw new Error('Use at least one whole SGD step.');
    var p = copy(initial), before = stage(p), first = batch(p), afterSingle, history = [{step:0, loss:before.meanLoss, losses:before.losses.slice()}];
    for (var it=1; it<=nsteps; it++) {
      p = step(p, it===1 ? first.grads : batch(p).grads, lr);
      if (it===1) afterSingle = stage(p);
      if (it===1 || it%10===0 || it===nsteps) {
        var f = stage(p); history.push({step:it, loss:f.meanLoss, losses:f.losses.slice()});
      }
    }
    var afterTraining = stage(p);
    var specs = [{name:'W_class', index:[0,0]}, {name:'W_patch', index:[0,0]}, {name:'W_Q', index:[0,0]}, {name:'W_K', index:[0,0]},
      {name:'W_V', index:[0,0]}, {name:'W_O', index:[0,0]}, {name:'cls', index:[0]}, {name:'positions', index:[1,0]},
      {name:'b_patch', index:[0]}, {name:'b_class', index:[0]}];
    function cell(p, spec) { return spec.index.reduce(function (v, k) { return v[k]; }, p[spec.name]); }
    var selectedCells = specs.map(function (s) { return {name:s.name, index:s.index, label:s.name+s.index.map(function (i) { return '['+i+']'; }).join(''),
      before:cell(initial,s), gradient:cell(first.grads,s), afterSingle:cell(afterSingle.params,s), afterTraining:cell(afterTraining.params,s)}; });
    return {before:before, afterSingle:afterSingle, afterTraining:afterTraining, firstBatch:first, lr:lr, nsteps:nsteps, batchSize:2,
      history:history, selectedCells:selectedCells, parameterCount:44, objective:'mean cross-entropy on these two training images'};
  }
  T.learning = {initial:initial, images:images, targets:targets, classes:classes, parameterNames:names,
    forward:forward, backward:backward, batch:batch, step:step, runExperiment:runExperiment};
  T.learning.experiment = runExperiment();
})();

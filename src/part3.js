(function () {
  'use strict';

  var AT = window.AT;
  var training = (AT.model && AT.model.training) || {};

  function etaKey(eta) {
    var wanted = Number(eta);
    var steps = training.single && training.single.steps ? training.single.steps : {};
    var keys = Object.keys(steps);
    for (var i = 0; i < keys.length; i++) {
      if (Math.abs(Number(keys[i]) - wanted) < 1e-12) return keys[i];
    }
    return null;
  }

  AT.train = {
    raw: training,
    sentence: function () { return (training.sentence || []).slice(); },
    target: function () { return training.target || ''; },
    single: function () { return training.single || {}; },
    step: function (eta) {
      var key = etaKey(eta);
      return key ? training.single.steps[key] : null;
    },
    etas: function () {
      var steps = training.single && training.single.steps ? training.single.steps : {};
      return Object.keys(steps).map(Number).sort(function (a, b) { return a - b; });
    },
    gradient: function (name) {
      var grads = training.single && training.single.gradients ? training.single.gradients : {};
      return grads[name];
    },
    parametersAfter: function (eta) {
      var step = this.step(eta);
      return step ? step.parameters : null;
    },
    parallel: function () { return training.parallel || {}; },
    finiteDifference: function () { return training.finite_difference || {}; }
  };

  AT.ln = function (row, eps) {
    var x = Array.isArray(row) ? row.map(function (v) { return Number(v) || 0; }) : [];
    var epsilon = eps == null ? 1e-5 : Math.max(0, Number(eps) || 0);
    if (!x.length) return { input: [], mean: 0, variance: 0, centered: [], denominator: 0, output: [], eps: epsilon };
    var mean = x.reduce(function (a, b) { return a + b; }, 0) / x.length;
    var centered = x.map(function (v) { return v - mean; });
    var variance = centered.reduce(function (a, v) { return a + v * v; }, 0) / x.length;
    var denominator = Math.sqrt(variance + epsilon);
    return {
      input: x.slice(),
      mean: mean,
      variance: variance,
      centered: centered,
      denominator: denominator,
      output: centered.map(function (v) { return denominator ? v / denominator : 0; }),
      eps: epsilon
    };
  };

  var W1 = [
    [0.8, -0.4, 0.2, 0.0, 0.5, -0.3, 0.1, 0.0],
    [-0.3, 0.9, 0.0, 0.2, -0.2, 0.4, 0.0, 0.1],
    [0.2, 0.1, 0.8, -0.4, 0.0, 0.2, 0.3, -0.1],
    [0.1, 0.2, -0.2, 0.8, 0.3, 0.0, -0.3, 0.4],
    [0.0, 0.1, 0.2, 0.1, 0.4, 0.3, -0.2, 0.6]
  ];
  var b1 = [0.1, -0.1, 0.0, 0.2, 0.0, -0.2, 0.1, 0.0];
  var W2 = [
    [0.5, -0.1, 0.1, 0.0, 0.0],
    [-0.2, 0.6, 0.0, 0.1, 0.0],
    [0.1, 0.0, 0.5, -0.1, 0.1],
    [0.0, 0.1, -0.2, 0.5, 0.0],
    [0.3, -0.2, 0.0, 0.1, 0.2],
    [-0.1, 0.3, 0.2, 0.0, 0.1],
    [0.2, 0.0, 0.3, -0.2, 0.0],
    [0.0, 0.1, -0.1, 0.2, 0.4]
  ];
  var b2 = [0.0, 0.0, 0.0, 0.0, 0.0];

  function addBias(row, bias) {
    return row.map(function (v, i) { return v + bias[i]; });
  }

  function ffn(row) {
    var input = Array.isArray(row) ? row.slice(0, 5).map(function (v) { return Number(v) || 0; }) : [];
    while (input.length < 5) input.push(0);
    var pre = addBias(AT.matmul(input, W1), b1);
    var hidden = pre.map(function (v) { return Math.max(0, v); });
    var output = addBias(AT.matmul(hidden, W2), b2);
    return { input: input, pre: pre, hidden: hidden, output: output };
  }
  ffn.W1 = W1;
  ffn.b1 = b1;
  ffn.W2 = W2;
  ffn.b2 = b2;
  ffn.d_model = 5;
  ffn.d_ff = 8;
  ffn.illustrative = true;
  AT.ffn = ffn;

  function causalMask(n) {
    return AT.range(n).map(function (i) {
      return AT.range(n).map(function (j) { return j <= i ? 0 : -Infinity; });
    });
  }

  AT.heads = {
    illustrative: true,
    causalMask: causalMask,
    examples: {
      priya: {
        label: 'Illustrative',
        tokens: ['Priya', 'opened', 'the', 'notebook', ';', 'she', 'smiled'],
        queryIndex: 5,
        queryLabel: 'she',
        rows: [
          { name: 'Referent', note: 'Who does she refer to?', weights: [0.58, 0.08, 0.06, 0.12, 0.10, 0.06, 0.00] },
          { name: 'Event', note: 'What happened before she appeared?', weights: [0.06, 0.28, 0.08, 0.40, 0.12, 0.06, 0.00] },
          { name: 'Local form', note: 'Which nearby marks shape the next relation?', weights: [0.03, 0.05, 0.08, 0.14, 0.48, 0.22, 0.00] }
        ],
        mask: causalMask(7)
      },
      river: {
        label: 'Illustrative',
        tokens: ['The', 'fisherman', 'sat', 'beside', 'the', 'river', 'bank'],
        queryIndex: 6,
        queryLabel: 'bank',
        rows: [
          { name: 'Setting', note: 'Which setting gives bank its sense here?', weights: [0.04, 0.14, 0.05, 0.07, 0.04, 0.48, 0.18] },
          { name: 'Nearby syntax', note: 'Which words form the local phrase?', weights: [0.02, 0.03, 0.05, 0.10, 0.24, 0.42, 0.14] },
          { name: 'Subject', note: 'Who is involved in this scene?', weights: [0.08, 0.44, 0.12, 0.06, 0.08, 0.14, 0.08] }
        ],
        mask: causalMask(7)
      }
    }
  };
})();

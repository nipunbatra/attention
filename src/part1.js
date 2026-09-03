(function () {
  'use strict';

  var AT = window.AT;
  var model = window.__TOY__ || {};
  var vocab = Array.isArray(model.vocab) ? model.vocab.slice() : [];
  var stoi = {};
  vocab.forEach(function (token, i) { stoi[token] = i; });
  var w = Number(model.w) || 3;

  function idsOf(input) {
    var ids = Array.isArray(input) ? input.slice() : [];
    ids = ids.map(function (value) {
      if (typeof value === 'string') return stoi[value] == null ? 0 : stoi[value];
      value = Number(value);
      return Number.isFinite(value) && value >= 0 && value < vocab.length ? Math.floor(value) : 0;
    });
    while (ids.length < w) ids.unshift(0);
    return ids.slice(-w);
  }

  function embedWith(parameters, ids) {
    return idsOf(ids).map(function (id) { return (parameters.E[id] || []).slice(); });
  }

  function embed(ids) {
    return embedWith(model, ids);
  }

  function concat(rows) {
    var out = [];
    (Array.isArray(rows) ? rows : []).forEach(function (row) {
      (Array.isArray(row) ? row : []).forEach(function (value) { out.push(Number(value) || 0); });
    });
    return out;
  }

  function addBias(row, bias) {
    return row.map(function (value, i) { return value + (Number(bias[i]) || 0); });
  }

  function forwardWith(parameters, ids) {
    var rows = embedWith(parameters, ids);
    var a0 = concat(rows);
    var pre = addBias(AT.matmul(a0, parameters.W1 || []), parameters.b1 || []);
    var a1 = pre.map(function (value) { return Math.tanh(value); });
    var z = addBias(AT.matmul(a1, parameters.W2 || []), parameters.b2 || []);
    var p = AT.softmax(z);
    return { a0: a0, a1: a1, z: z, p: p };
  }

  function forward(ids) { return forwardWith(model, ids); }

  // The displayed generation bars and the sampler share this exact distribution.
  function distribution(ids, temperature) {
    var result = forward(ids);
    var value = Math.max(0.05, Number(temperature) || 1);
    return { z: result.z, p: AT.softmax(result.z.map(function (logit) { return logit / value; })) };
  }

  function targetId(target) {
    if (typeof target === 'string') return stoi[target] == null ? 0 : stoi[target];
    target = Number(target);
    return Number.isFinite(target) && target >= 0 && target < vocab.length ? Math.floor(target) : 0;
  }

  function loss(ids, target) {
    var p = forward(ids).p[targetId(target)] || 0;
    return -Math.log(Math.max(p, 1e-12));
  }

  function mulberry32(seed) {
    var state = seed >>> 0;
    return function () {
      state += 0x6D2B79F5;
      var x = state;
      x = Math.imul(x ^ x >>> 15, x | 1);
      x ^= x + Math.imul(x ^ x >>> 7, x | 61);
      return ((x ^ x >>> 14) >>> 0) / 4294967296;
    };
  }

  function choose(probabilities, random) {
    var draw = random(), total = 0;
    for (var i = 0; i < probabilities.length; i++) {
      total += probabilities[i];
      if (draw <= total) return i;
    }
    return Math.max(0, probabilities.length - 1);
  }

  function generate(options) {
    options = options || {};
    var temperature = Math.max(0.05, Number(options.temperature) || 1);
    var greedy = !!options.greedy;
    var seed = Number.isFinite(Number(options.seed)) ? Number(options.seed) : Number(model.sample_seed) || 1;
    var maxLength = Math.max(1, Number(options.maxLength) || 18);
    var random = mulberry32(seed);
    var context = idsOf(options.ids || []);
    var chars = [];
    var trace = [];
    for (var step = 0; step < maxLength; step++) {
      var result = distribution(context, temperature);
      var probabilities = result.p;
      var next = greedy ? AT.argmax(probabilities) : choose(probabilities, random);
      var before = context.slice();
      var after = context.slice(1).concat([next]);
      trace.push({
        step: step + 1,
        context: before.map(function (id) { return vocab[id]; }),
        ids: before,
        z: result.z.slice(),
        probabilities: probabilities.slice(),
        chosen: vocab[next],
        chosen_id: next,
        next_context: after.map(function (id) { return vocab[id]; })
      });
      if (next === 0) break;
      chars.push(vocab[next]);
      context = after;
    }
    return { name: chars.join(''), trace: trace, temperature: temperature, greedy: greedy, seed: seed };
  }

  function tokenizeChars(text) {
    return Array.from(String(text == null ? '' : text).toLowerCase()).map(function (char) { return char === ' ' ? '␠' : char; });
  }

  function rawWords(text) {
    return String(text == null ? '' : text).toLowerCase().match(/[a-z]+|[^\s\w]/g) || [];
  }

  var knownWords = new Set(['deep', 'learning', 'is', 'useful', 'the', 'cat', 'sat', 'on', 'mat', 'transformer']);
  function tokenizeWords(text, options) {
    options = options || {};
    return rawWords(text).map(function (word) {
      return options.unknown && /^[a-z]+$/.test(word) && !knownWords.has(word) ? '<UNK>' : word;
    });
  }

  var mergePairs = [
    ['u', 'n'], ['b', 'e'], ['l', 'i'], ['e', 'v'], ['be', 'li'], ['beli', 'ev'],
    ['a', 'b'], ['l', 'e'], ['ab', 'le'], ['t', 'o'], ['to', 'k'], ['tok', 'e'],
    ['toke', 'n'], ['i', 'z'], ['a', 't'], ['i', 'o'], ['io', 'n'], ['at', 'ion'],
    ['i', 'n'], ['in', 'g'], ['l', 'e'], ['le', 'a'], ['lea', 'r'], ['lear', 'n']
  ];

  function mergeWord(word) {
    var pieces = Array.from(word);
    mergePairs.forEach(function (pair) {
      for (var i = 0; i < pieces.length - 1;) {
        if (pieces[i] === pair[0] && pieces[i + 1] === pair[1]) pieces.splice(i, 2, pair[0] + pair[1]);
        else i += 1;
      }
    });
    return pieces;
  }

  function tokenizeSubwords(text) {
    var out = [];
    rawWords(text).forEach(function (word) {
      if (/^[a-z]+$/.test(word)) out = out.concat(mergeWord(word));
      else out.push(word);
    });
    return out;
  }

  // Part 1 names only embedding coordinates.  The shared notation builder also
  // knows the Part 2 query, key, and value axes, so narrow its axes group here.
  if (AT.ui && AT.ui.notationCard) {
    var sharedNotationCard = AT.ui.notationCard;
    AT.ui.notationCard = function (options) {
    options = options || {};
    var requested = options.groups ? options.groups.slice() : ['mlp', 'sizes', 'axes'];
    var wantsAxes = requested.indexOf('axes') >= 0;
    var clean = {};
    Object.keys(options).forEach(function (key) { clean[key] = options[key]; });
    clean.groups = requested.filter(function (group) { return group !== 'axes'; });
    var root = sharedNotationCard(clean);
    if (!wantsAxes) return root;
    var group = AT.h('div', { class: 'notation-group' });
    group.appendChild(AT.h('p', { class: 'notation-title' }, 'Named embedding coordinates'));
    var table = AT.h('table', { class: 'dt dt-notation' });
    table.appendChild(AT.h('thead', {}, AT.h('tr', {}, AT.h('th', {}, 'symbol'), AT.h('th', {}, 'meaning'), AT.h('th', {}, 'shape'))));
    var names = (AT.axes && AT.axes.e ? AT.axes.e : []).map(function (name) { return '“' + name + '”'; }).join(', ');
    table.appendChild(AT.h('tbody', {}, AT.h('tr', {}, AT.h('td', { html: '$\\ve{e}$' }), AT.h('td', {}, 'coordinates ' + names + ' of one character embedding'), AT.h('td', { html: '$1\\times d$ = 1×' + (model.d_model || 0) }))));
    group.appendChild(AT.h('div', { class: 'dt-scroll' }, table));
    group.appendChild(AT.h('p', { class: 'notation-note' }, 'The first coordinate received a vowel sign constraint during training.'));
    root.appendChild(group);
    AT.renderMath(group);
      return root;
    };
  }

  AT.mlp = {
    model: model,
    vocab: vocab,
    stoi: stoi,
    w: w,
    embed: embed,
    concat: concat,
    forward: forward,
    forwardWith: forwardWith,
    distribution: distribution,
    loss: loss,
    generate: generate,
    tokenizeChars: tokenizeChars,
    tokenizeWords: tokenizeWords,
    tokenizeSubwords: tokenizeSubwords,
    mergeTable: mergePairs.map(function (pair) { return { left: pair[0], right: pair[1], merged: pair[0] + pair[1] }; })
  };
})();

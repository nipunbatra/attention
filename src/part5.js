/* part5.js: Vision I runtime on toy5.json (v2, named axes). Requires shared.js and vision-shared.js. */
(function () {
  'use strict';
  var AT = window.AT, V = AT.vision, T = window.__TOY__ || {};
  if (!V || !T.scenes) return;
  V.model = T; V.classes = T.classes; V.curve = V.curve; V.trainingCurve = T.curve || [];
  V.params = function (which) { return which === 'initial' ? T.initial : T.trained; };
  /* forward(scene, which): scene = 'A'..'E' or an 8x8 array; which = 'initial' | 'trained' (default trained) */
  V.forward = function (scene, which) { return V.attend(V.scene(scene), V.params(which || 'trained')); };
  V.encode = function (scene, which) { return V.forward(scene, which).Enew; };     // 17 updated rows (CLS first): the frozen encoder later parts reuse
  V.attention = function (scene, which, receiver) { var A = V.forward(scene, which).A; return receiver == null ? A[0] : A[receiver]; };
  V.predict = function (scene, which) { var f = V.forward(scene, which); var i = f.probs[0] >= f.probs[1] ? 0 : 1; return { label: T.classes[i], probs: f.probs, logits: f.logits }; };
  V.regions = function (scene) { var g = V.scene(scene), o = []; for (var j = 0; j < 16; j++) o.push(V.regionOf(g, j)); return o; };
  V.labelOf = function (key) { return T.scenes[key] && T.scenes[key].label; };
  V.tokensPerImage = function (side, patch) { var n = Math.floor(side / patch); return { patches: n * n, rows: n * n + 1, scores: (n * n + 1) * (n * n + 1) }; };
  /* notation rows for the card (same shape as the old runtime used) */
  function note(g, sym, mean, shape, dims) { AT.notation.push({ g: g, sym: sym, mean: mean, shape: shape, dims: function () { return dims || ''; }, parts: ['vision1'] }); }
  if (AT.notation && AT.notation.push) {
    note('token', 'r_j', 'Raw pixels of patch j, read left to right then top to bottom', '1\\times 4', '1×4');
    note('token', '\\ve{e_j}=r_jW_{\\mathrm{patch}}+p_j', 'Patch row on named axes: brightness, contrast, row, col', '1\\times d_{\\mathrm{model}}', '1×4');
    note('token', '\\ve{e_0}=c+p_0', 'The CLS row: no pixels, a fixed start, its own position', '1\\times d_{\\mathrm{model}}', '1×4');
    note('token', '\\vq{q_i}=\\ve{e_i}W_Q,\\ \\vk{k_j}=\\ve{e_j}W_K,\\ \\vv{v_j}=\\ve{e_j}W_V', 'Ask (bright region? on the right?), offer, send', '1\\times 2', '');
    note('token', '\\vd{\\Delta e_0}=m_0W_O,\\ \\vp{e_0^\\prime}=\\ve{e_0}+\\vd{\\Delta e_0}', 'The update CLS receives and its updated row', '1\\times d_{\\mathrm{model}}', '1×4');
    note('matrix', '\\ve{E},\\vq{Q},\\vk{K},\\vv{V},\\va{A}', 'CLS plus sixteen patches: 17 rows', '17\\times d', '');
    note('sizes', 'N,\\ s_{\\mathrm{patch}},\\ d_{\\mathrm{model}},\\ d_k,\\ d_v', 'Patches, patch side, widths', '', '16, 2, 4, 2, 2');
  }
})();

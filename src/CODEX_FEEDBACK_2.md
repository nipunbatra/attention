# Codex feedback, round 2 (assessed; all accepted with the notes below). Apply together with REV2_TASK.md.

E1 Keys and values numerically identical (W_K = W_V in toy v2). FIX: REV2 already gives values their own 2-column space (W_V 5x2). ALSO add the intervention
   in s08 Phase B (and reuse in s11 if cheap): a toggle "change W_V: mute the finance column" (or scale it by 0.5). Attention weights stay identical, the message
   and Delta e change. One line: keys choose where to read; values decide what is read.
E2 Row-vector convention in printed math. Under q_i = e_i W_Q the score is q_i k_j^T, not q_i^T k_j. FIX: write single-token scores as the dot product
   $\vq{q_i} \cdot \vk{k_j}$ everywhere (prose, KaTeX, SVG labels, strip tooltip in part2.json/shell), keep $S = QK^\top$ in matrix form, and write the update as
   $\Delta e_i = \big(\sum_j \alpha_{ij} v_j\big) W_O = m_i W_O$. Sweep: grep for "\^\\top" and "^T" in sections/, shell.html, part2.json, shared.js (notation card).
E3 Fixed window criticised too strongly (s03 around line 57): a big matrix CAN respond differently to river and the. FIX: say membership is position based and the
   slot weights are fixed by position, not chosen by reading content; attention adds explicit content-dependent weighting over the available positions;
   a larger window changes the first layer's input width (not necessarily later layers).
E4 s14 "attention off" shows a self-weight of 1 but computes Delta e = 0. FIX: label the mode "attention bypassed", show no attention weights in that mode.
E5 s12: the slider keeps the raw scores fixed, so it shows the effect of dividing by sqrt(d_k), not variance growth with d_k. FIX: say exactly that; keep the
   variance simulation as the argument for growth and state its assumptions (independent, zero-mean, unit-variance coordinates).
E6 Qualifications: (a) q/k/v are projected representations, not the updated token state (s11 ~line 194); (b) a zero key still gets positive softmax weight
   (score 0 is not weight 0): fix any sentence saying glue words get "no" weight; (c) s13 ~line 84: position 5 predicts river only, not river and bank
   (the second leaked token is the one after); (d) "70% of the message" -> "70% of the attention mass" (magnitudes of v also matter).
E7 Bugs: s15 weighted-value worksheet stops updating after the first query-token switch; s07 arithmetic headings can keep old token indices after switching;
   index/prev/next links to part1.html and part3.html 404 until those parts exist: render them as "coming" (no href) when the file is absent
   (assemble.py can check the sibling file at build time and set a flag in window.__PART__).

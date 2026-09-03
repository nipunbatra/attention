# GUIDE — per-section interaction design (what each fragment must contain)

These notes refine SPEC.md; SPEC.md wins on content and notation, BRIEF.md/CONTRACT.md win on engineering.
"S_A" = AT.sentences.river (The fisherman sat beside the river bank and watched the), "S_B" = AT.sentences.cheque
(She deposited the cheque at the bank and watched the). "bank" is position 7 in both; the prediction slot is after position 10.
The SHORT sequence is the first 7 tokens of S_A: The fisherman sat beside the river bank. Every number on the page comes from AT.forward / AT.baseline / AT.embed.
Standardise the sentence wording on "beside" (SPEC §1 says "near" once; use "beside" everywhere for consistency).

## s01 Predict the next token  (data-lit="")
- Sentence chips for S_A with a blank slot "___" at the end; below it probability bars for the top-6 candidates from AT.forward(S_A).probs[9] (label the rest "other").
- A control to switch the context to S_B → bars change. Message: context changes the answer. Treat the model as a black box: "some model" (do NOT mention attention, Q/K/V, or Δe).
- Equation $p(x_{t+1}\mid x_1,\ldots,x_t)$. Callout: "To predict the next token well, the model needs a useful representation of everything relevant in the preceding context."
- Introduce $\ve{e_i^{(0)}}$ = token embedding + positional information as the starting representation (show e^{(0)} for one token as a vec; show tok_emb + pos_emb = e^{(0)} for the last token).

## s02 Baseline: only the last token  (data-lit="e")
- Chips where only the last token is lit; all earlier tokens greyed with a "not used" style. Vec $\ve{e_t}$. Equation $\ell = W_{\text{vocab}} e_t + b$, $p = \operatorname{softmax}(\ell)$.
- Bars from AT.baseline(tokens).probs[9]. Switch context S_A ↔ S_B: the bars are IDENTICAL — say so explicitly ("same last token, same position, same numbers → the earlier context cannot matter").
- Show the diffuse, hedging distribution. Callout: this throws away most of the context.

## s03 Baseline: a fixed window of K tokens  (data-lit="")
- Slider K (1..10). Chips inside the window highlighted, outside greyed; a vertical "hard boundary" marker. Show $c_t = [e_{t-K+1};\ldots;e_t]$ as K stacked e-vectors (real numbers) with the live dimension $K\cdot d_{\text{model}}$ and the parameter count of $W$ ($K d_{\text{model}} \times |\mathcal V|$).
- Four compact discussion cards (hard boundary / everything included equally / relevant context changes dynamically / bigger K does not fix it). For "relevant context changes dynamically" show two prediction slots in S_A (after "river" → next is "bank": relevant = river, fisherman; after "watched the" → relevant = river, bank, fisherman) with relevant tokens outlined (label these outlines "illustrative").
- End with the question callout: "Could we dynamically retrieve only the information that is useful right now?"

## s04 Weighted pooling  (data-lit="a")
- Mean pooling: chips of the short sequence with equal weight bars (1/i) above them; the pooled vector $c_i = \frac1i\sum_{j\le i}\ve{e_j}$ computed from AT.embed.
- Weighted pooling: one slider per token (weights normalised to sum to 1, shown as $\va{\alpha_{ij}}$ bars); $c_i=\sum_{j\le i}\va{\alpha_{ij}}\ve{e_j}$ recomputed live. (Here the pooled objects are still e_j — values come later.)
- Crucial question callout: "Where should the weights α_ij come from?" → "They should depend on what token i currently needs and what information token j contains." No Q/K/V yet.

## s05 A detour through search  (data-lit="q k v")
- A fake search engine: fixed query "How does a neural network send gradient information backwards?"; six video cards (Backpropagation, Gradient Descent, CNNs, Transformers, Batch Normalization, Regularization). Each card shows a KEY (amber: short matching tags / a tiny key vector) and a VALUE (teal: the payload — a one-line summary the user gets back + a tiny 3-number topic vector, e.g. [gradients, optimisation, architecture]). Query shown in purple.
- Similarity table with the SPEC numbers: Gradient Descent 1.4, Backpropagation 4.8, CNN 0.6, Transformers 0.2 (add Batch Normalization 0.9, Regularization 0.5 as illustrative). Hovering a row highlights the card. Equation $s_j = \vq{q}^\top \vk{k_j}$.
- The three boxed slogans (Q: What am I looking for? K: When should you retrieve me? V: What information do I send if retrieved?) and the callout "The key is used to decide whether something matches. The value is the information we actually retrieve."
- Optional second preset query ("How do I stop overfitting?") with a different illustrative similarity column.

## s06 Hard → soft retrieval  (data-lit="")
- Toggle "Hard (argmax)" vs "Soft (softmax)". Sliders for the six similarity scores (defaults = the s05 table). Softmax weights as $\va{\alpha_j}$ bars; retrieved result = $\sum_j \va{\alpha_j}\vv{v_j}$ shown as the mixed topic vector (teal) and, for hard mode, exactly $v_{j^*}$.
- Equations $j^*=\arg\max_j q^\top k_j$, $\alpha_j=\operatorname{softmax}_j(q^\top k_j)$, boxed $\sum_j \alpha_j v_j$. Callout: "Attention does not necessarily retrieve one thing. It retrieves a weighted mixture of information."
- Optional temperature slider to show hard as the sharp limit.

## s07 Every token becomes a searchable record  (data-lit="")
- Short sequence chips. Click a token to act as the QUERY token i (default bank): it shows $\vq{q_i}=W_Q\ve{e_i}$ as a purple vec. Every token j (j ≤ i) flips open to show its record: $\vk{k_j}$ (amber) and $\vv{v_j}$ (teal) computed from $\ve{e_j}$. Show W_Q, W_K, W_V as small 4×3 matrices with dims. Boxed equations. No matrix-form S, A yet; one query at a time.

## s08 Two phases  (data-lit="")
- Two large panels. PHASE A — READ ROUTING (Q, K): SVG tree: $q_{\text{bank}}$ at the top, edges down to $k_j$ for each earlier token with the score $q_i^\top k_j$ on each edge; then a softmax row → $\va{\alpha_{ij}}$ bars. Callout "Q, K determine routing / relevance". State: no representation has been updated yet.
- PHASE B — MESSAGE PASSING (V): each token's $\vv{v_j}$; the message $\va{\alpha_{ij}}\vv{v_j}$ (teal, opacity/width ∝ α); animated dots flowing (AT.ui.flow) into $m_i=\sum_j\alpha_{ij}v_j$. Callout "V carries the information". Closing callout: "Queries and keys determine who communicates with whom. Values determine what gets communicated." Buttons: "Run phase A", "Run phase B", "Replay".

## s09 The contextual update Δe  (data-lit="d ep")  — CENTREPIECE
- Animated vertical SVG exactly in the spirit of the SPEC ascii diagram: $e_i$ splits; right branch Q_i → compare with K_j → attention weights → weighted V_j → $\Delta e_i$; left branch is the untouched $e_i$; they meet at ⊕ → $e_i'$. A "Play" button (and step controls) walks the highlight down the diagram; numbers for bank from the short sequence appear beside each stage (vec e, q, α row, m, Δe, e′).
- Equations: $m_i=\sum_j\alpha_{ij}v_j$, boxed $\Delta e_i = W_O m_i$, boxed $e_i^{\text{new}} = e_i + \Delta e_i$.
- PROMINENT caption: "Attention does not replace the existing representation. It computes context-dependent information that can be added to it."
- IMMEDIATELY after: the notation disclaimer callout (kind 'note'): "Calling this update Δe_i is pedagogical notation. Standard implementations usually call this the attention output and add it through a residual connection. The addition itself is standard: it is the residual connection around the attention sublayer."
- Include AT.motif synced with the animation.

## s10 "bank": same start, different context  (data-lit="")
- Two panels side by side: S_A (first 7 tokens) and S_B (first 7 tokens). In both, bank is the query and only looks LEFT (say so; causal).
- Show $e^{(0)}_{\text{bank}}$ in both — identical numbers (visually mark "identical"). Show the attention row (heat cells over the chips) for bank in both: S_A river highest, fisherman/beside moderate; S_B cheque & deposited highest. Show $\Delta e_{\text{bank}}=\alpha_1 v_1+\cdots+\alpha_7 v_7$ and $e'_{\text{bank}} = e_{\text{bank}}+\Delta e_{\text{bank}}$ with real numbers, visibly different.
- Conceptual visual: "generic BANK representation + river-context update → RIVER-BANK contextual representation" vs "+ cheque-context update → FINANCE-BANK". A 2-D plot (two coordinates or a fixed 2-D projection of e) showing the shared starting point and two different arrows Δe works well.

## s11 What exactly are Q, K, and V?  (data-lit="")
- Title exactly "What exactly are Q, K, and V?". Three-way branch diagram $e_i \to \{q_i=W_Qe_i,\ k_i=W_Ke_i,\ v_i=W_Ve_i\}$ with real numbers for one token. Callouts: "They are temporary linear projections of the current token representation." "They are not three new contextual embeddings. They play temporary functional roles during attention."
- A "lifetime" strip: e_i persists across the layer; q/k/v are computed, used, discarded; Δe_i is added to e_i. Large AT.motif with the boxed $e \to Q,K,V \to \Delta e \to e+\Delta e$.

## s12 Why divide by √d_k?  (data-lit="")
- Scores [2, 5, 9, 1]. Slider $d_k$ (1..64). Two bars panels: softmax([2,5,9,1]) (no scaling) vs softmax([2,5,9,1]/√d_k). Equation $s_{ij}=q_i^\top k_j/\sqrt{d_k}$; explanation: if entries are ~zero-mean unit-variance, $\operatorname{Var}(q^\top k)\propto d_k$ (show a small seeded simulation: sample variance of q·k for the chosen d_k, growing with d_k). "Large logits make softmax extremely sharp."

## s13 Causal masking  (data-lit="")
- The 4×4 ✓/× mask matrix (KaTeX bmatrix). Then the real 10×10 heatmap for S_A from AT.forward with a toggle "Causal mask: ON/OFF" (AT.ui.toggle). OFF → show Sfull/unmasked A; highlight the cells (i, i+1): "when predicting x_{i+1}, the token could read e_{i+1} — the answer is in the input". Show a leakage example from the toy (e.g. an early token putting weight on river/bank). Explain why training becomes invalid. Keep: $S=QK^\top$, tokens may only retrieve from $j\le i$.

## s14 From Δe to the next-token probabilities  (data-lit="")
- Context selector S_A / S_B. Chips → attention row of the(10) (heat over chips) → $\Delta e_t$ vec → $e'_t$ vec → logits bars → probability bars (top 6). Toggle "with attention / last token only" to show the contrast with s02. Equation $\ell = W_{\text{vocab}} e_t + b$, $p=\operatorname{softmax}(\ell)$.
- The boxed causal chain: previous tokens → attention → Δe_t → e′_t → next-token logits → next-token probabilities (as a horizontal chain of boxes that light up in sequence when the context changes).

## s15 Full walkthrough  (data-lit="")  — MAJOR VISUALISATION
- Uses S_A (all 10 tokens) so masking is visible; query token selectable at step 3 (default bank, position 7). A big "Next step" button (AT.ui.stepper big:true) with Previous/Reset; a numbered list of the 18 SPEC steps where ONLY the current one is highlighted; a stage area that shows exactly what that step produces (real numbers), and AT.motif synced (e → qkv → att → delta → add → ep).
- Step 18: the same computation runs for every position at once; for the last token the(10) the contextual $e'$ feeds the output head → show its probability bars (S_A: water/boats/fish/ducks).
- Steps: 1 tokens, 2 initial e_i, 3 choose query, 4 q_i, 5 each k_j, 6 q_i^T k_j, 7 scale, 8 mask, 9 softmax, 10 α_ij, 11 v_j, 12 α_ij v_j, 13 sum, 14 m_i, 15 W_O m_i, 16 "call this Δe_i (pedagogical)", 17 e_i' = e_i + Δe_i, 18 predict.

## s16 Matrix form  (data-lit="")
- Use the SHORT sequence (T = 7) so matrices fit. Stepper over: E (7×4) → Q, K, V (7×3, with W's 4×3) → S = QK^T/√d_k (7×7) → A = softmax(S + M) (heat, masked cells hatched) → H = AV (7×3) → ΔE = HW_O (7×4) → E' = E + ΔE (7×4). Dimensions always visible ($E\in\mathbb R^{T\times d_{\text{model}}}$ etc.). One operation highlighted at a time; the row for bank highlighted in every matrix to connect with s15.
- Restate: "ΔE is our pedagogical notation for the attention sublayer's residual update."

(Multiple heads and layer after layer moved to Part 3; their fragments are kept as raw material in sections3_seed/sec17.html and sec18.html.)

## s17 Attention versus the alternatives  (data-lit="")
- Four compact panels for the last token of the chosen context (S_A/S_B switch): fixed window (K=3, equal structural inclusion), mean pooling (uniform 1/t), fixed positional weights (e.g. decaying with distance, content-independent), attention (α from AT.forward — changes when the context changes). Weight bars over the chips in each panel. Boxed "Attention = content-dependent soft retrieval". Equations under each panel.

## s18 Pause and think  (data-lit="")
- The eight SPEC questions as AT.ui.reveal items with the SPEC answers (math rendered on open). Add the standard-operation equation $e_i^{\text{new}} = e_i^{\text{old}} + \text{AttentionOutput}_i$ and the Δ-notation disclaimer.

## s19 Three summaries  (data-lit="")
- Intuitive (quote), Operational (boxed chain), Mathematical (the four equations with boxed E' = E + ΔE), then the final chain E' → vocabulary logits → softmax → p(next token). Large AT.motif. A closing line that returns to the s01 sentence with the final probabilities, then the one-line pointer: Part 3 adds several heads, stacked layers and the rest of the Transformer.

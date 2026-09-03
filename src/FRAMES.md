# FRAMES.md — frame plan for Part 2 (attention.html) in presentation mode

Notation: F = frame, b = build. "held" = the drawing that stays while builds add one mark. Notes = presenter notes (first line = question before reveal).

## s01 Predict the next token
F1 "One blank": b0 sentence chips with the blank; b1 bars (top 6); b2 context switch button; b3 equation p(x_{t+1}|x_{<=t}); b4 callout (spec quote).
  Notes: "What could come next, and why do you think so?" Switch the context after collecting answers. Do not say attention.
F2 "Where a token starts": b0 table token + position = e^{(0)} (three aligned rows); b1 the equation; b2 note that every token gets such a row.
  Notes: "What does position add, and why is it small?"

## s02 Only the last token
F1: b0 chips with only the last token lit; b1 e_t row; b2 equation; b3 bars; b4 context switch (identical bars); b5 callout.
  Notes: "Before I switch the context: will the bars move?" (They cannot: same token, same position.)

## s03 A fixed window
F1 "Grow the window": b0 node network sketch with w = 3 inputs (e_1..e_3), hidden, 27-style outputs; b1 slider to 5 (network redraws, first layer takes 5d); b2 w = 100 (collapsed input column, parameter count); b3 claim "change the window, change the network".
  Notes: "What changes in the network if we use five tokens?"
F2 "Inside the window": b0 window slider with chips; b1 concatenation table; b2 dimension and parameter count readouts.
F3 "Four problems": b0..b3 the four cards one by one; b4 the question callout "Could we dynamically retrieve...".
  Notes: "Which of these does a bigger K fix?" (none of the last).

## s04 Weighted pooling
F1: b0 chips + equal-weight table (mean); b1 the pooled row; b2 sliders appear; b3 weighted footer; b4 question callout "Where should the weights come from?"; b5 answer callout.
  Notes: "Why not just average?" then "If you could set the weights by hand for bank, what would you choose and why?"

## s05 Search detour
F1 "A search box": b0 query text + six video cards; b1 the KEY tags highlighted on each card; b2 the VALUE payloads highlighted.
  Notes: "What does the site compare, and what does it hand back?"
F2 "Score the keys": b0 query row above the key table (axis headers); b1..b6 score column reveals row by row (Backpropagation 4.8 last or first, your call); b7 equation s_j = q^T k_j; b8 the three slogans.
  Notes: "Which column decides the winner here?"
F3 "The value is a different table": b0 value table with its own headers; b1 callout "key decides whether; value is what you get back".

## s06 Hard to soft
F1: b0 the two tables in hard mode (winner highlighted); b1 toggle to soft: alpha column; b2 weighted rows; b3 sum row; b4 sliders; b5 callout "weighted mixture"; b6 temperature slider.
  Notes: "If Gradient Descent's score rises to 4.8 too, what should come back?"

## s07 Tokens as records
F1: b0 chips; b1 pick query (bank) and its q row; b2 key rows for j <= i; b3 value rows; b4 the three boxed equations; b5 W tables.
  Notes: "Which of these three rows is the token's new representation?" (none)

## s08 Two phases
F1 "Read routing": b0 tree with q_bank on top; b1 scores on edges (dotTable alongside); b2 softmax; b3 alpha bars; b4 callout Q,K route.
F2 "Message passing": b0 value rows; b1 alpha column; b2 weighted rows; b3 flow animation into m_i; b4 footer sum; b5 callout V carries; b6 closing callout.
  Notes: "Has any representation changed yet?" (no) ... "What would change if only W_V changed?"

## s09 The update
F1: the centrepiece SVG; each stage is a build (e_i, q_i, compare K_j, alpha, weighted V, Delta e_i, plus, e_i'); then b8 caption (spec quote); b9 the notation disclaimer.
  Notes: "If the message were zero, what reaches the predictor?"

## s10 bank
F1: S_A table with alpha column and bank rows; F2: S_B the same; F3: both bank rows side by side + 2-D plot: b0 e^{(0)} identical, b1 the two Delta e arrows, b2 the two e'.
  Notes: "Same starting row: must the contextual row be the same?"

## s11 What are Q, K, V
F1: b0 e_i row; b1..b3 q, k, v rows; b4 callout temporary projections; b5 callout not new embeddings; b6 lifetime strip; b7 motif.

## s12 Scaling
F1: b0 the four scores; b1 softmax without scaling; b2 slider d_k; b3 softmax with scaling; b4 variance note.
  Notes: "Which of the two bar charts would you trust for d_k = 64?"

## s13 Causal mask
F1: b0 the 4x4 mask; b1 the rule j <= i. F2: b0 heatmap mask on; b1 toggle off; b2 the (i, i+1) cells outlined; b3 leakage callout; b4 why training breaks.
  Notes: "When predicting x_{i+1}, what is the worst thing token i could read?"

## s14 To the probabilities
F1: b0 chain boxes; b1 alpha row of the(10); b2 Delta e and e' rows; b3 logits; b4 probabilities; b5 switch context; b6 attention off toggle. F2: layer-boundary note.
  Notes: "Does the final 'the' read the updated bank row?" (not within this layer)

## s15 Walkthrough
One frame; the stepper's 18 steps are the builds. Notes: keep to the step list; ask before each reveal what the next quantity should be.

## s16 Matrix form
One frame; the operation stepper is the builds (E, Q/K/V, S, A, H, DeltaE, E'). Notes: "Which row is the one we did by hand?"

## s17 Alternatives
F1: b0..b3 the four panels one by one; b4 the comparison table; b5 context switch; b6 boxed claim.

## s18 Pause and think
F1: each question is a build; its answer is the next build (16 builds) or split into two frames of four questions.

## s19 Summaries
F1 intuitive; F2 operational chain (builds per arrow); F3 mathematical (four equations as builds) + final chain + motif; last build = the pointer to Part 3 (heads, stacked layers, the rest of the Transformer).

Timing guide (70 min): s01-s04 12 min, s05-s06 10, s07-s09 15, s10-s13 12, s14-s15 15, s16 4, s17-s19 6. Short-on-time: skip s03 F2, s10 F2, s17. Multi-head attention and stacked layers are Part 3 (raw material in sections3_seed/).

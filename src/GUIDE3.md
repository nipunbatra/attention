# GUIDE3.md — Part 3: Learning, heads, and the Transformer (mirrors L11M Lecture 2 end + Lecture 3)

Audience: students who finished Part 2. Job: show how the attention model learns from the observed next token, then build the full
decoder-only Transformer from parts they know, and compare it with the Part 1 character model. Same design system, named-axes toy from
Part 2 (toy.json copied as toy3.json plus precomputed training numbers), tables, worksheets, node sketches, frames + builds.

## Numbers (toy3.json + part3.js)
- Attention sections reuse the Part 2 toy (named axes water/finance/person/glue). Training numbers are precomputed offline with numpy
  (make_toy.py contains a reverse-mode autodiff): for S_A "The fisherman sat beside the river bank and watched the" with target "water":
  loss before, gradients for E (rows of the sentence), W_Q, W_K, W_V, W_O, W_vocab, b; the parameters after one SGD step for eta in {0.05, 0.1, 0.3};
  p(target) and loss after; and the parallel version (all ten positions, shifted targets, mean loss before/after). Export to toy3.json "training".
- part3.js: AT.train (lookup of those numbers), AT.ln(row) (LayerNorm live), AT.ffn (a tiny illustrative 4 -> 8 -> 4 FFN with fixed numbers),
  AT.heads (three illustrative head weight rows for the Priya sentence and for the short river sentence, labelled illustrative).

## Sections
s01 Where Part 2 left us   (recap: the motif; notation card; e'_bank for both contexts; "the numbers so far were fixed; where do they come from?")
s02 The observed next token tells us the error   F1: prefix, prediction bars, the observed target "water"; F2: cross-entropy worksheet -log p(target); the loss value.
s03 Which numbers will the optimizer change?   F1: table of parameters (E, W_Q, W_K, W_V, W_O, W_vocab, b, positions) vs intermediates (e, q, k, v, s, alpha, m, Delta e, e', logits); "a contextual update is not a learning update".
s04 The forward pass is a graph   F1: computational-graph diagram (builds: lookup -> projections -> scores -> softmax -> mix -> W_O -> add -> head -> loss); F2: autograd follows it backward (arrows reversed, gradient through the retained input); F3: gradient table for a few parameters (real numbers).
s05 One update   F1: theta <- theta - eta grad; learning-rate presets; before/after table (p(target), loss, e'_10); F2: "one example improved is not generalisation".
s06 Train every position at once   F1: shifted targets table (input row, target row); F2: the causal mask keeps it honest (heat map); F3: mean loss before/after; short PyTorch-style pseudocode (4 lines, labelled).
s07 Multiple heads   (from Part 2 s17, revised) F1: why one head is not enough (Priya sentence; three needs); F2: each head reads the full input through its own projections (diagram with builds: split -> heads -> concat -> W_O); F3: illustrative head weight rows with a head selector and a mask toggle; F4: widths (d_k = d_model / n_h) and the equations; W_O combines.
s08 Attention mixes rows; the FFN mixes features   F1: the two-axis grid (across tokens vs within a token); F2: per-token FFN as AT.netSketch (d_model -> d_ff -> d_model) with a worksheet for one token; F3: why it widens in the middle.
s09 The residual stream and LayerNorm   F1: the unbroken residual line with an attention branch and an FFN branch (builds); F2: LayerNorm calculator (edit a row: [1, 3, 5] -> [-2, 0, 2] -> [-1.22, 0, 1.22]); F3: pre-norm vs post-norm toggle, choose pre-norm.
s10 One block, then many   F1: block stepper x -> LN -> causal MHA -> + -> LN -> FFN -> +; the two update equations; F2: stacking (from Part 2 s18): e^{(0)} -> block 1 -> e^{(1)} -> ...; F3: multi-hop trace ("The trophy did not fit in the suitcase because it was too ___").
s11 The complete decoder-only model   F1: tokens -> embedding + position -> L blocks -> LM head -> probabilities (end-to-end diagram with builds); F2: shapes table; F3: LM head as a node sketch (d_model features -> |V| logits).
s12 Training: one pass, many guesses   F1: whole sequence in, shifted targets, loss at every position, one update; F2: the objective is the same as in Part 1 (aabid loss beside the Transformer loss).
s13 Generation: one token, then again   F1: stepper: prefix -> last-position distribution -> choose -> append -> repeat (two steps with bars); F2: KV cache: old keys and values do not change (animation appends k_t, v_t); F3: cost slider: T^2 scores and the cache growing with T.
s14 Three ways to arrange blocks   F1: encoder-only / decoder-only / encoder-decoder cards; F2: cross-attention uses the same calculation (queries from the decoder, keys and values from the source); F3: the 2017 original in one line.
s15 Sequence models side by side   F1: table: fixed-window MLP, RNN, temporal convolution, self-attention (context flexibility, path length, training parallelism, cost); F2: longer memory has a price.
s16 What attention gives, and what it does not   F1: gives learned content-dependent routing; does not guarantee facts, reasoning, interpretability, unlimited memory; F2: four quick checks as reveals (weights are explanations? one rule per head? masking only at generation? a Transformer remembers everything?).
s17 Back to aabid   F1: the two pipelines side by side (fixed three-row concatenation vs causal self-attention over any prefix); F2: "A Transformer language model differs from our aabid MLP mainly because ..." (reveal).
s18 Pause and think (eight reveals). s19 Three summaries + "Back to Part 1 / Part 2".

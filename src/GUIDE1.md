# GUIDE1.md — Part 1: From characters to next-token prediction (mirrors dl-teaching L11M Lecture 1)

Audience: students who know MLPs, softmax and cross-entropy. Job of the page: build the aabid name model end to end, generate real names,
then discover that a fixed window is baked into the network, which hands over to Part 2. Style: the same design system, tables with
named columns wherever coordinates mean something, node network sketches (AT.netSketch), worksheets for every calculation, companion prose,
frames + builds from the start. Running example: the name "aabid" with w = 3 and the boundary token "-"; vocabulary {-, a..z}, |V| = 27.

## Toy model (toy1.json, produced by train_names.py with numpy)
- Data: names.csv in the scratchpad (about 6,500 Indian given names, columns index,Name; lowercase, keep a-z only, dedupe). The page bundles only the trained weights and 20 sample names, not the list.
- Model: E (27 x 2), W_1 (6 x 32), b_1, W_2 (32 x 27), b_2, tanh hidden, w = 3. Train with cross-entropy, Adam, a few thousand steps.
- Make axis 1 of E readable: add a penalty so that vowels get positive axis-1 values and consonants negative ("vowel-ness"); axis 2 is learned ("learned axis 2").
  Report in toy1.json "axes": {"e": ["vowel-ness", "learned axis 2"]} and a note that axis 1 was constrained during training.
- Export: vocab, E, W1, b1, W2, b2, w, the six aabid rows, a training curve (loss every 50 steps), the before-training parameters too (for the s09 before/after),
  and 20 sampled names from the final model (seeded) for the page to show as examples. 2-decimal numbers.
- part1.js: AT.mlp = { embed(ids), concat(rows), forward(ids) -> {a0, a1, z, p}, loss(ids, target), generate({temperature, greedy, seed}) -> name with the step trace, tokenizeChars, tokenizeWords, tokenizeSubwords (a tiny illustrative BPE-like table) }.

## Sections
s01 What can a network do with text?  (data-lit "")
  F1: task cards (spam, sentiment, entities, translation, summary, question answering) one per build; F2: "generation has many valid answers" prompt with 4 continuations,
  then bars: generation is classification over the next token.
s02 Generate a name
  F1: the task; vocabulary strip with the boundary token "-"; a few real names from the list; F2: "aab -> ?" with a probability strip from the trained model (real numbers).
s03 Choose a context length
  F1: w = 1 / 3 / all cards; commit to w = 3; equation p(t_i | t_{i-3..i-1}). F2: sliding-window stepper over "- - - a a b i d -" producing the six rows (table). F3: every name yields L+1 rows; the count for the bundled list.
s04 Characters to numbers
  F1: one-hot table for a, b, i with the zero dot products (worksheet). F2: the embedding table E (27 x 2) with the two axis headers; lookup e_c = E[id]; the three selected rows highlighted. F3: shapes table (ids (3,), E[ids] 3 x 2).
s05 Concatenate the embeddings
  F1: a_0 = [e_1, e_2, e_3] as a table with position-coloured segments; F2: order test: a b i vs i b a (concatenate differs, sum is equal).
s06 Pass the vector through an MLP
  F1: AT.netSketch with 6 inputs, 5 drawn hidden nodes (label "32 hidden"), 27 outputs with the target lit; equations a_1 = sigma(a_0 W_1 + b_1), z = a_1 W_2 + b_2; shapes table.
  F2: worksheet for one hidden unit (a_0 dot W_1 column j + b) and one logit (a_1 dot W_2 column "i" + b), real numbers.
s07 27 scores to probabilities
  F1: logits table (top 5 + other) -> softmax worksheet (exp, sum, divide) -> bars with the target highlighted.
s08 The loss
  F1: cross-entropy -log p(target) live; the comparison p = 0.80 -> 0.223 vs p = 0.01 -> 4.605; F2: pick any of the six rows and read its loss.
s09 Learn the embeddings and the weights
  F1: what the optimizer changes (table of parameters and shapes); F2: training curve (loss vs steps, from toy1.json); F3: before/after bars for aab -> i.
s10 Sample, append, repeat   (THE HIGHLIGHT)
  F1: the loop as three boxes; F2: generator: buttons "Next character" / "Generate a name", greedy vs sample, temperature slider; the window slides visibly, bars show each step; F3: 20 sampled names.
s11 Training and generation
  F1: two-column comparison (known target, loss, update, many windows vs unknown target, choose, append, loop).
s12 What should a token be?
  F1: character / word / subword cards; F2: tokenizer widget (type text; see the three splits with |V| and T); F3: trade-off table; the <UNK> demo; note that attention cost will depend on T.
s13 Same model, word tokens
  F1: "a a b -> i" beside "The cat sat on -> the": identical pipeline; F2: "The cat sat on the ___" bars (illustrative, labelled); F3: chain-rule product for "deep learning is fun" as a stepper.
s14 The context keeps growing   (hand-over)
  F1: AT.netSketch with the window slider w = 3, 5, 10, 100 and the live W_1 shape and parameter count (d = 2 here; also show d = 256, d_h = 1024 giving about 26 million for w = 100); claim "change the window, and we must change the network".
  F2: the question "How should we represent arbitrarily long context?" and the link to Part 2.
s15 Pause and think (six reveals: why not one-hot; why concatenate; why 27 outputs; what changes with w; why the loss is -log p; what generation needs from training).
s16 Three summaries (intuitive, operational, mathematical) + "Next: Part 2".

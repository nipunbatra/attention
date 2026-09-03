# FRAMES.md — frame plan for Part 2 (attention.html) in presentation mode

Notation: F = frame, b = build. "held" = the drawing that stays while builds add one mark. Notes = presenter notes (first line = question before reveal).

## s01 Predict the next token
F1 "One blank": sentence chips, context switch, then the claim that the context matters. F2 show the resulting probability bars for the selected context. F3 state the conditional next-token task. F4 distinguish the learned vocabulary table E_tok from one looked-up row. F5 "Where a token starts": select a token and inspect its three aligned rows, token + position = e^{(0)}. F6 assemble E by stacking the current rows of this sentence. F7 explain same-width position addition, and explicitly disclose that this hand-designed toy ignores its dedicated position coordinate.
  Notes: "What could come next, and why do you think so?" Collect answers before switching context. The selected context and token survive continuation frames. Do not introduce attention yet.

## s02 Only the last token
F1 isolate the last token and its input row. F2 switch between contexts and observe identical probability bars. F3 connect that row to the vocabulary head and softmax. F4 compare the two identical last-token rows side by side. The full head-arithmetic worksheet remains in reading mode.
  Notes: "Before I switch the context: will the bars move?" They cannot: the baseline receives the same token at the same position.

## s03 A fixed window
F1 "Grow the window": interactive network sketch, then builds expand to five and one hundred inputs; dimensions and parameter counts update. F2 select the available window in the sentence. F3 concatenate its rows (long classroom tables explicitly summarize middle rows; the article retains every row). F4 show the enlarged prediction-head input and parameter count. F5 identify the hard context boundary. F6 distinguish fixed input slots from content-dependent MLP computation. F7 contrast useful clues in two contexts. F8 compare window-size costs and ask how to collect useful information.
  Notes: "What changes in the network if we use five tokens?" and "Can a token outside the window influence this prediction?"

## s04 Weighted pooling
F1 select the prefix and define its mean. F2 inspect the equal-weight table and reveal the pooled row. F3 replace equal contributions with chosen weights. F4 adjust presets or individual weights while a live pooled row stays visible. F5 inspect the weighted calculation and reveal its sum. F6 ask where useful weights should come from. Expanded arithmetic and coordinate explanations remain in reading mode.
  Notes: "Why not just average?" then "If you could set the weights by hand for bank, what would you choose and why?" The weights persist into the calculation frame.

## s05 Search detour
F1 choose a search query and separate matching from returning content. F2 view six ranked video results. F3 translate the request into a query vector with named toy features. F4 compare the same query with all six keys, revealing one score per build. F5 introduce the three jobs incrementally: query requests, key matches, value carries. F6 inspect the distinct value-feature table. F7 retrieve the current winner's actual payload and value vector. The selected query persists through all continuation frames; verbose card annotations remain in reading mode.
  Notes: "What does the site compare, and what does it hand back?" and "Which table decides the winner; which table supplies the result?"

## s06 Hard to soft
F1 explain hard retrieval of one winning record. F2 inspect the six match scores without repeating the keys. F3 define exponentiation and division by the common total. F4 compare hard/soft weights in an aligned three-decimal table. F5 reveal weights, weighted value rows, and their sum in three builds. F6 interpret the query-specific returned information. F7 change one score and observe the weight bars immediately. F8 change temperature and compare concentration. Full result cards and arithmetic worksheets remain in reading mode.
  Notes: "If Gradient Descent's score rises to 4.8 too, what should come back?" These are weights over available records, not next-token probabilities. Numeric tables identify the current temperature when it differs from one.

## s07 Tokens as records
F1 receiver/query/key/value jobs. F2 choose the receiving input row. F3 calculate q_bank. F4 inspect one source record. F5 calculate its key. F6 calculate its value. F7 compare one query with every key. F8 keep each key paired with its value. F9 shared W_Q/W_K/W_V and compact PyTorch.
  Notes: ask “Which row decides relevance, and which row carries content?” before F4. None of q, k, v is the token's new representation.

## s08 Two phases
F1 set up routing versus carrying. F2 show the seven query-key scores. F3 normalize those scores. F4 generalize one score. F5 generalize softmax. F6 see one query fan out to every key. F7 inspect a selected dot product. F8 pair weights with values. F9 add weighted values coordinate by coordinate. F10 intervene on values only. F11 read the changed message. In article mode, the F2/F3 tables combine into the original interactive phase-A table.
  Notes: "Has any representation changed yet?" (no) ... "What would change if only W_V changed?"

## s09 The update
F1 explain W_O. F2 show where Delta e rejoins e. F3 two-row end-to-end path (“choose where” then “carry information”) with staged highlights. F4 numerical message/update/residual calculation. F5 introduce W_O coordinates. F6 calculate W_O row by row. F7 distinguish message, update, and contextualized row.
  Notes: "If the message were zero, what reaches the predictor?"

## s10 bank
F1 compare the two bank sentences. F2 compare incoming source information. F3 calculate the river-context update. F4 calculate the cheque-context update. F5 place both updates on the same starting bank embedding and plot the result.
  Notes: "Same starting row: must the contextual row be the same?"

## s11 What are Q, K, V
F1 reject the three-table misconception. F2 one input row, three jobs. F3 show the learned projection matrices. F4 separate stored parameters from per-pass intermediates. F5 follow one row’s lifetime through a layer. F6 reconnect the branch to the residual update.

## s12 Scaling
F1 explain why larger dot-product dimension spreads scores. F2 hold raw scores fixed while changing the divisor. F3 verify the variance claim with a simulation. F4 compare bank weights with and without scaling. F5 interpret what scaling changed and what it did not.
  Notes: "Which of the two bar charts would you trust for d_k = 64?"

## s13 Causal mask
F1 derive the prefix-only rule from next-token prediction. F2 shift one sentence to show each target. F3 write the causal triangle and mask matrix. F4 apply -infinity before softmax. F5 toggle the full 10×10 attention matrix. F6 inspect token 5’s received information with the mask on/off. F7 state why the mask is non-negotiable.
  Notes: "When predicting x_{i+1}, what is the worst thing token i could read?"

## s14 To the probabilities
F1 receiver handoff: bank/q7 earlier versus final the/q10 now. F2 dynamic alpha row + context switch/bypass. F3 match → scale → source-softmax with two-line PyTorch. F4 residual table. F5 message → W_O → add, with shapes and two-line PyTorch. F6 vocabulary table. F7 attention-softmax versus vocabulary-softmax + head code. F8 one candidate logit. F9 exponentiate and normalize. F10 causal chain. F11 same-layer versus next-layer boundary.
  Notes: ask "Which known position supplies the query now?", "Are these weights over sources or probabilities over words?", and finally "Does the final 'the' read the updated bank row?" (not within this layer).

## s15 Walkthrough
F1: the compact classroom renderer replaces its result in place across the stepper's 18 builds; the full worksheets remain in reading mode. The presentation's own Next control advances the stepper, so the duplicate local toolbar is hidden. Notes: ask before each reveal what quantity and shape should come next.

## s16 Matrix form
F1 incremental contextualized attention diagram. F2 matrix-shape table. F3 operation stepper (E, Q/K/V, S, A, H, DeltaE, E') with only the current result in the classroom view. F4 routing equations. F5 message/project/add equations. F6 batching-is-not-a-new-idea callout. F7 one-head boundary + two-line PyTorch.
  Notes: "Which row is the one we did by hand?" and "Which matrix stores routing weights?"

## s17 Alternatives
F1 framing question + context switch. F2 four numerical weighting rows. F3 fixed window versus mean. F4 fixed positional weights versus attention. F5 boxed conclusion. The context control still drives the numerical table on F2.

## s18 Pause and think
F1–F8: one misconception question per frame and one reveal each. Presentation answers are compact; reading mode keeps every original numerical worksheet and arithmetic control. F9 three-space notation recap: match (q/k), send (v/m), model/update (e/Delta e/e').

## s19 Summaries
F1 intuitive sentence. F2 operational chain + synchronized motif. F3 routing matrix equations. F4 message/update matrix equations. F5 updated last row → head. F6 vocabulary table and bars. F7 explicit generation loop: predict → choose → append → new query, plus two-line PyTorch. F8 generation/training boundary: forward → loss → autograd/optimizer. F9 distinguish content routing from word-order sensitivity: this toy demonstrates only the former. F10 recap + Part 3 pointer.

Suggested pacing before discussion (about 75 min): s01-s04 12 min, s05-s06 10, s07-s09 15, s10-s13 12, s14-s15 15, s16 4, s17-s19 6. Allow longer for every worksheet or split the lesson across two meetings. Short-on-time: use either the full walkthrough in s15 or the matrix recap in s16, and leave s17 and detailed worksheets for self-study. Multi-head attention and stacked layers are Part 3.

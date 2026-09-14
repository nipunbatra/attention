# FRAMES.md — frame plan for Part 2 (attention.html) in presentation mode

Notation: F = frame, b = build. "held" = the drawing that stays while builds add one mark. Notes = presenter notes (first line = question before reveal).

## s01 Predict the next token
F1 "One blank": sentence chips and the prediction task. F2 show probability bars. F3 distinguish E_tok from a looked-up row. F4 explain the four illustrative word features with actual river/bank/the rows and matching colour labels. F5 motivate order with Maya/Ravi and bridge from Part I's concatenation. F6 add actual word/position rows at positions 1, 5, and 10, without an extra axis. F7 assemble the ten-by-four sentence matrix and define T and d_model. Keep technical position details in the article companion, not repeated recap slides.
  Notes: "What could come next, and why do you think so?" Collect answers before switching context. The selected context and token survive continuation frames. Do not introduce attention yet.

## s02 Only the last token
F1 state the context problem using the two full sentences. F2 isolate the last token as a deliberately limited baseline. F3 switch contexts and observe identical bars. F4 shows the actual 4→8→20 predictor: W1 and b1, ReLU hidden activations, W2 and b2, with matching input/hidden/parameter/logit colours. F5 turns all twenty scores into probabilities. F6 explains why identical last-token rows give identical predictions and motivates reading several rows. Full hidden-to-output arithmetic remains in reading mode. Baseline, pooling and post-attention examples all use this same hand-designed MLP; it is distinct from a Transformer block's FFN.
  Notes: "Before I switch the context: will the bars move?" They cannot: the baseline receives the same token at the same position.

## s03 A fixed window
F1 "Grow the window": interactive network sketch, then builds expand to five and one hundred inputs; dimensions and parameter counts update. F2 select the available window in the sentence. F3 concatenate its rows (long classroom tables explicitly summarize middle rows; the article retains every row). F4 show the enlarged prediction-head input and parameter count. F5 identify the hard context boundary. F6 distinguish fixed input slots from content-dependent MLP computation. F7 contrast useful clues in two contexts. F8 compare window-size costs and ask how to collect useful information.
  Notes: "What changes in the network if we use five tokens?" and "Can a token outside the window influence this prediction?"

## s04 Weighted pooling
F1 select the prefix and define its mean. F2 inspect the equal-weight table and reveal the pooled row. F3 replace equal contributions with chosen weights. F4 adjust presets or individual weights while a live pooled row stays visible. F5 inspect the weighted calculation and reveal its sum. F6 ask where useful weights should come from. Expanded arithmetic and coordinate explanations remain in reading mode.
  Notes: "Why not just average?" then "If you could set the weights by hand for bank, what would you choose and why?" The weights persist into the calculation frame.

## s05 Search detour
F1–F3 introduce attention and mark the retrieval detour. F4–F7 establish a query, define its four broad axes, show one numerical row and compare four questions. F8–F11 establish keys through a source description, the first key row, selectable source examples and the fixed collection below the changing query. F12 computes match scores. F13 ranks the six videos. F14 distinguishes matching topics from returned explanations. F15 defines eight specific value features with Backpropagation's numbers. F16 compares all six fixed eight-number value rows. F17 expands the matched video's content in plain language, then reveals its exact eight-coordinate row and the whole-row return. Teal ideas in the explanation match bold named coordinates; the four query choices cover Backpropagation, Regularisation, CNNs and Gradient descent. F18 recaps the three jobs with the actual question, winning video's key and paired value: exact 1×4 / 1×4 / 1×8 rows on labelled SVG charts, revealed Q then K then V. Four question controls update all three columns and the winning score without changing the stored source rows. The detail belongs on F17 and the compact comparison on F18. The selected query persists through continuation frames; detailed card worksheets remain in reading mode.
  Notes: "What does the site compare, and what does it hand back?" and "Which table decides the winner; which table supplies the result?"

## s06 Hard to soft
F1 explain hard retrieval with matching query/key/value colours in prose and math, label the match score and define the winning video index j*. F2 connects the six scores to the same query/key dot products and expands Backpropagation's 4.8 with matching query/key/score colours. F3 define exponentiation and division by the common total. F4 compare hard/soft weights in an aligned three-decimal table. F5 starts with the same eight original value coordinates, then reveals weights, weighted rows and their eight-coordinate sum. F6 interprets that 1×8 result on the same content axes. F7 changes temperature. The original query–key scores, eight source values and their order remain fixed throughout. Full result cards and arithmetic worksheets remain in reading mode.
  F5 header labels: feature names alone for original values, then alpha_j × feature name when the weighted cells appear. The multiplier stays for the column sums and reverses with backward navigation. Reading mode shows weighted headers with weighted cells.
  Notes: "If Gradient Descent's score rises to 4.8 too, what should come back?" These are weights over available records, not next-token probabilities. Numeric tables identify the current temperature when it differs from one.

## s07 Tokens as records
F1 is a section break returning to the original two next-word prediction prefixes, with river and cheque highlighted. F2 is a single recap of last-token/window/pooling attempts and the match → weight → mix operation. F3 starts with bank's English need and its supplied query [1.26, 1.26, 0.04], on named water?/finance?/who? axes. F4 supplies seven source keys, highlighting river. F5 supplies the paired two-coordinate values. F6 lets students inspect each source key/value record while bank remains the receiver.
  Notes: use these canonical example numbers without deriving them yet. Distinguish the known input bank at position 7 from the final the at position 10 used to predict word 11. The same values feed Section 8; no random resampling or model change. Q/K/V projection arithmetic moves to Section 11, after attention, residual addition and prediction have a purpose. None of q, k, v is the token's new representation.

## s08 Two phases
F1 sets up routing versus carrying with a colour-matched two-path diagram. Build 1 shows q7 and seven keys producing scaled scores and softmax weights. Build 2 carries the same seven weights down to the paired value products, then adds them to m7. Build 3 keeps the original takeaway. F2 shows the seven query-key scores, briefly defines d_k = 3 coordinates per query/key (not tokens), and identifies division by sqrt(3) as scale control before softmax; defer why the square root to Section 12. F3 normalizes those scores. F4 generalizes one score. F5 generalizes softmax. F6 shows one query fanning out to every key. F7 inspects a selected dot product. F8 pairs weights with values. F9 adds weighted values coordinate by coordinate. F10 intervenes on values only. F11 reads the changed message. In article mode, the F2/F3 tables combine into the original interactive phase-A table with the same first-use scaling note.
  Notes: "Has any representation changed yet?" (no) ... "What would change if we changed only the supplied values?" Hold Q/K fixed and zero the finance value coordinate. The intervention ends at the changed message; projection provenance and the output-space update are not prerequisites.

## s09 The update
F1 explain W_O. F2 show where Delta e rejoins e. F3 two-row end-to-end path (“choose where” then “carry information”) using the supplied query, with staged highlights. F4 numerical message/update/residual calculation. F5 explains W_O as a learned mapping from the two-coordinate message to a four-coordinate update, showing only the 1×2 · 2×4 = 1×4 shapes. The hand-chosen matrix and its coordinate arithmetic are in a collapsed optional reading detail, not a separate slide. F6 distinguish message, update, and contextualized row. F7 extends to the ten-token prefix and shows e_10 + Delta e_10 = e'_10. F8 passes that updated row to the familiar vocabulary head and displays four leading probabilities. River/cheque controls are synchronized across F7/F8. The same model and head are used throughout; Section 14 later opens up the head arithmetic.
  Notes: "If the message were zero, what reaches the predictor?"

## s10 bank
F1 compare the two bank sentences. F2 compare incoming source information. F3 calculate the river-context update. F4 calculate the cheque-context update. F5 place both updates on the same starting bank embedding and plot the result.
  Notes: "Same starting row: must the contextual row be the same?"

## s11 Where do queries, keys and values come from?
F1 introduces the three shared learned projections and distinguishes their parameters from computed rows. F2 recovers the supplied query with one worked calculation and a token selector. F3 reconnects the branch to the residual update with the complete small diagram. Key/value worksheets, all-matrix inspection and the lifetime walkthrough remain in reading mode.

## s12 Scaling
F1 pays off the earlier sqrt(3) signpost with the variance argument and its independence assumptions. F2 compares bank's actual weights with/without scaling and explains the preserved ranking but changed concentration. The fixed-score experiment, simulation and extended interpretation stay in reading mode.
  Notes: "Did the winning key change? What changed after softmax?"

## s13 Causal mask
F1 derives the prefix-only rule. F2 shows the causal triangle and mask equation. F3 toggles the full 10×10 matrix. F4 inspects token 5's received message, retaining the chosen mask state across the transition. The shifted-target staircase, extra code and repeated justification stay in reading mode.
  Notes: "When predicting x_{i+1}, what is the worst thing token i could read?"

## s14 To the probabilities
F1 shows the final-position-10 attention weights with context switch/bypass. F2 combines the head formula, actual vocabulary probabilities and the distinction between source and vocabulary softmax. F3 preserves the same-layer versus next-layer boundary. Repeated route/residual/head derivations and the causal chain stay in reading mode; the full diagram follows in Section 16.
  Notes: ask "Which known position supplies the query now?", "Are these weights over sources or probabilities over words?", and finally "Does the final 'the' read the updated bank row?" (not within this layer).

## s15 Walkthrough
F1 is an explicitly optional numerical inspector. Its local Next step / Previous / Reset controls inspect all 18 operations. Presentation Next skips directly to Section 16; the manual stepper does not add 18 mandatory presentation states. The classroom view keeps the operation, result and interpretation, without duplicate status/code; reading mode retains the full worksheets.

## s16 Matrix form
F1 preserves the full incremental flowchart, including scaling, causal mask, value branch, output projection, residual and final-token handoff. F2 collects the same computation in one four-equation matrix grid, with mask, row-wise normalization and output width explained. Matrix-shape/operation worksheets and duplicate summaries remain in reading mode.
  Notes: "Which row is the one we did by hand?" and "Which matrix stores routing weights?"

## s17 Alternatives
F1 combines the context controls, sentence and four weighting rows. Changing the words changes only attention's mixing weights; the fixed-window row denotes inclusion, not normalized weights. Extended comparisons and conclusion stay in reading mode.

## s18 Pause and think
F1 is one checkpoint on matching versus information sent. The other seven questions and notation recap remain available in reading mode, including their numerical worksheets and reveal controls.

## s19 Generation and training
F1 generation: predict → choose → append → new query, with fixed parameters. F2 training: forward → loss → autograd/optimizer. F3 closes with the single-head scope and Part 3's multiple heads, feed-forward layers, normalization and stacked blocks. Repeated intuitive, operational, matrix and vocabulary summaries stay in reading mode.

The Section 11–19 classroom tail is 20 frames, down from 63; the full flowchart still builds progressively. Suggested pacing for this tail: projections 4 min, scaling 3, masking 5, prediction/layer boundary 3, flowchart/matrix form 5, comparison/checkpoint/close 5. Skip the optional Section 15 inspector on a first pass. These are planning estimates, not a timed run. Detailed worksheets remain in reading mode; multi-head attention and stacked blocks are Part 3.

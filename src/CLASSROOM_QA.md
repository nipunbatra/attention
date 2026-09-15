# Classroom release checks

## 2026-09-15: plain-language explanation of hard retrieval

- Kept the hard-retrieval equation and index explanation, then added an always-visible plain-language reading and the existing Backpropagation example below them. It explicitly returns all eight value coordinates from the winning video and no contribution from the other five. The later question about multiple useful items retains its original reveal.
- Humanizer guided the direct wording; the presentation skill guided the existing typography and value colour. Visually inspected both reveal states and 390px reading layout. Narrow checks verify the explanation follows the equation, remains visible at both builds and preserves forward navigation. The full 227-state / 513-formula audit passes without overflow or math errors. No new frame, model, numerical-data or dependency changes.

## 2026-09-15: concrete keys, values and separate query/key mappings

- Rebuilt the video-to-value transition around the selected result's existing diagram thumbnail and a labelled illustrative transcript. All four query winners have their own excerpt; the source, thumbnail and transcript stay paired when the request changes. These are authored examples, not transcripts of real videos. No separate generated video photo was present in the repository or the supplied task asset folder.
- Added three classroom frames: a contact-name/phone-extension lookup, a pronoun receiving identity/number information, and a later two-feature projection example. The first two precede the value axes; the projection example stays after the complete token-to-prediction flow. The latter demonstrates equal query/key widths with different mappings and independently checked directional raw scores of 1 and 0. Notes distinguish the symmetric raw-score restriction of tied projections from masking and row-softmax behavior. The value lessons explicitly allow shared widths and overlapping information: the distinction is each representation's use, not a requirement that its entries differ.
- The humanizer skill guided concrete explanations and removal of the duplicated video-summary sentence. The presentation skill guided reuse of the existing design, editable tables, role colours, progressive reveals and visual checks. Inspected the video, lookup, pronoun and projection frames at desktop and 390px reading widths. All 227 states / 513 unique formulas pass the full frame audit; phone QA has no overflow, page errors or math errors. The new role regression checks reveals, source pairing, independent projection arithmetic, navigation and model immutability. Existing search and projection checks retain the source values, dot products and forward calculations. Eight live/reference model cases still match (9,260 finite values; maximum error 1.78e-15).
- The table audit's four existing narrow-column warnings remain outside the changed frames. One additional heuristic flag marks the short “Source” column in the phone pronoun table; visually checked that its label and both source names fit without clipping. No model, shared runtime, dependency, Lecture 3 or PDF changes.

## 2026-09-14: carry Lecture 2's model and teaching sequence into Lecture 3

- Part III now starts from the exact Part II model, including its 4→8→20 ReLU prediction MLP. This supersedes the earlier separate-linear-readout adapter. Both predictor matrices and biases participate in the forward pass, backward pass, SGD updates, parameter-shape tables, and numerical worksheets. Saved training values retain full precision; only displayed text is rounded. The larger Transformer schematic explicitly distinguishes its 4→8→4 block FFN and final linear vocabulary readout from the toy predictor.
- Applied the humanizer skill in this pass to make Lecture 3's explanations and transitions more direct, and to clarify Lecture 2's handoff. The presentation skill guided the existing theme, matched prose/math colours, editable diagrams, and example-first sequence. Added learning, block-building, and generation dividers. Reduced the classroom route from 70 frames to 51, retaining detailed arithmetic and repeated checks in reading mode. Preserved the complete forward/backward graphs, causal mask, and decoder flowchart; showed both MLP layers and actual residual bypasses. Backward arrows and labels match the red gradient equations.
- Numerical verification reproduces all 2,112 saved training entries. Central differences check 324 smooth coordinates of the canonical model; eight exact-zero ReLU directions are reported separately. A separate smooth probe checks all 332 gradients for both single-target and mean-loss objectives (maximum error 9.77e-11). Forty unused position perturbations leave the loss unchanged. The new continuity regression independently checks 1,578 numbers, all SGD presets, hidden activations, head/mask controls, LayerNorm sliders, cache/cost controls, diagrams, colours, reading companions, model immutability, and settled phone layout (maximum numerical error 2.67e-15).
- Inspected the revised desktop graphs, dividers, FFN, residual paths, decoder flow and closing slide, plus 390px reading layouts. Part III passes all 105 presentation states / 156 formulas and its table audit. Part II still passes 218 states / 508 formulas and eight independent model cases. Both phone articles have no horizontal overflow, math errors, or page errors. The 48-case capacity/generation check and eight-config metadata check pass. Fixed the metadata test fixture's missing existing Part I image; no dependency was added.

## 2026-09-14: bridge from the missed clue to a fixed-width summary

- Added exactly one classroom frame after “Ten tokens can miss the earlier clue” and before the summarizing section divider. Its three states recover river/cheque with twenty tokens, compare the wider concatenated input and first MLP matrix, then ask how a fixed-width summary could collect useful history. Each token still has four coordinates; the hidden width stays eight and W2 is unchanged. Presenter notes distinguish input visibility from prediction quality and fixed summary width from free or unlimited context.
- The presentation skill guided the existing clue/input/parameter colours, editable comparison table and progressive reveals. Inspected the final desktop and 390px reading layouts. Full-deck audit passes all 218 presentation states and 504 unique formulas, with no overflow or math failures; phone QA reports no page/console errors or horizontal overflow. Eight independent live/reference cases still agree (9,260 finite values; max error 1.78e-15).
- Updated the window and pooling flow regressions for the new transition, dimensions, reveal order, matched colours and phone table bounds. No model, runtime, dependency, Part I or Part III changes. The existing derivations remain in reading mode; this slide motivates the next section rather than repeating the parameter-count lesson.

## 2026-09-14: use the Part I-style hidden-layer predictor

- Section 2 now teaches an actual 4→8→20 ReLU MLP, not a direct linear 4→20 head. The native SVG shows all eight hidden activations and both matrices/biases; formulas and prose share input-blue, hidden-teal, parameter-purple and logit-grey roles. The presentation skill guided the editable diagram, compact definitions and phone equation wrapping. No classroom frames were added.
- Baseline, pooling presets, residual-to-prediction examples, numerical worksheets and the full flowchart all use the same saved two-layer predictor. Updated Python generation, independent JS reference, live runtime, numerical report and standalone SVG evidence. The predictor is explicitly hand-designed, not trained. The identical last-token baseline remains an eight-way tie; after attention, water and teller remain the respective top choices. Attention projections, scores, weights, messages and residual updates are unchanged.
- Added seven independent MLP probes, including negative pre-activations and a threshold-crossing check that would fail for an affine-only substitute. Verified 7,265 Python/reference numbers (max error 3.55e-15), eight live/reference cases including hidden rows (9,260 finite values, max error 1.78e-15), and all 215 classroom states without overflow or math errors. Inspected the revised slide and stacked phone equations; phone layout, matched colours, displayed edges/activations and worked arithmetic are covered by regression tests. Standalone preview passes all twelve SVG stages.
- Part III intentionally retains its existing separate linear-readout training experiment and full-Transformer architecture sketch. Its rebuild adapter now selects that architecture explicitly; all 1,386 saved training numbers and 212 gradient checks still agree. The 48-case capacity/generation check compares each assembled part with its own source model and passes. No Part I or Part III HTML/model artifacts were changed.

## 2026-09-14: concise classroom sequence from Section 11 onward

- Reduced Sections 11–19 from 63 classroom frames to 20. Kept the shared-projection introduction and one query calculation, the promised scaling explanation and actual comparison, all essential causal-mask demonstrations, the vocabulary/layer-boundary result, the complete 12-stage flowchart, one matrix summary, one alternatives table, one checkpoint, and the generation/training close. The presentation skill guided consolidation around distinct teaching steps rather than repeated derivations. Sections 1–10, Part I, canonical model data and shared runtime are unchanged.
- Moved repeated projection, lifetime, matrix, head-arithmetic and recap frames to reading-mode companions without removing their numerical hosts or controls. The 18-step numerical walkthrough is optional: its local toolbar inspects operations while presentation Next continues directly to the full flowchart. Its classroom view drops duplicate status/code. The comparison now includes its own context controls and numbered sentence; compact column positions refer to those chips while retaining full accessible labels.
- Preserved the chosen mask state when moving from the matrix to the close-up message, using the existing local state-retention attribute. The new check_concise_tail.mjs verifies the 20-frame order, companion visibility, forward/back navigation around the optional inspector, all 18 manual stages, all 12 flowchart stages, exact masked/unmasked tables and messages against the independent reference, context-dependent comparison weights, model immutability and phone reading.
- Visually inspected the merged slides, both flow diagrams, optional worksheet and 390px reading layouts. Full-deck audit passes 215 presentation states / 491 unique formulas with no overflow or math failures. Existing token-flow and routing/scaling checks, phone QA, and all eight live/reference model cases pass (7,980 finite values; maximum difference 8.88e-16). Detailed worksheets remain available rather than being deleted.

## 2026-09-14: explain the output projection as a learned mapping

- Replaced Section 9's zero-heavy output matrix slide with a concise explanation of the learned mapping into embedding space, using the 1×2 · 2×4 = 1×4 shapes and matching message/update/embedding colours. Moved the fixed matrix and its following arithmetic frame into one collapsed optional reading detail. It explicitly identifies the hand-chosen matrix and its zero columns as properties of this example, not requirements of a trained projection. The classroom sequence has one fewer slide and goes directly from the mapping to residual addition. Source weights and numerical outputs are unchanged.
- The presentation skill guided the flat, editable math layout and preservation of the existing colours. Visually checked desktop, phone, the next-slide transition and expanded reading detail. A local nowrap rule keeps the W_O subscript attached above its shape label on phones. Regression checks the learned-mapping explanation and shapes, absence of matrix entries in the presentation frame, optional-detail visibility and unchanged source values. All 316 presentation states / 558 formulas, phone QA and eight independent numerical-model cases pass.

## 2026-09-14: introduce the square-root factor before deferring its derivation

- Added a brief note beside the first scaled-score table in Section 8, in both reading and presentation modes. It defines d_k as the three coordinates per query/key, distinguishes that from the token count, and identifies division by sqrt(3) as scale control before softmax. The note points to Section 12, whose existing opening now explicitly returns to this example before the variance argument. No new frame, reveal, model or dependency changes. The presentation skill guided the existing layout and semantic math colours.
- Visually inspected the first-use note, the full later explanation and the phone reading layout. Regression verifies both copies of the note, the current model dimension, the later payoff and unchanged frame counts. All 317 presentation states / 557 formulas, phone QA, routing/scaling regression and eight independent numerical-model cases pass.

## 2026-09-14: diagram for the two attention phases

- Added an editable SVG overview to the existing Section 8 opening frame. Its upper path compares bank's query with seven keys and applies softmax. A rose connector takes those same weights to the lower path, where each weight multiplies its paired value and the contributions add into the two-coordinate message. Query/key/value/weight colours match the prose and equations. Phase A and Phase B reveal with builds 1 and 2, and the original takeaway stays at build 3. No additional frame or numerical-model change.
- The presentation skill guided the existing layout, semantic colours and native editable diagram. Visually inspected all builds and the phone scroll. Regression verifies forward/back reveal states, phase ordering, colour matching, the next-frame handoff and locally contained phone scrolling. The full 317-state frame audit, 554-formula check, phone QA, routing regression and eight numerical-reference cases pass.

## 2026-09-14: use Q/K/V before deriving their projections

- Section 7 now supplies the existing bank query, seven keys and their paired values on named axes, before any Q/K/V projection equation. The source inspector keeps bank as receiver. Section 8 carries those same rows through scores, shares and messages; its value-only experiment changes V directly and defers the output-space update. Section 9 completes the message → output projection → residual path, then previews the final-token update and actual next-word probabilities for both contexts.
- Moved projection teaching to its existing Section 11 home: first ask where the supplied rows came from, then recover them from the current input rows with the shared W_Q/W_K/W_V matrices. Three coordinate-by-coordinate calculation frames replace the early derivations. The total authored frame count across Sections 7–11 is unchanged. The presentation skill guided the use-before-derivation order, existing semantic colours, editable numerical tables and visual checks.
- `check_token_flow.mjs` verifies the supplied numbers against the independent reference, all seven paired source records, absence of premature projection equations, exact position-10 residuals/probabilities in both contexts, unchanged shared-matrix calculations, synchronized selectors, model preservation and phone layout. Desktop and phone views were visually inspected. The full frame audit passes 317 presentation states and 554 unique formulas; 152 control interactions, pooling/routing regression and eight independent model cases pass. The table audit found no issues in changed sections; its four narrow-column warnings are in unchanged Sections 3, 4 and 17.

## 2026-09-14: return to the prediction problem before token projections

- Reworked the existing Section 7 transition into a dedicated section break with both original ten-token prefixes and highlighted river/cheque clues. Added one recap frame before the receiving-token equations. It connects last-token, concatenation and pooling attempts to the remaining need to compute context shares, then previews query/key matching, softmax shares and the weighted value sum. It explicitly distinguishes bank at position 7 from the final the at position 10 used later to predict word 11. No model or projection arithmetic changed.
- The presentation skill guided the problem-first break, single-slide recap, existing colour conventions and staged explanation. Visually inspected the break, both recap builds, the following equation frame and phone layouts. Regression covers exact original prefixes, prerequisite order, forward/back navigation, matching text/math colours, the receiver/source indices and the unchanged model. All 325 presentation states / 556 formulas, 144 controls, 390px article QA and eight independent model cases pass. Updated the shifted query-calculation screenshot anchor.

## 2026-09-14: remove the score-change exercise

- Removed Section 6's score-change frame and its six sliders, reset button, bar chart and exercise hints from both presentation and reading modes. Removed obsolete score-override notices and styles. The presentation skill guided the shorter sequence: the returned-vector frame now leads directly to temperature, then back to the sentence example. The original query–key scores, source values, hard/soft comparison and temperature control remain intact. The removed exercise is recoverable in Git history.
- Regression verifies the seven-frame order, next/previous navigation, soft-mode activation at the new temperature index, and the handoff to Section 7. The four remaining hard/soft/temperature cases independently check 192 weighted products and 32 returned coordinates. Visually inspected desktop transitions and the phone article. All 324 presentation states / 556 formulas, 144 controls and 390px article QA pass.

## 2026-09-14: simplify the returned-information slide

- Removed the redundant hard/soft status line, including the arbitrary 2% contribution count, from Section 6's returned-row frame. Kept the eight-coordinate result, its interpretation and the next-build takeaway. Removed the unused status calculation and scoped styles. The presentation skill guided removal of the extra text without changing the deck's layout conventions.
- Regression checks that the status stays absent and the interpretation stays visible in all six retrieval cases, alongside unchanged weighted products and returned rows. Visually inspected both desktop builds and the 390px reading layout. All 325 presentation states / 556 formulas and phone article QA pass.

## 2026-09-14: label weighted contribution columns at their reveal

- Each of the eight value headers in Section 6's multiplication table now gains a rose alpha_j × prefix when the weighted cells appear at build 2. Feature names stay teal and keep their original order. The prefix remains for the column sums, disappears when returning to original values, and stays visible in reading mode with the weighted table. The presentation skill guided the existing palette, aligned header labels and unchanged reveal sequence. No additional slide, model, shared-runtime or dependency changes.
- Regression checks all eight prefixes against the cell reveal states, backward navigation, colour matching, feature order and column bounds, together with all existing retrieval arithmetic. A focused final-layout check verifies aligned multipliers and phone reading labels. Visually inspected the original, weighted and summed desktop states and phone scrolling. All 325 presentation states / 557 formulas, 151 controls and 390px article QA pass. Stored values, weighted products and column sums are unchanged.

## 2026-09-14: reconnect the score table to query–key matching

- Section 6's second frame now explicitly identifies the same gradient-information query and six video keys, defines each score as s_j = q dot k_j, and expands Backpropagation's four coordinate products to 4.8. The presentation skill guided reuse of the query/key colours and the existing rose score colour in both text and arithmetic. No new slide or reveal. The table, original numbers and clickable worksheets remain intact.
- The worked terms stay together when wrapping on phones. A compact notice appears if later score sliders overwrite the original dot products, including when navigating back to this frame, and disappears when the scores return to their starting values. Visually inspected the initial and edited desktop states and the phone reading layout. All 325 presentation states / 556 formulas, 151 controls and 390px article QA pass.
- Retrieval regression verifies the transition from the preceding frame, matching role colours, the original four query/key coordinates and their independently computed result, all six starting scores, and unchanged arithmetic across score interventions. All existing retrieval cases and the independent live/reference token-model check pass. No model, shared-runtime, dependency or Part I changes.

## 2026-09-14: match hard-retrieval prose and equation colours

- Updated Section 6's existing first frame so query, key and value names and inline symbols use the same purple, orange and teal as the display equation. Labelled the dot product as the match score and defined j* as the winning video's index. The presentation skill guided reuse of the existing colour and layout conventions. Kept the original follow-up reveal and slide count. Corrected the notes to describe this frame rather than later controls.
- Visually checked the desktop slide and 390px reading layout. The retrieval regression verifies matching text/math colours, symbol explanations, repeated reveal states and phone bounds, alongside all existing query/key/value calculations. All 325 presentation states / 550 formulas, 151 controls, phone QA and eight independent model cases pass. No model, shared-runtime, dependency or Part I changes.

## 2026-09-14: expand what the matched source returns

- Added one focused explanation before the three-role recap. Each of the four query choices now shows what its winning video teaches in three short explanations. Backpropagation unpacks the loss signal and chain rule, then distinguishes gradient computation from an optimiser's update. The other choices explain penalties/dropout, shared filters, and gradient steps/step size. Existing summaries and all source vectors remain unchanged.
- The presentation skill guided the content-first reveal, flat explanation/chart layout and matching teal emphasis. A second stage shows the same eight named value coordinates, with the corresponding content ideas in bold. The final stage explicitly returns every entry of the selected source's row, including small and zero values. Notes and reading text distinguish the illustrative feature encoding from a transcript or a trained text encoder. The Q/K/V recap remains compact on the following frame.
- Full-deck checks pass for all 325 presentation states / 547 formulas, all 151 controls, the existing pooling regression, eight independent token-model cases (7,980 finite values, max error 8.88e-16), and 390px article QA. One additional frame. No Part I, model, shared-runtime, dependency or PDF changes.
- The focused retrieval regression checks all four expanded explanations, source/subscript pairing, exact returned vectors, linked feature names, all three repeated reveal states, synchronized query controls and the natural transition to the recap. It also passes the existing 24 score, 48 stored-value, 288 weighted-product and 48 mixture-coordinate checks. Visually inspected all four desktop examples, the reveal sequence and phone explanation/value panels. Phone panels stack and keep every coordinate visible.

## 2026-09-14: concrete query/key/value recap

- Rebuilt the existing final retrieval frame as the three-role recap, after the query, key and eight-value-feature foundations. Each column keeps its role question and adds the actual learner request or winning video, its exact vector and a small named-axis SVG chart. Four synchronized question controls contrast gradients, overfitting, CNNs and weight updates. Query/key rows remain 1×4; values remain 1×8. The key/value title and subscript identify the same source, and the displayed winning score comes from all six unchanged keys.
- The presentation skill guided the preserved three-column colour convention, native data graphics and Q → K → V reveals. Removed the old role-only companion cards rather than adding a duplicate slide. The SVGs include accessible numeric descriptions; notes and reading text identify the hand-chosen numbers, row convention and bar scales. Phone reading stacks the cards in order. No source vectors, Part I, token model, shared runtime, dependencies or PDFs changed.
- Full-deck checks pass for all 322 presentation states / 547 formulas, all 147 controls, the existing pooling regression, eight independent live/reference model cases (7,980 finite values, max error 8.88e-16), and 390px article QA. The table audit reports only the four pre-existing warnings outside Section 5. Visually inspected all four desktop recaps and the query/key/value reveals.
- Expanded the retrieval regression to verify all four recap queries, exact SVG coordinates and axis order, common query/key bar scales, source pairing and score arithmetic, synchronized controls, repeated forward/back reveals, text/bar bounds and phone card stacking. All pass, together with the existing 24 score, 48 source-value, 288 weighted-product and 48 mixture-coordinate checks. Visually inspected the phone gradient and overfitting recaps.

## 2026-09-14: eight specific value features throughout retrieval

- Replaced the three broad value topics, which resembled the query/key topics, with eight specific content features: chain rule, gradient step, step size, shared filters, token mixing, activation scaling, weight penalty and dropout. A new introductory frame defines all eight and shows Backpropagation's actual entries before the six-video comparison. Strongest source features are underlined. The guide explicitly labels these zero-to-one strengths as hand-chosen and not probabilities or literal learned-coordinate names.
- All six videos now have fixed eight-number values. The matched source's row, result-card details, hard return, weighted contribution worksheet, final mixture and arithmetic expansion reuse all eight in the same order. Four-dimensional queries/keys, all match scores and controls remain unchanged. Regularisation's explanation now includes dropout as well as penalties. Section 7 explicitly distinguishes this video example from the original three-matching/two-value-coordinate sentence model, whose parameters and outputs are unchanged.
- The presentation skill guided the example-first axis definitions, native editable tables and matched key/value colours. Preserved the theme and progressive reveals. The original/product reveal removes hidden products from layout to avoid a doubled-width intermediate state. The activation-scaling header has extra room. Desktop tables show every coordinate and phone tables scroll locally. Visually inspected the axis guide, original values, hard/soft mixtures and phone layouts.
- Expanded the retrieval regression to independently check all 48 source values, 24 query-key scores, 288 weighted products and 48 mixture coordinates across hard, soft, tied-score, sharp/broad-temperature and alternate-source cases. It checks source/value pairing, unchanged stored rows, axis order, repeated reveals, column text bounds, phone reachability and unchanged model state. All 321 presentation states / 545 formulas, 143 controls, the existing pooling regression, eight independent token-model cases (7,980 finite values, max error 8.88e-16) and 390px article QA pass. No Part I, shared-runtime, dependency or PDF changes.

## 2026-09-14: establish queries before four key lessons

- Rebuilt the retrieval introduction around four query-first frames: a learner's request, four explicitly defined topic axes, one actual query row, and four contrasting questions with their vectors. These frames do not introduce keys or values. All rows use gradient flow, optimisation, architecture and generalisation in the same order; the strongest coordinate is underlined in purple. The slides explain the 1×4 shape, small versus zero signals, and that the hand-chosen coordinates are neither probabilities nor literal named axes in a trained model.
- Followed with four key lessons: what one source covers, Backpropagation on the same axes, three selectable source examples, and a query above all six fixed source keys. Orange source rows remain unchanged when the purple query changes. The first two query vectors and every existing key/value are unchanged; added architecture and weight-update requests supply two more contrasts. Dot products and ranked results now follow these foundations. The concrete matching-topic/returned-explanation comparison moves after matching, avoiding premature value terminology and another repeated explanation.
- The presentation skill guided example-first sequencing, native aligned tables, matching colours and progressive reveals. The four key slides define the source subscript and distinguish it from the coordinate index. Source selection is independent of query selection. Visually inspected the desktop sequence, all named-axis rows and settled mathematical subscripts, plus phone reading layouts and local table scrolling. Six net additional presentation frames; no Part I, token model, shared-runtime, dependency or PDF changes.
- Added a focused retrieval regression for prerequisite order, all four exact queries, six original keys, fixed-key invariance, all 24 independently recomputed match scores, winning sources, returned explanations, source controls, repeated reveals and navigation/reset behaviour. Updated the existing pooling/search regression for the moved concrete example. All 319 presentation states / 544 formulas, 143 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass. The table audit reports only the four existing unrelated narrow-prose warnings.

## 2026-09-14: make the search-box distinction concrete

- Replaced the abstract two-jobs callout on the existing search frame with a paired example: a named video's matching topic, followed by the explanation it returns. The gradients question shows Backpropagation and its chain-rule explanation; the overfitting question shows Regularisation and its explanation about penalising complexity. Both update from the existing winning source and summary, while the plain-language topic follows its strongest key feature. No numerical search data changed.
- The presentation skill guided the concrete example, flat side-by-side layout and matching key/value colours. The existing first reveal displays both parts together; the phone article stacks them in reading order. Teaching notes connect the topic description to the later illustrative key vector and distinguish the example from a live video search. No additional slide or repetition of the scoring arithmetic was added.
- Regression checks both queries at repeated forward/back reveal states, exact returned summaries, consistency with the later results, reset behaviour, colours and phone bounds. Visually inspected both desktop examples and phone layouts. All 309 presentation states / 542 formulas, 132 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass. Model parameters, Part I, shared runtime, dependencies and PDFs are unchanged.

## 2026-09-14: mark the retrieval detour and the return to tokens

- Turned the existing information-retrieval introduction into a prominent section break immediately before the search box. It previews query, key and value in their established colours, explains the connection to the alpha weights, and promises a return to bank and river. This replaces the brief introduction rather than adding a second detour slide.
- Added one return transition after soft retrieval, before the token-projection equations. The seven-token prefix appears first; a reveal maps bank's query q7 and river's key k6/value v6 to the search roles. The text explicitly compares against every available key before normalization and identifies alpha7,6 as river's share for bank, applied to its value. Teaching notes distinguish gathering context for the known bank token from predicting it, and the analogy from an actual external search.
- The presentation skill guided the clear topic boundaries, concrete-example-first reveal, and matching colours across role names, math and explanations. Visually inspected the detour and both return builds on desktop and the phone article, including settled mathematical subscripts. Updated the table-audit screenshot anchor for the shifted Section 7 frame.
- Regression verifies the new natural transitions, role definitions and colours, reveal order, search-query retention/reset behaviour, phone bounds and unchanged model outputs. All 309 presentation states / 542 formulas, 132 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass. The numerical examples, model, Part I, shared runtime, dependencies and PDFs are unchanged.

## 2026-09-14: original rows before weighted rows and their sum

- Reworked the existing weighted-row worksheet into three progressive stages on the same frame. Original input rows and their normalized shares appear first. The first reveal adds aligned alpha-times-input columns while leaving the originals visible. The second reveal adds m7 directly below the product columns. The four preset controls update every stage together, and all original rows remain unchanged, including when their share is zero.
- The presentation skill guided the native table, numbered column groups, matched colours, source-row alignment and sequential reveals. Kept all seven sources and four coordinates in each group, explicit rounding guidance, and the sum-cell arithmetic tooltips. Preset highlighting follows aria-pressed through navigation. The phone article scrolls the comparison locally and provides a scroll hint.
- Regression verifies original cells, normalized shares, every product and the final sum for all presets and custom weights. It checks repeated forward/back reveals, row alignment, retained state and the following prediction frame. Visually inspected the original/product/sum sequence, river and bank-only cases, and phone layout. All 307 presentation states / 541 formulas, 132 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass. No new slide, model, Part I, shared-runtime, dependency or PDF changes.

## 2026-09-14: connect hand-chosen summaries to word predictions

- The four preset controls now reveal a live prediction from the existing fixed vocabulary head after showing the pooled row m7. The weighted-row worksheet has synchronized preset controls and displays the actual alpha-times-input contributions, whose columns add into m7. It labels rounding explicitly. Custom sliders, zero-total fallback and tied highest scores remain supported.
- Added one comparison frame: four saved m7 rows first, then the top word, score and probability for each. All twenty unrounded logits enter softmax. Equal, river-focused and fisherman-focused presets rank water first, with probabilities approximately 0.223, 0.425 and 0.344; bank-only narrowly ranks teller first at 0.144. These are the unchanged hand-set head's outputs, not desirable language predictions. Slides and teaching notes explicitly identify position 8 after bank, the original continuation and, and the untrained-head limitation. This baseline feeds m7 directly to the head, without a residual addition.
- The presentation skill guided the context-before-prediction reveals, native editable comparison, and matching summary/parameter/score/probability colours. Visually inspected desktop controls, contribution sums, four-case comparison and phone article. A local horizontal scroll preserves the comparison table on narrow screens.
- Pooling regression independently recomputes every weighted contribution and all twenty logits/probabilities, tests the four presets in both control locations, custom keyboard input, zero weights, a two-word tie, state retention, reveal order, colour consistency and phone bounds. All 306 presentation states / 540 formulas, 132 controls, eight live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass. No model, Part I, shared-runtime, dependency or PDF changes.

## 2026-09-14: motivate unequal shares before alpha

- Inserted one example-led frame between the mean worksheet and the numbered hand-chosen-weight example. The seven-word prefix ends at the known input bank. Fisherman and river are bold, blue and marked as earlier clues, while bank keeps the teal receiving-token colour. The first question asks students to distinguish the riverside meaning from the financial meaning.
- Progressive reveals first explain the clues and question the mean's equal treatment of river and “the”, then propose unequal shares chosen by hand. There are no alpha symbols, numerical shares or query terminology on the motivation slide. The existing alpha example follows naturally and now continues the decision rather than reopening the motivation. Notes distinguish human-selected clues from measured weights and avoid implying that grammatical words are always unimportant.
- The presentation skill guided the example-before-notation order and simple, colour-matched text emphasis. Visually inspected all new content on desktop and phone. Regression verifies the exact prefix, highlighted/bold words, colour meanings, natural mean-to-motivation-to-alpha navigation, repeated reveals, and unchanged pooling/model arithmetic.
- All 303 presentation states / 539 formulas, the pooling regression, 128 controls, eight live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass. No JavaScript, KaTeX or overflow failures. The numerical examples, controls, model, Part I, shared runtime, dependencies and PDFs are unchanged.

## 2026-09-14: a summary section with numbered approaches

- Added a dedicated “Summarizing the prefix” divider after the fixed-window limitation. It defines prefix, states the fixed-width-summary problem, and numbers the progression: 1 averaging, 2 hand-chosen weights, 3 attention-computed weights. The same numbers appear at the existing approach introductions. The roadmap distinguishes how contributions are chosen, without suggesting that averaging and weighted pooling are unrelated operations.
- Moved the existing concatenation-versus-pooling bridge, its styles and initialization into Section 4 after the divider. Section 3 now ends at its longer-prefix counterexample. Section 4's article heading and navigation label agree. This adds only one teaching frame, preserves the existing examples, and leaves the attention explanation and retrieval detour in their established order.
- The presentation skill guided the visible topic boundary and numbered progression. Visually inspected the divider, moved bridge, numbered mean/alpha/attention introductions and phone article. The article reuses its section heading rather than repeating the divider title.
- Regression checks natural section navigation, consistent approach numbers and section labels, desktop/article title visibility, all seven prefix/mean positions, weighted presets and unchanged model outputs. Window regression passes all short/long widths and 100 MLP widths. All 300 presentation states / 540 formulas, 128 controls, eight live/reference cases (7,980 finite values, max error 8.88e-16), and 390px article QA pass with no JavaScript, KaTeX or overflow failures. No model, Part I, shared-runtime, dependency or PDF changes.

## 2026-09-14: consolidate the window and parameter-growth explanation

- Reduced Section 3 from seven teaching frames to five: available inputs, concatenation, the windowed MLP, a longer-prefix counterexample, and the pooling alternative. The two redundant linear-head frames remain as explicitly labelled optional article companions, with a definition of every count term and a worked reading of the first table row. The longer-prefix slide now focuses on lost information rather than repeating parameter arithmetic.
- Both visible network views use the same eight-hidden-unit MLP layout. The single network explanation introduces w, d, H and V before revealing the forward equations and then the four parameter groups. Matching colours connect inputs, hidden activations, weights, scores and probabilities to their definitions. Counts include W1, b1, W2 and b2 and exclude embedding/position tables. At w=20 the layer total is 640+8+160+20=828. Only W1 grows, by dH=32 weights per extra token slot.
- The presentation skill guided prerequisite order and progressive detail in one diagram instead of repeated growth slides. The chosen short window now synchronizes across token, concatenation, MLP and optional linear-head views. Larger MLP widths explicitly describe capacity beyond the ten-token example. Reveals preserve the chosen width. Stable frame IDs replace positional hooks, and retained SVG input labels update when switching between c10 and the generic ct context.
- Regression passes for the five-frame navigation, all ten short windows, all 100 MLP widths at every reveal, eleven long windows, both matrices and bias counts, token-coordinate connections, exact concatenation, optional output tracing, keyboard controls, retained state and model immutability. All 299 presentation states / 540 formulas, 128 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16) and phone layout checks pass. Visually inspected the revised network at small/large widths and all three reveals, the window boundary, and phone MLP/linear-count views. No model, Part I, shared runtime, dependency or PDF changes.

## 2026-09-14: introduce attention before the retrieval analogy

- Ended the weighted-pooling sequence with an explicit open question: how should the model calculate the source weights? Added a dedicated Attention section break, a three-step plain-language explanation (compare, normalize, mix), and a signposted information-retrieval detour before the existing search example. The explanation retains bank at position 7 and river at position 6, distinguishes input-dependent weights from learned parameters, and defers query/key/value terminology until the analogy introduces their roles.
- Updated the chapter and navigation titles together. The search reset hook now targets its stable frame ID rather than the first frame in the section, preserving the existing query interaction after inserting the three new frames. Regression checks natural navigation, progressive reveals, matching colours, absence of premature jargon, divider-only header hiding, and retained/reset search state.
- The presentation skill guided a clear topic break and concrete explanation before terminology, while preserving the deck's native HTML, colours and interactive format. Visually inspected the divider, explanation and retrieval detour on desktop, and the explanation and detour in the phone article.

## 2026-09-14: label both matrices in the growing-window network

- Labelled W1 above the input-to-hidden connections and W2 above the hidden-to-vocabulary connections. Matching purple readouts show both matrix shapes and weight counts, and explicitly state that only W1 grows with the window. At w=20, W1 has shape 80×8 and 640 weights; W2 remains 8×20 with 160 weights. These are weight-only counts, not whole-model totals.
- Regression exercises all 100 MLP window widths, checking both connection groups, labels, colour matches, matrix shapes, counts and slide bounds. Existing short-window, long-window, keyboard, reduced-motion and numerical-immutability checks still pass. Visually inspected w=1, 20 and 100; the labels and expanded readout fit without overlap.
- Combined release checks pass: all 304 presentation states / 529 unique formulas, 128 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16), pooling and window regressions, and 390px article QA. No JavaScript, KaTeX, slide-overflow or phone document-overflow errors. No model, Part I, shared runtime, dependency or PDF changes.

## 2026-09-12: explain alpha with a concrete source contribution

- Replaced the abstract weighted-pooling introduction with two frames. The first identifies receiver bank at i=7 and source river at j=6, reads alpha aloud, and works through a conditional, hand-chosen weight of 0.5. Both the input row and its half-weighted contribution come directly from the existing embedding data. The example changes no preset or model value, and explicitly leaves the other six weights a total share of 0.5.
- The second frame expands bank's seven-source sum before introducing Sigma and the general rule. Rose identifies scalar weights, blue identifies input rows, and teal identifies the four-number summary in both equations and explanatory text. The slide defines each symbol, distinguishes scalar and row shapes, states nonnegative weights summing to one for a fixed receiver, includes the current source, excludes later sources, and connects equal weights back to the mean.
- The presentation skill guided concrete arithmetic before general notation and retained the deck's colours, editable tables and progressive reveals. Regression now checks both indices and highlighted tokens, every product against independent arithmetic, colour matches, definition-column bounds, repeated hide/reveal states, actual summation pixels and phone table fit. Scoped visible/hidden rules prevent Chromium from retaining stale visibility inside nested KaTeX table spans.
- All 299 presentation states / 526 formulas, 128 controls, eight independent live/reference cases (7,980 finite values, max error 8.88e-16), the full pooling regression and 390px article QA pass. Visually inspected the worked example, expanded/general rule and phone layout. One additional teaching frame. No model, Part I, shared runtime, dependency or PDF changes.

## 2026-09-12: distinguish current position from window size

- Added one comparison frame before the mean calculation. The same current input position i drives two token strips: a fixed three-token window and all tokens so far. At bank (i=7), the window supplies positions 5–7, while full-prefix pooling supplies 1–7 and retains fisherman. At i<3 both rules show only the words available so far. Positions after i never contribute.
- Renamed the controls to “Current token position i” and synchronized comparison, mean result and worksheet state. Blue marks every contributing input; a separate outline identifies the current token. The mean slide uses the same distinction instead of highlighting only the current token. The article connects the user's k to the existing w notation and explains that averaging only the last k rows is a valid local variant that retains the window cutoff.
- The presentation skill guided a separate visual comparison before the arithmetic rather than adding another definition to the equation slide. Regression covers both source sets at all seven positions, exact named words and ranges, exclusion of later tokens, the fixed window width, bidirectional control synchronization, keyboard input, retained state and unchanged model outputs. Existing seven-mean/four-preset arithmetic and equation-paint checks still pass.
- All 297 presentation states / 516 formulas, the 128-control sweep, eight independent live/reference cases (7,980 finite values, max error 8.88e-16) and 390px article QA pass. Visually inspected the comparison at i=2, 4 and 7, the mean result and phone views. No model, Part I, shared runtime, dependency or PDF changes.

## 2026-09-12: repair the window-to-pooling teaching sequence

- The focused teaching audit found three prerequisite gaps: a slide highlighted outputs of a later query mechanism before teaching queries; the task silently switched from next-word prediction to gathering context for the already-known bank token; and c denoted both concatenation and a fixed-width pooled summary. Replaced the premature clue slide, moved parameter cost before the alternative, and kept the redundant fixed-slot recap in the article only.
- The new bridge uses the seven actual input words through bank, explicitly identifies position 7 as known, and contrasts seven four-coordinate blocks (one concatenated 1×28 row) with a proposed four-coordinate summary. Averaging is introduced as the first combining rule. Notes retain the compression/information-loss and computation caveats rather than implying that arbitrary context is free.
- Mean pooling now displays its computed four-coordinate result before revealing the colour-matched equation and definitions. The position control updates the visible result, source range, worksheet and table together. All Section 4 pooled summaries use m; c remains concatenation. Weighted presets and numerical values are unchanged. The model itself is never updated by these controls.
- Added a focused regression for sequence order, all seven mean positions, four weighted presets, the zero-weight fallback, retained navigation state, exact independent arithmetic, one-row concatenation, symbol colours and mobile layout. Screenshot pixels and computed visibility check the summation/fraction through repeated reveals; a scoped rule prevents Chromium's nested KaTeX table spans from retaining hidden visibility. The phone concatenation strip scrolls locally rather than wrapping into a matrix.
- All 296 presentation states / 516 formulas, the 127-control sweep, eight live/reference cases (7,980 finite values, max error 8.88e-16), short/long-window regression and 390px article QA pass. Visually inspected the bridge, means at positions 3 and 7, the full mean table, weighted presets, and phone views. One fewer presentation frame. This is a bounded Sections 3–4 sequence repair, not a whole-Part-II pedagogical sign-off. No model, Part I, shared runtime, dependencies or PDF changes.

## 2026-09-12: longer prefixes and the cost of widening the window

- Replaced “cut at ten” with an explicit most-recent-token rule and two twenty-token prefixes. Their final ten words are identical, while river at position 6 and cheque at position 5 lie outside a ten-token window. A separate 10–20 control marks available words, the moving boundary and prediction position 21. Plausible water/teller continuations motivate the lost clues without claiming measured predictions or sending new words through the toy model.
- The visible conclusion handles intermediate widths correctly: Sunday/yesterday already differ at w=12, river enters at w=15 and cheque at w=16. At twenty slots both prefixes fit, with 1,600 weights versus 800 at ten slots. Twenty biases stay fixed. The article adds a 35-token example showing that twenty slots can again miss a clue.
- The cost slide derives wdV + V for the direct linear head, defines each variable and separates weights, biases and head totals. Benchmarks cover w=10, 20 and 100. A labelled larger illustrative design, d=128 and V=10,000, needs 12.8 million weights at w=10 and 128 million at w=100. Notes distinguish head-only counts from embeddings, positional parameters, the earlier MLP and a whole transformer.
- Regression checks all eleven long-window settings, exact suffix equality, clue-entry thresholds, positions, parameter arithmetic, independent short-window state, progressive visibility, keyboard and retained navigation state. All 297 presentation states / 518 formulas, the 127-control sweep, eight live/reference cases (7,980 finite values, max error 8.88e-16) and 390px article QA pass. Visually inspected both window extremes, the cost slide and phone views, including every cost-table column. No new slides, model, Part I, shared runtime, dependencies or PDF changes.

## 2026-09-12: link window-head nodes to colour-matched mathematics

- Replaced the matrix-growth frame's bare equations with an interactive linear-head diagram. Its window control shares state with Inside the window and the concatenation example. Each token supplies four actual scalar coordinates; twenty vocabulary outputs remain fixed. Selecting an output traces exactly its incoming column of W, by mouse or Enter/Space. Dots explicitly account for omitted nodes.
- Blue inputs, purple weights/biases, grey logits and green probabilities follow Part I. Progressive builds connect the nodes to the shape-annotated row multiplication and then softmax. The slide defines c10 and its last-position subscript, counts weights separately from biases, and shows that each extra token needs eighty more weights. It does not add a hidden layer or invent predictions for differently shaped architectures.
- Regression covers all ten windows, every drawn edge/value, output selection, colour matches, matrix shapes, total parameters, both slider directions, keyboard, navigation state, progressive visibility, SVG bounds and model immutability. Visually inspected one-, three- and ten-token slides and the phone article. The general interaction sweep now supports accessible SVG buttons as well as HTML buttons.
- All 297 presentation states / 521 unique formulas pass; the 126-control sweep and 390px article QA report no errors or overflow. Eight independent live/reference cases agree on 7,980 finite values (max error 8.88e-16). No new frames, model, Part I, shared runtime, dependencies or PDF changes.

## 2026-09-12: distinguish stacked rows from concatenated context

- The concatenation frame now shows a labelled before table (w × 4), then reveals a single joined context row (1 × 4w). It defines c as the concatenated context and the subscript 10 as the last input position, with matching blue/orange notation and prose. The scope line identifies the selected input positions and prediction position 11. The table no longer calls its stacked shape the size of c10.
- Matching token blocks and actual values make the joining operation explicit. Longer windows show the same first two/final rows in both views, with counted omissions. All rows remain available in the article. The article explains the shorthand R^(4w); the following general equation uses concat and the explicit 1 × (wd) shape. No model or parameter values change.
- Regression checks all ten windows for exact flattened values/order, full coordinate counts, unchanged subscript, shape labels, colour matches, progressive visibility, one-row layout and both slide builds. Full audit passes 297 states and 516 formulas, numerical reference checks and the 121-control sweep pass, and the phone article has no document overflow. Visually inspected small/full windows and phone rendering. One extra reveal, no additional slide or regenerated PDF.
- A frame-entry refresh fixes Chromium retaining a blank paint cache for the notation column after the hidden table changes height. Pixel checks require both blue c text and orange position text in full-slide screenshots at w=3 and w=10, at both reveal steps.

## 2026-09-12: connect the window boundary to scalar network inputs

- The “Inside the window” slider now updates token groups, four scalar input nodes per token, their connections, the linear-head matrix shape and its weight count. Vocabulary outputs stay fixed at twenty. Dots explicitly account for omitted inputs and outputs. The existing concatenation table shares the same window state; no extra slide or control was added.
- Blue inputs, purple parameters and grey scores match Part I. The preceding MLP sketch now also counts scalar inputs correctly (wd rather than w), retains eight hidden units, and no longer marks an attention-model prediction as if it belonged to this architecture sketch. Notes distinguish the two architectures and explain that resizing requires a differently shaped matrix, not inference under unchanged weights.
- New regression covers all ten windows: retained positions, actual coordinate values, visible edges, omissions, parameter counts, fixed outputs, keyboard interaction, navigation state, reduced motion, phone track width, SVG bounds and unchanged model predictions. All 296 presentation states and 512 formulas, opening-flow regression, eight independent live/reference cases and phone article QA pass. Visually inspected small/wide windows, the MLP at w=100, and the phone layout. No model, Part I, shared runtime, dependency or PDF changes.

## 2026-09-12: identify the two baseline contexts on the comparison slide

- Replaced A/B row labels with River sentence and Cheque sentence. Both complete contexts now appear above the table, with the earlier clue in orange and the final “the” in blue. The title and visible explanation connect the identical starting rows to identical predictions. The difference row compares input coordinates rather than conflating them with a probability-gap statistic.
- The preceding context-switch buttons use the same descriptive names. Both comparison rows keep equal emphasis after either button is selected. No model, frame-count or numerical changes.
- Regression checks require the full saved sentences, correct named rows and four coordinates, matching control names, final-token highlights, zero differences and mobile table fit. Opening/interaction checks, 296 presentation states, 511 formulas, eight independent live/reference cases and mobile QA pass. Visually inspected the revised slide and phone article.

## 2026-09-12: diagram-first output head, matching Part I

- Replaced the bare head equations with a labelled four-input / twenty-output network and matching blue input, purple parameter, grey logit definitions. The actual toy values populate the nodes and the water worksheet. Dots explicitly omit sixteen output nodes; the drawing does not invent an extra hidden layer. Input, weight, bias and logit shapes are defined, and Part II's ell is linked to Part I's z.
- The next frame defines softmax and works through water's probability, 0.094, using all twenty logits in the denominator (10.678 rounded). The conclusion connects the arithmetic to the baseline failure. Full conditional-probability notation is explained in the article instead of appearing unexplained on the slide.
- Browser regression verifies all drawn edges against the model weights, every displayed node value, colour matches across SVG/math/prose, output omissions, worksheet products, normalization, opening order and phone layout. Visually inspected both new frames, the phone network and the complete phone worksheet. One additional teaching frame; no model, Part I, or shared runtime changes.
- Full Part II frame audit, independent live/reference numerical check, mobile article QA and the 121-control interaction sweep pass. No JavaScript or KaTeX errors, slide overflow or phone document overflow. PDFs were not regenerated.

## 2026-09-12: explicit last-token baseline takeaway

- The context-switch slide now states its result in the title: this baseline cannot tell the contexts apart. A visible explanation identifies the shared input, the final “the” at position 10. The closing line says that earlier clues need to influence the prediction. The eight-way tie is explicitly incidental to the toy.
- Preserved all eight candidate bars and the aggregate. Browser regression switches both contexts, checks identical rendered probabilities, and requires the reason and takeaway to be visible at build zero. Visually checked both states and the phone article.
- Opening-flow regression, all 294 presentation states / 490 formulas, phone-width QA, and 8 independent live/reference numerical cases pass. No model or slide-count changes.

## 2026-09-12: remove the dedicated position axis consistently

- The bank model now uses four coordinates (water, finance, person, glue) and adds small hand-chosen position vectors across those coordinates. This supersedes older five-coordinate/ignored-position notes below. The new opening frame defines the four feature columns using river, bank, fisherman and the, with matching coloured definitions and an explicit real-embedding caveat.
- The position example now reads the actual model tables; the ten-by-four matrix, baseline, Q/K/V worksheets, walkthrough, SVG exports and Part III training results all use the same data. Removed the repeated late position frame; the diagnostic explanation is an article companion. Part III's FFN is 4→8→4.
- Python and independent JavaScript agree on 6,465 intermediate numbers (max error 1.78e-15). All toy targets pass, including the position-on/off prefix-permutation regression. Saved training agrees on 1,386 entries; finite differences verify all 212 used parameter scalars (max error 5.38e-11), and 40 unused-position scalars do not affect the loss.
- Browser/reference comparison: 48 capacity/mask/scale cases, 82,544 finite values, 12 invalid-input checks; independent FFN arithmetic agrees. Live diagram evidence: 299 values checked; all 12 classroom stages fit. Routing/value ablation and score-scaling regressions pass.
- Full presentation audits: Part II 294 progressive states / 491 formulas, Part III 138 states / 139 formulas; no overflow, JavaScript errors or math failures. Interaction sweeps exercised 121 and 34 controls with zero errors. Both articles fit 390px width. The opening test checks actual table values, colours, flow, all three position choices, and phone tables.
- Visually inspected the coordinate primer, position sums, sentence matrix, corrected last-token vector, query worksheet, residual calculation, diagram and Part III FFN; also inspected the phone coordinate primer. No new dependencies or changes to Part I.

## 2026-09-11: character placeholder versus literal

- Section 4 now displays c = "a" with a grey placeholder and an orange quoted literal, each explained in matching
  prose. The note explicitly excludes the quote marks from the name. Alternate values "b"/"i", id("a"), the
  concrete embedding subscript, and generic lookup equation follow the same distinction.
- Added the Part 1 pSymbol role and notation regressions for exact quoted values, distinct colours, and both generic
  c occurrences. Visually inspected the changed character, ID, embedding, and full-equation frames. Article/print
  colour-guide checks and slide-fit checks pass. No new frames, model changes, code snippets, or PDFs.

## 2026-09-11: window and embedding dimensions together

- Replaced the section 14 window-only sketch with two labelled controls and a coordinate-grid/concatenation diagram.
  Both w and d change the wd input width, W1 shape, and weight count. Ellipses mark omitted coordinates and positions.
  The diagram retains fixed hidden/output widths, labelled weights/biases, and says b1 is added before ReLU.
- Added a synchronized five-row parameter worksheet including the embedding table and biases. It explains shared
  token rows, separates the effects of w and d, and displays the total with thousands separators. Controls preserve
  their state across frames and expose actual values to assistive technology rather than only preset indices.
- The five-line PyTorch example and downloadable script create a separate model with w=5,d=4 and 1,671 parameters.
  All 33 snippets execute. Independent PyTorch models verify all 25 choices, parameter counts, and batched outputs.
  Existing saved-model forward, sampling, and 1,169-parameter gradient regressions still pass. No trained data changed.
- Browser checks cover all 25 pairs, real tile counts and omissions, shape metadata, bounded SVG text, frame fit,
  keyboard changes, retained state, and model immutability. The 84-instance SVG suite checks 2,680 bounded labels.
  All 230 progressive states, 121 formulas, and 19 equation guides pass. Tables pass at reading, phone, and slide sizes.
- Visually checked default and maximum settings, the full parameter worksheet, code, larger-width arithmetic, and
  portrait slides. Phone controls fit. Print media retains the chosen width and five worksheet rows. No PDF regenerated.
  Existing deck typography/colours and vanilla controls are retained. No dependency or shared presentation-style changes.

## 2026-09-11: concrete examples of training and generation inputs

- Replaced the abstract "Many windows or one growing output" table with two diagrams. The training frame shows all
  six observed aabid input/target pairs and six separate output rows from one MLP batch. It explicitly says that a
  wrong prediction cannot replace observed i in the next training window abi. The existing batch code now follows
  that motivation, with X [6, 3], logits [6, 27], and target IDs y [6] explained.
- The generation frame follows the actual seed-1, temperature-1 run from aab: draw h (0.090), shift to abh, then draw
  stop (0.183), keeping the name aabh. Its lock symbols preserve the frozen-parameter convention. Text distinguishes
  sequential steps within one name from batching independent names. No model values or executable snippets changed.
- New browser tests verify each training pair against the saved data, each generation transition/probability/name
  against the live sampler, the boundary stop, and the new teaching order. Independent Torch checks reproduce both
  displayed probabilities. All 33 literal snippets and existing 1,169-parameter gradient checks pass.
- All 230 progressive states, 126 formulas, 18 equation guides, and table checks pass without slide overflow. SVG
  checks cover 80 instances and 2,592 bounded labels. Visually inspected both new frames, the adjacent code frame,
  and portrait presentation. Print media retains all six training rows and both generation calls. No PDF regenerated.

## 2026-09-11: visible frozen parameters in the two-loop comparison

- Kept the three-stage section 11 diagram. Training now has a trainable-parameter view and an orange gradient/optimizer
  return arrow. Generation has a lock and five vector snowflakes, one for the embedding table and each weight/bias
  group. Its next-input arrow remains separate. Plain-text keys distinguish fixed parameters from recomputed activations
  and probabilities. The article clarifies no_grad(), no optimizer step, and why eval() alone does not freeze weights.
- The actual seed-1 draw of h after a a b and its next window a b h are unchanged. SVG regressions now require every
  frozen marker to belong to a parameter, not an activation, and verify the progressive reveal of the two loop arrows.
  All 72 SVG instances, 2,388 bounded labels, and existing forward/sampling numerical checks pass.
- All 229 progressive states and 126 formulas fit without overflow. The 18 equation guides pass. Visually checked
  every changed reveal and portrait presentation. Print media keeps all five snowflakes. No PDF or new code snippet
  was generated, and no other lesson or shared presentation style changed.

## 2026-09-11: seed and starting text in the live generator

- Added labelled seed and starting-text inputs. Starting fragments become lowercase, pad with boundary tokens when
  shorter than three letters, and retain the whole prefix in the name while using only its last three letters as input.
  Generate replays the exact entered seed; New seed is explicit. Start over preserves settings. Changing a setting
  clears the run. Invalid text or non-uint32 seeds show an inline error and disable generation.
- Added `check_part1_generator.mjs`: exact Generate/Next replay, seed zero and uint32 maximum, short/long/uppercase
  prefixes, invalid settings, probability bars and grouped mass, chosen tail tokens, greedy invariance, boundary/cap
  stopping, native typing/Enter/Escape, and retained settings across navigation. Model weights are unchanged.
- Visually checked default presentation, the maximum 34-letter output (16 supplied + 18 generated), portrait slides,
  and 390px article layout. All fit without slide scrollbars or horizontal article overflow. Print-media checks retain
  the entered values and current output. No PDF regenerated. The duplicate card heading was removed to keep space
  for the controls; the existing typography, colours, and 1280×720 stage are retained.
- All 229 progressive states, 126 formulas, and 18 equation guides pass. Existing 72-instance SVG and numerical
  generator regressions pass. Tables pass in reading, phone, and presentation modes. Metadata checks pass for all
  eight parts. No other lesson, shared runtime/style, or model data changed.

## 2026-09-11: temperature before the generation control

- Added three paced frames before temperature first appears in executable code: a same-input comparison at
  0.5/1.0/1.5, a colour-keyed softmax(z/T) explanation, and actual divided-logit arithmetic for n and m after `- s a`.
  The charts retain a common scale and token order. Other probability mass is labelled as a total over 22 tokens.
  The numerical softmax includes all 27 tokens and uses unrounded logits.
- The text distinguishes changed sampling probabilities from unchanged weights and greedy choices. T must be positive.
  The next frame's existing five-line PyTorch function implements the operation. The live generator adds a mode-aware
  temperature note. Its generation and slider algorithms are unchanged. No PDF was regenerated.
- All 229 progressive states, 126 unique formulas, and 18 equation guides pass fit/notation checks. Table checks pass
  in reading, phone, and presentation modes. The chart, formula, numerical worksheet, and greedy-mode control were
  visually inspected. All 33 snippets execute. Independent Torch tests reproduce n's probabilities at all three
  temperatures, confirm normalization and fixed argmax, and show increasing entropy. Existing gradient tests pass.
- SVG checks cover 72 instances and 2,342 bounded labels. New regressions compare every displayed temperature bar
  and worksheet cell with the live generator, and require the explanation before its code/control.

## 2026-09-11: a complete animated generation run

- Replaced the abstract loop with 12 progressive states covering four actual saved-model calls: `---`, `--s`, `-sa`,
  then `sam`, which samples the boundary and stops. Each call separates its probabilities, draw, and append/shift.
  The new window and accumulated name stay distinct. The sampled token stays visible even outside the top five.
- Manual Next/Back, Play/Pause, replay, leave-frame cancellation, and reduced-motion support pass browser checks.
  A separate four-row trace preserves the full example in final-state slide exports. No PDF was regenerated.
- Added a same-context greedy/sampling comparison immediately before the unchanged live generator. The browser's
  seed-3 sample yields sam; greedy yields san. Both draw from all 27 probabilities. Python RNG differences are explicit.
- All 33 literal PyTorch snippets execute and match the download. Independent Torch checks reproduce all four chosen
  probabilities and the full greedy run. Existing model and all 1,169 NumPy-gradient checks remain passing.
- All 226 presentation states and 124 unique formulas pass fit/notation checks with no nested scrolling. Table checks
  pass for desktop, phone, and presentation. SVG checks cover 68 instances, 2,150 bounded labels, local marker IDs,
  every worked stage, and actual stored-model probabilities. Classroom start/shift/stop, full trace, and comparison
  views were visually inspected. No shared presentation styling or other part changed.

## 2026-09-11: full embedding table during backpropagation

- Added a single follow-on frame after the learning graph showing all 27 rows and 54 stored coordinates of E_tok.
  Three blocks keep the full table readable without omissions. Only a and b are outlined. The explanation distinguishes
  the two accumulated a contributions, one b contribution, and zero embedding gradient for the remaining 25 rows.
  Target i remains outside the input lookup. The claim explicitly concerns this example's cross-entropy only.
- Preserved the original four-stage computation graph. Its separate frame keeps both views in default final-state
  slide exports. No PDF was regenerated. Classroom code, model parameters, and the downloadable snippets are unchanged.
- New PyTorch checks confirm exactly rows 1 and 2 have nonzero gradients, row a equals the sum from its two positions,
  row b equals its single contribution, and a plain SGD step changes only these two embedding rows. All 32 snippets
  and existing saved-model/NumPy gradient checks still pass.
- SVG checks cover 34 instances, all 27 displayed rows and their actual values, row highlights, 1,174 bounded labels,
  and local arrow markers. Matching-colour checks pass. Visually inspected the new classroom view; phone reading has
  no horizontal overflow. Browser print media retains the separate graph and table frames. Metadata builds pass.
- All 216 progressive presentation states and 124 unique formulas pass, with no overflow or nested scrolling.

## 2026-09-11: compact MLP-to-probability sequence

- Reduced the nine classroom frames beginning with the worked hidden unit to four. Kept a combined MLP calculation,
  softmax with matching-colour explanations and code, one numerical table, and the final distribution. Full article
  mode retains the product worksheets and numerical-stability derivation/code. No snippets or model data were removed.
- The table now uses direct exp(z), matching the displayed softmax formula, with three-decimal logits, exponentials,
  and probabilities. It includes all 27 tokens in the total, grouping 22 in the last row. Target i remains 0.1169677634.
  The hidden/logit calculation and final bars use the same three-decimal convention. Computation retains full precision.
- All 32 PyTorch snippets pass, including the stable-softmax equivalence and saved-model checks. New browser assertions
  verify compact-frame count, retained article worksheets, numerical sums, every table row, and the target probability.
- All 215 progressive states and 124 unique formulas pass with no overflow or nested scrolling. Seventeen equation
  guides pass matching-colour checks. Tables pass at reading, phone, and classroom widths (35/35/24 tables). Visually
  inspected all four classroom frames. Isolated metadata builds pass. No other lesson, shared runtime, or PDF changed.

## 2026-09-11: examples versus features in tensor shapes

- Split the section 6 shape table into activation and parameter frames. The activation table explicitly labels rows
  as examples and columns as input features, hidden activations, or vocabulary scores. Each entire context window is
  one example. Matching symbol/prose colours remain intact. The parameter frame distinguishes feature axes from the
  batch dimension and explains shared bias rows and PyTorch broadcasting.
- Added a five-line, four-example forward pass and synchronized the standalone download. All 32 snippets execute;
  [4, 6], [4, 32], and [4, 27] match the displayed shapes. Every batched output agrees with a separate forward pass.
  Saved probabilities and the NumPy/autograd gradient checks still pass; model parameters were not changed.
- All 223 progressive states and 122 unique formulas pass, with no overflow or nested scrolling. The table audit
  passes 36 reading, 36 phone, and 27 classroom tables. Sixteen equation guides pass article/print colour checks.
  Visually inspected the activation table, parameter table, and batch-code frames. Eight isolated metadata builds pass.
- Only Part 1 content, checks, and its download changed. No shared runtime, other lesson, or PDF changes.

## 2026-09-11: ReLU and visible weights/biases

- Switched Part 1 consistently from tanh to ReLU, including NumPy forward/backward, JS runtime, worksheets, diagram,
  recap, word-model example, and downloadable PyTorch code. Retrained the 1,169-parameter model using the existing
  dataset, split, seed, dimensions, 6,000 steps, and embedding sign penalty. Train loss is 2.200583 and held-out loss
  2.203077. These are results from the regenerated checkpoint, not unchanged tanh values.
- Added W1/W2 labels on the network connections and b1/b2 labels at the hidden/output columns. Parameters and their
  prose use the same purple. Hidden activations use teal. Explanations separately identify each matrix/bias and shape.
  A simple negative/zero/positive example and three-line PyTorch snippet introduce ReLU's rule.
- All 31 literal snippets execute and match the standalone download. PyTorch and JS match the saved six probability
  rows within 2.78e-16. NumPy's objective and all 1,169 gradients agree with independent autograd checks for both the
  initial and fitted checkpoints. Generation retains the original seed; it now samples h from a a b, yielding a b h.
- All 221 presentation states pass, with 122 unique formulas, no overflow, and no nested scrolling. Sixteen equation
  guides pass matching-colour checks in article and print media. Tables pass in reading, phone, and classroom modes.
  Both MLP sketches, 30 original diagram instances, and all 11 embedding-primer diagrams pass their regression checks.
  Visually inspected the labelled network, equations, ReLU example, and regenerated hidden/logit worksheets.
- No shared runtime, other lesson, or PDF files were changed in this checkpoint.

## 2026-09-11: honest layer widths in the MLP sketches

- The full-network frame shows all six input coordinates, eight numbered hidden units (1–4, 29–32), and representative
  outputs. Dots mark every omitted range, including both sides of output i. The hidden column is taller than either
  neighbouring column. The window-size sketch uses the same correct 32-unit convention.
- Regression checks cover real unit numbers, visible labels, gap counts, output order, true accessible widths, target
  highlighting, and dense edges between displayed nodes only. Saved arithmetic and generation remain unchanged.
- Part 1 passes all 220 progressive frame states with 121 unique formulas and no overflow or nested scrolling. Visually
  inspected both changed classroom frames and the full-network diagram at 390px reading width. Shared presentation
  interaction tests and eight isolated metadata builds pass. All eight HTML outputs were rebuilt for the shared helper.
- No training data, computational snippets, or PDFs changed.

## 2026-09-11: the embedding lookup, symbol by symbol

- Replaced the combined table/equation frame with six paced frames: character, ID mapping, whole table, square brackets,
  retrieved vector, and complete equation. Added a seventh, three-line PyTorch example after constructing the layer.
  The explanation includes parentheses, equals, capital/lowercase E/e, subscripts, and the token abbreviation.
- Characters use rust and IDs muted violet, with matching symbol/prose colours. Embedding rows and their table retain
  the established blue. Explicit row-ID columns distinguish labels from the two learned coordinate columns. Displayed
  rows still come from the unchanged saved model. The complete equation appears only after its pieces are introduced.
- All 220 progressive states pass frame checks, with 121 unique formulas, no overflow, and no internal scrolling.
  Sixteen equation-guide groups pass article/print colour checks. The table checks pass 35 reading, 35 phone, and 26
  presentation tables. All seven lookup frames were visually inspected at classroom size, including the selected row.
- The 30 executable snippets pass, including ID 1 selecting the exact embedding row with shape [1, 2]. The standalone
  Python file stays synchronized. Saved model probabilities still match within 1.53e-16. JS arithmetic/generation,
  embedding-primer, original SVG regression, and isolated metadata checks pass. No PDF export or shared-runtime rewrite.

## 2026-09-11: motivate embeddings before the lookup table

- Section 4 adds 22 paced frames: motivation, a row as a 2D point, several points, training's role, representations across
  domains, and a return to the same `a a b` context. The original trained coordinates and numerical model are unchanged.
  The existing vowel-sign constraint is disclosed beside the plot; no claim of spontaneously discovered semantic axes.
- Per Nipun's follow-up, Word2vec stays simple: six frames around king/queen/man/woman, including a prediction-learning
  diagram, points, matching offsets, a same-colour equation reading key, and a four-line toy-vector calculation. The
  coordinates are explicitly invented. The analogy is neither an exact general rule nor the training objective.
- Documents, images, and signals each have an encoder diagram, task intuition, and short executable code. Their code uses
  separate fresh layers, not pretrained encoders. The existing two-mug scene is embedded for offline use. No dependencies
  or shared CSS/presentation runtime were changed; no new PDF was exported.
- `check_embedding_primer.mjs` passes 11 SVG instances, bounds/accessibility, saved point coordinates, analogy arithmetic,
  full-size diagram checks, 23 new/revised frame screenshots, and phone layout. Visually reviewed character plots/table,
  learning flows, analogy points/arrows/equation/code, document/image/signal diagrams, and the return to character lookups.
- `frame_audit.mjs` passes 212 progressive states, 115 unique formulas, with no overflow or internal scrollbars.
  The table audit passes 33 reading, 33 phone, and 24 presentation tables. All 11 equation guides pass article and print
  colour/layout checks. These are browser print-media checks, not a fresh PDF export.
- All 29 literal PyTorch snippets execute; analogy and encoder shapes agree with the slides. The saved model's six
  probability rows still agree within 1.53e-16. Original JS arithmetic and generation, 30 existing SVG regression instances,
  and eight isolated metadata builds pass. The standalone Python download is synchronized with the HTML snippets.

## 2026-09-11: Part 1 executable PyTorch stages

- Added 25 code frames, each with 2–5 literal lines, nearby explanations, and relevant tensor shapes. The same character
  model runs from IDs through `nn.Embedding`, concatenation, `nn.Linear`, softmax, loss, an SGD update, and generation.
  The stored-weight transpose, logits-versus-probabilities distinction, and train/eval-versus-gradient modes are explicit.
- `check_part1_torch.py` executes the exact HTML snippets in order with PyTorch 2.13.0. All shapes, loss equivalence,
  finite parameter gradients, parameter changes, sampling context, and standalone-file synchronization pass. Loading the
  saved checkpoint into those layers reproduces all six full vocabulary distributions within 1.53e-16.
- `examples/part1_pytorch.py` is the downloadable code in lesson order. It starts untrained and illustrates one update,
  not the saved model's entire training run. No training data, numerical model, shared runtime, or dependencies changed.
- The frame audit passes all 185 progressive states, with 114 unique formulas and no overflow, internal scrollbars, or
  runtime errors. The table audit passes 33 reading, 33 phone, and 24 presentation views. Phone reading has no document-width
  overflow. The notation test still passes all ten equation guides in article and print-media modes.
- Visually checked the embedding, storage transpose, cross-entropy, training update, generation loop, and forward recap
  at the 1280×720 classroom size. `check_part1.mjs` and eight isolated metadata builds also pass. No new PDF was exported.

## 2026-09-11: Part 1 equation reading keys

- Added ten introductory/recap equation-guide groups, including the long-form general probability expression.
  Symbols and adjacent plain-language explanations share scoped Part 1 colours. Conditioning, indices, operators,
  shapes, and the distinction between the letter “i” and an index are explained in words.
- Split the two MLP equations, stable softmax, loss examples, and mathematical recap across appropriate frames.
  All 160 progressive states pass `frame_audit.mjs`, with 114 unique formulas parsed and no overflow or nested scrolling.
- `notation_test.mjs` checks exact rendered symbol/prose colours, shape-table meanings, 16 representative full-build
  frames, the 390px article layout, and print-media styling. Ten guide groups pass. This is not a new PDF export.
- `check_part1.mjs` still matches all six trained rows to 1.6653345369377348e-16 and generation at four temperatures.
  The table audit passes 33 reading, 33 phone, and 24 presentation table views. Metadata and isolated builds pass.
- Visually reviewed probability, hidden-layer, loss, stable-softmax, parameter, shape-table, recap, and phone views.
  The new authoring convention applies to future equation revisions across the series; this pass changes Part 1 only.

## 2026-09-11: presentation controls and the Part 1 introduction

Baseline: local `main`, `origin/main`, and the successful Pages deployment all pointed to `632093b`. This includes Claude's
September 5 rebuild and cover slides. The unfinished whiteboard pass remains separate on `wip/whiteboard-pass`.

- Navigation uses one compact row and disappears after three idle seconds. Pointer hover and keyboard focus keep it open.
  **C**, pointer movement at the lower-right corner, and touch restore access. Touch hover styling is explicitly scoped
  so a tapped button does not remain painted after the timer expires. No teaching-widget controls are hidden by this timer.
- The shared cover contains its label, title, and subtitle only. Part 1 removes the generated-name teaser and the promise
  to write every parameter by hand. Six language tasks now get separate worked input/output examples before their output
  shapes are compared. `aabid` bridges to character prediction, and one frame explains the indexed probability in words.
- `controls_test.mjs` passes pointer, idle, focus, Tab, Escape, overview, and touch checks. Its stage geometry checks pass
  at 1280×720, 2048×1011, 3650×1802, 1024×768, and 390×844. The stage retains its 16:9 ratio and logical 28px body type.
- `pres_test.mjs` passes with zero errors/failures, including managed/manual controls, notes, presenter window, print
  preparation/restoration, and real overflow warnings. `check_metadata.py` passes eight configs and eight isolated builds.
  Part 1's six numerical rows and generation at four temperatures still pass `check_part1.mjs`.
- The eight `frame_audit.mjs` walks pass 1,123 progressive states: Part 1 155, Part 2 298, Part 3 138, Part 4 86, Vision I
  147, Vision II 96, Vision III 118, Vision IV 85. No frame overflow, nested scrollbar, runtime error, or strict math-parse
  failure. Part 1 has 70 frames including the cover. Its table audit passes 33 desktop, 33 phone, and 24 presentation table
  views. The article also passes 390px document-width checks.
- Visually inspected the cover with controls open/hidden, Hindi translation, entity labels, summarization, next-word bars,
  probability notation, and phone entity layout. Removed a redundant bar-chart caption that crowded the following line.
- **Known test maintenance:** `interaction_test.mjs` reaches an obsolete Vision II selector `#s03-slider` after its earlier
  shared-layout checks. Claude's September 5 vision rebuild replaced that widget. This run does not claim that full legacy
  script passed. The independent presentation and new controls suites pass. No new PDF export was made in this UI pass.

## 2026-09-05: CLIP geometry and temperature checkpoint

Seven short frames connect a fixed classifier's output menu to candidate descriptions, show normalization geometrically, plot the actual six unit vectors, follow their training updates, and separate directional learning from temperature. The first pair has zero third coordinates and uses an exact 2D slice; subsequent plots use a fixed orthographic view of all three coordinates. Numbered image thumbnails stay attached to captions and paired cosines. The 0/1/20/60-step control opens at step 60, while earlier authored frames preserve steps 0 and 1 for static readers and PDF export.

- All 1,160 reference/invariant comparisons and 246 finite-difference gradients pass. The independent NumPy training reproduction is unchanged. New tests verify checkpoint values, the fixed projection/scale/origin, actual plotted vectors, probability bar heights, and a fixed-temperature comparison using the trained encoders.
- All 35 frames / 101 progressive states fit without an internal scrollbar or overflow; 65 formulas strictly parse. Browser tests pass 12 candidate-control layouts, four training-control layouts, and four geometry-control states. All 27 table views pass reading, phone, and presentation checks. Eight isolated builds pass metadata checks.
- New normalization, geometry, temperature, and comparison-table slides were visually inspected, along with the 390px phone geometry layout. The 35-page temporary PDF exports at 16:9 with its reveal answer shown; its normalization and trained-vector pages were rendered with Poppler and inspected. PDF text remains rasterized. This is a targeted visual review, not an exhaustive inspection of every PDF page.
- No model weights, saved training data, shared presentation CSS, or production dependencies changed. Provenance and limitations remain in article/speaker notes and asset credits; the classroom does not repeat a production caption under every scene.

## 2026-09-05: classroom caption cleanup

Removed the repeated production caption from the shared scene figure in all four vision lessons. Asset provenance remains in `figures/vision-scene/README.md` and the article scope/speaker notes. Human judgments and generated illustrations are still distinguished from calculated model outputs. All 17 scene frames pass offline asset decoding, crop geometry, SVG text bounds, and phone containment checks. No model parameters, numerical examples, or presentation styles changed.

## 2026-09-05: VLM source-contribution checkpoint

Vision IV now shows all four image-value messages and all three prompt-value messages before their sum. A spatially arranged image-slot diagram retains the original attention mass on text. The same-query comparison and signed logit-difference chart use the fitted model's actual outputs. The image remains visible throughout all three generation stages. Five additional short frames replace no existing arithmetic, training data, or parameters.

The model checks pass 5,631 reference/invariant comparisons and 1,096 finite-difference gradients. New checks cover every source message, projected contribution, and complete logit contrast for both images, both snapshots, and three prefix lengths. The existing one-image training regression is unchanged. The 41 frames / 90 progressive states pass the frame audit with 63 valid formulas, no overflow, and no nested slide scrolling. All 62 table views pass desktop, phone, and presentation checks. Browser tests also verify the image persists through generation and inspect SVG bounds. Representative new classroom frames were visually reviewed. This checkpoint does not include a fresh Vision IV PDF export.

## 2026-09-05: masked-image learning checkpoint

Vision II now follows an actual 72-parameter image-to-prediction-to-update experiment, alongside the original loss calculator. The example exposes visible-patch projection, encoder rows, decoder slot restoration, values and weights, residual update, predicted pixels, masked loss, one SGD update, reconstruction checkpoints, and held-out results. A repeated-tile baseline and an ambiguous hidden-tile failure bound the interpretation. The frozen-encoder probe reports its actual unchanged 100% accuracy instead of claiming improvement.

- The saved training run reproduces all 881 reference numbers exactly. All 288 scalar finite differences pass (maximum error 1.60e-11). Mutating only hidden target pixels leaves predictions unchanged. Initial parameters are not mutated by the update helper.
- Legacy Vision II arithmetic still passes all 40 cases. Twelve new browser control states reproduce their corresponding saved-parameter predictions and replace, rather than accumulate, figures. SVG text bounds pass.
- All 43 classroom frames / 79 progressive states fit, with 41 strictly parsed formulas and no runtime error, internal slide scrollbar, or overflow. Phone reading at 390×844 has no document-width overflow. Table checks cover reading, phone, and presentation modes.
- The 43-page temporary PDF exports with all three reveal answers shown. Representative calculation, reconstruction, and computation-graph pages were visually inspected. A text/arrow overlap and SVG edge padding were corrected before the final export. This is a targeted visual check, not a claim of reviewing every PDF page.
- Metadata checks pass all eight isolated lesson builds. No shared stylesheet or presentation runtime changed. New lesson code has no production dependencies.

Reproduce with the commands in `HANDOVER.md`, including `check_vision2.mjs --browser --page vision2.html`, `check_vision2_learning.mjs`, and `train_vision2.mjs --check`. DINO/I-JEPA, CLIP geometry, and VLM source-contribution teaching improvements remain separate checkpoints in the audit plan.

## 2026-09-05: notation and continuity checkpoint

Patch side length now uses `s_patch`; `P` consistently denotes position vectors. Vision IV's notation includes the connector output `B`. The transitions distinguish reusing an encoder architecture from reusing its trained checkpoint. Numerical parameters are unchanged.

All four numerical checkers pass. The four frame audits cover 127, 62, 84, and 85 progressive states, respectively, with no runtime errors, invalid formulas, or overflowing frames. A caption that initially exceeded Vision II's stage by four pixels was shortened and rechecked. This checkpoint does not include new PDF exports or claim a complete visual audit of every frame. The larger teaching revision is tracked in `VISION_AUDIT_2026-09-05.md`.

## 2026-09-04: shared-scene vision polish

The four vision lessons now follow one generated photographic illustration: two mugs, a book, and a plant. Scene views motivate the task; the existing 4×4 models remain separate numerical worksheets. Captions identify generated imagery and distinguish human judgments, withheld targets, symbolic encoder features, and actual computed outputs. No new photo-based model scores, attention maps, or reconstructions are invented.

| Lesson | Frames / final PDF pages | Progressive states | Strictly reparsed formulas | Table views |
| --- | ---: | ---: | ---: | ---: |
| Vision I | 60 | 127 | 78 | 47 |
| Vision II | 26 | 62 | 35 | 34 |
| Vision III | 28 | 84 | 61 | 25 |
| Vision IV | 36 | 85 | 55 | 53 |

- All 150 frames, 358 progressive states, 229 formulas, and 159 table views pass the frame/math/table checks. Table coverage includes desktop reading, 390px phone reading, and classroom presentation. No stage overflow, nested classroom scrollbar, or runtime error was reported.
- Desktop/phone QA and control sweeps pass for all four pages. All 17 new scene frames pass asset decoding, crop geometry, SVG-label bounds, and offline checks. Phone scene comparisons unfold vertically; no scene introduces horizontal document overflow. The patch enlargement uses the actual selected image region, and the narrow reading layout retains its source outline.
- Numerical model data and training snapshots are unchanged. Independent Vision I verification passes 396 finite differences (maximum error 1.02e-9) and NumPy/browser parity (4.62e-14). Vision II passes 40 numerical cases and invariants. Vision III passes 900 reference comparisons and 246 gradient checks (maximum error 2.52e-11); its saved loss trajectory is reproduced. Vision IV passes 4,779 reference comparisons and 1,096 gradient checks (maximum error 6.94e-8); its saved update and the regression on the other image are retained.
- Metadata checks pass for all eight lesson configs, 91 sections, navigation labels, isolated builds, and planned links. No shared presentation theme or production dependency changed. Diagram labels were enlarged and geometry adjusted within the existing semantic colors.
- PDFs contain the completed classroom frames, with reveal answers shown, at 16:9 and 2× raster resolution. Actual PDF pages were rendered with Poppler and visually inspected: scene/crop stages, target-encoder paths, calculations, generation, and revealed answers. A small Vision I gradient-label collision found during this inspection was corrected before its final export. PDF text remains rasterized, not selectable.

Reproduce with the vision numerical checkers, `verify_vision1_learning.py`, `train_vision3.py --check`, `train_vision4.py --check`, `check_metadata.py`, `check_vision_scene.mjs`, and the shared frame/table/QA/sweep/export commands in `HANDOVER.md`. Asset prompts and limitations are in `figures/vision-scene/README.md`. These checks are not a claim of exhaustive browser compatibility, accessibility conformance, or learned-model generalization. Earlier release counts below are historical.

## 2026-09-04: independent review repair pass

`REVIEW_RESOLUTION_2026-09-04.md` records the accepted findings, independent corrections to proposed fixes, teaching additions, and remaining limits. This entry supersedes the counts in earlier historical entries below.

| Lesson | Frames / final PDF pages | Progressive states |
| --- | ---: | ---: |
| Part I | 61 | 132 |
| Part II | 133 | 297 |
| Part III | 68 | 134 |
| Part IV | 32 | 85 |
| Vision I | 57 | 123 |
| Vision II | 25 | 62 |
| Vision III | 26 | 81 |
| Vision IV | 31 | 74 |

All 988 progressive states pass strict math validation and frame preflight/live-fit checks: no clipped stage, nested classroom scrollbar, or runtime error. The table checks cover desktop reading, 390px phone reading, and presentation. Two Part II narrow-label advisories were inspected: the short labels intentionally wrap into two lines; they do not overlap neighboring cells. Phone tables pan within their containers.

The focused interaction regression passes 49 checks, including repeated Vision II slider/mask changes, their visible numbers and losses, arithmetic popup bounds at several stage scales, notes reservation, native slider keys, frame-title overview, full-width notation, and Part III chip destinations. QA and sweep intentional-error fixtures now exit nonzero. The broad release batch checks all eight pages at desktop 1280-by-720 and phone 390-by-844, exercises their controls, and runs the presentation and PDF-export contracts. Final Vision II/Vision IV diagram/header changes are checked again after rebuilding.

Numerical checks reproduce the unchanged model arrays and the corrected full-precision diagnostics: 6,615 shared-model references; 8,220 live-model values; 84,968 position-capacity values; all 1,678 stored training entries; 260 scalar gradients with maximum error 4.51e-11; and 50 perturbations of unused position rows. Part IV passes 2,274 references and 188 gradients/updates. The vision checkers cover their disclosed forward passes, objectives, training updates, image permutations, grayscale invariants, and the new held-out counting failure. Metadata checks verify all eight configs, 91 sections, matching navigation names, eight isolated clean-directory builds, and three planned-link cases.

All eight PDFs use the final 16:9 classroom view at 2x raster resolution, with reveal answers shown. Actual PDF pages were rendered for visual inspection, including the new Vision I comparisons, CLIP question, corrected arithmetic tables, grayscale transforms and VLM scores. That inspection caught two Vision II label collisions missed by bounding-box tests; the diagrams were spaced out before the final export. PDF text remains rasterized rather than selectable. Presenter notes are not printed over slides.

Reproduce with the current all-eight build/check commands in `README.md`, plus `node src/interaction_test.mjs` and `python3 src/check_metadata.py`. For future changes, run checks after changing widget state, not only when the page first opens. Testing in Chromium is not exhaustive browser, accessibility, projector, or classroom validation.

## 2026-09-04: Vision I teaching rebuild

The first Vision I adaptation met numerical and layout checks but did not give students enough motivation or intermediate workings. This rebuild was reviewed against the text lessons as a teaching sequence, not scored by slide count. It keeps the same block images visible through projection, matching, mixing, prediction, and learning; uses direct teacher questions; and moves optional formal detail into article companions.

- 55 authored classroom frames and PDF pages; 120 progressive states. Preflight and live navigation have no overflow, nested slide scrollbars, JavaScript errors, or invalid mathematics (78 strictly reparsed formulas).
- 21 reusable SVG stages keep crops attached to their vectors. Individual projection columns, a changed pixel, score/exponential/normalization stages, value contributions, and a value-only intervention expose previously skipped operations. Every rendered frame was visually reviewed across the team; this caught and fixed a source-label overlap that canvas-bound checks alone missed.
- The original encoder arithmetic is unchanged. Independent NumPy calculations reproduce the 44-parameter learning model and its training checkpoints within 4.62e-14. All 396 central-difference comparisons pass (44 parameters × three objectives × three snapshots), maximum error 1.02e-9.
- Full-batch SGD on the two pictured training images: learning rate 0.05, 600 updates. Mean loss is 0.809919 initially, 0.693813 after one step, and 0.004388 after training. The first step worsens one image; that regression stays visible. The fitted examples are not evidence of general counting ability.
- 47 table views across desktop article, 390px phone, and classroom modes pass with no issues. Phone reading has no document-width overflow; interactive controls and reveals report no runtime errors.
- The exported PDF has 55 valid 16:9 slide images at 2× raster resolution. All three reveal answers are open. Actual PDF pages 1, 28, and 55 were rendered and visually inspected after export.
- Vision IV's fixed initial encoder still passes its reference checks. The separate Vision I fitted snapshot does not silently replace it.

Reproduce with `check_vision1.mjs --browser`, `verify_vision1_learning.py`, and the frame/table/article/PDF commands in `README.md`. The following four-part release record describes the earlier adaptation, before this teaching rewrite.

## 2026-09-04: four-part Vision to language extension

The adapted articles use the existing slide-first runtime without new production dependencies or changes to shared CSS/JS. Original vision articles and lecture sources are preserved. `VISION_SOURCE_AUDIT.md` records the source corrections and explicit toy/full-model boundaries.

| Vision part | Frames / PDF pages | Progressive states | Strictly reparsed formulas |
|---|---:|---:|---:|
| I: Vision Transformer | 30 | 85 | 67 |
| II: visual pretraining | 25 | 62 | 32 |
| III: CLIP | 25 | 78 | 55 |
| IV: vision-language generation | 31 | 74 | 49 |

- All 299 states pass preflight and live navigation with no stage overflow, nested slide scrolling, JavaScript errors, or invalid maths. Open reveals and managed builds are included.
- All 151 table views pass desktop-reading, 390px-phone and classroom checks with no issues. CLIP's narrow-phone comparison tables unfold into labelled value cards; the same numerical data remain visible.
- Vision I: exact patch/CLS arithmetic, same-width position addition, permutation behavior, causal/unmasked comparison, and pixel edits pass. The full pre-norm ViT block stays separate from the simplified numerical worksheet.
- Vision II: 40 numeric cases, masked MSE, normalized teacher/student distributions, EMA, all 10 SVG-stage bounds, and assembled-page MAE/DINO controls pass. Calculator outputs are explicitly illustrative, not pretrained-model evidence.
- Vision III: 900 JS/reference scalars agree within 4.45e-16; 246 finite-difference gradient checks within 2.53e-11. Exact NumPy regeneration passes. All 12 candidate/template/temperature states and four training checkpoints fit. Normalization gradients, simultaneous parameter updates, duplicate candidates, missing candidates, and frozen-encoder readouts are checked.
- Vision IV: 4,779 reference/invariant comparisons agree within 5.69e-14; 1,096 finite-difference checks cover all 137 parameters, both images and both snapshots (maximum 6.94e-8). Exact NumPy regeneration, the shared Vision I front end, causal prefixes, SGD update, image sensitivity, and actual greedy generation pass. A final independent content pass checked notation and lesson transitions.
- The VLM update deliberately shows a regression: two-image-response loss 0.0850 → 0.0077, but one-image-response loss 0.0790 → 0.6372. The latter answer changes from “one block” to “two blocks”. This is a training-set conditioning demonstration, not evidence of general counting ability.
- PDFs contain 111 fully revealed 16:9 pages at 2× raster resolution; all eight quiz/reveal answers are open. Every page has a valid slide image. Representative actual PDF pages, dense calculations, architecture diagrams, generation states, and phone views were visually inspected.
- Final notation fixes reserve `M` for a mask, use `n_r` for resampler row count, distinguish `d_k` and `d_v`, and explicitly disclose omitted CLS in the MAE schematic.

Core browser checks work with external requests blocked. These tests do not establish exhaustive browser compatibility or learned-model generalization. Very wide SVGs remain small overview diagrams on narrow phones; classroom and desktop views retain readable labels.

Reproduce with `check_vision1.mjs` through `check_vision4.mjs`, `train_vision3.py --check`, `train_vision4.py --check`, and the shared frame/table/PDF checks documented in `README.md`. Exported PDFs remain ignored local build artifacts.

## 2026-09-04: Part IV, cross-attention and translation

- 8 sections, 32 classroom frames, 85 progressive states. All states pass strict fit and math checks (62 formulas).
- 82 table views across desktop, phone, and presentation pass without warnings or failures.
- Phone reading at 390px has no horizontal overflow; interaction sweep reports no runtime errors.
- 2,274 JS/reference scalar checks agree within 3.56e-15. All 188 parameter gradients pass central differences
  within 6.13e-11. Independent review also checked causal prefixes, source sensitivity, and two genuine greedy translations.
- Python `train_part4.py --check` reproduces every saved tensor, the 507 warm-up steps, and the displayed update.
- SVG label bounds pass all nine diagram stages. Generation frames show only the current step and do not announce EOS early.
- PDF: 32 pages at 2x raster resolution; both reveal answers are open. Every page is 16:9 and contains its slide image.
- The three-dimensional two-pair fitted toy omits FFN, LayerNorm, and dropout. Its further river-only update improves
  that example's mean loss (0.043213 to 0.030730), not every position or the financial example.

## 2026-09-04: tone, whitespace, and PDF answers

- Humanizer pass: all 54 section files in Parts I–III; computation and notation retained and embedded scripts parsed.
- No persistent presentation bars. One heading per slide, quiet margins, and on-demand keyboard-accessible controls.
- Presentation regression suite passes: hidden-control focus, scale invariance on opening controls, navigation, overview,
  presenter window, print reveal/restore, mobile stage, live fit diagnostics, and preflight restoration.
- All 563 existing live states pass strict maths and fit checks: Part I 132, Part II 297, Part III 134.
- Table flow: 507 desktop/mobile/presentation table views checked. No errors or alignment/overflow failures; two existing
  narrow-prose-column advisories remain in Part II's comparison tables.
- `export_test.mjs`: final/all-build × shown/authored answer modes, 12 PDF pages, actual PNG answer pixels checked.
  Default exports show quiz answers on completed frames; earlier all-build pages retain the question.
- Rebuilt PDFs: Part I 61 pages (6 newly opened answers), Part II 133 (11), Part III 68 (13).
- Part I arithmetic and sampler match reference values. Part II routing/scaling checks pass, including value-only changes,
  independent score normalization, mobile width, ≤4-line code snippets, and no JS/KaTeX errors.

## Previous release record

Verified locally on 2026-09-03. All three parts use the same slide-first runtime. Each frame fits a logical 1280 × 720 stage; the article unfolds the same content and its companion explanations. Exact-slide PDF exports contain no browser header or footer.

## Classroom and reading checks

| Part | Authored frames / PDF pages | Presentation states walked | Unique rendered formulas reparsed |
|---|---:|---:|---:|
| 1: characters to prediction | 61 | 132 | 79 |
| 2: self-attention | 133 | 297 | 487 |
| 3: learning and Transformer blocks | 68 | 134 | 140 |

- All 563 states passed: no stage/frame overflow, nested scrolling, JavaScript errors, or invalid rendered formulas. Preflight also checks open reveals and intermediate managed steps. The walkthrough explicitly starts at frame 1, independent of the article's current scroll position.
- Strict math validation reparses each unique rendered formula with KaTeX `throwOnError: true`. A regression fixture confirmed that unknown commands, malformed math, and fallback text fail the audit, including commands introduced only in a later step.
- Reading layouts fit 390px phone width without horizontal document overflow. Interaction sweeps exercised 32 controls in Part 1, 128 in Part 2, and 34 in Part 3 without errors.
- Runtime regression checks passed for forward/back navigation, multiple managed steppers, manual-widget state/focus/selection, pending-control accessibility, overview focus, presenter view, fullscreen fallback, print restoration, deep links, and exit/reload. Managed steppers use the global presentation controls; manual widgets retain their local toolbar.
- Title-only overflow, native disclosure changes, and intermediate-step disclosure overflow are detected. Preflight preserves edited manual widgets rather than rebuilding their DOM.
- Representative diagram, dense-table, intro, and continuation frames were inspected visually. The browser header and footer are compact; body/title/caption/math sizes are 28/42/22/32 logical pixels.

## Table and Part 1 diagram refinement

The dedicated table audit sampled 507 tables across desktop reading, 390px phone reading, and presentation. It checks cell-text overlap, table overlap, semantic numeric/text/code styling, header sizes, contained mobile scrolling, and runtime errors. Classification, update, footer, and colour-tint regression fixtures pass for all three assembled parts. Numeric precision was preserved; prose no longer inherits right-aligned number styling. Dense token headers and projection columns were visually checked, and 42 targeted screenshots were saved during review. Intentional row-label wrapping remains readable.

Four Part 1 diagrams adapt the original handwritten notes' visual sequence: the learned embedding scatter, repeated lookup to ordered concatenation, the prediction/learning graph, and the two training/generation loops. The regression creates 30 SVG instances and checks all 662 labels against their viewBoxes, all 184 arrow-marker references against their owning SVG, unique IDs, and accessible titles/descriptions. It verifies all 27 plotted embedding coordinates, six concatenated values, model shapes, target probability/loss, and a reproducible sample of `i` followed by the context `a b i`. A sampled boundary stops generation. Drawing diagrams does not mutate model parameters; the observed target has its own loss branch and is not an input.

Part 1 one-hot worksheets use the neutral label `products`, not query/key notation. Diagram stages use the full classroom width. Wide diagrams and tables may pan within a bounded container in phone reading mode; presentation frames never scroll.

## Numerical and conceptual checks

| Check | Result |
|---|---|
| Part 1 forward pass | All six saved example rows agree with the JS model; maximum error 1.67e−16 |
| Part 1 generation | Displayed probabilities and sampler agree at four temperatures |
| Part 2 reference versus Python | 6,615 values; maximum error 3.55e−15; all hard targets pass |
| Live article versus reference | Eight original cases; 8,220 finite values and 180 masked infinities; maximum error 1.34e−15 |
| Position capacity and generation | 48 cases at lengths 10, 11, and 20 across both attention pages; 84,968 finite values and 2,680 masked infinities; maximum error 1.34e−15 |
| Invalid inputs | Twelve checks reject unsupported length, unknown tokens, and missing, wrong-width, or non-finite position vectors |
| Part 3 generation UI | Appends `water`, evaluates position 11, and displays the correct distribution for position 12 |
| Stored training results | All 1,678 saved numeric entries match recomputation at stored precision |
| Backward implementation | All 260 used parameter gradients checked by central differences; maximum absolute error 4.51e−11 |
| Diagram evidence | 307 comparisons; shapes, causal zeros, normalization, prefix equivalence, provenance, and deep freeze pass |
| Staged diagram | All twelve stages fit; receiver changes from bank at position 7 to the final token at position 10 |
| Routing/scaling | Value-only intervention leaves weights unchanged; divisor/softmax calculations and variance simulation pass |

The review also corrected the incomplete vocabulary-softmax worksheet, vocabulary bias shape, residual-gradient graph, LayerNorm epsilon, final pre-norm LayerNorm, and a future-leaking multi-layer example. The three-part notation separates the learned vocabulary table `E_tok` from the sequence stack `E`, values from embeddings, message matrix `H` from causal mask `M`, and attention weights from vocabulary probabilities. Head-message equations are introduced before concatenation.

The learner-facing backpropagation explanation uses the branched computation graph and autograd, not a manual Jacobian derivation. Numerical gradients remain a behind-the-scenes regression test.

## PDF verification

The exact-slide exporter produced 61-, 133-, and 68-page PDFs at 2× resolution. Every page has a 16:9 media box and an embedded slide image. First, middle, and last pages of each exported PDF were rendered and visually inspected, along with the new Part 1 diagrams and representative changed tables.

The exporter also passed progressive-build and multiple-stepper fixtures, and rejected an intentionally overfull frame instead of writing a clipped PDF. `--builds all` records each managed step; default export records the final state of each authored frame. Sliders, quizzes, and manual disclosures remain at their authored defaults.

These PDFs are appearance-faithful raster snapshots: text is not selectable, and controls are no longer interactive. Local PDF outputs are ignored by Git and can be regenerated from the committed sources.

## Deliberate teaching boundaries

Part 1 uses a trained small name model. Part 2 is a hand-designed, single-head model with widths 5/3/2 and a 20-token vocabulary, not a trained language model. Part 3 demonstrates actual updates to that toy before explaining a larger Transformer architecture.

The toy has twenty same-width position vectors. Its initial projection/head matrices ignore the dedicated position coordinate. Therefore it demonstrates content routing, not word-order sensitivity; this is now explained explicitly in the introduction and summary. Position is added, not appended as an obligatory extra dimension. The causal mask still restricts available sources.

Full Transformer diagrams are architectural explanations. The original one-head numerical worksheet does not silently become a multi-head, normalized, stacked model. This release is a checked teaching implementation, not a claim of exhaustive browser/accessibility coverage or an absence of every possible pedagogical improvement.

## Reproduce from the repository root

```sh
python3 src/assemble.py --part 1 --out part1.html
python3 src/assemble.py --part 2 --out attention.html
python3 src/assemble.py --part 3 --out part3.html
node src/check_part1.mjs
node src/check_part1_diagrams.mjs part1.html
node src/toy_ref.mjs src/toy.json --compare src/py_check.json
node src/check-live-model.mjs attention.html
node src/check_token_flow.mjs attention.html
python3 src/check_training.py
node src/check_position_capacity.mjs
node src/check-routing-scaling.mjs
node src/check-diagram.mjs attention.html
node src/pres_test.mjs
node src/frame_audit.mjs part1.html
node src/frame_audit.mjs attention.html
node src/frame_audit.mjs part3.html
node src/check_tables.mjs part1.html attention.html part3.html
node src/qa.mjs attention.html --width 390 --height 844
node src/sweep.mjs attention.html
node figures/attention-diagram-preview/check-data.mjs
node src/export_slides.mjs attention.html output/pdf/attention-part2-slides.pdf
git diff --check
```

Repeat article QA, interaction sweeps, and PDF export for each changed part. Check the GitHub Pages workflow before assuming the latest pushed source is live.

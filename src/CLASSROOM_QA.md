# Classroom release checks

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
